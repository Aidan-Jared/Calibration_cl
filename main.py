import argparse
import ast
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import pandas as pd
from optax import add_decayed_weights, chain, sgd

from src.buffer_selection import reservoir_sampling
from src.dataloader import CL_DataLoader
from src.der import train_der
from src.models.resnet18 import singleHeadResNet18
from src.models.resnet32 import singleHeadResNet32
from src.models.vit import VisionTransformer
from src.utils import load_data

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def parse_list(arg):
    return ast.literal_eval(arg)


def parse_bool(arg):
    return arg == "True"


parser = argparse.ArgumentParser()
# experiment settings
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model-runs", type=int, default=1)

parser.add_argument(
    "--model",
    type=str,
    default="singleHeadResNet32",
    choices=["singleHeadResNet32", "singleHeadResNet18", "VisionTransformer"],
)

# training hyperparameters
parser.add_argument("--replay-size", type=int, default=32)
parser.add_argument("--task-splits", type=int, default=5)

parser.add_argument("--task-epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--momentum", type=float, default=0.0)
parser.add_argument("--batch-size", type=int, default=32)

parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--transform", type=parse_bool, default="True")
parser.add_argument("--task-shuffle", type=parse_bool, default=False)

# dataset parameters
parser.add_argument("--data_set", type=str, default="CIFAR10")
parser.add_argument(
    "--norm",
    type=parse_list,
    default=[(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)],
)

# Dataset Mean Std
# MNIST (0.1307,) (0.3081,)
# FashionMNIST (0.2860,) (0.3530,)
# CIFAR-10 (0.4914, 0.4822, 0.4465) (0.2470, 0.2435, 0.2616)
# CIFAR-100 (0.5071, 0.4867, 0.4408)(0.2675, 0.2565, 0.2761)
# Food-101 (0.5450, 0.4435, 0.3436) (0.2262, 0.2326, 0.2377)
# ImageNet (0.485,  0.456,  0.406)  (0.229,  0.224,  0.225)

# method parameters
parser.add_argument("--method", type=str, default="DER", choices=["DER"])
parser.add_argument("--der-alpha", type=float, default=0.5)
parser.add_argument("--der-beta", type=float, default=0.5)
parser.add_argument(
    "--selection-method",
    type=str,
    default="reservoir_sampling",
    choices=["reservoir_sampling", ""],
)
parser.add_argument("--buffer-size", type=int, default=320)

# socrates parameters
parser.add_argument("--loss", type=str, default="", choices=["", "socrates"])
parser.add_argument("--soc-alpha", type=float, default=0.0)
parser.add_argument("--soc-gamma", type=float, default=0.0)


args = vars(parser.parse_args())

model_dict = {
    "singleHeadResNet32": singleHeadResNet32,
    "singleHeadResNet18": singleHeadResNet18,
    "VisionTransformer": VisionTransformer,
}

selection_dict = {
    "reservoir_sampling": reservoir_sampling,
    "": None,
}


def main():
    BATCH = args["batch_size"]
    SPLITS = args["task_splits"]
    LR = args["lr"]
    MOMENTUM = args["momentum"]
    EPOCHS = args["task_epochs"]

    train, test = load_data(args["data_set"])

    key = jax.random.PRNGKey(args["seed"])
    seeds = jax.random.randint(key, (args["model_runs"],), 1, 5000)
    df = pd.DataFrame()

    for SEED in seeds:
        KEY = jax.random.PRNGKey(SEED)
        norm = args["norm"]

        subkey1, subkey2, subkey3, subkey4 = jax.random.split(KEY, 4)

        # prepare data loaders
        if args["loss"] == "socrates":
            soc = True
        else:
            soc = False

        if not args["task_shuffle"]:
            subkey1 = None

        trainloader = CL_DataLoader(
            train,
            batch_size=BATCH,
            splits=SPLITS,
            dtype=jnp.float32,
            key=subkey1,
            buffer=True,
            buffer_size=args["buffer_size"],
            buff_size_mem=args["replay_size"],
            transform=args["transform"],
            socrates=soc,
        )

        testloader = CL_DataLoader(
            test,
            batch_size=BATCH,
            splits=SPLITS,
            dtype=jnp.float32,
            transform=args["transform"],
            key=subkey1,
            buffer=False,
        )

        trainloader.normilization_values(norm[0], norm[1])
        testloader.normilization_values(norm[0], norm[1])

        # make model
        if args["loss"] == "soc":
            num_classes = trainloader.num_classes + 1
        else:
            num_classes = trainloader.num_classes

        model, state = eqx.nn.make_with_state(model_dict[args["model"]])(
            trainloader.all_data[0].shape[0],
            num_classes=num_classes,
            num_splits=SPLITS,
            dropout=args["dropout"],
            dtype=jnp.float32,
            key=subkey3,
        )

        if args["loss"] == "soc":
            # initialize socrates state
            prob_history = jnp.zeros(
                (trainloader.all_data.shape[0], trainloader.num_classes),
                dtype=jnp.float32,
            )

            updated = jnp.zeros((trainloader.all_data.shape[0],), dtype=jnp.uint32)
            soc_alpha = args["soc_alpha"]
            gamma = args["soc_gamma"]
        else:
            prob_history = None
            updated = None
            soc_alpha = None
            gamma = None

        optim = chain(
            # add_decayed_weights(1e-4),
            sgd(LR, momentum=MOMENTUM),
        )

        res = train_der(
            model,
            trainloader,
            testloader,
            SPLITS,
            EPOCHS,
            state,
            optim,
            der_alpha=args["der_alpha"],
            beta=args["der_beta"],
            selection_method=selection_dict[args["selection_method"]],
            prob_history=prob_history,
            gamma=gamma,
            soc_alpha=soc_alpha,
            updated=updated,
            key=subkey4,
        )

        results = [{"seed": SEED.item()} | res for res in res]

        df = pd.concat([df, pd.DataFrame(results)])

    df.to_parquet("Runs/test_control.parquet")


if __name__ == "__main__":
    main()
