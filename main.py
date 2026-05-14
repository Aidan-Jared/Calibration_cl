import jax
import jax.numpy as jnp
import equinox as eqx

from optax import sgd

from src.der import train_der
from src.dataloader import CL_DataLoader
from src.models.resnet32 import singleHeadResNet32
from src.models.resnet18 import singleHeadResNet18
from src.models.vit import VisionTransformer

from src.utils import load_data


import argparse
import os
import ast
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def parse_list(arg):
    return ast.literal_eval(arg)

parser = argparse.ArgumentParser()
# training hyperparameters
parser.add_argument("--seed", type=parse_list, default=[42])
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--splits", type=int, default=5)
parser.add_argument("--task-epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--model", type=str, default="singleHeadResNet32", choices=["singleHeadResNet32", "singleHeadResNet18"])

# dataset parameters
parser.add_argument("--data_set", type=str, default="CIFAR10")
parser.add_argument("--norm", type=parse_list, default=[(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)])

# Dataset Mean Std
# MNIST (0.1307,) (0.3081,)
# FashionMNIST (0.2860,) (0.3530,)
# CIFAR-10 (0.4914, 0.4822, 0.4465) (0.2470, 0.2435, 0.2616)
# CIFAR-100 (0.5071, 0.4867, 0.4408)(0.2675, 0.2565, 0.2761)

# method parameters
parser.add_argument("--method", type=str, default="DER", choices=["DER"])
parser.add_argument("--der-alpha", type=float, default=.5)
parser.add_argument("--der-beta", type=float, default=.5)
parser.add_argument("--buffer-size", type=int, default=320)

# socrates parameters
parser.add_argument("--loss", type=str, default="", choices=["", "socrates"])
parser.add_argument("--soc-alpha", type=float, default=0.)
parser.add_argument("--soc-gamma", type=float, default=0.)


args = vars(parser.parse_args())

model_dict = {
    "singleHeadResNet32": singleHeadResNet32,
    "singleHeadResNet18": singleHeadResNet18,
    "VisionTransformer": VisionTransformer,
}


def main():
    
    BATCH = args["batch_size"]
    SPLITS = args["splits"]
    LR = args["lr"]
    EPOCHS = args["epochs"]

    train, test = load_data(args["data_set"])
    
    for SEED in args['seed']:
        KEY = jax.random.PRNGKey(SEED)
        norm = args["norm"]
        
    
        subkey1, subkey2, subkey3, subkey4 = jax.random.split(KEY, 4)
        # prepare data loaders
        trainloader = CL_DataLoader(
            train, batch_size=BATCH, splits=SPLITS, dtype=jnp.float32, key=subkey1, buffer = True, buffer_size = args["buffer_size"]
        )
        testloader = CL_DataLoader(
            test, batch_size=BATCH, splits=SPLITS, dtype=jnp.float32, key=subkey2, buffer = False
        )
    
        trainloader.normilization_values(norm[0], norm[1])
        testloader.normilization_values(norm[0], norm[1])
        
        # make model
        if args["loss"] == "soc":
            num_classes = trainloader.num_classes + 1
        else:
            num_classes = trainloader.num_classes
    
        model, state = eqx.nn.make_with_state(model_dict[args["model"]])(
            trainloader.all_data[0].shape[0], num_classes = num_classes,
            num_splits = SPLITS, dropout=args["dropout"], dtype=jnp.float32, key=subkey3
        )
    
        if args["loss"] == "soc":
            # initialize socrates state
            prob_history = jnp.zeros((trainloader.all_data.shape[0], trainloader.num_classes), dtype = jnp.float32)
        
            updated = jnp.zeros((trainloader.all_data.shape[0],), dtype = jnp.uint32)
            soc_alpha = args["soc_alpha"]
            gamma = args["soc_gamma"]
        else:
            prob_history = None
            updated = None
            soc_alpha = None
            gamma = None
        
        optim = sgd(LR)
        
        train_der(
            model, trainloader, testloader,
            SPLITS, EPOCHS, state, optim,
            der_alpha= args["der_alpha"], beta = args["der_beta"],
            prob_history = prob_history, gamma = gamma, soc_alpha = soc_alpha, updated = updated, key = subkey4
        )
    
if __name__ == "__main__":
    main()
