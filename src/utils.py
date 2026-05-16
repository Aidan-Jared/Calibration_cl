import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx
from tqdm import tqdm
from optax import softmax_cross_entropy_with_integer_labels
from torchvision import transforms

from equinox.nn._stateful import State


def model_forward(
    model,
    X: Array,
    state: State,
    key: PRNGKeyArray,
) -> tuple[Array, State]:
    y, state = model(X, state, key=key)
    return y, state


def load_data(data_set: str):
    normalize_data = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    if data_set == "CIFAR100":
        from torchvision.datasets import CIFAR100

        train = CIFAR100(
            root="Data/",
            train=True,
            transform=normalize_data,
            download=True,
        )

        test = CIFAR100(
            root="Data/",
            train=False,
            transform=normalize_data,
            download=True,
        )

    elif data_set == "FashionMNIST":
        from torchvision.datasets import FashionMNIST

        train = FashionMNIST(
            root="Data/",
            train=True,
            transform=normalize_data,
            download=True,
        )

        test = FashionMNIST(
            root="Data/",
            train=False,
            transform=normalize_data,
            download=True,
        )

    elif data_set == "MNIST":
        from torchvision.datasets import MNIST

        train = MNIST(
            root="Data/",
            train=True,
            transform=normalize_data,
            download=True,
        )

        test = MNIST(
            root="Data/",
            train=False,
            transform=normalize_data,
            download=True,
        )

    elif data_set == "Food101":
        from torchvision.datasets import Food101

        train = Food101(
            root="Data/",
            split="train",
            transform=normalize_data,
            download=True,
        )

        test = Food101(
            root="Data/",
            split="test",
            transform=normalize_data,
            download=True,
        )

    elif data_set == "ImageNet":
        from torchvision.datasets import ImageNet

        train = ImageNet(
            root="Data/",
            split="train",
            transform=normalize_data,
            download=True,
        )

        test = ImageNet(
            root="Data/",
            split="test",
            transform=normalize_data,
            download=True,
        )

    else:
        from torchvision.datasets import CIFAR10

        train = CIFAR10(
            root="Data/",
            train=True,
            transform=normalize_data,
            download=True,
        )

        test = CIFAR10(
            root="Data/",
            train=False,
            transform=normalize_data,
            download=True,
        )
    return train, test


def eval(model, state, tasks, testloader, *, key):
    model = eqx.nn.inference_mode(model, value=True)

    def loss_fn(model, x, y, state, key):
        logits, _ = jax.vmap(
            model_forward,
            in_axes=(None, 0, None, None),
            out_axes=(0, None),
            axis_name="batch",
        )(model, x, state, key)

        loss = softmax_cross_entropy_with_integer_labels(logits, y)
        acc = jnp.mean(jnp.argmax(logits, axis=1) == y)

        return loss, acc

    loss_fn = eqx.filter_jit(loss_fn)
    results = dict()
    for p_task in range(tasks):
        key, subkey = jax.random.split(key)
        task_loss = []
        task_acc = []
        pbar = tqdm(
            enumerate(testloader.sample(p_task, key=subkey)),
            total=testloader.iters(p_task),
            ncols=75,
        )
        for step, (x, y, _, _, _) in pbar:
            key, subkey = jax.random.split(key)
            loss, acc = loss_fn(model, x, y, state, subkey)
            task_loss.append(loss)
            task_acc.append(acc)
            if step % 10 == 0:
                pbar.set_postfix(
                    {
                        "task_eval": p_task,
                        "loss": jnp.mean(jnp.array(loss)).item(),
                        "acc": jnp.mean(jnp.array(acc)).item(),
                    }
                )
        results[p_task] = {
            "loss": jnp.mean(jnp.array(task_loss)).item(),
            "acc": jnp.mean(jnp.array(task_acc)).item(),
        }

    return results
