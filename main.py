import jax
import jax.numpy as jnp
import equinox as eqx

from src.dataloader import CL_DataLoader
from src.models.resnet32 import singleHeadResNet32

from torchvision.datasets import CIFAR10
from torchvision import transforms

SEED = 42
BATCH = 32
SPLITS = 5

def main():
    KEY = jax.random.PRNGKey(SEED)
    norm = [(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)]
    normalize_data = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
        
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
    

    subkey1, subkey2 = jax.random.split(KEY)
    trainloader = CL_DataLoader(
        train, batch_size=BATCH, splits=SPLITS, dtype=jnp.float32, key=subkey1, buffer = True, buffer_size = 320
    )

    trainloader.normalize(norm[0], norm[1])
    
    p_model, state = eqx.nn.make_with_state(singleHeadResNet32)(
        trainloader.all_data[0].shape[0], num_classes = trainloader.num_classes, num_splits = SPLITS, dropout=0.0, dtype=jnp.float32, key=subkey2
    )

    trainloader.add_to_buffer(0, p_model, state, key = subkey1)


if __name__ == "__main__":
    main()
