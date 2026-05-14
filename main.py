import jax
import jax.numpy as jnp
import equinox as eqx

from optax import sgd

from src.der import train_der
from src.dataloader import CL_DataLoader
from src.models.resnet32 import singleHeadResNet32

from torchvision.datasets import CIFAR10
from torchvision import transforms

SEED = 42
BATCH = 32
SPLITS = 5
EPOCHS = 1
LR = 1e-3

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
    testloader = CL_DataLoader(
        test, batch_size=BATCH, splits=SPLITS, dtype=jnp.float32, key=subkey1, buffer = False
    )

    trainloader.normilization_values(norm[0], norm[1])
    testloader.normilization_values(norm[0], norm[1])
    
    p_model, state = eqx.nn.make_with_state(singleHeadResNet32)(
        trainloader.all_data[0].shape[0], num_classes = trainloader.num_classes, num_splits = SPLITS, dropout=0.0, dtype=jnp.float32, key=subkey2
    )

    # trainloader.add_to_buffer(0, p_model, state, key = subkey1)

    prob_history = jnp.zeros((trainloader.all_data.shape[0], trainloader.num_classes), dtype = jnp.float32)

    updated = jnp.zeros((trainloader.all_data.shape[0],), dtype = jnp.uint32)
    
    optim = sgd(LR)
    
    train_der(
        p_model, trainloader, testloader,
        SPLITS, EPOCHS, state, optim,
        beta = .5, prob_history = prob_history, gamma = 2.0, soc_alpha = 0.9, updated = updated, key = subkey1
    )

if __name__ == "__main__":
    main()
