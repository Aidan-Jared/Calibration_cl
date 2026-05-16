import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from jaxtyping import Array, PRNGKeyArray

from src.models.resnet18 import ResNet18
from src.models.resnet32 import ResNet32
from src.dataloader import CL_DataLoader
from src.utils import model_forward


def reservoir_sampling(
    dataloader: CL_DataLoader,
    task_n,
    buffer_idx: Array,
    buffer_targets: Array,
    buffer_logits: Array,
    model: ResNet18 | ResNet32,
    state: eqx.nn._stateful.State,
    *,
    key: PRNGKeyArray,
):
    task_idx = dataloader.tasks[task_n]

    model_forward_jit = eqx.filter_jit(model_forward)
    if task_n == 0:
        start = 0
        end = buffer_idx.shape[0] // task_idx.shape[0]
        for task in task_idx:
            key, subkey = jax.random.split(key)
            choices = jax.random.choice(
                subkey,
                dataloader.class_indicies[task],
                shape=(buffer_idx.shape[0] // task_idx.shape[0],),
                replace=False,
            )
            labels = jnp.full(
                (buffer_idx.shape[0] // task_idx.shape[0],), task, dtype=jnp.uint32
            )

            X: Array = dataloader.all_data[choices]
            X = dataloader._norm(
                X.astype(jnp.float32) / 255, dataloader.mean, dataloader.std
            )
            X = jax.device_put(X, jax.devices(dataloader.iter_device)[0])

            logits, _ = jax.vmap(
                model_forward_jit,
                in_axes=(None, 0, None, None),
                out_axes=(0, None),
                axis_name="batch",
            )(model, X, state, key)

            logits = jax.device_put(logits, jax.devices(dataloader.device)[0])

            buffer_idx = buffer_idx.at[start:end].set(choices)
            buffer_targets = buffer_targets.at[start:end].set(labels)
            buffer_logits = buffer_logits.at[start:end].set(logits)
            start = end
            end += buffer_idx.shape[0] // task_idx.shape[0]
            del logits, X
    else:
        for task in task_idx:
            nunique = jnp.unique(buffer_targets).shape[0]
            key, subkey1, subkey2 = jax.random.split(key, 3)
            replace = jax.random.bernoulli(
                subkey1, p=1 / (nunique), shape=(buffer_idx.shape[0],)
            )

            nsamples = jnp.sum(replace).item()

            choices = jax.random.choice(
                subkey2,
                dataloader.class_indicies[task],
                shape=(nsamples,),
                replace=False,
            )

            labels = jnp.full((nsamples,), task, dtype=jnp.uint32)

            X: Array = dataloader.all_data[choices]
            X = dataloader._norm(
                X.astype(jnp.float32) / 255, dataloader.mean, dataloader.std
            )
            X = jax.device_put(X, jax.devices(dataloader.iter_device)[0])

            logits, _ = jax.vmap(
                model_forward_jit,
                in_axes=(None, 0, None, None),
                out_axes=(0, None),
                axis_name="batch",
            )(model, X, state, key)

            logits = jax.device_put(logits, jax.devices(dataloader.device)[0])

            replace = jnp.where(replace)

            buffer_idx = buffer_idx.at[replace].set(choices)
            buffer_targets = buffer_targets.at[replace].set(labels)
            buffer_logits = buffer_logits.at[replace].set(logits)

            del logits, X

    return buffer_idx, buffer_targets, buffer_logits


def calibration_balanced_class_selection(
    dataloader: CL_DataLoader,
    task_n,
    buffer_idx: Array,
    buffer_targets: Array,
    buffer_logits: Array,
    model: ResNet18 | ResNet32,
    state: eqx.nn._stateful.State,
    *,
    key: PRNGKeyArray,
):
    unique_targets = jnp.unique(buffer_targets)
    replace_samples = []
    calibration = buffer_logits[:, -1]
    removed = (buffer_idx.shape[0] // unique_targets.shape) // (task_n + 1)
    for i in unique_targets:
        target_idxes = jnp.argwhere(buffer_targets == i)

        removed = jnp.argsort(calibration[target_idxes])[:removed]

        replace_samples.append(removed)
    replace_samples = jnp.concatenate(replace_samples)

    task_idx = dataloader.tasks[task_n]

    model_jit = eqx.filter_jit(model_forward)

    for x, _, class_idx, task, _ in dataloader.sample(task_n, key=key):
        logits, _ = model_jit(model, x, state, key=key)

    buffer_idx = buffer_idx.at[replace_samples].set(samples)
    buffer_targets = buffer_targets.at[replace_samples].set(labels)
    buffer_logits = buffer_logits.at[replace_samples].set(logits)
    return buffer_idx, buffer_targets, buffer_logits
