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

    if task_n == 0:
        start = 0
        end = buffer_idx.shape[0] // task_idx.shape[0]       
        for task in task_idx:
            key, subkey = jax.random.split(key)
            choices = jax.random.choice(subkey, dataloader.class_indicies[task], shape=(buffer_idx.shape[0] // task_idx.shape[0],))
            labels = jnp.full((buffer_idx.shape[0] // task_idx.shape[0],), task, dtype=jnp.uint32)
            samples = dataloader.class_indicies[task][choices]

            X = dataloader.all_data[samples]

            logits, _ = model_forward(model, X, state, key=key)

            buffer_idx = buffer_idx.at[start:end].set(samples)
            buffer_targets = buffer_targets.at[start:end].set(labels)
            buffer_logits = buffer_logits.at[start:end].set(logits)
            start = end
            end += buffer_idx.shape[0] // task_idx.shape[0]
    else:
        for task in task_idx:
            key, subkey1, subkey2 = jax.random.split(key, 3)
            replace = jax.random.bernoulli(subkey1, p=1 / (task + 1), shape=(buffer_idx.shape[0],))

            nsamples = jnp.sum(replace)
           
            choices = jax.random.choice(subkey2, dataloader.class_indicies[task], shape=(nsamples,))
            labels = jnp.full((nsamples,), task, dtype=jnp.uint32)
            samples = dataloader.class_indicies[task][choices]

            X = dataloader.all_data[samples]

            logits, _ = model_forward(model, X, state, key=key)

            buffer_idx = jnp.where(replace, samples, buffer_idx)
            buffer_targets = jnp.where(replace, labels, buffer_targets)
            buffer_logits = jnp.where(replace, logits, buffer_logits)

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
    calibration = buffer_logits[:,-1]
    removed = (buffer_idx.shape[0] // unique_targets.shape) // (task_n + 1)
    for i in unique_targets:
        target_idxes = jnp.argwhere(buffer_targets == i)

        removed = jnp.argsort(calibration[target_idxes])[:removed]
        
        replace_samples.append(removed)
    replace_samples = jnp.concatenate(replace_samples)

    task_idx = dataloader.tasks[task_n]

    model_jit = eqx.filter_jit(model_forward)

    for x, _, class_idx, task, _ in dataloader.sample(task_n, key=key):
        logits,_ = model_jit(model, x, state, key=key)
        

    buffer_idx = buffer_idx.at[replace_samples].set(samples)
    buffer_targets = buffer_targets.at[replace_samples].set(labels)
    buffer_logits = buffer_logits.at[replace_samples].set(logits)
    return buffer_idx, buffer_targets, buffer_logits