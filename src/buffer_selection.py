import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from jaxtyping import Array, PRNGKeyArray

from src.models.resnet18 import ResNet18
from src.models.resnet32 import ResNet32
from src.dataloader import CL_DataLoader
from src.util import model_forward

def random_balanced_class_selection(
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
    prob = (unique_targets.shape[0] - 1) / unique_targets.shape[0]
    for i in unique_targets:
        target_idxes = jnp.argwhere(buffer_targets == i)
        key, subkey = jax.random.split(key)
        
        mask = jax.random.bernoulli(subkey, p= prob, shape=target_idxes.shape) <= prob

        replace_samples.append(target_idxes[mask])
    replace_samples = jnp.concatenate(replace_samples)

    task_idx = dataloader.tasks[task_n]

    key, subkey = jax.random.split(key)
    choices = jax.random.choice(subkey, dataloader.class_indicies[task_idx], shape=(replace_samples.shape[0],))
    labels = np.repeat(task_idx, dataloader.class_lengths[task_idx])[choices]
    samples = dataloader.class_indicies[task_idx][choices]

    X = dataloader.all_data[samples]

    logits, _ = model_forward(model, X, state, key=key)

    buffer_idx = buffer_idx.at[replace_samples].set(samples)
    buffer_targets = buffer_targets.at[replace_samples].set(labels)
    buffer_logits = buffer_logits.at[replace_samples].set(logits)
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