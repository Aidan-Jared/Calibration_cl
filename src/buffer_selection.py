import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Array, PRNGKeyArray

from src.models.resnet18 import ResNet18
from src.models.resnet32 import ResNet32


def reservoir_sampling(
    sample_idx: Array,
    labels: Array,
    logits: Array,
    buffer_idx: Array,
    buffer_targets: Array,
    buffer_logits: Array,
    seen_examples: int,
    *,
    device,
    key: PRNGKeyArray,
):
    batch_size = sample_idx.shape[0]
    buffer_size = buffer_idx.shape[0]
    replace = []
    choices = []
    for i in range(batch_size):
        if seen_examples < buffer_size:
            replace.append(seen_examples)
            choices.append(i)
        else:
            key, subkey = jax.random.split(key)
            rand_idx = jax.random.randint(subkey, (), 0, seen_examples).item()

            if rand_idx < buffer_size:
                replace.append(rand_idx)
                choices.append(i)

        seen_examples += 1

    choices = jnp.array(choices, device=device, dtype=jnp.uint32)
    replace = jnp.array(replace, device=device, dtype=jnp.uint32)
    buffer_idx = buffer_idx.at[replace].set(sample_idx[choices])
    buffer_targets = buffer_targets.at[replace].set(labels[choices].astype(jnp.uint32))
    buffer_logits = buffer_logits.at[replace].set(logits[choices])

    return buffer_idx, buffer_targets, buffer_logits, seen_examples


#  not implemented yet
def calibration_balanced_class_selection(
    dataloaderh,
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
