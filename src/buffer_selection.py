import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Array, PRNGKeyArray

from src.models.resnet18 import ResNet18
from src.models.resnet32 import ResNet32


@jax.jit(static_argnames=("device",))
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

    batch_idxes = jnp.arange(1, batch_size + 1, dtype=jnp.int32)

    def rand_selection(key, n, i):
        rand_idx = jax.random.randint(key, (), 0, n, dtype=jnp.int32)
        replace, choice = jax.lax.cond(
            rand_idx < buffer_size, lambda: (rand_idx, i), lambda: (-1, -1)
        )
        return replace, choice

    def add_to_buffer(batch_idx, seen_examples, key):
        n = seen_examples + batch_idx
        replace, choice = jax.lax.cond(
            n < buffer_size,
            lambda k, n, i: (n, i),
            lambda k, n, i: rand_selection(k, n, i),
            key,
            n,
            batch_idx,
        )
        return replace, choice

    keys = jax.random.split(key, batch_size)
    replace, choices = jax.vmap(add_to_buffer, in_axes=(0, None, 0))(
        batch_idxes, seen_examples, keys
    )
    
    seen_examples += batch_size

    choices = jnp.array(choices, device=device, dtype=jnp.int32)
    replace = jnp.array(replace, device=device, dtype=jnp.int32)
    buffer_idx = buffer_idx.at[replace].set(sample_idx[choices])
    buffer_targets = buffer_targets.at[replace].set(labels[choices].astype(jnp.uint32))
    buffer_logits = buffer_logits.at[replace].set(logits[choices])

    return buffer_idx, buffer_targets, buffer_logits, seen_examples


    replace, choices = jax.vmap(add_to_buffer, in_axes=(0, None, 0))(
        batch_idxes, seen_examples, keys
    )
    
    seen_examples += batch_size

    choices = jnp.array(choices, device=device, dtype=jnp.int32)
    replace = jnp.array(replace, device=device, dtype=jnp.int32)
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
