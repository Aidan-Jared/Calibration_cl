import jax
import jax.numpy as jnp
import equinox as eqx

from typing import Callable
from jaxtyping import Array, PRNGKeyArray

from src.models.resnet18 import ResNet18
from src.models.resnet32 import ResNet32


def make_buffer(
    buffer_size, num_classes, device: str = "gpu", socrates: bool = False
) -> tuple[Array, Array, Array]:
    if socrates:
        num_classes = num_classes + 1
    else:
        num_classes = num_classes

    device = jax.devices(device)[0]
    buffer_logits = jnp.empty(
        (buffer_size + 1, num_classes), device=device, dtype=jnp.float32
    )
    buffer_idx = jnp.full((buffer_size,), -1, device=device, dtype=jnp.int32)
    buffer_targets = jnp.zeros((buffer_size,), device=device, dtype=jnp.uint32)
    return buffer_idx, buffer_targets, buffer_logits


def add_to_buffer(
    sample_idx: Array,
    labels: Array,
    logits: Array,
    buffer_idx: Array,
    buffer_targets: Array,
    buffer_logits: Array,
    seen_examples: int,
    selection_method: Callable | None = None,
    device: str = "gpu",
    *,
    key: PRNGKeyArray,
) -> tuple[Array, Array, Array, int]:
    key, subkey = jax.random.split(key)

    device = jax.devices(device)[0]
    sample_idx: Array = jax.device_put(sample_idx, device)
    labels: Array = jax.device_put(labels, device)
    logits: Array = jax.device_put(logits, device)
    if selection_method is None:
        selection_method = reservoir_sampling
    buffer_idx, buffer_targets, buffer_logits, seen_examples = selection_method(
        sample_idx,
        labels,
        logits,
        buffer_idx,
        buffer_targets,
        buffer_logits,
        seen_examples,
        device=device,
        key=key,
    )
    return buffer_idx, buffer_targets, buffer_logits, seen_examples


def get_from_buffer(
    buffer_idx: Array,
    buffer_targets: Array,
    buffer_logits: Array,
    replay_size: int,
    has_buffer: bool,
    trainloader,
    *,
    key: PRNGKeyArray,
    soc: bool = False,
):
    valid_mask = buffer_idx >= 0
    num_filled = jnp.sum(valid_mask)
    has_buffer = jnp.logical_and(has_buffer, num_filled > 0)

    probs: Array = valid_mask.astype(jnp.float32)
    probs: Array = probs / jnp.maximum(jnp.sum(probs), 1.0)
    # jax.debug.print("{}", probs)
    # jax.debug.breakpoint()
    key, subkey1, subkey2 = jax.random.split(key, 3)
    buffer_samples = jax.random.choice(
        subkey1,
        buffer_idx.shape[0],
        shape=(replay_size,),
        replace=False,
        p=probs,
    )

    # jax.debug.print("{}", buffer_samples)
    # jax.debug.print("{}", key)
    idx = buffer_idx[buffer_samples]
    X = trainloader.all_data[idx]

    # jax.debug.print("{}", X)
    # jax.debug.breakpoint()
    y = buffer_targets[buffer_samples]
    # jax.debug.print("{}", y)
    logits = buffer_logits[buffer_samples]

    device = jax.devices(trainloader.iter_device)[0]

    if hasattr(trainloader, "mean") and hasattr(trainloader, "std"):
        X: Array = trainloader._norm(X, trainloader.mean, trainloader.std)

    X = trainloader.transform_batch(
        subkey2, X, trainloader.crop, trainloader.padding, trainloader.flip_p
    )
    X: Array = jax.device_put(X, device)
    y: Array = jax.device_put(y, device)
    logits: Array = jax.device_put(logits, device)
    if not soc:
        return X, y, logits, has_buffer, key
    else:
        return X, y, logits, buffer_samples, has_buffer, key


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

    batch_idxes = jnp.arange(0, batch_size, dtype=jnp.int32)

    def rand_selection(key, n, i):
        rand_idx = jax.random.randint(key, (), 0, n, dtype=jnp.int32)
        replace, choice = jax.lax.cond(
            rand_idx < buffer_size, lambda: (rand_idx, i), lambda: (buffer_size + 1, i)
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

    # jax.debug.print("replace {}", replace)
    # jax.debug.print("choices {}", choices)

    seen_examples += batch_size

    choices = jnp.array(choices, device=device, dtype=jnp.int32)
    replace = jnp.array(replace, device=device, dtype=jnp.int32)
    buffer_idx = buffer_idx.at[replace].set(sample_idx[choices], mode="drop")
    buffer_targets = buffer_targets.at[replace].set(
        labels[choices].astype(jnp.uint32), mode="drop"
    )
    buffer_logits = buffer_logits.at[replace].set(logits[choices], mode="drop")

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
