import jax
import jax.numpy as jnp

from jaxtyping import Array


def ECE(logits: Array, y: Array, M: int):
    probs = jax.nn.softmax(logits, axis=-1)
    confidences = jnp.max(probs, axis=-1)
    predictions = jnp.argmax(probs, axis=-1)
    accuracies = (predictions == y).astype(jnp.float32)

    bin_edges = jnp.linspace(0, 1, M + 1)
    bin_idx = jnp.clip(jnp.digitize(confidences, bin_edges) - 1, 0, M - 1)

    masks = jax.nn.one_hot(bin_idx, M, dtype=jnp.bool_).T

    bin_count = jnp.sum(masks, axis=1)
    conf_sum = jnp.sum(jnp.where(masks, confidences[None, :], 0), axis=1)
    acc_sum = jnp.sum(jnp.where(masks, accuracies[None, :], 0), axis=1)

    safe_count = jnp.where(bin_count > 0, bin_count, 1)
    conf = conf_sum / safe_count
    acc = acc_sum / safe_count

    weight = bin_count / y.shape[0]
    return jnp.sum(jnp.where(bin_count > 0, weight * jnp.abs(acc - conf), 0))
