import jax
import jax.numpy as jnp

from jaxtyping import Array


def ece(logits: Array, y: Array, M: int):
    bin_edges = jnp.linspace(0, 1, M + 1)
    bin_idx = jnp.digitize(logits, bin_edges)
    masks = jax.nn.one_hot(bin_idx, M + 1, dtype=jnp.bool).T
    bins = jnp.where(masks, logits[None, :], 0)
    bin_count = jnp.sum(bins > 0, axis=1)
    conf = jnp.sum(bins, axis=1) / bin_count
    acc = (
        jnp.sum(jnp.where(masks, (jnp.argmax(logits, axis=1) == y)[:None], 0))
        / bin_count
    )
    return jnp.sum(bin_count / y.shape[0] * jnp.abs(acc - conf))
