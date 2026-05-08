import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from equinox.nn._stateful import State

def model_forward(
    model,
    X: Array,
    state: State,
    *,
    key: PRNGKeyArray,
) -> tuple[Array, State]:
    key, *keys = jax.random.split(key, X.shape[0])
    keys = jnp.concatenate([key, *keys])
    y, state = jax.vmap(model,
        in_axes=(0, None, 0),
        out_axes=(0, None),
        axis_name="batch"
    )(X, state, key=keys)
    return y, state