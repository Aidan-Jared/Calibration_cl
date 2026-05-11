import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from equinox.nn._stateful import State

def model_forward(
    model,
    X: Array,
    state: State,
    key: PRNGKeyArray,
) -> tuple[Array, State]:
    y, state = model(X, state, key=key)
    return y, state