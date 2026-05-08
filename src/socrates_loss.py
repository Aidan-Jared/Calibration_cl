import jax
import jax.numpy as jnp
from jaxtyping import Array

def adaptive_target(
    prob: Array,
    prob_history: Array,
    target,
    updated: Array,
    alpha: float,   
) -> tuple[Array, Array, Array]:
    oh_target = jnp.zeros_like(prob_history).at[target].set(1)
    
    p_momentum = jnp.where(updated == 1, prob_history, oh_target)
    
    p_momentum = alpha * p_momentum + (1 - alpha) * prob
    updated = jnp.array(1)
    
    prob_history = p_momentum
    
    return p_momentum, updated, prob_history    

def socrates_loss(
    logits: Array,
    prob_history: Array,
    target: Array,
    updated: Array,
    gamma: float,
    alpha: float,
):
    probs = jax.nn.softmax(
        jax.lax.stop_gradient(logits)
    )
    
    t, updated, prob_history = adaptive_target(
        probs[:-1], prob_history,target, updated, alpha
    )
    
    beta = jnp.max(probs[:-1]) - probs[-1]    
    adaptive_component = t * jnp.log(probs[:-1]) + beta * (1 - t) * jnp.log(probs[-1])
    
    socrates = jnp.sum(jnp.power(1 - probs[:-1], gamma) * adaptive_component)
    
    return -socrates / (probs.shape[-1] - 1), updated, prob_history

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    logits = jax.random.normal(key, (2, 5))
    targets = jnp.array([[1], [2]], jnp.uint8)
    prob_history = jnp.zeros(logits[:,:-1].shape, dtype=jnp.float32)
    updated = jnp.zeros(targets.shape[0])
    gamma = 2.0
    alpha = 0.9
    index = jnp.arange(logits.shape[0])
    
    socrates_loss_vmap = jax.vmap(socrates_loss, in_axes=(0, 0, 0, 0, None, None))
    for _ in range(2):
        loss, updated, up_prob_history = socrates_loss_vmap(logits, prob_history[index], targets, updated, gamma, alpha)
        prob_history = prob_history.at[index].set(up_prob_history)
        print(loss)
    