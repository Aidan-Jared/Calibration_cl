import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree, PRNGKeyArray
import equinox as eqx
import numpy as np

from tqdm import tqdm

from equinox.nn._stateful import State
from optax import softmax_cross_entropy_with_integer_labels, GradientTransformationExtraArgs

from src.socrates_loss import socrates_loss

def model_forward(
    model, x: Array, state: State, *, key: PRNGKeyArray
):
    logits, state = model(x, state, key=key)
    return logits, state


def der_loss(
    Model,
    x: Array,
    y: Array,
    state: State,
    old_logits: Array,
    buffer_size: int,
    alpha_: float = .5,
    beta: float = 0.0,
    prob_history: Array | None = None,
    indexes: Array | None = None,
    updated: Array | None = None,
    gamma: float | None = None,
    alpha: float | None = None,
    *,
    key: PRNGKeyArray,
):
    key, *keys = jax.random.split(key, x.shape[0] + 1)
    keys = jnp.array(keys)
    
    logits, state = jax.vmap(
        model_forward, in_axes = (None, 0, None, 0), out_axes = (0, None)
    )(Model, x, state, key=keys)
    
    acc = jnp.mean(jnp.argmax(logits) == y)
    
    if prob_history is None or updated is None or alpha is None or gamma is None or indexes is None:
        loss = softmax_cross_entropy_with_integer_labels(logits[:-buffer_size], y[:-buffer_size])
        loss += alpha_ * jnp.mean((logits[-buffer_size:] - old_logits)**2)
        if beta != 0.0:
            loss += beta * softmax_cross_entropy_with_integer_labels(logits[-buffer_size:], y[-buffer_size:])
        return loss, (acc, state, None, None)
    else:
        loss, updated, up_prob_history = jax.vmap(
            socrates_loss, in_axes = (0, 0, 0, 0, None, None)
        )(logits[:-buffer_size], prob_history[indexes[:-buffer_size]], y, updated, gamma, alpha)
        loss += jnp.mean((logits[-buffer_size:] - old_logits)**2)
        prob_history = prob_history.at[indexes[:-buffer_size]].set(up_prob_history)
        if beta != 0:
            loss, updated, up_prob_history = beta * jax.vmap(
                socrates_loss, in_axes = (0, 0, 0, 0, None, None)
            )(logits[-buffer_size:], prob_history[indexes[-buffer_size:]], y, updated, gamma, alpha)
        
            prob_history = prob_history.at[indexes[-buffer_size:]].set(up_prob_history)
        return loss, (acc, state, updated, prob_history)
        
def train_step(
    model,
    x: Array,
    y: Array,
    state: State,
    old_logits: Array,
    buffer_size: int,
    optim:GradientTransformationExtraArgs,
    opt_state: PyTree,
    beta: float = 0.0,
    prob_history: Array | None = None,
    indexes: Array | None = None,
    updated: Array | None = None,
    gamma: float | None = None,
    alpha: float | None = None,
    *,
    key: PRNGKeyArray,
):
    (loss, acc, state, updated, prob_history), grads = eqx.filter_value_and_grad(der_loss, has_aux = True)(
        model, x, y, state, old_logits, buffer_size, beta,
        prob_history, indexes, updated, gamma, alpha, key = key
    )
    
    updates = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    
    model = eqx.apply_updates(model, updates)
    
    return model, loss, acc, state, updated, prob_history
    
def train(
    model,
    trainloader,
    testloader,
    tasks: int,
    epochs: int,
    state: State,
    buffer_size: int,
    optim:GradientTransformationExtraArgs,
    beta: float = 0.0,
    prob_history: Array | None = None,
    updated: Array | None = None,
    gamma: float | None = None,
    alpha: float | None = None,
    print_every: int = 10,
    *,
    key: PRNGKeyArray,    
):
    for task in range(tasks):
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        for epoch in range(epochs):
            key, subkey = jax.random.split(key)
            
            epoch_loss = []
            epoch_acc = []
            
            pbar = tqdm(
                enumerate(trainloader.sample(task, key=subkey)),
                total=trainloader.iters(task),
                # ncols=75,
            )
            train_step_jit = eqx.filter_jit(train_step)
            for step, (x,y, indexes, old_logits) in pbar:
                key, subkey = jax.random.split(key)
                model, loss, acc, state, updated, prob_history = train_step_jit(
                    model, x, y, state, buffer_size, optim, opt_state, beta,
                    prob_history, indexes,updated, gamma, alpha, key = subkey
                )
                if step % print_every == 0:
                    pbar.set_postfix(
                        {
                            "task_train": task,
                            "epoch": epoch + 1,
                            "batch": step + 1,
                            "loss": np.mean(epoch_loss),
                            "acc": np.mean(epoch_acc),
                        }
                    )
