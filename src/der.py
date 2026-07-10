from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox.nn._stateful import State
from jaxtyping import Array, PRNGKeyArray, PyTree
from optax import (
    GradientTransformationExtraArgs,
    softmax_cross_entropy_with_integer_labels,
)
from tqdm import tqdm

from src.dataloader import CL_DataLoader
from src.buffer_selection import make_buffer, add_to_buffer, get_from_buffer
from src.socrates_loss import socrates_loss
from src.utils import eval, model_forward


def der_loss(
    Model,
    x: Array,
    y: Array,
    state: State,
    trainloader: CL_DataLoader,
    buffer_idx: Array,
    buffer_targets: Array,
    buffer_logits: Array,
    replay_size: int,
    has_buffer: bool,
    batch_size: int,
    der_alpha: float = 0.5,
    beta: float = 0.0,
    prob_history: Array | None = None,
    indexes: Array | None = None,
    updated: Array | None = None,
    gamma: float | None = None,
    soc_alpha: float | None = None,
    *,
    key: PRNGKeyArray,
):
    key, *keys = jax.random.split(key, x.shape[0] + 1)
    keys = jnp.array(keys)

    logits, state = jax.vmap(
        model_forward,
        in_axes=(None, 0, None, 0),
        out_axes=(0, None),
        axis_name="batch",
    )(Model, x, state, keys)

    acc = jnp.mean(jnp.argmax(logits, axis=1) == y)

    key, *keys = jax.random.split(key, x.shape[0] + 1)
    keys = jnp.array(keys)

    def _der_loss(
        key,
        buffer_idx,
        buffer_targets,
        buffer_logits,
        replay_size,
        has_buffer,
        trainloader,
        Model,
        state,
        keys,
    ):
        params, static = eqx.partition(Model, eqx.is_array)
        x, y, old_logits, has_buffer, key = get_from_buffer(
            buffer_idx,
            buffer_targets,
            buffer_logits,
            replay_size,
            has_buffer,
            trainloader,
            key=key,
        )

        def true_fn(operands):
            M, x_b, s, k, target_logits = operands
            M = eqx.combine(M, static)
            preds = jax.vmap(
                model_forward,
                in_axes=(None, 0, None, 0),
                out_axes=(0, None),
                axis_name="batch",
            )(M, x_b, s, k)[0]
            return der_alpha * jnp.mean((preds - target_logits) ** 2)

        loss = jax.lax.cond(
            has_buffer,
            true_fn,
            lambda operands: 0.0,
            operand=(
                params,
                x,
                state,
                keys,
                old_logits,
            ),
        )
        return loss

    if (
        prob_history is None
        or updated is None
        or soc_alpha is None
        or gamma is None
        or indexes is None
    ):
        loss = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, y))

        # jax.debug.breakpoint()

        key, subkey = jax.random.split(key)
        loss += _der_loss(
            subkey,
            buffer_idx,
            buffer_targets,
            buffer_logits,
            replay_size,
            has_buffer,
            trainloader,
            Model,
            state,
            keys,
        )
        # jax.debug.breakpoint()
        if beta != 0:
            key, subkey = jax.random.split(key)
            x, y, old_logits, has_buffer, key = get_from_buffer(
                buffer_idx,
                buffer_targets,
                buffer_logits,
                replay_size,
                has_buffer,
                trainloader,
                key=subkey,
            )

            key, *keys_list = jax.random.split(key, x.shape[0] + 1)
            keys = jnp.array(keys_list)
            params, static = eqx.partition(Model, eqx.is_array)

            def beta_true_fn(operands):
                M, x_b, y_b, k_b = operands
                M = eqx.combine(M, static)
                preds = jax.vmap(
                    model_forward,
                    in_axes=(None, 0, None, 0),
                    out_axes=(0, None),
                    axis_name="batch",
                )(M, x_b, state, k_b)[0]

                return beta * jnp.mean(
                    softmax_cross_entropy_with_integer_labels(preds, y_b)
                )

            loss += jax.lax.cond(
                has_buffer,
                beta_true_fn,
                lambda operands: 0.0,
                operand=(
                    params,
                    x,
                    y,
                    keys,
                ),
            )
        return loss, (logits, acc, state, None, None)  # typing: ignore
    else:
        loss, up_prob_history = jax.vmap(
            socrates_loss, in_axes=(0, 0, 0, 0, None, None)
        )(
            logits,
            prob_history,
            y,
            updated,
            gamma,
            soc_alpha,
        )
        loss = jnp.mean(loss)

        loss += _der_loss(
            key,
            buffer_idx,
            buffer_targets,
            buffer_logits,
            replay_size,
            has_buffer,
            trainloader,
            Model,
            state,
            keys,
        )
        prob_history = prob_history.at[indexes].set(up_prob_history)
        updated = updated.at[indexes].set(1)

        if beta != 0:

            def socrates_loss_with_old_logits(
                x, y, state, prob_history, indexes, updated, gamma, soc_alpha
            ):
                logits = jax.vmap(
                    model_forward,
                    in_axes=(None, 0, None, 0),
                    out_axes=(0, None),
                    axis_name="batch",
                )(Model, x, state, keys)[0]

                sloss, up_prob_history = jax.vmap(
                    socrates_loss, in_axes=(0, 0, 0, 0, None, None)
                )(
                    logits,
                    prob_history[indexes],
                    y,
                    updated[indexes],
                    gamma,
                    soc_alpha,
                )

                prob_history = prob_history.at[indexes].set(up_prob_history)
                updated = updated.at[indexes].set(1)

                return jnp.mean(sloss), prob_history, updated

            key, subkey = jax.random.split(key)
            x, y, old_logits, indexes, has_buffer, key = get_from_buffer(
                buffer_idx,
                buffer_targets,
                buffer_logits,
                replay_size,
                has_buffer,
                trainloader,
                key=subkey,
                soc=True,
            )
            sloss, prob_history, updated = jax.lax.cond(
                has_buffer,
                lambda: socrates_loss_with_old_logits(
                    x, y, state, prob_history, indexes, updated, gamma, soc_alpha
                ),
                lambda: (jnp.array(0.0), prob_history, updated),
            )
            loss = loss + beta * sloss

        return loss, (logits, acc, state, updated, prob_history)


def train_step(
    model,
    x: Array,
    y: Array,
    state: State,
    trainloader: CL_DataLoader,
    buffer_idx: Array,
    buffer_targets: Array,
    buffer_logits: Array,
    replay_size: int,
    seen_examples: int,
    has_buffer: bool,
    batch_size: int,
    optim: GradientTransformationExtraArgs,
    opt_state: PyTree,
    der_alpha: float = 0.5,
    beta: float = 0.0,
    selection_method: Callable | None = None,
    prob_history: Array | None = None,
    indexes: Array | None = None,
    updated: Array | None = None,
    gamma: float | None = None,
    soc_alpha: float | None = None,
    *,
    key: PRNGKeyArray,
):
    (
        subkey1,
        subkey2,
    ) = jax.random.split(key)

    # 2. Define a wrapper function that recombines them so JAX only traces the arrays
    # def trace_wrapper(
    #     dyn_model,
    #     x,
    #     y,
    #     state,
    #     buffer_idx,
    #     buffer_targets,
    #     buffer_logits,
    #     prob_history,
    #     indexes,
    #     updated,
    #     key,
    # ):
    #     full_model = eqx.combine(dyn_model, static_model)
    #     loss_and_grad_fn = eqx.filter_value_and_grad(der_loss, has_aux=True)
    #     return loss_and_grad_fn(
    #         full_model,
    #         x,
    #         y,
    #         state,
    #         trainloader,
    #         buffer_idx,
    #         buffer_targets,
    #         buffer_logits,
    #         replay_size,
    #         has_buffer,
    #         batch_size,
    #         der_alpha,
    #         beta,
    #         prob_history,
    #         indexes,
    #         updated,
    #         gamma,
    #         soc_alpha,
    #         key=key,
    #     )

    # # 3. Print the Jaxpr safely using only the dynamic/array arguments
    # lowered = jax.make_jaxpr(trace_wrapper)(
    #     dynamic_model,
    #     x,
    #     y,
    #     state,
    #     buffer_idx,
    #     buffer_targets,
    #     buffer_logits,
    #     prob_history,
    #     indexes,
    #     updated,
    #     subkey1,
    # )
    # jax.debug.print("my thing: {}", lowered)

    # jax.debug.breakpoint()
    (loss, (logits, acc, state, updated, prob_history)), grads = (
        eqx.filter_value_and_grad(der_loss, has_aux=True)(
            model,
            x,
            y,
            state,
            trainloader,
            buffer_idx,
            buffer_targets,
            buffer_logits,
            replay_size,
            has_buffer,
            batch_size,
            der_alpha,
            beta,
            prob_history,
            indexes,
            updated,
            gamma,
            soc_alpha,
            key=subkey1,
        )
    )
    # jax.debug.breakpoint()
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))

    buffer_idx, buffer_targets, buffer_logits, seen_examples = add_to_buffer(
        indexes,
        y,
        logits,
        buffer_idx,
        buffer_targets,
        buffer_logits,
        seen_examples,
        selection_method,
        key=subkey2,
    )
    model = eqx.apply_updates(model, updates)

    return (
        model,
        logits,
        loss,
        acc,
        state,
        updated,
        prob_history,
        opt_state,
        buffer_idx,
        buffer_targets,
        buffer_logits,
        seen_examples,
    )


def DER_train(
    model,
    trainloader: CL_DataLoader,
    testloader: CL_DataLoader,
    tasks: int,
    epochs: int,
    state: State,
    optim: GradientTransformationExtraArgs,
    buffer_size: int,
    replay_size: int,
    der_alpha: float = 0.5,
    beta: float = 0.0,
    selection_method: Callable | None = None,
    prob_history: Array | None = None,
    updated: Array | None = None,
    gamma: float | None = None,
    soc_alpha: float | None = None,
    print_every: int = 10,
    *,
    key: PRNGKeyArray,
):
    batch_size = trainloader.batch_size
    results = []
    train_step_jit = eqx.filter_jit(train_step)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    if prob_history is not None:
        soc = True
    else:
        soc = False
    buffer_idx, buffer_targets, buffer_logits = make_buffer(
        buffer_size, trainloader.num_classes, socrates=soc
    )
    seen_examples = 0
    has_buffer = True
    for task in range(tasks):
        model = eqx.nn.inference_mode(model, False)
        print(f"training task {task}")
        print("-" * 50)
        for epoch in range(epochs):
            key, subkey = jax.random.split(key)

            epoch_loss = []
            epoch_acc = []

            model = eqx.nn.inference_mode(model, False)
            pbar = tqdm(
                enumerate(trainloader.sample(task, beta=beta, key=subkey)),
                total=trainloader.iters(task),
            )  # train_step_jit = train_step
            for step, (x, y, indexes, task_n) in pbar:
                key, subkey1, subkey2 = jax.random.split(key, 3)

                (
                    model,
                    logits,
                    loss,
                    acc,
                    state,
                    updated,
                    prob_history,
                    opt_state,
                    buffer_idx,
                    buffer_targets,
                    buffer_logits,
                    seen_examples,
                ) = train_step_jit(
                    model,
                    x,
                    y,
                    state,
                    trainloader,
                    buffer_idx,
                    buffer_targets,
                    buffer_logits,
                    replay_size,
                    seen_examples,
                    has_buffer,
                    batch_size,
                    optim,
                    opt_state,
                    der_alpha,
                    beta,
                    selection_method,
                    prob_history,
                    indexes,
                    updated,
                    gamma,
                    soc_alpha,
                    key=subkey1,
                )

                epoch_loss.append(loss)
                epoch_acc.append(acc)
                if (step + 1) % print_every == 0:
                    pbar.set_postfix(
                        {
                            "task_train": task,
                            "epoch": epoch + 1,
                            "batch": step + 1,
                            "loss": np.mean(epoch_loss),
                            "acc": np.mean(epoch_acc),
                        }
                    )
                    epoch_loss = []
                    epoch_acc = []

            print("task eval")

            model_forward_jit = eqx.filter_jit(model_forward)
            eval_acc = []
            eval_loss = []
            model = eqx.nn.inference_mode(model, True)
            for step, (x, y, indexes, task_n) in enumerate(
                testloader.sample(task, beta=beta, key=subkey)
            ):
                logits, _ = jax.vmap(
                    model_forward_jit,
                    in_axes=(None, 0, None, None),
                    out_axes=(0, None),
                    axis_name="batch",
                )(model, x, state, key)

                eval_loss.append(softmax_cross_entropy_with_integer_labels(logits, y))
                eval_acc.append(jnp.mean(jnp.argmax(logits, axis=1) == y))
                if (step + 1) == testloader.iters(task):
                    print("eval loss: ", np.mean(eval_loss))
                    print("eval acc: ", np.mean(eval_acc))
                    print()

        print("eval")
        print("-" * 50)
        res = eval(model, state, tasks, testloader, key=subkey)

        results.append(res)
    return results
