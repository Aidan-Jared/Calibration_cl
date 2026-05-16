import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from torch.utils.data import Dataset
from jaxtyping import PRNGKeyArray, Array
from typing import Callable
from concurrent.futures import ThreadPoolExecutor
from src.models.resnet18 import ResNet18
from src.models.resnet32 import ResNet32
from src.utils import model_forward


class CL_DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        splits: int,
        device: str = "cpu",
        iter_device: str = "gpu",
        workers: int = 1,
        buffer: bool = False,
        buffer_size: int = 100,
        buff_size_mem: int | None = None,
        socrates: bool = False,
        dtype=jnp.float32,
        *,
        key: PRNGKeyArray | None = None,
    ) -> None:
        self.splits: int = splits
        self.batch_size: int = batch_size
        self.seen_tasks: list[int] = []
        self.device: str = device
        self.iter_device: str = iter_device
        self.workers: int = workers
        self.buffer: bool = buffer
        self.buffer_size: int = buffer_size

        self.buff_size_mem = buff_size_mem if buff_size_mem is not None else batch_size

        self.dtype: jnp.dtype = dtype

        self.len = getattr(dataset, "__len__", batch_size)

        class_to_indices = {}
        all_data = []
        for idx, (data, label) in enumerate(dataset):  # type: ignore
            if isinstance(data, jnp.ndarray):
                all_data.append(np.array(data))
            else:
                all_data.append(data.numpy() * 255)

            label_int = int(label)
            if label_int not in class_to_indices:
                class_to_indices[label_int] = []

            class_to_indices[label_int].append(idx)

        device = jax.devices(device)[0]

        all_data_np = np.stack(all_data)

        self.all_data = jax.device_put(all_data_np, device).astype(jnp.uint8)

        self.num_classes = len(class_to_indices)

        max_samples_per_class = max(len(v) for v in class_to_indices.values())

        self.class_indicies = jax.device_put(
            jnp.full((self.num_classes, max_samples_per_class), -1, dtype=jnp.int32),
            device,
        )

        self.class_lengths = jax.device_put(
            jnp.zeros(self.num_classes, dtype=jnp.int32), device
        )

        for class_idx, (label, idx) in enumerate(sorted(class_to_indices.items())):
            num_samples = len(idx)
            self.class_indicies = self.class_indicies.at[class_idx, :num_samples].set(
                jnp.array(idx, dtype=jnp.int32)
            )

            self.class_lengths = self.class_lengths.at[class_idx].set(num_samples)

        if key is None:
            self.tasks = np.arange(self.num_classes).reshape((self.splits, -1))

        else:
            self.tasks = jax.random.choice(
                key, self.num_classes, (self.num_classes,), replace=False
            ).reshape(self.splits, -1)
            self.tasks = jax.device_put(self.tasks, device)

        if self.buffer:
            if socrates:
                num_classes = self.num_classes + 1
            else:
                num_classes = self.num_classes

            self.buffer_logits = jnp.empty(
                (self.buffer_size, num_classes), device=device, dtype=jnp.float32
            )
            self.buffer_idx = jnp.full(
                (self.buffer_size,), -1, device=device, dtype=jnp.int32
            )
            self.buffer_targets = jnp.zeros(
                (self.buffer_size,), device=device, dtype=jnp.uint32
            )

    @staticmethod
    @jax.jit
    def _norm(X, mean, std) -> Array:
        return (X - mean) / std

    def __len__(self) -> int:
        return self.len

    def normilization_values(
        self,
        mean: tuple | float,
        std: tuple | float,
    ):
        self.mean = jnp.array(mean)  # typing:ignore
        self.std = jnp.array(std)  # typing:ignore
        self.mean = jnp.expand_dims(self.mean, axis=(0, 2, 3))
        self.std = jnp.expand_dims(self.std, axis=(0, 2, 3))

    def iters(self, task_n: int) -> int:
        task_idx = self.tasks[task_n]
        n = jnp.sum(self.class_lengths[task_idx]).item()
        return n // self.batch_size

    def update_batch_size(self, new_batch_size: int):
        self.batch_size = new_batch_size

    def _prepare_batch(
        self, X, y, logits, class_idx, task, device, *, key
    ) -> tuple[Array, Array, Array, int, Array]:
        if logits is None:
            logits = jnp.zeros((1,))
        X: Array = X.astype(jnp.float32) / 255.0
        if hasattr(self, "mean") and hasattr(self, "std"):
            X = self._norm(X, self.mean, self.std)
        X = jax.device_put(X, device)
        y = jax.device_put(y, device)
        logits = jax.device_put(logits, device)
        return X.astype(self.dtype), y.astype(jnp.int32), class_idx, task, logits

    def sample(self, task_n: int, *, key: PRNGKeyArray):
        task_idx: Array[int] = self.tasks[task_n]
        n: int = jnp.sum(self.class_lengths[task_idx]).item()
        class_idx: Array[int] = self.class_indicies[task_idx].reshape(-1)
        labels: Array[int] = np.repeat(task_idx, self.class_lengths[task_idx])
        key, subkey = jax.random.split(key)

        if key is not None:
            shuffle = jax.random.permutation(key=key, x=n)
            class_idx: Array[int] = class_idx[shuffle]
            labels: Array[int] = labels[shuffle]

        batches = n // self.batch_size
        class_idx = class_idx[: batches * self.batch_size].reshape(
            batches, self.batch_size
        )
        labels = labels[: batches * self.batch_size].reshape(batches, self.batch_size)

        if self.buffer and (task_n > 0 or jnp.any(self.buffer_idx > 0)):
            filled = int(jnp.sum(self.buffer_idx >= 0))
        else:
            filled = None

        device = jax.devices(self.iter_device)[0]

        def raw_generator():
            nonlocal key
            for i in range(batches):
                if self.buffer and filled is not None:
                    key, subkey = jax.random.split(key)
                    buffer_samples = jax.random.choice(
                        subkey, filled, shape=(self.buff_size_mem,), replace=False
                    )

                    idx = jnp.concatenate(
                        (class_idx[i], self.buffer_idx[buffer_samples])
                    )

                    X = self.all_data[idx]

                    y: Array[int] = jnp.concatenate(
                        (labels[i], self.buffer_targets[buffer_samples])
                    )
                    logits = self.buffer_logits[buffer_samples]

                else:
                    X: Array = self.all_data[class_idx[i]]
                    y: Array[int] = labels[i]
                    logits = None

                yield (X, y, logits, class_idx[i], task_n)

        yield from self._prefetch(raw_generator(), device, key=key)

    def _prefetch(self, generator, device, *, key):
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for item in generator:
                if key is not None:
                    key, subkey = jax.random.split(key)
                else:
                    subkey = None
                futures.append(
                    executor.submit(
                        self._prepare_batch,
                        item[0],
                        item[1],
                        item[2],
                        item[3],
                        item[4],
                        device,
                        key=subkey,
                    )
                )
            while futures:
                yield futures.pop(0).result()

    def add_to_buffer(
        self,
        task_n: int,
        model: ResNet18 | ResNet32,
        state: eqx.nn._stateful.State,
        selection_method: Callable | None = None,
        *,
        key: PRNGKeyArray,
    ):
        if not self.buffer:
            return
        model = eqx.nn.inference_mode(model, True)
        key, subkey = jax.random.split(key)
        if selection_method is None:
            task_idx: Array[int] = self.tasks[: task_n + 1]
            slots_per_task: int = self.buffer_size // task_idx.size
            start = 0
            end: int = slots_per_task
            for c in task_idx.flatten():
                labels = np.repeat(c, slots_per_task)
                key, subkey = jax.random.split(key)
                tidx = jax.random.choice(
                    subkey,
                    self.class_indicies[c],
                    shape=(slots_per_task,),
                    replace=False,
                )
                self.buffer_idx = self.buffer_idx.at[start:end].set(tidx)
                self.buffer_targets = self.buffer_targets.at[start:end].set(labels)
                X = self.all_data[tidx]
                if hasattr(self, "mean") and hasattr(self, "std"):
                    X = self._norm(X.astype(np.float32) / 255, self.mean, self.std)

                X = jax.device_put(X, jax.devices(self.iter_device)[0])
                logits, _ = jax.vmap(
                    model_forward,
                    in_axes=(None, 0, None, None),
                    out_axes=(0, None),
                    axis_name="batch",
                )(model, X, state, key)
                logits = jax.device_put(logits, jax.devices(self.device)[0])
                self.buffer_logits = self.buffer_logits.at[start:end].set(logits)
                start = end
                end = start + slots_per_task
        else:
            self.buffer_idx, self.buffer_targets, self.buffer_logits = selection_method(
                self,
                task_n,
                self.buffer_idx,
                self.buffer_targets,
                self.buffer_logits,
                model,
                state,
                key=key,
            )
