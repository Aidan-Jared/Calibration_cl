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
        dtype=jnp.float32,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.splits: int = splits
        self.batch_size: int = batch_size
        self.seen_tasks: list[int] = []
        self.device: str = device
        self.iter_device: str = iter_device
        self.workers: int = workers
        self.buffer: bool = buffer
        self.buffer_size: int = buffer_size
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

        self.tasks = np.arange(self.num_classes).reshape((self.splits, -1))
        if self.buffer:
            self.buffer_logits = jnp.zeros((self.buffer_size, self.num_classes + 1), device=device, dtype=jnp.float32)
            self.buffer_idx = jnp.full((self.buffer_size,), -1, device=device, dtype=jnp.int32)
            self.buffer_targets = jnp.zeros((self.buffer_size,), device=device, dtype=jnp.uint32)

    @staticmethod
    @jax.jit
    def _norm(X, mean, std):
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

    def _prepare_batch(self, X, y, class_idx, task, device, *, key):
        X = jax.jit(lambda x: x / 255.0)(X.astype(jnp.float32))
        if hasattr(self, "mean") and hasattr(self, "std"):
            X = self._norm(X, self.mean, self.std)
        X = jax.device_put(X, device)
        y = jax.device_put(y, device)
        return X.astype(self.dtype), y.astype(jnp.int32), class_idx, task

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
            n_buff = self.buffer_idx.shape[0] // batches
            filled = int(jnp.sum(self.buffer_idx >= 0))
            key, subkey = jax.random.split(key)
            idx = jax.random.choice(subkey, filled, shape=(n_buff * batches,)).reshape(batches, n_buff)

            class_idx: Array[int] = jnp.concatenate([class_idx, self.buffer_idx[idx]], axis=1)
            labels: Array[int] = jnp.concatenate([labels, self.buffer_targets[idx]], axis=1)
            logits: Array[float] = self.buffer_logits[idx]
        else:
            logits = None
        
        device = jax.devices(self.iter_device)[0]

        def raw_generator():
            for i in range(batches):
                X: Array = self.all_data[class_idx[i]]
                y:Array[int] = labels[i] 
                yield (X, y, class_idx[i], task_n)

        
        if logits is not None:
            i = 0
            for X, y, class_idx_i, task_n_i in self._prefetch(raw_generator(), device, key=key):
                old_logits = jax.device_put(logits[i], device)
                class_idx_i = jax.device_put(class_idx_i, device)
                yield (X, y, class_idx_i, task_n_i, old_logits)
                i += 1
        else:
            for X, y, class_idx_i, task_n_i in self._prefetch(raw_generator(), device, key=key):
                class_idx_i = jax.device_put(class_idx_i, device)
                yield (X, y, class_idx_i, task_n_i, jnp.zeros((1,)))

    def _prefetch(self, generator, device, *, key):
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for item in generator:
                if key is not None:
                    key, subkey = jax.random.split(key)
                else:
                    subkey = None
                futures.append(
                    executor.submit(self._prepare_batch, item[0], item[1], item[2], item[3], device, key=subkey)
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
            task_idx: Array[int] = self.tasks[:task_n + 1]
            slots_per_task: int = self.buffer_size // task_idx.size
            start = 0
            end: int = slots_per_task
            for c in task_idx.flatten():
                labels = np.repeat(c, slots_per_task)
                key, subkey = jax.random.split(key)
                tidx = jax.random.choice(subkey, self.class_indicies[c], shape = (slots_per_task,))
                self.buffer_idx = self.buffer_idx.at[start:end].set(tidx)
                self.buffer_targets = self.buffer_targets.at[start:end].set(labels)
                X = self.all_data[tidx]
                if hasattr(self, "mean") and hasattr(self, "std"):
                    norm_fn = jax.jit(lambda x : self._norm(x / 255, self.mean, self.std))
                    X = norm_fn(X.astype(np.float32))
                X = jax.device_put(
                    X,
                    jax.devices(self.iter_device)[0]
                )
                logits, _ = jax.vmap(model_forward,
                    in_axes = (None, 0, None, None),
                    out_axes = (0, None),
                    axis_name = "batch",
                )(model, X, state, key)
                logits = jax.device_put(
                    logits,
                    jax.devices(self.device)[0]
                )
                self.buffer_logits = self.buffer_logits.at[start:end].set(logits)
                start = end
                end = start + slots_per_task