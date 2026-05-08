import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from torch.utils.data import Dataset
from jaxtyping import PRNGKeyArray
from typing import Callable
from concurrent.futures import ThreadPoolExecutor
from src.models.resnet18 import ResNet18
from src.models.resnet32 import ResNet32
from src.util import model_forward


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
        self.splits = splits
        self.batch_size = batch_size
        self.seen_tasks = []
        self.device = device
        self.iter_device = iter_device
        self.workers = workers
        self.buffer = buffer
        self.buffer_size = buffer_size
        self.dtype = dtype

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
            self.buffer_logits = jnp.zeros((self.buffer_size, self.num_classes + 1), device=device)
            self.buffer_idx = jnp.zeros((self.buffer_size,), device=device)
            self.buffer_targets = jnp.zeros((self.buffer_size, self.num_classes), device=device)

    @staticmethod
    @jax.jit
    def _norm(X, mean, std):
        return (X - mean) / std
    
    def __len__(self) -> int:
        return self.len

    def normalize(
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
        X = jax.jit(lambda x: x / 255.0)(X)
        if hasattr(self, "mean") and hasattr(self, "std"):
            X = self._norm(X, self.mean, self.std)
        X = jax.device_put(X, device)
        y = jax.device_put(y, device)
        return X.astype(self.dtype), y.astype(jnp.int32), class_idx

    def sample(self, task_n: int, *, key: PRNGKeyArray):

        task_idx = self.tasks[task_n]
        n = jnp.sum(self.class_lengths[task_idx]).item()
        class_idx = self.class_indicies[task_idx].reshape(-1)
        labels = np.repeat(task_idx, self.class_lengths[task_idx])
        key, subkey = jax.random.split(key)
            
        if key is not None:
            shuffle = jax.random.permutation(key=key, x=n)
            class_idx = class_idx[shuffle]
            labels = labels[shuffle]

        batches = n // self.batch_size
        class_idx = class_idx.reshape(
            batches, self.batch_size
        )
        labels = labels.reshape(batches, self.batch_size)

        if self.buffer and (task_n > 0 or jnp.any(self.buffer_idx > 0)):
            n_buff = self.buffer_idx.shape[0] // batches
            filled = int(jnp.sum(self.buffer_idx > 0))
            key, subkey = jax.random.split(key)
            idx = jax.random.choice(subkey, filled, shape=(n_buff * batches,)).reshape(batches, n_buff)

            class_idx = jnp.concatenate([class_idx, self.buffer_idx[idx]], axis=1)
            labels = jnp.concatenate([labels, self.buffer_targets[idx]], axis=1)
            logits = self.buffer_logits[idx]
        else:
            logits = None
        
        device = jax.devices(self.iter_device)[0]

        def raw_generator():
            for i in range(batches):
                X = self.all_data[class_idx[i]]
                y = labels[i] 
                yield (X, y, class_idx[i], task_n)

        for i, (X, y, class_idx_i, task_n_i) in enumerate(raw_generator()):
            if logits is not None:
                yield (X, y, class_idx_i, task_n_i, logits[i])
            else:
                yield (X, y, class_idx_i, task_n_i, None)

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
            task_idx = self.tasks[:task_n]
            slots_per_task = self.buffer_size // task_n
            start = 0
            end = slots_per_task
            for task in task_idx:
                labels = np.repeat(task_idx, self.class_lengths[task])
                key, subkey = jax.random.split(key)
                tidx = jax.random.choice(subkey, self.class_indicies[task], shape = (slots_per_task,))
                self.buffer_idx = self.buffer_idx.at[start:end].set(tidx)
                self.buffer_targets = self.buffer_targets.at[start:end].set(labels[task, tidx])
                X = self.all_data[tidx[start:end]]
                if hasattr(self, "mean") and hasattr(self, "std"):
                    X = self._norm(X, self.mean, self.std)
                X = jax.device_put(
                    X,
                    jax.devices(self.iter_device)[0]
                )
                logits, _ = jax.vmap(model_forward, in_axes = (None, 0, None, None))(model, X, state, key)
                logits = jax.device_put(
                    logits,
                    jax.devices(self.device)[0]
                )
                self.buffer_logits = self.buffer_logits.at[start:end].set(logits)
                start = end
                end = start + slots_per_task