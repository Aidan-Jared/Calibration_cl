from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray
from torch.utils.data import Dataset

from src.buffer_selection import reservoir_sampling


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
        transform: bool = True,
        crop: tuple[int, int] = (32, 32),
        padding: int = 4,
        flip_p: float = 0.5,
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
        self.seen_examples = 0

        self.buff_size_mem = buff_size_mem if buff_size_mem is not None else batch_size
        self.Transform = transform
        self.crop = crop
        self.padding = padding
        self.flip_p = flip_p

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
                (self.buffer_size + 1, num_classes), device=device, dtype=jnp.float32
            )
            self.buffer_idx = jnp.full(
                (self.buffer_size + 1,), -1, device=device, dtype=jnp.int32
            )
            self.buffer_targets = jnp.zeros(
                (self.buffer_size + 1,), device=device, dtype=jnp.uint32
            )

    @staticmethod
    @jax.jit
    def _norm(X, mean, std) -> Array:
        return (X - mean) / std

    @staticmethod
    @jax.jit
    def _random_hflip(key, X, p):
        flip = jax.random.uniform(key) < p
        return jax.lax.cond(flip, lambda x: jnp.flip(x, axis=1), lambda x: x, X)

    @staticmethod
    @jax.jit(static_argnames=("crop_size", "padding"))
    def _random_crop(key, X, crop_size, padding=None):
        if padding is not None:
            X = jnp.pad(
                X, ((0, 0), (padding, padding), (padding, padding)), mode="reflect"
            )
        h, w = X.shape[1:]
        crop_h, crop_w = crop_size

        subkey1, subkey2 = jax.random.split(key)

        top = jax.random.randint(subkey1, (), 0, h - crop_h + 1)
        left = jax.random.randint(subkey2, (), 0, w - crop_w + 1)

        return jax.lax.dynamic_slice(X, (0, top, left), (X.shape[0], crop_h, crop_w))

    def _transform(self, key, X, crop_size, padding=4, flip_p=0.05):
        subkey1, subkey2 = jax.random.split(key)
        X = self._random_crop(subkey1, X, crop_size, padding)
        X = self._random_hflip(subkey2, X, flip_p)
        return X

    def transform_batch(self, key, batch, crop_size, padding=4, flip_p=0.5):
        keys = jax.random.split(key, batch.shape[0])
        return jax.vmap(
            lambda k, img: self._transform(k, img, crop_size, padding, flip_p),
            in_axes=(0, 0),
        )(keys, batch)

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
        self,
        X: Array,
        y: Array,
        logits: Array | None,
        class_idx: Array,
        task: int,
        device: jax.Device,
        *,
        key,
    ) -> tuple[Array, Array, Array, int, Array]:
        if logits is None:
            logits = jnp.zeros((1, self.num_classes))
        if self.Transform:
            X = self.transform_batch(key, X, self.crop, self.padding, self.flip_p)
        X: Array = X.astype(jnp.float32) / 255.0
        if hasattr(self, "mean") and hasattr(self, "std"):
            X: Array = self._norm(X, self.mean, self.std)
        X: Array = jax.device_put(X, device)
        y: Array = jax.device_put(y, device)
        logits: Array = jax.device_put(logits, device)
        class_idx: Array = jax.device_put(class_idx, device)

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

        if self.buffer and (task_n > 0 and jnp.any(self.buffer_idx > 0)):
            filled = int(jnp.sum(self.buffer_idx[:-1] >= 0))
        else:
            filled = None

        device = jax.devices(self.iter_device)[0]

        def raw_generator():
            nonlocal key
            for i in range(batches):
                if self.buffer and filled is not None:
                    key, subkey = jax.random.split(key)
                    buffer_samples = jax.random.choice(
                        subkey, filled, shape=(self.buff_size_mem,), replace=True
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
                    idx = class_idx[i]

                yield (X, y, logits, idx, task_n)

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
        sample_idx: Array,
        labels: Array,
        logits: Array,
        selection_method: Callable | None = None,
        *,
        key: PRNGKeyArray,
    ):
        if not self.buffer:
            return
        key, subkey = jax.random.split(key)

        device: jax.Device = jax.devices(self.device)[0]
        sample_idx: Array = jax.device_put(sample_idx, device)
        labels: Array = jax.device_put(labels, device)
        logits: Array = jax.device_put(logits, device)
        if selection_method is None:
            selection_method = reservoir_sampling
        self.buffer_idx, self.buffer_targets, self.buffer_logits, self.seen_examples = (
            selection_method(
                sample_idx,
                labels,
                logits,
                self.buffer_idx,
                self.buffer_targets,
                self.buffer_logits,
                self.seen_examples,
                device=device,
                key=key,
            )
        )
