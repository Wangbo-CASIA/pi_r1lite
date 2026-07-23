from collections.abc import Iterator, Sequence
import json
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class ActionLossMaskDataset(Dataset[dict]):
    """Adds an action loss mask while retaining the full underlying episodes."""

    def __init__(
        self,
        dataset: Dataset[dict],
        action_horizon: int,
        *,
        sample_mode: Literal["all", "intervention", "valid_non_intervention"] = "all",
        intervention_ranges_path: str | None = None,
        fps: float | None = None,
        pre_intervention_seconds: float = 2.0,
    ):
        self._dataset = dataset
        self._action_horizon = action_horizon
        self._sample_mode = sample_mode
        self._valid_action_frames: list[np.ndarray] = []
        self._samples = self._load_samples(intervention_ranges_path, fps, pre_intervention_seconds)

    def _load_samples(
        self,
        intervention_ranges_path: str | None,
        fps: float | None,
        pre_intervention_seconds: float,
    ) -> list[tuple[int, int, int, int | None]] | None:
        if self._sample_mode == "all":
            return None
        if intervention_ranges_path is None:
            raise ValueError(f"sample_mode={self._sample_mode!r} requires intervention_ranges_path.")
        if fps is None or fps <= 0:
            raise ValueError(f"sample_mode={self._sample_mode!r} requires a positive dataset fps.")
        if pre_intervention_seconds < 0:
            raise ValueError("pre_intervention_seconds must be non-negative.")

        metadata_path = pathlib.Path(intervention_ranges_path)
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Intervention metadata not found: {metadata_path}")

        with metadata_path.open() as f:
            episodes = [json.loads(line) for line in f if line.strip()]
        episodes.sort(key=lambda episode: episode["episode_index"])

        samples: list[tuple[int, int, int, int | None]] = []
        dataset_offset = 0
        pre_intervention_frames = round(pre_intervention_seconds * fps)
        for expected_episode_index, episode in enumerate(episodes):
            episode_index = int(episode["episode_index"])
            if episode_index != expected_episode_index:
                raise ValueError(
                    "Intervention metadata must contain contiguous episode indices starting at 0; "
                    f"expected {expected_episode_index}, got {episode_index}."
                )

            episode_length = int(episode["length"])
            intervention_frames = np.zeros(episode_length, dtype=bool)
            invalid_pre_intervention_frames = np.zeros(episode_length, dtype=bool)
            previous_end = -1
            for intervention_range in episode.get("intervention_ranges", ()):
                start_frame = int(intervention_range["start_frame"])
                end_frame = int(intervention_range["end_frame"])
                if not 0 <= start_frame <= end_frame < episode_length:
                    raise ValueError(
                        f"Invalid intervention range [{start_frame}, {end_frame}] "
                        f"for episode {episode_index} with length {episode_length}."
                    )
                if start_frame <= previous_end:
                    raise ValueError(f"Overlapping intervention ranges in episode {episode_index} are not supported.")

                intervention_frames[start_frame : end_frame + 1] = True
                invalid_pre_intervention_frames[max(0, start_frame - pre_intervention_frames) : start_frame] = True
                if self._sample_mode == "intervention":
                    samples.extend(
                        (dataset_offset + frame_index, episode_index, frame_index, end_frame)
                        for frame_index in range(start_frame, end_frame + 1)
                    )
                previous_end = end_frame

            # Intervention targets are corrections and always remain valid, even
            # if a preceding-error window overlaps an earlier intervention.
            invalid_pre_intervention_frames &= ~intervention_frames
            self._valid_action_frames.append(~invalid_pre_intervention_frames)
            if self._sample_mode == "valid_non_intervention":
                valid_starts = ~intervention_frames & ~invalid_pre_intervention_frames
                samples.extend(
                    (dataset_offset + int(frame_index), episode_index, int(frame_index), None)
                    for frame_index in np.flatnonzero(valid_starts)
                )

            dataset_offset += episode_length

        if dataset_offset != len(self._dataset):
            raise ValueError(
                f"Intervention metadata describes {dataset_offset} frames, "
                f"but the LeRobot dataset contains {len(self._dataset)} frames."
            )
        if not samples:
            raise ValueError(f"No intervention frames found in {metadata_path}.")
        return samples

    def __getitem__(self, index: SupportsIndex) -> dict:
        item_index = index.__index__()
        if self._samples is None:
            dataset_index = item_index
            valid_steps = self._action_horizon
        else:
            dataset_index, episode_index, frame_index, intervention_end = self._samples[item_index]
            if self._sample_mode == "intervention":
                assert intervention_end is not None
                valid_steps = min(self._action_horizon, intervention_end - frame_index + 1)
            else:
                future_frames = frame_index + np.arange(self._action_horizon)
                within_episode = future_frames < len(self._valid_action_frames[episode_index])
                action_loss_mask = np.zeros(self._action_horizon, dtype=bool)
                action_loss_mask[within_episode] = self._valid_action_frames[episode_index][
                    future_frames[within_episode]
                ]

        sample = dict(self._dataset[dataset_index])
        if self._samples is None or self._sample_mode == "intervention":
            action_loss_mask = np.arange(self._action_horizon) < valid_steps
        sample["action_loss_mask"] = action_loss_mask
        return sample

    def __len__(self) -> int:
        return len(self._dataset) if self._samples is None else len(self._samples)


class WeightedConcatDataset(torch.utils.data.ConcatDataset):
    """Concatenated datasets carrying source-level sampling weights."""

    def __init__(self, datasets: Sequence[Dataset], weights: Sequence[float]):
        if len(datasets) != len(weights):
            raise ValueError("Each LeRobot dataset must have exactly one sampling weight.")
        if not datasets:
            raise ValueError("At least one LeRobot dataset is required for a mixture.")
        if any(weight <= 0 for weight in weights):
            raise ValueError("LeRobot dataset sampling weights must all be positive.")

        super().__init__(typing.cast(Sequence[torch.utils.data.Dataset], datasets))
        if any(len(dataset) == 0 for dataset in self.datasets):
            raise ValueError("LeRobot datasets in a mixture must not be empty.")
        total_weight = sum(weights)
        self.source_weights = tuple(weight / total_weight for weight in weights)

    def item_sampling_weights(self) -> torch.Tensor:
        """Return per-item weights whose mass equals each source's weight."""
        return torch.cat(
            [
                torch.full((len(dataset),), source_weight / len(dataset), dtype=torch.double)
                for dataset, source_weight in zip(self.datasets, self.source_weights, strict=True)
            ]
        )


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_cache: dict[
        tuple[str, str | None, tuple[str, ...]], tuple[Dataset, lerobot_dataset.LeRobotDatasetMetadata]
    ] = {}

    def create_single_dataset(spec: _config.LeRobotDatasetConfig, *, add_action_loss_mask: bool) -> Dataset:
        cache_key = (spec.repo_id, spec.root, tuple(spec.action_sequence_keys))
        if cache_key not in dataset_cache:
            dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(spec.repo_id, root=spec.root)
            dataset = lerobot_dataset.LeRobotDataset(
                spec.repo_id,
                root=spec.root,
                delta_timestamps={
                    key: [t / dataset_meta.fps for t in range(action_horizon)] for key in spec.action_sequence_keys
                },
            )
            dataset_cache[cache_key] = (dataset, dataset_meta)
        dataset, dataset_meta = dataset_cache[cache_key]

        if add_action_loss_mask:
            dataset = ActionLossMaskDataset(
                dataset,
                action_horizon,
                sample_mode=spec.sample_mode,
                intervention_ranges_path=spec.intervention_ranges_path,
                fps=dataset_meta.fps,
                pre_intervention_seconds=spec.pre_intervention_seconds,
            )
        if data_config.prompt_from_task:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
        return dataset

    if data_config.lerobot_datasets:
        add_action_loss_mask = any(spec.sample_mode != "all" for spec in data_config.lerobot_datasets)
        datasets = [
            create_single_dataset(spec, add_action_loss_mask=add_action_loss_mask)
            for spec in data_config.lerobot_datasets
        ]
        return WeightedConcatDataset(datasets, [spec.weight for spec in data_config.lerobot_datasets])

    return create_single_dataset(
        _config.LeRobotDatasetConfig(
            repo_id=data_config.repo_id,
            root=data_config.root,
            action_sequence_keys=data_config.action_sequence_keys,
        ),
        add_action_loss_mask=False,
    )


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    mixture = dataset if isinstance(dataset, WeightedConcatDataset) else None
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if mixture is not None:
        sampler = torch.utils.data.WeightedRandomSampler(
            mixture.item_sampling_weights(),
            num_samples=len(mixture),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        logging.info(f"Using weighted LeRobot mixture with source probabilities {mixture.source_weights}")

    if framework == "pytorch":
        if torch.distributed.is_initialized():
            if sampler is not None:
                raise NotImplementedError("Weighted LeRobot mixtures are not supported with PyTorch DDP.")
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
