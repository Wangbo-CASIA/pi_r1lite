import dataclasses
import json

import jax
import pytest
import torch

from openpi.models import pi0_config
from openpi.policies import r1lite_policy
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class _SourceDataset:
    def __init__(self, source: int, length: int):
        self._source = source
        self._length = length

    def __getitem__(self, index):
        return self._source, index

    def __len__(self):
        return self._length


class _DictDataset:
    def __init__(self, length: int):
        self._length = length

    def __getitem__(self, index):
        return {"dataset_index": index}

    def __len__(self):
        return self._length


class _PythonIntOnlyDataset(_DictDataset):
    def __getitem__(self, index):
        if type(index) is not int:
            raise TypeError(f"Expected a Python int index, got {type(index)}")
        return super().__getitem__(index)


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_weighted_concat_dataset_assigns_weight_by_source():
    dataset = _data_loader.WeightedConcatDataset(
        [_SourceDataset(0, 2), _SourceDataset(1, 8)],
        weights=[0.5, 0.5],
    )

    item_weights = dataset.item_sampling_weights()
    assert item_weights[:2].sum().item() == pytest.approx(0.5)
    assert item_weights[2:].sum().item() == pytest.approx(0.5)

    sampler = torch.utils.data.WeightedRandomSampler(
        item_weights,
        num_samples=10_000,
        replacement=True,
        generator=torch.Generator().manual_seed(42),
    )
    source_zero_fraction = sum(index < 2 for index in sampler) / 10_000
    assert source_zero_fraction == pytest.approx(0.5, abs=0.02)


def test_action_loss_mask_dataset_filters_intervention_starts(tmp_path):
    metadata_path = tmp_path / "episodes.jsonl"
    episodes = [
        {
            "episode_index": 0,
            "length": 6,
            "intervention_ranges": [{"start_frame": 1, "end_frame": 3}],
        },
        {
            "episode_index": 1,
            "length": 5,
            "intervention_ranges": [{"start_frame": 0, "end_frame": 1}],
        },
    ]
    metadata_path.write_text("".join(json.dumps(episode) + "\n" for episode in episodes))

    dataset = _data_loader.ActionLossMaskDataset(
        _DictDataset(11),
        action_horizon=4,
        sample_mode="intervention",
        intervention_ranges_path=str(metadata_path),
        fps=1.0,
    )

    assert len(dataset) == 5
    assert [dataset[index]["dataset_index"] for index in range(len(dataset))] == [1, 2, 3, 6, 7]
    assert dataset[0]["action_loss_mask"].tolist() == [True, True, True, False]
    assert dataset[1]["action_loss_mask"].tolist() == [True, True, False, False]
    assert dataset[2]["action_loss_mask"].tolist() == [True, False, False, False]
    assert dataset[3]["action_loss_mask"].tolist() == [True, True, False, False]


def test_action_loss_mask_dataset_uses_full_mask_without_interventions():
    dataset = _data_loader.ActionLossMaskDataset(_DictDataset(2), action_horizon=4)

    assert len(dataset) == 2
    assert dataset[1]["dataset_index"] == 1
    assert dataset[1]["action_loss_mask"].tolist() == [True, True, True, True]


def test_action_loss_mask_dataset_keeps_valid_non_intervention_with_per_action_mask(tmp_path):
    metadata_path = tmp_path / "episodes.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "episode_index": 0,
                "length": 12,
                "intervention_ranges": [{"start_frame": 5, "end_frame": 7}],
            }
        )
        + "\n"
    )
    dataset = _data_loader.ActionLossMaskDataset(
        _PythonIntOnlyDataset(12),
        action_horizon=7,
        sample_mode="valid_non_intervention",
        intervention_ranges_path=str(metadata_path),
        fps=2.0,
        pre_intervention_seconds=2.0,
    )

    # Frames 1..4 are the two-second pre-intervention window, while
    # frames 5..7 belong to the intervention view.
    assert [dataset[index]["dataset_index"] for index in range(len(dataset))] == [0, 8, 9, 10, 11]
    # A chunk starting at frame 0 masks bad frames 1..4 and then resumes
    # supervision on intervention frames 5..6.
    assert dataset[0]["action_loss_mask"].tolist() == [True, False, False, False, False, True, True]
    # Episode overflow is masked as well.
    assert dataset[1]["action_loss_mask"].tolist() == [True, True, True, True, False, False, False]


def test_dagger_config_uses_five_requested_source_weights():
    config = _config.get_config("r1lite_pack_phone_abs_joint_crop_head_image_dager")
    data_config = config.data.create(config.assets_dirs, config.model)

    assert [source.weight for source in data_config.lerobot_datasets] == [0.40, 0.25, 0.05, 0.25, 0.05]
    assert [source.sample_mode for source in data_config.lerobot_datasets] == [
        "all",
        "intervention",
        "valid_non_intervention",
        "intervention",
        "valid_non_intervention",
    ]
    assert [source.action_sequence_keys for source in data_config.lerobot_datasets] == [
        ("action.qpos",),
        ("action",),
        ("action",),
        ("action.qpos",),
        ("action.qpos",),
    ]


def test_r1lite_repack_preserves_action_loss_mask():
    mask = jax.numpy.asarray([True, False])
    result = r1lite_policy.R1LiteRepack()(
        {
            "images": {"head": None, "left_wrist": None, "right_wrist": None},
            "state": jax.numpy.zeros(14),
            "actions": jax.numpy.zeros((2, 14)),
            "action_loss_mask": mask,
        }
    )

    assert result["action_loss_mask"] is mask


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
