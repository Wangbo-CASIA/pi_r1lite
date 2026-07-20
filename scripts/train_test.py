import dataclasses
import os
import pathlib

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import pytest

from openpi.training import config as _config

from . import train


def test_reduce_chunked_loss_without_mask():
    loss = jnp.asarray([[1.0, 3.0], [5.0, 7.0]])

    assert train._reduce_chunked_loss(loss, None) == pytest.approx(4.0)  # noqa: SLF001


def test_reduce_chunked_loss_normalizes_each_sample_mask():
    loss = jnp.asarray([[1.0, 3.0, 100.0], [2.0, 4.0, 6.0]])
    mask = jnp.asarray([[True, True, False], [True, False, False]])

    # First sample: (1 + 3) / 2 = 2. Second sample: 2 / 1 = 2.
    assert train._reduce_chunked_loss(loss, mask) == pytest.approx(2.0)  # noqa: SLF001


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
