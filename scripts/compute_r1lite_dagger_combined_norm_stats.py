"""Compute R1Lite normalization statistics from every frame of SFT and DAgger datasets.

Unlike the training loader, this script does not apply source sampling weights or
intervention filtering. Every parquet row from all datasets contributes exactly
once. Images are intentionally skipped because normalization only uses state and
actions.
"""

import dataclasses
import pathlib

import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

import openpi.shared.normalize as normalize


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: pathlib.Path
    state_key: str
    action_key: str


DEFAULT_SFT_ROOT = pathlib.Path("/home/robot/wangbo/project/VLA_own/data/r1lite_pack_phone_0707/pack_up_a_smart_phone")
DEFAULT_DAGGER_ROOT = pathlib.Path("/home/robot/wangbo/project/VLA_own/data/r1lite-pack-phone-dagger-it1-0717")
DEFAULT_DAGGER_IT2_ROOT = pathlib.Path("/home/robot/wangbo/project/VLA_own/data/r1lite-pack-phone-dagger-it2-0721")
DEFAULT_OUTPUT_DIR = pathlib.Path(
    "assets/r1lite_pack_phone_abs_joint_crop_head_image_dager/r1lite_pack_phone_abs_joint_crop_head_image_dager"
)


def _parquet_files(root: pathlib.Path) -> list[pathlib.Path]:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
    return files


def _as_vectors(column, *, dataset_name: str, key: str, expected_dim: int) -> np.ndarray:
    values = np.asarray(column.to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != expected_dim:
        raise ValueError(f"Expected {dataset_name}:{key} to have shape [N, {expected_dim}], got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Found non-finite values in {dataset_name}:{key}.")
    return values


def _update_from_dataset(
    spec: DatasetSpec,
    stats: dict[str, normalize.RunningStats],
    observed_min: dict[str, np.ndarray],
    observed_max: dict[str, np.ndarray],
    *,
    batch_size: int,
    expected_dim: int,
) -> int:
    files = _parquet_files(spec.root)
    total_rows = 0
    progress = tqdm.tqdm(files, desc=f"Reading {spec.name}", unit="episode")
    for path in progress:
        parquet = pq.ParquetFile(path)
        available_columns = set(parquet.schema_arrow.names)
        required_columns = {spec.state_key, spec.action_key}
        if missing := required_columns - available_columns:
            raise KeyError(f"Missing columns {sorted(missing)} in {path}.")

        for batch in parquet.iter_batches(batch_size=batch_size, columns=[spec.state_key, spec.action_key]):
            state = _as_vectors(
                batch.column(spec.state_key),
                dataset_name=spec.name,
                key=spec.state_key,
                expected_dim=expected_dim,
            )
            actions = _as_vectors(
                batch.column(spec.action_key),
                dataset_name=spec.name,
                key=spec.action_key,
                expected_dim=expected_dim,
            )
            if len(state) != len(actions):
                raise ValueError(f"State/action row mismatch in {path}: {len(state)} != {len(actions)}")

            stats["state"].update(state)
            stats["actions"].update(actions)
            observed_min["state"] = np.minimum(observed_min["state"], state.min(axis=0))
            observed_max["state"] = np.maximum(observed_max["state"], state.max(axis=0))
            observed_min["actions"] = np.minimum(observed_min["actions"], actions.min(axis=0))
            observed_max["actions"] = np.maximum(observed_max["actions"], actions.max(axis=0))
            total_rows += len(state)
        progress.set_postfix(rows=total_rows)
    return total_rows


def _repair_collapsed_quantiles(
    norm_stats: dict[str, normalize.NormStats],
    observed_min: dict[str, np.ndarray],
    observed_max: dict[str, np.ndarray],
) -> dict[str, list[int]]:
    """Use the observed range when rare values make q01 and q99 identical."""
    repaired: dict[str, list[int]] = {}
    for key, value in norm_stats.items():
        q01 = np.asarray(value.q01).copy()
        q99 = np.asarray(value.q99).copy()
        collapsed = np.isclose(q01, q99)
        repairable = collapsed & ~np.isclose(observed_min[key], observed_max[key])
        if np.any(repairable):
            q01[repairable] = observed_min[key][repairable]
            q99[repairable] = observed_max[key][repairable]
            value.q01 = q01
            value.q99 = q99
            repaired[key] = np.flatnonzero(repairable).tolist()
    return repaired


def main(
    sft_root: pathlib.Path = DEFAULT_SFT_ROOT,
    dagger_root: pathlib.Path = DEFAULT_DAGGER_ROOT,
    dagger_it2_root: pathlib.Path = DEFAULT_DAGGER_IT2_ROOT,
    output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = 8192,
    expected_dim: int = 14,
) -> None:
    """Compute and save combined full-frame state/action normalization stats."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    specs = (
        DatasetSpec(
            name="SFT",
            root=sft_root,
            state_key="observations.state.qpos",
            action_key="action.qpos",
        ),
        DatasetSpec(
            name="DAgger-it1",
            root=dagger_root,
            state_key="observations.state.qpos",
            action_key="action",
        ),
        DatasetSpec(
            name="DAgger-it2",
            root=dagger_it2_root,
            state_key="observations.state.qpos",
            action_key="action.qpos",
        ),
    )
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    observed_min = {key: np.full(expected_dim, np.inf, dtype=np.float64) for key in stats}
    observed_max = {key: np.full(expected_dim, -np.inf, dtype=np.float64) for key in stats}
    source_counts = {
        spec.name: _update_from_dataset(
            spec,
            stats,
            observed_min,
            observed_max,
            batch_size=batch_size,
            expected_dim=expected_dim,
        )
        for spec in specs
    }

    norm_stats = {key: running_stats.get_statistics() for key, running_stats in stats.items()}
    repaired = _repair_collapsed_quantiles(norm_stats, observed_min, observed_max)
    normalize.save(output_dir, norm_stats)

    print(f"Source frame counts: {source_counts}")
    print(f"Total frames: {sum(source_counts.values())}")
    if repaired:
        print(f"Collapsed quantiles repaired with observed min/max: {repaired}")
    print(f"Writing stats to: {(output_dir / 'norm_stats.json').resolve()}")
    for key, value in norm_stats.items():
        print(f"{key}.q01 = {value.q01.tolist()}")
        print(f"{key}.q99 = {value.q99.tolist()}")


if __name__ == "__main__":
    tyro.cli(main)
