#!/usr/bin/env python3
"""Convert R1Lite dual-arm RAW MCAP episodes to a LeRobot dataset for openpi.
主要结构：
- 定义lerobot每一帧的数据格式是什么样的
-

"""


import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from r1lite_experiment_config import add_experiment_args
from r1lite_experiment_config import apply_converter_config

# CONRFT_ROOT = Path("/home/ps/VLA-RL/conrft-r1lite")
# SARM_EXAMPLES = CONRFT_ROOT / "examples" / "sarm"
# if str(SARM_EXAMPLES) not in sys.path:
#     sys.path.insert(0, str(SARM_EXAMPLES))

from rosbag_sarm_utils import DEFAULT_TOPICS  # noqa: E402
from rosbag_sarm_utils import nearest_value_at  # noqa: E402
from rosbag_sarm_utils import overlapping_timeline  # noqa: E402
from rosbag_sarm_utils import read_topic_series  # noqa: E402
from rosbag_sarm_utils import resolve_input_dirs  # noqa: E402
from rosbag_sarm_utils import value_at  # noqa: E402


DEFAULT_TASK_DESC = "move the white box from the left to the center, then pick up the yellow-red mango on the right and place it inside the box"


# 下面这几个定义用于告诉 LeRobot，每一帧数据长什么样：

# 1 状态：lerobot里，按照你需要的state进行表征；工程上为了统一 R1Lite 的 observation.state 格式；包括左右臂的 joint 或 EEF pose
STATE_NAMES = (
    [f"left_tcp_pose_{i}" for i in range(7)]
    + [f"left_tcp_vel_{i}" for i in range(6)]
    + [f"left_joint_pos_{i}" for i in range(6)]
    + [f"left_joint_vel_{i}" for i in range(6)]
    + ["left_gripper"]
    + [f"right_tcp_pose_{i}" for i in range(7)]
    + [f"right_tcp_vel_{i}" for i in range(6)]
    + [f"right_joint_pos_{i}" for i in range(6)]
    + [f"right_joint_vel_{i}" for i in range(6)]
    + ["right_gripper", "torso"]
)

# 下面对应于统一状态格式里 你当前脚本想要填充的状态 主要是关节角 和 两个夹爪
LEFT_JOINT_POS_SLICE = slice(13, 19)
LEFT_GRIPPER_INDEX = 25
RIGHT_JOINT_POS_SLICE = slice(39, 45)
RIGHT_GRIPPER_INDEX = 51

# 2：动作；定义lerobot里 按照你需要的action表示进行输入
ACTION_NAMES = (
    [f"left_joint_delta_{i}" for i in range(6)]
    + ["left_gripper_target"]
    + [f"right_joint_delta_{i}" for i in range(6)]
    + ["right_gripper_target"]
)

DEFAULT_BASE_SOURCE_LABEL = "base_demo"

JOINT_TOPIC_KEYS = (
    "head",
    "left_wrist",
    "right_wrist",
    "left_joint",
    "right_joint",
    "left_gripper",
    "right_gripper",
)



# 3）图像；下面会根据原始数据填充 image_shape
def _image_feature(image_shape: tuple[int, ...], dtype: str) -> dict[str, Any]:
    if len(image_shape) != 3:
        raise ValueError(f"Expected image shape (height, width, channel), got {image_shape}")
    return {"dtype": dtype, "shape": image_shape, "names": ["height", "width", "channel"]}




# 最终的 features会告诉 LeRobot，每一帧数据长什么样；
def _features(image_shapes: dict[str, tuple[int, ...]], image_dtype: str) -> dict[str, Any]:
    return {
        "observation.images.head": _image_feature(image_shapes["head"], image_dtype),
        "observation.images.left_wrist": _image_feature(image_shapes["left_wrist"], image_dtype),
        "observation.images.right_wrist": _image_feature(image_shapes["right_wrist"], image_dtype),
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": list(STATE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": list(ACTION_NAMES),
        },
    }



def _feature_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_feature_value(item) for item in value]
    if isinstance(value, list):
        return [_feature_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _feature_value(item) for key, item in value.items() if key != "info"}
    return value


def _validate_append_features(dataset: LeRobotDataset, expected_features: dict[str, Any]) -> None:
    for key, expected in expected_features.items():
        actual = dataset.features.get(key)
        if actual is None:
            raise ValueError(f"Existing dataset is missing feature required for append: {key}")
        if _feature_value(actual) != _feature_value(expected):
            raise ValueError(
                "Existing dataset feature does not match converter output for append: "
                f"{key}: actual={actual} expected={expected}"
            )


####### 下面是将每帧的实际数据填充到lerobot的每一帧里面

def _binary_gripper(gripper: np.ndarray, threshold: float) -> np.ndarray:
    value = float(np.asarray(gripper, dtype=np.float32).reshape(-1)[0])
    return np.asarray([0.0 if value > threshold else 1.0], dtype=np.float32)

def _float_gripper(gripper: np.ndarray) -> np.ndarray:
    value = float(np.asarray(gripper, dtype=np.float32).reshape(-1)[0])
    return np.asarray(value, dtype=np.float32)

# 将原始数据里每一帧的 状态数据 填充到 lerobot格式的状态里
def _state_vector(sample: dict[str, Any], left_gripper_threshold: float, right_gripper_threshold: float) -> np.ndarray:
    state = np.zeros((len(STATE_NAMES),), dtype=np.float32)
    state[LEFT_JOINT_POS_SLICE] = np.asarray(sample["left_joint"]["position"], dtype=np.float32).reshape(-1)[:6]
    state[LEFT_GRIPPER_INDEX] = _binary_gripper(sample["left_gripper"], left_gripper_threshold)[0]
    state[RIGHT_JOINT_POS_SLICE] = np.asarray(sample["right_joint"]["position"], dtype=np.float32).reshape(-1)[:6]
    state[RIGHT_GRIPPER_INDEX] = _binary_gripper(sample["right_gripper"], right_gripper_threshold)[0]
    return state

# 将原始数据里每一帧的 动作数据 填充到 lerobot格式的动作里
# def _action_vector(
#     current: dict[str, Any],
#     next_sample: dict[str, Any],
#     left_gripper_threshold: float,
#     right_gripper_threshold: float,
# ) -> np.ndarray:
#     left_joint = np.asarray(current["left_joint"]["position"], dtype=np.float32).reshape(-1)[:6]
#     next_left_joint = np.asarray(next_sample["left_joint"]["position"], dtype=np.float32).reshape(-1)[:6]
#     right_joint = np.asarray(current["right_joint"]["position"], dtype=np.float32).reshape(-1)[:6]
#     next_right_joint = np.asarray(next_sample["right_joint"]["position"], dtype=np.float32).reshape(-1)[:6]
#     return np.concatenate(
#         [
#             next_left_joint - left_joint,
#             _binary_gripper(next_sample["left_gripper"], left_gripper_threshold),
#             next_right_joint - right_joint,
#             _binary_gripper(next_sample["right_gripper"], right_gripper_threshold),
#         ],
#         axis=0,
#     ).astype(np.float32)

def _action_vector(
    current: dict[str, Any],
    next_sample: dict[str, Any],
    left_gripper_threshold: float,
    right_gripper_threshold: float,
) -> np.ndarray:
    left_joint = np.asarray(current["left_joint"]["position"], dtype=np.float32).reshape(-1)[:6]
    next_left_joint = np.asarray(next_sample["left_joint"]["position"], dtype=np.float32).reshape(-1)[:6]
    right_joint = np.asarray(current["right_joint"]["position"], dtype=np.float32).reshape(-1)[:6]
    next_right_joint = np.asarray(next_sample["right_joint"]["position"], dtype=np.float32).reshape(-1)[:6]
    return np.concatenate(
        [
            next_left_joint,
            _binary_gripper(next_sample["left_gripper"], left_gripper_threshold),
            next_right_joint,
            _binary_gripper(next_sample["right_gripper"], right_gripper_threshold),
        ],
        axis=0,
    ).astype(np.float32)



def _build_joint_episode_samples(input_dir: Path, fps: float, topics: dict[str, str]) -> list[dict[str, Any]]:
    selected_topics = [topics[key] for key in JOINT_TOPIC_KEYS]
    series_map = read_topic_series(input_dir, selected_topics)
    timeline = overlapping_timeline(series_map, fps)
    samples = []
    for ts in timeline:
        timestamp = int(ts)
        samples.append(
            {
                "timestamp_ns": timestamp,
                "head": nearest_value_at(series_map[topics["head"]], timestamp),
                "left_wrist": nearest_value_at(series_map[topics["left_wrist"]], timestamp),
                "right_wrist": nearest_value_at(series_map[topics["right_wrist"]], timestamp),
                "left_joint": value_at(series_map[topics["left_joint"]], timestamp),
                "right_joint": value_at(series_map[topics["right_joint"]], timestamp),
                "left_gripper": value_at(series_map[topics["left_gripper"]], timestamp),
                "right_gripper": value_at(series_map[topics["right_gripper"]], timestamp),
            }
        )
    return samples


def _add_episode(
    dataset: LeRobotDataset,
    samples: list[dict[str, Any]],
    task_desc: str,
    left_gripper_threshold: float,
    right_gripper_threshold: float,
) -> None:
    if len(samples) < 2:
        raise ValueError("Need at least two synchronized samples to export an episode.")

    for idx, sample in enumerate(samples):
        next_sample = samples[idx + 1] if idx + 1 < len(samples) else sample
        frame = {
            "observation.images.head": sample["head"],
            "observation.images.left_wrist": sample["left_wrist"],
            "observation.images.right_wrist": sample["right_wrist"],
            "observation.state": _state_vector(sample, left_gripper_threshold, right_gripper_threshold),
            "action": _action_vector(sample, next_sample, left_gripper_threshold, right_gripper_threshold),
            "task": task_desc,
        }
        dataset.add_frame(frame)
    dataset.save_episode()


def _topic_overrides(args: argparse.Namespace) -> dict[str, str]:
    topics = dict(DEFAULT_TOPICS)
    for key in DEFAULT_TOPICS:
        value = getattr(args, f"{key}_topic")
        if value:
            topics[key] = value
    return topics


def _manifest_path(output_dir: Path, source_manifest: str | None) -> Path:
    if source_manifest:
        return Path(source_manifest).expanduser().resolve()
    return output_dir / "meta" / "source_episodes.jsonl"


def _failed_manifest_path(output_dir: Path, failed_source_manifest: str | None) -> Path:
    if failed_source_manifest:
        return Path(failed_source_manifest).expanduser().resolve()
    return output_dir / "meta" / "source_episodes_failed.jsonl"


def _read_source_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in source manifest {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Invalid source manifest row in {path}:{line_number}: expected object")
        records.append(record)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _prepare_source_manifest(
    path: Path,
    existing_episode_count: int,
    input_dirs: list[Path],
    source_label: str,
    allow_duplicate_source: bool,
    skip_existing_source: bool,
) -> set[str]:
    if path.exists():
        records = _read_source_manifest(path)
        if len(records) != existing_episode_count:
            raise ValueError(
                f"Existing source manifest {path} has {len(records)} rows, "
                f"but dataset has {existing_episode_count} episodes"
            )
        episode_indices = [record.get("episode_index") for record in records]
        expected_indices = list(range(existing_episode_count))
        if episode_indices[:existing_episode_count] != expected_indices:
            raise ValueError(
                f"Existing source manifest {path} does not cover dataset episodes 0..{existing_episode_count - 1}"
            )
    else:
        records = [
            {"episode_index": episode_index, "source": DEFAULT_BASE_SOURCE_LABEL, "raw_path": None}
            for episode_index in range(existing_episode_count)
        ]
        _write_jsonl(path, records)

    existing_raw_paths = {
        record.get("raw_path")
        for record in records
        if isinstance(record.get("raw_path"), str) and record.get("raw_path")
    }
    new_raw_paths = [str(path.resolve()) for path in input_dirs]
    duplicated_existing = sorted(set(new_raw_paths) & existing_raw_paths)
    duplicated_new = sorted({path for path in new_raw_paths if new_raw_paths.count(path) > 1})
    if (
        (duplicated_existing and not skip_existing_source)
        or duplicated_new
    ) and not allow_duplicate_source:
        details = []
        if duplicated_existing and not skip_existing_source:
            details.append(f"already in manifest: {duplicated_existing[:5]}")
        if duplicated_new:
            details.append(f"duplicated in input: {duplicated_new[:5]}")
        raise ValueError(
            "Refusing to append duplicate RAW source paths. "
            + "; ".join(details)
            + ". Use --allow-duplicate-source only if this is intentional."
        )

    if not source_label:
        raise ValueError("--source-label must be non-empty when appending")
    return existing_raw_paths


def _append_source_record(path: Path, episode_index: int, source_label: str, raw_path: Path) -> None:
    _write_jsonl(
        path,
        [
            {
                "episode_index": episode_index,
                "source": source_label,
                "raw_path": str(raw_path.resolve()),
            }
        ],
        append=True,
    )


def _append_failed_source_record(path: Path, source_label: str, raw_path: Path, error: Exception) -> None:
    _write_jsonl(
        path,
        [
            {
                "source": source_label,
                "raw_path": str(raw_path.resolve()),
                "error": str(error),
                "error_type": type(error).__name__,
            }
        ],
        append=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_args(parser, default_experiment="r1lite_pack_phone_new_state",default_action_space="abs_joint")
    parser.add_argument("--input-dir", default=None, help="RAW episode directory or parent directory.")
    parser.add_argument("--raw-dir-glob", default=None)
    parser.add_argument("--recursive", action="store_true", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--task-desc", default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--gripper-threshold", type=float, default=None)
    parser.add_argument("--left-gripper-threshold", type=float, default=None)
    parser.add_argument("--right-gripper-threshold", type=float, default=None)
    parser.add_argument("--overwrite", type=bool, default=True)
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--append", action="store_true", help="Append RAW episodes to an existing LeRobot dataset.")
    parser.add_argument("--source-label", default=None, help="Source label written to meta/source_episodes.jsonl.")
    parser.add_argument("--source-manifest", default=None, help="Optional source manifest path. Defaults to output meta.")
    parser.add_argument(
        "--failed-source-manifest",
        default=None,
        help="Optional failed source manifest path. Defaults to output meta.",
    )
    parser.add_argument(
        "--skip-existing-source",
        action="store_true",
        help="In append mode, skip RAW paths already present in the source manifest.",
    )
    parser.add_argument(
        "--skip-invalid-episodes",
        action="store_true",
        help="Record invalid RAW episodes in source_episodes_failed.jsonl and continue.",
    )
    parser.add_argument(
        "--allow-duplicate-source",
        action="store_true",
        help="Allow appending RAW paths already recorded in the source manifest.",
    )
    for key, default_topic in DEFAULT_TOPICS.items():
        parser.add_argument(
            f"--{key.replace('_', '-')}-topic",
            dest=f"{key}_topic",
            default=None,
            help=f"Override topic for {key}: {default_topic}",
        )
    args = apply_converter_config(parser.parse_args())
    if args.input_dir is None:
        parser.error("--input-dir is required unless provided by --experiment/--config")
    if args.output_dir is None:
        parser.error("--output-dir is required unless provided by --experiment/--config")
    if args.repo_id is None:
        parser.error("--repo-id is required unless provided by --experiment/--config")
    if args.raw_dir_glob is None:
        args.raw_dir_glob = "*_RAW"
    if args.recursive is None:
        args.recursive = False
    if args.task_desc is None:
        args.task_desc = DEFAULT_TASK_DESC
    if args.fps is None:
        args.fps = 10.0
    if args.gripper_threshold is None:
        args.gripper_threshold = 75.0
    if args.left_gripper_threshold is None:
        args.left_gripper_threshold = args.gripper_threshold
    if args.right_gripper_threshold is None:
        args.right_gripper_threshold = args.gripper_threshold
    if args.append and args.overwrite:
        parser.error("--append cannot be used with --overwrite")
    if args.append and not args.source_label:
        parser.error("--append requires --source-label")
    if not args.append and (args.skip_existing_source or args.skip_invalid_episodes):
        parser.error("--skip-existing-source and --skip-invalid-episodes require --append")
    return args


def main() -> None:
    args = _parse_args()
    fps = round(float(args.fps))
    if fps <= 0 or abs(float(args.fps) - fps) > 1e-6:
        raise ValueError(f"--fps must be a positive integer value, got {args.fps}")

    input_dirs = resolve_input_dirs([args.input_dir], args.raw_dir_glob, args.recursive)
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError(f"--max-episodes must be positive, got {args.max_episodes}")
        input_dirs = input_dirs[: args.max_episodes]
    if not input_dirs:
        raise ValueError("No RAW episodes resolved.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and not args.append:
        if not args.overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_dir}")
        shutil.rmtree(output_dir)
    if args.append and not output_dir.exists():
        raise FileNotFoundError(f"--append requires an existing output dataset: {output_dir}")

    topics = _topic_overrides(args)
    source_manifest_path = _manifest_path(output_dir, args.source_manifest)
    failed_source_manifest_path = _failed_manifest_path(output_dir, args.failed_source_manifest)
    if args.append:
        dataset = LeRobotDataset(args.repo_id, root=output_dir, video_backend=args.video_backend)
        if dataset.fps != fps:
            raise ValueError(f"Existing dataset fps={dataset.fps} does not match requested fps={fps}")
        existing_raw_paths = _prepare_source_manifest(
            source_manifest_path,
            dataset.meta.total_episodes,
            input_dirs,
            args.source_label,
            args.allow_duplicate_source,
            args.skip_existing_source,
        )
        if args.skip_existing_source:
            before_count = len(input_dirs)
            input_dirs = [path for path in input_dirs if str(path.resolve()) not in existing_raw_paths]
            skipped_count = before_count - len(input_dirs)
            if skipped_count:
                print(f"Skipped {skipped_count} RAW episodes already recorded in {source_manifest_path}", flush=True)
        if not input_dirs:
            print("No new RAW episodes to append.")
            return
        if args.image_writer_processes or args.image_writer_threads:
            dataset.start_image_writer(args.image_writer_processes, args.image_writer_threads)
    else:
        print(f"Reading episode 1/{len(input_dirs)}: {input_dirs[0]}", flush=True)
        first_samples = _build_joint_episode_samples(input_dirs[0], fps, topics)
        if not first_samples:
            raise ValueError(f"No synchronized samples resolved for first episode: {input_dirs[0]}")
        image_shapes = {
            "head": tuple(np.asarray(first_samples[0]["head"]).shape),
            "left_wrist": tuple(np.asarray(first_samples[0]["left_wrist"]).shape),
            "right_wrist": tuple(np.asarray(first_samples[0]["right_wrist"]).shape),
        }
        expected_features = _features(image_shapes, "image" if args.no_videos else "video")
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=fps,
            root=output_dir,
            robot_type="r1lite_dual",
            features=expected_features,
            use_videos=not args.no_videos,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
            video_backend=args.video_backend,
        )

    append_features_validated = False
    try:
        for index, input_dir in enumerate(input_dirs):
            try:
                if not args.append and index == 0:
                    samples = first_samples
                else:
                    print(f"Reading episode {index + 1}/{len(input_dirs)}: {input_dir}", flush=True)
                    samples = _build_joint_episode_samples(input_dir, fps, topics)
                if not samples:
                    raise ValueError(f"No synchronized samples resolved for episode: {input_dir}")
                if not args.append and index == 0:
                    first_samples = []
                if args.append and not append_features_validated:
                    image_shapes = {
                        "head": tuple(np.asarray(samples[0]["head"]).shape),
                        "left_wrist": tuple(np.asarray(samples[0]["left_wrist"]).shape),
                        "right_wrist": tuple(np.asarray(samples[0]["right_wrist"]).shape),
                    }
                    expected_features = _features(image_shapes, "image" if args.no_videos else "video")
                    _validate_append_features(dataset, expected_features)
                    append_features_validated = True
                episode_index = dataset.meta.total_episodes
                print(
                    f"{'Appending' if args.append else 'Exporting'} episode {index + 1}/{len(input_dirs)} "
                    f"as dataset episode {episode_index} with {len(samples)} frames",
                    flush=True,
                )
                _add_episode(
                    dataset,
                    samples,
                    args.task_desc,
                    args.left_gripper_threshold,
                    args.right_gripper_threshold,
                )
                if args.append:
                    _append_source_record(source_manifest_path, episode_index, args.source_label, input_dir)
            except Exception as exc:
                if not (args.append and args.skip_invalid_episodes):
                    raise
                print(f"FAILED episode {index + 1}/{len(input_dirs)}: {input_dir}: {exc}", flush=True)
                if dataset.episode_buffer is not None and dataset.episode_buffer.get("size", 0) > 0:
                    dataset.clear_episode_buffer()
                _append_failed_source_record(failed_source_manifest_path, args.source_label, input_dir, exc)
    finally:
        dataset.stop_image_writer()

    action = "Appended" if args.append else "Exported"
    print(f"{action} {len(input_dirs)} episodes to {output_dir}")


if __name__ == "__main__":
    main()
