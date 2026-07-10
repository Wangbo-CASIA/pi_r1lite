#!/usr/bin/env python3
"""Convert R1Lite RAW MCAP episodes to an EVA-VLA-style LeRobot v2.1 dataset."""

import argparse
from pathlib import Path
import shutil
import sys
from typing import Any

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_module
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import lerobot.common.datasets.video_utils as video_utils
import numpy as np

CONRFT_ROOT = Path("/home/ps/VLA-RL/conrft-r1lite")
SARM_EXAMPLES = CONRFT_ROOT / "examples" / "sarm"
if str(SARM_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(SARM_EXAMPLES))

from rosbag_sarm_utils import nearest_value_at  # noqa: E402,I001
from rosbag_sarm_utils import DEFAULT_TOPICS  # noqa: E402
from rosbag_sarm_utils import read_topic_series  # noqa: E402
from rosbag_sarm_utils import resolve_input_dirs  # noqa: E402
from rosbag_sarm_utils import value_at  # noqa: E402


DEFAULT_INPUT_DIR = "/home/ps/VLA-RL/conrft-r1lite/data/RAW/r1lite_pack_phone_new"
DEFAULT_REPO_ID = "r1lite_pack_phone_new_eva_vla_abs"
DEFAULT_OUTPUT_DIR = f"/home/ps/VLA-RL/conrft-r1lite/data/lerobot_openpi/{DEFAULT_REPO_ID}"
DEFAULT_TASK_DESC = (
    "first scoop up the black foam and place it in the box, then scoop up the phone and place it in the box, "
    "and finally pick up the lid, put it on the box, and press it down firmly"
)

JOINT_NAMES = (
    [f"left.j{i}" for i in range(6)]
    + ["left.gripper"]
    + [f"right.j{i}" for i in range(6)]
    + ["right.gripper"]
)
EEF_NAMES = (
    "left.x",
    "left.y",
    "left.z",
    "left.qw",
    "left.qx",
    "left.qy",
    "left.qz",
    "left.gripper",
    "right.x",
    "right.y",
    "right.z",
    "right.qw",
    "right.qx",
    "right.qy",
    "right.qz",
    "right.gripper",
)
BASE_ACTION_NAMES = ["base.linear_x", "base.angular_z"]
CAMERA_KEY_MAP = {
    "head": "observation.images.cam_high",
    "left_wrist": "observation.images.cam_left_wrist",
    "right_wrist": "observation.images.cam_right_wrist",
}


def _image_feature(image_shape: tuple[int, ...], dtype: str) -> dict[str, Any]:
    if len(image_shape) != 3:
        raise ValueError(f"Expected image shape (height, width, channels), got {image_shape}")
    return {"dtype": dtype, "shape": image_shape, "names": ["height", "width", "channels"]}


def _features(image_shapes: dict[str, tuple[int, ...]], image_dtype: str) -> dict[str, Any]:
    return {
        CAMERA_KEY_MAP["head"]: _image_feature(image_shapes["head"], image_dtype),
        CAMERA_KEY_MAP["left_wrist"]: _image_feature(image_shapes["left_wrist"], image_dtype),
        CAMERA_KEY_MAP["right_wrist"]: _image_feature(image_shapes["right_wrist"], image_dtype),
        "observation.qpos": {"dtype": "float32", "shape": (14,), "names": list(JOINT_NAMES)},
        "observation.qvel": {"dtype": "float32", "shape": (14,), "names": list(JOINT_NAMES)},
        "observation.effort": {"dtype": "float32", "shape": (14,), "names": list(JOINT_NAMES)},
        "observation.eef": {"dtype": "float32", "shape": (16,), "names": list(EEF_NAMES)},
        "action": {"dtype": "float32", "shape": (14,), "names": list(JOINT_NAMES)},
        "action_eef": {"dtype": "float32", "shape": (16,), "names": list(EEF_NAMES)},
        "base_action": {"dtype": "float32", "shape": (2,), "names": list(BASE_ACTION_NAMES)},
    }


def _topic_overrides(args: argparse.Namespace) -> dict[str, str]:
    topics = dict(DEFAULT_TOPICS)
    for key in DEFAULT_TOPICS:
        value = getattr(args, f"{key}_topic")
        if value:
            topics[key] = value
    return topics


def _binary_gripper(gripper: np.ndarray, threshold: float) -> np.ndarray:
    value = float(np.asarray(gripper, dtype=np.float32).reshape(-1)[0])
    return np.asarray([0.0 if value > threshold else 1.0], dtype=np.float32)


def _unit_xyzw_quat(pose: np.ndarray, label: str) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)[:7].copy()
    quat_norm = float(np.linalg.norm(pose[3:7]))
    if not np.isfinite(quat_norm) or quat_norm <= 0.0:
        raise ValueError(f"{label} has invalid quaternion norm: {quat_norm}")
    pose[3:7] /= quat_norm
    return pose


def _eef_arm(pose: np.ndarray, gripper: np.ndarray, threshold: float, label: str) -> np.ndarray:
    pose = _unit_xyzw_quat(pose, label)
    # RAW stores x, y, z, qx, qy, qz, qw. EVA-VLA schema uses x, y, z, qw, qx, qy, qz.
    return np.concatenate([pose[:3], pose[6:7], pose[3:6], _binary_gripper(gripper, threshold)], axis=0)


def _qpos(sample: dict[str, Any], left_threshold: float, right_threshold: float) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(sample["left_joint"]["position"], dtype=np.float32).reshape(-1)[:6],
            _binary_gripper(sample["left_gripper"], left_threshold),
            np.asarray(sample["right_joint"]["position"], dtype=np.float32).reshape(-1)[:6],
            _binary_gripper(sample["right_gripper"], right_threshold),
        ],
        axis=0,
    ).astype(np.float32)


def _qvel(
    sample: dict[str, Any],
    next_sample: dict[str, Any],
    dt: float,
    left_threshold: float,
    right_threshold: float,
) -> np.ndarray:
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    left_gripper_vel = (
        _binary_gripper(next_sample["left_gripper"], left_threshold) - _binary_gripper(sample["left_gripper"], left_threshold)
    ) / dt
    right_gripper_vel = (
        _binary_gripper(next_sample["right_gripper"], right_threshold)
        - _binary_gripper(sample["right_gripper"], right_threshold)
    ) / dt
    return np.concatenate(
        [
            np.asarray(sample["left_joint"]["velocity"], dtype=np.float32).reshape(-1)[:6],
            left_gripper_vel,
            np.asarray(sample["right_joint"]["velocity"], dtype=np.float32).reshape(-1)[:6],
            right_gripper_vel,
        ],
        axis=0,
    ).astype(np.float32)


def _effort(sample: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(sample["left_joint"]["effort"], dtype=np.float32).reshape(-1)[:6],
            np.zeros((1,), dtype=np.float32),
            np.asarray(sample["right_joint"]["effort"], dtype=np.float32).reshape(-1)[:6],
            np.zeros((1,), dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def _eef(sample: dict[str, Any], left_threshold: float, right_threshold: float) -> np.ndarray:
    return np.concatenate(
        [
            _eef_arm(sample["left_tcp_pose"], sample["left_gripper"], left_threshold, "left tcp pose"),
            _eef_arm(sample["right_tcp_pose"], sample["right_gripper"], right_threshold, "right tcp pose"),
        ],
        axis=0,
    ).astype(np.float32)


def _selected_topics(topics: dict[str, str]) -> list[str]:
    return [
        topics["head"],
        topics["left_wrist"],
        topics["right_wrist"],
        topics["left_tcp_pose"],
        topics["right_tcp_pose"],
        topics["left_joint"],
        topics["right_joint"],
        topics["left_gripper"],
        topics["right_gripper"],
    ]


def _overlap_bounds(series_map: dict[str, Any]) -> tuple[int, int]:
    start_ns = max(int(series.timestamps[0]) for series in series_map.values())
    end_ns = min(int(series.timestamps[-1]) for series in series_map.values())
    if end_ns <= start_ns:
        raise ValueError("No overlapping time interval across selected topics")
    return start_ns, end_ns


def _infer_fps(head_timestamps: np.ndarray) -> int:
    if len(head_timestamps) < 2:
        raise ValueError("Need at least two head camera timestamps to infer fps")
    dt = np.diff(head_timestamps.astype(np.int64)) / 1e9
    median_dt = float(np.median(dt))
    if not np.isfinite(median_dt) or median_dt <= 0:
        raise ValueError(f"Invalid median head camera dt: {median_dt}")
    fps = round(1.0 / median_dt)
    if fps <= 0:
        raise ValueError(f"Invalid inferred fps: {fps}")
    return fps


def _episode_samples(input_dir: Path, topics: dict[str, str], expected_fps: int | None = None) -> tuple[list[dict[str, Any]], int]:
    series_map = read_topic_series(input_dir, _selected_topics(topics))
    start_ns, end_ns = _overlap_bounds(series_map)
    head_series = series_map[topics["head"]]
    head_timestamps = head_series.timestamps[
        (head_series.timestamps >= start_ns) & (head_series.timestamps <= end_ns)
    ]
    if len(head_timestamps) < 2:
        raise ValueError(f"Need at least two synchronized head frames for episode: {input_dir}")

    fps = _infer_fps(head_timestamps)
    if expected_fps is not None and fps != expected_fps:
        raise ValueError(f"Episode {input_dir} inferred fps {fps}, expected {expected_fps}")

    samples: list[dict[str, Any]] = []
    for timestamp in head_timestamps:
        timestamp_ns = int(timestamp)
        samples.append(
            {
                "timestamp_ns": timestamp_ns,
                "head": nearest_value_at(series_map[topics["head"]], timestamp_ns),
                "left_wrist": nearest_value_at(series_map[topics["left_wrist"]], timestamp_ns),
                "right_wrist": nearest_value_at(series_map[topics["right_wrist"]], timestamp_ns),
                "left_tcp_pose": value_at(series_map[topics["left_tcp_pose"]], timestamp_ns),
                "right_tcp_pose": value_at(series_map[topics["right_tcp_pose"]], timestamp_ns),
                "left_joint": value_at(series_map[topics["left_joint"]], timestamp_ns),
                "right_joint": value_at(series_map[topics["right_joint"]], timestamp_ns),
                "left_gripper": value_at(series_map[topics["left_gripper"]], timestamp_ns),
                "right_gripper": value_at(series_map[topics["right_gripper"]], timestamp_ns),
            }
        )
    return samples, fps


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
        if idx + 1 < len(samples):
            dt = max(1e-6, (next_sample["timestamp_ns"] - sample["timestamp_ns"]) / 1e9)
        else:
            dt = 1.0 / dataset.fps
        qpos = _qpos(sample, left_gripper_threshold, right_gripper_threshold)
        eef = _eef(sample, left_gripper_threshold, right_gripper_threshold)
        frame = {
            CAMERA_KEY_MAP["head"]: sample["head"],
            CAMERA_KEY_MAP["left_wrist"]: sample["left_wrist"],
            CAMERA_KEY_MAP["right_wrist"]: sample["right_wrist"],
            "observation.qpos": qpos,
            "observation.qvel": _qvel(sample, next_sample, dt, left_gripper_threshold, right_gripper_threshold),
            "observation.effort": _effort(sample),
            "observation.eef": eef,
            "action": _qpos(next_sample, left_gripper_threshold, right_gripper_threshold),
            "action_eef": _eef(next_sample, left_gripper_threshold, right_gripper_threshold),
            "base_action": np.zeros((2,), dtype=np.float32),
            "task": task_desc,
        }
        dataset.add_frame(frame)
    dataset.save_episode()


def _patch_video_encoder(vcodec: str) -> None:
    if vcodec not in {"h264", "hevc", "libsvtav1"}:
        raise ValueError(f"Unsupported video codec: {vcodec}")
    original_encode_video_frames = video_utils.encode_video_frames

    def encode_video_frames_with_codec(
        imgs_dir: Path | str,
        video_path: Path | str,
        fps: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        kwargs["vcodec"] = vcodec
        original_encode_video_frames(imgs_dir, video_path, fps, *args, **kwargs)

    lerobot_dataset_module.encode_video_frames = encode_video_frames_with_codec


def _write_format_doc(root: Path, repo_id: str, fps: int, video_codec: str) -> None:
    content = f"""# EVA-VLA R1Lite LeRobot Dataset Format

This dataset is a LeRobot v2.1 local dataset converted from R1Lite RAW MCAP episodes.

## Identity

- `repo_id`: `{repo_id}`
- `robot_type`: `r1lite_dual`
- `fps`: `{fps}`
- Video codec: `{video_codec}`
- Data source: `/home/ps/VLA-RL/conrft-r1lite/data/RAW/r1lite_pack_phone_new`
- Sampling timeline: RAW head camera timestamps, exported to LeRobot's nominal `{fps}` Hz timestamp grid

## Cameras

- `observation.images.cam_high`: RAW head camera, RGB uint8, no resize
- `observation.images.cam_left_wrist`: RAW left wrist camera, RGB uint8, no resize
- `observation.images.cam_right_wrist`: RAW right wrist camera, RGB uint8, no resize

## State And Action Fields

- `observation.qpos`: `float32[14]`
  - `left.j0..left.j5`, `left.gripper`, `right.j0..right.j5`, `right.gripper`
- `observation.qvel`: `float32[14]`
  - arm joint velocities from RAW joint feedback
  - gripper velocity is finite-difference velocity of the binary gripper state
- `observation.effort`: `float32[14]`
  - arm joint effort from RAW joint feedback
  - gripper effort values are explicit `0.0` placeholders because RAW has no gripper effort source
- `observation.eef`: `float32[16]`
  - `left.x`, `left.y`, `left.z`, `left.qw`, `left.qx`, `left.qy`, `left.qz`, `left.gripper`
  - `right.x`, `right.y`, `right.z`, `right.qw`, `right.qx`, `right.qy`, `right.qz`, `right.gripper`
- `action`: `float32[14]`, next-frame absolute qpos target
- `action_eef`: `float32[16]`, next-frame absolute eef target
- `base_action`: `float32[2]`, explicit `[0.0, 0.0]` placeholder because RAW has no mobile-base action source

## Gripper Convention

Gripper values are binary:

- `0.0`: open
- `1.0`: closed

Thresholds:

- left gripper: raw value `<= 50.0` is closed
- right gripper: raw value `<= 98.0` is closed

## Quaternion Convention

RAW TCP pose stores quaternions as `qx, qy, qz, qw`. This dataset exports EEF quaternions as `qw, qx, qy, qz`. Quaternions are normalized during conversion; invalid zero-norm quaternions fail conversion.

## Action Alignment

For frame `t`, `action[t] = observation.qpos[t + 1]` and `action_eef[t] = observation.eef[t + 1]`. The final frame reuses its own qpos/eef as the target to keep each episode length unchanged.
"""
    (root / "FORMAT.md").write_text(content, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="RAW episode directory or parent directory.")
    parser.add_argument("--raw-dir-glob", default="*_RAW")
    parser.add_argument("--recursive", action="store_true", default=False)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--task-desc", default=DEFAULT_TASK_DESC)
    parser.add_argument("--gripper-threshold", type=float, default=75.0)
    parser.add_argument("--left-gripper-threshold", type=float, default=50.0)
    parser.add_argument("--right-gripper-threshold", type=float, default=98.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--video-codec", choices=["h264", "hevc", "libsvtav1"], default="h264")
    parser.add_argument("--max-episodes", type=int, default=None)
    for key, default_topic in DEFAULT_TOPICS.items():
        parser.add_argument(
            f"--{key.replace('_', '-')}-topic",
            dest=f"{key}_topic",
            default=None,
            help=f"Override topic for {key}: {default_topic}",
        )
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error(f"--max-episodes must be positive, got {args.max_episodes}")
    if args.left_gripper_threshold is None:
        args.left_gripper_threshold = args.gripper_threshold
    if args.right_gripper_threshold is None:
        args.right_gripper_threshold = args.gripper_threshold
    return args


def main() -> None:
    args = _parse_args()
    if not args.no_videos:
        _patch_video_encoder(args.video_codec)
    input_dirs = resolve_input_dirs([args.input_dir], args.raw_dir_glob, args.recursive)
    if args.max_episodes is not None:
        input_dirs = input_dirs[: args.max_episodes]
    if not input_dirs:
        raise ValueError("No RAW episodes resolved.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_dir}")
        shutil.rmtree(output_dir)

    topics = _topic_overrides(args)
    print(f"Reading episode 1/{len(input_dirs)}: {input_dirs[0]}", flush=True)
    first_samples, fps = _episode_samples(input_dirs[0], topics)
    print(f"Inferred dataset fps from head camera: {fps}", flush=True)
    image_shapes = {
        "head": tuple(np.asarray(first_samples[0]["head"]).shape),
        "left_wrist": tuple(np.asarray(first_samples[0]["left_wrist"]).shape),
        "right_wrist": tuple(np.asarray(first_samples[0]["right_wrist"]).shape),
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        root=output_dir,
        robot_type="r1lite_dual",
        features=_features(image_shapes, "image" if args.no_videos else "video"),
        use_videos=not args.no_videos,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        video_backend=args.video_backend,
    )

    failed: list[tuple[Path, str]] = []
    for index, input_dir in enumerate(input_dirs):
        try:
            if index == 0:
                samples = first_samples
            else:
                print(f"Reading episode {index + 1}/{len(input_dirs)}: {input_dir}", flush=True)
                samples, _ = _episode_samples(input_dir, topics, expected_fps=fps)
            print(f"Exporting episode {index + 1}/{len(input_dirs)} with {len(samples)} frames", flush=True)
            _add_episode(
                dataset,
                samples,
                args.task_desc,
                args.left_gripper_threshold,
                args.right_gripper_threshold,
            )
        except Exception as exc:
            failed.append((input_dir, str(exc)))
            print(f"FAILED episode {index + 1}/{len(input_dirs)}: {input_dir}: {exc}", flush=True)
            break
        finally:
            if index == 0:
                first_samples = []

    if failed:
        details = "\n".join(f"- {path}: {error}" for path, error in failed)
        raise RuntimeError(f"Conversion failed; no episodes were silently skipped:\n{details}")

    _write_format_doc(output_dir, args.repo_id, fps, "image" if args.no_videos else args.video_codec)
    print(f"Exported {len(input_dirs)} episodes to {output_dir}", flush=True)
    print(f"Wrote dataset format documentation to {output_dir / 'FORMAT.md'}", flush=True)


if __name__ == "__main__":
    main()
