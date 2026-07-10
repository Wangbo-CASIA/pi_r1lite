#!/usr/bin/env python3
"""Convert R1Lite dual-arm RAW MCAP episodes to an absolute-EEF LeRobot dataset."""

import argparse
from pathlib import Path
import shutil
import sys
from typing import Any

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from r1lite_experiment_config import add_experiment_args
from r1lite_experiment_config import apply_converter_config

CONRFT_ROOT = Path("/home/ps/VLA-RL/conrft-r1lite")
SARM_EXAMPLES = CONRFT_ROOT / "examples" / "sarm"
if str(SARM_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(SARM_EXAMPLES))

from rosbag_sarm_utils import DEFAULT_TOPICS  # noqa: E402
from rosbag_sarm_utils import build_episode_samples  # noqa: E402
from rosbag_sarm_utils import resolve_input_dirs  # noqa: E402

DEFAULT_TASK_DESC = (
    "move the white box from the left to the center, then pick up the yellow-red mango on the right and place it "
    "inside the box"
)
DEFAULT_REPO_ID = "r1lite_dual_mango_box_abs_eef_pose16"
DEFAULT_OUTPUT_DIR = "/home/ps/VLA-RL/conrft-r1lite/data/lerobot_openpi/r1lite_dual_mango_box_abs_eef_pose16"
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
ACTION_NAMES = (
    [f"left_eef_target_{name}" for name in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    + ["left_gripper01_target"]
    + [f"right_eef_target_{name}" for name in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    + ["right_gripper01_target"]
)
LEFT_TCP_POSE_SLICE = slice(0, 7)
LEFT_TCP_VEL_SLICE = slice(7, 13)
LEFT_JOINT_POS_SLICE = slice(13, 19)
LEFT_JOINT_VEL_SLICE = slice(19, 25)
LEFT_GRIPPER_INDEX = 25
RIGHT_TCP_POSE_SLICE = slice(26, 33)
RIGHT_TCP_VEL_SLICE = slice(33, 39)
RIGHT_JOINT_POS_SLICE = slice(39, 45)
RIGHT_JOINT_VEL_SLICE = slice(45, 51)
RIGHT_GRIPPER_INDEX = 51


def _image_feature(image_shape: tuple[int, ...], dtype: str) -> dict[str, Any]:
    if len(image_shape) != 3:
        raise ValueError(f"Expected image shape (height, width, channel), got {image_shape}")
    return {"dtype": dtype, "shape": image_shape, "names": ["height", "width", "channel"]}


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


def _binary_gripper(gripper: np.ndarray, threshold: float) -> np.ndarray:
    value = float(np.asarray(gripper, dtype=np.float32).reshape(-1)[0])
    return np.asarray([0.0 if value > threshold else 1.0], dtype=np.float32)


def _unit_quat(pose: np.ndarray, label: str) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)[:7].copy()
    quat_norm = float(np.linalg.norm(pose[3:7]))
    if not np.isfinite(quat_norm) or quat_norm <= 0.0:
        raise ValueError(f"{label} has invalid quaternion norm: {quat_norm}")
    pose[3:7] /= quat_norm
    return pose


def _tcp_velocity(current_pose: np.ndarray, next_pose: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0:
        return np.zeros((6,), dtype=np.float32)
    current_pose = _unit_quat(current_pose, "current tcp pose")
    next_pose = _unit_quat(next_pose, "next tcp pose")
    linear = (next_pose[:3] - current_pose[:3]) / dt
    # Keep angular velocity zero here; abs-EEF training uses pose state and does not consume velocity for control.
    return np.concatenate([linear, np.zeros((3,), dtype=np.float32)], axis=0).astype(np.float32)


def _state_vector(
    sample: dict[str, Any],
    next_sample: dict[str, Any],
    dt: float,
    gripper_threshold: float,
) -> np.ndarray:
    state = np.zeros((len(STATE_NAMES),), dtype=np.float32)
    state[LEFT_TCP_POSE_SLICE] = _unit_quat(sample["left_tcp_pose"], "left tcp pose")
    state[LEFT_TCP_VEL_SLICE] = _tcp_velocity(sample["left_tcp_pose"], next_sample["left_tcp_pose"], dt)
    state[LEFT_JOINT_POS_SLICE] = np.asarray(sample["left_joint"]["position"], dtype=np.float32).reshape(-1)[:6]
    state[LEFT_JOINT_VEL_SLICE] = np.asarray(sample["left_joint"]["velocity"], dtype=np.float32).reshape(-1)[:6]
    state[LEFT_GRIPPER_INDEX] = _binary_gripper(sample["left_gripper"], gripper_threshold)[0]
    state[RIGHT_TCP_POSE_SLICE] = _unit_quat(sample["right_tcp_pose"], "right tcp pose")
    state[RIGHT_TCP_VEL_SLICE] = _tcp_velocity(sample["right_tcp_pose"], next_sample["right_tcp_pose"], dt)
    state[RIGHT_JOINT_POS_SLICE] = np.asarray(sample["right_joint"]["position"], dtype=np.float32).reshape(-1)[:6]
    state[RIGHT_JOINT_VEL_SLICE] = np.asarray(sample["right_joint"]["velocity"], dtype=np.float32).reshape(-1)[:6]
    state[RIGHT_GRIPPER_INDEX] = _binary_gripper(sample["right_gripper"], gripper_threshold)[0]
    return state


def _action_vector(next_sample: dict[str, Any], gripper_threshold: float) -> np.ndarray:
    return np.concatenate(
        [
            _unit_quat(next_sample["left_tcp_pose"], "left action pose"),
            _binary_gripper(next_sample["left_gripper"], gripper_threshold),
            _unit_quat(next_sample["right_tcp_pose"], "right action pose"),
            _binary_gripper(next_sample["right_gripper"], gripper_threshold),
        ],
        axis=0,
    ).astype(np.float32)


def _add_episode(
    dataset: LeRobotDataset,
    samples: list[dict[str, Any]],
    task_desc: str,
    fps: int,
    gripper_threshold: float,
) -> None:
    if len(samples) < 2:
        raise ValueError("Need at least two synchronized samples to export an episode.")

    for idx, sample in enumerate(samples):
        next_sample = samples[idx + 1] if idx + 1 < len(samples) else sample
        if idx + 1 < len(samples):
            dt = max(1e-6, (next_sample["timestamp_ns"] - sample["timestamp_ns"]) / 1e9)
        else:
            dt = 1.0 / fps
        frame = {
            "observation.images.head": sample["head"],
            "observation.images.left_wrist": sample["left_wrist"],
            "observation.images.right_wrist": sample["right_wrist"],
            "observation.state": _state_vector(sample, next_sample, dt, gripper_threshold),
            "action": _action_vector(next_sample, gripper_threshold),
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_args(parser, default_action_space="abs_eef")
    parser.add_argument("--input-dir", default=None, help="RAW episode directory or parent directory.")
    parser.add_argument("--raw-dir-glob", default=None)
    parser.add_argument("--recursive", action="store_true", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--task-desc", default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--gripper-threshold", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
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
        args.output_dir = DEFAULT_OUTPUT_DIR
    if args.repo_id is None:
        args.repo_id = DEFAULT_REPO_ID
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
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_dir}")
        shutil.rmtree(output_dir)

    topics = _topic_overrides(args)
    episode_samples = [build_episode_samples(path, fps, topics) for path in input_dirs]
    image_shapes = {
        "head": tuple(np.asarray(episode_samples[0][0]["head"]).shape),
        "left_wrist": tuple(np.asarray(episode_samples[0][0]["left_wrist"]).shape),
        "right_wrist": tuple(np.asarray(episode_samples[0][0]["right_wrist"]).shape),
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

    for index, samples in enumerate(episode_samples):
        print(f"Exporting episode {index + 1}/{len(episode_samples)} with {len(samples)} frames")
        _add_episode(dataset, samples, args.task_desc, fps, args.gripper_threshold)

    print(f"Exported {len(episode_samples)} episodes to {output_dir}")


if __name__ == "__main__":
    main()
