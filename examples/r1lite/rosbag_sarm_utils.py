#!/usr/bin/env python3

import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


DEFAULT_TOPICS = {
    "head": "/hdas/camera_head/right_raw/image_raw_color/compressed",
    "left_wrist": "/hdas/camera_wrist_left/color/image_raw/compressed",
    "right_wrist": "/hdas/camera_wrist_right/color/image_raw/compressed",
    "left_tcp_pose": "/motion_control/pose_ee_arm_left",
    "right_tcp_pose": "/motion_control/pose_ee_arm_right",
    "left_joint": "/hdas/feedback_arm_left",
    "right_joint": "/hdas/feedback_arm_right",
    "left_gripper": "/hdas/feedback_gripper_left",
    "right_gripper": "/hdas/feedback_gripper_right",
}


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def is_raw_episode_dir(path: Path) -> bool:
    return (path / "metadata.yaml").exists() and bool(list(path.glob("*.mcap")))


def resolve_input_dirs(raw_inputs: List[str], raw_dir_glob: str = "*_RAW", recursive: bool = False) -> List[Path]:
    resolved: List[Path] = []
    for item in raw_inputs:
        path = Path(item).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        if is_raw_episode_dir(path):
            resolved.append(path)
            continue
        if not path.is_dir():
            raise ValueError(f"Input path is not a RAW episode directory or parent directory: {path}")

        candidates = path.rglob(raw_dir_glob) if recursive else path.glob(raw_dir_glob)
        raw_dirs = sorted(candidate.resolve() for candidate in candidates if candidate.is_dir() and is_raw_episode_dir(candidate))
        if not raw_dirs:
            mode = "recursively " if recursive else ""
            raise FileNotFoundError(f"No RAW episode directories matching {raw_dir_glob!r} found {mode}under {path}")
        resolved.extend(raw_dirs)

    unique: List[Path] = []
    seen = set()
    for path in resolved:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


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


EEF_ACTION_NAMES = (
    [f"left_eef_delta_{name}" for name in ("x", "y", "z", "rx", "ry", "rz")]
    + ["left_gripper_delta"]
    + [f"right_eef_delta_{name}" for name in ("x", "y", "z", "rx", "ry", "rz")]
    + ["right_gripper_delta"]
)


JOINT_ACTION_NAMES = (
    [f"left_joint_delta_{i}" for i in range(6)]
    + ["left_gripper_delta"]
    + [f"right_joint_delta_{i}" for i in range(6)]
    + ["right_gripper_delta"]
)


@dataclass
class TopicSeries:
    timestamps: np.ndarray
    values: List[Any]


def resolve_metadata_path(input_dir: Path) -> Path:
    metadata_path = input_dir / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.yaml not found under {input_dir}")
    if not list(input_dir.glob("*.mcap")):
        raise FileNotFoundError(f"No .mcap file found under {input_dir}")
    return metadata_path


def load_metadata(input_dir: Path) -> dict:
    with resolve_metadata_path(input_dir).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_rosbags_input_dir(input_dir: Path, metadata: dict) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    bag_info = metadata["rosbag2_bagfile_information"]
    relative_paths = [str(x) for x in bag_info.get("relative_file_paths", [])]
    if not relative_paths:
        return input_dir, None

    missing = [name for name in relative_paths if not (input_dir / name).exists()]
    if not missing:
        return input_dir, None

    actual_mcap_files = sorted(input_dir.glob("*.mcap"))
    if len(actual_mcap_files) != 1:
        raise FileNotFoundError(f"metadata.yaml expects missing MCAP files {missing} under {input_dir}")

    temp_dir = tempfile.TemporaryDirectory(prefix="r1lite_sarm_rosbags_compat_", dir="/tmp")
    compat_dir = Path(temp_dir.name)
    shutil.copy2(input_dir / "metadata.yaml", compat_dir / "metadata.yaml")
    actual_mcap = actual_mcap_files[0]
    for relative_name in relative_paths:
        target = compat_dir / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(actual_mcap)
    return compat_dir, temp_dir


def load_rosbags_reader():
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise RuntimeError("rosbags is required. Install it with: python -m pip install rosbags") from exc
    return AnyReader


def _fit_vector(values: Sequence[float], length: int) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32)
    out = np.zeros((length,), dtype=np.float32)
    if arr.size:
        out[: min(length, arr.size)] = arr[:length]
    return out


def decode_compressed_image(msg) -> np.ndarray:
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode CompressedImage")
    return bgr[..., ::-1].astype(np.uint8)


def decode_image(msg) -> np.ndarray:
    height = int(msg.height)
    width = int(msg.width)
    encoding = str(msg.encoding).lower()
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if encoding in ("rgb8", "bgr8"):
        img = raw.reshape(height, width, 3)
        if encoding == "bgr8":
            img = img[..., ::-1]
        return img.astype(np.uint8)
    if encoding in ("mono8", "8uc1"):
        img = raw.reshape(height, width)
        return np.repeat(img[..., None], 3, axis=-1).astype(np.uint8)
    raise ValueError(f"Unsupported sensor_msgs/Image encoding for SARM export: {msg.encoding}")


def extract_pose_stamped(msg) -> np.ndarray:
    pose = msg.pose
    return np.asarray(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=np.float32,
    )


def extract_joint_state(msg, length: int = 6) -> Dict[str, np.ndarray]:
    return {
        "position": _fit_vector(getattr(msg, "position", []), length),
        "velocity": _fit_vector(getattr(msg, "velocity", []), length),
        "effort": _fit_vector(getattr(msg, "effort", []), length),
    }


def extract_gripper_state(msg) -> np.ndarray:
    position = list(getattr(msg, "position", []))
    return np.asarray([float(position[0]) if position else 0.0], dtype=np.float32)


def parse_message(topic: str, msg) -> Any:
    if "compressed" in topic:
        return decode_compressed_image(msg)
    if "image_raw" in topic:
        return decode_image(msg)
    if "pose_ee_arm_" in topic:
        return extract_pose_stamped(msg)
    if "feedback_arm_" in topic:
        return extract_joint_state(msg)
    if "feedback_gripper_" in topic:
        return extract_gripper_state(msg)
    return msg


def read_topic_series(input_dir: Path, topics: Sequence[str]) -> Dict[str, TopicSeries]:
    input_dir = input_dir.expanduser().resolve()
    metadata = load_metadata(input_dir)
    AnyReader = load_rosbags_reader()
    topic_set = set(topics)
    buckets: Dict[str, List[Tuple[int, Any]]] = defaultdict(list)
    reader_input_dir, temp_dir = prepare_rosbags_input_dir(input_dir, metadata)
    try:
        with AnyReader([reader_input_dir]) as reader:
            connections = [conn for conn in reader.connections if conn.topic in topic_set]
            missing = topic_set.difference({conn.topic for conn in connections})
            if missing:
                raise ValueError(f"Missing topics in rosbag {input_dir}: {sorted(missing)}")
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                buckets[connection.topic].append((int(timestamp), parse_message(connection.topic, msg)))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    result = {}
    for topic in topics:
        samples = buckets.get(topic, [])
        if not samples:
            raise ValueError(f"No samples read for topic: {topic}")
        result[topic] = TopicSeries(
            timestamps=np.asarray([item[0] for item in samples], dtype=np.int64),
            values=[item[1] for item in samples],
        )
    return result


def value_at(series: TopicSeries, timestamp_ns: int) -> Any:
    idx = int(np.searchsorted(series.timestamps, timestamp_ns, side="right") - 1)
    if idx < 0:
        idx = 0
    return series.values[idx]


def nearest_value_at(series: TopicSeries, timestamp_ns: int) -> Any:
    idx = int(np.searchsorted(series.timestamps, timestamp_ns, side="left"))
    if idx <= 0:
        return series.values[0]
    if idx >= len(series.timestamps):
        return series.values[-1]
    prev_idx = idx - 1
    if abs(int(series.timestamps[idx]) - int(timestamp_ns)) < abs(int(series.timestamps[prev_idx]) - int(timestamp_ns)):
        return series.values[idx]
    return series.values[prev_idx]


def overlapping_timeline(series_map: Dict[str, TopicSeries], hz: float) -> np.ndarray:
    starts = [int(series.timestamps[0]) for series in series_map.values()]
    ends = [int(series.timestamps[-1]) for series in series_map.values()]
    start_ns = max(starts)
    end_ns = min(ends)
    if end_ns <= start_ns:
        raise ValueError("No overlapping time interval across selected topics")
    step_ns = int(round(1e9 / hz))
    if step_ns <= 0:
        raise ValueError(f"Invalid fps/hz: {hz}")
    return np.arange(start_ns, end_ns + 1, step_ns, dtype=np.int64)


def rotvec_delta(current_xyzw: np.ndarray, next_xyzw: np.ndarray) -> np.ndarray:
    return (Rotation.from_quat(current_xyzw).inv() * Rotation.from_quat(next_xyzw)).as_rotvec().astype(np.float32)


def euler_left_delta_xyz(current_xyzw: np.ndarray, next_xyzw: np.ndarray) -> np.ndarray:
    current = Rotation.from_quat(current_xyzw)
    nxt = Rotation.from_quat(next_xyzw)
    return (nxt * current.inv()).as_euler("xyz").astype(np.float32)


def tcp_velocity(current_pose: np.ndarray, next_pose: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0:
        return np.zeros((6,), dtype=np.float32)
    linear = (next_pose[:3] - current_pose[:3]) / dt
    angular = rotvec_delta(current_pose[3:], next_pose[3:]) / dt
    return np.concatenate([linear, angular], axis=0).astype(np.float32)


def state_vector(sample: Dict[str, Any], next_sample: Optional[Dict[str, Any]], dt: float) -> np.ndarray:
    next_sample = sample if next_sample is None else next_sample
    left_tcp_vel = tcp_velocity(sample["left_tcp_pose"], next_sample["left_tcp_pose"], dt)
    right_tcp_vel = tcp_velocity(sample["right_tcp_pose"], next_sample["right_tcp_pose"], dt)
    return np.concatenate(
        [
            sample["left_tcp_pose"],
            left_tcp_vel,
            sample["left_joint"]["position"],
            sample["left_joint"]["velocity"],
            sample["left_gripper"],
            sample["right_tcp_pose"],
            right_tcp_vel,
            sample["right_joint"]["position"],
            sample["right_joint"]["velocity"],
            sample["right_gripper"],
            np.zeros((1,), dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def action_vector(current: Dict[str, Any], nxt: Dict[str, Any], action_space: str) -> np.ndarray:
    left_gripper_delta = nxt["left_gripper"] - current["left_gripper"]
    right_gripper_delta = nxt["right_gripper"] - current["right_gripper"]
    if action_space == "eef":
        left_delta = np.concatenate(
            [
                nxt["left_tcp_pose"][:3] - current["left_tcp_pose"][:3],
                euler_left_delta_xyz(current["left_tcp_pose"][3:], nxt["left_tcp_pose"][3:]),
            ]
        )
        right_delta = np.concatenate(
            [
                nxt["right_tcp_pose"][:3] - current["right_tcp_pose"][:3],
                euler_left_delta_xyz(current["right_tcp_pose"][3:], nxt["right_tcp_pose"][3:]),
            ]
        )
    elif action_space == "joint":
        left_delta = nxt["left_joint"]["position"] - current["left_joint"]["position"]
        right_delta = nxt["right_joint"]["position"] - current["right_joint"]["position"]
    else:
        raise ValueError(f"Unsupported action_space: {action_space}")
    return np.concatenate([left_delta, left_gripper_delta, right_delta, right_gripper_delta], axis=0).astype(np.float32)


def build_episode_samples(input_dir: Path, fps: float, topics: Dict[str, str]) -> List[Dict[str, Any]]:
    selected_topics = [
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
    series_map = read_topic_series(input_dir, selected_topics)
    timeline = overlapping_timeline(series_map, fps)
    samples: List[Dict[str, Any]] = []
    for ts in timeline:
        samples.append(
            {
                "timestamp_ns": int(ts),
                "head": nearest_value_at(series_map[topics["head"]], int(ts)),
                "left_wrist": nearest_value_at(series_map[topics["left_wrist"]], int(ts)),
                "right_wrist": nearest_value_at(series_map[topics["right_wrist"]], int(ts)),
                "left_tcp_pose": value_at(series_map[topics["left_tcp_pose"]], int(ts)),
                "right_tcp_pose": value_at(series_map[topics["right_tcp_pose"]], int(ts)),
                "left_joint": value_at(series_map[topics["left_joint"]], int(ts)),
                "right_joint": value_at(series_map[topics["right_joint"]], int(ts)),
                "left_gripper": value_at(series_map[topics["left_gripper"]], int(ts)),
                "right_gripper": value_at(series_map[topics["right_gripper"]], int(ts)),
            }
        )
    return samples
