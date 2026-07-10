#!/usr/bin/env python3
"""Import successful R1Lite LeRobot episodes into ALOE replay v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from lerobot.common.datasets.video_utils import decode_video_frames

from openpi.aloe import schema
from openpi.aloe.config import load_aloe_config
from openpi.aloe.config import require_dict
from openpi.aloe.writer import AsyncPklReplayWriter
from openpi.aloe.writer import next_collection_run_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from r1lite_experiment_config import load_experiment_config  # noqa: E402


def _task_prompt(root: Path) -> str:
    tasks_path = root / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        raise FileNotFoundError(f"LeRobot dataset is missing meta/tasks.jsonl: {root}")
    with tasks_path.open("r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    task = first.get("task")
    if not isinstance(task, str) or not task:
        raise ValueError(f"invalid task entry in {tasks_path}")
    return task


def _episode_paths(root: Path, limit: int | None) -> list[Path]:
    paths = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no LeRobot parquet episodes found under {root / 'data'}")
    return paths[:limit] if limit is not None else paths


def _episode_index(path: Path) -> int:
    return int(path.stem.removeprefix("episode_"))


def _video_frame(root: Path, camera: str, episode_index: int, frame_index: int, fps: float) -> np.ndarray:
    video_path = root / "videos" / "chunk-000" / f"observation.images.{camera}" / f"episode_{episode_index:06d}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"missing LeRobot video for {camera} episode {episode_index}: {video_path}")
    frames = decode_video_frames(video_path, [float(frame_index) / float(fps)], tolerance_s=0.5 / float(fps))
    if frames.shape[0] != 1:
        raise RuntimeError(f"expected one decoded frame from {video_path}, got {frames.shape}")
    frame = frames[0].detach().cpu().numpy()
    if frame.shape[0] == 3:
        frame = np.moveaxis(frame, 0, -1)
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
    return frame.astype(np.uint8, copy=False)


def _observation(root: Path, episode_index: int, frame_index: int, state: np.ndarray, fps: float) -> dict[str, Any]:
    return {
        "state": np.asarray(state, dtype=np.float32).reshape(schema.STATE_DIM),
        "images": {
            camera: _video_frame(root, camera, episode_index, frame_index, fps)
            for camera in schema.CAMERA_KEYS
        },
    }


def _import_episode(
    writer: AsyncPklReplayWriter,
    root: Path,
    path: Path,
    *,
    run_id: str,
    prompt: str,
    action_horizon: int,
    fps: float,
) -> int:
    table = pq.read_table(path, columns=["observation.state", "action", "frame_index"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    if states.ndim != 2 or states.shape[1] != schema.STATE_DIM:
        raise ValueError(f"{path} observation.state must have shape (T, {schema.STATE_DIM}), got {states.shape}")
    if actions.ndim != 2 or actions.shape[1] != schema.ACTION_DIM:
        raise ValueError(f"{path} action must have shape (T, {schema.ACTION_DIM}), got {actions.shape}")
    if len(actions) < 2:
        raise ValueError(f"{path} must contain at least two frames")

    episode_index = _episode_index(path)
    segment_id = episode_index
    chunks = 0
    for start in range(0, len(actions), action_horizon):
        end = min(start + action_horizon, len(actions))
        chunk_actions = actions[start:end]
        next_index = min(end, len(states) - 1)
        terminal = end >= len(actions)
        rewards = np.full((len(chunk_actions),), -1.0, dtype=np.float32)
        dones = np.zeros((len(chunk_actions),), dtype=bool)
        terminal_outcome = None
        terminal_reason = None
        if terminal:
            rewards[-1] = schema.terminal_reward("success", cfail=1.0)
            dones[-1] = True
            terminal_outcome = "success"
            terminal_reason = "imported_success_demo"
        record = schema.make_chunk_record(
            run_id=run_id,
            segment_id=segment_id,
            source="human_demo",
            start_seq=start,
            prompt=prompt,
            policy_checkpoint=None,
            action_space="joint_delta",
            observation=_observation(root, episode_index, int(frame_indices[start]), states[start], fps),
            actions=chunk_actions,
            next_observation=_observation(root, episode_index, int(frame_indices[next_index]), states[next_index], fps),
            rewards=rewards,
            dones=dones,
            valid_mask=np.ones((len(chunk_actions),), dtype=bool),
            terminal_outcome=terminal_outcome,
            terminal_reason=terminal_reason,
            infos={
                "run_id": run_id,
                "prompt": prompt,
                "action_space": "joint_delta",
                "source_dataset": str(root),
                "source_episode": episode_index,
            },
        )
        writer.append_chunk(record)
        chunks += 1
    return chunks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="r1lite_pack_phone")
    parser.add_argument("--config", default=None, help="Path to experiments/r1lite/<name>/aloe.yaml")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--collection", default="human_demo")
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-size-chunks", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    aloe_cfg = load_aloe_config(args.experiment, args.config)
    replay_cfg = require_dict(aloe_cfg, "replay")
    actor_cfg = require_dict(aloe_cfg, "actor")
    exp_cfg = load_experiment_config(args.experiment, None, "joint_delta")
    if exp_cfg is None:
        raise RuntimeError(f"missing R1Lite experiment config for {args.experiment}")
    action_cfg = exp_cfg["action_spaces"]["joint_delta"]
    input_root = args.input or Path(action_cfg["lerobot_dir"])
    output_root = args.output or Path(replay_cfg["root"])
    action_horizon = args.action_horizon or int(actor_cfg["action_horizon"])
    if action_horizon <= 0:
        raise ValueError("--action-horizon must be positive")

    prompt = _task_prompt(input_root)
    info_path = input_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"LeRobot dataset is missing meta/info.json: {input_root}")
    fps = float(json.loads(info_path.read_text(encoding="utf-8"))["fps"])
    run_id, run_dir = next_collection_run_dir(output_root, args.collection)
    episode_paths = _episode_paths(input_root, args.limit)
    with AsyncPklReplayWriter(
        run_dir,
        run_id=run_id,
        prompt=prompt,
        policy_checkpoint=None,
        action_space="joint_delta",
        shard_size_chunks=args.shard_size_chunks,
        metadata={
            "importer": Path(__file__).name,
            "source_dataset": str(input_root),
            "episodes": len(episode_paths),
            "action_horizon": action_horizon,
        },
    ) as writer:
        total_chunks = 0
        for path in episode_paths:
            total_chunks += _import_episode(
                writer,
                input_root,
                path,
                run_id=run_id,
                prompt=prompt,
                action_horizon=action_horizon,
                fps=fps,
            )
    print(f"imported episodes={len(episode_paths)} chunks={total_chunks} to {run_dir}")


if __name__ == "__main__":
    main()
