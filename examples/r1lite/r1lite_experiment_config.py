#!/usr/bin/env python3
"""Experiment config loading for R1Lite OpenPI workflows."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "r1lite"
ACTION_SPACE_ALIASES = {
    "abs_joint": "abs_joint",
    "joint_abs": "abs_joint",
    "delta_joint": "joint_delta",
    "joint_delta": "joint_delta",

    "abs_eef": "abs_eef",
    "delta_eef": "delta_eef",
    "eef_abs": "abs_eef",
    "eef_delta": "delta_eef",
}


def normalize_action_space(value: str) -> str:
    try:
        return ACTION_SPACE_ALIASES[value]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(ACTION_SPACE_ALIASES.values())))
        raise ValueError(f"unknown action space {value!r}; expected one of: {allowed}") from exc


def add_experiment_args(parser: argparse.ArgumentParser,default_experiment: str, default_action_space: str) -> None:
    parser.add_argument("--experiment", default=default_experiment, help="Experiment name under experiments/r1lite/<name>.")
    parser.add_argument("--config", default=None, help="Path to an explicit R1Lite experiment config.yaml.")
    parser.add_argument(
        "--action-space",
        default=default_action_space,
        choices=("joint_delta","abs_joint" "abs_eef", "delta_eef"),
        help="Action-space section to use from the experiment config.",
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"R1Lite experiment config does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"R1Lite experiment config must be a YAML object: {path}")
    return data


def _config_path(experiment: str | None, config: str | None) -> Path | None:
    if config and experiment:
        raise ValueError("--config and --experiment are mutually exclusive")
    if config:
        return Path(config).expanduser().resolve()
    if experiment:
        return (DEFAULT_EXPERIMENT_ROOT / experiment / "config.yaml").resolve()
    return None


def load_experiment_config(experiment: str | None, config: str | None, action_space: str) -> dict[str, Any] | None:
    path = _config_path(experiment, config)
    if path is None:
        return None
    data = _load_yaml(path)
    data["_config_path"] = str(path)
    action_space = normalize_action_space(action_space)
    action_spaces = _required_dict(data, "action_spaces")
    if action_space not in action_spaces:
        raise ValueError(f"config {path} has no action_spaces.{action_space} section")
    return data


def _required_dict(root: dict[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"R1Lite experiment config is missing object section: {key}")
    return value


def _set_if_none(args: argparse.Namespace, attr: str, value: Any) -> None:
    if value is not None and hasattr(args, attr) and getattr(args, attr) is None:
        setattr(args, attr, value)


def _set_bool_default(args: argparse.Namespace, attr: str, value: Any) -> None:
    if value is not None and hasattr(args, attr) and getattr(args, attr) is None:
        setattr(args, attr, bool(value))


def apply_rollout_config(args: argparse.Namespace) -> argparse.Namespace:
    cfg = load_experiment_config(args.experiment, args.config, args.action_space)
    if cfg is None:
        return args
    action_space = normalize_action_space(args.action_space)
    args.action_space = action_space
    args.experiment_config_path = cfg["_config_path"]

    experiment = _required_dict(cfg, "experiment")
    robot = _required_dict(cfg, "robot")
    rollout = _required_dict(cfg, "rollout")
    hg_dagger = _required_dict(cfg, "hg_dagger")
    local = _required_dict(cfg, "local")

    _set_if_none(args, "prompt", experiment.get("prompt") or experiment.get("task_desc"))
    _set_if_none(args, "robot_server", robot.get("server_url"))
    _set_if_none(args, "policy_host", rollout.get("policy_host"))
    _set_if_none(args, "policy_port", rollout.get("policy_port"))
    _set_if_none(args, "control_hz", rollout.get("control_hz"))
    _set_if_none(args, "actions_per_infer", rollout.get("actions_per_infer"))
    _set_if_none(args, "timeout", rollout.get("timeout"))
    _set_if_none(args, "gripper_open_value", rollout.get("gripper_open_value"))
    _set_if_none(args, "gripper_close_value", rollout.get("gripper_close_value"))
    _set_if_none(args, "gripper_threshold", rollout.get("gripper_threshold"))
    _set_if_none(args, "left_gripper_threshold", rollout.get("left_gripper_threshold"))
    _set_if_none(args, "right_gripper_threshold", rollout.get("right_gripper_threshold"))
    rtc = rollout.get("rtc")
    if rtc is not None:
        if not isinstance(rtc, dict):
            raise ValueError("rollout.rtc must be an object when provided")
        _set_bool_default(args, "rtc", rtc.get("enabled"))
        _set_if_none(args, "rtc_latency_k", rtc.get("latency_k"))
        _set_if_none(args, "rtc_inference_rate", rtc.get("inference_rate"))
        _set_if_none(args, "rtc_max_guidance_weight", rtc.get("max_guidance_weight"))

    _set_if_none(args, "record_output_dir", robot.get("record_output_dir"))
    _set_if_none(args, "record_stop_timeout", robot.get("record_stop_timeout"))
    _set_if_none(args, "record_topics", robot.get("record_topics"))
    for attr in (
        "intervention_arm",
        "left_spacemouse_path",
        "right_spacemouse_path",
        "teleop_calibrate_seconds",
        "teleop_trans_deadzone",
        "teleop_rot_deadzone",
        "intervention_activate_threshold",
        "intervention_release_threshold",
        "teleop_xyz_scale",
        "teleop_rot_scale",
        "teleop_source",
        "teleop_idle_seconds",
        "takeover_http_path",
        "takeover_switch_delay_sec",
        "tabletop_release_key",
        "tabletop_restore_timeout_sec",
        "tabletop_restore_joint_tolerance",
        "operator_event_host",
        "operator_event_port",
        "operator_event_path",
    ):
        _set_if_none(args, attr, hg_dagger.get(attr))

    return args


def apply_converter_config(args: argparse.Namespace) -> argparse.Namespace:
    cfg = load_experiment_config(args.experiment, args.config, args.action_space)
    if cfg is None:
        return args
    action_space = normalize_action_space(args.action_space)
    args.action_space = action_space
    args.experiment_config_path = cfg["_config_path"]

    experiment = _required_dict(cfg, "experiment")
    data = _required_dict(cfg, "data")
    local = _required_dict(cfg, "local")
    action_cfg = copy.deepcopy(_required_dict(_required_dict(cfg, "action_spaces"), action_space))

    _set_if_none(args, "input_dir", local.get("raw_dir"))
    _set_if_none(args, "raw_dir_glob", data.get("raw_dir_glob"))
    _set_bool_default(args, "recursive", data.get("recursive"))
    _set_if_none(args, "output_dir", action_cfg.get("lerobot_dir"))
    _set_if_none(args, "repo_id", action_cfg.get("repo_id"))
    _set_if_none(args, "task_desc", experiment.get("task_desc") or experiment.get("prompt"))
    _set_if_none(args, "fps", data.get("fps"))
    _set_if_none(args, "gripper_threshold", data.get("gripper_threshold"))
    _set_if_none(args, "left_gripper_threshold", data.get("left_gripper_threshold"))
    _set_if_none(args, "right_gripper_threshold", data.get("right_gripper_threshold"))
    _set_if_none(args, "image_writer_processes", data.get("image_writer_processes"))
    _set_if_none(args, "image_writer_threads", data.get("image_writer_threads"))
    _set_if_none(args, "video_backend", data.get("video_backend"))

    topics = data.get("topics")
    if topics is not None:
        if not isinstance(topics, dict):
            raise ValueError("data.topics must be an object when provided")
        for key, value in topics.items():
            attr = f"{key}_topic"
            if hasattr(args, attr):
                _set_if_none(args, attr, value)
    return args
