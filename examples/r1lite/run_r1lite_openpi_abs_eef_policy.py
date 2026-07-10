#!/usr/bin/env python3
"""Run an absolute-EEF openpi R1Lite policy against the R1Lite body service."""

import argparse
import base64
import dataclasses
import json
import sys
import termios
import time
import tty
from typing import Any
import urllib.error
import urllib.request

import cv2
import numpy as np
from openpi_client import rtc_client_policy
from openpi_client import websocket_client_policy
from r1lite_experiment_config import add_experiment_args
from r1lite_experiment_config import apply_rollout_config
from r1lite_hg_dagger import HgDaggerInterventionRuntime
from r1lite_hg_dagger import add_hg_dagger_args
from r1lite_hg_dagger import list_spacemouse_devices
from r1lite_hg_dagger import validate_hg_dagger_args
from r1lite_rtc import RtcInferenceLoop
from r1lite_rtc import add_rtc_args
from r1lite_rtc import apply_rtc_defaults
from r1lite_rtc import validate_rtc_args

DEFAULT_PROMPT = (
    "move the white box from the left to the center, then pick up the yellow-red mango on the right and place it "
    "inside the box"
)
LEFT_TCP_POSE_SLICE = slice(0, 7)
LEFT_GRIPPER_INDEX = 25
RIGHT_TCP_POSE_SLICE = slice(26, 33)
RIGHT_GRIPPER_INDEX = 51
STATE_DIM = 53
ACTION_DIM = 16


class Color:
    RESET = "\033[0m"
    TARGET = "\033[96m"
    CURRENT = "\033[93m"
    ERROR = "\033[91m"
    DIM = "\033[90m"


@dataclasses.dataclass(frozen=True)
class StepRecord:
    seq: int
    left_target: np.ndarray
    right_target: np.ndarray
    left_gripper: float
    right_gripper: float
    action: np.ndarray


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _request_json(method: str, url: str, timeout: float, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc

    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{method} {url} returned non-JSON response: {body[:500]}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"{method} {url} returned JSON {type(result).__name__}, expected object")
    return result


def _get_robot_state(robot_server: str, timeout: float) -> dict[str, Any]:
    return _request_json("GET", _url(robot_server, "/state"), timeout)


def _post_action(robot_server: str, timeout: float, payload: dict[str, Any]) -> dict[str, Any]:
    return _request_json("POST", _url(robot_server, "/action"), timeout, payload)


def _decode_image(encoded: Any, key: str) -> np.ndarray:
    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"Expected images.{key} to be a non-empty base64 string")
    image_bytes = base64.b64decode(encoded, validate=True)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to decode images.{key} as an encoded image")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _arm_state(raw_state: dict[str, Any], side: str) -> dict[str, Any]:
    state = raw_state.get("state")
    if not isinstance(state, dict):
        raise ValueError("Robot state response is missing object field: state")
    arm = state.get(side)
    if not isinstance(arm, dict):
        raise ValueError(f"Robot state response is missing object field: state.{side}")
    return arm


def _tcp_pose(raw_state: dict[str, Any], side: str) -> np.ndarray:
    tcp_pose = _arm_state(raw_state, side).get("tcp_pose")
    pose = np.asarray(tcp_pose, dtype=np.float32).reshape(-1)
    if pose.shape[0] < 7:
        raise ValueError(f"Expected state.{side}.tcp_pose to contain at least 7 values, got {pose.shape[0]}")
    return _unit_quat(pose[:7], f"state.{side}.tcp_pose")


def _gripper_pose(raw_state: dict[str, Any], side: str) -> float:
    gripper_pose = _arm_state(raw_state, side).get("gripper_pose")
    gripper = np.asarray(gripper_pose, dtype=np.float32).reshape(-1)
    if gripper.shape[0] < 1:
        raise ValueError(f"Expected state.{side}.gripper_pose to contain at least 1 value, got {gripper.shape[0]}")
    return float(gripper[0])


def _binary_gripper(value: float, threshold: float) -> float:
    return 0.0 if float(value) > threshold else 1.0


def _unit_quat(pose: np.ndarray, label: str) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)[:7].copy()
    norm = float(np.linalg.norm(pose[3:7]))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{label} has invalid quaternion norm: {norm}")
    pose[3:7] /= norm
    return pose


def _state_vector(raw_state: dict[str, Any], gripper_threshold: float) -> np.ndarray:
    state = np.zeros((STATE_DIM,), dtype=np.float32)
    state[LEFT_TCP_POSE_SLICE] = _tcp_pose(raw_state, "left")
    state[LEFT_GRIPPER_INDEX] = _binary_gripper(_gripper_pose(raw_state, "left"), gripper_threshold)
    state[RIGHT_TCP_POSE_SLICE] = _tcp_pose(raw_state, "right")
    state[RIGHT_GRIPPER_INDEX] = _binary_gripper(_gripper_pose(raw_state, "right"), gripper_threshold)
    return state


def _validate_freshness(raw_state: dict[str, Any]) -> None:
    meta = raw_state.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("Robot state response is missing object field: meta")
    validity = meta.get("validity")
    if not isinstance(validity, dict):
        raise ValueError("Robot state response is missing object field: meta.validity")

    image_validity = validity.get("images")
    if not isinstance(image_validity, dict):
        raise ValueError("Robot state response is missing object field: meta.validity.images")
    stale_images = [key for key in ("head", "left_wrist", "right_wrist") if not bool(image_validity.get(key))]
    if stale_images:
        raise ValueError(f"Robot state has stale or invalid images: {stale_images}")

    for side in ("left", "right"):
        arm_validity = validity.get(side)
        if not isinstance(arm_validity, dict):
            raise ValueError(f"Robot state response is missing object field: meta.validity.{side}")
        missing = [key for key in ("joint_state", "gripper") if not bool(arm_validity.get(key))]
        if missing:
            raise ValueError(f"Robot state has invalid {side} fields: {missing}")


def _observation(raw_state: dict[str, Any], prompt: str, gripper_threshold: float) -> dict[str, Any]:
    _validate_freshness(raw_state)
    images = raw_state.get("images")
    if not isinstance(images, dict):
        raise ValueError("Robot state response is missing object field: images")
    return {
        "images": {
            "head": _decode_image(images.get("head"), "head"),
            "left_wrist": _decode_image(images.get("left_wrist"), "left_wrist"),
            "right_wrist": _decode_image(images.get("right_wrist"), "right_wrist"),
        },
        "state": _state_vector(raw_state, gripper_threshold),
        "prompt": prompt,
    }


def _policy_actions(client: websocket_client_policy.WebsocketClientPolicy, observation: dict[str, Any]) -> np.ndarray:
    result = client.infer(observation)
    if "actions" not in result:
        raise RuntimeError(f"Policy response is missing 'actions'. Keys: {sorted(result)}")
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise RuntimeError(f"Expected policy actions with shape (horizon, {ACTION_DIM}), got {actions.shape}")
    if actions.shape[0] < 1:
        raise RuntimeError("Policy returned an empty action chunk")
    return actions


def _gripper_command(binary_value: float, open_value: float, close_value: float) -> float:
    return float(close_value if float(binary_value) >= 0.5 else open_value)


def _payload_from_action(
    action: np.ndarray,
    seq: int,
    gripper_open_value: float,
    gripper_close_value: float,
    include_gripper_command: bool,  # noqa: FBT001
) -> tuple[dict[str, Any], StepRecord]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != ACTION_DIM:
        raise ValueError(f"Expected one action with {ACTION_DIM} values, got {action.shape[0]}")

    left_target = _unit_quat(action[:7], "left action pose")
    right_target = _unit_quat(action[8:15], "right action pose")
    left_gripper = _gripper_command(action[7], gripper_open_value, gripper_close_value)
    right_gripper = _gripper_command(action[15], gripper_open_value, gripper_close_value)
    left_payload = {
        "pose_target": [float(value) for value in left_target],
        "preset": "free_space",
    }
    right_payload = {
        "pose_target": [float(value) for value in right_target],
        "preset": "free_space",
    }
    if include_gripper_command:
        left_payload["gripper"] = left_gripper
        right_payload["gripper"] = right_gripper
    payload = {
        "mode": "ee_pose_servo",
        "owner": "policy",
        "seq": seq,
        "left": left_payload,
        "right": right_payload,
    }
    return payload, StepRecord(
        seq=seq,
        left_target=left_target,
        right_target=right_target,
        left_gripper=left_gripper,
        right_gripper=right_gripper,
        action=action.copy(),
    )


def _format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value): .5f}" for value in values.reshape(-1)) + "]"


def _print_error_report(record: StepRecord, raw_state: dict[str, Any]) -> None:
    for side, target in (("left", record.left_target), ("right", record.right_target)):
        current = _tcp_pose(raw_state, side)
        position_error = target[:3] - current[:3]
        quat_dot = float(abs(np.dot(target[3:7], current[3:7])))
        quat_dot = min(1.0, max(-1.0, quat_dot))
        ori_error = 2.0 * np.arccos(quat_dot)
        print(f"{Color.DIM}{side} tcp pose, seq={record.seq}{Color.RESET}")
        print(f"  {Color.TARGET}target : {_format_vector(target)}{Color.RESET}")
        print(f"  {Color.CURRENT}current: {_format_vector(current)}{Color.RESET}")
        print(f"  {Color.ERROR}pos err: {_format_vector(position_error)}, ori err rad: {ori_error: .5f}{Color.RESET}")


def _read_debug_key() -> str:
    prompt = "debug command [c=execute next action, e=print tcp error, q=quit]: "
    print(prompt, end="", flush=True)
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        key = line[:1].lower()
        print(key)
        return key

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    print(key)
    return key


def _print_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _execute_or_print(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    if args.execute:
        response = _post_action(args.robot_server, args.timeout, payload)
        print(f"accepted seq={payload['seq']}: {response}")
    else:
        print(f"dry-run seq={payload['seq']}:")
        _print_payload(payload)


def _next_action_record(
    args: argparse.Namespace,
    action: np.ndarray,
    seq: int,
    include_gripper_command: bool,  # noqa: FBT001
) -> StepRecord:
    payload, record = _payload_from_action(
        action,
        seq,
        args.gripper_open_value,
        args.gripper_close_value,
        include_gripper_command,
    )
    _execute_or_print(args, payload)
    return record


def _rtc_executable_chunk(args: argparse.Namespace, raw_state: dict[str, Any], actions: np.ndarray) -> np.ndarray:
    _ = raw_state
    chunk_len = min(args.actions_per_infer, actions.shape[0])
    chunk = np.asarray(actions[:chunk_len], dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected RTC absolute EEF chunk with shape (horizon, {ACTION_DIM}), got {chunk.shape}")
    return chunk.copy()


def _next_rtc_action_record(args: argparse.Namespace, action: np.ndarray, seq: int) -> StepRecord:
    return _next_action_record(args, action, seq, include_gripper_command=True)


def _run_debug_chunk(args: argparse.Namespace, actions: np.ndarray, steps_done: int, seq: int) -> tuple[int, int, bool]:
    chunk_len = min(args.actions_per_infer, actions.shape[0], args.max_steps - steps_done)
    chunk = actions[:chunk_len]
    action_index = 0
    last_record: StepRecord | None = None
    print(f"debug chunk ready: {chunk_len} actions. Press c to execute one action.")

    while action_index < chunk_len:
        key = _read_debug_key()
        if key == "q":
            return steps_done, seq, True
        if key == "e":
            if last_record is None:
                print("No action has been executed in this chunk yet.")
                continue
            latest_state = _get_robot_state(args.robot_server, args.timeout)
            _print_error_report(last_record, latest_state)
            continue
        if key != "c":
            print(f"Unsupported debug command: {key!r}")
            continue

        last_record = _next_action_record(
            args,
            chunk[action_index],
            seq,
            include_gripper_command=action_index == 0,
        )
        steps_done += 1
        seq += 1
        action_index += 1
        if args.execute:
            time.sleep(1.0 / args.control_hz)
        print("action complete. Press e for error report or c for the next action.")
    return steps_done, seq, False


def _run_policy_loop(args: argparse.Namespace) -> None:
    intervention_runtime = None
    if args.rtc:
        client = rtc_client_policy.RtcWebsocketClientPolicy(
            host=args.policy_host,
            port=args.policy_port,
            latency_k=args.rtc_latency_k,
            execution_horizon=args.actions_per_infer,
            max_guidance_weight=args.rtc_max_guidance_weight,
        )
    else:
        client = websocket_client_policy.WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
    rtc_loop = None
    steps_done = 0
    seq = 0
    stop_status = None
    stop_error = None
    try:
        if args.intervene:
            intervention_runtime = HgDaggerInterventionRuntime(args, control_script=__file__)
            intervention_runtime.start()

        if args.rtc:
            def fetch_rtc_chunk() -> np.ndarray:
                raw_state_for_rtc = _get_robot_state(args.robot_server, args.timeout)
                obs_for_rtc = _observation(raw_state_for_rtc, args.prompt, args.gripper_threshold)
                return _rtc_executable_chunk(args, raw_state_for_rtc, _policy_actions(client, obs_for_rtc))

            rtc_loop = RtcInferenceLoop(
                fetch_executable_chunk=fetch_rtc_chunk,
                latency_k=args.rtc_latency_k,
                inference_rate=args.rtc_inference_rate,
            )
            rtc_loop.start()

        while steps_done < args.max_steps:
            if intervention_runtime is not None:
                intervention = intervention_runtime.maybe_run_step(
                    _get_robot_state,
                    _execute_or_print,
                    steps_done,
                    seq,
                )
                steps_done, seq = intervention.steps_done, intervention.seq
                if intervention.handled or intervention.released:
                    if args.rtc and rtc_loop is not None:
                        rtc_loop.reset()
                        client.reset()
                        rtc_loop.start()
                    continue

            if args.rtc:
                assert rtc_loop is not None
                action = rtc_loop.pop_next_action()
                if action is None:
                    time.sleep(min(0.01, 1.0 / args.control_hz))
                    continue
                _next_rtc_action_record(args, action, seq)
                steps_done += 1
                seq += 1
                if args.execute:
                    time.sleep(1.0 / args.control_hz)
                continue

            raw_state = _get_robot_state(args.robot_server, args.timeout)
            obs = _observation(raw_state, args.prompt, args.gripper_threshold)
            actions = _policy_actions(client, obs)

            if args.debug:
                steps_done, seq, should_quit = _run_debug_chunk(args, actions, steps_done, seq)
                if should_quit:
                    stop_status = "interrupted"
                    return
                continue

            chunk_len = min(args.actions_per_infer, actions.shape[0], args.max_steps - steps_done)
            for action_index, action in enumerate(actions[:chunk_len]):
                if intervention_runtime is not None:
                    intervention = intervention_runtime.maybe_run_step(
                        _get_robot_state,
                        _execute_or_print,
                        steps_done,
                        seq,
                    )
                    steps_done, seq = intervention.steps_done, intervention.seq
                    if intervention.handled or intervention.released:
                        break
                _next_action_record(
                    args,
                    action,
                    seq,
                    include_gripper_command=action_index == 0,
                )
                steps_done += 1
                seq += 1
                if args.execute:
                    time.sleep(1.0 / args.control_hz)
        stop_status = "completed"
    except KeyboardInterrupt:
        stop_status = "interrupted"
        print("interrupted by operator")
    except Exception as exc:
        stop_status = "failed"
        stop_error = repr(exc)
        raise
    finally:
        if rtc_loop is not None:
            rtc_loop.stop()
        if intervention_runtime is not None:
            intervention_runtime.close(status=stop_status or "failed", error=stop_error)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_args(parser, default_action_space="abs_eef")
    parser.add_argument("--policy-host", default=None)
    parser.add_argument("--policy-port", type=int, default=None)
    parser.add_argument("--robot-server", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--control-hz", type=float, default=None)
    parser.add_argument("--actions-per-infer", type=int, default=None)
    parser.add_argument("--gripper-open-value", type=float, default=None)
    parser.add_argument("--gripper-close-value", type=float, default=None)
    parser.add_argument("--gripper-threshold", type=float, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--timeout", type=float, default=2.0)
    add_rtc_args(parser)
    add_hg_dagger_args(parser)
    args = apply_rollout_config(parser.parse_args())
    if args.list_spacemouse_devices:
        list_spacemouse_devices()
        raise SystemExit(0)

    if args.policy_host is None:
        args.policy_host = "localhost"
    if args.policy_port is None:
        args.policy_port = 8000
    if args.robot_server is None:
        args.robot_server = "http://127.0.0.1:8001"
    if args.prompt is None:
        args.prompt = DEFAULT_PROMPT
    if args.control_hz is None:
        args.control_hz = 10.0
    if args.actions_per_infer is None:
        args.actions_per_infer = 5
    if args.gripper_open_value is None:
        args.gripper_open_value = 100.0
    if args.gripper_close_value is None:
        args.gripper_close_value = 0.0
    if args.gripper_threshold is None:
        args.gripper_threshold = 75.0
    if args.timeout is None:
        args.timeout = 2.0
    apply_rtc_defaults(args)

    if args.control_hz <= 0:
        parser.error("--control-hz must be positive")
    if args.actions_per_infer <= 0:
        parser.error("--actions-per-infer must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    validate_rtc_args(parser, args)
    validate_hg_dagger_args(parser, args)
    if args.execute and args.max_steps is None:
        parser.error("--max-steps is required when --execute is set")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.max_steps is None:
        args.max_steps = args.actions_per_infer
        print(f"dry-run without --max-steps: running one chunk ({args.max_steps} actions)")
    return args


def main() -> None:
    args = _parse_args()
    _run_policy_loop(args)


if __name__ == "__main__":
    main()
