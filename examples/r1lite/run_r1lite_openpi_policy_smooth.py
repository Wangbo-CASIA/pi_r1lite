#!/usr/bin/env python3
"""Run an openpi R1Lite policy with action-chunk smoothing."""

import argparse
import base64
import dataclasses
import json
import sys
import termios
import threading
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
LEFT_JOINT_POS_SLICE = slice(13, 19)
LEFT_GRIPPER_INDEX = 25
RIGHT_JOINT_POS_SLICE = slice(39, 45)
RIGHT_GRIPPER_INDEX = 51
STATE_DIM = 53
ACTION_DIM = 14


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
    left_reference: np.ndarray
    right_reference: np.ndarray
    left_gripper: float
    right_gripper: float
    action: np.ndarray


@dataclasses.dataclass
class RightSoftCloseControlPanel:
    enabled_ui: bool = True
    default_enabled: bool = False
    enabled: bool = False
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _stop_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _window_name: str = "R1Lite right gripper soft-close"

    def __post_init__(self) -> None:
        self.enabled = bool(self.default_enabled)

    def start(self) -> None:
        if not self.enabled_ui:
            print(f"right soft-close mode: {'ON' if self.is_enabled() else 'OFF'}")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="right-soft-close-control-panel", daemon=True)
        self._thread.start()
        print(
            "right soft-close OpenCV control panel: click the window to toggle "
            f"(initial={'ON' if self.is_enabled() else 'OFF'})"
        )

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        try:
            cv2.destroyWindow(self._window_name)
        except cv2.error:
            pass

    def is_enabled(self) -> bool:
        with self._lock:
            return bool(self.enabled)

    def _toggle(self) -> None:
        with self._lock:
            self.enabled = not self.enabled
            enabled = self.enabled
        print(f"\nright soft-close mode: {'ON' if enabled else 'OFF'}", flush=True)

    def _run(self) -> None:
        try:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._window_name, 420, 180)
            cv2.setMouseCallback(self._window_name, self._on_mouse)
            while not self._stop_event.is_set():
                cv2.imshow(self._window_name, self._panel_image())
                key = cv2.waitKey(100) & 0xFF
                if key in (ord("g"), ord("G")):
                    self._toggle()
        except cv2.error as exc:
            print(f"right soft-close OpenCV control panel disabled: {exc}")
        finally:
            try:
                cv2.destroyWindow(self._window_name)
            except cv2.error:
                pass

    def _on_mouse(self, event: int, _x: int, _y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._toggle()

    def _panel_image(self) -> np.ndarray:
        enabled = self.is_enabled()
        image = np.full((180, 420, 3), 245, dtype=np.uint8)
        color = (60, 170, 60) if enabled else (60, 60, 220)
        label = "SOFT CLOSE: ON" if enabled else "SOFT CLOSE: OFF"
        cv2.rectangle(image, (25, 45), (395, 135), color, thickness=-1)
        cv2.rectangle(image, (25, 45), (395, 135), (30, 30, 30), thickness=2)
        cv2.putText(image, label, (55, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "Click window or press G to toggle", (55, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        return image


@dataclasses.dataclass
class ActionChunkSmoother:
    """Post-process policy chunks before sending them to the robot."""

    args: argparse.Namespace
    previous_action: np.ndarray | None = None
    left_gripper_command: float | None = None
    right_gripper_command: float | None = None

    def reset(self) -> None:
        self.previous_action = None
        self.left_gripper_command = None
        self.right_gripper_command = None

    def smooth_chunk(self, actions: np.ndarray, chunk_len: int) -> np.ndarray:
        chunk = np.asarray(actions[:chunk_len], dtype=np.float32).copy()
        if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM:
            raise ValueError(f"Expected action chunk with shape (n, {ACTION_DIM}), got {chunk.shape}")
        if chunk_len <= 0:
            return chunk

        if self.args.smooth_actions:
            chunk = self._smooth_joint_targets(chunk)

        if self.args.smooth_grippers:
            self._stabilize_grippers(chunk)

        self.previous_action = chunk[-1].copy()
        return chunk

    def _smooth_joint_targets(self, chunk: np.ndarray) -> np.ndarray:
        joint_indices = np.r_[0:6, 7:13]
        smoothed = chunk.copy()

        window = max(1, int(self.args.chunk_smooth_window))
        if window > 1 and chunk.shape[0] > 1:
            radius = window // 2
            padded = np.pad(smoothed[:, joint_indices], ((radius, radius), (0, 0)), mode="edge")
            averaged = np.empty_like(smoothed[:, joint_indices])
            for i in range(smoothed.shape[0]):
                averaged[i] = np.mean(padded[i : i + window], axis=0)
            smoothed[:, joint_indices] = averaged

        alpha = float(self.args.joint_ema_alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("--joint-ema-alpha must be in (0, 1]")
        last = None if self.previous_action is None else self.previous_action[joint_indices].copy()
        for i in range(smoothed.shape[0]):
            raw = smoothed[i, joint_indices].copy()
            if last is not None:
                raw = alpha * raw + (1.0 - alpha) * last
                max_step = float(self.args.joint_max_step)
                if max_step > 0.0:
                    raw = last + np.clip(raw - last, -max_step, max_step)
            smoothed[i, joint_indices] = raw
            last = raw

        return smoothed

    def _stabilize_grippers(self, chunk: np.ndarray) -> None:
        if self.left_gripper_command is None:
            self.left_gripper_command = float(self.args.gripper_open_value)
        if self.right_gripper_command is None:
            self.right_gripper_command = float(self.args.gripper_open_value)

        chunk[:, 6] = self._stable_gripper_value(chunk[:, 6], self.left_gripper_command)
        self.left_gripper_command = float(chunk[-1, 6])

        chunk[:, 13] = self._stable_gripper_value(chunk[:, 13], self.right_gripper_command)
        self.right_gripper_command = float(chunk[-1, 13])

    def _stable_gripper_value(self, values: np.ndarray, previous_command: float) -> float:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        open_votes = float(np.mean(values >= float(self.args.gripper_vote_threshold)))
        min_vote = float(self.args.gripper_min_vote_fraction)
        max_close_vote = 1.0 - min_vote

        if open_votes >= min_vote:
            return float(self.args.gripper_open_value)
        if open_votes <= max_close_vote:
            return float(self.args.gripper_close_value)
        return float(previous_command)


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

# TODO
def _crop_head_bottom_right_2_3(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return image[h // 3 :, w // 3 :]

def _arm_state(raw_state: dict[str, Any], side: str) -> dict[str, Any]:
    state = raw_state.get("state")
    if not isinstance(state, dict):
        raise ValueError("Robot state response is missing object field: state")
    arm = state.get(side)
    if not isinstance(arm, dict):
        raise ValueError(f"Robot state response is missing object field: state.{side}")
    return arm


def _joint_pos(raw_state: dict[str, Any], side: str) -> np.ndarray:
    joint_pos = _arm_state(raw_state, side).get("joint_pos")
    joint = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
    if joint.shape[0] < 6:
        raise ValueError(f"Expected state.{side}.joint_pos to contain at least 6 values, got {joint.shape[0]}")
    return joint[:6].copy()


def _gripper_pose(raw_state: dict[str, Any], side: str) -> float:
    gripper_pose = _arm_state(raw_state, side).get("gripper_pose")
    gripper = np.asarray(gripper_pose, dtype=np.float32).reshape(-1)
    if gripper.shape[0] < 1:
        raise ValueError(f"Expected state.{side}.gripper_pose to contain at least 1 value, got {gripper.shape[0]}")
    return float(gripper[0])


# def _binary_gripper(value: float, threshold: float) -> float:
#     return 1.0 if float(value) > threshold else 0.0

def _binary_gripper(value: float, threshold: float) -> float:
    return float(value)


def _state_vector(
    raw_state: dict[str, Any], left_gripper_threshold: float, right_gripper_threshold: float
) -> np.ndarray:
    state = np.zeros((STATE_DIM,), dtype=np.float32)
    state[LEFT_JOINT_POS_SLICE] = _joint_pos(raw_state, "left")
    state[LEFT_GRIPPER_INDEX] = _binary_gripper(_gripper_pose(raw_state, "left"), left_gripper_threshold)
    state[RIGHT_JOINT_POS_SLICE] = _joint_pos(raw_state, "right")
    state[RIGHT_GRIPPER_INDEX] = _binary_gripper(_gripper_pose(raw_state, "right"), right_gripper_threshold)
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


def _observation(
    raw_state: dict[str, Any],
    prompt: str,
    left_gripper_threshold: float,
    right_gripper_threshold: float,
) -> dict[str, Any]:
    _validate_freshness(raw_state)
    images = raw_state.get("images")
    if not isinstance(images, dict):
        raise ValueError("Robot state response is missing object field: images")
    return {
        "images": {
            #head": _decode_image(images.get("head"), "head"),
            "head": _crop_head_bottom_right_2_3(_decode_image(images.get("head"), "head")),
            "left_wrist": _decode_image(images.get("left_wrist"), "left_wrist"),
            "right_wrist": _decode_image(images.get("right_wrist"), "right_wrist"),
        },
        "state": _state_vector(raw_state, left_gripper_threshold, right_gripper_threshold),
        "prompt": prompt,
    }


def _policy_actions(client: websocket_client_policy.WebsocketClientPolicy, observation: dict[str, Any]) -> tuple[np.ndarray, float]:
    # 新增推理延迟计时
    t0 = time.perf_counter()
    result = client.infer(observation)
    infer_latency = time.perf_counter() - t0

    if "actions" not in result:
        raise RuntimeError(f"Policy response is missing 'actions'. Keys: {sorted(result)}")
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise RuntimeError(f"Expected policy actions with shape (horizon, {ACTION_DIM}), got {actions.shape}")
    if actions.shape[0] < 1:
        raise RuntimeError("Policy returned an empty action chunk")
    return actions, infer_latency


def _gripper_command(binary_value: float, open_value: float, close_value: float) -> float:
    return float(open_value if float(binary_value) >= 50.0 else close_value)


def _gripper_command1(binary_value: float, open_value: float, close_value: float) -> float:
    return _gripper_command(binary_value, open_value, close_value)


def _gripper_command2(value: float, open_value: float, close_value: float) -> float:
    return _gripper_command(value, open_value, close_value)
# def _gripper_command(binary_value: float, open_value: float, close_value: float) -> float:
#     return float(binary_value)


def _apply_right_soft_close_rule(
    seq: int,
    command: float,
    close_value: float,
    soft_close_start_step: int | None,
    soft_close_end_step: int | None,
    soft_close_value: float,
    force_enabled: bool = False,  # noqa: FBT001, FBT002
) -> float:
    if not force_enabled and (soft_close_start_step is None or seq < soft_close_start_step):
        return float(command)
    if not force_enabled and soft_close_end_step is not None and seq >= soft_close_end_step:
        return float(command)
    if np.isclose(float(command), float(close_value)):
        return float(soft_close_value)
    return float(command)


def _right_soft_close_keyboard_enabled(args: argparse.Namespace) -> bool:
    switch = getattr(args, "right_soft_close_keyboard_switch", None)
    return bool(switch is not None and switch.is_enabled())


def _payload_from_action(
    action: np.ndarray,
    seq: int,
    left_reference: np.ndarray,
    right_reference: np.ndarray,
    gripper_open_value: float,
    gripper_close_value: float,
    right_soft_close_start_step: int | None,
    right_soft_close_end_step: int | None,
    right_soft_close_value: float,
    right_soft_close_keyboard_enabled: bool,
    include_gripper_command: bool,  # noqa: FBT001
) -> tuple[dict[str, Any], StepRecord]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != ACTION_DIM:
        raise ValueError(f"Expected one action with {ACTION_DIM} values, got {action.shape[0]}")
    # 如果是相对的joint
    # left_target = left_reference + action[:6]
    # right_target = right_reference + action[7:13]
    # 如果是绝对的joint
    left_target = action[:6]
    right_target = action[7:13]

    left_gripper = _gripper_command1(action[6], gripper_open_value, gripper_close_value)
    right_gripper = _gripper_command2(action[13], gripper_open_value, gripper_close_value)
    right_gripper = _apply_right_soft_close_rule(
        seq,
        right_gripper,
        gripper_close_value,
        right_soft_close_start_step,
        right_soft_close_end_step,
        right_soft_close_value,
        force_enabled=right_soft_close_keyboard_enabled,
    )
    
    left_payload = {
        "joint_target": [float(value) for value in left_target],
        "preset": "free_space",
    }
    right_payload = {
        "joint_target": [float(value) for value in right_target],
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
        left_reference=left_reference,
        right_reference=right_reference,
        left_gripper=left_gripper,
        right_gripper=right_gripper,
        action=action.copy(),
    )


def _format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value): .5f}" for value in values.reshape(-1)) + "]"


def _print_error_report(record: StepRecord, raw_state: dict[str, Any]) -> None:
    for side, target in (("left", record.left_target), ("right", record.right_target)):
        current = _joint_pos(raw_state, side)
        error = target - current
        print(f"{Color.DIM}{side} joints, seq={record.seq}, error=target-current{Color.RESET}")
        print(f"  {Color.TARGET}target : {_format_vector(target)}{Color.RESET}")
        print(f"  {Color.CURRENT}current: {_format_vector(current)}{Color.RESET}")
        print(f"  {Color.ERROR}error  : {_format_vector(error)}{Color.RESET}")


def _read_debug_key() -> str:
    prompt = "debug command [c=execute next action, e=print joint error, q=quit]: "
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
    left_reference: np.ndarray,
    right_reference: np.ndarray,
    include_gripper_command: bool,  # noqa: FBT001
) -> StepRecord:
    payload, record = _payload_from_action(
        action,
        seq,
        left_reference,
        right_reference,
        args.gripper_open_value,
        args.gripper_close_value,
        args.right_soft_close_start_step,
        args.right_soft_close_end_step,
        args.right_soft_close_value,
        _right_soft_close_keyboard_enabled(args),
        include_gripper_command,
    )
    _execute_or_print(args, payload)
    return record


def _rtc_executable_chunk(args: argparse.Namespace, raw_state: dict[str, Any], actions: np.ndarray) -> np.ndarray:
    left_reference = _joint_pos(raw_state, "left")
    right_reference = _joint_pos(raw_state, "right")
    chunk_len = min(args.actions_per_infer, actions.shape[0])
    rows = []
    for action in actions[:chunk_len]:
        row = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if row.shape[0] != ACTION_DIM:
            raise ValueError(f"Expected one action with {ACTION_DIM} values, got {row.shape[0]}")
        row[:6] = left_reference + row[:6]
        row[7:13] = right_reference + row[7:13]
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def _next_rtc_action_record(args: argparse.Namespace, action: np.ndarray, seq: int) -> StepRecord:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != ACTION_DIM:
        raise ValueError(f"Expected one RTC action with {ACTION_DIM} values, got {action.shape[0]}")
    left_target = action[:6]
    right_target = action[7:13]
    left_gripper = _gripper_command(action[6], args.gripper_open_value, args.gripper_close_value)
    right_gripper = _gripper_command(action[13], args.gripper_open_value, args.gripper_close_value)
    right_gripper = _apply_right_soft_close_rule(
        seq,
        right_gripper,
        args.gripper_close_value,
        args.right_soft_close_start_step,
        args.right_soft_close_end_step,
        args.right_soft_close_value,
        force_enabled=_right_soft_close_keyboard_enabled(args),
    )
    payload = {
        "mode": "ee_pose_servo",
        "owner": "policy",
        "seq": seq,
        "left": {
            "joint_target": [float(value) for value in left_target],
            "gripper": left_gripper,
            "preset": "free_space",
        },
        "right": {
            "joint_target": [float(value) for value in right_target],
            "gripper": right_gripper,
            "preset": "free_space",
        },
    }
    record = StepRecord(
        seq=seq,
        left_target=left_target,
        right_target=right_target,
        left_reference=left_target.copy(),
        right_reference=right_target.copy(),
        left_gripper=left_gripper,
        right_gripper=right_gripper,
        action=action.copy(),
    )
    _execute_or_print(args, payload)
    return record


def _run_debug_chunk(
    args: argparse.Namespace,
    actions: np.ndarray,
    smoother: ActionChunkSmoother,
    steps_done: int,
    seq: int,
) -> tuple[int, int, bool]:
    raw_state = _get_robot_state(args.robot_server, args.timeout)
    left_reference = _joint_pos(raw_state, "left")
    right_reference = _joint_pos(raw_state, "right")
    chunk_len = min(args.actions_per_infer, actions.shape[0], args.max_steps - steps_done)
    chunk = smoother.smooth_chunk(actions, chunk_len)
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
            left_reference,
            right_reference,
            include_gripper_command=True,
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
    smoother = ActionChunkSmoother(args)
    soft_close_switch = RightSoftCloseControlPanel(
        enabled_ui=args.right_soft_close_ui,
        default_enabled=args.right_soft_close_keyboard_default_on,
    )
    args.right_soft_close_keyboard_switch = soft_close_switch
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
        soft_close_switch.start()
        if args.intervene:
            intervention_runtime = HgDaggerInterventionRuntime(args, control_script=__file__)
            intervention_runtime.start()

        if args.rtc:
            def fetch_rtc_chunk() -> np.ndarray:
                raw_state_for_rtc = _get_robot_state(args.robot_server, args.timeout)
                obs_for_rtc = _observation(
                    raw_state_for_rtc,
                    args.prompt,
                    args.left_gripper_threshold,
                    args.right_gripper_threshold,
                )
                # 调用推理并拿到延迟
                actions, infer_latency = _policy_actions(client, obs_for_rtc)
                print(f"[RTC-Infer Latency] {infer_latency * 1000:.2f} ms")
                executable = _rtc_executable_chunk(args, raw_state_for_rtc, actions)
                return smoother.smooth_chunk(executable, executable.shape[0])

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
                        smoother.reset()
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
            obs = _observation(
                raw_state,
                args.prompt,
                args.left_gripper_threshold,
                args.right_gripper_threshold,
            )
            # 普通模式，获取动作+推理时延
            actions, infer_latency = _policy_actions(client, obs)
            print(f"[Policy-Infer Latency] seq:{seq}, latency={infer_latency * 1000:.2f} ms")

            if args.debug:
                steps_done, seq, should_quit = _run_debug_chunk(args, actions, smoother, steps_done, seq)
                if should_quit:
                    stop_status = "interrupted"
                    return
                continue

            left_reference = _joint_pos(raw_state, "left")
            right_reference = _joint_pos(raw_state, "right")
            chunk_len = min(args.actions_per_infer, actions.shape[0], args.max_steps - steps_done)

            chunk = smoother.smooth_chunk(actions, chunk_len)
            for action_index, action in enumerate(chunk):
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
                    left_reference,
                    right_reference,
                    include_gripper_command=True,
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
        soft_close_switch.stop()
        if rtc_loop is not None:
            rtc_loop.stop()
        if intervention_runtime is not None:
            intervention_runtime.close(status=stop_status or "failed", error=stop_error)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_args(parser,default_experiment=None, default_action_space="abs_joint")
    parser.add_argument("--policy-host", default=None)
    parser.add_argument("--policy-port", type=int, default=None)
    parser.add_argument("--robot-server", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--control-hz", type=float, default=None)
    parser.add_argument("--actions-per-infer", type=int, default=None)
    parser.add_argument("--gripper-open-value", type=float, default=None)
    parser.add_argument("--gripper-close-value", type=float, default=None)
    parser.add_argument("--gripper-threshold", type=float, default=None)
    parser.add_argument("--left-gripper-threshold", type=float, default=None)
    parser.add_argument("--right-gripper-threshold", type=float, default=None)
    parser.add_argument("--no-action-smoothing", dest="smooth_actions", action="store_false")
    parser.set_defaults(smooth_actions=True)
    parser.add_argument("--chunk-smooth-window", type=int, default=3)
    parser.add_argument("--joint-ema-alpha", type=float, default=0.65)
    parser.add_argument("--joint-max-step", type=float, default=0.08)
    parser.add_argument("--no-gripper-smoothing", dest="smooth_grippers", action="store_false")
    parser.set_defaults(smooth_grippers=True)
    parser.add_argument("--gripper-vote-threshold", type=float, default=50.0)
    parser.add_argument("--gripper-min-vote-fraction", type=float, default=0.6)
    parser.add_argument("--right-soft-close-start-step", type=int, default=None)
    parser.add_argument("--right-soft-close-end-step", type=int, default=None)
    parser.add_argument("--right-soft-close-value", type=float, default=90.0)
    parser.add_argument("--no-right-soft-close-ui", dest="right_soft_close_ui", action="store_false")
    parser.set_defaults(right_soft_close_ui=True)
    parser.add_argument("--right-soft-close-default-on", dest="right_soft_close_keyboard_default_on", action="store_true")
    parser.add_argument("--right-soft-close-keyboard-default-on", action="store_true")
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
    if args.left_gripper_threshold is None:
        args.left_gripper_threshold = args.gripper_threshold
    if args.right_gripper_threshold is None:
        args.right_gripper_threshold = args.gripper_threshold
    if args.timeout is None:
        args.timeout = 2.0
    apply_rtc_defaults(args)

    if args.control_hz <= 0:
        parser.error("--control-hz must be positive")
    if args.actions_per_infer <= 0:
        parser.error("--actions-per-infer must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.chunk_smooth_window <= 0:
        parser.error("--chunk-smooth-window must be positive")
    if args.chunk_smooth_window % 2 == 0:
        parser.error("--chunk-smooth-window must be odd")
    if not 0.0 < args.joint_ema_alpha <= 1.0:
        parser.error("--joint-ema-alpha must be in (0, 1]")
    if args.joint_max_step < 0.0:
        parser.error("--joint-max-step must be non-negative")
    if not 0.0 <= args.gripper_vote_threshold <= 100.0:
        parser.error("--gripper-vote-threshold must be in [0, 100]")
    if not 0.5 <= args.gripper_min_vote_fraction <= 1.0:
        parser.error("--gripper-min-vote-fraction must be in [0.5, 1.0]")
    if args.right_soft_close_start_step is not None and args.right_soft_close_start_step < 0:
        parser.error("--right-soft-close-start-step must be non-negative")
    if args.right_soft_close_end_step is not None and args.right_soft_close_end_step < 0:
        parser.error("--right-soft-close-end-step must be non-negative")
    if (
        args.right_soft_close_start_step is not None
        and args.right_soft_close_end_step is not None
        and args.right_soft_close_end_step <= args.right_soft_close_start_step
    ):
        parser.error("--right-soft-close-end-step must be greater than --right-soft-close-start-step")
    if not 0.0 <= args.right_soft_close_value <= args.gripper_open_value:
        parser.error("--right-soft-close-value must be between 0 and --gripper-open-value")
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
