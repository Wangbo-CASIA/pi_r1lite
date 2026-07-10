#!/usr/bin/env python3
"""Shared HG-Dagger helpers for R1Lite OpenPI policy runners."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import dataclasses
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import posixpath
import queue
import select
import sys
import termios
import threading
import time
import tty
from typing import Any
import urllib.error
import urllib.request

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

DEFAULT_ROBOT_RECORD_OUTPUT_DIR = "/home/r1lite/GalaxeaDataset/hgdagger"
DEFAULT_RECORD_STOP_TIMEOUT = 60.0
DEFAULT_RECORD_TOPICS = [
    "/hdas/camera_head/right_raw/image_raw_color/compressed",
    "/hdas/camera_wrist_left/color/image_raw/compressed",
    "/hdas/camera_wrist_right/color/image_raw/compressed",
    "/hdas/feedback_arm_left",
    "/hdas/feedback_arm_right",
    "/hdas/feedback_torso",
    "/hdas/feedback_gripper_left",
    "/hdas/feedback_gripper_right",
]


def _record_session_dir_name() -> str:
    now_ns = time.time_ns()
    seconds = now_ns // 1_000_000_000
    milliseconds = (now_ns // 1_000_000) % 1_000
    return time.strftime("session_%Y%m%d_%H%M%S_", time.localtime(seconds)) + f"{milliseconds:03d}"


def _record_session_output_dir(parent: str, session_dir_name: str) -> str:
    if not parent:
        raise ValueError("record output directory must be non-empty")
    parent_text = str(parent).rstrip("/") or "/"
    return posixpath.join(parent_text, session_dir_name)


@dataclasses.dataclass(frozen=True)
class InterventionStepResult:
    handled: bool
    released: bool
    steps_done: int
    seq: int


@dataclasses.dataclass
class OperatorEventItem:
    payload: dict[str, Any]
    requires_ack: bool
    _done: threading.Event = dataclasses.field(default_factory=threading.Event)
    _success: bool = False
    _message: str = "pending"
    _status: int = 200

    def wait(self) -> tuple[bool, str, int]:
        if not self.requires_ack:
            return True, "queued", 200
        self._done.wait()
        return self._success, self._message, self._status

    def resolve(self, *, success: bool, message: str, status: int = 200) -> None:
        if not self.requires_ack:
            return
        self._success = bool(success)
        self._message = str(message)
        self._status = int(status)
        self._done.set()


@dataclasses.dataclass
class OperatorControlEvent:
    action: str
    item: OperatorEventItem


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


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    try:
        handler.wfile.write(body)
    except BrokenPipeError:
        print(f"[hg-dagger] operator event response dropped: status={status} payload={payload}", flush=True)


def _require_operator_field(payload: dict[str, Any], name: str, expected_type: type) -> Any:
    if name not in payload:
        raise ValueError(f"missing field: {name}")
    value = payload[name]
    if not isinstance(value, expected_type):
        raise ValueError(f"{name} must be {expected_type.__name__}")
    return value


def _validate_operator_event(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("operator event must be a JSON object")
    seq = _require_operator_field(payload, "seq", int)
    if seq < 0:
        raise ValueError("seq must be non-negative")
    stamp = payload.get("stamp")
    if not isinstance(stamp, int | float):
        raise ValueError("stamp must be number")
    for name in ("source", "topic", "button", "event", "action"):
        value = _require_operator_field(payload, name, str)
        if not value:
            raise ValueError(f"{name} must be non-empty")
    pressed = _require_operator_field(payload, "pressed", bool)
    event = payload["event"]
    action = payload["action"]
    if event == "takeover":
        if action not in (
            "takeover_start",
            "discard_without_save_and_end_takeover",
            "save_and_end_takeover",
        ):
            raise ValueError(f"invalid takeover action: {action}")
        active = payload.get("active")
        if active is None:
            raise ValueError("takeover event requires active")
        if action == "takeover_start" and active is not True:
            raise ValueError("takeover_start requires active=true")
        if action in ("discard_without_save_and_end_takeover", "save_and_end_takeover") and active is not False:
            raise ValueError(f"{action} requires active=false")
        if pressed is not True:
            raise ValueError("takeover event must be sent on button press")
        if payload.get("side") is not None:
            raise ValueError("takeover event must not include side")
    elif event == "trajectory":
        if action != "discard_trajectory":
            raise ValueError(f"invalid trajectory action for HG-Dagger: {action}")
        if pressed is not True:
            raise ValueError("trajectory event must be sent on button press")
        if payload.get("side") is not None or payload.get("active") is not None:
            raise ValueError("trajectory event must not include side or active")
    elif event == "gripper":
        if action not in (
            "left_gripper_open",
            "left_gripper_close",
            "right_gripper_open",
            "right_gripper_close",
        ):
            raise ValueError(f"invalid gripper action: {action}")
        side = payload.get("side")
        if side not in ("left", "right"):
            raise ValueError("gripper event requires side=left or side=right")
    else:
        raise ValueError(f"unsupported operator event: {event}")
    return dict(payload)


def _operator_event_requires_ack(event: dict[str, Any]) -> bool:
    if event["event"] == "takeover":
        return event["action"] in (
            "discard_without_save_and_end_takeover",
            "save_and_end_takeover",
        )
    return event["event"] == "trajectory"


def _operator_control_action(event: dict[str, Any]) -> str | None:
    if event["event"] == "takeover":
        return str(event["action"])
    if event["event"] == "trajectory":
        return str(event["action"])
    return None


def _estimate_idle_bias(expert: Any, duration_sec: float) -> np.ndarray:
    if duration_sec <= 0.0:
        action, _ = expert.get_action()
        return np.zeros_like(np.asarray(action, dtype=np.float64))

    deadline = time.time() + duration_sec
    samples = []
    while time.time() < deadline:
        action, _ = expert.get_action()
        samples.append(np.asarray(action, dtype=np.float64))
        time.sleep(0.01)
    if not samples:
        action, _ = expert.get_action()
        return np.zeros_like(np.asarray(action, dtype=np.float64))
    return np.median(np.stack(samples, axis=0), axis=0)


def _apply_deadzone(action: np.ndarray, trans_deadzone: float, rot_deadzone: float) -> np.ndarray:
    filtered = np.asarray(action, dtype=np.float64).copy()
    if filtered.shape[0] >= 3:
        filtered[:3][np.abs(filtered[:3]) < trans_deadzone] = 0.0
    if filtered.shape[0] >= 6:
        filtered[3:6][np.abs(filtered[3:6]) < rot_deadzone] = 0.0
    if filtered.shape[0] >= 9:
        filtered[6:9][np.abs(filtered[6:9]) < trans_deadzone] = 0.0
    if filtered.shape[0] >= 12:
        filtered[9:12][np.abs(filtered[9:12]) < rot_deadzone] = 0.0
    return filtered


def _spacemouse_state_to_action(state: Any) -> list[float]:
    return [
        -float(state.y),
        float(state.x),
        float(state.z),
        -float(state.roll),
        -float(state.pitch),
        -float(state.yaw),
    ]


def _decode_hid_path(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def list_spacemouse_devices() -> None:
    try:
        import pyspacemouse
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyspacemouse is required to list SpaceMouse devices") from exc

    from easyhid import Enumeration

    specs = pyspacemouse.get_device_specs()
    hid = Enumeration()
    print("SpaceMouse-capable HID devices:")
    matched = 0
    seen_paths: set[str] = set()
    for dev in hid.find():
        for name, spec in specs.items():
            if dev.vendor_id == spec.vendor_id and dev.product_id == spec.product_id:
                path = _decode_hid_path(dev.path)
                if path in seen_paths:
                    break
                seen_paths.add(path)
                matched += 1
                print(
                    f"  path={path} "
                    f"name={name} vid=0x{dev.vendor_id:04x} pid=0x{dev.product_id:04x} "
                    f"product={dev.product_string!r} manufacturer={dev.manufacturer_string!r}"
                )
                break
    if matched == 0:
        print("  none")


def _matching_spacemouse_paths(pyspacemouse: Any) -> list[str]:
    from easyhid import Enumeration

    specs = pyspacemouse.get_device_specs()
    hid = Enumeration()
    matches = []
    seen_paths: set[str] = set()
    selected_spec = None
    for dev in hid.find():
        for name, spec in specs.items():
            if dev.vendor_id == spec.vendor_id and dev.product_id == spec.product_id:
                if selected_spec is None:
                    selected_spec = (name, spec.vendor_id, spec.product_id)
                if (name, spec.vendor_id, spec.product_id) == selected_spec:
                    path = _decode_hid_path(dev.path)
                    if path not in seen_paths:
                        seen_paths.add(path)
                        matches.append(path)
                break
    return matches


class PySpaceMouseExpert:
    """Small OpenPI-local wrapper around pyspacemouse 2.x."""

    def __init__(self, *, dual: bool, left_path: str | None = None, right_path: str | None = None):
        try:
            import pyspacemouse
        except ModuleNotFoundError as exc:
            raise RuntimeError("pyspacemouse is required for --intervene; install it in the openpi uv env") from exc

        self.pyspacemouse = pyspacemouse
        self.devices = []
        if dual:
            if left_path and right_path:
                if Path(left_path).resolve() == Path(right_path).resolve():
                    raise RuntimeError("left and right SpaceMouse paths must be different")
                self.devices.append(pyspacemouse.open_by_path(left_path, nonblocking=True))
                self.devices.append(pyspacemouse.open_by_path(right_path, nonblocking=True))
            else:
                paths = _matching_spacemouse_paths(pyspacemouse)
                if len(paths) < 2:
                    raise RuntimeError(
                        f"dual SpaceMouse intervention requires two HID devices of the same supported type, found {len(paths)}: {paths}"
                    )
                print(f"[hg-dagger] auto-selected SpaceMouse paths: left={paths[0]} right={paths[1]}")
                self.devices.append(pyspacemouse.open_by_path(paths[0], nonblocking=True))
                self.devices.append(pyspacemouse.open_by_path(paths[1], nonblocking=True))
        else:
            path = left_path or right_path
            if path:
                self.devices.append(pyspacemouse.open_by_path(path, nonblocking=True))
            else:
                self.devices.append(pyspacemouse.open(nonblocking=True))
        self._lock = threading.Lock()
        self._running = True
        self._latest_actions = [np.zeros(6, dtype=np.float64) for _ in self.devices]
        self._latest_buttons: list[list[Any]] = [[] for _ in self.devices]
        self._reader = threading.Thread(target=self._read_loop, name="spacemouse-reader", daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        while self._running:
            actions = []
            buttons = []
            for device in self.devices:
                state = device.read()
                actions.append(np.asarray(_spacemouse_state_to_action(state), dtype=np.float64))
                buttons.append(list(state.buttons))
            with self._lock:
                self._latest_actions = actions
                self._latest_buttons = buttons
            time.sleep(0.001)

    def get_action(self) -> tuple[np.ndarray, list[Any]]:
        groups = self.get_action_groups()
        actions: list[float] = []
        buttons: list[Any] = []
        for action_group, button_group in groups:
            actions.extend(action_group.tolist())
            buttons.extend(button_group)
        return np.asarray(actions, dtype=np.float64), buttons

    def get_action_groups(self) -> list[tuple[np.ndarray, list[Any]]]:
        with self._lock:
            return [
                (action.copy(), list(buttons))
                for action, buttons in zip(self._latest_actions, self._latest_buttons, strict=True)
            ]

    def close(self) -> None:
        self._running = False
        self._reader.join(timeout=1.0)
        for device in self.devices:
            device.close()


def _arm_state(raw_state: dict[str, Any], side: str) -> dict[str, Any]:
    state = raw_state.get("state")
    if not isinstance(state, dict):
        raise ValueError("Robot state response is missing object field: state")
    arm = state.get(side)
    if not isinstance(arm, dict):
        raise ValueError(f"Robot state response is missing object field: state.{side}")
    return arm


def _tcp_pose(raw_state: dict[str, Any], side: str) -> np.ndarray:
    pose = np.asarray(_arm_state(raw_state, side).get("tcp_pose"), dtype=np.float64).reshape(-1)
    if pose.shape[0] < 7:
        raise ValueError(f"Expected state.{side}.tcp_pose to contain at least 7 values, got {pose.shape[0]}")
    pose = pose[:7].copy()
    quat_norm = float(np.linalg.norm(pose[3:7]))
    if not np.isfinite(quat_norm) or quat_norm <= 0.0:
        raise ValueError(f"state.{side}.tcp_pose has invalid quaternion norm: {quat_norm}")
    pose[3:7] /= quat_norm
    return pose


def _pose_target_from_action(
    tcp_pose: np.ndarray, action: np.ndarray, xyz_scale: float, rot_scale: float
) -> np.ndarray:
    pose = np.asarray(tcp_pose, dtype=np.float64).copy()
    delta = np.asarray(action[:6], dtype=np.float64)
    pose[:3] = pose[:3] + delta[:3] * xyz_scale
    pose[3:7] = (Rotation.from_euler("xyz", delta[3:6] * rot_scale) * Rotation.from_quat(pose[3:7])).as_quat()
    return pose


def _single_arm_gripper(buttons: list[Any]) -> float | None:
    if len(buttons) >= 1 and buttons[0]:
        return 0.0
    if len(buttons) >= 2 and buttons[-1]:
        return 100.0
    return None


def _dual_arm_grippers(buttons: list[Any]) -> tuple[float | None, float | None]:
    if len(buttons) != 4:
        return None, None
    left = 0.0 if buttons[0] else 100.0 if buttons[1] else None
    right = 0.0 if buttons[2] else 100.0 if buttons[3] else None
    return left, right


def _joint_pos(raw_state: dict[str, Any], side: str) -> np.ndarray:
    joint = np.asarray(_arm_state(raw_state, side).get("joint_pos"), dtype=np.float64).reshape(-1)
    if joint.shape[0] < 6:
        raise ValueError(f"Expected state.{side}.joint_pos to contain at least 6 values, got {joint.shape[0]}")
    return joint[:6].copy()


def _gripper_pose(raw_state: dict[str, Any], side: str) -> float:
    gripper = np.asarray(_arm_state(raw_state, side).get("gripper_pose"), dtype=np.float64).reshape(-1)
    if gripper.shape[0] < 1:
        raise ValueError(f"Expected state.{side}.gripper_pose to contain at least 1 value, got {gripper.shape[0]}")
    return float(gripper[0])


def _hold_current_joint_target(
    args: argparse.Namespace,
    raw_state: dict[str, Any],
    execute_payload: Callable[[argparse.Namespace, dict[str, Any]], None],
    seq: int,
) -> int:
    payload = {
        "mode": "ee_pose_servo",
        "owner": "policy",
        "seq": seq,
        "left": {
            "joint_target": [float(value) for value in _joint_pos(raw_state, "left")],
            "gripper": _gripper_pose(raw_state, "left"),
            "preset": "free_space",
        },
        "right": {
            "joint_target": [float(value) for value in _joint_pos(raw_state, "right")],
            "gripper": _gripper_pose(raw_state, "right"),
            "preset": "free_space",
        },
    }
    execute_payload(args, payload)
    time.sleep(1.0 / args.control_hz)
    return seq + 1


def _max_joint_error(raw_state: dict[str, Any], target_raw_state: dict[str, Any]) -> float:
    left_error = np.max(np.abs(_joint_pos(raw_state, "left") - _joint_pos(target_raw_state, "left")))
    right_error = np.max(np.abs(_joint_pos(raw_state, "right") - _joint_pos(target_raw_state, "right")))
    return float(max(left_error, right_error))


def _wait_until_robot_reaches_state(
    args: argparse.Namespace,
    get_state: Callable[[str, float], dict[str, Any]],
    target_raw_state: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + args.tabletop_restore_timeout_sec
    latest_state = get_state(args.robot_server, args.timeout)
    latest_error = _max_joint_error(latest_state, target_raw_state)
    while latest_error > args.tabletop_restore_joint_tolerance:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"{label} did not reach target within {args.tabletop_restore_timeout_sec:.3f}s: "
                f"max_joint_error={latest_error:.6f}, tolerance={args.tabletop_restore_joint_tolerance:.6f}"
            )
        time.sleep(1.0 / args.control_hz)
        latest_state = get_state(args.robot_server, args.timeout)
        latest_error = _max_joint_error(latest_state, target_raw_state)
    print(f"[hg-dagger] {label} reached target: max_joint_error={latest_error:.6f}", flush=True)
    return latest_state


def _poll_terminal_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    if key in ("\r", "\n"):
        return "enter"
    if key == "\x1b":
        return "esc"
    if key:
        return key.lower()
    return None


def _read_save_or_discard() -> str:
    print("Save this intervention rosbag? ENTER=save, ESC=discard and retry from takeover start", flush=True)
    if not sys.stdin.isatty():
        line = sys.stdin.readline().strip().lower()
        if line in ("", "enter", "save", "s"):
            return "enter"
        if line in ("esc", "discard", "d"):
            return "esc"
        raise RuntimeError(f"unsupported save/discard input: {line!r}")
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            key = sys.stdin.read(1)
            if key in ("\r", "\n"):
                print("enter")
                return "enter"
            if key == "\x1b":
                print("esc")
                return "esc"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class RecordingClient:
    def __init__(self, server_url: str, timeout: float, stop_timeout: float):
        self.server_url = server_url
        self.timeout = timeout
        self.stop_timeout = stop_timeout
        self.started = False
        self.episode_stem: str | None = None

    def start(self, args: argparse.Namespace, control_script: str, output_dir: str) -> dict[str, Any]:
        payload = {
            "output_dir": output_dir,
            "control_script": control_script,
            "policy_host": args.policy_host,
            "policy_port": args.policy_port,
            "prompt": args.prompt,
            "max_steps": args.max_steps,
            "control_hz": args.control_hz,
            "actions_per_infer": args.actions_per_infer,
            "record_topics": args.record_topics,
        }
        result = _request_json("POST", _url(self.server_url, "/record/start"), self.timeout, payload)
        self.started = True
        self.episode_stem = str(result.get("episode_stem", ""))
        print(f"recording started: {result}")
        return result

    def mark(
        self,
        event: str,
        *,
        rollout_step: int,
        seq: int,
        source: str,
        device: str,
        arm: str,
        end_reason: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event": event,
            "rollout_step": rollout_step,
            "seq": seq,
            "source": source,
            "device": device,
            "arm": arm,
        }
        if end_reason is not None:
            payload["end_reason"] = end_reason
        result = _request_json("POST", _url(self.server_url, "/record/mark"), self.timeout, payload)
        print(f"record marker: {event} step={rollout_step} seq={seq}")
        return result

    def stop(self, status: str, error: str | None = None) -> dict[str, Any] | None:
        if not self.started:
            return None
        payload: dict[str, Any] = {"status": status}
        if error is not None:
            payload["error"] = error
        result = _request_json(
            "POST", _url(self.server_url, "/record/stop"), max(self.timeout, self.stop_timeout), payload
        )
        self.started = False
        print(f"recording stopped: {result}")
        return result


class InterventionRecordingSession:
    def __init__(self, args: argparse.Namespace, control_script: str):
        self.args = args
        self.control_script = control_script
        self.session_dir_name = _record_session_dir_name()
        self.record_output_dir = _record_session_output_dir(args.record_output_dir, self.session_dir_name)
        self.client = RecordingClient(args.robot_server, args.timeout, args.record_stop_timeout)
        self.active_source: str | None = None
        self.active_arm: str = "dual"
        self.segment_index = 0
        print(f"[hg-dagger] rollout recording session output_dir={self.record_output_dir}", flush=True)

    @property
    def active(self) -> bool:
        return self.client.started

    def start(self, *, source: str, rollout_step: int, seq: int, arm: str) -> None:
        if self.active:
            raise RuntimeError("cannot start a new intervention recording while another recording is active")
        self.active_source = source
        self.active_arm = arm
        self.segment_index += 1
        result = self.client.start(self.args, control_script=self.control_script, output_dir=self.record_output_dir)
        print(
            f"[hg-dagger] intervention recording segment {self.segment_index} started: "
            f"source={source} arm={arm} result={result}",
            flush=True,
        )
        self.client.mark(
            "intervention_start",
            rollout_step=rollout_step,
            seq=seq,
            source=source,
            device=source,
            arm=arm,
        )

    def save(self, *, rollout_step: int, seq: int, end_reason: str) -> None:
        if not self.active:
            raise RuntimeError("cannot save intervention recording because no recording is active")
        source = self.active_source or "unknown"
        arm = self.active_arm
        self.client.mark(
            "intervention_end",
            rollout_step=rollout_step,
            seq=seq,
            source=source,
            device=source,
            arm=arm,
            end_reason=end_reason,
        )
        self.client.stop("completed")
        self.active_source = None
        self.active_arm = "dual"

    def discard(self, *, rollout_step: int, seq: int, end_reason: str) -> None:
        if not self.active:
            raise RuntimeError("cannot discard intervention recording because no recording is active")
        source = self.active_source or "unknown"
        arm = self.active_arm
        self.client.mark(
            "intervention_end",
            rollout_step=rollout_step,
            seq=seq,
            source=source,
            device=source,
            arm=arm,
            end_reason=end_reason,
        )
        self.client.stop("discarded")
        self.active_source = None
        self.active_arm = "dual"

    def stop_failed(self, status: str, error: str | None = None) -> None:
        if not self.active:
            return
        self.client.stop(status, error=error)
        self.active_source = None
        self.active_arm = "dual"


class OperatorEventReceiver:
    def __init__(self, *, host: str, port: int, path: str):
        self.host = host
        self.port = int(port)
        self.path = "/" + path.strip("/")
        self._events: queue.Queue[OperatorEventItem] = queue.Queue(maxsize=1024)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                if self.path != receiver.path:
                    _json_response(self, 404, {"success": False, "message": "not_found"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                try:
                    raw = self.rfile.read(length)
                    payload = json.loads(raw.decode("utf-8"))
                    event = _validate_operator_event(payload)
                    item = receiver.submit(event)
                    success, message, status = item.wait()
                except Exception as exc:
                    _json_response(self, 400, {"success": False, "message": str(exc)})
                    return
                _json_response(self, status, {"success": success, "message": message})

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="hg-dagger-operator-event-http", daemon=True)
        self._thread.start()
        print(f"[hg-dagger] operator event server listening on http://{self.host}:{self.port}{self.path}", flush=True)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def submit(self, event: dict[str, Any]) -> OperatorEventItem:
        item = OperatorEventItem(payload=event, requires_ack=_operator_event_requires_ack(event))
        try:
            self._events.put_nowait(item)
        except queue.Full as exc:
            raise RuntimeError("operator event queue is full") from exc
        return item

    def get(self) -> OperatorEventItem | None:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None


def _next_operator_control_event(operator_events: OperatorEventReceiver | None) -> OperatorControlEvent | None:
    if operator_events is None:
        return None
    while True:
        item = operator_events.get()
        if item is None:
            return None
        action = _operator_control_action(item.payload)
        if action is not None:
            return OperatorControlEvent(action=action, item=item)


def _resolve_operator_control_event(
    control_event: OperatorControlEvent | None,
    *,
    success: bool,
    message: str,
    status: int = 200,
) -> None:
    if control_event is not None:
        control_event.item.resolve(success=success, message=message, status=status)


def _post_takeover_active(args: argparse.Namespace, *, active: bool) -> dict[str, Any]:
    try:
        return _request_json(
            "POST",
            _url(args.robot_server, args.takeover_http_path),
            args.timeout,
            {"active": bool(active), "owner": "policy"},
        )
    except RuntimeError as exc:
        raise RuntimeError(f"takeover HTTP request active={active} failed: {exc}") from exc


def _wait_takeover_switch_delay(args: argparse.Namespace, transition: str) -> None:
    print(f"[hg-dagger] waiting {args.takeover_switch_delay_sec:.3f}s before tabletop {transition}", flush=True)
    time.sleep(args.takeover_switch_delay_sec)


class SpaceMouseInterventionController:
    def __init__(self, args: argparse.Namespace):
        self.expert = PySpaceMouseExpert(
            dual=args.intervention_arm == "dual",
            left_path=args.left_spacemouse_path,
            right_path=args.right_spacemouse_path,
        )
        self.arm = args.intervention_arm
        self.xyz_scale = args.teleop_xyz_scale
        self.rot_scale = args.teleop_rot_scale
        self.trans_deadzone = args.teleop_trans_deadzone
        self.rot_deadzone = args.teleop_rot_deadzone
        self.activate_threshold = args.intervention_activate_threshold
        self.release_threshold = args.intervention_release_threshold
        self.active_by_arm = dict.fromkeys(self._controlled_arms(), False)
        print(f"[hg-dagger] calibrating SpaceMouse for {args.teleop_calibrate_seconds:.2f}s, keep it untouched...")
        self.bias = _estimate_idle_bias(self.expert, args.teleop_calibrate_seconds)
        print(
            "[hg-dagger] calibration complete: " f"bias={np.array2string(self.bias, precision=4, suppress_small=True)}"
        )

    def close(self) -> None:
        self.expert.close()

    def _controlled_arms(self) -> tuple[str, ...]:
        if self.arm == "dual":
            return ("left", "right")
        return (self.arm,)

    def _read_filtered_groups(self) -> list[tuple[np.ndarray, list[Any]]]:
        groups = self.expert.get_action_groups()
        filtered_groups = []
        offset = 0
        for raw_action, buttons in groups:
            action = np.asarray(raw_action, dtype=np.float64)
            bias = self.bias[offset : offset + action.shape[0]]
            if bias.shape[0] != action.shape[0]:
                raise RuntimeError(f"SpaceMouse bias/action shape mismatch: bias={bias.shape}, action={action.shape}")
            filtered_groups.append(
                (
                    _apply_deadzone(
                        action - bias,
                        trans_deadzone=self.trans_deadzone,
                        rot_deadzone=self.rot_deadzone,
                    ),
                    list(buttons),
                )
            )
            offset += action.shape[0]
        return filtered_groups

    def _is_arm_active(self, arm: str, action: np.ndarray, buttons: list[Any]) -> bool:
        if action.shape[0] < 6:
            raise RuntimeError("failed to read a valid SpaceMouse 6DoF action")
        motion_norm = float(np.linalg.norm(action[:6]))
        button_pressed = any(bool(value) for value in buttons)
        threshold = self.release_threshold if self.active_by_arm[arm] else self.activate_threshold
        return button_pressed or motion_norm > threshold

    def _arm_payload(
        self, raw_state: dict[str, Any], arm: str, action: np.ndarray, buttons: list[Any]
    ) -> dict[str, Any]:
        arm_payload: dict[str, Any] = {
            "pose_target": _pose_target_from_action(
                _tcp_pose(raw_state, arm),
                action[:6],
                self.xyz_scale,
                self.rot_scale,
            ).tolist(),
            "preset": "free_space",
        }
        gripper = _single_arm_gripper(buttons)
        if gripper is not None:
            arm_payload["gripper"] = gripper
        return arm_payload

    def _payload_for_active_actions(
        self,
        raw_state: dict[str, Any],
        active_inputs: list[tuple[str, np.ndarray, list[Any]]],
        seq: int,
        *,
        owner: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "owner": owner,
            "mode": "ee_pose_servo",
            "seq": seq,
        }
        if owner == "teleop":
            payload["teleop_source"] = "spacemouse"
        for arm, action, buttons in active_inputs:
            payload[arm] = self._arm_payload(raw_state, arm, action, buttons)
        return payload

    def poll(self) -> tuple[list[tuple[str, np.ndarray, list[Any]]], list[tuple[str, str]]]:
        groups = self._read_filtered_groups()
        arms = self._controlled_arms()
        if len(groups) < len(arms):
            raise RuntimeError(f"{self.arm} intervention requires {len(arms)} SpaceMouse device(s), got {len(groups)}")

        events: list[tuple[str, str]] = []
        active_inputs: list[tuple[str, np.ndarray, list[Any]]] = []
        for arm, (action, buttons) in zip(arms, groups, strict=True):
            was_active = self.active_by_arm[arm]
            active_now = self._is_arm_active(arm, action, buttons)
            if active_now and not was_active:
                events.append(("intervention_start", arm))
            elif was_active and not active_now:
                events.append(("intervention_end", arm))
            self.active_by_arm[arm] = active_now
            if active_now:
                active_inputs.append((arm, action, buttons))
        return active_inputs, events

    def decision(
        self,
        raw_state: dict[str, Any],
        seq: int,
        *,
        owner: str = "policy",
    ) -> tuple[dict[str, Any] | None, list[tuple[str, str]]]:
        active_inputs, events = self.poll()
        if not active_inputs:
            return None, events
        return self._payload_for_active_actions(raw_state, active_inputs, seq, owner=owner), events

    def payload_for_inputs(
        self,
        raw_state: dict[str, Any],
        active_inputs: list[tuple[str, np.ndarray, list[Any]]],
        seq: int,
        *,
        owner: str = "policy",
    ) -> dict[str, Any]:
        return self._payload_for_active_actions(raw_state, active_inputs, seq, owner=owner)


class HgDaggerInterventionRuntime:
    def __init__(self, args: argparse.Namespace, *, control_script: str):
        self.args = args
        self.control_script = control_script
        self.controller = SpaceMouseInterventionController(args) if args.intervene and args.teleop_source == "spacemouse" else None
        self.operator_events = (
            OperatorEventReceiver(
                host=args.operator_event_host,
                port=args.operator_event_port,
                path=args.operator_event_path,
            )
            if args.intervene and args.teleop_source == "tabletop"
            else None
        )
        self.recorder = InterventionRecordingSession(args, control_script) if args.record else None
        self._tabletop_active = False
        self._tabletop_takeover_state: dict[str, Any] | None = None

    def start(self) -> None:
        if self.operator_events is not None:
            self.operator_events.start()

    def close(self, *, status: str = "interrupted", error: str | None = None) -> None:
        if self.recorder is not None and self.recorder.active:
            self.recorder.stop_failed(status, error=error)
        if self.operator_events is not None:
            self.operator_events.stop()
        if self.controller is not None:
            self.controller.close()

    def maybe_run_step(
        self,
        get_state: Callable[[str, float], dict[str, Any]],
        execute_payload: Callable[[argparse.Namespace, dict[str, Any]], None],
        steps_done: int,
        seq: int,
    ) -> InterventionStepResult:
        if not self.args.intervene:
            return InterventionStepResult(handled=False, released=False, steps_done=steps_done, seq=seq)
        if self.args.teleop_source == "spacemouse":
            return self._maybe_run_spacemouse_step(get_state, execute_payload, steps_done, seq)
        if self.args.teleop_source == "tabletop":
            return self._maybe_run_tabletop_step(get_state, execute_payload, steps_done, seq)
        raise RuntimeError(f"unsupported teleop_source: {self.args.teleop_source}")

    def _maybe_start_recording(self, *, source: str, rollout_step: int, seq: int, arm: str) -> None:
        if self.recorder is None:
            print(f"[hg-dagger] intervention_start step={rollout_step} seq={seq} arm={arm} recording=off")
            return
        if not self.recorder.active:
            self.recorder.start(source=source, rollout_step=rollout_step, seq=seq, arm=arm)

    def _save_recording(self, *, rollout_step: int, seq: int, end_reason: str) -> None:
        if self.recorder is not None and self.recorder.active:
            self.recorder.save(rollout_step=rollout_step, seq=seq, end_reason=end_reason)

    def _discard_recording(self, *, rollout_step: int, seq: int, end_reason: str) -> None:
        if self.recorder is not None and self.recorder.active:
            self.recorder.discard(rollout_step=rollout_step, seq=seq, end_reason=end_reason)

    def _restore_takeover_start(
        self,
        get_state: Callable[[str, float], dict[str, Any]],
        execute_payload: Callable[[argparse.Namespace, dict[str, Any]], None],
        takeover_state: dict[str, Any],
        seq: int,
        *,
        label: str,
    ) -> int:
        seq = _hold_current_joint_target(self.args, takeover_state, execute_payload, seq)
        _wait_until_robot_reaches_state(self.args, get_state, takeover_state, label=label)
        print(
            "[hg-dagger] discarded intervention. Robot joints restored to takeover start. "
            "Restore moved objects if needed before retrying takeover.",
            flush=True,
        )
        return seq

    def _maybe_run_spacemouse_step(
        self,
        get_state: Callable[[str, float], dict[str, Any]],
        execute_payload: Callable[[argparse.Namespace, dict[str, Any]], None],
        steps_done: int,
        seq: int,
    ) -> InterventionStepResult:
        if self.controller is None:
            return InterventionStepResult(handled=False, released=False, steps_done=steps_done, seq=seq)

        active_inputs, events = self.controller.poll()
        if active_inputs:
            arm = self.args.intervention_arm
            if self.recorder is None or not self.recorder.active:
                self._spacemouse_takeover_state = get_state(self.args.robot_server, self.args.timeout)
                self._maybe_start_recording(source="spacemouse", rollout_step=steps_done, seq=seq, arm=arm)
            raw_state = get_state(self.args.robot_server, self.args.timeout)
            payload = self.controller.payload_for_inputs(raw_state, active_inputs, seq, owner="teleop")
            execute_payload(self.args, payload)
            steps_done += 1
            seq += 1
            time.sleep(1.0 / self.args.control_hz)
            return InterventionStepResult(handled=True, released=False, steps_done=steps_done, seq=seq)

        if any(event == "intervention_end" for event, _arm in events):
            print(f"[hg-dagger] intervention_end step={steps_done} seq={seq} arm={self.args.intervention_arm}")
            if self.recorder is None or not self.recorder.active:
                return InterventionStepResult(handled=False, released=True, steps_done=steps_done, seq=seq)
            idle_deadline = time.monotonic() + self.args.teleop_idle_seconds
            while time.monotonic() < idle_deadline:
                active_inputs, _events = self.controller.poll()
                if active_inputs:
                    return InterventionStepResult(handled=True, released=False, steps_done=steps_done, seq=seq)
                time.sleep(min(0.05, max(0.0, idle_deadline - time.monotonic())))
            decision = _read_save_or_discard()
            if decision == "enter":
                self._save_recording(rollout_step=steps_done, seq=seq, end_reason="operator_save")
                return InterventionStepResult(handled=True, released=True, steps_done=steps_done, seq=seq)
            takeover_state = getattr(self, "_spacemouse_takeover_state", None)
            if takeover_state is None:
                raise RuntimeError("SpaceMouse discard requested but takeover start state was not captured")
            self._discard_recording(rollout_step=steps_done, seq=seq, end_reason="operator_discard")
            seq = self._restore_takeover_start(
                get_state,
                execute_payload,
                takeover_state,
                seq,
                label="spacemouse discard restore",
            )
            return InterventionStepResult(handled=True, released=True, steps_done=steps_done, seq=seq)

        return InterventionStepResult(handled=False, released=False, steps_done=steps_done, seq=seq)

    def _maybe_run_tabletop_step(
        self,
        get_state: Callable[[str, float], dict[str, Any]],
        execute_payload: Callable[[argparse.Namespace, dict[str, Any]], None],
        steps_done: int,
        seq: int,
    ) -> InterventionStepResult:
        operator_control = _next_operator_control_event(self.operator_events)
        if operator_control is not None:
            if operator_control.action == "takeover_start":
                if self._tabletop_active:
                    _resolve_operator_control_event(
                        operator_control,
                        success=False,
                        message="duplicate_takeover_start_while_recording_intervention",
                        status=409,
                    )
                    return InterventionStepResult(handled=True, released=False, steps_done=steps_done, seq=seq)
                self._tabletop_takeover_state = get_state(self.args.robot_server, self.args.timeout)
                _wait_takeover_switch_delay(self.args, "takeover")
                result = _post_takeover_active(self.args, active=True)
                print(f"[hg-dagger] tabletop takeover active: {result}", flush=True)
                self._tabletop_active = True
                self._maybe_start_recording(source="tabletop", rollout_step=steps_done, seq=seq, arm="dual")
                _resolve_operator_control_event(operator_control, success=True, message="intervention_start")
                return InterventionStepResult(handled=True, released=False, steps_done=steps_done, seq=seq)
            if operator_control.action == "save_and_end_takeover":
                if not self._tabletop_active:
                    _resolve_operator_control_event(
                        operator_control,
                        success=False,
                        message="save_requires_active_tabletop_takeover",
                        status=409,
                    )
                    return InterventionStepResult(handled=True, released=False, steps_done=steps_done, seq=seq)
                latest_state = get_state(self.args.robot_server, self.args.timeout)
                seq = _hold_current_joint_target(self.args, latest_state, execute_payload, seq)
                self._save_recording(rollout_step=steps_done, seq=seq, end_reason="operator_save")
                _wait_takeover_switch_delay(self.args, "release")
                result = _post_takeover_active(self.args, active=False)
                print(f"[hg-dagger] tabletop takeover released: {result}", flush=True)
                self._tabletop_active = False
                self._tabletop_takeover_state = None
                _resolve_operator_control_event(operator_control, success=True, message="intervention_save_end")
                return InterventionStepResult(handled=True, released=True, steps_done=steps_done, seq=seq)
            if operator_control.action == "discard_without_save_and_end_takeover":
                if not self._tabletop_active:
                    _resolve_operator_control_event(
                        operator_control,
                        success=False,
                        message="discard_requires_active_tabletop_takeover",
                        status=409,
                    )
                    return InterventionStepResult(handled=True, released=False, steps_done=steps_done, seq=seq)
                if self._tabletop_takeover_state is None:
                    raise RuntimeError("tabletop discard requested but takeover start state was not captured")
                seq = self._restore_takeover_start(
                    get_state,
                    execute_payload,
                    self._tabletop_takeover_state,
                    seq,
                    label="tabletop discard restore",
                )
                self._discard_recording(rollout_step=steps_done, seq=seq, end_reason="operator_discard")
                _wait_takeover_switch_delay(self.args, "release")
                result = _post_takeover_active(self.args, active=False)
                print(f"[hg-dagger] tabletop takeover released: {result}", flush=True)
                self._tabletop_active = False
                self._tabletop_takeover_state = None
                _resolve_operator_control_event(operator_control, success=True, message="intervention_discard_end")
                return InterventionStepResult(handled=True, released=True, steps_done=steps_done, seq=seq)
            if operator_control.action == "discard_trajectory":
                _resolve_operator_control_event(
                    operator_control,
                    success=False,
                    message="discard_trajectory_not_supported_in_hg_dagger",
                    status=409,
                )
                return InterventionStepResult(handled=True, released=False, steps_done=steps_done, seq=seq)
            _resolve_operator_control_event(
                operator_control,
                success=False,
                message=f"unsupported_operator_action:{operator_control.action}",
                status=409,
            )
            return InterventionStepResult(handled=True, released=False, steps_done=steps_done, seq=seq)

        if not self._tabletop_active:
            return InterventionStepResult(handled=False, released=False, steps_done=steps_done, seq=seq)
        key = _poll_terminal_key()
        if key == self.args.tabletop_release_key:
            latest_state = get_state(self.args.robot_server, self.args.timeout)
            seq = _hold_current_joint_target(self.args, latest_state, execute_payload, seq)
            decision = _read_save_or_discard()
            if decision == "enter":
                self._save_recording(rollout_step=steps_done, seq=seq, end_reason="keyboard_save")
            else:
                if self._tabletop_takeover_state is None:
                    raise RuntimeError("tabletop discard requested but takeover start state was not captured")
                self._discard_recording(rollout_step=steps_done, seq=seq, end_reason="keyboard_discard")
                seq = self._restore_takeover_start(
                    get_state,
                    execute_payload,
                    self._tabletop_takeover_state,
                    seq,
                    label="tabletop discard restore",
                )
            _wait_takeover_switch_delay(self.args, "release")
            result = _post_takeover_active(self.args, active=False)
            print(f"[hg-dagger] tabletop takeover released: {result}", flush=True)
            self._tabletop_active = False
            self._tabletop_takeover_state = None
            return InterventionStepResult(handled=True, released=True, steps_done=steps_done, seq=seq)
        if key == "q":
            raise KeyboardInterrupt
        time.sleep(1.0 / self.args.control_hz)
        return InterventionStepResult(handled=True, released=False, steps_done=steps_done, seq=seq)


def add_spacemouse_intervention_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--intervention-arm", choices=("left", "right", "dual"), default=None)
    parser.add_argument(
        "--left-spacemouse-path", default=None, help="Explicit left SpaceMouse HID path, e.g. /dev/hidraw3."
    )
    parser.add_argument(
        "--right-spacemouse-path", default=None, help="Explicit right SpaceMouse HID path, e.g. /dev/hidraw4."
    )
    parser.add_argument(
        "--list-spacemouse-devices",
        action="store_true",
        help="List supported SpaceMouse HID devices and exit.",
    )
    parser.add_argument("--teleop-calibrate-seconds", type=float, default=None)
    parser.add_argument("--teleop-trans-deadzone", type=float, default=None)
    parser.add_argument("--teleop-rot-deadzone", type=float, default=None)
    parser.add_argument("--intervention-activate-threshold", type=float, default=None)
    parser.add_argument("--intervention-release-threshold", type=float, default=None)
    parser.add_argument("--teleop-xyz-scale", type=float, default=None)
    parser.add_argument("--teleop-rot-scale", type=float, default=None)


def add_hg_dagger_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--record", action="store_true", help="Record only HG-Dagger human intervention segments on the robot server."
    )
    parser.add_argument(
        "--intervene",
        action="store_true",
        help="Enable configured human intervention during policy rollout without requiring robot-side recording.",
    )
    parser.add_argument(
        "--record-output-dir",
        default=None,
        help="Robot-server-local RAW parent directory. This path is interpreted on the robot, not on the inference PC.",
    )
    parser.add_argument(
        "--record-stop-timeout",
        type=float,
        default=None,
        help="HTTP timeout in seconds for finalizing robot-side rosbag recording.",
    )
    parser.add_argument(
        "--record-topic",
        dest="record_topics",
        action="append",
        default=None,
        help="ROS topic to record on the robot. Repeat this flag for multiple topics.",
    )
    add_spacemouse_intervention_args(parser)
    parser.add_argument("--teleop-source", choices=("spacemouse", "tabletop"), default=None)
    parser.add_argument("--teleop-idle-seconds", type=float, default=None)
    parser.add_argument("--takeover-http-path", default=None)
    parser.add_argument("--takeover-switch-delay-sec", type=float, default=None)
    parser.add_argument("--tabletop-release-key", choices=("enter", "esc"), default=None)
    parser.add_argument("--tabletop-restore-timeout-sec", type=float, default=None)
    parser.add_argument("--tabletop-restore-joint-tolerance", type=float, default=None)
    parser.add_argument("--operator-event-host", default=None)
    parser.add_argument("--operator-event-port", type=int, default=None)
    parser.add_argument("--operator-event-path", default=None)


def fill_spacemouse_intervention_defaults(args: argparse.Namespace) -> None:
    if args.intervention_arm is None:
        args.intervention_arm = "dual"
    defaults = {
        "teleop_calibrate_seconds": 0.5,
        "teleop_trans_deadzone": 0.08,
        "teleop_rot_deadzone": 0.08,
        "intervention_activate_threshold": 0.001,
        "intervention_release_threshold": 0.001,
        "teleop_xyz_scale": 0.03,
        "teleop_rot_scale": 0.20,
        "teleop_source": "spacemouse",
        "teleop_idle_seconds": 1.0,
    }
    for attr, value in defaults.items():
        if hasattr(args, attr) and getattr(args, attr) is None:
            setattr(args, attr, value)
    if not hasattr(args, "left_spacemouse_path"):
        args.left_spacemouse_path = None
    if not hasattr(args, "right_spacemouse_path"):
        args.right_spacemouse_path = None
    config_path = getattr(args, "experiment_config_path", None)
    if config_path and (args.left_spacemouse_path is None or args.right_spacemouse_path is None):
        with Path(config_path).open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        hg_dagger = cfg.get("hg_dagger", {}) if isinstance(cfg, dict) else {}
        if isinstance(hg_dagger, dict):
            if args.left_spacemouse_path is None:
                args.left_spacemouse_path = hg_dagger.get("left_spacemouse_path")
            if args.right_spacemouse_path is None:
                args.right_spacemouse_path = hg_dagger.get("right_spacemouse_path")


def fill_hg_dagger_defaults(args: argparse.Namespace) -> None:
    if args.record_output_dir is None:
        args.record_output_dir = DEFAULT_ROBOT_RECORD_OUTPUT_DIR
    if args.record_stop_timeout is None:
        args.record_stop_timeout = DEFAULT_RECORD_STOP_TIMEOUT
    if args.record_topics is None:
        args.record_topics = list(DEFAULT_RECORD_TOPICS)
    fill_spacemouse_intervention_defaults(args)
    if args.teleop_source is not None:
        args.teleop_source = str(args.teleop_source).lower()
    if args.takeover_http_path is not None:
        args.takeover_http_path = "/" + str(args.takeover_http_path).strip("/")
    if args.tabletop_release_key is not None:
        args.tabletop_release_key = str(args.tabletop_release_key).lower()
    if args.operator_event_path is not None:
        args.operator_event_path = "/" + str(args.operator_event_path).strip("/")


def validate_hg_dagger_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    fill_hg_dagger_defaults(args)
    if not args.execute:
        if args.record:
            parser.error("--record requires --execute")
        if args.intervene:
            parser.error("--intervene requires --execute")
    if args.debug:
        if args.record:
            parser.error("--record is incompatible with --debug")
        if args.intervene:
            parser.error("--intervene is incompatible with --debug")
    if args.record:
        args.intervene = True
    if args.record and args.max_steps is None:
        parser.error("--record requires --max-steps")
    if args.record_stop_timeout <= 0:
        parser.error("--record-stop-timeout must be positive")
    if not args.record_topics:
        parser.error("--record requires at least one --record-topic")
    invalid_record_topics = [
        topic for topic in args.record_topics if not isinstance(topic, str) or not topic.startswith("/")
    ]
    if invalid_record_topics:
        parser.error(f"record topics must be absolute ROS topic names: {invalid_record_topics}")
    for name in (
        "teleop_calibrate_seconds",
        "teleop_trans_deadzone",
        "teleop_rot_deadzone",
        "intervention_activate_threshold",
        "intervention_release_threshold",
        "teleop_xyz_scale",
        "teleop_rot_scale",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.teleop_source not in ("spacemouse", "tabletop"):
        parser.error("--teleop-source must be one of: spacemouse, tabletop")
    if args.teleop_idle_seconds < 0:
        parser.error("--teleop-idle-seconds must be non-negative")
    if args.teleop_source == "tabletop":
        required = (
            "takeover_http_path",
            "takeover_switch_delay_sec",
            "tabletop_release_key",
            "tabletop_restore_timeout_sec",
            "tabletop_restore_joint_tolerance",
            "operator_event_host",
            "operator_event_port",
            "operator_event_path",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"--teleop-source=tabletop requires hg_dagger/CLI values for: {', '.join(missing)}")
        if args.takeover_switch_delay_sec < 0:
            parser.error("--takeover-switch-delay-sec must be non-negative")
        if args.tabletop_release_key not in ("enter", "esc"):
            parser.error("--tabletop-release-key must be one of: enter, esc")
        if args.tabletop_restore_timeout_sec <= 0:
            parser.error("--tabletop-restore-timeout-sec must be positive")
        if args.tabletop_restore_joint_tolerance <= 0:
            parser.error("--tabletop-restore-joint-tolerance must be positive")
        if args.operator_event_port <= 0:
            parser.error("--operator-event-port must be positive")


def maybe_run_intervention_step(
    args: argparse.Namespace,
    recorder: RecordingClient | None,
    controller: SpaceMouseInterventionController | None,
    get_state: Callable[[str, float], dict[str, Any]],
    execute_payload: Callable[[argparse.Namespace, dict[str, Any]], None],
    steps_done: int,
    seq: int,
) -> InterventionStepResult:
    if controller is None:
        return InterventionStepResult(handled=False, released=False, steps_done=steps_done, seq=seq)

    active_inputs, events = controller.poll()
    for event, arm in events:
        if recorder is None:
            print(f"[hg-dagger] {event} step={steps_done} seq={seq} arm={arm} recording=off")
        else:
            recorder.mark(
                event,
                rollout_step=steps_done,
                seq=seq,
                source="spacemouse",
                device="spacemouse",
                arm=arm,
            )

    if not active_inputs:
        return InterventionStepResult(
            handled=False,
            released=any(event == "intervention_end" for event, _arm in events),
            steps_done=steps_done,
            seq=seq,
        )

    raw_state = get_state(args.robot_server, args.timeout)
    payload = controller.payload_for_inputs(raw_state, active_inputs, seq, owner="teleop")
    execute_payload(args, payload)
    steps_done += 1
    seq += 1
    time.sleep(1.0 / args.control_hz)
    return InterventionStepResult(handled=True, released=False, steps_done=steps_done, seq=seq)
