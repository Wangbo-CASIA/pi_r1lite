#!/usr/bin/env python3
"""Collect chunked ALOE replay directly from the R1Lite robot server."""

from __future__ import annotations

import argparse
import dataclasses
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

import numpy as np
from openpi_client import websocket_client_policy
from r1lite_experiment_config import add_experiment_args
from r1lite_experiment_config import apply_rollout_config
from r1lite_hg_dagger import SpaceMouseInterventionController
from r1lite_hg_dagger import add_spacemouse_intervention_args
from r1lite_hg_dagger import fill_spacemouse_intervention_defaults
from r1lite_hg_dagger import list_spacemouse_devices

from openpi.aloe import r1lite_runtime as rt
from openpi.aloe import schema
from openpi.aloe.config import load_aloe_config
from openpi.aloe.config import require_dict
from openpi.aloe.writer import AsyncPklReplayWriter
from openpi.aloe.writer import next_run_dir

DEFAULT_PROMPT = (
    "first scoop up the black foam and place it in the box, then scoop up the phone and place it in the box, "
    "and finally pick up the lid, put it on the box, and press it down firmly"
)


@dataclasses.dataclass
class ChunkDraft:
    segment_id: int
    source: schema.SegmentSource
    start_seq: int
    observation: dict[str, Any]
    actions: list[np.ndarray]
    next_observation: dict[str, Any]
    infos: dict[str, Any]

    @property
    def length(self) -> int:
        return len(self.actions)


class TailSegmentBuffer:
    def __init__(self, *, segment_id: int, source: schema.SegmentSource):
        self.segment_id = segment_id
        self.source = source
        self._tail: ChunkDraft | None = None

    def append(self, writer: AsyncPklReplayWriter, draft: ChunkDraft, *, cfail: float) -> None:
        if draft.length < 1:
            raise ValueError("cannot append empty chunk")
        if self._tail is not None:
            writer.append_chunk(
                _chunk_record_from_draft(self._tail, terminal_outcome=None, terminal_reason=None, cfail=cfail)
            )
        self._tail = draft

    def clear(self) -> None:
        self._tail = None

    def flush(
        self,
        writer: AsyncPklReplayWriter,
        *,
        terminal_outcome: schema.TerminalOutcome | None,
        terminal_reason: str | None,
        cfail: float,
    ) -> None:
        if self._tail is None:
            return
        writer.append_chunk(
            _chunk_record_from_draft(
                self._tail,
                terminal_outcome=terminal_outcome,
                terminal_reason=terminal_reason,
                cfail=cfail,
            )
        )
        self._tail = None


class CorrectionBuffer:
    def __init__(self):
        self._drafts: list[ChunkDraft] = []

    @property
    def has_data(self) -> bool:
        return bool(self._drafts)

    def append(self, draft: ChunkDraft) -> None:
        if draft.length < 1:
            raise ValueError("cannot append empty correction chunk")
        self._drafts.append(draft)

    def clear(self) -> None:
        self._drafts.clear()

    def flush(
        self,
        writer: AsyncPklReplayWriter,
        *,
        cfail: float,
        terminal_outcome: schema.TerminalOutcome | None = None,
        terminal_reason: str | None = None,
    ) -> None:
        for index, draft in enumerate(self._drafts):
            is_tail = index == len(self._drafts) - 1
            writer.append_chunk(
                _chunk_record_from_draft(
                    draft,
                    terminal_outcome=terminal_outcome if is_tail else None,
                    terminal_reason=terminal_reason if is_tail else None,
                    cfail=cfail,
                )
            )
        self.clear()


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


class KeyEventThread:
    def __init__(self):
        self._events: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="aloe-key-events", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get(self) -> str | None:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def put_back(self, key: str) -> None:
        self._events.put(key)

    def wait_for(self, allowed: set[str]) -> str:
        while True:
            key = self.get()
            if key in allowed:
                return key
            time.sleep(0.05)

    def _run(self) -> None:
        while not self._stop.is_set():
            key = rt.poll_key()
            if key in ("f", "s", "q", "enter", "esc"):
                self._events.put(key)
            time.sleep(0.01)


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    try:
        handler.wfile.write(body)
    except BrokenPipeError:
        print(
            f"[aloe] operator event response dropped: client disconnected before status={status} payload={payload}",
            flush=True,
        )


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
    if event == "gripper":
        side_by_action = {
            "left_gripper_open": "left",
            "left_gripper_close": "left",
            "right_gripper_open": "right",
            "right_gripper_close": "right",
        }
        if action not in side_by_action:
            raise ValueError(f"invalid gripper action: {action}")
        side = payload.get("side")
        if side != side_by_action[action]:
            raise ValueError(f"gripper action {action} requires side={side_by_action[action]}")
        if payload.get("active") is not None:
            raise ValueError("gripper event must not include active")
    elif event == "takeover":
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
        if (
            action
            in (
                "discard_without_save_and_end_takeover",
                "save_and_end_takeover",
            )
            and active is not False
        ):
            raise ValueError(f"{action} requires active=false")
        if pressed is not True:
            raise ValueError("takeover event must be sent on button press")
        if payload.get("side") is not None:
            raise ValueError("takeover event must not include side")
    elif event == "trajectory":
        if action not in ("mark_success", "discard_trajectory"):
            raise ValueError(f"invalid trajectory action: {action}")
        if pressed is not True:
            raise ValueError("trajectory event must be sent on button press")
        if payload.get("side") is not None or payload.get("active") is not None:
            raise ValueError("trajectory event must not include side or active")
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


class OperatorEventReceiver:
    def __init__(self, *, host: str, port: int, path: str):
        self.host = host
        self.port = int(port)
        self.path = "/" + path.strip("/")
        self._events: queue.Queue[OperatorEventItem] = queue.Queue(maxsize=1024)
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._gripper_pressed = {
            "left_gripper_open": False,
            "left_gripper_close": False,
            "right_gripper_open": False,
            "right_gripper_close": False,
        }
        self._gripper_active_action: dict[str, str | None] = {"left": None, "right": None}
        self._last_takeover_action: str | None = None
        self._last_trajectory_action: str | None = None

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
        self._thread = threading.Thread(target=self._server.serve_forever, name="aloe-operator-event-http", daemon=True)
        self._thread.start()
        print(f"[aloe] operator event server listening on http://{self.host}:{self.port}{self.path}", flush=True)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def submit(self, event: dict[str, Any]) -> OperatorEventItem:
        item = OperatorEventItem(payload=event, requires_ack=_operator_event_requires_ack(event))
        with self._lock:
            if event["event"] == "gripper":
                self._gripper_pressed[event["action"]] = bool(event["pressed"])
                self._update_active_gripper_action(
                    str(event["side"]),
                    str(event["action"]),
                    pressed=bool(event["pressed"]),
                )
            elif event["event"] == "takeover":
                self._last_takeover_action = event["action"]
            elif event["event"] == "trajectory":
                self._last_trajectory_action = event["action"]
        try:
            self._events.put_nowait(item)
        except queue.Full as exc:
            raise RuntimeError("operator event queue is full") from exc
        return item

    def _update_active_gripper_action(self, side: str, action: str, *, pressed: bool) -> None:
        if pressed:
            self._gripper_active_action[side] = action
            return
        if self._gripper_active_action.get(side) != action:
            return
        candidates = (
            (f"{side}_gripper_close", f"{side}_gripper_open")
            if action.endswith("_open")
            else (f"{side}_gripper_open", f"{side}_gripper_close")
        )
        self._gripper_active_action[side] = next(
            (candidate for candidate in candidates if self._gripper_pressed.get(candidate)),
            None,
        )

    def get(self) -> OperatorEventItem | None:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def gripper_labels(self) -> dict[str, float]:
        with self._lock:
            labels: dict[str, float] = {}
            for side, action in self._gripper_active_action.items():
                if action is not None:
                    labels[side] = 1.0 if action.endswith("_close") else 0.0
            return labels


def _operator_control_action(event: dict[str, Any]) -> str | None:
    if event["event"] == "takeover":
        return str(event["action"])
    if event["event"] == "trajectory":
        return str(event["action"])
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


def _policy_actions(client: websocket_client_policy.WebsocketClientPolicy, observation: dict[str, Any]) -> np.ndarray:
    result = client.infer(observation)
    if "actions" not in result:
        raise RuntimeError(f"Policy response is missing 'actions'. Keys: {sorted(result)}")
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != schema.ACTION_DIM:
        raise RuntimeError(f"Expected policy actions with shape (horizon, {schema.ACTION_DIM}), got {actions.shape}")
    if actions.shape[0] < 1:
        raise RuntimeError("Policy returned an empty action chunk")
    return actions


def _obs_from_raw(args: argparse.Namespace, raw_state: dict[str, Any]) -> dict[str, Any]:
    return rt.observation_from_state(
        raw_state,
        args.prompt,
        args.left_gripper_threshold,
        args.right_gripper_threshold,
    )


def _chunk_record_from_draft(
    draft: ChunkDraft,
    *,
    terminal_outcome: schema.TerminalOutcome | None,
    terminal_reason: str | None,
    cfail: float,
) -> dict[str, Any]:
    actions = np.asarray(draft.actions, dtype=np.float32)
    rewards = np.full((actions.shape[0],), -1.0, dtype=np.float32)
    dones = np.zeros((actions.shape[0],), dtype=bool)
    valid_mask = np.ones((actions.shape[0],), dtype=bool)
    if terminal_outcome is not None:
        rewards[-1] = schema.terminal_reward(terminal_outcome, cfail)
        dones[-1] = True
    return schema.make_chunk_record(
        run_id=draft.infos["run_id"],
        segment_id=draft.segment_id,
        source=draft.source,
        start_seq=draft.start_seq,
        prompt=draft.infos["prompt"],
        policy_checkpoint=draft.infos.get("policy_checkpoint"),
        action_space="joint_delta",
        observation=draft.observation,
        actions=actions,
        next_observation=draft.next_observation,
        rewards=rewards,
        dones=dones,
        valid_mask=valid_mask,
        terminal_outcome=terminal_outcome,
        terminal_reason=terminal_reason,
        infos=draft.infos,
    )


def _extract_joint_targets(
    response: dict[str, Any], seq: int
) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    if not isinstance(response, dict):
        raise RuntimeError(f"robot action response for seq={seq} must be a JSON object")
    left = response.get("left")
    right = response.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise KeyError("left/right")
    left_target = np.asarray(left.get("joint_target"), dtype=np.float32).reshape(-1)
    right_target = np.asarray(right.get("joint_target"), dtype=np.float32).reshape(-1)
    if left_target.shape[0] < 6 or right_target.shape[0] < 6:
        raise KeyError("joint_target")
    left_gripper = None if left.get("gripper") is None else float(left["gripper"])
    right_gripper = None if right.get("gripper") is None else float(right["gripper"])
    return left_target[:6].copy(), right_target[:6].copy(), left_gripper, right_gripper


def _extract_joint_targets_from_health(
    health: dict[str, Any], seq: int
) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    commands = health.get("commands")
    if not isinstance(commands, dict):
        raise RuntimeError(f"robot health response for seq={seq} is missing commands")
    left = commands.get("left")
    right = commands.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise RuntimeError(f"robot health response for seq={seq} is missing commands.left/right")
    left_target = np.asarray(left.get("desired_joint"), dtype=np.float32).reshape(-1)
    right_target = np.asarray(right.get("desired_joint"), dtype=np.float32).reshape(-1)
    if left_target.shape[0] < 6 or right_target.shape[0] < 6:
        raise RuntimeError(
            f"robot health response for seq={seq} does not expose 6D commands.left/right.desired_joint. "
            "For joint_delta ALOE correction labels, the robot server must expose accepted joint targets "
            "in /action response or /health commands."
        )
    left_gripper = None if left.get("gripper") is None else float(left["gripper"])
    right_gripper = None if right.get("gripper") is None else float(right["gripper"])
    return left_target[:6].copy(), right_target[:6].copy(), left_gripper, right_gripper


def _accepted_joint_targets(
    args: argparse.Namespace, response: dict[str, Any], seq: int
) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    try:
        return _extract_joint_targets(response, seq)
    except KeyError:
        health = rt.get_robot_health(args.robot_server, args.timeout)
        return _extract_joint_targets_from_health(health, seq)


def _binary_gripper_from_command(command: float | None, open_value: float, close_value: float) -> float:
    if command is None:
        return 0.0
    open_distance = abs(float(command) - float(open_value))
    close_distance = abs(float(command) - float(close_value))
    return 1.0 if close_distance <= open_distance else 0.0


def _payload_gripper_command(payload: dict[str, Any], side: str) -> float | None:
    arm_payload = payload.get(side)
    if not isinstance(arm_payload, dict) or arm_payload.get("gripper") is None:
        return None
    return float(arm_payload["gripper"])


def _binary_gripper_from_state(raw_state: dict[str, Any], side: str, args: argparse.Namespace) -> float:
    threshold = args.left_gripper_threshold if side == "left" else args.right_gripper_threshold
    return rt.binary_gripper(rt.gripper_pose(raw_state, side), threshold)


def _gripper_label_from_payload_or_state(
    payload: dict[str, Any], after_state: dict[str, Any], side: str, args: argparse.Namespace
) -> float:
    command = _payload_gripper_command(payload, side)
    if command is not None:
        return _binary_gripper_from_command(command, args.gripper_open_value, args.gripper_close_value)
    return _binary_gripper_from_state(after_state, side, args)


def _action_from_targets(
    *,
    left_reference: np.ndarray,
    right_reference: np.ndarray,
    left_target: np.ndarray,
    right_target: np.ndarray,
    left_gripper: float | None,
    right_gripper: float | None,
    args: argparse.Namespace,
) -> np.ndarray:
    action = np.zeros((schema.ACTION_DIM,), dtype=np.float32)
    action[:6] = np.asarray(left_target, dtype=np.float32).reshape(-1)[:6] - left_reference
    action[6] = _binary_gripper_from_command(left_gripper, args.gripper_open_value, args.gripper_close_value)
    action[7:13] = np.asarray(right_target, dtype=np.float32).reshape(-1)[:6] - right_reference
    action[13] = _binary_gripper_from_command(right_gripper, args.gripper_open_value, args.gripper_close_value)
    return action


def _action_from_observed_transition(
    *,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    payload: dict[str, Any],
    args: argparse.Namespace,
    gripper_labels: dict[str, float] | None = None,
) -> np.ndarray:
    action = np.zeros((schema.ACTION_DIM,), dtype=np.float32)
    action[:6] = rt.joint_pos(after_state, "left") - rt.joint_pos(before_state, "left")
    action[6] = (
        float(gripper_labels["left"])
        if gripper_labels is not None and "left" in gripper_labels
        else _gripper_label_from_payload_or_state(payload, after_state, "left", args)
    )
    action[7:13] = rt.joint_pos(after_state, "right") - rt.joint_pos(before_state, "right")
    action[13] = (
        float(gripper_labels["right"])
        if gripper_labels is not None and "right" in gripper_labels
        else _gripper_label_from_payload_or_state(payload, after_state, "right", args)
    )
    return action


def _execute_policy_chunk(
    args: argparse.Namespace,
    client: websocket_client_policy.WebsocketClientPolicy,
    key_events: KeyEventThread,
    operator_events: OperatorEventReceiver | None,
    *,
    run_id: str,
    segment_id: int,
    raw_state: dict[str, Any],
    obs: dict[str, Any],
    seq: int,
    remaining_steps: int,
) -> tuple[ChunkDraft | None, dict[str, Any], dict[str, Any], int, str | OperatorControlEvent | None]:
    actions = _policy_actions(client, obs)
    chunk_len = min(args.actions_per_infer, actions.shape[0], remaining_steps)
    left_reference = rt.joint_pos(raw_state, "left")
    right_reference = rt.joint_pos(raw_state, "right")
    executed: list[np.ndarray] = []
    interrupt: str | OperatorControlEvent | None = None
    for action in actions[:chunk_len]:
        key = key_events.get()
        if key in ("f", "s", "q"):
            interrupt = key
            break
        operator_event = _next_operator_control_event(operator_events)
        if operator_event is not None:
            interrupt = operator_event
            break
        payload = rt.joint_delta_payload(
            action,
            seq,
            raw_state,
            owner="policy",
            gripper_open_value=args.gripper_open_value,
            gripper_close_value=args.gripper_close_value,
        )
        payload["left"]["joint_target"] = [float(value) for value in left_reference + action[:6]]
        payload["right"]["joint_target"] = [float(value) for value in right_reference + action[7:13]]
        rt.post_action(args.robot_server, args.timeout, payload)
        executed.append(np.asarray(action, dtype=np.float32).copy())
        seq += 1
        time.sleep(1.0 / args.control_hz)
    if not executed:
        return None, raw_state, obs, seq, interrupt
    next_raw_state = rt.get_robot_state(args.robot_server, args.timeout)
    next_obs = _obs_from_raw(args, next_raw_state)
    draft = ChunkDraft(
        segment_id=segment_id,
        source="policy",
        start_seq=seq - len(executed),
        observation=schema.as_observation(obs),
        actions=executed,
        next_observation=schema.as_observation(next_obs),
        infos={
            "run_id": run_id,
            "prompt": args.prompt,
            "policy_checkpoint": args.policy_checkpoint,
            "control_hz": args.control_hz,
            "action_space": "joint_delta",
        },
    )
    return draft, next_raw_state, next_obs, seq, interrupt


def _execute_teleop_chunk(
    args: argparse.Namespace,
    controller: SpaceMouseInterventionController,
    *,
    run_id: str,
    segment_id: int,
    raw_state: dict[str, Any],
    obs: dict[str, Any],
    seq: int,
) -> tuple[ChunkDraft | None, dict[str, Any], dict[str, Any], int]:
    chunk_reference_raw_state = raw_state
    before_state = raw_state
    left_reference = rt.joint_pos(raw_state, "left")
    right_reference = rt.joint_pos(raw_state, "right")
    executed: list[np.ndarray] = []
    for _ in range(args.teleop_chunk_size):
        active_inputs, _events = controller.poll()
        if not active_inputs:
            if executed:
                break
            return None, raw_state, obs, seq
        payload = controller.payload_for_inputs(chunk_reference_raw_state, active_inputs, seq, owner="teleop")
        response = rt.post_action(args.robot_server, args.timeout, payload)
        if args.teleop_label_source == "accepted_joint_target":
            left_target, right_target, left_gripper, right_gripper = _accepted_joint_targets(args, response, seq)
            executed.append(
                _action_from_targets(
                    left_reference=left_reference,
                    right_reference=right_reference,
                    left_target=left_target,
                    right_target=right_target,
                    left_gripper=left_gripper,
                    right_gripper=right_gripper,
                    args=args,
                )
            )
            left_reference = left_target
            right_reference = right_target
            time.sleep(1.0 / args.control_hz)
        elif args.teleop_label_source == "observed_joint_delta":
            time.sleep(1.0 / args.control_hz)
            after_state = rt.get_robot_kinematic_state(args.robot_server, args.timeout)
            executed.append(
                _action_from_observed_transition(
                    before_state=before_state,
                    after_state=after_state,
                    payload=payload,
                    args=args,
                )
            )
            before_state = after_state
        else:
            raise RuntimeError(f"unsupported teleop_label_source: {args.teleop_label_source}")
        seq += 1
    next_raw_state = rt.get_robot_state(args.robot_server, args.timeout)
    next_obs = _obs_from_raw(args, next_raw_state)
    draft = ChunkDraft(
        segment_id=segment_id,
        source="human_correction",
        start_seq=seq - len(executed),
        observation=schema.as_observation(obs),
        actions=executed,
        next_observation=schema.as_observation(next_obs),
        infos={
            "run_id": run_id,
            "prompt": args.prompt,
            "policy_checkpoint": args.policy_checkpoint,
            "control_hz": args.control_hz,
            "action_space": "joint_delta",
            "teleop_source": "spacemouse",
            "teleop_label_source": args.teleop_label_source,
        },
    )
    return draft, next_raw_state, next_obs, seq


def _execute_tabletop_chunk(
    args: argparse.Namespace,
    operator_events: OperatorEventReceiver | None,
    key_events: KeyEventThread,
    *,
    run_id: str,
    segment_id: int,
    raw_state: dict[str, Any],
    obs: dict[str, Any],
    seq: int,
) -> tuple[ChunkDraft | None, dict[str, Any], dict[str, Any], int, str | OperatorControlEvent | None]:
    before_state = raw_state
    executed: list[np.ndarray] = []
    interrupt: str | OperatorControlEvent | None = None
    for _ in range(args.teleop_chunk_size):
        deadline = time.monotonic() + (1.0 / args.control_hz)
        while time.monotonic() < deadline:
            key = key_events.get()
            if key == "q":
                interrupt = "q"
                break
            if key == args.tabletop_release_key:
                interrupt = "keyboard_release"
                break
            operator_event = _next_operator_control_event(operator_events)
            if operator_event is not None:
                interrupt = operator_event
                break
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        if interrupt is not None:
            break
        after_state = rt.get_robot_kinematic_state(args.robot_server, args.timeout)
        gripper_labels = operator_events.gripper_labels() if operator_events is not None else None
        executed.append(
            _action_from_observed_transition(
                before_state=before_state,
                after_state=after_state,
                payload={},
                args=args,
                gripper_labels=gripper_labels,
            )
        )
        before_state = after_state
        seq += 1
    if not executed:
        return None, raw_state, obs, seq, interrupt
    next_raw_state = rt.get_robot_state(args.robot_server, args.timeout)
    next_obs = _obs_from_raw(args, next_raw_state)
    draft = ChunkDraft(
        segment_id=segment_id,
        source="human_correction",
        start_seq=seq - len(executed),
        observation=schema.as_observation(obs),
        actions=executed,
        next_observation=schema.as_observation(next_obs),
        infos={
            "run_id": run_id,
            "prompt": args.prompt,
            "policy_checkpoint": args.policy_checkpoint,
            "control_hz": args.control_hz,
            "action_space": "joint_delta",
            "teleop_source": "tabletop",
            "teleop_label_source": "observed_joint_delta",
            "gripper_label_source": "tabletop_joycon_button",
        },
    )
    return draft, next_raw_state, next_obs, seq, interrupt


def _wait_save_or_discard(key_events: KeyEventThread) -> str:
    print("Save this correction? ENTER=save, ESC=discard and retry from failed state", flush=True)
    return key_events.wait_for({"enter", "esc"})


def _drain_writer(writer: AsyncPklReplayWriter) -> None:
    print("[aloe] waiting for replay writer to finish...", flush=True)
    writer.drain()


def _print_progress(label: str, steps: int, max_steps: int) -> None:
    print(f"[aloe] {label}: policy_steps={steps}/{max_steps}", flush=True)


def _max_joint_error(raw_state: dict[str, Any], target_raw_state: dict[str, Any]) -> float:
    left_error = np.max(np.abs(rt.joint_pos(raw_state, "left") - rt.joint_pos(target_raw_state, "left")))
    right_error = np.max(np.abs(rt.joint_pos(raw_state, "right") - rt.joint_pos(target_raw_state, "right")))
    return float(max(left_error, right_error))


def _wait_until_robot_reaches_state(
    args: argparse.Namespace,
    target_raw_state: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + args.tabletop_restore_timeout_sec
    latest_state = rt.get_robot_state(args.robot_server, args.timeout)
    latest_error = _max_joint_error(latest_state, target_raw_state)
    while latest_error > args.tabletop_restore_joint_tolerance:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"{label} did not reach target within {args.tabletop_restore_timeout_sec:.3f}s: "
                f"max_joint_error={latest_error:.6f}, tolerance={args.tabletop_restore_joint_tolerance:.6f}"
            )
        time.sleep(1.0 / args.control_hz)
        latest_state = rt.get_robot_state(args.robot_server, args.timeout)
        latest_error = _max_joint_error(latest_state, target_raw_state)
    print(f"[aloe] {label} reached target: max_joint_error={latest_error:.6f}", flush=True)
    return latest_state


def _restore_failure_robot_state(args: argparse.Namespace, failed_raw_state: dict[str, Any], seq: int) -> int:
    payload = {
        "mode": "ee_pose_servo",
        "owner": "policy",
        "seq": seq,
        "left": {
            "joint_target": [float(value) for value in rt.joint_pos(failed_raw_state, "left")],
            "gripper": rt.gripper_pose(failed_raw_state, "left"),
            "preset": "free_space",
        },
        "right": {
            "joint_target": [float(value) for value in rt.joint_pos(failed_raw_state, "right")],
            "gripper": rt.gripper_pose(failed_raw_state, "right"),
            "preset": "free_space",
        },
    }
    rt.post_action(args.robot_server, args.timeout, payload)
    time.sleep(1.0 / args.control_hz)
    return seq + 1


def _post_takeover_active(args: argparse.Namespace, *, active: bool) -> dict[str, Any]:
    try:
        return rt.request_json(
            "POST",
            rt.url(args.robot_server, args.takeover_http_path),
            args.timeout,
            {"active": bool(active), "owner": "policy"},
        )
    except RuntimeError as exc:
        raise RuntimeError(f"takeover HTTP request active={active} failed: {exc}") from exc


def _wait_takeover_switch_delay(args: argparse.Namespace, transition: str) -> None:
    print(f"[aloe] waiting {args.takeover_switch_delay_sec:.3f}s before tabletop {transition}", flush=True)
    time.sleep(args.takeover_switch_delay_sec)


def _enter_tabletop_takeover(args: argparse.Namespace) -> dict[str, Any]:
    _wait_takeover_switch_delay(args, "takeover")
    return _post_takeover_active(args, active=True)


def _hold_current_auto_target(args: argparse.Namespace, raw_state: dict[str, Any], seq: int) -> int:
    payload = {
        "mode": "ee_pose_servo",
        "owner": "policy",
        "seq": seq,
        "left": {
            "joint_target": [float(value) for value in rt.joint_pos(raw_state, "left")],
            "gripper": rt.gripper_pose(raw_state, "left"),
            "preset": "free_space",
        },
        "right": {
            "joint_target": [float(value) for value in rt.joint_pos(raw_state, "right")],
            "gripper": rt.gripper_pose(raw_state, "right"),
            "preset": "free_space",
        },
    }
    rt.post_action(args.robot_server, args.timeout, payload)
    time.sleep(1.0 / args.control_hz)
    return seq + 1


def _release_tabletop_takeover(args: argparse.Namespace, seq: int) -> int:
    _wait_takeover_switch_delay(args, "release")
    latest_state = rt.get_robot_state(args.robot_server, args.timeout)
    seq = _hold_current_auto_target(args, latest_state, seq)
    release_result = _post_takeover_active(args, active=False)
    print(f"[aloe] tabletop takeover released: {release_result}")
    return seq


def _finish_tabletop_operator_exit(
    args: argparse.Namespace,
    writer: AsyncPklReplayWriter,
    correction: CorrectionBuffer,
    *,
    control_event: OperatorControlEvent,
    cfail: float,
    steps: int,
    max_steps: int,
    seq: int,
    segment_id: int,
    failed_raw_state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], int, int, TailSegmentBuffer] | None:
    if control_event.action == "save_and_end_takeover":
        if not correction.has_data:
            _resolve_operator_control_event(
                control_event,
                success=False,
                message="no_tabletop_correction_samples_to_save",
                status=409,
            )
            print("[aloe] JoyCon Y save/end rejected: no correction samples recorded yet", flush=True)
            return None
        raw_state = rt.get_robot_state(args.robot_server, args.timeout)
        print("[aloe] holding tabletop corrected endpoint before JoyCon Y release", flush=True)
        seq = _hold_current_auto_target(args, raw_state, seq)
        correction.flush(writer, cfail=cfail)
        _print_progress("tabletop correction saved", steps, max_steps)
        print("[aloe] tabletop correction saved from JoyCon Y; ending takeover")
        _resolve_operator_control_event(control_event, success=True, message="intervention_save_end")
    elif control_event.action == "discard_without_save_and_end_takeover":
        print("[aloe] holding tabletop takeover start before JoyCon X release", flush=True)
        seq = _hold_current_auto_target(args, failed_raw_state, seq)
        correction.clear()
        _print_progress("tabletop correction discarded", steps, max_steps)
        print("[aloe] tabletop correction discarded by JoyCon X; ending takeover")
        _resolve_operator_control_event(control_event, success=True, message="intervention_discard_end")
    else:
        _resolve_operator_control_event(
            control_event,
            success=False,
            message=f"invalid_tabletop_end_action:{control_event.action}",
            status=409,
        )
        raise RuntimeError(f"invalid tabletop end action: {control_event.action}")
    _wait_takeover_switch_delay(args, "operator release")
    raw_state = rt.get_robot_state(args.robot_server, args.timeout)
    if control_event.action == "discard_without_save_and_end_takeover":
        raw_state = _wait_until_robot_reaches_state(args, failed_raw_state, label="tabletop discard restore")
    obs = _obs_from_raw(args, raw_state)
    segment_id += 1
    return raw_state, obs, seq, segment_id, TailSegmentBuffer(segment_id=segment_id, source="policy")


def _apply_aloe_collector_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.experiment is None and args.aloe_config is None:
        return args
    if args.aloe_config is None and args.experiment is None:
        return args
    cfg = load_aloe_config(args.experiment or "r1lite_pack_phone", args.aloe_config)
    collector = require_dict(cfg, "collector")
    replay = require_dict(cfg, "replay")
    reward = require_dict(cfg, "reward")
    if args.replay_root is None and replay.get("root") is not None:
        args.replay_root = Path(replay["root"])
    if args.policy_host is None and collector.get("policy_host") is not None:
        args.policy_host = str(collector["policy_host"])
    if args.policy_port is None and collector.get("policy_port") is not None:
        args.policy_port = int(collector["policy_port"])
    if args.control_hz is None and collector.get("control_hz") is not None:
        args.control_hz = float(collector["control_hz"])
    if args.actions_per_infer is None and collector.get("actions_per_infer") is not None:
        args.actions_per_infer = int(collector["actions_per_infer"])
    if args.max_steps is None and collector.get("max_steps") is not None:
        args.max_steps = int(collector["max_steps"])
    if args.teleop_idle_seconds is None and collector.get("teleop_idle_seconds") is not None:
        args.teleop_idle_seconds = float(collector["teleop_idle_seconds"])
    if args.teleop_source is None and collector.get("teleop_source") is not None:
        args.teleop_source = str(collector["teleop_source"])
    if args.takeover_http_path is None and collector.get("takeover_http_path") is not None:
        args.takeover_http_path = str(collector["takeover_http_path"])
    if args.takeover_switch_delay_sec is None and collector.get("takeover_switch_delay_sec") is not None:
        args.takeover_switch_delay_sec = float(collector["takeover_switch_delay_sec"])
    if args.tabletop_release_key is None and collector.get("tabletop_release_key") is not None:
        args.tabletop_release_key = str(collector["tabletop_release_key"]).lower()
    if args.tabletop_restore_timeout_sec is None and collector.get("tabletop_restore_timeout_sec") is not None:
        args.tabletop_restore_timeout_sec = float(collector["tabletop_restore_timeout_sec"])
    if args.tabletop_restore_joint_tolerance is None and collector.get("tabletop_restore_joint_tolerance") is not None:
        args.tabletop_restore_joint_tolerance = float(collector["tabletop_restore_joint_tolerance"])
    if args.operator_event_host is None and collector.get("operator_event_host") is not None:
        args.operator_event_host = str(collector["operator_event_host"])
    if args.operator_event_port is None and collector.get("operator_event_port") is not None:
        args.operator_event_port = int(collector["operator_event_port"])
    if args.operator_event_path is None and collector.get("operator_event_path") is not None:
        args.operator_event_path = str(collector["operator_event_path"])
    if args.cfail is None and reward.get("cfail") is not None:
        args.cfail = float(reward["cfail"])
    return args


def _run_collection(args: argparse.Namespace) -> None:
    run_id, run_dir = next_run_dir(args.replay_root, args.iteration)
    client = websocket_client_policy.WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
    controller = SpaceMouseInterventionController(args) if args.teleop_source == "spacemouse" else None
    operator_events = (
        OperatorEventReceiver(
            host=args.operator_event_host, port=args.operator_event_port, path=args.operator_event_path
        )
        if args.teleop_source == "tabletop"
        else None
    )
    effective_teleop_label_source = (
        "observed_joint_delta" if args.teleop_source == "tabletop" else args.teleop_label_source
    )
    key_events = KeyEventThread()
    seq = 0
    segment_id = 0
    steps = 0
    cfail = float(args.cfail if args.cfail is not None else args.max_steps)
    policy_segment = TailSegmentBuffer(segment_id=segment_id, source="policy")
    print(f"[aloe] writing run {run_id} to {run_dir}")
    print(f"[aloe] max policy steps: {args.max_steps}")
    print(f"[aloe] teleop source: {args.teleop_source}")
    print(f"[aloe] teleop correction label source: {effective_teleop_label_source}")
    _print_progress("rollout start", steps, args.max_steps)
    with AsyncPklReplayWriter(
        run_dir,
        run_id=run_id,
        prompt=args.prompt,
        policy_checkpoint=args.policy_checkpoint,
        action_space="joint_delta",
        shard_size_chunks=args.shard_size_chunks,
        queue_size=args.writer_queue_size,
        metadata={
            "iteration": args.iteration,
            "control_hz": args.control_hz,
            "action_space": "joint_delta",
            "max_steps": args.max_steps,
            "actions_per_infer": args.actions_per_infer,
            "teleop_chunk_size": args.teleop_chunk_size,
            "teleop_source": args.teleop_source,
            "teleop_label_source": effective_teleop_label_source,
            "takeover_switch_delay_sec": args.takeover_switch_delay_sec if args.teleop_source == "tabletop" else None,
            "tabletop_restore_timeout_sec": (
                args.tabletop_restore_timeout_sec if args.teleop_source == "tabletop" else None
            ),
            "tabletop_restore_joint_tolerance": (
                args.tabletop_restore_joint_tolerance if args.teleop_source == "tabletop" else None
            ),
            "operator_event_url": (
                f"http://{args.operator_event_host}:{args.operator_event_port}{args.operator_event_path}"
                if operator_events is not None
                else None
            ),
        },
    ) as writer:
        try:
            with rt.RawTerminal():
                if operator_events is not None:
                    operator_events.start()
                key_events.start()
                raw_state = rt.get_robot_state(args.robot_server, args.timeout)
                obs = _obs_from_raw(args, raw_state)
                pending_operator_control: OperatorControlEvent | None = None
                while steps < args.max_steps:
                    operator_control = pending_operator_control or _next_operator_control_event(operator_events)
                    pending_operator_control = None
                    if operator_control is not None and operator_control.action == "takeover_start":
                        key = "operator_takeover_start"
                    elif operator_control is not None and operator_control.action == "mark_success":
                        key = "operator_mark_success"
                    elif operator_control is not None and operator_control.action == "discard_trajectory":
                        key = "operator_discard_trajectory"
                    elif operator_control is not None:
                        _resolve_operator_control_event(
                            operator_control,
                            success=False,
                            message=f"{operator_control.action}_requires_active_tabletop_correction",
                            status=409,
                        )
                        continue
                    else:
                        key = key_events.get()
                    if key in ("q", "operator_discard_trajectory"):
                        writer.mark_ignored(
                            "operator_discard_trajectory" if key == "operator_discard_trajectory" else "operator_abort"
                        )
                        _resolve_operator_control_event(
                            operator_control,
                            success=True,
                            message="trajectory_discarded",
                        )
                        _print_progress("operator abort", steps, args.max_steps)
                        print("[aloe] rollout aborted; run marked ignored")
                        return
                    if key in ("s", "operator_mark_success"):
                        policy_segment.flush(
                            writer,
                            terminal_outcome="success",
                            terminal_reason="operator_success",
                            cfail=cfail,
                        )
                        _drain_writer(writer)
                        _resolve_operator_control_event(
                            operator_control,
                            success=True,
                            message="trajectory_marked_success",
                        )
                        _print_progress("operator success", steps, args.max_steps)
                        print("[aloe] success marked; resetting robot and waiting for scene reset")
                        rt.post_reset(
                            args.robot_server,
                            args.timeout,
                            owner=args.maintenance_owner,
                            left_gripper=args.gripper_open_value,
                            right_gripper=args.gripper_open_value,
                        )
                        break
                    if key in ("f", "operator_takeover_start"):
                        started_by_operator = key == "operator_takeover_start"
                        failed_raw_state = rt.get_robot_state(args.robot_server, args.timeout)
                        failed_obs = _obs_from_raw(args, failed_raw_state)
                        raw_state = failed_raw_state
                        obs = failed_obs
                        policy_segment.flush(
                            writer,
                            terminal_outcome="failure",
                            terminal_reason="operator_failure_or_unsafe",
                            cfail=cfail,
                        )
                        _print_progress("operator failure", steps, args.max_steps)
                        print(f"[aloe] failure marked; starting {args.teleop_source} correction")
                        segment_id += 1
                        correction = CorrectionBuffer()
                        idle_start = None
                        if args.teleop_source == "tabletop":
                            if started_by_operator:
                                print("[aloe] tabletop takeover active from JoyCon event")
                                _wait_takeover_switch_delay(args, "operator takeover")
                            else:
                                enter_result = _enter_tabletop_takeover(args)
                                print(f"[aloe] tabletop takeover active: {enter_result}")
                            raw_state = rt.get_robot_state(args.robot_server, args.timeout)
                            obs = _obs_from_raw(args, raw_state)
                            print(
                                "[aloe] recording tabletop correction; JoyCon Y=save/end, JoyCon X=discard/end, "
                                "JoyCon A=success, JoyCon B=discard trajectory, "
                                f"{args.tabletop_release_key.upper()}=keyboard release, q=abort"
                            )
                            _resolve_operator_control_event(
                                operator_control,
                                success=True,
                                message="intervention_start",
                            )
                            while True:
                                operator_control = _next_operator_control_event(operator_events)
                                if operator_control is not None and operator_control.action in (
                                    "save_and_end_takeover",
                                    "discard_without_save_and_end_takeover",
                                ):
                                    finish_result = _finish_tabletop_operator_exit(
                                        args,
                                        writer,
                                        correction,
                                        control_event=operator_control,
                                        cfail=cfail,
                                        steps=steps,
                                        max_steps=args.max_steps,
                                        seq=seq,
                                        segment_id=segment_id,
                                        failed_raw_state=failed_raw_state,
                                    )
                                    if finish_result is None:
                                        continue
                                    raw_state, obs, seq, segment_id, policy_segment = finish_result
                                    break
                                if operator_control is not None and operator_control.action == "takeover_start":
                                    _resolve_operator_control_event(
                                        operator_control,
                                        success=False,
                                        message="duplicate_takeover_start_while_recording_correction",
                                        status=409,
                                    )
                                    print(
                                        "[aloe] duplicate takeover_start ignored while already recording tabletop correction",
                                        flush=True,
                                    )
                                    continue
                                if operator_control is not None and operator_control.action == "mark_success":
                                    if not correction.has_data:
                                        _resolve_operator_control_event(
                                            operator_control,
                                            success=False,
                                            message="no_tabletop_correction_samples_to_mark_success",
                                            status=409,
                                        )
                                        print(
                                            "[aloe] JoyCon A success rejected: no correction samples recorded yet",
                                            flush=True,
                                        )
                                        continue
                                    correction.flush(
                                        writer,
                                        cfail=cfail,
                                        terminal_outcome="success",
                                        terminal_reason="operator_success",
                                    )
                                    _drain_writer(writer)
                                    _resolve_operator_control_event(
                                        operator_control,
                                        success=True,
                                        message="trajectory_marked_success",
                                    )
                                    _print_progress("operator success", steps, args.max_steps)
                                    print("[aloe] success marked during tabletop correction; resetting robot")
                                    rt.post_reset(
                                        args.robot_server,
                                        args.timeout,
                                        owner=args.maintenance_owner,
                                        left_gripper=args.gripper_open_value,
                                        right_gripper=args.gripper_open_value,
                                    )
                                    return
                                if operator_control is not None and operator_control.action == "discard_trajectory":
                                    correction.clear()
                                    writer.mark_ignored("operator_discard_trajectory")
                                    _resolve_operator_control_event(
                                        operator_control,
                                        success=True,
                                        message="trajectory_discarded",
                                    )
                                    _print_progress("operator discard", steps, args.max_steps)
                                    print("[aloe] rollout discarded by JoyCon B; run marked ignored")
                                    return
                                correction_key = key_events.get()
                                if correction_key == "q":
                                    seq = _release_tabletop_takeover(args, seq)
                                    writer.mark_ignored("operator_abort")
                                    _print_progress("operator abort", steps, args.max_steps)
                                    print("[aloe] rollout aborted; run marked ignored")
                                    return
                                if correction_key == args.tabletop_release_key:
                                    if not correction.has_data:
                                        print(
                                            "[aloe] tabletop release ignored: no correction samples recorded yet",
                                            flush=True,
                                        )
                                        continue
                                    seq = _release_tabletop_takeover(args, seq)
                                    decision = _wait_save_or_discard(key_events)
                                    if decision == "enter":
                                        correction.flush(writer, cfail=cfail)
                                        _print_progress("tabletop correction saved", steps, args.max_steps)
                                        print("[aloe] tabletop correction saved; resuming policy")
                                        raw_state = rt.get_robot_state(args.robot_server, args.timeout)
                                        obs = _obs_from_raw(args, raw_state)
                                        segment_id += 1
                                        policy_segment = TailSegmentBuffer(segment_id=segment_id, source="policy")
                                        break
                                    correction.clear()
                                    seq = _restore_failure_robot_state(args, failed_raw_state, seq)
                                    print(
                                        "[aloe] tabletop correction discarded. Robot joints restored to the failed state. "
                                        "Restore moved objects if needed, then press ENTER to retry tabletop correction.",
                                        flush=True,
                                    )
                                    key_events.wait_for({"enter"})
                                    retry_result = _enter_tabletop_takeover(args)
                                    print(f"[aloe] tabletop takeover active: {retry_result}")
                                    raw_state = rt.get_robot_state(args.robot_server, args.timeout)
                                    obs = _obs_from_raw(args, raw_state)
                                    continue
                                draft, raw_state, obs, seq, tabletop_interrupt = _execute_tabletop_chunk(
                                    args,
                                    operator_events,
                                    key_events,
                                    run_id=run_id,
                                    segment_id=segment_id,
                                    raw_state=raw_state,
                                    obs=obs,
                                    seq=seq,
                                )
                                if draft is not None:
                                    correction.append(draft)
                                if isinstance(tabletop_interrupt, OperatorControlEvent):
                                    if tabletop_interrupt.action in (
                                        "save_and_end_takeover",
                                        "discard_without_save_and_end_takeover",
                                    ):
                                        finish_result = _finish_tabletop_operator_exit(
                                            args,
                                            writer,
                                            correction,
                                            control_event=tabletop_interrupt,
                                            cfail=cfail,
                                            steps=steps,
                                            max_steps=args.max_steps,
                                            seq=seq,
                                            segment_id=segment_id,
                                            failed_raw_state=failed_raw_state,
                                        )
                                        if finish_result is None:
                                            continue
                                        raw_state, obs, seq, segment_id, policy_segment = finish_result
                                        break
                                    if tabletop_interrupt.action == "takeover_start":
                                        _resolve_operator_control_event(
                                            tabletop_interrupt,
                                            success=False,
                                            message="duplicate_takeover_start_while_recording_correction",
                                            status=409,
                                        )
                                        print(
                                            "[aloe] duplicate takeover_start ignored while already recording tabletop correction",
                                            flush=True,
                                        )
                                        continue
                                    if tabletop_interrupt.action == "mark_success":
                                        if not correction.has_data:
                                            _resolve_operator_control_event(
                                                tabletop_interrupt,
                                                success=False,
                                                message="no_tabletop_correction_samples_to_mark_success",
                                                status=409,
                                            )
                                            print(
                                                "[aloe] JoyCon A success rejected: no correction samples recorded yet",
                                                flush=True,
                                            )
                                            continue
                                        correction.flush(
                                            writer,
                                            cfail=cfail,
                                            terminal_outcome="success",
                                            terminal_reason="operator_success",
                                        )
                                        _drain_writer(writer)
                                        _resolve_operator_control_event(
                                            tabletop_interrupt,
                                            success=True,
                                            message="trajectory_marked_success",
                                        )
                                        _print_progress("operator success", steps, args.max_steps)
                                        print("[aloe] success marked during tabletop correction; resetting robot")
                                        rt.post_reset(
                                            args.robot_server,
                                            args.timeout,
                                            owner=args.maintenance_owner,
                                            left_gripper=args.gripper_open_value,
                                            right_gripper=args.gripper_open_value,
                                        )
                                        return
                                    if tabletop_interrupt.action == "discard_trajectory":
                                        correction.clear()
                                        writer.mark_ignored("operator_discard_trajectory")
                                        _resolve_operator_control_event(
                                            tabletop_interrupt,
                                            success=True,
                                            message="trajectory_discarded",
                                        )
                                        _print_progress("operator discard", steps, args.max_steps)
                                        print("[aloe] rollout discarded by JoyCon B; run marked ignored")
                                        return
                                    _resolve_operator_control_event(
                                        tabletop_interrupt,
                                        success=False,
                                        message=f"unsupported_operator_action:{tabletop_interrupt.action}",
                                        status=409,
                                    )
                                    continue
                                if tabletop_interrupt == "q":
                                    seq = _release_tabletop_takeover(args, seq)
                                    writer.mark_ignored("operator_abort")
                                    _print_progress("operator abort", steps, args.max_steps)
                                    print("[aloe] rollout aborted; run marked ignored")
                                    return
                                if tabletop_interrupt == "keyboard_release":
                                    if not correction.has_data:
                                        print(
                                            "[aloe] tabletop release ignored: no correction samples recorded yet",
                                            flush=True,
                                        )
                                        continue
                                    seq = _release_tabletop_takeover(args, seq)
                                    decision = _wait_save_or_discard(key_events)
                                    if decision == "enter":
                                        correction.flush(writer, cfail=cfail)
                                        _print_progress("tabletop correction saved", steps, args.max_steps)
                                        print("[aloe] tabletop correction saved; resuming policy")
                                        raw_state = rt.get_robot_state(args.robot_server, args.timeout)
                                        obs = _obs_from_raw(args, raw_state)
                                        segment_id += 1
                                        policy_segment = TailSegmentBuffer(segment_id=segment_id, source="policy")
                                        break
                                    correction.clear()
                                    seq = _restore_failure_robot_state(args, failed_raw_state, seq)
                                    print(
                                        "[aloe] tabletop correction discarded. Robot joints restored to the failed state. "
                                        "Restore moved objects if needed, then press ENTER to retry tabletop correction.",
                                        flush=True,
                                    )
                                    key_events.wait_for({"enter"})
                                    retry_result = _enter_tabletop_takeover(args)
                                    print(f"[aloe] tabletop takeover active: {retry_result}")
                                    raw_state = rt.get_robot_state(args.robot_server, args.timeout)
                                    obs = _obs_from_raw(args, raw_state)
                                    continue
                            continue
                        if controller is None:
                            raise RuntimeError("spacemouse controller is not initialized")
                        print("[aloe] waiting for SpaceMouse input before recording correction")
                        while True:
                            correction_key = key_events.get()
                            if correction_key == "q":
                                writer.mark_ignored("operator_abort")
                                _print_progress("operator abort", steps, args.max_steps)
                                print("[aloe] rollout aborted; run marked ignored")
                                return
                            draft, raw_state, obs, seq = _execute_teleop_chunk(
                                args,
                                controller,
                                run_id=run_id,
                                segment_id=segment_id,
                                raw_state=raw_state,
                                obs=obs,
                                seq=seq,
                            )
                            if draft is not None:
                                correction.append(draft)
                                idle_start = None
                                continue
                            if not correction.has_data:
                                idle_start = None
                                time.sleep(max(0.0, 1.0 / args.control_hz))
                                continue
                            if idle_start is None:
                                idle_start = time.monotonic()
                            if idle_start is not None and time.monotonic() - idle_start >= args.teleop_idle_seconds:
                                decision = _wait_save_or_discard(key_events)
                                if decision == "enter":
                                    correction.flush(writer, cfail=cfail)
                                    _print_progress("correction saved", steps, args.max_steps)
                                    print("[aloe] correction saved; resuming policy")
                                    segment_id += 1
                                    policy_segment = TailSegmentBuffer(segment_id=segment_id, source="policy")
                                    break
                                correction.clear()
                                seq = _restore_failure_robot_state(args, failed_raw_state, seq)
                                print(
                                    "[aloe] correction discarded. Robot joints restored to the failed state. "
                                    "Restore moved objects if needed, then press ENTER to retry teleop correction.",
                                    flush=True,
                                )
                                key_events.wait_for({"enter"})
                                raw_state = failed_raw_state
                                obs = failed_obs
                                print("[aloe] waiting for SpaceMouse input before recording correction")
                                idle_start = None
                            time.sleep(max(0.0, 1.0 / args.control_hz))
                        continue

                    draft, raw_state, obs, seq, interrupt = _execute_policy_chunk(
                        args,
                        client,
                        key_events,
                        operator_events,
                        run_id=run_id,
                        segment_id=segment_id,
                        raw_state=raw_state,
                        obs=obs,
                        seq=seq,
                        remaining_steps=args.max_steps - steps,
                    )
                    if draft is not None:
                        policy_segment.append(writer, draft, cfail=cfail)
                        steps += draft.length
                        _print_progress("policy chunk complete", steps, args.max_steps)
                    if interrupt is not None:
                        if isinstance(interrupt, OperatorControlEvent):
                            pending_operator_control = interrupt
                        else:
                            key_events.put_back(interrupt)

                if steps >= args.max_steps:
                    policy_segment.flush(
                        writer,
                        terminal_outcome="timeout",
                        terminal_reason="max_steps",
                        cfail=cfail,
                    )
                    _drain_writer(writer)
                    _print_progress("max_steps reached", steps, args.max_steps)
                    print("[aloe] max_steps reached; timeout/failure terminal recorded")
        finally:
            key_events.stop()
            if operator_events is not None:
                operator_events.stop()
            if controller is not None:
                controller.close()

    rt.wait_for_enter("Recover the scene manually, then press ENTER to start the next rollout outside this process.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_args(parser, default_action_space="joint_delta")
    parser.add_argument(
        "--aloe-config", default=None, help="Path to an explicit experiments/r1lite/<experiment>/aloe.yaml."
    )
    parser.add_argument("--replay-root", type=Path, default=None)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--policy-host", default=None)
    parser.add_argument("--policy-port", type=int, default=None)
    parser.add_argument("--policy-checkpoint", default=None)
    parser.add_argument("--robot-server", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--control-hz", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--actions-per-infer", type=int, default=None)
    parser.add_argument("--teleop-chunk-size", type=int, default=None)
    parser.add_argument("--shard-size-chunks", type=int, default=64)
    parser.add_argument("--writer-queue-size", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--gripper-open-value", type=float, default=None)
    parser.add_argument("--gripper-close-value", type=float, default=None)
    parser.add_argument("--gripper-threshold", type=float, default=None)
    parser.add_argument("--left-gripper-threshold", type=float, default=None)
    parser.add_argument("--right-gripper-threshold", type=float, default=None)
    parser.add_argument("--teleop-idle-seconds", type=float, default=None)
    parser.add_argument("--teleop-source", choices=("spacemouse", "tabletop"), default=None)
    parser.add_argument("--takeover-http-path", default=None)
    parser.add_argument("--takeover-switch-delay-sec", type=float, default=None)
    parser.add_argument("--tabletop-release-key", choices=("enter", "esc"), default=None)
    parser.add_argument("--tabletop-restore-timeout-sec", type=float, default=None)
    parser.add_argument("--tabletop-restore-joint-tolerance", type=float, default=None)
    parser.add_argument("--operator-event-host", default=None)
    parser.add_argument("--operator-event-port", type=int, default=None)
    parser.add_argument("--operator-event-path", default=None)
    parser.add_argument(
        "--teleop-label-source",
        choices=("observed_joint_delta", "accepted_joint_target"),
        default="observed_joint_delta",
        help=(
            "How to label SpaceMouse correction actions. observed_joint_delta records "
            "state_after - state_before from /state/robot. accepted_joint_target requires "
            "the robot server to expose accepted joint targets in /action response or /health."
        ),
    )
    parser.add_argument("--cfail", type=float, default=None)
    parser.add_argument("--maintenance-owner", default="debug")
    add_spacemouse_intervention_args(parser)
    args = _apply_aloe_collector_config(parser.parse_args())
    args = apply_rollout_config(args)
    if args.list_spacemouse_devices:
        list_spacemouse_devices()
        raise SystemExit(0)
    fill_spacemouse_intervention_defaults(args)
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
    if args.max_steps is None:
        args.max_steps = 2000
    if args.timeout is None:
        args.timeout = 2.0
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
    if args.teleop_chunk_size is None:
        args.teleop_chunk_size = args.actions_per_infer
    if args.teleop_idle_seconds is None:
        args.teleop_idle_seconds = 1.0
    if args.teleop_source is None:
        args.teleop_source = "spacemouse"
    args.teleop_source = str(args.teleop_source).lower()
    if args.takeover_http_path is None:
        args.takeover_http_path = "/takeover/set_active"
    if args.takeover_switch_delay_sec is None:
        args.takeover_switch_delay_sec = 0.5
    if args.tabletop_release_key is None:
        args.tabletop_release_key = "enter"
    args.tabletop_release_key = str(args.tabletop_release_key).lower()
    if args.tabletop_restore_timeout_sec is None:
        args.tabletop_restore_timeout_sec = 5.0
    if args.tabletop_restore_joint_tolerance is None:
        args.tabletop_restore_joint_tolerance = 0.03
    if args.operator_event_host is None:
        args.operator_event_host = "0.0.0.0"
    if args.operator_event_port is None:
        args.operator_event_port = 18001
    if args.operator_event_path is None:
        args.operator_event_path = "/teleop/operator_event"
    args.operator_event_path = "/" + str(args.operator_event_path).strip("/")
    if args.replay_root is None:
        args.replay_root = Path("data/aloe/r1lite_pack_phone")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.actions_per_infer <= 0:
        parser.error("--actions-per-infer must be positive")
    if args.teleop_chunk_size <= 0:
        parser.error("--teleop-chunk-size must be positive")
    if args.shard_size_chunks <= 0:
        parser.error("--shard-size-chunks must be positive")
    if args.writer_queue_size <= 0:
        parser.error("--writer-queue-size must be positive")
    if args.control_hz <= 0:
        parser.error("--control-hz must be positive")
    if args.teleop_idle_seconds < 0:
        parser.error("--teleop-idle-seconds must be non-negative")
    if args.teleop_source not in ("spacemouse", "tabletop"):
        parser.error("--teleop-source must be one of: spacemouse, tabletop")
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
    if args.teleop_source == "tabletop" and args.teleop_label_source != "observed_joint_delta":
        parser.error("--teleop-source=tabletop requires --teleop-label-source=observed_joint_delta")
    return args


def main() -> None:
    try:
        _run_collection(_parse_args())
    except KeyboardInterrupt:
        print("\n[aloe] interrupted by operator", file=sys.stderr)


if __name__ == "__main__":
    main()
