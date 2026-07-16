#!/usr/bin/env python3
"""Run an absolute-joint R1Lite policy with latency-aware ACT temporal ensembling.

The policy runs asynchronously.  Every returned action chunk is aligned to the
15 Hz control timeline using its measured observation-to-result latency.  Joint
predictions from overlapping chunks are exponentially ensembled at each global
control tick.  Grippers come from the newest contributing chunk and are not
averaged as continuous joint values.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import dataclasses
import math
import time

import numpy as np
from openpi_client import websocket_client_policy
from r1lite_experiment_config import apply_rollout_config
from r1lite_rtc import add_rtc_args
from r1lite_rtc import apply_rtc_defaults
from r1lite_rtc import validate_rtc_args
import run_r1lite_openpi_policy_smooth as base

JOINT_INDICES = np.r_[0:6, 7:13]


@dataclasses.dataclass(frozen=True)
class TimedPrediction:
    """One chunk's prediction for one global execution tick."""

    source_tick: int
    action: np.ndarray


@dataclasses.dataclass(frozen=True)
class InferenceResult:
    scheduled_tick: int
    observation_time: float
    finished_time: float
    model_latency: float
    end_to_end_latency: float
    actions: np.ndarray
    reference_joints: np.ndarray


@dataclasses.dataclass(frozen=True)
class EnsembleResult:
    action: np.ndarray
    candidate_count: int
    source_ticks: tuple[int, ...]


class LatencyAwareTemporalEnsembler:
    """Align and ensemble overlapping absolute-joint action chunks."""

    def __init__(
        self,
        *,
        control_hz: float,
        weight_decay: float,
        actions_per_infer: int,
        gripper_smoother: base.ActionChunkSmoother | None = None,
    ) -> None:
        self._control_hz = float(control_hz)
        self._period = 1.0 / self._control_hz
        self._weight_decay = float(weight_decay)
        self._actions_per_infer = int(actions_per_infer)
        self._gripper_smoother = gripper_smoother
        self._predictions: dict[int, list[TimedPrediction]] = collections.defaultdict(list)

    def add_chunk(
        self,
        result: InferenceResult,
        *,
        clock_start: float,
        available_tick: int,
    ) -> tuple[int, int, int]:
        actions = np.asarray(result.actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != base.ACTION_DIM:
            raise ValueError(f"Expected actions with shape (horizon, {base.ACTION_DIM}), got {actions.shape}")

        # Map the actual observation time onto the fixed-rate control timeline.
        observation_tick = max(
            result.scheduled_tick,
            round((result.observation_time - clock_start) / self._period),
        )

        # This is measured for every request; there is no fixed 100 ms assumption.
        latency_steps = max(0, math.ceil(result.model_latency * self._control_hz - 1e-9))

        # A result can be noticed later than its model completion.  Both sources of
        # delay are respected, and every action whose execution tick has passed is dropped.
        first_valid_index = max(latency_steps, available_tick - observation_tick)
        usable_len = min(len(actions), self._actions_per_infer)

        # Restore the smooth runner's gripper behavior on the executable part of
        # this chunk: vote once per side, fill the whole valid chunk with that
        # command, and keep the previous chunk's command when the vote is ambiguous.
        # Expired rows are deliberately excluded because they will never be executed.
        if self._gripper_smoother is not None and first_valid_index < usable_len:
            valid_chunk = self._gripper_smoother.smooth_chunk(
                actions[first_valid_index:usable_len],
                usable_len - first_valid_index,
            )
            actions = actions.copy()
            actions[first_valid_index:usable_len, 6] = valid_chunk[:, 6]
            actions[first_valid_index:usable_len, 13] = valid_chunk[:, 13]

        for action_index in range(first_valid_index, usable_len):
            target_tick = observation_tick + action_index
            if target_tick < available_tick:
                continue
            self._predictions[target_tick].append(
                TimedPrediction(
                    source_tick=observation_tick,
                    action=actions[action_index].copy(),
                )
            )

        self.discard_before(available_tick)
        added = max(0, usable_len - first_valid_index)
        return observation_tick, latency_steps, added

    def pop(self, target_tick: int) -> EnsembleResult | None:
        self.discard_before(target_tick)
        candidates = self._predictions.pop(target_tick, None)
        if not candidates:
            return None

        newest_source_tick = max(item.source_tick for item in candidates)
        weights = np.asarray(
            [math.exp(-self._weight_decay * (newest_source_tick - item.source_tick)) for item in candidates],
            dtype=np.float32,
        )
        weights /= np.sum(weights)

        joint_predictions = np.stack([item.action[JOINT_INDICES] for item in candidates], axis=0)
        ensemble_joints = np.sum(weights[:, None] * joint_predictions, axis=0)

        # Grippers are discrete commands.  Use the newest observation's prediction
        # instead of averaging open and close into a meaningless intermediate value.
        newest = max(candidates, key=lambda item: item.source_tick)
        action = newest.action.copy()
        action[JOINT_INDICES] = ensemble_joints
        return EnsembleResult(
            action=action,
            candidate_count=len(candidates),
            source_ticks=tuple(item.source_tick for item in candidates),
        )

    def discard_before(self, tick: int) -> None:
        for expired_tick in [key for key in self._predictions if key < tick]:
            del self._predictions[expired_tick]

    def reset(self) -> None:
        self._predictions.clear()
        if self._gripper_smoother is not None:
            self._gripper_smoother.reset()


class JointStepLimiter:
    """Limit each absolute joint command relative to the previous sent command."""

    def __init__(self, max_step: float) -> None:
        self._max_step = float(max_step)
        self._previous: np.ndarray | None = None

    def initialize(self, joints: np.ndarray) -> None:
        if self._previous is None:
            self._previous = np.asarray(joints, dtype=np.float32).reshape(12).copy()

    def apply(self, joints: np.ndarray) -> np.ndarray:
        joints = np.asarray(joints, dtype=np.float32).reshape(12)
        if self._previous is None or self._max_step <= 0.0:
            result = joints.copy()
        else:
            delta = np.clip(joints - self._previous, -self._max_step, self._max_step)
            result = self._previous + delta
        self._previous = result.copy()
        return result


class LatencyStats:
    def __init__(self, alpha: float) -> None:
        self._alpha = float(alpha)
        self.model_ema: float | None = None
        self.end_to_end_ema: float | None = None

    def update(self, result: InferenceResult) -> None:
        self.model_ema = self._update_one(self.model_ema, result.model_latency)
        self.end_to_end_ema = self._update_one(self.end_to_end_ema, result.end_to_end_latency)

    def _update_one(self, previous: float | None, value: float) -> float:
        if previous is None:
            return float(value)
        return self._alpha * float(value) + (1.0 - self._alpha) * previous


def _fetch_inference_result(
    client: websocket_client_policy.WebsocketClientPolicy,
    args: argparse.Namespace,
    scheduled_tick: int,
) -> InferenceResult:
    started = time.monotonic()
    raw_state = base._get_robot_state(args.robot_server, args.timeout)  # noqa: SLF001
    observation = base._observation(  # noqa: SLF001
        raw_state,
        args.prompt,
        args.left_gripper_threshold,
        args.right_gripper_threshold,
    )
    observation_time = time.monotonic()
    actions, model_latency = base._policy_actions(client, observation)  # noqa: SLF001
    finished = time.monotonic()
    reference_joints = np.r_[
        base._joint_pos(raw_state, "left"),  # noqa: SLF001
        base._joint_pos(raw_state, "right"),  # noqa: SLF001
    ]
    return InferenceResult(
        scheduled_tick=scheduled_tick,
        observation_time=observation_time,
        finished_time=finished,
        model_latency=model_latency,
        end_to_end_latency=finished - started,
        actions=actions,
        reference_joints=reference_joints.astype(np.float32),
    )


def _apply_limited_joints(action: np.ndarray, limiter: JointStepLimiter) -> np.ndarray:
    result = np.asarray(action, dtype=np.float32).reshape(base.ACTION_DIM).copy()
    result[JOINT_INDICES] = limiter.apply(result[JOINT_INDICES])
    return result


def _run_act_loop(args: argparse.Namespace) -> None:
    if args.action_space != "abs_joint":
        raise ValueError("This ACT runner currently supports only --action-space abs_joint")

    client = websocket_client_policy.WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
    gripper_smoother = base.ActionChunkSmoother(args)
    ensembler = LatencyAwareTemporalEnsembler(
        control_hz=args.control_hz,
        weight_decay=args.act_weight_decay,
        actions_per_infer=args.actions_per_infer,
        gripper_smoother=gripper_smoother,
    )
    limiter = JointStepLimiter(args.joint_max_step)
    latency_stats = LatencyStats(args.latency_ema_alpha)
    soft_close_switch = base.RightSoftCloseControlPanel(
        enabled_ui=args.right_soft_close_ui,
        default_enabled=args.right_soft_close_keyboard_default_on,
    )
    args.right_soft_close_keyboard_switch = soft_close_switch

    period = 1.0 / args.control_hz
    clock_start = time.monotonic()
    global_tick = 0
    seq = 0
    steps_done = 0
    pending: concurrent.futures.Future[InferenceResult] | None = None
    last_action: np.ndarray | None = None
    consecutive_holds = 0
    inference_count = 0

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="r1lite-act-infer")
    soft_close_switch.start()
    print(
        "ACT rollout: "
        f"hz={args.control_hz:.2f}, horizon={args.actions_per_infer}, "
        f"weight_decay={args.act_weight_decay:.3f}, joint_max_step={args.joint_max_step:.4f}, "
        f"gripper_vote={args.gripper_vote_threshold:.1f}/{args.gripper_min_vote_fraction:.2f}"
    )

    try:
        while steps_done < args.max_steps:
            deadline = clock_start + global_tick * period
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)

            # If HTTP execution made the loop late, align to wall-clock time rather
            # than pretending that missed 15 Hz ticks still happened on schedule.
            wall_tick = max(0, int((time.monotonic() - clock_start) // period))
            current_tick = max(global_tick, wall_tick)
            if current_tick > global_tick:
                ensembler.discard_before(current_tick)
                global_tick = current_tick

            if pending is not None and pending.done():
                inference = pending.result()
                pending = None
                limiter.initialize(inference.reference_joints)
                latency_stats.update(inference)
                observation_tick, latency_steps, added = ensembler.add_chunk(
                    inference,
                    clock_start=clock_start,
                    available_tick=current_tick,
                )
                inference_count += 1
                if inference_count % args.log_every_infers == 0 or inference_count == 1:
                    print(
                        f"[ACT infer] n={inference_count} obs_tick={observation_tick} now={current_tick} "
                        f"model={inference.model_latency * 1000:.1f}ms "
                        f"e2e={inference.end_to_end_latency * 1000:.1f}ms "
                        f"latency_steps={latency_steps} added={added} "
                        f"model_ema={latency_stats.model_ema * 1000:.1f}ms"
                    )

            # Keep exactly one inference in flight.  The next request starts as
            # soon as the previous measured request completes; no fixed latency or
            # fixed inference-rate assumption is used.
            if pending is None:
                pending = executor.submit(_fetch_inference_result, client, args, current_tick)

            ensemble = ensembler.pop(current_tick)
            if ensemble is not None:
                action = _apply_limited_joints(ensemble.action, limiter)
                last_action = action.copy()
                consecutive_holds = 0
                if args.log_every_steps > 0 and seq % args.log_every_steps == 0:
                    print(
                        f"[ACT action] seq={seq} tick={current_tick} candidates={ensemble.candidate_count} "
                        f"sources={ensemble.source_ticks}"
                    )
            elif last_action is not None:
                consecutive_holds += 1
                if consecutive_holds > args.max_hold_steps:
                    raise RuntimeError(
                        f"ACT buffer had no action for {consecutive_holds} consecutive ticks; "
                        "policy inference is too slow or the action horizon is too short"
                    )
                action = last_action.copy()
                print(f"[ACT hold] tick={current_tick} consecutive={consecutive_holds}")
            else:
                # Startup warm-up: the robot receives no command until the first
                # measured-latency-aligned policy action becomes available.
                global_tick = current_tick + 1
                continue

            left_reference = action[:6].copy()
            right_reference = action[7:13].copy()
            base._next_action_record(  # noqa: SLF001
                args,
                action,
                seq,
                left_reference,
                right_reference,
                include_gripper_command=True,
            )
            seq += 1
            steps_done += 1
            global_tick = current_tick + 1
    except KeyboardInterrupt:
        print("interrupted by operator")
    finally:
        soft_close_switch.stop()
        if pending is not None:
            pending.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        reset = getattr(client, "reset", None)
        if callable(reset):
            reset()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=None, help="Experiment name under experiments/r1lite/<name>.")
    parser.add_argument("--config", default=None, help="Path to an explicit R1Lite experiment config.yaml.")
    parser.add_argument("--action-space", default="abs_joint", choices=("abs_joint",))
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
    parser.add_argument("--no-gripper-smoothing", dest="smooth_grippers", action="store_false")
    parser.set_defaults(smooth_actions=False, smooth_grippers=True)
    parser.add_argument("--gripper-vote-threshold", type=float, default=50.0)
    parser.add_argument("--gripper-min-vote-fraction", type=float, default=0.6)
    parser.add_argument("--joint-max-step", type=float, default=0.053)
    parser.add_argument("--act-weight-decay", type=float, default=0.25)
    parser.add_argument("--latency-ema-alpha", type=float, default=0.2)
    parser.add_argument("--max-hold-steps", type=int, default=4)
    parser.add_argument("--log-every-infers", type=int, default=10)
    parser.add_argument("--log-every-steps", type=int, default=15)
    parser.add_argument("--right-soft-close-start-step", type=int, default=None)
    parser.add_argument("--right-soft-close-end-step", type=int, default=None)
    parser.add_argument("--right-soft-close-value", type=float, default=90.0)
    parser.add_argument("--no-right-soft-close-ui", dest="right_soft_close_ui", action="store_false")
    parser.set_defaults(right_soft_close_ui=True)
    parser.add_argument(
        "--right-soft-close-default-on",
        dest="right_soft_close_keyboard_default_on",
        action="store_true",
    )
    parser.add_argument("--right-soft-close-keyboard-default-on", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    add_rtc_args(parser)
    args = apply_rollout_config(parser.parse_args())

    if args.policy_host is None:
        args.policy_host = "localhost"
    if args.policy_port is None:
        args.policy_port = 8000
    if args.robot_server is None:
        args.robot_server = "http://127.0.0.1:8001"
    if args.prompt is None:
        args.prompt = base.DEFAULT_PROMPT
    if args.control_hz is None:
        args.control_hz = 15.0
    if args.actions_per_infer is None:
        args.actions_per_infer = 10
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

    args.debug = False
    apply_rtc_defaults(args)
    validate_rtc_args(parser, args)

    if args.rtc:
        parser.error("This runner implements ACT temporal ensembling; use --no-rtc")
    if args.action_space != "abs_joint":
        parser.error("This runner currently supports only abs_joint actions")
    if args.control_hz <= 0.0:
        parser.error("--control-hz must be positive")
    if args.actions_per_infer <= 0:
        parser.error("--actions-per-infer must be positive")
    if args.joint_max_step < 0.0:
        parser.error("--joint-max-step must be non-negative")
    if not 0.0 <= args.gripper_vote_threshold <= 100.0:
        parser.error("--gripper-vote-threshold must be in [0, 100]")
    if not 0.5 <= args.gripper_min_vote_fraction <= 1.0:
        parser.error("--gripper-min-vote-fraction must be in [0.5, 1.0]")
    if args.act_weight_decay < 0.0:
        parser.error("--act-weight-decay must be non-negative")
    if not 0.0 < args.latency_ema_alpha <= 1.0:
        parser.error("--latency-ema-alpha must be in (0, 1]")
    if args.max_hold_steps < 0:
        parser.error("--max-hold-steps must be non-negative")
    if args.log_every_infers <= 0:
        parser.error("--log-every-infers must be positive")
    if args.log_every_steps < 0:
        parser.error("--log-every-steps must be non-negative")
    if args.execute and args.max_steps is None:
        parser.error("--max-steps is required when --execute is set")
    if args.max_steps is None:
        args.max_steps = args.actions_per_infer
        print(f"dry-run without --max-steps: running {args.max_steps} sent actions")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if not 0.0 <= args.right_soft_close_value <= args.gripper_open_value:
        parser.error("--right-soft-close-value must be between 0 and --gripper-open-value")
    return args


def main() -> None:
    _run_act_loop(_parse_args())


if __name__ == "__main__":
    main()
