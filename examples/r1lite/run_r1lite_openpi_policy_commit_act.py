#!/usr/bin/env python3
"""Run R1Lite with latency-aligned, commit-aware ACT action selection.

Each policy chunk is placed on one global 15 Hz timeline.  The first model row
belongs to the first tick after the observation, expired rows are discarded
using the measured observation-to-result delay, and overlapping predictions
for the same tick are ACT-ensembled.  A newly completed chunk may take control
only at a commit boundary, preventing every inference result from immediately
changing the robot's short-term intention.
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
import run_r1lite_openpi_policy_async as async_base
import run_r1lite_openpi_policy_smooth as base

JOINT_INDICES = np.r_[0:6, 7:13]


@dataclasses.dataclass(frozen=True)
class TimedPrediction:
    source_tick: int
    action: np.ndarray


@dataclasses.dataclass(frozen=True)
class InferenceResult:
    request_id: int
    scheduled_tick: int
    observation_time: float
    finished_time: float
    model_latency: float
    observation_to_result_latency: float
    end_to_end_latency: float
    actions: np.ndarray
    reference_joints: np.ndarray


@dataclasses.dataclass(frozen=True)
class ChunkUpdate:
    request_id: int
    observation_tick: int
    latency_steps: int
    expired_actions: int
    accepted_actions: int
    source_tick: int
    voted_left_gripper: float
    voted_right_gripper: float


@dataclasses.dataclass(frozen=True)
class CommitDecision:
    action: np.ndarray
    active_source_tick: int
    commit_until_tick: int
    candidate_count: int
    source_ticks: tuple[int, ...]
    switched: bool


class PublishedGripperVote:
    """Vote per chunk, but hold ambiguous decisions at acknowledged state."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._left = float(args.gripper_open_value)
        self._right = float(args.gripper_open_value)

    def vote(self, actions: np.ndarray) -> tuple[float, float]:
        actions = np.asarray(actions, dtype=np.float32)
        count = min(len(actions), int(self._args.gripper_vote_actions))
        if count <= 0:
            return self._left, self._right
        return (
            self._vote_one(actions[:count, 6], self._left),
            self._vote_one(actions[:count, 13], self._right),
        )

    def mark_published(self, left: float, right: float) -> None:
        self._left = float(left)
        self._right = float(right)

    def _vote_one(self, values: np.ndarray, previous: float) -> float:
        open_fraction = float(np.mean(values >= float(self._args.gripper_vote_threshold)))
        min_fraction = float(self._args.gripper_min_vote_fraction)
        if open_fraction >= min_fraction:
            return float(self._args.gripper_open_value)
        if open_fraction <= 1.0 - min_fraction:
            return float(self._args.gripper_close_value)
        return float(previous)


class ExecutedGripperFilter:
    """Require a stable discrete intent before changing each gripper state."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        initial = float(args.gripper_open_value)
        self._states = [initial, initial]
        self._pending: list[float | None] = [None, None]
        self._pending_counts = [0, 0]
        self.last_raw = (initial, initial)
        self.last_output = (initial, initial)
        self.last_pending_counts = (0, 0)
        self.last_switched = (False, False)

    def apply(self, action: np.ndarray) -> np.ndarray:
        result = np.asarray(action, dtype=np.float32).reshape(base.ACTION_DIM).copy()
        raw = (float(result[6]), float(result[13]))
        outputs = []
        switched = []
        for arm, value in enumerate(raw):
            output, did_switch = self._apply_one(arm, value)
            outputs.append(output)
            switched.append(did_switch)
        result[6], result[13] = outputs
        self.last_raw = raw
        self.last_output = (outputs[0], outputs[1])
        self.last_pending_counts = (self._pending_counts[0], self._pending_counts[1])
        self.last_switched = (switched[0], switched[1])
        return result

    def _apply_one(self, arm: int, value: float) -> tuple[float, bool]:
        threshold = float(self._args.gripper_vote_threshold)
        target = (
            float(self._args.gripper_open_value)
            if value >= threshold
            else float(self._args.gripper_close_value)
        )
        if np.isclose(target, self._states[arm]):
            self._pending[arm] = None
            self._pending_counts[arm] = 0
            return self._states[arm], False

        if self._pending[arm] is not None and np.isclose(target, self._pending[arm]):
            self._pending_counts[arm] += 1
        else:
            self._pending[arm] = target
            self._pending_counts[arm] = 1

        required = (
            int(self._args.gripper_open_confirm_steps)
            if np.isclose(target, float(self._args.gripper_open_value))
            else int(self._args.gripper_close_confirm_steps)
        )
        if self._pending_counts[arm] < required:
            return self._states[arm], False

        self._states[arm] = target
        self._pending[arm] = None
        self._pending_counts[arm] = 0
        return self._states[arm], True


class CommitAwareActBuffer:
    """ACT prediction table with an explicit minimum plan commitment window."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._period = 1.0 / float(args.control_hz)
        self._predictions: dict[int, list[TimedPrediction]] = collections.defaultdict(list)
        self._active_source_tick: int | None = None
        self._commit_until_tick = 0
        self.grippers = PublishedGripperVote(args)

    def add_chunk(
        self,
        result: InferenceResult,
        *,
        clock_start: float,
        available_tick: int,
    ) -> ChunkUpdate:
        actions = np.asarray(result.actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != base.ACTION_DIM:
            raise ValueError(f"Expected actions with shape (horizon, {base.ACTION_DIM}), got {actions.shape}")

        observation_tick = max(
            result.scheduled_tick,
            round((result.observation_time - clock_start) / self._period),
        )
        latency_steps = max(
            0,
            math.ceil(result.observation_to_result_latency * float(self._args.control_hz) - 1e-9),
        )

        # Model row i is scheduled for observation_tick + i + 1.  Therefore a
        # two-tick delay expires row 0 and starts execution from row 1.
        first_valid = max(
            0,
            latency_steps - 1,
            available_tick - observation_tick - 1,
        )
        usable_len = min(len(actions), int(self._args.actions_per_infer))
        accepted = max(0, usable_len - first_valid)
        voted_left = float(self._args.gripper_open_value)
        voted_right = float(self._args.gripper_open_value)

        if accepted > 0:
            actions = actions.copy()
            voted_left, voted_right = self.grippers.vote(actions[first_valid:usable_len])
            actions[first_valid:usable_len, 6] = voted_left
            actions[first_valid:usable_len, 13] = voted_right
            for action_index in range(first_valid, usable_len):
                target_tick = observation_tick + action_index + 1
                if target_tick >= available_tick:
                    self._predictions[target_tick].append(
                        TimedPrediction(source_tick=observation_tick, action=actions[action_index].copy())
                    )

        self.discard_before(available_tick)
        return ChunkUpdate(
            request_id=result.request_id,
            observation_tick=observation_tick,
            latency_steps=latency_steps,
            expired_actions=min(first_valid, usable_len),
            accepted_actions=accepted,
            source_tick=observation_tick,
            voted_left_gripper=voted_left,
            voted_right_gripper=voted_right,
        )

    def pop(self, target_tick: int) -> CommitDecision | None:
        self.discard_before(target_tick)
        candidates = self._predictions.pop(target_tick, None)
        if not candidates:
            return None

        source_ticks = sorted({item.source_tick for item in candidates})
        active_available = self._active_source_tick in source_ticks
        may_switch = (
            self._active_source_tick is None
            or target_tick >= self._commit_until_tick
            or not active_available
        )
        switched = False
        if may_switch:
            newest_source = source_ticks[-1]
            switched = newest_source != self._active_source_tick
            if switched:
                self._active_source_tick = newest_source
                self._commit_until_tick = target_tick + int(self._args.min_plan_commit_steps)

        assert self._active_source_tick is not None
        # While committed, predictions from newer observations wait for the next
        # boundary.  Older same-tick predictions remain useful ACT context.
        eligible = [item for item in candidates if item.source_tick <= self._active_source_tick]
        if not eligible:
            eligible = [max(candidates, key=lambda item: item.source_tick)]
            self._active_source_tick = eligible[0].source_tick
            self._commit_until_tick = target_tick + int(self._args.min_plan_commit_steps)
            switched = True

        eligible.sort(key=lambda item: item.source_tick, reverse=True)
        eligible = eligible[: int(self._args.act_max_candidates)]
        newest_eligible = eligible[0].source_tick
        weights = np.asarray(
            [
                math.exp(-float(self._args.act_weight_decay) * (newest_eligible - item.source_tick))
                for item in eligible
            ],
            dtype=np.float32,
        )
        weights /= np.sum(weights)
        joint_values = np.stack([item.action[JOINT_INDICES] for item in eligible], axis=0)
        action = eligible[0].action.copy()
        action[JOINT_INDICES] = np.sum(weights[:, None] * joint_values, axis=0)
        return CommitDecision(
            action=action,
            active_source_tick=self._active_source_tick,
            commit_until_tick=self._commit_until_tick,
            candidate_count=len(eligible),
            source_ticks=tuple(item.source_tick for item in eligible),
            switched=switched,
        )

    def discard_before(self, tick: int) -> None:
        for expired_tick in [key for key in self._predictions if key < tick]:
            del self._predictions[expired_tick]


class LatencyStats:
    def __init__(self, alpha: float) -> None:
        self._alpha = float(alpha)
        self.model_ema: float | None = None
        self.observation_to_result_ema: float | None = None

    def update(self, result: InferenceResult) -> None:
        self.model_ema = self._update(self.model_ema, result.model_latency)
        self.observation_to_result_ema = self._update(
            self.observation_to_result_ema,
            result.observation_to_result_latency,
        )

    def _update(self, previous: float | None, value: float) -> float:
        if previous is None:
            return float(value)
        return self._alpha * float(value) + (1.0 - self._alpha) * previous


def _fetch_inference_result(
    client: websocket_client_policy.WebsocketClientPolicy,
    args: argparse.Namespace,
    request_id: int,
    scheduled_tick: int,
) -> InferenceResult:
    started = time.monotonic()
    raw_state = base._get_robot_state(args.robot_server, args.timeout)  # noqa: SLF001
    # This is the real local receive time of the observation snapshot.  Decode,
    # packing, transport and inference are all included in the alignment delay.
    observation_time = time.monotonic()
    observation = base._observation(  # noqa: SLF001
        raw_state,
        args.prompt,
        args.left_gripper_threshold,
        args.right_gripper_threshold,
    )
    actions, model_latency = base._policy_actions(client, observation)  # noqa: SLF001
    finished_time = time.monotonic()
    reference_joints = np.r_[
        base._joint_pos(raw_state, "left"),  # noqa: SLF001
        base._joint_pos(raw_state, "right"),  # noqa: SLF001
    ].astype(np.float32)
    return InferenceResult(
        request_id=request_id,
        scheduled_tick=scheduled_tick,
        observation_time=observation_time,
        finished_time=finished_time,
        model_latency=model_latency,
        observation_to_result_latency=finished_time - observation_time,
        end_to_end_latency=finished_time - started,
        actions=actions,
        reference_joints=reference_joints,
    )


def _run_commit_act_loop(args: argparse.Namespace) -> None:
    client = websocket_client_policy.WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
    plan = CommitAwareActBuffer(args)
    joint_filter = async_base.ExecutedJointFilter(args)
    gripper_filter = ExecutedGripperFilter(args)
    publisher = async_base.LatestCommandPublisher(args)
    latency_stats = LatencyStats(args.latency_ema_alpha)
    soft_close_switch = base.RightSoftCloseControlPanel(
        enabled_ui=args.right_soft_close_ui,
        default_enabled=args.right_soft_close_keyboard_default_on,
    )
    args.right_soft_close_keyboard_switch = soft_close_switch

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="r1lite-commit-act")
    pending: concurrent.futures.Future[InferenceResult] | None = None
    request_id = 0
    inference_count = 0
    publish_count = 0
    publish_latency_ema: float | None = None
    steps_done = 0
    seq = 0
    holds = 0
    last_action: np.ndarray | None = None
    period = 1.0 / float(args.control_hz)
    clock_start = time.monotonic()
    global_tick = 0

    print(
        "commit-aware ACT rollout: "
        f"publish_hz={args.control_hz:.2f}, horizon={args.actions_per_infer}, "
        f"commit_steps={args.min_plan_commit_steps}, act_candidates={args.act_max_candidates}, "
        f"act_decay={args.act_weight_decay:.3f}"
    )

    try:
        soft_close_switch.start()
        publisher.start()
        while steps_done < args.max_steps:
            deadline = clock_start + global_tick * period
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            wall_tick = max(0, int((time.monotonic() - clock_start) // period))
            current_tick = max(global_tick, wall_tick)

            for completion in publisher.poll():
                if completion.error is not None:
                    raise RuntimeError(
                        f"Action publish failed for seq={completion.command.seq}: {completion.error}"
                    ) from completion.error
                joint_filter.mark_published(completion.command.action)
                # Keep the model's binary intent (0/100) as state.  The right
                # soft-close payload may be 90, but that is actuator effort and
                # must never be fed back as an open/close model decision.
                plan.grippers.mark_published(
                    float(completion.command.action[6]),
                    float(completion.command.action[13]),
                )
                publish_count += 1
                alpha = float(args.latency_ema_alpha)
                publish_latency_ema = (
                    completion.latency
                    if publish_latency_ema is None
                    else alpha * completion.latency + (1.0 - alpha) * publish_latency_ema
                )
                if args.log_every_steps > 0 and publish_count % args.log_every_steps == 0:
                    print(
                        f"[commit publish] n={publish_count} seq={completion.command.seq} "
                        f"http={completion.latency * 1000:.1f}ms "
                        f"http_ema={publish_latency_ema * 1000:.1f}ms dropped={publisher.dropped_commands}"
                    )

            if pending is not None and pending.done():
                result = pending.result()
                pending = None
                latency_stats.update(result)
                update = plan.add_chunk(result, clock_start=clock_start, available_tick=current_tick)
                inference_count += 1
                if update.accepted_actions > 0:
                    joint_filter.initialize(result.reference_joints)
                if inference_count == 1 or inference_count % args.log_every_infers == 0:
                    print(
                        f"[commit infer] n={inference_count} request={result.request_id} "
                        f"model={result.model_latency * 1000:.1f}ms "
                        f"obs_to_result={result.observation_to_result_latency * 1000:.1f}ms "
                        f"e2e={result.end_to_end_latency * 1000:.1f}ms "
                        f"obs_tick={update.observation_tick} latency_ticks={update.latency_steps} "
                        f"expired={update.expired_actions} accepted={update.accepted_actions} "
                        f"gripper_vote=({update.voted_left_gripper:.1f},{update.voted_right_gripper:.1f}) "
                        f"latency_ema={latency_stats.observation_to_result_ema * 1000:.1f}ms"
                    )

            # Latest-only inference: no observation queue is allowed to build.
            if pending is None:
                request_id += 1
                pending = executor.submit(
                    _fetch_inference_result,
                    client,
                    args,
                    request_id,
                    current_tick,
                )

            decision = plan.pop(current_tick)
            if decision is None:
                if last_action is None:
                    global_tick = current_tick + 1
                    continue
                holds += 1
                if holds > args.max_hold_steps:
                    raise RuntimeError(
                        f"No aligned ACT prediction for {holds} ticks; increase --actions-per-infer/"
                        "--max-hold-steps or inspect inference latency"
                    )
                selected = last_action.copy()
            else:
                holds = 0
                selected = decision.action

            filtered_action = joint_filter.apply(selected)
            filtered_action = gripper_filter.apply(filtered_action)
            last_action = filtered_action.copy()
            command = async_base._prepare_interpolated_command(args, filtered_action, seq)  # noqa: SLF001
            publisher.submit(command)
            if args.log_every_steps > 0 and seq % args.log_every_steps == 0:
                decision_text = "hold" if decision is None else (
                    f"source={decision.active_source_tick} commit_until={decision.commit_until_tick} "
                    f"switched={decision.switched} candidates={decision.candidate_count} "
                    f"sources={decision.source_ticks}"
                )
                print(
                    f"[commit action] seq={seq} tick={current_tick} {decision_text} holds={holds} "
                    f"raw_jump={joint_filter.last_raw_jump:.5f} "
                    f"sent_step={joint_filter.last_sent_step:.5f} "
                    f"clipped={joint_filter.last_was_clipped} dropped={publisher.dropped_commands}"
                    f" gripper_raw={gripper_filter.last_raw} "
                    f"gripper_out={gripper_filter.last_output} "
                    f"gripper_pending={gripper_filter.last_pending_counts} "
                    f"gripper_switched={gripper_filter.last_switched}"
                )
            steps_done += 1
            seq += 1
            global_tick = current_tick + 1
    except KeyboardInterrupt:
        print("interrupted by operator")
    finally:
        publisher.close()
        soft_close_switch.stop()
        if pending is not None:
            pending.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        reset = getattr(client, "reset", None)
        if callable(reset):
            reset()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--config", default=None)
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
    parser.add_argument("--gripper-vote-threshold", type=float, default=50.0)
    parser.add_argument("--gripper-min-vote-fraction", type=float, default=0.6)
    parser.add_argument("--gripper-vote-actions", type=int, default=5)
    parser.add_argument("--gripper-close-confirm-steps", type=int, default=2)
    parser.add_argument("--gripper-open-confirm-steps", type=int, default=3)
    parser.add_argument("--joint-ema-alpha", type=float, default=0.8)
    parser.add_argument("--joint-max-step", type=float, default=0.10)
    parser.add_argument("--act-weight-decay", type=float, default=0.8)
    parser.add_argument("--act-max-candidates", type=int, default=3)
    parser.add_argument("--min-plan-commit-steps", type=int, default=3)
    parser.add_argument("--max-hold-steps", type=int, default=4)
    parser.add_argument("--latency-ema-alpha", type=float, default=0.2)
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
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    args = base.apply_rollout_config(parser.parse_args())

    defaults = {
        "policy_host": "localhost",
        "policy_port": 8000,
        "robot_server": "http://127.0.0.1:8001",
        "prompt": base.DEFAULT_PROMPT,
        "control_hz": 15.0,
        "actions_per_infer": 15,
        "gripper_open_value": 100.0,
        "gripper_close_value": 0.0,
        "gripper_threshold": 75.0,
        "timeout": 2.0,
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.left_gripper_threshold is None:
        args.left_gripper_threshold = args.gripper_threshold
    if args.right_gripper_threshold is None:
        args.right_gripper_threshold = args.gripper_threshold
    args.debug = False

    if args.control_hz <= 0.0:
        parser.error("--control-hz must be positive")
    if args.actions_per_infer <= 0:
        parser.error("--actions-per-infer must be positive")
    if not 0.0 < args.joint_ema_alpha <= 1.0:
        parser.error("--joint-ema-alpha must be in (0, 1]")
    if args.joint_max_step < 0.0:
        parser.error("--joint-max-step must be non-negative")
    if args.act_weight_decay < 0.0:
        parser.error("--act-weight-decay must be non-negative")
    if args.act_max_candidates <= 0:
        parser.error("--act-max-candidates must be positive")
    if args.min_plan_commit_steps <= 0:
        parser.error("--min-plan-commit-steps must be positive")
    if args.max_hold_steps < 0:
        parser.error("--max-hold-steps must be non-negative")
    if not 0.5 <= args.gripper_min_vote_fraction <= 1.0:
        parser.error("--gripper-min-vote-fraction must be in [0.5, 1.0]")
    if args.gripper_vote_actions <= 0:
        parser.error("--gripper-vote-actions must be positive")
    if args.gripper_close_confirm_steps <= 0:
        parser.error("--gripper-close-confirm-steps must be positive")
    if args.gripper_open_confirm_steps <= 0:
        parser.error("--gripper-open-confirm-steps must be positive")
    if not 0.0 < args.latency_ema_alpha <= 1.0:
        parser.error("--latency-ema-alpha must be in (0, 1]")
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if args.execute and args.max_steps is None:
        parser.error("--max-steps is required when --execute is set")
    if args.max_steps is None:
        args.max_steps = args.actions_per_infer
        print(f"dry-run without --max-steps: running {args.max_steps} sent actions")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if not 0.0 <= args.right_soft_close_value <= args.gripper_open_value:
        parser.error("--right-soft-close-value must be between 0 and --gripper-open-value")
    return args


def main() -> None:
    _run_commit_act_loop(_parse_args())


if __name__ == "__main__":
    main()
