#!/usr/bin/env python3
"""Run the R1Lite policy with timestamp-aligned asynchronous execution.

The policy predicts 15 Hz absolute-joint action chunks.  Every chunk is placed
on the same monotonic clock used by the execution loop:

* observation time is stamped immediately after the local system receives the
  robot snapshot;
* action ``i`` is timestamped at ``observation_time + (i + 1) / 15``;
* each completed chunk atomically replaces the previous chunk;
* immediately before every command is published, all 14 action dimensions
  (12 joints and 2 grippers) are linearly interpolated on the latest chunk at
  the current monotonic time.

It intentionally uses the normal WebsocketClientPolicy protocol.  It does not
require the RTC policy client or ROS topics used by the reference async scripts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import queue
import threading
import time

import numpy as np
from openpi_client import websocket_client_policy
import run_r1lite_openpi_policy_smooth as base

MODEL_ACTION_HZ = 15.0
MODEL_ACTION_DT = 1.0 / MODEL_ACTION_HZ
GRIPPER_VOTE_ACTIONS = 10
JOINT_INDICES = np.r_[0:6, 7:13]


@dataclasses.dataclass(frozen=True)
class InferenceResult:
    """One model result placed on the local monotonic timeline."""

    request_id: int
    generation: int
    observation_time: float
    finished_time: float
    model_latency: float
    end_to_end_latency: float
    actions: np.ndarray
    action_times: np.ndarray


@dataclasses.dataclass(frozen=True)
class PlanUpdate:
    request_id: int
    accepted_actions: int
    expired_actions: int
    first_action_time: float
    last_action_time: float
    voted_left_gripper: float
    voted_right_gripper: float
    blended_actions: int


@dataclasses.dataclass(frozen=True)
class InterpolatedAction:
    action: np.ndarray
    left_index: int
    right_index: int
    left_time: float
    right_time: float
    alpha: float
    before_start: bool
    after_end: bool


def interp_linear(q0: np.ndarray, q1: np.ndarray, t0: float, t1: float, t: float) -> np.ndarray:
    """Linearly interpolate every action dimension between two timestamps."""
    q0 = np.asarray(q0)
    q1 = np.asarray(q1)
    if t1 <= t0:
        raise ValueError(f"Interpolation timestamps must increase, got t0={t0:.9f}, t1={t1:.9f}")
    delta = q1 - q0
    alpha = (t - t0) / (t1 - t0)
    if isinstance(alpha, np.ndarray) and isinstance(delta, np.ndarray) and alpha.ndim < delta.ndim:
        alpha.shape = alpha.shape + (1,) * (delta.ndim - alpha.ndim)
    update = alpha * delta
    return q0 + update.astype(q0.dtype)


class LatestTimedActionPlan:
    """Latest complete action chunk and its absolute monotonic timestamps."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._request_id: int | None = None
        self._actions: np.ndarray | None = None
        self._action_times: np.ndarray | None = None
        self._last_published_left_gripper: float | None = None
        self._last_published_right_gripper: float | None = None

    def integrate(self, result: InferenceResult, *, now: float) -> PlanUpdate:
        actions = np.asarray(result.actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != base.ACTION_DIM:
            raise ValueError(f"Expected actions with shape (horizon, {base.ACTION_DIM}), got {actions.shape}")

        usable_len = min(len(actions), int(self._args.actions_per_infer))
        if usable_len <= 0:
            raise ValueError("Policy returned no usable actions")
        action_times = np.asarray(result.action_times[:usable_len], dtype=np.float64)
        if action_times.shape != (usable_len,):
            raise ValueError(f"Expected {usable_len} action timestamps, got {action_times.shape}")
        if not np.all(np.diff(action_times) > 0.0):
            raise ValueError(f"Action timestamps must be strictly increasing: {action_times}")

        expired_actions = int(np.searchsorted(action_times, now, side="right"))
        # Include the left interpolation endpoint in the voting window, while
        # excluding older rows that can no longer affect any published action.
        vote_start = max(0, min(expired_actions - 1, usable_len - 1))
        vote_end = min(usable_len, vote_start + GRIPPER_VOTE_ACTIONS)
        voted_left = self._vote_gripper(
            actions[vote_start:vote_end, 6],
            self._last_published_left_gripper,
        )
        voted_right = self._vote_gripper(
            actions[vote_start:vote_end, 13],
            self._last_published_right_gripper,
        )
        actions = actions[:usable_len].copy()
        actions[:, 6] = voted_left
        actions[:, 13] = voted_right

        # Blend only the beginning of a new chunk against the previous chunk at
        # the same absolute timestamps.  This removes replacement discontinuity
        # without building an action backlog or changing the model time axis.
        blend_len = 0
        if self._actions is not None and self._action_times is not None:
            blend_len = min(int(self._args.plan_blend_steps), usable_len)
            for index in range(blend_len):
                old_action = self._sample_arrays(self._actions, self._action_times, float(action_times[index]))
                new_weight = float(index + 1) / float(blend_len + 1)
                old_weight = 1.0 - new_weight
                actions[index, JOINT_INDICES] = (
                    old_weight * old_action[JOINT_INDICES] + new_weight * actions[index, JOINT_INDICES]
                )

        # Latest-result replacement is intentional.  We retain expired leading
        # rows so sample() can locate the continuous-time position directly.
        self._request_id = result.request_id
        self._actions = actions
        self._action_times = action_times.copy()
        return PlanUpdate(
            request_id=result.request_id,
            accepted_actions=usable_len,
            expired_actions=expired_actions,
            first_action_time=float(action_times[0]),
            last_action_time=float(action_times[-1]),
            voted_left_gripper=voted_left,
            voted_right_gripper=voted_right,
            blended_actions=blend_len,
        )

    def sample(self, now: float) -> InterpolatedAction | None:
        if self._actions is None or self._action_times is None:
            return None

        actions = self._actions
        times = self._action_times
        if now <= times[0]:
            return InterpolatedAction(
                action=actions[0].copy(),
                left_index=0,
                right_index=0,
                left_time=float(times[0]),
                right_time=float(times[0]),
                alpha=0.0,
                before_start=True,
                after_end=False,
            )
        if now >= times[-1]:
            last = len(times) - 1
            return InterpolatedAction(
                action=actions[last].copy(),
                left_index=last,
                right_index=last,
                left_time=float(times[last]),
                right_time=float(times[last]),
                alpha=1.0,
                before_start=False,
                after_end=True,
            )

        right = int(np.searchsorted(times, now, side="right"))
        left = right - 1
        t0 = float(times[left])
        t1 = float(times[right])
        alpha = (now - t0) / (t1 - t0)
        action = interp_linear(actions[left], actions[right], t0, t1, now)
        return InterpolatedAction(
            action=np.asarray(action, dtype=np.float32),
            left_index=left,
            right_index=right,
            left_time=t0,
            right_time=t1,
            alpha=float(alpha),
            before_start=False,
            after_end=False,
        )

    def reset(self) -> None:
        self._request_id = None
        self._actions = None
        self._action_times = None
        self._last_published_left_gripper = None
        self._last_published_right_gripper = None

    def mark_published(self, left_gripper: float, right_gripper: float) -> None:
        """Advance cross-chunk gripper state only after a command is published."""
        self._last_published_left_gripper = float(left_gripper)
        self._last_published_right_gripper = float(right_gripper)

    def __len__(self) -> int:
        return 0 if self._actions is None else len(self._actions)

    def _vote_gripper(self, values: np.ndarray, previous: float | None) -> float:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if len(values) == 0:
            raise ValueError("Cannot vote on an empty gripper action segment")
        if previous is None:
            previous = float(self._args.gripper_open_value)
        open_fraction = float(np.mean(values >= float(self._args.gripper_vote_threshold)))
        minimum = float(self._args.gripper_min_vote_fraction)
        if open_fraction >= minimum:
            return float(self._args.gripper_open_value)
        if open_fraction <= 1.0 - minimum:
            return float(self._args.gripper_close_value)
        return float(previous)

    @staticmethod
    def _sample_arrays(actions: np.ndarray, times: np.ndarray, now: float) -> np.ndarray:
        if now <= times[0]:
            return actions[0].copy()
        if now >= times[-1]:
            return actions[-1].copy()
        right = int(np.searchsorted(times, now, side="right"))
        left = right - 1
        return np.asarray(
            interp_linear(actions[left], actions[right], float(times[left]), float(times[right]), now),
            dtype=np.float32,
        )


class LatencyStats:
    def __init__(self, alpha: float) -> None:
        self._alpha = float(alpha)
        self.model_ema: float | None = None
        self.end_to_end_ema: float | None = None

    def update(self, result: InferenceResult) -> None:
        self.model_ema = self._update(self.model_ema, result.model_latency)
        self.end_to_end_ema = self._update(self.end_to_end_ema, result.end_to_end_latency)

    def _update(self, previous: float | None, value: float) -> float:
        if previous is None:
            return float(value)
        return self._alpha * float(value) + (1.0 - self._alpha) * previous


@dataclasses.dataclass(frozen=True)
class PreparedCommand:
    seq: int
    action: np.ndarray
    payload: dict
    left_gripper: float
    right_gripper: float


@dataclasses.dataclass(frozen=True)
class PublishCompletion:
    command: PreparedCommand
    latency: float
    error: BaseException | None


class LatestCommandPublisher:
    """Publish robot commands off-thread, retaining at most one unsent command."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._condition = threading.Condition()
        self._pending: PreparedCommand | None = None
        self._closing = False
        self._completions: queue.SimpleQueue[PublishCompletion] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, name="r1lite-action-publisher", daemon=True)
        self.dropped_commands = 0

    def start(self) -> None:
        self._thread.start()

    def submit(self, command: PreparedCommand) -> None:
        with self._condition:
            if self._closing:
                raise RuntimeError("Action publisher is closing")
            if self._pending is not None:
                self.dropped_commands += 1
            self._pending = command
            self._condition.notify()

    def poll(self) -> list[PublishCompletion]:
        completed = []
        while True:
            try:
                completed.append(self._completions.get_nowait())
            except queue.Empty:
                return completed

    def close(self) -> None:
        with self._condition:
            self._closing = True
            if self._pending is not None:
                self.dropped_commands += 1
                self._pending = None
            self._condition.notify()
        self._thread.join(timeout=float(self._args.timeout) + 1.0)
        if self._thread.is_alive():
            raise RuntimeError("Action publisher did not stop before its HTTP timeout")

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closing:
                    self._condition.wait()
                if self._pending is None and self._closing:
                    return
                command = self._pending
                self._pending = None

            assert command is not None
            started = time.monotonic()
            error = None
            try:
                if self._args.execute:
                    base._post_action(self._args.robot_server, self._args.timeout, command.payload)  # noqa: SLF001
                else:
                    base._execute_or_print(self._args, command.payload)  # noqa: SLF001
            except BaseException as exc:  # Surface worker failures in the control loop.
                error = exc
            self._completions.put(
                PublishCompletion(
                    command=command,
                    latency=time.monotonic() - started,
                    error=error,
                )
            )


def _fetch_inference_result(
    client: websocket_client_policy.WebsocketClientPolicy,
    args: argparse.Namespace,
    request_id: int,
    generation: int,
) -> InferenceResult:
    started = time.monotonic()
    raw_state = base._get_robot_state(args.robot_server, args.timeout)  # noqa: SLF001
    # Timestamp the snapshot before image decoding/observation packing so that
    # those costs are also reflected when stale action rows are discarded.
    observation_time = time.monotonic()
    observation = base._observation(  # noqa: SLF001
        raw_state,
        args.prompt,
        args.left_gripper_threshold,
        args.right_gripper_threshold,
    )
    actions, model_latency = base._policy_actions(client, observation)  # noqa: SLF001
    finished_time = time.monotonic()
    action_times = observation_time + (np.arange(len(actions), dtype=np.float64) + 1.0) * MODEL_ACTION_DT
    return InferenceResult(
        request_id=request_id,
        generation=generation,
        observation_time=observation_time,
        finished_time=finished_time,
        model_latency=model_latency,
        end_to_end_latency=finished_time - started,
        actions=actions,
        action_times=action_times,
    )


def _prepare_interpolated_command(
    args: argparse.Namespace,
    action: np.ndarray,
    seq: int,
) -> PreparedCommand:
    """Build one interpolated command and apply the final soft-close override."""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (base.ACTION_DIM,):
        raise ValueError(f"Expected one action with {base.ACTION_DIM} values, got {action.shape}")
    left_gripper = float(action[6])
    right_gripper = base._apply_right_soft_close_rule(  # noqa: SLF001
        seq,
        float(action[13]),
        args.gripper_close_value,
        args.right_soft_close_start_step,
        args.right_soft_close_end_step,
        args.right_soft_close_value,
        force_enabled=base._right_soft_close_keyboard_enabled(args),  # noqa: SLF001
    )
    payload = {
        "mode": "ee_pose_servo",
        "owner": "policy",
        "seq": seq,
        "left": {
            "joint_target": [float(value) for value in action[:6]],
            "gripper": left_gripper,
            "preset": "free_space",
        },
        "right": {
            "joint_target": [float(value) for value in action[7:13]],
            "gripper": right_gripper,
            "preset": "free_space",
        },
    }
    return PreparedCommand(
        seq=seq,
        action=action.copy(),
        payload=payload,
        left_gripper=left_gripper,
        right_gripper=right_gripper,
    )


def _run_async_loop(args: argparse.Namespace) -> None:
    if args.action_space != "abs_joint":
        raise ValueError("The async runner currently supports only --action-space abs_joint")

    client = websocket_client_policy.WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
    plan = LatestTimedActionPlan(args)
    latency_stats = LatencyStats(args.latency_ema_alpha)
    command_publisher = LatestCommandPublisher(args)
    soft_close_switch = base.RightSoftCloseControlPanel(
        enabled_ui=args.right_soft_close_ui,
        default_enabled=args.right_soft_close_keyboard_default_on,
    )
    args.right_soft_close_keyboard_switch = soft_close_switch

    intervention_runtime = None
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="r1lite-async-infer")
    pending: concurrent.futures.Future[InferenceResult] | None = None
    request_id = 0
    generation = 0
    inference_count = 0
    publish_count = 0
    publish_latency_ema: float | None = None
    steps_done = 0
    seq = 0
    consecutive_holds = 0
    stop_status: str | None = None
    stop_error: str | None = None

    period = 1.0 / float(args.control_hz)
    clock_start = time.monotonic()
    global_tick = 0

    print(
        "async timestamp-interpolated rollout: "
        f"publish_hz={args.control_hz:.2f}, model_action_hz={MODEL_ACTION_HZ:.2f}, "
        f"horizon={args.actions_per_infer}, max_hold_steps={args.max_hold_steps}"
    )

    try:
        soft_close_switch.start()
        command_publisher.start()
        if args.intervene:
            intervention_runtime = base.HgDaggerInterventionRuntime(args, control_script=__file__)
            intervention_runtime.start()

        while steps_done < args.max_steps:
            deadline = clock_start + global_tick * period
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)

            # Do not replay missed ticks after a slow HTTP request.  Rejoin the
            # wall-clock timeline and continue with exactly one current command.
            wall_tick = max(0, int((time.monotonic() - clock_start) // period))
            current_tick = max(global_tick, wall_tick)

            for completion in command_publisher.poll():
                if completion.error is not None:
                    raise RuntimeError(
                        f"Action publish failed for seq={completion.command.seq}: {completion.error}"
                    ) from completion.error
                plan.mark_published(
                    completion.command.left_gripper,
                    completion.command.right_gripper,
                )
                publish_count += 1
                if publish_latency_ema is None:
                    publish_latency_ema = completion.latency
                else:
                    alpha = float(args.latency_ema_alpha)
                    publish_latency_ema = alpha * completion.latency + (1.0 - alpha) * publish_latency_ema
                if args.log_every_steps > 0 and publish_count % args.log_every_steps == 0:
                    print(
                        f"[async publish] n={publish_count} seq={completion.command.seq} "
                        f"http={completion.latency * 1000:.1f}ms "
                        f"http_ema={publish_latency_ema * 1000:.1f}ms "
                        f"dropped={command_publisher.dropped_commands}"
                    )

            if intervention_runtime is not None:
                intervention = intervention_runtime.maybe_run_step(
                    base._get_robot_state,  # noqa: SLF001
                    base._execute_or_print,  # noqa: SLF001
                    steps_done,
                    seq,
                )
                steps_done, seq = intervention.steps_done, intervention.seq
                if intervention.handled or intervention.released:
                    generation += 1
                    plan.reset()
                    consecutive_holds = 0
                    global_tick = current_tick + 1
                    continue

            if pending is not None and pending.done():
                result = pending.result()
                pending = None
                if result.generation != generation:
                    print(
                        f"[async stale] request={result.request_id} "
                        f"result_generation={result.generation} current_generation={generation}"
                    )
                else:
                    latency_stats.update(result)
                    update = plan.integrate(result, now=time.monotonic())
                    inference_count += 1
                    if inference_count == 1 or inference_count % args.log_every_infers == 0:
                        print(
                            f"[async infer] n={inference_count} request={update.request_id} "
                            f"model={result.model_latency * 1000:.1f}ms "
                            f"e2e={result.end_to_end_latency * 1000:.1f}ms "
                            f"obs_t={result.observation_time:.6f} first_t={update.first_action_time:.6f} "
                            f"last_t={update.last_action_time:.6f} expired={update.expired_actions} "
                            f"actions={update.accepted_actions} "
                            f"gripper_vote=({update.voted_left_gripper:.1f},{update.voted_right_gripper:.1f}) "
                            f"blended={update.blended_actions} "
                            f"model_ema={latency_stats.model_ema * 1000:.1f}ms"
                        )

            # Only one request may use the WebSocket client at a time.  Starting
            # immediately after completion naturally gives the newest available
            # robot observation without building an observation backlog.
            if pending is None:
                request_id += 1
                pending = executor.submit(
                    _fetch_inference_result,
                    client,
                    args,
                    request_id,
                    generation,
                )

            # Sample as late as possible: every command is computed from the
            # current monotonic time and the newest completed model chunk.
            publish_time = time.monotonic()
            sampled = plan.sample(publish_time)
            if sampled is None:
                # Startup is deliberately command-free until the first plan.
                global_tick = current_tick + 1
                continue
            if sampled.after_end:
                consecutive_holds += 1
                if consecutive_holds > args.max_hold_steps:
                    raise RuntimeError(
                        f"Latest timed action chunk ended {consecutive_holds} control ticks ago; "
                        "increase --actions-per-infer/--max-hold-steps or reduce inference latency"
                    )
            else:
                consecutive_holds = 0

            command = _prepare_interpolated_command(args, sampled.action, seq)
            command_publisher.submit(command)
            if args.log_every_steps > 0 and seq % args.log_every_steps == 0:
                print(
                    f"[async action] seq={seq} tick={current_tick} now={publish_time:.6f} "
                    f"points=({sampled.left_index},{sampled.right_index}) "
                    f"times=({sampled.left_time:.6f},{sampled.right_time:.6f}) "
                    f"alpha={sampled.alpha:.4f} before={sampled.before_start} "
                    f"after={sampled.after_end} holds={consecutive_holds} "
                    f"publish_dropped={command_publisher.dropped_commands}"
                )
            steps_done += 1
            seq += 1
            global_tick = current_tick + 1

        stop_status = "completed"
    except KeyboardInterrupt:
        stop_status = "interrupted"
        print("interrupted by operator")
    except Exception as exc:
        stop_status = "failed"
        stop_error = repr(exc)
        raise
    finally:
        command_publisher.close()
        soft_close_switch.stop()
        if intervention_runtime is not None:
            intervention_runtime.close(status=stop_status or "failed", error=stop_error)
        if pending is not None:
            pending.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        reset = getattr(client, "reset", None)
        if callable(reset):
            reset()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Keep this local until r1lite_experiment_config.add_experiment_args fixes
    # its accidentally concatenated "abs_joint" "abs_eef" choices entry.
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
    parser.add_argument("--no-action-smoothing", dest="smooth_actions", action="store_false")
    parser.set_defaults(smooth_actions=True)
    parser.add_argument("--chunk-smooth-window", type=int, default=3)
    parser.add_argument("--joint-ema-alpha", type=float, default=0.65)
    parser.add_argument("--joint-max-step", type=float, default=0.08)
    parser.add_argument("--no-gripper-smoothing", dest="smooth_grippers", action="store_false")
    parser.set_defaults(smooth_grippers=True)
    parser.add_argument("--gripper-vote-threshold", type=float, default=50.0)
    parser.add_argument("--gripper-min-vote-fraction", type=float, default=0.6)
    parser.add_argument("--plan-blend-steps", type=int, default=3)
    parser.add_argument("--max-hold-steps", type=int, default=4)
    parser.add_argument("--max-empty-infers", type=int, default=3)
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
    parser.add_argument("--right-soft-close-keyboard-default-on", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    base.add_hg_dagger_args(parser)
    args = base.apply_rollout_config(parser.parse_args())

    if args.list_spacemouse_devices:
        base.list_spacemouse_devices()
        raise SystemExit(0)
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
    base.validate_hg_dagger_args(parser, args)
    if args.action_space != "abs_joint":
        parser.error("This runner currently supports only --action-space abs_joint")
    if args.control_hz <= 0.0:
        parser.error("--control-hz must be positive")
    if args.actions_per_infer <= 0:
        parser.error("--actions-per-infer must be positive")
    if args.chunk_smooth_window <= 0 or args.chunk_smooth_window % 2 == 0:
        parser.error("--chunk-smooth-window must be a positive odd number")
    if not 0.0 < args.joint_ema_alpha <= 1.0:
        parser.error("--joint-ema-alpha must be in (0, 1]")
    if args.joint_max_step < 0.0:
        parser.error("--joint-max-step must be non-negative")
    if not 0.0 <= args.gripper_vote_threshold <= 100.0:
        parser.error("--gripper-vote-threshold must be in [0, 100]")
    if not 0.5 <= args.gripper_min_vote_fraction <= 1.0:
        parser.error("--gripper-min-vote-fraction must be in [0.5, 1.0]")
    if args.plan_blend_steps < 0:
        parser.error("--plan-blend-steps must be non-negative")
    if args.max_hold_steps < 0:
        parser.error("--max-hold-steps must be non-negative")
    if args.max_empty_infers < 0:
        parser.error("--max-empty-infers must be non-negative")
    if not 0.0 < args.latency_ema_alpha <= 1.0:
        parser.error("--latency-ema-alpha must be in (0, 1]")
    if args.log_every_infers <= 0:
        parser.error("--log-every-infers must be positive")
    if args.log_every_steps < 0:
        parser.error("--log-every-steps must be non-negative")
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
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
    _run_async_loop(_parse_args())


if __name__ == "__main__":
    main()
