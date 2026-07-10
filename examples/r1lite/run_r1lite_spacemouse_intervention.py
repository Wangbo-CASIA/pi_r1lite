#!/usr/bin/env python3
"""Run SpaceMouse-only R1Lite intervention without policy inference."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any
import urllib.error
import urllib.request

from r1lite_experiment_config import add_experiment_args
from r1lite_experiment_config import apply_rollout_config
from r1lite_hg_dagger import SpaceMouseInterventionController
from r1lite_hg_dagger import add_spacemouse_intervention_args
from r1lite_hg_dagger import fill_spacemouse_intervention_defaults
from r1lite_hg_dagger import list_spacemouse_devices
from r1lite_hg_dagger import maybe_run_intervention_step


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


def _execute_or_print(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    if args.execute:
        response = _post_action(args.robot_server, args.timeout, payload)
        print(f"accepted seq={payload['seq']}: {response}")
    else:
        print(f"dry-run seq={payload['seq']}:")
        print(json.dumps(payload, indent=2, sort_keys=True))


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    fill_spacemouse_intervention_defaults(args)
    if args.robot_server is None:
        args.robot_server = "http://127.0.0.1:8001"
    if args.control_hz is None:
        args.control_hz = 10.0
    if args.timeout is None:
        args.timeout = 2.0
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.control_hz <= 0:
        parser.error("--control-hz must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
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
    return args


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_args(parser, default_action_space="delta_eef")
    parser.add_argument("--robot-server", default=None)
    parser.add_argument("--control-hz", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--execute", action="store_true", help="Send SpaceMouse commands to the robot server.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional maximum number of control ticks.")
    add_spacemouse_intervention_args(parser)
    args = apply_rollout_config(parser.parse_args())
    if args.list_spacemouse_devices:
        list_spacemouse_devices()
        raise SystemExit(0)
    return _validate_args(parser, args)


def main() -> None:
    args = _parse_args()
    controller = SpaceMouseInterventionController(args)
    steps_done = 0
    seq = 0
    print(
        "[intervene-only] SpaceMouse control ready: "
        f"arm={args.intervention_arm}, execute={args.execute}, hz={args.control_hz:.2f}, robot={args.robot_server}"
    )
    try:
        while args.max_steps is None or steps_done < args.max_steps:
            intervention = maybe_run_intervention_step(
                args,
                None,
                controller,
                _get_robot_state,
                _execute_or_print,
                steps_done,
                seq,
            )
            steps_done, seq = intervention.steps_done, intervention.seq
            if not intervention.handled:
                time.sleep(1.0 / args.control_hz)
    except KeyboardInterrupt:
        print("interrupted by operator")
    finally:
        controller.close()


if __name__ == "__main__":
    main()
