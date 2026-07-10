#!/usr/bin/env python3
"""Run one explicit ALOE training iteration from an existing replay buffer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any

from openpi.aloe.config import load_aloe_config
from openpi.aloe.config import require_dict
from openpi.aloe.dataset import replay_summary


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="r1lite_pack_phone")
    parser.add_argument("--config", default=None)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("runs/aloe"))
    parser.add_argument("--policy-host", default=None)
    parser.add_argument("--policy-port", type=int, default=None)
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--collect-target-successes", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    cfg = load_aloe_config(args.experiment, args.config)
    replay_cfg = require_dict(cfg, "replay")
    critic_cfg = require_dict(cfg, "critic")
    actor_cfg = require_dict(cfg, "actor")
    collector_cfg = require_dict(cfg, "collector")
    replay_root = args.replay_root or Path(replay_cfg["root"])
    run_root = args.output_root / args.experiment / f"iteration_{args.iteration:03d}"
    critic_dir = run_root / "critic"
    actor_dir = run_root / "actor"
    if not args.dry_run:
        run_root.mkdir(parents=True, exist_ok=False)

    policy_host = args.policy_host or str(collector_cfg.get("policy_host", "localhost"))
    policy_port = int(args.policy_port or collector_cfg.get("policy_port", 8000))
    pi05_base_checkpoint = critic_cfg.get("pi05_base_checkpoint")
    if critic_cfg.get("vision_mode", "openpi_pi05_siglip") == "openpi_pi05_siglip" and not pi05_base_checkpoint:
        raise ValueError("critic.pi05_base_checkpoint is required for vision_mode=openpi_pi05_siglip")

    started = time.time()
    if not args.skip_collect:
        collect_cmd = [
            "uv",
            "run",
            "python",
            "scripts/collect_r1lite_aloe_replay.py",
            "--experiment",
            args.experiment,
            "--iteration",
            str(args.iteration),
            "--replay-root",
            str(replay_root),
            "--policy-checkpoint",
            str(args.actor_checkpoint),
        ]
        if args.config is not None:
            collect_cmd.extend(["--aloe-config", args.config])
        if collector_cfg.get("max_steps") is not None:
            collect_cmd.extend(["--max-steps", str(int(collector_cfg["max_steps"]))])
        if args.collect_target_successes is not None:
            print(
                "warning: --collect-target-successes is recorded in the manifest only; "
                "collector is still one rollout per process.",
                flush=True,
            )
        _run(collect_cmd, dry_run=args.dry_run)

    validate_cmd = ["uv", "run", "python", "scripts/validate_aloe_replay.py", str(replay_root)]
    _run(validate_cmd, dry_run=args.dry_run)

    critic_ckpt = critic_dir / f"critic_step_{int(critic_cfg['steps_per_iteration'])}.pt"
    train_critic_cmd = [
        "uv",
        "run",
        "python",
        "scripts/train_aloe_critic.py",
        "--replay-root",
        str(replay_root),
        "--output-dir",
        str(critic_dir),
        "--policy-host",
        policy_host,
        "--policy-port",
        str(policy_port),
        "--action-horizon",
        str(critic_cfg["action_horizon"]),
        "--steps",
        str(critic_cfg["steps_per_iteration"]),
        "--batch-size",
        str(critic_cfg["batch_size"]),
        "--lr",
        str(critic_cfg["lr"]),
        "--gamma",
        str(critic_cfg["gamma"]),
        "--polyak",
        str(critic_cfg["polyak"]),
        "--ensemble",
        str(critic_cfg["ensemble"]),
        "--embed-dim",
        str(critic_cfg["embed_dim"]),
        "--layers",
        str(critic_cfg["layers"]),
        "--heads",
        str(critic_cfg["heads"]),
        "--pi05-base-checkpoint",
        str(pi05_base_checkpoint),
        "--vision-mode",
        str(critic_cfg.get("vision_mode", "openpi_pi05_siglip")),
        "--save-interval",
        str(critic_cfg["steps_per_iteration"]),
    ]
    _run(train_critic_cmd, dry_run=args.dry_run)

    train_actor_cmd = [
        "uv",
        "run",
        "python",
        "scripts/train_aloe_actor.py",
        "--replay-root",
        str(replay_root),
        "--critic-checkpoint",
        str(critic_ckpt),
        "--actor-config",
        str(cfg["actor_train_config"]),
        "--actor-checkpoint",
        str(args.actor_checkpoint),
        "--output-dir",
        str(actor_dir),
        "--policy-host",
        policy_host,
        "--policy-port",
        str(policy_port),
        "--action-horizon",
        str(actor_cfg["action_horizon"]),
        "--batch-size",
        str(actor_cfg["batch_size"]),
        "--steps",
        str(actor_cfg["steps_per_iteration"]),
        "--lr",
        str(actor_cfg["lr"]),
        "--beta",
        str(actor_cfg["beta"]),
        "--adv-clip",
        str(actor_cfg["adv_clip"]),
        "--save-interval",
        str(actor_cfg["steps_per_iteration"]),
        "--overwrite",
    ]
    _run(train_actor_cmd, dry_run=args.dry_run)

    summary = None if args.dry_run else replay_summary(replay_root)
    payload = {
        "experiment": args.experiment,
        "iteration": args.iteration,
        "started_unix": started,
        "finished_unix": time.time(),
        "actor_input_checkpoint": str(args.actor_checkpoint),
        "critic_checkpoint": str(critic_ckpt),
        "actor_output_dir": str(actor_dir),
        "replay_root": str(replay_root),
        "replay_summary": summary,
        "dry_run": bool(args.dry_run),
        "config_path": cfg["_config_path"],
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _manifest(run_root / "manifest.json", payload)
        print(f"wrote iteration manifest: {run_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
