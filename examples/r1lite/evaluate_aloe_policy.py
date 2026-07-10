#!/usr/bin/env python3
"""Run operator-scored ALOE policy evaluation rollouts and write a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="r1lite_pack_phone")
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    started = time.time()
    for episode in range(args.episodes):
        print(f"evaluation episode {episode}: press s/f/q in the rollout terminal according to the task outcome")
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/collect_r1lite_aloe_replay.py",
            "--experiment",
            args.experiment,
            "--iteration",
            str(episode),
            "--replay-root",
            str(args.output / "eval_replay"),
            "--policy-checkpoint",
            str(args.actor_checkpoint),
            "--max-steps",
            str(args.max_steps),
        ]
        episode_start = time.time()
        if not args.dry_run:
            subprocess.run(cmd, check=True)
        results.append(
            {
                "episode": episode,
                "command": cmd,
                "duration_seconds": time.time() - episode_start,
                "scoring": "operator_terminal_key_in_replay_manifest",
            }
        )
    manifest = {
        "experiment": args.experiment,
        "actor_checkpoint": str(args.actor_checkpoint),
        "episodes": args.episodes,
        "started_unix": started,
        "finished_unix": time.time(),
        "dry_run": bool(args.dry_run),
        "results": results,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote eval manifest: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
