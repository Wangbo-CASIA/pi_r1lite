#!/usr/bin/env python3
"""Explicitly migrate ALOE pkl-shard replay v1 runs to v2 joint_delta records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

from openpi.aloe import schema


def _migrate_run(run_dir: Path) -> int:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") == schema.FORMAT_VERSION:
        return 0
    if manifest.get("format") not in (None, "aloe_pkl_shards_v1"):
        raise ValueError(f"unsupported source format in {manifest_path}: {manifest.get('format')!r}")
    changed = 0
    for shard_path in sorted((run_dir / "shards").glob("shard_*.pkl")):
        with shard_path.open("rb") as f:
            shard = pickle.load(f)
        if not isinstance(shard, list):
            raise ValueError(f"shard must contain a list: {shard_path}")
        migrated = []
        for chunk in shard:
            if not isinstance(chunk, dict):
                raise ValueError(f"invalid chunk type in {shard_path}: {type(chunk).__name__}")
            chunk = dict(chunk)
            chunk["format_version"] = schema.FORMAT_VERSION
            chunk["action_space"] = "joint_delta"
            schema.validate_chunk_record(chunk)
            migrated.append(chunk)
            changed += 1
        with shard_path.open("wb") as f:
            pickle.dump(migrated, f, protocol=pickle.HIGHEST_PROTOCOL)
    manifest["format"] = "aloe_pkl_shards_v2"
    manifest["format_version"] = schema.FORMAT_VERSION
    manifest["action_space"] = "joint_delta"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_root", type=Path)
    args = parser.parse_args()
    if not args.replay_root.exists():
        raise FileNotFoundError(args.replay_root)
    total = 0
    run_dirs = sorted(args.replay_root.glob("iteration_*/run_*"))
    run_dirs.extend(sorted(args.replay_root.glob("human_demo/run_*")))
    for run_dir in run_dirs:
        if run_dir.is_dir():
            total += _migrate_run(run_dir)
    print(f"migrated chunks={total} under {args.replay_root}")


if __name__ == "__main__":
    main()
