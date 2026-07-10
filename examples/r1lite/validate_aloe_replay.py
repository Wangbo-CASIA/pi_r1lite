#!/usr/bin/env python3
"""Validate an ALOE pkl-shard replay directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from openpi.aloe.dataset import grouped_segments
from openpi.aloe.dataset import load_chunks
from openpi.aloe.dataset import replay_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_root", type=Path)
    args = parser.parse_args()
    chunks = load_chunks(args.replay_root)
    segments = grouped_segments(chunks)
    terminal = sum(1 for chunk in chunks if chunk.get("terminal_outcome") is not None)
    actions = sum(int(chunk["valid_mask"].sum()) for chunk in chunks)
    summary = replay_summary(args.replay_root)
    print(
        f"valid aloe replay: chunks={len(chunks)} actions={actions} segments={len(segments)} "
        f"terminal_chunks={terminal} sources={summary['sources']} terminal={summary['terminal']}"
    )


if __name__ == "__main__":
    main()
