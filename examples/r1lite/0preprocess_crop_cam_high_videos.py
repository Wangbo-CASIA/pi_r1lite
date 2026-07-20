#!/usr/bin/env python3
"""Crop every R1Lite ``cam_high`` frame to its bottom-right two thirds.

MP4 files are read from ``videos/chunk-000/observation.images.cam_high`` and
the cropped videos are saved under
``videos/chunk-000/observation.images.cam_high_crop``.  Each frame is cropped
exactly like ``image[h // 3 :, w // 3 :]``.  Every video's actual resolution
and frame rate are probed independently; cropped frames are not resized.

The default mode is a read-only preflight.  Pass ``--apply`` to encode the
videos.  Source videos are never modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

DEFAULT_DATASET_ROOT = Path("/home/robot/wangbo/project/VLA_own/data/r1lite-pack-phone-dagger-it1-0717")
SOURCE_CAMERA_RELATIVE_DIR = Path("videos/chunk-000/observation.images.cam_high")
OUTPUT_CAMERA_RELATIVE_DIR = Path("videos/chunk-000/observation.images.cam_high_crop")


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frame_count: int
    fps: float
    duration: float


def _parse_fraction(value: str) -> float:
    numerator, denominator = value.split("/", maxsplit=1)
    return float(numerator) / float(denominator)


def _probe(path: Path) -> VideoInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams: list[dict[str, Any]] = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}, found {len(streams)}")

    stream = streams[0]
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        frame_count=int(stream["nb_read_frames"]),
        fps=_parse_fraction(stream["avg_frame_rate"]),
        duration=float(stream["duration"]),
    )


def _validate_output(source: Path, output: Path, before: VideoInfo) -> None:
    after = _probe(output)
    expected_width = before.width - before.width // 3
    expected_height = before.height - before.height // 3
    if (after.width, after.height) != (expected_width, expected_height):
        raise RuntimeError(
            f"Output resolution mismatch for {source.name}: "
            f"{after.width}x{after.height} != {expected_width}x{expected_height}"
        )
    if after.frame_count != before.frame_count:
        raise RuntimeError(
            f"Output frame-count mismatch for {source.name}: {after.frame_count} != {before.frame_count}"
        )
    if abs(after.fps - before.fps) > 1e-3:
        raise RuntimeError(f"Output FPS mismatch for {source.name}: {after.fps} != {before.fps}")


def _encode_one(source: Path, temporary_dir: Path, crf: int, preset: str) -> tuple[Path, Path, VideoInfo]:
    before = _probe(source)
    output = temporary_dir / source.name

    # exact=1 preserves the same integer slicing boundary as NumPy:
    # frame[h // 3 :, w // 3 :].  yuv444p supports odd cropped dimensions
    # (for example, 640x360 becomes 427x240).
    video_filter = (
        "crop="
        "w=iw-trunc(iw/3):"
        "h=ih-trunc(ih/3):"
        "x=trunc(iw/3):"
        "y=trunc(ih/3):"
        "exact=1,"
        "format=yuv444p"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        video_filter,
        "-vsync",
        "0",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv444p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)
    _validate_output(source, output, before)
    return source, output, before


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable is not available: {name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--workers", type=int, default=2, help="Concurrent ffmpeg processes (default: 2).")
    parser.add_argument("--crf", type=int, default=18, help="libx264 quality; lower is higher quality (default: 18).")
    parser.add_argument("--preset", default="medium", help="libx264 preset (default: medium).")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Encode cropped videos into cam_high_crop. Without this flag, only run the read-only preflight.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError(f"--workers must be at least 1, got {args.workers}")
    if not 0 <= args.crf <= 51:
        raise ValueError(f"--crf must be in [0, 51], got {args.crf}")

    _require_tool("ffmpeg")
    _require_tool("ffprobe")

    dataset_root = args.dataset_root.expanduser().resolve()
    source_dir = dataset_root / SOURCE_CAMERA_RELATIVE_DIR
    output_dir = dataset_root / OUTPUT_CAMERA_RELATIVE_DIR
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Camera video directory does not exist: {source_dir}")

    videos = sorted(source_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No MP4 files found in {source_dir}")

    video_infos = [(video, _probe(video)) for video in videos]
    geometry_counts: dict[tuple[int, int, int, int, float], int] = {}
    for _, info in video_infos:
        geometry = (
            info.width,
            info.height,
            info.width - info.width // 3,
            info.height - info.height // 3,
            info.fps,
        )
        geometry_counts[geometry] = geometry_counts.get(geometry, 0) + 1

    print(f"Dataset: {dataset_root}")
    print(f"Source:  {source_dir}")
    print(f"Output:  {output_dir}")
    print(f"Videos:  {len(videos)}")
    for (width, height, crop_width, crop_height, fps), count in sorted(geometry_counts.items()):
        print(f"Format:  {count} video(s): {width}x{height} -> {crop_width}x{crop_height}, {fps:g} FPS")

    if not args.apply:
        print("Preflight only: no files were changed. Re-run with --apply to process all listed videos.")
        return 0

    existing_outputs = [output_dir / video.name for video in videos if (output_dir / video.name).exists()]
    if existing_outputs:
        raise FileExistsError(
            f"Refusing to overwrite {len(existing_outputs)} existing output video(s); first file: {existing_outputs[0]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".encoding_", dir=output_dir))
    encoded: list[tuple[Path, Path, VideoInfo]] = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_encode_one, video, temporary_dir, args.crf, args.preset): video for video in videos
            }
            for completed_count, future in enumerate(as_completed(futures), start=1):
                source = futures[future]
                encoded.append(future.result())
                print(f"[{completed_count}/{len(videos)}] validated {source.name}", flush=True)

        # Publish only after every newly encoded video has passed validation.
        for source, temporary_output, _ in sorted(encoded, key=lambda item: item[0].name):
            final_output = output_dir / source.name
            os.chmod(temporary_output, source.stat().st_mode)
            os.replace(temporary_output, final_output)
        print(f"Done: saved {len(encoded)} cropped videos to {output_dir}.")
        return 0
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
