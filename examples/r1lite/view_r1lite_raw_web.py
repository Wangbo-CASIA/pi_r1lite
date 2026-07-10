#!/usr/bin/env python3
"""Local web viewer for R1Lite RAW MCAP episodes."""

from __future__ import annotations

import argparse
import atexit
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any
from urllib.parse import unquote

import cv2
from flask import Flask
from flask import Response
from flask import jsonify
import numpy as np

CONRFT_ROOT = Path("/home/ps/VLA-RL/conrft-r1lite")
SARM_EXAMPLES = CONRFT_ROOT / "examples" / "sarm"
DEFAULT_DATA_DIR = CONRFT_ROOT / "data" / "RAW" / "r1lite_pack_phone_new"

if str(SARM_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(SARM_EXAMPLES))

from rosbag_sarm_utils import DEFAULT_TOPICS  # noqa: E402
from rosbag_sarm_utils import build_episode_samples  # noqa: E402
from rosbag_sarm_utils import is_raw_episode_dir  # noqa: E402
from rosbag_sarm_utils import load_metadata  # noqa: E402

CAMERAS = ("head", "left_wrist", "right_wrist")
GRIPPERS = ("left_gripper", "right_gripper")


@dataclass(frozen=True)
class EpisodeInfo:
    name: str
    path: Path
    duration_seconds: float | None
    message_count: int | None
    bag_size_mb: float | None
    created_time: str | None


@dataclass
class LoadedEpisode:
    samples: list[dict[str, Any]]
    image_dir: Path


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>R1Lite RAW Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #667085;
      --line: #d8dee8;
      --accent: #2764c5;
      --accent-dark: #174ea6;
      --danger: #b42318;
      --shadow: 0 1px 3px rgba(16, 24, 40, 0.12);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }

    button,
    select,
    input {
      font: inherit;
    }

    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }

    .topbar {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 18px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      box-shadow: var(--shadow);
    }

    .brand {
      font-weight: 700;
      white-space: nowrap;
      margin-right: 6px;
    }

    .episode-select {
      flex: 1;
      min-width: 220px;
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      color: var(--text);
      padding: 0 10px;
    }

    .button {
      height: 36px;
      min-width: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      color: var(--text);
      cursor: pointer;
      padding: 0 12px;
    }

    .button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }

    .button:hover {
      border-color: var(--accent);
    }

    .button.primary:hover {
      background: var(--accent-dark);
    }

    .button:disabled,
    .episode-select:disabled,
    input:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }

    .content {
      padding: 16px 18px 18px;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 14px;
    }

    .status-line {
      min-height: 24px;
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px 18px;
    }

    .status-line.error {
      color: var(--danger);
      font-weight: 600;
    }

    .camera-grid {
      display: grid;
      grid-template-columns: 2fr 1fr 1fr;
      gap: 14px;
      align-items: start;
    }

    .camera {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: var(--shadow);
    }

    .camera-title {
      height: 34px;
      display: flex;
      align-items: center;
      padding: 0 10px;
      font-weight: 650;
      color: var(--muted);
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }

    .camera img {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #101828;
    }

    .bottom {
      background: var(--panel);
      border-top: 1px solid var(--line);
      padding: 12px 18px 14px;
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 16px;
      align-items: center;
      box-shadow: 0 -1px 3px rgba(16, 24, 40, 0.08);
    }

    .transport {
      display: flex;
      gap: 8px;
      align-items: center;
    }

    .progress-wrap {
      display: grid;
      grid-template-columns: 1fr;
      gap: 6px;
    }

    .progress-meta {
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      min-height: 18px;
    }

    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }

    .readouts {
      display: grid;
      grid-template-columns: repeat(2, minmax(140px, auto));
      gap: 8px 14px;
      align-items: center;
      color: var(--muted);
      white-space: nowrap;
    }

    .readout-value {
      color: var(--text);
      font-variant-numeric: tabular-nums;
      font-weight: 650;
      margin-left: 6px;
    }

    @media (max-width: 900px) {
      .topbar {
        flex-wrap: wrap;
      }

      .brand {
        width: 100%;
      }

      .camera-grid {
        grid-template-columns: 1fr;
      }

      .bottom {
        grid-template-columns: 1fr;
      }

      .transport {
        justify-content: center;
      }

      .readouts {
        grid-template-columns: 1fr 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <div class="brand">R1Lite RAW Viewer</div>
      <select id="episodeSelect" class="episode-select" aria-label="Episode"></select>
      <button id="loadButton" class="button primary" type="button">Load</button>
      <button id="reloadButton" class="button" type="button">Reload list</button>
    </header>

    <main class="content">
      <div id="status" class="status-line">Loading episode list...</div>
      <section class="camera-grid" aria-label="Camera frames">
        <article class="camera">
          <div class="camera-title">head</div>
          <img id="imgHead" alt="head camera">
        </article>
        <article class="camera">
          <div class="camera-title">left_wrist</div>
          <img id="imgLeftWrist" alt="left wrist camera">
        </article>
        <article class="camera">
          <div class="camera-title">right_wrist</div>
          <img id="imgRightWrist" alt="right wrist camera">
        </article>
      </section>
    </main>

    <footer class="bottom">
      <div class="transport">
        <button id="prevButton" class="button" title="Previous frame" type="button">&lt;</button>
        <button id="playButton" class="button primary" title="Play or pause" type="button">Play</button>
        <button id="nextButton" class="button" title="Next frame" type="button">&gt;</button>
      </div>
      <div class="progress-wrap">
        <input id="frameRange" type="range" min="0" max="0" step="1" value="0" aria-label="Frame index">
        <div class="progress-meta">
          <span id="frameLabel">Frame 0 / 0</span>
          <span id="timestampLabel">timestamp: -</span>
        </div>
      </div>
      <div class="readouts">
        <div>left gripper<span id="leftGripper" class="readout-value">-</span></div>
        <div>right gripper<span id="rightGripper" class="readout-value">-</span></div>
      </div>
    </footer>
  </div>

  <script>
    const state = {
      episodes: [],
      episode: null,
      frameCount: 0,
      frameIndex: 0,
      fps: 10,
      timer: null,
      loading: false,
    };

    const el = {
      episodeSelect: document.getElementById("episodeSelect"),
      loadButton: document.getElementById("loadButton"),
      reloadButton: document.getElementById("reloadButton"),
      status: document.getElementById("status"),
      imgHead: document.getElementById("imgHead"),
      imgLeftWrist: document.getElementById("imgLeftWrist"),
      imgRightWrist: document.getElementById("imgRightWrist"),
      prevButton: document.getElementById("prevButton"),
      playButton: document.getElementById("playButton"),
      nextButton: document.getElementById("nextButton"),
      frameRange: document.getElementById("frameRange"),
      frameLabel: document.getElementById("frameLabel"),
      timestampLabel: document.getElementById("timestampLabel"),
      leftGripper: document.getElementById("leftGripper"),
      rightGripper: document.getElementById("rightGripper"),
    };

    function setStatus(message, isError = false) {
      el.status.textContent = message;
      el.status.className = isError ? "status-line error" : "status-line";
    }

    function setBusy(isBusy) {
      state.loading = isBusy;
      el.episodeSelect.disabled = isBusy;
      el.loadButton.disabled = isBusy;
      el.reloadButton.disabled = isBusy;
      el.prevButton.disabled = isBusy || state.frameCount <= 0;
      el.nextButton.disabled = isBusy || state.frameCount <= 0;
      el.playButton.disabled = isBusy || state.frameCount <= 0;
      el.frameRange.disabled = isBusy || state.frameCount <= 0;
    }

    function stopPlayback() {
      if (state.timer !== null) {
        window.clearInterval(state.timer);
        state.timer = null;
      }
      el.playButton.textContent = "Play";
    }

    async function api(path, options = {}) {
      const response = await fetch(path, options);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || `Request failed: ${response.status}`);
      }
      return payload;
    }

    async function loadEpisodeList() {
      stopPlayback();
      setBusy(true);
      setStatus("Loading episode list...");
      try {
        const payload = await api("/api/episodes");
        state.episodes = payload.episodes;
        state.fps = payload.fps;
        el.episodeSelect.innerHTML = "";
        for (const episode of state.episodes) {
          const option = document.createElement("option");
          option.value = episode.name;
          option.textContent = episode.name;
          el.episodeSelect.appendChild(option);
        }
        setStatus(`${state.episodes.length} episodes found`);
      } catch (error) {
        setStatus(error.message, true);
      } finally {
        setBusy(false);
      }
    }

    async function loadSelectedEpisode() {
      const name = el.episodeSelect.value;
      if (!name) {
        setStatus("No episode selected", true);
        return;
      }
      stopPlayback();
      setBusy(true);
      setStatus(`Loading ${name}...`);
      try {
        const payload = await api(`/api/episodes/${encodeURIComponent(name)}/load`, { method: "POST" });
        state.episode = name;
        state.frameCount = payload.frame_count;
        state.frameIndex = 0;
        el.frameRange.min = "0";
        el.frameRange.max = String(Math.max(0, state.frameCount - 1));
        el.frameRange.value = "0";
        await showFrame(0);
        setStatus(`${name} loaded: ${payload.frame_count} frames`);
      } catch (error) {
        state.episode = null;
        state.frameCount = 0;
        setStatus(error.message, true);
      } finally {
        setBusy(false);
      }
    }

    async function showFrame(index) {
      if (!state.episode || state.frameCount <= 0) {
        return;
      }
      const clamped = Math.max(0, Math.min(state.frameCount - 1, Number(index)));
      const payload = await api(`/api/episodes/${encodeURIComponent(state.episode)}/frames/${clamped}`);
      state.frameIndex = payload.index;
      el.frameRange.value = String(payload.index);
      el.frameLabel.textContent = `Frame ${payload.index + 1} / ${state.frameCount}`;
      el.timestampLabel.textContent = `timestamp: ${payload.timestamp_ns}`;
      el.leftGripper.textContent = payload.left_gripper.toFixed(6);
      el.rightGripper.textContent = payload.right_gripper.toFixed(6);
      el.imgHead.src = `${payload.images.head}&t=${Date.now()}`;
      el.imgLeftWrist.src = `${payload.images.left_wrist}&t=${Date.now()}`;
      el.imgRightWrist.src = `${payload.images.right_wrist}&t=${Date.now()}`;
    }

    function togglePlayback() {
      if (state.timer !== null) {
        stopPlayback();
        return;
      }
      if (!state.episode || state.frameCount <= 0) {
        return;
      }
      el.playButton.textContent = "Pause";
      const intervalMs = Math.max(20, Math.round(1000 / state.fps));
      state.timer = window.setInterval(async () => {
        const next = state.frameIndex + 1;
        if (next >= state.frameCount) {
          stopPlayback();
          return;
        }
        try {
          await showFrame(next);
        } catch (error) {
          stopPlayback();
          setStatus(error.message, true);
        }
      }, intervalMs);
    }

    el.loadButton.addEventListener("click", loadSelectedEpisode);
    el.reloadButton.addEventListener("click", loadEpisodeList);
    el.prevButton.addEventListener("click", () => {
      stopPlayback();
      showFrame(state.frameIndex - 1).catch(error => setStatus(error.message, true));
    });
    el.nextButton.addEventListener("click", () => {
      stopPlayback();
      showFrame(state.frameIndex + 1).catch(error => setStatus(error.message, true));
    });
    el.playButton.addEventListener("click", togglePlayback);
    el.frameRange.addEventListener("input", () => {
      stopPlayback();
      showFrame(Number(el.frameRange.value)).catch(error => setStatus(error.message, true));
    });
    el.episodeSelect.addEventListener("change", loadSelectedEpisode);

    setBusy(true);
    loadEpisodeList().then(() => {
      if (state.episodes.length > 0) {
        loadSelectedEpisode();
      }
    });
  </script>
</body>
</html>
"""


def _json_error(message: str, status_code: int) -> tuple[Response, int]:
    return jsonify({"error": message}), status_code


def _load_bag_info(path: Path) -> dict[str, Any]:
    metadata = load_metadata(path)
    return metadata.get("rosbag2_bagfile_information", {})


def _duration_seconds(bag_info: dict[str, Any]) -> float | None:
    duration = bag_info.get("duration")
    if not isinstance(duration, dict):
        return None
    nanoseconds = duration.get("nanoseconds")
    if nanoseconds is None:
        return None
    return float(nanoseconds) / 1e9


def _bag_json_path(path: Path) -> Path:
    return path.parent / f"{path.name}.json"


def _created_time(path: Path) -> str | None:
    json_path = _bag_json_path(path)
    if not json_path.exists():
        return None
    try:
        import json

        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return None
    bag_info = payload.get("bag_info")
    if not isinstance(bag_info, dict):
        return None
    created_time = bag_info.get("created_time")
    return str(created_time) if created_time is not None else None


def discover_episodes(data_dir: Path, raw_dir_glob: str) -> list[EpisodeInfo]:
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {data_dir}")

    paths = sorted(path for path in data_dir.glob(raw_dir_glob) if path.is_dir() and is_raw_episode_dir(path))
    if not paths and is_raw_episode_dir(data_dir):
        paths = [data_dir]
    if not paths:
        raise FileNotFoundError(f"No RAW episode directories matching {raw_dir_glob!r} found under {data_dir}")

    episodes = []
    for path in paths:
        bag_info = _load_bag_info(path)
        episodes.append(
            EpisodeInfo(
                name=path.name,
                path=path,
                duration_seconds=_duration_seconds(bag_info),
                message_count=bag_info.get("message_count"),
                bag_size_mb=None,
                created_time=_created_time(path),
            )
        )
    return episodes


def _topic_overrides(args: argparse.Namespace) -> dict[str, str]:
    topics = dict(DEFAULT_TOPICS)
    for key in (*CAMERAS, *GRIPPERS):
        value = getattr(args, f"{key}_topic")
        if value:
            topics[key] = value
    return topics


def _episode_payload(episode: EpisodeInfo) -> dict[str, Any]:
    return {
        "name": episode.name,
        "path": str(episode.path),
        "duration_seconds": episode.duration_seconds,
        "message_count": episode.message_count,
        "bag_size_mb": episode.bag_size_mb,
        "created_time": episode.created_time,
    }


def _sample_gripper(sample: dict[str, Any], key: str) -> float:
    value = np.asarray(sample[key], dtype=np.float32).reshape(-1)
    if value.size < 1:
        raise ValueError(f"{key} has no gripper position value")
    return float(value[0])


def _encode_jpeg(image: np.ndarray, output_path: Path, quality: int) -> None:
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (height, width, 3), got {rgb.shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = rgb[..., ::-1]
    ok = cv2.imwrite(str(output_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError(f"Failed to encode JPEG image: {output_path}")


def create_app(args: argparse.Namespace) -> Flask:
    data_dir = Path(args.data_dir)
    topics = _topic_overrides(args)
    temp_root = Path(tempfile.mkdtemp(prefix="r1lite_raw_viewer_"))
    loaded: dict[str, LoadedEpisode] = {}

    def cleanup() -> None:
        shutil.rmtree(temp_root, ignore_errors=True)

    atexit.register(cleanup)
    app = Flask(__name__)

    def episodes_by_name() -> dict[str, EpisodeInfo]:
        return {episode.name: episode for episode in discover_episodes(data_dir, args.raw_dir_glob)}

    def require_loaded_episode(name: str) -> LoadedEpisode:
        decoded = unquote(name)
        episode = loaded.get(decoded)
        if episode is None:
            raise KeyError(f"Episode is not loaded: {decoded}")
        return episode

    def frame_image_path(episode_name: str, loaded_episode: LoadedEpisode, index: int, camera: str) -> Path:
        output_path = loaded_episode.image_dir / f"{index:06d}_{camera}.jpg"
        if output_path.exists():
            return output_path
        sample = loaded_episode.samples[index]
        if camera not in CAMERAS:
            raise KeyError(f"Unsupported camera: {camera}")
        _encode_jpeg(sample[camera], output_path, args.jpeg_quality)
        return output_path

    @app.get("/")
    def index() -> Response:
        return Response(HTML, mimetype="text/html")

    @app.get("/api/episodes")
    def list_episodes() -> Response | tuple[Response, int]:
        try:
            episodes = discover_episodes(data_dir, args.raw_dir_glob)
            return jsonify(
                {
                    "data_dir": str(data_dir.expanduser().resolve()),
                    "fps": args.fps,
                    "topics": {key: topics[key] for key in (*CAMERAS, *GRIPPERS)},
                    "episodes": [_episode_payload(episode) for episode in episodes],
                }
            )
        except Exception as exc:
            return _json_error(str(exc), 500)

    @app.post("/api/episodes/<path:name>/load")
    def load_episode(name: str) -> Response | tuple[Response, int]:
        decoded = unquote(name)
        try:
            episodes = episodes_by_name()
            episode = episodes.get(decoded)
            if episode is None:
                return _json_error(f"Unknown episode: {decoded}", 404)
            samples = build_episode_samples(episode.path, args.fps, topics)
            if not samples:
                raise ValueError(f"No synchronized frames were built for episode: {decoded}")
            image_dir = temp_root / decoded
            if image_dir.exists():
                shutil.rmtree(image_dir)
            image_dir.mkdir(parents=True)
            loaded[decoded] = LoadedEpisode(samples=samples, image_dir=image_dir)
            return jsonify(
                {
                    "name": decoded,
                    "frame_count": len(samples),
                    "first_timestamp_ns": int(samples[0]["timestamp_ns"]),
                    "last_timestamp_ns": int(samples[-1]["timestamp_ns"]),
                }
            )
        except Exception as exc:
            loaded.pop(decoded, None)
            return _json_error(str(exc), 500)

    @app.get("/api/episodes/<path:name>/frames/<int:index>")
    def frame_metadata(name: str, index: int) -> Response | tuple[Response, int]:
        decoded = unquote(name)
        try:
            loaded_episode = require_loaded_episode(decoded)
            if index < 0 or index >= len(loaded_episode.samples):
                return _json_error(f"Frame index out of range: {index}", 400)
            sample = loaded_episode.samples[index]
            images = {}
            for camera in CAMERAS:
                frame_image_path(decoded, loaded_episode, index, camera)
                images[camera] = f"/api/episodes/{decoded}/frames/{index}/{camera}.jpg?episode={decoded}"
            return jsonify(
                {
                    "episode": decoded,
                    "index": index,
                    "frame_count": len(loaded_episode.samples),
                    "timestamp_ns": int(sample["timestamp_ns"]),
                    "left_gripper": _sample_gripper(sample, "left_gripper"),
                    "right_gripper": _sample_gripper(sample, "right_gripper"),
                    "images": images,
                }
            )
        except KeyError as exc:
            return _json_error(str(exc), 404)
        except Exception as exc:
            return _json_error(str(exc), 500)

    @app.get("/api/episodes/<path:name>/frames/<int:index>/<camera>.jpg")
    def frame_image(name: str, index: int, camera: str) -> Response | tuple[Response, int]:
        decoded = unquote(name)
        try:
            loaded_episode = require_loaded_episode(decoded)
            if index < 0 or index >= len(loaded_episode.samples):
                return _json_error(f"Frame index out of range: {index}", 400)
            if camera not in CAMERAS:
                return _json_error(f"Unsupported camera: {camera}", 400)
            image_path = frame_image_path(decoded, loaded_episode, index, camera)
            return Response(image_path.read_bytes(), mimetype="image/jpeg")
        except KeyError as exc:
            return _json_error(str(exc), 404)
        except Exception as exc:
            return _json_error(str(exc), 500)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--raw-dir-glob", default="*_RAW")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    for key in (*CAMERAS, *GRIPPERS):
        default = DEFAULT_TOPICS[key]
        parser.add_argument(
            f"--{key.replace('_', '-')}-topic",
            dest=f"{key}_topic",
            default=None,
            help=f"Override topic for {key}: {default}",
        )
    args = parser.parse_args()
    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")
    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        raise ValueError(f"--jpeg-quality must be in [1, 100], got {args.jpeg_quality}")
    return args


def main() -> None:
    args = parse_args()
    app = create_app(args)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
