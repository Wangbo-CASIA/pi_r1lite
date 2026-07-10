#!/usr/bin/env python3
"""Web viewer for chunked open-loop R1Lite policy actions on LeRobot episodes."""

from __future__ import annotations

import argparse
import atexit
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Any

import cv2
from flask import Flask
from flask import Response
from flask import jsonify
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np
from openpi_client import rtc_client_policy
from openpi_client import websocket_client_policy
from r1lite_experiment_config import _config_path
from r1lite_experiment_config import _load_yaml
from r1lite_experiment_config import normalize_action_space
from r1lite_rtc import add_rtc_args
from r1lite_rtc import apply_rtc_defaults
import torch

CAMERAS = ("head", "left_wrist", "right_wrist")
IMAGE_KEYS = {
    "head": "observation.images.head",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    length: int
    task: str | None


@dataclass
class LoadedEpisode:
    dataset: LeRobotDataset
    image_dir: Path
    pred: np.ndarray
    gt: np.ndarray
    action_names: list[str]
    plot_order: list[int]


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>R1Lite Open-Loop Viewer</title>
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
      --predict: #2764c5;
      --gt: #c2410c;
      --cursor: #111827;
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
      grid-template-rows: auto auto 1fr;
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

    .camera,
    .plot-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: var(--shadow);
    }

    .camera-title,
    .plot-title {
      min-height: 34px;
      display: flex;
      align-items: center;
      gap: 14px;
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

    .plot-title {
      justify-content: space-between;
      flex-wrap: wrap;
      padding-top: 6px;
      padding-bottom: 6px;
    }

    .legend {
      display: flex;
      gap: 14px;
      align-items: center;
      font-weight: 500;
    }

    .legend-item {
      display: flex;
      gap: 6px;
      align-items: center;
      white-space: nowrap;
    }

    .swatch {
      width: 18px;
      height: 3px;
      border-radius: 99px;
      display: inline-block;
    }

    .swatch.predict {
      background: var(--predict);
    }

    .swatch.gt {
      background: var(--gt);
    }

    .plot-wrap {
      width: 100%;
      overflow-x: auto;
      background: #ffffff;
    }

    #actionCanvas {
      display: block;
      width: 100%;
      min-height: 520px;
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
      grid-template-columns: repeat(3, minmax(110px, auto));
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

    @media (max-width: 980px) {
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
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <div class="brand">R1Lite Open-Loop Viewer</div>
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
      <section class="plot-panel" aria-label="Open-loop action plots">
        <div class="plot-title">
          <span id="plotTitleText">Open-loop actions</span>
          <span class="legend">
            <span class="legend-item"><span class="swatch predict"></span>predict</span>
            <span class="legend-item"><span class="swatch gt"></span>gt</span>
          </span>
        </div>
        <div class="plot-wrap">
          <canvas id="actionCanvas"></canvas>
        </div>
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
        <div>chunk<span id="chunkLabel" class="readout-value">-</span></div>
        <div>predict[0]<span id="predictLabel" class="readout-value">-</span></div>
        <div>gt[0]<span id="gtLabel" class="readout-value">-</span></div>
      </div>
    </footer>
  </div>

  <script>
    const state = {
      episodes: [],
      episodeIndex: null,
      frameCount: 0,
      frameIndex: 0,
      fps: 10,
      actionsPerInfer: 1,
      timer: null,
      loading: false,
      actions: null,
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
      chunkLabel: document.getElementById("chunkLabel"),
      predictLabel: document.getElementById("predictLabel"),
      gtLabel: document.getElementById("gtLabel"),
      actionCanvas: document.getElementById("actionCanvas"),
      plotTitleText: document.getElementById("plotTitleText"),
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
        state.actionsPerInfer = payload.actions_per_infer;
        state.rtcEnabled = payload.rtc && payload.rtc.enabled;
        el.episodeSelect.innerHTML = "";
        for (const episode of state.episodes) {
          const option = document.createElement("option");
          option.value = String(episode.episode_index);
          option.textContent = `episode ${episode.episode_index} (${episode.length} frames)`;
          el.episodeSelect.appendChild(option);
        }
        setStatus(`${state.episodes.length} episodes found; policy server ${payload.policy_host}:${payload.policy_port}; rtc=${state.rtcEnabled ? "on" : "off"}`);
      } catch (error) {
        setStatus(error.message, true);
      } finally {
        setBusy(false);
      }
    }

    async function loadSelectedEpisode() {
      const episodeIndex = el.episodeSelect.value;
      if (episodeIndex === "") {
        setStatus("No episode selected", true);
        return;
      }
      stopPlayback();
      setBusy(true);
      setStatus(`Loading episode ${episodeIndex} and running open-loop inference...`);
      try {
        const payload = await api(`/api/episodes/${encodeURIComponent(episodeIndex)}/load`, { method: "POST" });
        state.episodeIndex = Number(episodeIndex);
        state.frameCount = payload.frame_count;
        state.frameIndex = 0;
        state.actionsPerInfer = payload.actions_per_infer;
        state.rtcEnabled = payload.rtc_enabled;
        state.actions = await api(`/api/episodes/${encodeURIComponent(episodeIndex)}/actions`);
        el.frameRange.min = "0";
        el.frameRange.max = String(Math.max(0, state.frameCount - 1));
        el.frameRange.value = "0";
        el.plotTitleText.textContent = `Open-loop actions, actions_per_infer=${state.actionsPerInfer}, rtc=${state.rtcEnabled ? "on" : "off"}`;
        await showFrame(0);
        drawActions();
        setStatus(`episode ${episodeIndex} loaded: ${payload.frame_count} frames`);
      } catch (error) {
        state.episodeIndex = null;
        state.frameCount = 0;
        state.actions = null;
        drawActions();
        setStatus(error.message, true);
      } finally {
        setBusy(false);
      }
    }

    async function showFrame(index) {
      if (state.episodeIndex === null || state.frameCount <= 0) {
        return;
      }
      const clamped = Math.max(0, Math.min(state.frameCount - 1, Number(index)));
      const payload = await api(`/api/episodes/${encodeURIComponent(state.episodeIndex)}/frames/${clamped}`);
      const cacheToken = Date.now();
      const imageSources = {
        head: `${payload.images.head}?t=${cacheToken}`,
        left_wrist: `${payload.images.left_wrist}?t=${cacheToken}`,
        right_wrist: `${payload.images.right_wrist}?t=${cacheToken}`,
      };
      await preloadImages(Object.values(imageSources));
      state.frameIndex = payload.index;
      el.frameRange.value = String(payload.index);
      el.frameLabel.textContent = `Frame ${payload.index + 1} / ${state.frameCount}`;
      el.timestampLabel.textContent = `timestamp: ${payload.timestamp}`;
      el.chunkLabel.textContent = String(Math.floor(payload.index / state.actionsPerInfer));
      el.predictLabel.textContent = formatNumber(payload.predict[0]);
      el.gtLabel.textContent = formatNumber(payload.gt[0]);
      el.imgHead.src = imageSources.head;
      el.imgLeftWrist.src = imageSources.left_wrist;
      el.imgRightWrist.src = imageSources.right_wrist;
      drawActions();
    }

    function preloadImages(srcs) {
      return Promise.all(srcs.map(src => new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = resolve;
        image.onerror = () => reject(new Error(`Failed to load image: ${src}`));
        image.src = src;
      })));
    }

    function formatNumber(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "-";
      }
      return Number(value).toFixed(4);
    }

    function togglePlayback() {
      if (state.timer !== null) {
        stopPlayback();
        return;
      }
      if (state.episodeIndex === null || state.frameCount <= 0) {
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

    function drawActions() {
      const canvas = el.actionCanvas;
      const ctx = canvas.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      const containerWidth = canvas.parentElement.clientWidth || 900;

      if (!state.actions || !state.actions.gt || !state.actions.pred) {
        canvas.width = Math.floor(containerWidth * dpr);
        canvas.height = Math.floor(260 * dpr);
        canvas.style.height = "260px";
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, containerWidth, 260);
        ctx.fillStyle = "#667085";
        ctx.font = "14px sans-serif";
        ctx.fillText("Load an episode to render action curves.", 16, 32);
        return;
      }

      const pred = state.actions.pred;
      const gt = state.actions.gt;
      const order = state.actions.plot_order;
      const names = state.actions.action_names;
      const frameCount = gt.length;
      const rowHeight = 86;
      const leftPad = 176;
      const rightPad = 18;
      const topPad = 16;
      const bottomPad = 18;
      const width = Math.max(containerWidth, 980);
      const height = topPad + bottomPad + rowHeight * order.length;

      canvas.style.height = `${height}px`;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      const plotWidth = width - leftPad - rightPad;
      const xForFrame = (frame) => leftPad + (frameCount <= 1 ? 0 : frame * plotWidth / (frameCount - 1));
      const cursorX = xForFrame(state.frameIndex);

      ctx.font = "12px sans-serif";
      for (let row = 0; row < order.length; row += 1) {
        const dim = order[row];
        const y0 = topPad + row * rowHeight;
        const plotTop = y0 + 10;
        const plotBottom = y0 + rowHeight - 18;
        const plotMid = (plotTop + plotBottom) / 2;
        const values = [];
        for (let i = 0; i < frameCount; i += 1) {
          values.push(pred[i][dim], gt[i][dim]);
        }
        let minValue = Math.min(...values);
        let maxValue = Math.max(...values);
        if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
          minValue = -1;
          maxValue = 1;
        }
        if (Math.abs(maxValue - minValue) < 1e-6) {
          minValue -= 1;
          maxValue += 1;
        }
        const pad = 0.08 * (maxValue - minValue);
        minValue -= pad;
        maxValue += pad;
        const yForValue = (value) => plotBottom - (value - minValue) * (plotBottom - plotTop) / (maxValue - minValue);

        ctx.strokeStyle = "#e5e7eb";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(leftPad, plotMid);
        ctx.lineTo(width - rightPad, plotMid);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(leftPad, plotBottom);
        ctx.lineTo(width - rightPad, plotBottom);
        ctx.stroke();

        ctx.fillStyle = "#17202a";
        ctx.textAlign = "right";
        ctx.fillText(names[dim], leftPad - 12, plotTop + 12);
        ctx.fillStyle = "#667085";
        ctx.fillText(maxValue.toFixed(3), leftPad - 12, plotTop + 28);
        ctx.fillText(minValue.toFixed(3), leftPad - 12, plotBottom);

        drawLine(ctx, pred, dim, frameCount, xForFrame, yForValue, "#2764c5", 1.6);
        drawLine(ctx, gt, dim, frameCount, xForFrame, yForValue, "#c2410c", 1.3);

        ctx.strokeStyle = "#f1f3f7";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, y0 + rowHeight - 1);
        ctx.lineTo(width, y0 + rowHeight - 1);
        ctx.stroke();
      }

      ctx.strokeStyle = "#111827";
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(cursorX, topPad);
      ctx.lineTo(cursorX, height - bottomPad);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#111827";
      ctx.textAlign = "center";
      ctx.fillText(String(state.frameIndex), cursorX, 12);
    }

    function drawLine(ctx, values, dim, frameCount, xForFrame, yForValue, color, width) {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      for (let i = 0; i < frameCount; i += 1) {
        const x = xForFrame(i);
        const y = yForValue(values[i][dim]);
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
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
    window.addEventListener("resize", drawActions);

    setBusy(true);
    drawActions();
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


def _select_action_space(exp_cfg: dict[str, Any], requested: str | None) -> str:
    action_spaces = exp_cfg.get("action_spaces")
    if not isinstance(action_spaces, dict) or not action_spaces:
        raise ValueError("experiment config must contain a non-empty action_spaces object")

    if requested is not None:
        action_space = normalize_action_space(requested)
        if action_space not in action_spaces:
            raise ValueError(f"experiment config has no action_spaces.{action_space} section")
        return action_space

    if len(action_spaces) == 1:
        return next(iter(action_spaces))

    available = ", ".join(sorted(action_spaces))
    raise ValueError(f"experiment config has multiple action spaces ({available}); pass --action-space")


def _load_experiment(experiment: str | None, config: str | None) -> tuple[Path, dict[str, Any]]:
    path = _config_path(experiment, config)
    if path is None:
        raise ValueError("pass either --experiment or --config")
    return path, _load_yaml(path)


def _get_required_dict(root: dict[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"experiment config is missing object section: {key}")
    return value


def _get_actions_per_infer(exp_cfg: dict[str, Any]) -> int:
    rollout = _get_required_dict(exp_cfg, "rollout")
    value = rollout.get("actions_per_infer")
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"rollout.actions_per_infer must be a positive integer, got {value!r}")
    return value


def _policy_endpoint(exp_cfg: dict[str, Any], args: argparse.Namespace) -> tuple[str, int]:
    rollout = _get_required_dict(exp_cfg, "rollout")
    host = args.policy_host or rollout.get("policy_host") or "localhost"
    port = args.policy_port if args.policy_port is not None else rollout.get("policy_port", 8000)
    if not isinstance(host, str) or not host:
        raise ValueError(f"policy host must be a non-empty string, got {host!r}")
    if not isinstance(port, int) or port <= 0:
        raise ValueError(f"policy port must be a positive integer, got {port!r}")
    return host, port


def _apply_rtc_config(exp_cfg: dict[str, Any], args: argparse.Namespace) -> None:
    rollout = _get_required_dict(exp_cfg, "rollout")
    rtc = rollout.get("rtc")
    if rtc is not None:
        if not isinstance(rtc, dict):
            raise ValueError("rollout.rtc must be an object when provided")
        if args.rtc is None and rtc.get("enabled") is not None:
            args.rtc = bool(rtc["enabled"])
        if args.rtc_latency_k is None and rtc.get("latency_k") is not None:
            args.rtc_latency_k = int(rtc["latency_k"])
        if args.rtc_inference_rate is None and rtc.get("inference_rate") is not None:
            args.rtc_inference_rate = float(rtc["inference_rate"])
        if args.rtc_max_guidance_weight is None and rtc.get("max_guidance_weight") is not None:
            args.rtc_max_guidance_weight = float(rtc["max_guidance_weight"])
    apply_rtc_defaults(args)
    if args.rtc_latency_k < 0:
        raise ValueError(f"--rtc-latency-k must be non-negative, got {args.rtc_latency_k}")
    if args.rtc_inference_rate <= 0:
        raise ValueError(f"--rtc-inference-rate must be positive, got {args.rtc_inference_rate}")
    if args.rtc_max_guidance_weight <= 0:
        raise ValueError(f"--rtc-max-guidance-weight must be positive, got {args.rtc_max_guidance_weight}")


def _create_policy(args: argparse.Namespace, policy_host: str, policy_port: int, actions_per_infer: int):
    if args.rtc:
        return rtc_client_policy.RtcWebsocketClientPolicy(
            host=policy_host,
            port=policy_port,
            latency_k=args.rtc_latency_k,
            execution_horizon=actions_per_infer,
            max_guidance_weight=args.rtc_max_guidance_weight,
        )
    return websocket_client_policy.WebsocketClientPolicy(host=policy_host, port=policy_port)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _image_to_rgb_uint8(value: Any) -> np.ndarray:
    image = _as_numpy(value)
    if image.ndim != 3:
        raise ValueError(f"expected image with 3 dims, got shape {image.shape}")
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"expected RGB image with 3 channels, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    return image.astype(np.uint8)


def _encode_jpeg(image: np.ndarray, output_path: Path, quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = image[..., ::-1]
    ok = cv2.imwrite(str(output_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError(f"failed to encode JPEG image: {output_path}")


def _sample_for_policy(sample: dict[str, Any]) -> dict[str, Any]:
    obs = {
        "images": {camera: _to_msgpack_value(sample[IMAGE_KEYS[camera]]) for camera in CAMERAS},
        "state": _to_msgpack_value(sample["observation.state"]),
    }
    task = sample.get("task")
    if task is not None:
        obs["prompt"] = np.asarray(task)
    return obs


def _to_msgpack_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _to_msgpack_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_to_msgpack_value(item) for item in value)
    return value


def _feature_action_names(metadata: LeRobotDatasetMetadata, action_dim: int) -> list[str]:
    features = getattr(metadata, "features", None)
    names = None
    if isinstance(features, dict):
        action_feature = features.get("action")
        if isinstance(action_feature, dict):
            names = action_feature.get("names")

    if not isinstance(names, list | tuple) or len(names) != action_dim:
        return [f"action_{idx}" for idx in range(action_dim)]
    return [str(name) for name in names]


def _plot_order(action_names: list[str]) -> list[int]:
    grippers = [idx for idx, name in enumerate(action_names) if "gripper" in name.lower()]
    others = [idx for idx, name in enumerate(action_names) if "gripper" not in name.lower()]
    return [*grippers, *others]


def _episode_infos(metadata: LeRobotDatasetMetadata) -> list[EpisodeInfo]:
    episodes = getattr(metadata, "episodes", None)
    if isinstance(episodes, dict):
        return [
            EpisodeInfo(
                episode_index=int(index),
                length=int(episode.get("length", 0)),
                task=(episode.get("tasks") or [None])[0],
            )
            for index, episode in sorted(episodes.items())
        ]

    infos = []
    for episode_index in range(metadata.total_episodes):
        dataset = LeRobotDataset(metadata.repo_id, root=metadata.root, episodes=[episode_index])
        task = dataset[0].get("task") if len(dataset) > 0 else None
        infos.append(EpisodeInfo(episode_index=episode_index, length=len(dataset), task=task))
    return infos


def _episode_payload(info: EpisodeInfo) -> dict[str, Any]:
    return {"episode_index": info.episode_index, "length": info.length, "task": info.task}


def _run_open_loop(
    dataset: LeRobotDataset,
    policy: websocket_client_policy.WebsocketClientPolicy,
    actions_per_infer: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []
    for start in range(0, len(dataset), actions_per_infer):
        segment_len = min(actions_per_infer, len(dataset) - start)
        sample = dataset[start]
        output = policy.infer(_sample_for_policy(sample))
        pred_chunk = _as_numpy(output["actions"])
        if pred_chunk.ndim != 2:
            raise ValueError(f"policy output actions must have shape (horizon, dim), got {pred_chunk.shape}")
        if pred_chunk.shape[0] < segment_len:
            raise ValueError(
                f"policy output horizon {pred_chunk.shape[0]} is shorter than required segment length {segment_len}"
            )

        gt_chunk = np.stack([_as_numpy(dataset[idx]["action"]) for idx in range(start, start + segment_len)], axis=0)
        pred_chunk = pred_chunk[:segment_len]
        if pred_chunk.shape[-1] != gt_chunk.shape[-1]:
            raise ValueError(f"action dim mismatch: predict {pred_chunk.shape[-1]} vs gt {gt_chunk.shape[-1]}")
        pred_chunks.append(pred_chunk.astype(np.float32))
        gt_chunks.append(gt_chunk.astype(np.float32))
        print(f"processed frames {start}:{start + segment_len}", flush=True)

    pred = np.concatenate(pred_chunks, axis=0)
    gt = np.concatenate(gt_chunks, axis=0)
    if pred.shape != gt.shape:
        raise ValueError(f"predict and gt shapes differ after concatenation: {pred.shape} vs {gt.shape}")
    return pred, gt


def create_app(args: argparse.Namespace) -> Flask:
    exp_path, exp_cfg = _load_experiment(args.experiment, args.config)
    action_space = _select_action_space(exp_cfg, args.action_space)
    actions_per_infer = _get_actions_per_infer(exp_cfg)
    policy_host, policy_port = _policy_endpoint(exp_cfg, args)
    _apply_rtc_config(exp_cfg, args)
    action_cfg = _get_required_dict(_get_required_dict(exp_cfg, "action_spaces"), action_space)

    repo_id = action_cfg.get("repo_id")
    lerobot_dir = action_cfg.get("lerobot_dir")
    train_config = action_cfg.get("train_config")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError(f"action_spaces.{action_space}.repo_id must be a non-empty string")
    if not isinstance(lerobot_dir, str) or not lerobot_dir:
        raise ValueError(f"action_spaces.{action_space}.lerobot_dir must be a non-empty string")

    dataset_root = Path(lerobot_dir).expanduser()
    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    episode_infos = _episode_infos(metadata)
    if not episode_infos:
        raise ValueError(f"dataset has no episodes: {repo_id}")
    temp_root = Path(tempfile.mkdtemp(prefix="r1lite_open_loop_viewer_"))
    loaded: dict[int, LoadedEpisode] = {}
    policy = _create_policy(args, policy_host, policy_port, actions_per_infer)

    def cleanup() -> None:
        shutil.rmtree(temp_root, ignore_errors=True)

    atexit.register(cleanup)
    app = Flask(__name__)

    def require_loaded_episode(episode_index: int) -> LoadedEpisode:
        episode = loaded.get(episode_index)
        if episode is None:
            raise KeyError(f"episode is not loaded: {episode_index}")
        return episode

    def frame_image_path(episode_index: int, loaded_episode: LoadedEpisode, index: int, camera: str) -> Path:
        output_path = loaded_episode.image_dir / f"{index:06d}_{camera}.jpg"
        if output_path.exists():
            return output_path
        if camera not in CAMERAS:
            raise KeyError(f"unsupported camera: {camera}")
        sample = loaded_episode.dataset[index]
        image = _image_to_rgb_uint8(sample[IMAGE_KEYS[camera]])
        _encode_jpeg(image, output_path, args.jpeg_quality)
        return output_path

    @app.get("/")
    def index() -> Response:
        return Response(HTML, mimetype="text/html")

    @app.get("/api/episodes")
    def list_episodes() -> Response | tuple[Response, int]:
        try:
            return jsonify(
                {
                    "experiment": args.experiment or exp_path.parent.name,
                    "action_space": action_space,
                    "actions_per_infer": actions_per_infer,
                    "fps": metadata.fps,
                    "repo_id": repo_id,
                    "lerobot_dir": str(dataset_root),
                    "train_config": train_config,
                    "policy_host": policy_host,
                    "policy_port": policy_port,
                    "rtc": {
                        "enabled": args.rtc,
                        "latency_k": args.rtc_latency_k,
                        "inference_rate": args.rtc_inference_rate,
                        "max_guidance_weight": args.rtc_max_guidance_weight,
                    },
                    "episodes": [_episode_payload(info) for info in episode_infos],
                }
            )
        except Exception as exc:
            return _json_error(str(exc), 500)

    @app.post("/api/episodes/<int:episode_index>/load")
    def load_episode(episode_index: int) -> Response | tuple[Response, int]:
        try:
            if episode_index < 0 or episode_index >= metadata.total_episodes:
                return _json_error(f"episode index out of range: {episode_index}", 400)
            dataset = LeRobotDataset(repo_id, root=dataset_root, episodes=[episode_index])
            if len(dataset) == 0:
                raise ValueError(f"episode {episode_index} has no frames")
            policy.reset()
            pred, gt = _run_open_loop(dataset, policy, actions_per_infer)
            action_names = _feature_action_names(metadata, gt.shape[1])
            image_dir = temp_root / f"episode_{episode_index:06d}"
            if image_dir.exists():
                shutil.rmtree(image_dir)
            image_dir.mkdir(parents=True)
            loaded[episode_index] = LoadedEpisode(
                dataset=dataset,
                image_dir=image_dir,
                pred=pred,
                gt=gt,
                action_names=action_names,
                plot_order=_plot_order(action_names),
            )
            return jsonify(
                {
                    "episode_index": episode_index,
                    "frame_count": len(dataset),
                    "actions_per_infer": actions_per_infer,
                    "rtc_enabled": args.rtc,
                    "action_dim": gt.shape[1],
                }
            )
        except Exception as exc:
            loaded.pop(episode_index, None)
            return _json_error(str(exc), 500)

    @app.get("/api/episodes/<int:episode_index>/actions")
    def episode_actions(episode_index: int) -> Response | tuple[Response, int]:
        try:
            episode = require_loaded_episode(episode_index)
            return jsonify(
                {
                    "episode_index": episode_index,
                    "pred": episode.pred.tolist(),
                    "gt": episode.gt.tolist(),
                    "action_names": episode.action_names,
                    "plot_order": episode.plot_order,
                    "actions_per_infer": actions_per_infer,
                }
            )
        except KeyError as exc:
            return _json_error(str(exc), 404)
        except Exception as exc:
            return _json_error(str(exc), 500)

    @app.get("/api/episodes/<int:episode_index>/frames/<int:index>")
    def frame_metadata(episode_index: int, index: int) -> Response | tuple[Response, int]:
        try:
            episode = require_loaded_episode(episode_index)
            if index < 0 or index >= len(episode.dataset):
                return _json_error(f"frame index out of range: {index}", 400)
            sample = episode.dataset[index]
            images = {}
            for camera in CAMERAS:
                frame_image_path(episode_index, episode, index, camera)
                images[camera] = f"/api/episodes/{episode_index}/frames/{index}/{camera}.jpg"
            return jsonify(
                {
                    "episode_index": episode_index,
                    "index": index,
                    "frame_count": len(episode.dataset),
                    "timestamp": float(_as_numpy(sample["timestamp"])),
                    "frame_index": int(_as_numpy(sample["frame_index"])),
                    "predict": episode.pred[index].tolist(),
                    "gt": episode.gt[index].tolist(),
                    "images": images,
                }
            )
        except KeyError as exc:
            return _json_error(str(exc), 404)
        except Exception as exc:
            return _json_error(str(exc), 500)

    @app.get("/api/episodes/<int:episode_index>/frames/<int:index>/<camera>.jpg")
    def frame_image(episode_index: int, index: int, camera: str) -> Response | tuple[Response, int]:
        try:
            episode = require_loaded_episode(episode_index)
            if index < 0 or index >= len(episode.dataset):
                return _json_error(f"frame index out of range: {index}", 400)
            if camera not in CAMERAS:
                return _json_error(f"unsupported camera: {camera}", 400)
            image_path = frame_image_path(episode_index, episode, index, camera)
            return Response(image_path.read_bytes(), mimetype="image/jpeg")
        except KeyError as exc:
            return _json_error(str(exc), 404)
        except Exception as exc:
            return _json_error(str(exc), 500)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=None, help="Experiment name under experiments/r1lite/<name>.")
    parser.add_argument("--config", default=None, help="Path to an explicit R1Lite experiment config.yaml.")
    parser.add_argument(
        "--action-space",
        default=None,
        choices=("joint_delta", "abs_eef", "delta_eef"),
        help="Action-space section to use. Required when the experiment config has multiple action spaces.",
    )
    parser.add_argument("--policy-host", default=None, help="Policy server host. Defaults to rollout.policy_host.")
    parser.add_argument("--policy-port", type=int, default=None, help="Policy server port. Defaults to rollout.policy_port.")
    parser.add_argument("--host", default="127.0.0.1", help="Web server bind host.")
    parser.add_argument("--port", type=int, default=7861, help="Web server bind port.")
    parser.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality for served video frames.")
    add_rtc_args(parser)
    args = parser.parse_args()
    if args.port <= 0:
        raise ValueError(f"--port must be positive, got {args.port}")
    if args.policy_port is not None and args.policy_port <= 0:
        raise ValueError(f"--policy-port must be positive, got {args.policy_port}")
    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        raise ValueError(f"--jpeg-quality must be in [1, 100], got {args.jpeg_quality}")
    return args


def main() -> None:
    args = parse_args()
    app = create_app(args)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
