#!/usr/bin/env python3
"""Browser monitor for the R1Lite robot service."""

from __future__ import annotations

import argparse
import json
from typing import Any
import urllib.error
import urllib.request

from flask import Flask
from flask import Response
from flask import jsonify
from flask import request

DEFAULT_RESET_TORSO = "-0.607 1.394 0.649"

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>R1Lite Monitor</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f6f8;
      --panel: #ffffff;
      --text: #18212f;
      --muted: #657083;
      --line: #d7dde7;
      --accent: #1f6feb;
      --accent-dark: #174ea6;
      --danger: #b42318;
      --ok: #087443;
      --warn: #b54708;
      --panel-gap: 12px;
      --image-height: 360px;
      --shadow: 0 1px 3px rgba(16, 24, 40, 0.10);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 13px;
      line-height: 1.35;
    }

    button,
    input {
      font: inherit;
    }

    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }

    .topbar {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 14px;
      align-items: center;
      padding: 12px 16px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      box-shadow: var(--shadow);
      position: sticky;
      top: 0;
      z-index: 10;
    }

    .brand {
      font-size: 16px;
      font-weight: 750;
      white-space: nowrap;
    }

    .summary {
      display: grid;
      grid-template-columns: repeat(7, minmax(110px, 1fr));
      gap: 8px;
      min-width: 0;
    }

    .summary-item {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 8px;
      background: #fbfcfe;
      min-width: 0;
    }

    .summary-label {
      color: var(--muted);
      font-size: 11px;
      white-space: nowrap;
    }

    .summary-value {
      font-weight: 650;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .status {
      min-width: 180px;
      color: var(--muted);
      text-align: right;
      white-space: nowrap;
    }

    .content {
      padding: var(--panel-gap);
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      gap: var(--panel-gap);
    }

    .controls {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: var(--panel-gap);
      align-items: stretch;
    }

    .toolbar,
    .sliderbar {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      box-shadow: var(--shadow);
    }

    .sliderbar label {
      display: grid;
      grid-template-columns: auto 130px auto;
      gap: 8px;
      align-items: center;
      color: var(--muted);
    }

    input[type="range"] {
      accent-color: var(--accent);
    }

    .button {
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      color: var(--text);
      cursor: pointer;
      padding: 0 10px;
      white-space: nowrap;
    }

    .button:hover {
      border-color: var(--accent);
    }

    .button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }

    .button.primary:hover {
      background: var(--accent-dark);
    }

    .button.danger {
      color: var(--danger);
    }

    .button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }

    .camera-grid {
      display: grid;
      grid-template-columns: 2fr 1fr 1fr;
      gap: var(--panel-gap);
      align-items: start;
    }

    .state-grid {
      display: grid;
      grid-template-columns: 1fr 0.82fr 1fr;
      gap: var(--panel-gap);
      min-height: 0;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: var(--shadow);
      min-width: 0;
    }

    .panel-header {
      min-height: 34px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 0 10px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
      color: var(--muted);
      font-weight: 700;
    }

    .panel-body {
      padding: 10px;
      min-width: 0;
    }

    .camera .panel-body {
      padding: 0;
      background: #101828;
    }

    .camera img {
      display: block;
      width: 100%;
      height: var(--image-height);
      object-fit: contain;
      background: #101828;
    }

    .missing-image {
      height: var(--image-height);
      display: grid;
      place-items: center;
      color: #cbd5e1;
      background: #101828;
    }

    .split {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      line-height: 1.45;
    }

    .logs {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: var(--panel-gap);
    }

    .log-body {
      height: 150px;
      overflow: auto;
    }

    .ok {
      color: var(--ok);
    }

    .warn {
      color: var(--warn);
    }

    .error {
      color: var(--danger);
      font-weight: 700;
    }

    @media (max-width: 1180px) {
      .topbar,
      .controls,
      .camera-grid,
      .state-grid,
      .logs {
        grid-template-columns: 1fr;
      }

      .summary {
        grid-template-columns: repeat(2, minmax(120px, 1fr));
      }

      .status {
        text-align: left;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <div class="brand">R1Lite Monitor</div>
      <div class="summary" id="summary"></div>
      <div class="status" id="connection">connecting</div>
    </header>

    <main class="content">
      <section class="controls">
        <div class="toolbar">
          <button class="button" data-action="brake-toggle">Brake Toggle</button>
          <button class="button" data-action="reset">Reset</button>
          <button class="button" data-action="clear-fault">Clear Fault</button>
          <button class="button" data-action="recover">Recover</button>
        </div>
        <div class="sliderbar">
          <label>Panel gap <input id="gapRange" type="range" min="0" max="32" value="12"><span id="gapValue">12px</span></label>
          <label>Image height <input id="heightRange" type="range" min="180" max="760" value="360"><span id="heightValue">360px</span></label>
        </div>
      </section>

      <section class="camera-grid" id="cameraGrid"></section>

      <section class="state-grid">
        <article class="panel">
          <div class="panel-header"><span>Left Arm</span><span id="leftValidity"></span></div>
          <div class="panel-body split">
            <pre id="leftCommand">waiting for data</pre>
            <pre id="leftState">waiting for data</pre>
          </div>
        </article>

        <article class="panel">
          <div class="panel-header"><span>Torso</span><span id="torsoValidity"></span></div>
          <div class="panel-body">
            <pre id="torsoState">waiting for data</pre>
          </div>
        </article>

        <article class="panel">
          <div class="panel-header"><span>Right Arm</span><span id="rightValidity"></span></div>
          <div class="panel-body split">
            <pre id="rightCommand">waiting for data</pre>
            <pre id="rightState">waiting for data</pre>
          </div>
        </article>
      </section>

      <section class="logs">
        <article class="panel">
          <div class="panel-header">Info</div>
          <div class="panel-body log-body"><pre id="infoLog">none</pre></div>
        </article>
        <article class="panel">
          <div class="panel-header">Warnings</div>
          <div class="panel-body log-body"><pre id="warnLog">none</pre></div>
        </article>
        <article class="panel">
          <div class="panel-header">Faults</div>
          <div class="panel-body log-body"><pre id="faultLog">none</pre></div>
        </article>
      </section>
    </main>
  </div>

  <script>
    const CONFIG = __CONFIG__;
    const CAMERAS = ["head", "left_wrist", "right_wrist"];
    const summaryItems = [
      ["Server", "server"],
      ["Mode", "mode"],
      ["Owner", "owner"],
      ["Teleop", "teleop_source"],
      ["Cmd Age", "last_command_age"],
      ["Brake", "brake"],
      ["Updated", "updated"],
    ];
    const logs = { info: [], warn: [], fault: [] };
    const last = {
      owner: undefined,
      teleop: undefined,
      brake: undefined,
      warningKeys: new Set(),
      faults: [],
    };
    let latestPacket = null;
    let latestHealth = null;
    let lastTextUpdate = 0;
    let inFlightState = false;
    let inFlightHealth = false;
    let actionBusy = false;

    function el(id) {
      return document.getElementById(id);
    }

    function nowTime() {
      return new Date().toLocaleTimeString("en-GB", { hour12: false });
    }

    function scalar(value) {
      if (value === null || value === undefined) return "-";
      if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(3) : String(value);
      return String(value);
    }

    function vector(value) {
      if (value === null || value === undefined) return "-";
      const arr = Array.isArray(value) ? value.flat(Infinity) : [value];
      return "[" + arr.map((v) => {
        const n = Number(v);
        return Number.isFinite(n) ? n.toFixed(3).padStart(7, " ") : String(v);
      }).join(", ") + "]";
    }

    function quatToEulerXyz(q) {
      if (!Array.isArray(q) || q.length < 4) return null;
      const [x, y, z, w] = q.map(Number);
      const n = Math.hypot(x, y, z, w);
      if (!Number.isFinite(n) || n <= 1e-12) return null;
      const qx = x / n, qy = y / n, qz = z / n, qw = w / n;
      const sinr = 2 * (qw * qx + qy * qz);
      const cosr = 1 - 2 * (qx * qx + qy * qy);
      const roll = Math.atan2(sinr, cosr);
      const sinp = 2 * (qw * qy - qz * qx);
      const pitch = Math.abs(sinp) >= 1 ? Math.sign(sinp) * Math.PI / 2 : Math.asin(sinp);
      const siny = 2 * (qw * qz + qx * qy);
      const cosy = 1 - 2 * (qy * qy + qz * qz);
      const yaw = Math.atan2(siny, cosy);
      return [roll, pitch, yaw];
    }

    function pose(value) {
      if (!Array.isArray(value) || value.length !== 7) return vector(value);
      const xyz = value.slice(0, 3);
      const quat = value.slice(3, 7);
      const euler = quatToEulerXyz(quat);
      let out = "xyz=" + vector(xyz) + "\nquat=" + vector(quat);
      if (euler) out += "\nrpy=" + vector(euler);
      return out;
    }

    function validityText(validity) {
      if (!validity || typeof validity !== "object") return "";
      const bad = Object.entries(validity).filter(([, value]) => value === false).map(([key]) => key);
      if (bad.length === 0) return "valid";
      return "invalid: " + bad.join(", ");
    }

    function addLog(level, text) {
      const line = `[${nowTime()}] ${text}`;
      const bucket = logs[level];
      if (bucket[bucket.length - 1] !== line) bucket.push(line);
      if (bucket.length > 120) bucket.splice(0, bucket.length - 120);
      el(level === "info" ? "infoLog" : level === "warn" ? "warnLog" : "faultLog").textContent =
        bucket.length ? bucket.join("\n") : "none";
    }

    function renderSummary(values) {
      const root = el("summary");
      if (!root.dataset.ready) {
        root.innerHTML = summaryItems.map(([label, key]) =>
          `<div class="summary-item"><div class="summary-label">${label}</div><div class="summary-value" id="summary-${key}">-</div></div>`
        ).join("");
        root.dataset.ready = "1";
      }
      for (const [, key] of summaryItems) {
        el(`summary-${key}`).textContent = values[key] ?? "-";
      }
    }

    function renderCameras(images) {
      const grid = el("cameraGrid");
      if (!grid.dataset.ready) {
        grid.innerHTML = CAMERAS.map((key) =>
          `<article class="panel camera"><div class="panel-header">${key}</div><div class="panel-body" id="camera-${key}"></div></article>`
        ).join("");
        grid.dataset.ready = "1";
      }
      for (const key of CAMERAS) {
        const container = el(`camera-${key}`);
        const encoded = images && images[key];
        if (typeof encoded === "string" && encoded.length > 0) {
          container.innerHTML = `<img alt="${key}" src="data:image/jpeg;base64,${encoded}">`;
        } else {
          container.innerHTML = `<div class="missing-image">${key} missing</div>`;
        }
      }
    }

    function formatCommand(command, owner, teleopSource) {
      command = command && typeof command === "object" ? command : {};
      const lines = [
        `Command`,
        `owner: ${scalar(owner)}`,
        `teleop_source: ${scalar(teleopSource)}`,
        `preset: ${scalar(command.preset)}`,
        `gripper_target: ${scalar(command.gripper)}`,
        `tcp_target:\n${pose(command.desired_pose)}`,
        `joint_target: ${vector(command.desired_joint)}`,
        `updated_at: ${scalar(command.updated_at)}`,
        `last_sent:\n${scalar(command.last_sent_target).replaceAll("; ", "\n")}`,
      ];
      return lines.join("\n");
    }

    function formatArmState(side, arm, packet) {
      arm = arm && typeof arm === "object" ? arm : {};
      const validity = packet?.meta?.validity?.[side] ?? {};
      const lines = [
        `State`,
        `joint_pos: ${vector(arm.joint_pos)}`,
        `joint_vel: ${vector(arm.joint_vel)}`,
        `joint_effort: ${vector(arm.joint_effort)}`,
        `gripper_pose: ${vector(arm.gripper_pose)}`,
        `tcp_pose:\n${pose(arm.tcp_pose)}`,
        `tcp_vel: ${vector(arm.tcp_vel)}`,
        `tcp_force: ${vector(arm.tcp_force)}`,
        `tcp_torque: ${vector(arm.tcp_torque)}`,
        `state_preset: ${scalar(arm.preset)}`,
        `validity: ${JSON.stringify(validity)}`,
      ];
      return lines.join("\n");
    }

    function formatTorso(torso, packet) {
      torso = torso && typeof torso === "object" ? torso : {};
      const validity = packet?.meta?.validity?.torso ?? {};
      return [
        `joint_pos: ${vector(torso.joint_pos)}`,
        `joint_vel: ${vector(torso.joint_vel)}`,
        `joint_effort: ${vector(torso.joint_effort)}`,
        `validity: ${JSON.stringify(validity)}`,
      ].join("\n");
    }

    function syncLogs(packet, health) {
      const meta = packet.meta || {};
      const owner = meta.command_owner;
      const teleop = meta.active_teleop_source;
      const brake = meta.brake_enabled;
      if (owner !== last.owner) {
        addLog("info", `command owner changed: ${scalar(last.owner)} -> ${scalar(owner)}`);
        last.owner = owner;
      }
      if (teleop !== last.teleop) {
        addLog("info", `teleop source changed: ${scalar(last.teleop)} -> ${scalar(teleop)}`);
        last.teleop = teleop;
      }
      if (brake !== last.brake) {
        addLog("info", `brake ${brake ? "enabled" : "disabled"}`);
        last.brake = brake;
      }

      const warningKeys = [];
      const freshness = health?.freshness || {};
      const validity = meta.validity || {};
      for (const [key, value] of Object.entries(freshness)) if (value === false) warningKeys.push(`freshness:${key}`);
      for (const side of ["left", "right"]) {
        for (const [key, value] of Object.entries(validity[side] || {})) {
          if (value === false) warningKeys.push(`validity:${side}:${key}`);
        }
      }
      for (const [key, value] of Object.entries(validity.torso || {})) if (value === false) warningKeys.push(`validity:torso:${key}`);
      for (const [key, value] of Object.entries(validity.images || {})) if (value === false) warningKeys.push(`validity:image:${key}`);

      const current = new Set(warningKeys);
      for (const key of [...current].sort()) if (!last.warningKeys.has(key)) addLog("warn", key);
      for (const key of [...last.warningKeys].sort()) if (!current.has(key)) addLog("info", `warning cleared: ${key}`);
      last.warningKeys = current;

      const faults = Array.isArray(health?.faults) ? health.faults.map(String).sort() : [];
      if (JSON.stringify(faults) !== JSON.stringify(last.faults)) {
        if (faults.length === 0 && last.faults.length > 0) addLog("info", "all faults cleared");
        for (const fault of faults) if (!last.faults.includes(fault)) addLog("fault", fault);
        last.faults = faults;
      }
    }

    function renderText(packet, health) {
      const meta = packet.meta || {};
      const currentHealth = health || meta.health || {};
      renderSummary({
        server: CONFIG.serverUrl,
        mode: scalar(meta.mode),
        owner: scalar(meta.command_owner),
        teleop_source: scalar(meta.active_teleop_source),
        last_command_age: scalar(currentHealth.last_command_age_sec),
        brake: meta.brake_enabled ? "enabled" : "disabled",
        updated: nowTime(),
      });
      const commands = meta.commands || {};
      const state = packet.state || {};
      el("leftCommand").textContent = formatCommand(commands.left, meta.command_owner, meta.active_teleop_source);
      el("rightCommand").textContent = formatCommand(commands.right, meta.command_owner, meta.active_teleop_source);
      el("leftState").textContent = formatArmState("left", state.left, packet);
      el("rightState").textContent = formatArmState("right", state.right, packet);
      el("torsoState").textContent = formatTorso(state.torso, packet);
      el("leftValidity").textContent = validityText(meta.validity?.left);
      el("rightValidity").textContent = validityText(meta.validity?.right);
      el("torsoValidity").textContent = validityText(meta.validity?.torso);
      syncLogs(packet, currentHealth);
    }

    async function requestJson(url, options = {}) {
      const response = await fetch(url, {
        headers: { "Accept": "application/json", "Content-Type": "application/json" },
        ...options,
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(body.error || `${response.status} ${response.statusText}`);
      }
      return body;
    }

    async function pollState() {
      if (inFlightState) return;
      inFlightState = true;
      try {
        const [robotPacket, imagePacket] = await Promise.all([
          requestJson("/api/state/robot"),
          requestJson("/api/state/images"),
        ]);
        latestPacket = {
          ...robotPacket,
          images: imagePacket.images || {},
          meta: {
            ...(robotPacket.meta || {}),
            image_meta: imagePacket.meta || {},
          },
        };
        renderCameras(latestPacket.images || {});
        el("connection").textContent = "connected";
        el("connection").className = "status ok";
        const now = performance.now();
        if (now - lastTextUpdate >= CONFIG.statePeriodMs) {
          lastTextUpdate = now;
          renderText(latestPacket, latestHealth);
        }
      } catch (error) {
        el("connection").textContent = error.message;
        el("connection").className = "status error";
        addLog("fault", `state request error: ${error.message}`);
      } finally {
        inFlightState = false;
      }
    }

    async function pollHealth() {
      if (inFlightHealth) return;
      inFlightHealth = true;
      try {
        latestHealth = await requestJson("/api/health");
      } catch (error) {
        addLog("fault", `health request error: ${error.message}`);
      } finally {
        inFlightHealth = false;
      }
    }

    async function sendAction(url, payload = {}) {
      if (actionBusy) return;
      actionBusy = true;
      document.querySelectorAll("button").forEach((button) => button.disabled = true);
      try {
        const result = await requestJson(url, { method: "POST", body: JSON.stringify(payload) });
        addLog("info", `action accepted: ${JSON.stringify(result)}`);
        await pollState();
        await pollHealth();
      } catch (error) {
        addLog("fault", `action failed: ${error.message}`);
      } finally {
        actionBusy = false;
        document.querySelectorAll("button").forEach((button) => button.disabled = false);
      }
    }

    function bindControls() {
      el("gapRange").addEventListener("input", (event) => {
        const value = `${event.target.value}px`;
        document.documentElement.style.setProperty("--panel-gap", value);
        el("gapValue").textContent = value;
      });
      el("heightRange").addEventListener("input", (event) => {
        const value = `${event.target.value}px`;
        document.documentElement.style.setProperty("--image-height", value);
        el("heightValue").textContent = value;
      });

      document.querySelector("[data-action='brake-toggle']").addEventListener("click", () => {
        const enabled = !(latestPacket?.meta?.brake_enabled === true);
        sendAction("/api/brake", { enabled });
      });
      document.querySelector("[data-action='reset']").addEventListener("click", () => sendAction("/api/reset"));
      document.querySelector("[data-action='clear-fault']").addEventListener("click", () => sendAction("/api/clear_fault"));
      document.querySelector("[data-action='recover']").addEventListener("click", () => sendAction("/api/recover"));
    }

    renderSummary({
      server: CONFIG.serverUrl,
      mode: "-",
      owner: "-",
      teleop_source: "-",
      last_command_age: "-",
      brake: "-",
      updated: "-",
    });
    bindControls();
    pollState();
    pollHealth();
    setInterval(pollState, CONFIG.imagePeriodMs);
    setInterval(pollHealth, CONFIG.statePeriodMs);
  </script>
</body>
</html>
"""


class RobotClient:
    def __init__(self, server_url: str, timeout: float):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.server_url}/{path.lstrip('/')}"

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self._url(path), data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {self._url(path)} failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{method} {self._url(path)} failed: {exc}") from exc

        if not body:
            return {}
        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{method} {self._url(path)} returned non-JSON response: {body[:500]}") from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"{method} {self._url(path)} returned JSON {type(result).__name__}, expected object")
        return result

    def get_state(self) -> dict[str, Any]:
        return self.request("GET", "/state")

    def get_robot_state(self) -> dict[str, Any]:
        return self.request("GET", "/state/robot")

    def get_images(self) -> dict[str, Any]:
        return self.request("GET", "/state/images")

    def get_health(self) -> dict[str, Any]:
        return self.request("GET", "/health")

    def brake(self, *, enabled: bool, owner: str) -> dict[str, Any]:
        return self.request("POST", "/brake", {"enabled": bool(enabled), "owner": owner})

    def reset(
        self,
        owner: str,
        left_joint: list[float] | None,
        right_joint: list[float] | None,
        torso: list[float] | None,
        *,
        left_gripper: float | None,
        right_gripper: float | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"owner": owner}
        if left_joint is not None:
            payload["left_joint"] = left_joint
        if right_joint is not None:
            payload["right_joint"] = right_joint
        if torso is not None:
            payload["torso"] = torso
        if left_gripper is not None:
            payload["left_gripper"] = float(left_gripper)
        if right_gripper is not None:
            payload["right_gripper"] = float(right_gripper)
        return self.request("POST", "/reset", payload)

    def clear_fault(self, owner: str) -> dict[str, Any]:
        return self.request("POST", "/clear_fault", {"owner": owner})

    def recover(self, owner: str) -> dict[str, Any]:
        return self.request("POST", "/recover", {"owner": owner})


def _json_error(message: str, status: int) -> tuple[Response, int]:
    return jsonify({"error": message}), status


def _parse_float_vector(text: str | None, expected_len: int, name: str) -> list[float] | None:
    if text is None or str(text).strip() == "":
        return None
    raw = str(text).replace("[", " ").replace("]", " ").replace(",", " ").split()
    if len(raw) != expected_len:
        raise ValueError(f"{name} expects {expected_len} floats, got {len(raw)}")
    return [float(item) for item in raw]


def _request_json_body() -> dict[str, Any]:
    data = request.get_json(silent=False)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("JSON request body must be an object")
    return data


def create_app(args: argparse.Namespace) -> Flask:
    client = RobotClient(args.server_url, args.timeout)
    reset_left_joint = _parse_float_vector(args.reset_left_joint, 6, "--reset-left-joint")
    reset_right_joint = _parse_float_vector(args.reset_right_joint, 6, "--reset-right-joint")
    reset_torso = _parse_float_vector(args.reset_torso, 3, "--reset-torso")
    app = Flask(__name__)

    @app.get("/")
    def index() -> Response:
        config = {
            "serverUrl": args.server_url.rstrip("/"),
            "imagePeriodMs": int(max(1.0 / max(args.image_hz, 1e-3), 0.05) * 1000),
            "statePeriodMs": int(max(args.state_period, 0.1) * 1000),
        }
        html = HTML.replace("__CONFIG__", json.dumps(config))
        return Response(html, mimetype="text/html")

    @app.get("/api/state")
    def state() -> Response | tuple[Response, int]:
        try:
            return jsonify(client.get_state())
        except Exception as exc:
            return _json_error(str(exc), 502)

    @app.get("/api/state/robot")
    def robot_state() -> Response | tuple[Response, int]:
        try:
            return jsonify(client.get_robot_state())
        except Exception as exc:
            return _json_error(str(exc), 502)

    @app.get("/api/state/images")
    def images() -> Response | tuple[Response, int]:
        try:
            return jsonify(client.get_images())
        except Exception as exc:
            return _json_error(str(exc), 502)

    @app.get("/api/health")
    def health() -> Response | tuple[Response, int]:
        try:
            return jsonify(client.get_health())
        except Exception as exc:
            return _json_error(str(exc), 502)

    @app.post("/api/brake")
    def brake() -> Response | tuple[Response, int]:
        try:
            payload = _request_json_body()
            if "enabled" not in payload:
                return _json_error("Missing required field: enabled", 400)
            return jsonify(client.brake(enabled=bool(payload["enabled"]), owner=args.maintenance_owner))
        except Exception as exc:
            return _json_error(str(exc), 502)

    @app.post("/api/reset")
    def reset() -> Response | tuple[Response, int]:
        try:
            reset_response = client.reset(
                args.maintenance_owner,
                reset_left_joint,
                reset_right_joint,
                reset_torso,
                left_gripper=float(args.gripper_open_value),
                right_gripper=float(args.gripper_open_value),
            )
            return jsonify({"reset": reset_response})
        except Exception as exc:
            return _json_error(str(exc), 502)

    @app.post("/api/clear_fault")
    def clear_fault() -> Response | tuple[Response, int]:
        try:
            return jsonify(client.clear_fault(args.maintenance_owner))
        except Exception as exc:
            return _json_error(str(exc), 502)

    @app.post("/api/recover")
    def recover() -> Response | tuple[Response, int]:
        try:
            return jsonify(client.recover(args.maintenance_owner))
        except Exception as exc:
            return _json_error(str(exc), 502)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://192.168.12.12:8001/")
    parser.add_argument("--host", default="127.0.0.1", help="Host for this browser GUI server.")
    parser.add_argument("--port", type=int, default=8010, help="Port for this browser GUI server.")
    parser.add_argument("--image-hz", type=float, default=5.0, help="Polling rate for image/state packets.")
    parser.add_argument("--state-period", type=float, default=0.5, help="Refresh period for text and health panels.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Robot HTTP request timeout.")
    parser.add_argument("--maintenance-owner", default="debug", help="Owner string used for maintenance actions.")
    parser.add_argument("--gripper-open-value", type=float, default=100.0)
    parser.add_argument("--reset-left-joint", default=None, help="Optional 6 floats for reset left arm joint target.")
    parser.add_argument("--reset-right-joint", default=None, help="Optional 6 floats for reset right arm joint target.")
    parser.add_argument("--reset-torso", default=DEFAULT_RESET_TORSO, help="3 floats for reset torso joint target.")
    args = parser.parse_args()

    if args.image_hz <= 0:
        parser.error("--image-hz must be positive")
    if args.state_period <= 0:
        parser.error("--state-period must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    _parse_float_vector(args.reset_left_joint, 6, "--reset-left-joint")
    _parse_float_vector(args.reset_right_joint, 6, "--reset-right-joint")
    _parse_float_vector(args.reset_torso, 3, "--reset-torso")
    return args


def main() -> None:
    args = parse_args()
    app = create_app(args)
    url = f"http://{args.host}:{args.port}"
    print(f"R1Lite monitor: {url}")
    print(f"Robot server: {args.server_url.rstrip('/')}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
