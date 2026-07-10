#!/usr/bin/env python3
"""Shared RTC helpers for R1Lite OpenPI policy runners."""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable
import threading
import time

import numpy as np


def add_rtc_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rtc", dest="rtc", action="store_true", default=None, help="Enable RTC rollout mode.")
    parser.add_argument("--no-rtc", dest="rtc", action="store_false", help="Disable RTC rollout mode.")
    parser.add_argument("--rtc-latency-k", type=int, default=None)
    parser.add_argument("--rtc-inference-rate", type=float, default=None)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=None)


def apply_rtc_defaults(args: argparse.Namespace) -> None:
    if args.rtc is None:
        args.rtc = False
    if args.rtc_latency_k is None:
        args.rtc_latency_k = 4
    if args.rtc_inference_rate is None:
        args.rtc_inference_rate = 3.0
    if args.rtc_max_guidance_weight is None:
        args.rtc_max_guidance_weight = 10.0


def validate_rtc_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.rtc_latency_k < 0:
        parser.error("--rtc-latency-k must be non-negative")
    if args.rtc_inference_rate <= 0:
        parser.error("--rtc-inference-rate must be positive")
    if args.rtc_max_guidance_weight <= 0:
        parser.error("--rtc-max-guidance-weight must be positive")
    if args.rtc and args.debug:
        parser.error("--rtc cannot be combined with --debug")


class RtcActionBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cur_chunk: deque[np.ndarray] = deque()
        self._k = 0
        self._last_action: np.ndarray | None = None

    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int, min_m: int = 8) -> None:
        with self._lock:
            chunk = np.asarray(actions_chunk, dtype=np.float32)
            if chunk.ndim != 2:
                raise ValueError(f"Expected RTC executable chunk with 2 dims, got {chunk.shape}")
            if len(chunk) == 0:
                raise ValueError("RTC executable chunk is empty")

            drop_n = min(self._k, max(0, int(max_k)))
            if drop_n >= len(chunk):
                raise ValueError(f"RTC latency trim dropped entire chunk: drop_n={drop_n}, len={len(chunk)}")
            new_chunk = [row.copy() for row in chunk[drop_n:]]

            if len(self._cur_chunk) == 0 and self._last_action is not None:
                old_list = [self._last_action.copy() for _ in range(max(1, int(min_m)))]
                self._last_action = None
            else:
                old_list = list(self._cur_chunk)
                if len(old_list) == 0:
                    self._cur_chunk = deque(new_chunk)
                    self._k = 0
                    return
                if len(old_list) < min_m:
                    old_list.extend([old_list[-1].copy() for _ in range(min_m - len(old_list))])

            overlap_len = min(len(old_list), len(new_chunk))
            if overlap_len == 1:
                w_old = np.array([1.0], dtype=np.float32)
            else:
                w_old = np.linspace(1.0, 0.0, overlap_len, dtype=np.float32)
            w_new = 1.0 - w_old
            smoothed = [w_old[i] * old_list[i] + w_new[i] * new_chunk[i] for i in range(overlap_len)]
            self._cur_chunk = deque(row.copy() for row in smoothed + new_chunk[overlap_len:])
            self._k = 0

    def pop_next_action(self) -> np.ndarray | None:
        with self._lock:
            if len(self._cur_chunk) == 0:
                return None
            action = np.asarray(self._cur_chunk.popleft(), dtype=np.float32)
            self._last_action = action.copy()
            self._k += 1
            return action

    def reset(self) -> None:
        with self._lock:
            self._cur_chunk.clear()
            self._k = 0
            self._last_action = None


class RtcInferenceLoop:
    def __init__(
        self,
        *,
        fetch_executable_chunk: Callable[[], np.ndarray],
        latency_k: int,
        inference_rate: float,
    ):
        self._fetch_executable_chunk = fetch_executable_chunk
        self._latency_k = int(latency_k)
        self._interval = 1.0 / float(inference_rate)
        self._buffer = RtcActionBuffer()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._error: BaseException | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._error = None
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="r1lite-rtc-inference", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop_event.set()
            thread = self._thread
            self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

    def reset(self) -> None:
        self.stop()
        self._buffer.reset()
        with self._lock:
            self._error = None

    def pop_next_action(self) -> np.ndarray | None:
        self.raise_if_failed()
        return self._buffer.pop_next_action()

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError("RTC inference loop failed") from error

    def _run(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                self._buffer.integrate_new_chunk(self._fetch_executable_chunk(), max_k=self._latency_k)
            except BaseException as error:
                with self._lock:
                    self._error = error
                self._stop_event.set()
                return
            remaining = max(0.0, self._interval - (time.monotonic() - started))
            if remaining > 0:
                self._stop_event.wait(remaining)
