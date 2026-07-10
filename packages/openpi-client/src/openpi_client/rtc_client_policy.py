from __future__ import annotations

import copy
import threading
from typing import Dict, Optional

import numpy as np

from openpi_client import websocket_client_policy


class RtcWebsocketClientPolicy(websocket_client_policy.WebsocketClientPolicy):
    """Websocket policy client that feeds the previous action chunk back for RTC."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        *,
        latency_k: int,
        execution_horizon: int,
        max_guidance_weight: float,
    ) -> None:
        super().__init__(host=host, port=port, api_key=api_key)
        self._latency_k = int(latency_k)
        self._execution_horizon = int(execution_horizon)
        self._max_guidance_weight = float(max_guidance_weight)
        self._active_response: Optional[dict] = None
        self._lock = threading.Lock()
        self._request_lock = threading.Lock()

    @property
    def is_rtc_client(self) -> bool:
        return True

    def reset(self) -> None:
        with self._lock:
            self._active_response = None

    def update_rtc_strategy(
        self,
        *,
        latency_k: int,
        execution_horizon: int,
        max_guidance_weight: float,
    ) -> None:
        with self._lock:
            self._latency_k = int(latency_k)
            self._execution_horizon = int(execution_horizon)
            self._max_guidance_weight = float(max_guidance_weight)

    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        request = dict(obs)
        with self._lock:
            active_response = copy.deepcopy(self._active_response)
            latency_k = self._latency_k
            execution_horizon = self._execution_horizon
            max_guidance_weight = self._max_guidance_weight

        if active_response is not None:
            actions = np.asarray(active_response["actions"], dtype=np.float32)
            if actions.ndim != 2:
                raise RuntimeError(f"RTC expected previous actions with 2 dims, got {actions.shape}")
            if 0 < latency_k < actions.shape[0]:
                shifted = np.empty_like(actions)
                shifted[: actions.shape[0] - latency_k] = actions[latency_k:]
                shifted[actions.shape[0] - latency_k :] = actions[-1]
                actions = shifted
            request["prev_action"] = actions
            request["rtc_params"] = {
                "s": latency_k,
                "execution_horizon": execution_horizon,
                "max_guidance_weight": max_guidance_weight,
            }

        with self._request_lock:
            response = super().infer(request)

        with self._lock:
            self._active_response = copy.deepcopy(response)
        return response
