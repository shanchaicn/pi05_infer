#!/usr/bin/env python3
"""OpenPI-protocol placeholder server for pipeline tests.

This server never runs a learned model.  It returns a short action chunk that
holds the observed TB6-R5 state, allowing Docker, websocket, API, validation,
and logging paths to be tested without a completed checkpoint.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class HoldPositionPolicy:
    """Return the current seven-dimensional state as every target action."""

    def __init__(self, action_horizon: int = 10) -> None:
        self.action_horizon = max(1, int(action_horizon))

    def infer(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        state = np.asarray(observation.get("observation/state"), dtype=np.float32).reshape(-1)
        if state.shape != (7,) or not np.all(np.isfinite(state)):
            raise ValueError(f"expected finite observation/state shape (7,), got {state.shape}")
        return {"actions": np.repeat(state[None, :], self.action_horizon, axis=0)}


def main() -> None:
    port = int(os.environ.get("POLICY_PORT", "8002"))
    horizon = int(os.environ.get("MOCK_ACTION_HORIZON", "10"))
    metadata = {
        "placeholder": True,
        "policy_kind": "tb6r5_hold_position_mock",
        "hardware_allowed": False,
        "warning": "No learned model is loaded. Dry-run pipeline validation only.",
    }
    logging.warning("Starting TB6-R5 placeholder policy on port %d; hardware use is forbidden", port)
    WebsocketPolicyServer(
        policy=HoldPositionPolicy(horizon),
        host="0.0.0.0",
        port=port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
