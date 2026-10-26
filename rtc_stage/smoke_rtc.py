#!/usr/bin/env python3
"""Model-only RTC protocol check. Does not open the robot command channel."""

import json
from pathlib import Path
import time

import numpy as np

from openpi_runner import PolicyClient, action_array, prepare_observation


def main():
    stats_path = Path("/home/ai/pi0.5/model/pytorch/pp_white_red/19999/assets/norm_stats.json")
    stats = json.loads(stats_path.read_text())["norm_stats"]
    state = np.asarray(stats["state"]["mean"], dtype=np.float32)
    # This is a model protocol test only; no robot or camera is connected.
    images = {name: np.zeros((224, 224, 3), dtype=np.uint8) for name in ("realsense_0", "realsense_1")}
    observation = prepare_observation(state, images, "Pick up the white object and place it on the red object.", allow_constant_images=True)
    with PolicyClient("127.0.0.1", 8012, 30) as client:
        if not client.metadata.get("rtc_supported"):
            raise RuntimeError(f"Server lacks RTC support: {client.metadata}")
        print("metadata", client.metadata, flush=True)
        initial = {**observation, "rtc": {"num_steps": 5}}
        started = time.monotonic()
        first = action_array(client.infer(initial, 120))
        print("initial", first.shape, f"{time.monotonic() - started:.3f}s", flush=True)
        if first.shape != (10, 8):
            raise RuntimeError("Unexpected initial action shape")
        guided = {**observation, "rtc": {
            "previous_actions": first[2:].copy(), "executed_steps": 2,
            "delay_steps": 4, "num_steps": 5, "guidance_beta": 5.0,
        }}
        started = time.monotonic()
        second = action_array(client.infer(guided, 120))
        print("guided", second.shape, f"{time.monotonic() - started:.3f}s", flush=True)
        if second.shape != first.shape:
            raise RuntimeError("Guided action shape differs from initial chunk")
        first_overlap = first[2:6]
        guided_overlap = second[:4]
        print("frozen prefix TCP error (m)", np.linalg.norm(first_overlap[:, :3] - guided_overlap[:, :3], axis=1), flush=True)
        print("frozen prefix gripper error (mm)", np.abs(first_overlap[:, 7] - guided_overlap[:, 7]) * 1000, flush=True)


if __name__ == "__main__":
    main()
