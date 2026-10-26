"""Small synchronous bridge from the inference process to LeRobotDataset."""
from __future__ import annotations

from pathlib import Path
import struct
import subprocess

import cv2
import msgpack
import numpy as np


class InferenceRecorder:
    def __init__(self, root, python, fps, episode_frames):
        root = Path(root).expanduser().resolve()
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"Recording directory is not empty: {root}")
        self.root = root
        script = Path(__file__).with_name("lerobot_recorder.py")
        self.process = subprocess.Popen(
            [str(python), "-u", str(script), "--root", str(root),
             "--fps", str(fps), "--episode-frames", str(episode_frames)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
        reply = self.process.stdout.readline().decode().strip()
        if reply != "READY":
            raise RuntimeError(f"Recorder failed to start ({self.process.poll()}): {reply}")

    def record(self, images, state, action, task):
        packet = {"kind": "frame", "task": task,
                  "observation.state": np.asarray(state, dtype=np.float32).tolist(),
                  "action": np.asarray(action, dtype=np.float32).tolist()}
        for camera, key in (("realsense_0", "observation.images.front"),
                            ("realsense_1", "observation.images.wrist")):
            image = np.asarray(images[camera], dtype=np.uint8)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Bad {camera} frame shape: {image.shape}")
            packet[key] = cv2.resize(image, (640, 480)).tobytes()
        self._send(packet)
        reply = self.process.stdout.readline().decode().strip()
        if not reply.startswith("OK "):
            raise RuntimeError(f"Recorder failed ({self.process.poll()}): {reply}")
        return reply

    def _send(self, packet):
        data = msgpack.packb(packet, use_bin_type=True)
        self.process.stdin.write(struct.pack(">I", len(data)) + data)
        self.process.stdin.flush()

    def close(self):
        if self.process.poll() is None:
            try:
                self._send({"kind": "close"})
                self.process.stdin.close()
                reply = self.process.stdout.readline().decode().strip()
                code = self.process.wait(timeout=120)
                if code or not reply.startswith("SAVED "):
                    raise RuntimeError(f"Recorder failed to save ({code}): {reply}")
                print(f"[RECORD] {reply}: {self.root}", flush=True)
            finally:
                if self.process.poll() is None:
                    self.process.terminate()
        else:
            raise RuntimeError(f"Recorder exited unexpectedly: {self.process.returncode}")
