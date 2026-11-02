"""Small synchronous bridge from the inference process to LeRobotDataset."""
from __future__ import annotations

from pathlib import Path
import os
import socket
import struct
import subprocess

import cv2
import msgpack
import numpy as np


DEFAULT_SIDECAR_SOCKET = "/tmp/pi05_lerobot_recorder.sock"


def _frame_packet(images, state, action, task):
    packet = {"kind": "frame", "task": task,
              "observation.state": np.asarray(state, dtype=np.float32).tolist(),
              "action": np.asarray(action, dtype=np.float32).tolist()}
    for camera, key in (("realsense_0", "observation.images.front"),
                        ("realsense_1", "observation.images.wrist")):
        image = np.asarray(images[camera], dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Bad {camera} frame shape: {image.shape}")
        packet[key] = cv2.resize(image, (640, 480)).tobytes()
    return packet


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
            # Let the runner own terminal Ctrl+C.  The writer must remain
            # alive long enough to receive the close packet, encode videos,
            # and exit successfully after saving the partial episode.
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True,
        )
        reply = self.process.stdout.readline().decode().strip()
        if reply != "READY":
            raise RuntimeError(f"Recorder failed to start ({self.process.poll()}): {reply}")

    def record(self, images, state, action, task):
        packet = _frame_packet(images, state, action, task)
        if self.fps is not None:
            packet["_fps"] = float(self.fps)
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
                if self.process.stdout is not None:
                    self.process.stdout.close()
        else:
            raise RuntimeError(f"Recorder exited unexpectedly: {self.process.returncode}")

    def is_active(self):
        return True


class SidecarRecorder:
    """Best-effort publisher for a separately started one-episode recorder."""

    def __init__(self, path=None, fps=None):
        self.path = path or os.environ.get("PI05_RECORDER_SOCKET", DEFAULT_SIDECAR_SOCKET)
        self.fps = fps
        self.socket = None
        self.frames = 0
        self._announced = False

    def is_active(self):
        if self.socket is not None:
            return True
        if not os.path.exists(self.path):
            return False
        candidate = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        candidate.settimeout(.25)
        try:
            candidate.connect(self.path)
            candidate.settimeout(5.0)
            self.socket = candidate
            if not self._announced:
                print(f"[RECORD] connected to sidecar {self.path}", flush=True)
                self._announced = True
            return True
        except OSError:
            candidate.close()
            return False

    def record(self, images, state, action, task):
        if not self.is_active():
            return False
        packet = _frame_packet(images, state, action, task)
        data = msgpack.packb(packet, use_bin_type=True)
        try:
            self.socket.sendall(struct.pack(">I", len(data)) + data)
            self.frames += 1
            return True
        except OSError as exc:
            print(f"[RECORD] sidecar stopped after {self.frames} frames: {exc}", flush=True)
            self.socket.close()
            self.socket = None
            return False

    def close(self):
        if self.socket is not None:
            try:
                self.socket.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            self.socket.close()
            self.socket = None
