"""Bounded asynchronous bridge from robot control to LeRobotDataset."""
from __future__ import annotations

from pathlib import Path
import queue
import struct
import subprocess
import threading

import cv2
import msgpack
import numpy as np


class InferenceRecorder:
    def __init__(self, root, python, fps, episode_frames, *, queue_frames=32):
        root = Path(root).expanduser().resolve()
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"Recording directory is not empty: {root}")
        if queue_frames < 1:
            raise ValueError("queue_frames must be positive")
        self.root = root
        self._queue = queue.Queue(maxsize=queue_frames)
        self._closing = threading.Event()
        self._error = None
        self._closed = False
        self._saved = None
        self._accepted = 0
        script = Path(__file__).with_name("lerobot_recorder.py")
        # Terminal Ctrl+C belongs to the runner. The writer finishes via EOF/close.
        self.process = subprocess.Popen(
            [str(python), "-u", str(script), "--root", str(root),
             "--fps", str(fps), "--episode-frames", str(episode_frames)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True,
        )
        reply = self.process.stdout.readline().decode().strip()
        if reply != "READY":
            if self.process.poll() is None:
                self.process.terminate()
            self.process.wait(timeout=10)
            raise RuntimeError(f"Recorder failed to start: {reply}")
        self._worker = threading.Thread(target=self._write_loop, name="lerobot-writer", daemon=True)
        self._worker.start()
        print(f"[RECORD] Async recording: queue capacity {queue_frames} frames", flush=True)

    def record(self, images, state, action, task):
        if self._closing.is_set():
            raise RuntimeError("Recorder is closing")
        if self._error is not None:
            raise RuntimeError(f"Background recorder failed: {self._error}") from self._error
        if self.process.poll() is not None:
            raise RuntimeError(f"Recorder exited unexpectedly: {self.process.returncode}")
        packet = {"kind": "frame", "task": task}
        for key, value in (("observation.state", state), ("action", action)):
            value = np.asarray(value, dtype=np.float32)
            if value.shape != (8,) or not np.isfinite(value).all():
                raise ValueError(f"{key} must be finite 8D")
            packet[key] = value.tolist()
        for camera, key in (("realsense_0", "observation.images.front"),
                            ("realsense_1", "observation.images.wrist")):
            image = np.asarray(images[camera])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
                raise ValueError(f"Bad {camera} RGB uint8 frame: {image.shape}")
            # Own the frame before the producer reuses camera buffers.
            packet[key] = image.copy()
        try:
            self._queue.put_nowait(packet)
        except queue.Full as exc:
            raise RuntimeError("Recording queue full: writer cannot keep up; stopping rather than dropping frames") from exc
        self._accepted += 1

    def _send(self, packet):
        data = msgpack.packb(packet, use_bin_type=True)
        self.process.stdin.write(struct.pack(">I", len(data)) + data)
        self.process.stdin.flush()

    def _write_loop(self):
        try:
            while True:
                try:
                    packet = self._queue.get(timeout=.1)
                except queue.Empty:
                    if self._closing.is_set():
                        break
                    continue
                try:
                    for key in ("observation.images.front", "observation.images.wrist"):
                        packet[key] = cv2.resize(packet[key], (640, 480)).tobytes()
                    self._send(packet)
                    reply = self.process.stdout.readline().decode().strip()
                    if not reply.startswith("OK "):
                        raise RuntimeError(f"Recorder rejected frame: {reply}")
                finally:
                    self._queue.task_done()
            self._send({"kind": "close"})
            self.process.stdin.close()
            reply = self.process.stdout.readline().decode().strip()
            code = self.process.wait(timeout=120)
            parts = reply.split()
            if code or len(parts) != 3 or parts[0] != "SAVED" or int(parts[1]) != self._accepted:
                raise RuntimeError(f"Recorder final count mismatch/failure ({code}): {reply}; queued={self._accepted}")
            self._saved = reply
        except Exception as exc:
            self._error = exc
        finally:
            # EOF asks the child to save any buffered frames on a protocol failure.
            try:
                self.process.stdin.close()
            except (OSError, ValueError):
                pass

    def close(self):
        if self._closed:
            return
        self._closing.set()
        print(f"[RECORD] Draining background writer ({self._queue.qsize()} queued frames)...", flush=True)
        self._worker.join(timeout=300)
        if self._worker.is_alive():
            self.process.kill()
            self.process.wait(timeout=10)
            self._worker.join(timeout=10)
            self._closed = True
            raise RuntimeError("Recorder drain exceeded 300s; dataset may be incomplete")
        try:
            self.process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
            raise RuntimeError("Recorder could not finish saving")
        finally:
            self.process.stdout.close()
            self._closed = True
        if self._error is not None:
            raise RuntimeError(f"Background recorder failed: {self._error}") from self._error
        print(f"[RECORD] {self._saved}: {self.root}", flush=True)
