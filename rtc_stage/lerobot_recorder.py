"""Write LeRobot v3 episodes in a Python environment separate from robot control."""
from __future__ import annotations

import argparse
import inspect
import os
import signal
import socket
import struct
import sys

import msgpack
import numpy as np


FEATURE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper_distance_m"]
FRAME_SHAPE = (480, 640, 3)


def _create_dataset(root, fps):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = ["height", "width", "channel"]
    features = {
        "observation.images.front": {"dtype": "video", "shape": FRAME_SHAPE, "names": names},
        "observation.images.wrist": {"dtype": "video", "shape": FRAME_SHAPE, "names": names},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": FEATURE_NAMES},
        "action": {"dtype": "float32", "shape": (8,), "names": FEATURE_NAMES},
    }
    kwargs = dict(
        repo_id="local/pi05_inference", root=root, fps=fps, features=features,
        robot_type="tb6r5", use_videos=True, image_writer_threads=2,
    )
    accepted = inspect.signature(LeRobotDataset.create).parameters
    # Older LeRobot builds take vcodec on create(); current ones encode on save.
    if "vcodec" in accepted:
        kwargs["vcodec"] = "h264"
    return LeRobotDataset.create(**{key: value for key, value in kwargs.items() if key in accepted})


def _add_frame(dataset, frame):
    accepted = inspect.signature(dataset.add_frame).parameters
    if "task" in accepted:
        task = frame.pop("task")
        dataset.add_frame(frame, task=task)
    else:
        dataset.add_frame(frame)


def read_packet(stream):
    header = stream.read(4)
    if not header:
        return None
    if len(header) != 4:
        raise EOFError("incomplete recorder packet header")
    size = struct.unpack(">I", header)[0]
    if size > 8 * 1024 * 1024:
        raise ValueError("recorder packet too large")
    payload = stream.read(size)
    if len(payload) != size:
        raise EOFError("incomplete recorder packet")
    return msgpack.unpackb(payload, raw=False)


def main():
    def stop_recording(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop_recording)
    signal.signal(signal.SIGTERM, stop_recording)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--episode-frames", type=int, default=100)
    parser.add_argument("--socket", help="Unix socket for independent sidecar recording")
    args = parser.parse_args()
    if args.fps < 1 or args.episode_frames < 1:
        parser.error("fps and episode-frames must be positive")

    root = os.path.abspath(os.path.expanduser(args.root))
    if os.path.isdir(root) and os.listdir(root):
        parser.error(f"recording directory is not empty: {root}")
    # Finish the relatively expensive LeRobot import/initialization before the
    # socket becomes visible to a live control loop.
    dataset = _create_dataset(root, args.fps)
    pending = total = episodes = 0
    server = connection = stream = None
    socket_path = None

    def consume(input_stream):
        nonlocal pending, total, episodes
        while (packet := read_packet(input_stream)) is not None:
            if packet.get("kind") == "close":
                return
            if packet.get("kind") != "frame":
                raise ValueError("unknown recorder packet")
            if args.socket and "_fps" in packet:
                packet_fps = int(round(float(packet["_fps"])))
                if packet_fps != args.fps:
                    raise ValueError(
                        f"recorder FPS={args.fps}, inference FPS={packet_fps}; "
                        f"restart with FPS={packet_fps} ./record_one_episode.sh"
                    )
            frame = {"task": packet["task"]}
            for key in ("observation.images.front", "observation.images.wrist"):
                frame[key] = np.frombuffer(packet[key], dtype=np.uint8).reshape(FRAME_SHAPE).copy()
            for key in ("observation.state", "action"):
                frame[key] = np.asarray(packet[key], dtype=np.float32)
                if frame[key].shape != (8,):
                    raise ValueError(f"{key} must have 8 dimensions")
            _add_frame(dataset, frame)
            pending += 1
            total += 1
            if pending >= args.episode_frames:
                dataset.save_episode()
                pending = 0
                episodes += 1
            if not args.socket or total % 25 == 0:
                print(f"[RECORD] frames={total}", flush=True) if args.socket else print(
                    f"OK {total} {episodes}", flush=True
                )

    try:
        if args.socket:
            socket_path = os.path.abspath(os.path.expanduser(args.socket))
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(socket_path)
            os.chmod(socket_path, 0o666)
            server.listen(1)
            print(f"READY {root}", flush=True)
            while True:
                print(f"[RECORD] waiting for inference at {socket_path}", flush=True)
                connection, _ = server.accept()
                stream = connection.makefile("rb")
                print("[RECORD] inference connected; press Ctrl+C to save this episode", flush=True)
                try:
                    consume(stream)
                finally:
                    stream.close()
                    connection.close()
                    stream = connection = None
                print(f"[RECORD] inference stage disconnected; retained {total} frames", flush=True)
        else:
            stream = sys.stdin.buffer
            print("READY", flush=True)
            consume(stream)
    except KeyboardInterrupt:
        print(f"[RECORD] stopping at {total} frames", flush=True)
    finally:
        if pending:
            dataset.save_episode()
            episodes += 1
        if stream is not None and stream is not sys.stdin.buffer:
            stream.close()
        if connection is not None:
            connection.close()
        if server is not None:
            server.close()
        if socket_path and os.path.exists(socket_path):
            os.unlink(socket_path)
        print(f"SAVED {total} {episodes} {root}", flush=True)


if __name__ == "__main__":
    main()
