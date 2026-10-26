"""Write LeRobot v3 episodes in a Python environment separate from robot control."""
from __future__ import annotations

import argparse
import inspect
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--episode-frames", type=int, default=100)
    args = parser.parse_args()
    if args.fps < 1 or args.episode_frames < 1:
        parser.error("fps and episode-frames must be positive")

    dataset = _create_dataset(args.root, args.fps)
    pending = total = episodes = 0
    print("READY", flush=True)
    try:
        while (packet := read_packet(sys.stdin.buffer)) is not None:
            if packet.get("kind") == "close":
                break
            if packet.get("kind") != "frame":
                raise ValueError("unknown recorder packet")
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
            print(f"OK {total} {episodes}", flush=True)
    finally:
        if pending:
            dataset.save_episode()
            episodes += 1
        print(f"SAVED {total} {episodes}", flush=True)


if __name__ == "__main__":
    main()
