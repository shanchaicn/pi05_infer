"""Host-side OpenPI websocket inference and TB6-R5 control.

The learned model stays in Docker. Dry-run opens only cameras and the read-only
Topic subscriber; it never connects the robot RPC command channel.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = Path(os.environ.get("TB6R5_PACKAGE_ROOT", ROOT.parent / "tb6r5_policy_infer"))
for source in (
    PACKAGE_ROOT,
    ROOT.parent,
    ROOT / "app/packages/openpi-client/src",
    ROOT.parent / "app/packages/openpi-client/src",
    ROOT.parent / "docker_source/packages/openpi-client/src",
):
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

from openpi_client import msgpack_numpy

DEFAULT_TASK = "Pick up the yellow ring and place it on the white plate."
DEFAULT_POLICY_DIR = Path(os.environ.get(
    "TB6R5_OPENPI_POLICY_DIR", os.environ.get(
        "CHECKPOINT_HOST_DIR", "/home/ai/pi0.5/model/pytorch/pp_white_red/19999"
    )
))
DEFAULT_DRY_STATE = np.array([-.04, -.616, .26, .47, -.855, .129, .112, .04], dtype=np.float32)
DEFAULT_DRY_STATE[3:7] /= np.linalg.norm(DEFAULT_DRY_STATE[3:7])


class SafetyAbort(RuntimeError):
    """Invalid input, feedback, or policy output: stop before sending motion."""


pack_message = msgpack_numpy.packb
unpack_message = msgpack_numpy.unpackb


def validate_policy_server_metadata(metadata, *, dry_run):
    if not isinstance(metadata, dict):
        raise SafetyAbort("Policy metadata must be a dictionary")
    if not dry_run and (metadata.get("placeholder") or metadata.get("hardware_allowed") is False):
        raise SafetyAbort("This policy server forbids hardware control")


class PolicyClient:
    def __init__(self, host, port, connect_timeout_s=10.0):
        from websockets.sync.client import connect

        self.socket = connect(
            f"ws://{host}:{port}", open_timeout=connect_timeout_s,
            compression=None, max_size=32 * 1024 * 1024, proxy=None, ping_interval=None,
        )
        try:
            self.metadata = self._receive(connect_timeout_s)
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout):
        response = self.socket.recv(timeout=timeout)
        if isinstance(response, str):
            raise SafetyAbort(f"Policy server error:\n{response}")
        result = unpack_message(response)
        if not isinstance(result, dict):
            raise SafetyAbort("Policy response must be a dictionary")
        return result

    def infer(self, observation, timeout_s=30.0):
        self.socket.send(pack_message(observation))
        return self._receive(timeout_s)

    def close(self):
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def finite_vector(value, dim, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (dim,) or not np.all(np.isfinite(array)):
        raise SafetyAbort(f"{name}: expected finite shape ({dim},), got {array.shape}")
    return array.copy()


def parse_xyz(value):
    try:
        xyz = np.asarray([float(part.strip()) for part in value.split(",")], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected x,y,z in metres") from exc
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise argparse.ArgumentTypeError("expected three finite values: x,y,z")
    return xyz


def quantize_gripper_actions(actions, args):
    """Map model gripper semantics to two stable physical endpoints."""
    actions = np.asarray(actions, dtype=np.float32).copy()
    if not args.binary_gripper:
        return actions
    raw = actions[:, -1].copy()
    actions[:, -1] = np.where(
        raw < args.gripper_close_threshold_m,
        args.gripper_close_command_m,
        args.gripper_open_command_m,
    )
    return actions


@dataclass
class TaskCompletionDetector:
    target_xyz: np.ndarray
    radius_m: float
    gripper_open_m: float
    hold_steps: int
    min_steps: int
    require_grasp: bool
    grasp_threshold_m: float
    consecutive: int = 0
    grasp_seen: bool = False

    def update(self, state, executed_steps):
        state = np.asarray(state, dtype=np.float32)
        if state[-1] < self.grasp_threshold_m:
            self.grasp_seen = True
        distance = float(np.linalg.norm(state[:3] - self.target_xyz))
        eligible = (
            executed_steps >= self.min_steps
            and distance <= self.radius_m
            and float(state[-1]) > self.gripper_open_m
            and (self.grasp_seen or not self.require_grasp)
        )
        self.consecutive = self.consecutive + 1 if eligible else 0
        return self.consecutive >= self.hold_steps, distance


def checked_quaternion(value):
    quat = finite_vector(value, 4, "quaternion")
    norm = np.linalg.norm(quat)
    if abs(norm - 1) > .1:
        raise SafetyAbort(f"Quaternion norm must be near 1, got {norm:.4f}")
    quat /= norm
    # Keep the dataset quaternion sign; forcing w >= 0 can leave its training bounds.
    return quat


def limit_quaternion_step(previous, target, max_angle):
    """SLERP target toward previous so their geodesic angle is at most max_angle."""
    previous = checked_quaternion(previous)
    target = checked_quaternion(target)
    dot = float(np.clip(np.dot(previous, target), -1.0, 1.0))
    if dot < 0:
        target = -target
        dot = -dot
    angle = 2.0 * math.acos(float(np.clip(dot, 0.0, 1.0)))
    if angle <= max_angle:
        return target, angle, False
    fraction = max_angle / angle
    half_angle = math.acos(float(np.clip(dot, 0.0, 1.0)))
    sin_half = math.sin(half_angle)
    if sin_half < 1e-6:
        limited = previous + fraction * (target - previous)
    else:
        limited = (
            math.sin((1.0 - fraction) * half_angle) / sin_half * previous
            + math.sin(fraction * half_angle) / sin_half * target
        )
    limited /= np.linalg.norm(limited)
    return limited.astype(np.float32), angle, True


# pp_white_red stores 1280x720 cameras stretched to this 4:3 frame.
TRAINING_FRAME_WIDTH = 640
TRAINING_FRAME_HEIGHT = 480


def match_training_frame(image):
    """Reproduce the converter's PIL LANCZOS stretch to the stored 640x480 frame.

    A direct resize from 1280x720 makes objects about 33% taller. The policy
    server applies the same OpenPI ResizeImages(224, 224) transform used during
    training, so this client deliberately stops at the stored dataset size.
    """
    from PIL import Image

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1 or image.dtype != np.uint8:
        raise SafetyAbort("Camera image must be a non-empty RGB uint8 HWC array")
    if image.shape[0] == TRAINING_FRAME_HEIGHT and image.shape[1] == TRAINING_FRAME_WIDTH:
        return image
    return np.asarray(
        Image.fromarray(image, mode="RGB").resize(
            (TRAINING_FRAME_WIDTH, TRAINING_FRAME_HEIGHT),
            Image.Resampling.LANCZOS,
        )
    )


def resize_with_pad(image, size=224):
    import cv2

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1 or image.dtype != np.uint8:
        raise SafetyAbort("Camera image must be a non-empty RGB uint8 HWC array")
    height, width = image.shape[:2]
    scale = size / max(height, width)
    height, width = max(1, round(height * scale)), max(1, round(width * scale))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    output = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - height) // 2, (size - width) // 2
    output[top:top + height, left:left + width] = resized
    return output


def prepare_observation(state, images, task, *, allow_constant_images=False):
    state = np.asarray(state, dtype=np.float32)
    if state.shape not in ((7,), (8,)) or not np.all(np.isfinite(state)):
        raise SafetyAbort("State must be a finite 7D joint or 8D TCP vector")
    if not isinstance(task, str) or not task.strip():
        raise SafetyAbort("A non-empty task is required")
    observation = {"observation/state": state.copy(), "prompt": task}
    for name, key in (("realsense_0", "observation/image"), ("realsense_1", "observation/wrist_image")):
        if name not in images:
            raise SafetyAbort(f"Missing camera: {name}")
        raw = np.asarray(images[name])
        # Send the stored training resolution. The policy server owns the
        # OpenPI ResizeImages(224, 224) transform for both training and inference.
        image = match_training_frame(raw)
        if not allow_constant_images and (raw.max() == raw.min() or float(raw.std()) < 1):
            raise SafetyAbort(f"Camera {name} returned an empty/constant frame")
        observation[key] = image
    return observation


def action_array(result):
    if not isinstance(result, dict) or "actions" not in result:
        raise SafetyAbort("Policy response has no actions")
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim != 2 or min(actions.shape) < 1 or not np.all(np.isfinite(actions)):
        raise SafetyAbort(f"Expected finite non-empty action chunk (T, D), got {actions.shape}")
    return actions.copy()


@dataclass
class SafetyLimits:
    bounds: dict
    margin: float = .15
    z_margin: float = .20
    action_space: str = "ee_pose"
    joint_step: float = .03
    tcp_step: float = .03
    rotation_step: float = .15
    gripper_min_mm: float = 0
    gripper_max_mm: float = 65

    @classmethod
    def load(cls, path, margin=.15, **kwargs):
        stats = json.loads(Path(path).read_text())["norm_stats"]
        bounds = {}
        for key in ("state", "actions"):
            low = np.asarray(stats[key]["q01"], dtype=np.float32)
            high = finite_vector(stats[key]["q99"], len(low), f"{key}.q99")
            low = finite_vector(low, len(high), f"{key}.q01")
            if len(low) not in (7, 8) or np.any(low > high):
                raise SafetyAbort(f"Invalid {key} normalization statistics")
            bounds[key] = (low, high)
        if len(bounds["state"][0]) != len(bounds["actions"][0]):
            raise SafetyAbort("State and action statistics dimensions disagree")
        result = cls(bounds, margin, **kwargs)
        expected = 8 if result.action_space == "ee_pose" else 7
        if result.dimension != expected:
            raise SafetyAbort(f"{result.action_space} requires {expected}D statistics, got {result.dimension}D")
        return result

    @property
    def dimension(self):
        return len(self.bounds["state"][0])

    def validate_state(self, value, kind="state"):
        value = finite_vector(value, self.dimension, kind)
        if self.action_space == "ee_pose":
            value[3:7] = checked_quaternion(value[3:7])
        low, high = self.bounds[kind]
        if self.action_space == "ee_pose":
            # q and -q encode the same rotation. Controllers may return the
            # opposite sign from the dataset, so align to the checkpoint
            # quaternion hemisphere before component-wise envelope checks.
            training_center = (low[3:7] + high[3:7]) * .5
            if float(np.dot(value[3:7], training_center)) < 0:
                value[3:7] *= -1
        padding = np.maximum(high - low, .01) * self.margin
        if self.action_space == "ee_pose":
            padding[2] = max(high[2] - low[2], .01) * self.z_margin
        lower = low - padding
        upper = high + padding
        outside = (value < lower) | (value > upper)
        if self.action_space == "ee_pose":
            # Quaternion components are coupled and q/-q are equivalent, so
            # per-component quantile bounds can reject a valid nearby pose.
            # Quaternion norm and per-step angular distance remain checked.
            outside[3:7] = False
        if np.any(outside):
            names = ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper") if self.action_space == "ee_pose" else tuple(
                f"joint_{index}" for index in range(self.dimension - 1)
            ) + ("gripper",)
            details = ", ".join(
                f"{names[index]}={value[index]:.6f} outside "
                f"[{lower[index]:.6f}, {upper[index]:.6f}]"
                for index in np.flatnonzero(outside)
            )
            raise SafetyAbort(f"{kind} outside checkpoint training envelope: {details}")
        return value

    def validate_actions(self, actions, state):
        actions = action_array({"actions": actions})
        if actions.shape[1] != self.dimension:
            raise SafetyAbort(f"Action dimension {actions.shape[1]} != checkpoint dimension {self.dimension}; no truncation or padding allowed")
        previous = self.validate_state(state)
        for index, raw in enumerate(actions):
            if self.action_space == "ee_pose":
                # The command path already enforces the physical gripper
                # range.  Apply that limit before the statistical envelope so
                # a harmless over-open prediction cannot reject the entire
                # Cartesian action chunk.
                raw[-1] = np.clip(
                    raw[-1], self.gripper_min_mm / 1000.0, self.gripper_max_mm / 1000.0
                )
            target = self.validate_state(raw, "actions")
            if self.action_space == "ee_pose":
                distance = float(np.linalg.norm(target[:3] - previous[:3]))
                limited_quat, angle, _ = limit_quaternion_step(
                    previous[3:7], target[3:7], self.rotation_step
                )
                if distance > self.tcp_step:
                    raise SafetyAbort(
                        f"Unsafe TCP position jump at action {index}: {distance:.4f}m; "
                        f"previous={previous[:3].tolist()}, target={target[:3].tolist()}"
                    )
                target[3:7] = limited_quat
            elif np.max(np.abs(target[:6] - previous[:6])) > self.joint_step:
                raise SafetyAbort(f"Unsafe joint jump at action {index}")
            actions[index] = target
            previous = target
        return actions

    def gripper_mm(self, action):
        raw_mm = float(action[-1]) * (1000 if self.action_space == "ee_pose" else 1)
        return float(np.clip(raw_mm, self.gripper_min_mm, self.gripper_max_mm))


class RobotFeedback:
    """Synchronous read-only Topic snapshots, with controller timestamp expiry."""
    def __init__(self, args):
        from tb6r5_policy_infer.hardware.topic_feedback import TopicFeedback

        self.reader = TopicFeedback(args.robot_ip, g_model=args.g_model)
        self.args = args
        self.last_header = None
        self.last_update = 0.0

    def start(self):
        from tb6r5_policy_infer.hardware.topic_feedback import setup_topic_import, validate_topic_sdk

        validate_topic_sdk()
        setup_topic_import()
        import topic

        self.reader._topic = topic
        topic.start_subscriber(self.args.robot_ip)
        deadline = time.monotonic() + self.args.feedback_timeout_s
        last_error = "no data"
        while time.monotonic() < deadline:
            try:
                self.read()
                return
            except SafetyAbort as exc:
                last_error = str(exc)
                time.sleep(.05)
        raise SafetyAbort(f"No valid Topic feedback from {self.args.robot_ip}: {last_error}")

    def read(self):
        snapshot = self.reader._topic.get_system_state()
        if not snapshot.has_rt():
            raise SafetyAbort("Missing Topic RT feedback")
        rt = snapshot.get_rt()
        header = rt.header_timestamp
        now = time.monotonic()
        if header != self.last_header:
            self.last_header, self.last_update = header, now
        if not header or now - self.last_update > self.args.feedback_max_age_s:
            raise SafetyAbort("Stale Topic feedback")
        # Parse this exact snapshot instead of fetching another one through the
        # shared reader, which also suppresses field/index exceptions.
        if not rt.models:
            raise SafetyAbort("Missing arm model: RT models is empty")
        if self.args.g_model == 3:
            if len(rt.models) < 2 or int(rt.models[1].joint_count) < 1:
                raise SafetyAbort(
                    f"Missing gripper model: --g-model 3 requires models[1] with a joint; received {len(rt.models)} models"
                )
            start = int(rt.models[1].joint_start_idx)
            if not 0 <= start < len(rt.models_joints):
                raise SafetyAbort(
                    f"Missing gripper joint: start={start}, RT joint count={len(rt.models_joints)}"
                )
            grip_m = float(rt.models_joints[start].position)
            # Physics/controller feedback can cross a mechanical endpoint by
            # floating-point roundoff. Allow at most one micrometre.
            tolerance_m = 1e-6
            if not math.isfinite(grip_m) or not -tolerance_m <= grip_m <= .08 + tolerance_m:
                raise SafetyAbort(f"Invalid gripper joint position: {grip_m}m (expected 0..0.08m)")
            grip_m = float(np.clip(grip_m, 0, .08))
            grip_mm = self.reader._gripper_joint_to_mm(grip_m)
        else:
            grip_mm, grip_ok = self.reader._read_gripper_mm_nrt()
            if not grip_ok or grip_mm is None:
                raise SafetyAbort("Missing gripper NRT subsystem feedback for --g-model 2")
            if not math.isfinite(grip_mm) or not 0 <= grip_mm <= 80:
                raise SafetyAbort(f"Invalid gripper NRT position: {grip_mm}mm")
            grip_m = grip_mm / 1000
        if self.args.action_space == "ee_pose":
            if not rt.models_current_points:
                raise SafetyAbort("Missing TCP robottarget feedback")
            tcp = finite_vector(rt.models_current_points[0].robottarget, 7, "TCP robottarget")
            return np.concatenate((tcp[:3], checked_quaternion(tcp[3:7]), [grip_m])).astype(np.float32)
        start = int(rt.models[0].joint_start_idx)
        count = int(rt.models[0].joint_count)
        if count < 6 or start < 0 or start + 6 > len(rt.models_joints):
            raise SafetyAbort(
                f"Missing arm joint feedback: start={start}, arm joint count={count}, RT joint count={len(rt.models_joints)}"
            )
        joints = finite_vector([rt.models_joints[i].position for i in range(start, start + 6)], 6, "Arm joints")
        return np.concatenate((joints, [grip_mm])).astype(np.float32)

    def close(self):
        self.reader.stop()


class Cameras:
    """Two-camera input from Topic SDK, local devices, RealSense, or HTTP."""
    def __init__(self, args):
        self.args = args
        self.sources = {}
        self.caps = {}
        self.realsense = None
        self.http_stream = None
        self.topic_camera = None
        self.transport = ""

    @staticmethod
    def _parse_mapping(value, option):
        result = {}
        for entry in value.split(","):
            if "=" not in entry:
                raise SafetyAbort(f"{option} entries must be alias=value")
            name, source = (part.strip() for part in entry.split("=", 1))
            if not name or not source or name in result:
                raise SafetyAbort(f"Invalid or duplicate {option} entry: {entry!r}")
            result[name] = source
        return result

    def _start_http(self):
        from tb6r5_policy_infer.camera import HttpCameraStream
        self.http_stream = HttpCameraStream(
            self.sources,
            width=1280,
            height=720,
            fps=max(1, int(self.args.fps)),
            timeout_s=self.args.camera_timeout_s,
        )
        self.http_stream.start()
        self.http_stream.wait_ready(self.args.camera_timeout_s * self.args.camera_retries)
        if set(self.http_stream.get_images()) != set(self.sources):
            raise SafetyAbort("Not all HTTP cameras produced an initial frame")
        self.transport = "HTTP MJPEG"

    def start(self):
        from tb6r5_policy_infer.camera import parse_camera_urls, parse_camera_devices, parse_camera_serials

        args = self.args
        if args.camera_topic_sensors:
            self.sources = self._parse_mapping(args.camera_topic_sensors, "--camera-topic-sensors")
            try:
                from camera_topic_sdk.robot_state.giant_frame_camera import GiantFrameCamera
                self.topic_camera = GiantFrameCamera(args.robot_ip, self.sources, args.camera_topic_port)
                self.topic_camera.start()
                timeout = args.camera_timeout_s * args.camera_retries
                if not self.topic_camera.wait_ready(timeout):
                    raise RuntimeError(f"no complete two-camera frame within {timeout:g}s; diagnostics={self.topic_camera.diagnostics}")
                self.transport = f"Topic SDK tcp://{args.robot_ip}:{args.camera_topic_port}"
            except Exception as exc:
                if self.topic_camera is not None:
                    self.topic_camera.stop()
                    self.topic_camera = None
                if not args.camera_topic_fallback_http:
                    raise SafetyAbort(f"Camera Topic SDK unavailable: {exc}") from exc
                print(f"[CAMERA] Topic SDK unavailable, falling back to HTTP: {exc}", flush=True)
                self.sources = {
                    "realsense_0": f"http://{args.robot_ip}:6888/RsCameraSensor/0/0/color",
                    "realsense_1": f"http://{args.robot_ip}:6888/RsCameraSensor/1/0/color",
                }
                self._start_http()
        elif args.camera_serials:
            from tb6r5_policy_infer.hardware.realsense import RealSenseCameraInterface
            self.sources = parse_camera_serials(args.camera_serials, {})
            self.realsense = RealSenseCameraInterface(serial_numbers=list(self.sources.values()))
            self.realsense.start()
            self.transport = "local RealSense SDK"
        elif args.camera_devices:
            import cv2
            self.sources = parse_camera_devices(args.camera_devices)
            for name, device in self.sources.items():
                cap = cv2.VideoCapture(int(device) if device.isdigit() else device)
                self.caps[name] = cap
                if not cap.isOpened():
                    raise SafetyAbort(f"Cannot open camera {name}: {device}")
            self.transport = "local video device"
        else:
            self.sources = parse_camera_urls(args.camera_urls) if args.camera_urls else {
                "realsense_0": f"http://{args.robot_ip}:6888/RsCameraSensor/0/0/color",
                "realsense_1": f"http://{args.robot_ip}:6888/RsCameraSensor/1/0/color",
            }
            self._start_http()
        if set(self.sources) != {"realsense_0", "realsense_1"}:
            raise SafetyAbort("Camera names must be realsense_0 and realsense_1")
        print(f"[OPENPI] Cameras ({self.transport}): {self.sources}", flush=True)

    def capture(self):
        import cv2

        if self.topic_camera is not None:
            deadline = time.monotonic() + self.args.camera_timeout_s * self.args.camera_retries
            while True:
                frames = self.topic_camera.get_frames()
                now_ns = time.time_ns()
                missing = set(self.sources) - set(frames)
                stale = {
                    name: (now_ns - frames[name]["timestamp_ns"]) / 1e9
                    for name in frames if name in self.sources
                    if (now_ns - frames[name]["timestamp_ns"]) / 1e9 > self.args.camera_max_age_s
                }
                if not missing and not stale:
                    images = {name: frames[name]["image"] for name in self.sources}
                    break
                if time.monotonic() >= deadline:
                    raise SafetyAbort(
                        f"Camera Topic SDK unavailable: missing={sorted(missing)}, stale_seconds={stale}, "
                        f"diagnostics={self.topic_camera.diagnostics}"
                    )
                time.sleep(.02)
        elif self.realsense is not None:
            self.realsense.update_frames()
            frames = self.realsense.get_frames()
            images = {name: frames[serial]["color"] for name, serial in self.sources.items() if serial in frames}
        elif self.caps:
            images = {}
            for name, cap in self.caps.items():
                ok, frame = cap.read()
                if not ok:
                    raise SafetyAbort(f"Camera disconnected: {name}")
                images[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            deadline = time.monotonic() + self.args.camera_timeout_s * self.args.camera_retries
            while True:
                images = self.http_stream.get_images()
                ages = self.http_stream.get_frame_ages()
                missing = set(self.sources) - set(images)
                stale = {name: ages.get(name, float("inf")) for name in self.sources if ages.get(name, float("inf")) > self.args.camera_max_age_s}
                if not missing and not stale:
                    break
                if time.monotonic() >= deadline:
                    raise SafetyAbort(f"HTTP camera unavailable: missing={sorted(missing)}, stale_seconds={stale}")
                time.sleep(.02)
        if self.args.show_camera:
            for name, image in images.items():
                cv2.imshow(name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xff == 27:
                raise KeyboardInterrupt
        return images

    def close(self):
        for cap in self.caps.values():
            cap.release()
        if self.realsense is not None:
            self.realsense.stop()
        if self.http_stream is not None:
            self.http_stream.stop()
        if self.topic_camera is not None:
            self.topic_camera.stop()
        if self.args.show_camera:
            import cv2
            cv2.destroyAllWindows()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.2.174")
    parser.add_argument("--rpc-port", type=int, default=5868)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8002)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offline-dry-run", action="store_true", help="Synthetic cameras/state; no robot connection (requires --dry-run --no-camera)")
    parser.add_argument("--record-dataset", type=Path, help="New local LeRobot v3 dataset directory")
    parser.add_argument("--recorder-python", default="/home/ai/condaenv/tb6r5/bin/python")
    parser.add_argument("--record-fps", type=int, default=2)
    parser.add_argument("--record-episode-frames", type=int, default=100)
    parser.add_argument("--confirm-hardware")
    parser.add_argument("--local-check", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Test model responses only; implies --dry-run and requires --no-camera")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--print-rpc", action="store_true")
    parser.add_argument("--show-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-inference-calls", type=int, default=0, help="0: unlimited (smoke-test defaults to 1)")
    parser.add_argument("--first-action-timeout-s", type=float, default=120)
    parser.add_argument("--action-timeout-s", type=float, default=30)
    parser.add_argument("--feedback-timeout-s", type=float, default=5)
    parser.add_argument("--feedback-max-age-s", type=float, default=.5)
    parser.add_argument("--camera-timeout-s", type=float, default=5)
    parser.add_argument("--camera-retries", type=int, default=3)
    parser.add_argument("--camera-max-age-s", type=float, default=2.0)
    camera_group = parser.add_mutually_exclusive_group()
    for option in ("--camera-urls", "--camera-devices", "--camera-serials", "--camera-topic-sensors"):
        camera_group.add_argument(option)
    parser.add_argument("--camera-topic-port", type=int, default=19091)
    parser.add_argument("--camera-topic-fallback-http", action="store_true", help="Fall back to robot HTTP MJPEG when no giant_frame_data Topic frames arrive")
    parser.add_argument("--norm-stats", type=Path, default=DEFAULT_POLICY_DIR / "assets/norm_stats.json")
    parser.add_argument("--action-space", choices=("ee_pose", "joints"), default="ee_pose")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--n-action-steps", type=int, default=4)
    parser.add_argument("--rtc", action="store_true", help="Asynchronous guided Real-Time Chunking (JAX or PyTorch server)")
    parser.add_argument("--rtc-warmup-only", action="store_true", help="Compile and validate RTC shapes, then exit before the control loop")
    parser.add_argument("--rtc-num-steps", type=int, default=5)
    parser.add_argument("--rtc-min-execution-steps", type=int, default=2)
    parser.add_argument("--rtc-guidance-beta", type=float, default=5.0)
    parser.add_argument(
        "--rtc-latency-margin-steps",
        type=int,
        default=2,
        help="Extra control ticks reserved beyond measured steady-state inference latency",
    )
    parser.add_argument(
        "--rtc-latency-floor-s",
        type=float,
        default=0.0,
        help="Known steady-state latency floor used to choose the first guided RTC shape",
    )
    parser.add_argument("--binary-gripper", action="store_true", help="Map model gripper output to closed/open endpoints")
    parser.add_argument("--gripper-close-threshold-m", type=float, default=.035)
    parser.add_argument("--gripper-close-command-m", type=float, default=0.0)
    parser.add_argument("--gripper-open-command-m", type=float, default=.065)
    parser.add_argument("--binary-gripper-resend-delta-mm", type=float, default=1.0, help="Do not resend the same binary endpoint within this distance")
    parser.add_argument("--task-complete-tcp", type=parse_xyz, help="Completion target x,y,z in metres")
    parser.add_argument("--task-complete-radius-m", type=float, default=.060)
    parser.add_argument("--task-complete-gripper-open-m", type=float, default=.040)
    parser.add_argument("--task-complete-hold-steps", type=int, default=5)
    parser.add_argument("--task-complete-min-steps", type=int, default=30)
    parser.add_argument("--task-complete-grace-steps", type=int, default=30, help="Feedback-only ticks allowed after the final action")
    parser.add_argument("--task-complete-require-grasp", action="store_true")
    parser.add_argument("--reuse-robot-enabled", action="store_true", help="Reconnect to a controller deliberately left enabled by the sequence")
    parser.add_argument("--keep-robot-enabled", action="store_true", help="After verified task completion, disconnect RPC without Disable for sequence handoff")
    parser.add_argument("--joint-step-max-rad", type=float, default=.05)
    parser.add_argument("--ee-step-max-m", type=float, default=.05)
    parser.add_argument("--ee-z-envelope-margin", type=float, default=.20)
    parser.add_argument("--rotation-step-max-rad", type=float, default=.15)
    parser.add_argument("--gripper-min-distance", type=float, default=0)
    parser.add_argument("--gripper-max-distance", type=float, default=65)
    parser.add_argument("--g-model", type=int, choices=(2, 3), default=3)
    parser.add_argument("--cd-version", type=int, choices=(44, 45), default=45)
    parser.add_argument("--subloop", type=int, choices=(0, 1), default=0)
    return parser


def validate_args(args):
    if args.offline_dry_run and not (args.dry_run and args.no_camera):
        raise SafetyAbort("--offline-dry-run requires --dry-run --no-camera")
    if args.record_dataset and args.action_space != "ee_pose":
        raise SafetyAbort("Dataset recorder currently supports 8D ee_pose only")
    if args.record_dataset and args.smoke_test:
        raise SafetyAbort("Dataset recorder requires validated actions, not --smoke-test")
    if args.record_dataset and (args.record_fps < 1 or args.record_episode_frames < 1):
        raise SafetyAbort("Recording FPS and episode length must be positive")
    if args.smoke_test:
        if not args.no_camera or args.print_rpc:
            raise SafetyAbort("--smoke-test requires --no-camera without --print-rpc; it always uses dry-run")
        args.dry_run = True
    if not (args.dry_run or args.local_check) and args.confirm_hardware != "TB6R5_REAL_INFERENCE":
        raise SafetyAbort("Hardware requires --confirm-hardware TB6R5_REAL_INFERENCE")
    if args.no_camera and not (args.dry_run or args.local_check):
        raise SafetyAbort("Hardware requires real camera frames")
    if args.max_inference_calls < 0 or args.n_action_steps < 1:
        raise SafetyAbort("Invalid inference/action count")
    if args.camera_retries < 1:
        raise SafetyAbort("camera_retries must be at least 1")
    if args.rtc and (args.rtc_num_steps < 1 or args.rtc_min_execution_steps < 1):
        raise SafetyAbort("RTC requires positive denoising and minimum execution steps")
    if args.rtc and (not math.isfinite(args.rtc_guidance_beta) or args.rtc_guidance_beta <= 0):
        raise SafetyAbort("RTC guidance beta must be positive and finite")
    if args.rtc and not 0 <= args.rtc_latency_margin_steps <= 10:
        raise SafetyAbort("RTC latency margin steps must be between 0 and 10")
    if args.rtc and not (math.isfinite(args.rtc_latency_floor_s) and 0 <= args.rtc_latency_floor_s <= 10):
        raise SafetyAbort("RTC latency floor must be finite and between 0 and 10 seconds")
    if args.rtc and args.smoke_test:
        raise SafetyAbort("RTC needs a real-time observation and feedback stream")
    if args.rtc_warmup_only and not (args.rtc and args.dry_run):
        raise SafetyAbort("--rtc-warmup-only requires --rtc --dry-run")
    if args.binary_gripper:
        values = (
            args.gripper_close_command_m,
            args.gripper_close_threshold_m,
            args.gripper_open_command_m,
        )
        if not all(math.isfinite(value) for value in values) or not (0 <= values[0] < values[1] < values[2] <= .080):
            raise SafetyAbort("Binary gripper values must satisfy 0 <= close < threshold < open <= 0.080m")
        if not math.isfinite(args.binary_gripper_resend_delta_mm) or args.binary_gripper_resend_delta_mm <= 0:
            raise SafetyAbort("binary_gripper_resend_delta_mm must be positive")
    if args.task_complete_tcp is not None:
        if args.action_space != "ee_pose":
            raise SafetyAbort("TCP task completion requires --action-space ee_pose")
        if not (math.isfinite(args.task_complete_radius_m) and args.task_complete_radius_m > 0):
            raise SafetyAbort("task_complete_radius_m must be positive")
        if not (math.isfinite(args.task_complete_gripper_open_m) and 0 < args.task_complete_gripper_open_m <= .080):
            raise SafetyAbort("task_complete_gripper_open_m must be in (0, 0.080]")
        if args.task_complete_hold_steps < 1 or args.task_complete_min_steps < 0 or args.task_complete_grace_steps < args.task_complete_hold_steps:
            raise SafetyAbort("Task completion hold/min steps are invalid")
    for name in ("fps", "first_action_timeout_s", "action_timeout_s", "feedback_timeout_s", "feedback_max_age_s", "camera_timeout_s", "joint_step_max_rad", "ee_step_max_m", "rotation_step_max_rad"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise SafetyAbort(f"{name} must be positive and finite")
    if not math.isfinite(args.ee_z_envelope_margin) or not .15 <= args.ee_z_envelope_margin <= .50:
        raise SafetyAbort("ee_z_envelope_margin must be between 0.15 and 0.50")
    if not (math.isfinite(args.gripper_min_distance) and math.isfinite(args.gripper_max_distance)
            and 0 <= args.gripper_min_distance < args.gripper_max_distance <= 80):
        raise SafetyAbort("Gripper limits must satisfy 0 <= min < max <= 80 mm")


def _load_sibling_recorder():
    """Use this directory's writer even if another inference_recorder is on sys.path."""
    path = Path(__file__).with_name("inference_recorder.py")
    name = "rtc_stage_inference_recorder"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def _checkpoint_mean_state(args):
    stats = json.loads(Path(args.norm_stats).read_text())["norm_stats"]
    return np.asarray(stats["state"]["mean"], dtype=np.float32)


def _live_or_offline_state(args, feedback, limits):
    if feedback is not None:
        return limits.validate_state(feedback.read())
    if not getattr(args, "offline_dry_run", False):
        raise SafetyAbort("RTC requires live robot feedback")
    return limits.validate_state(_checkpoint_mean_state(args))


def _record_step(recorder, limits, images, state, target, task):
    recorded = np.array(target, dtype=np.float32, copy=True)
    recorded[-1] = limits.gripper_mm(target) / 1000.0
    recorder.record(images, state, recorded, task)


def run_rtc_loop(args, client, feedback, cameras, formatter, limits, recorder=None):
    """Consume a chunk on control ticks while the next guided chunk is sampled."""

    def observe():
        images = cameras.capture() if cameras else {
            name: np.zeros((224, 224, 3), dtype=np.uint8) for name in ("realsense_0", "realsense_1")
        }
        state = _live_or_offline_state(args, feedback, limits)
        return state, images, prepare_observation(state, images, args.task, allow_constant_images=args.no_camera)

    arm = None
    completion = None
    task_completed = False
    if args.task_complete_tcp is not None and not args.dry_run:
        completion = TaskCompletionDetector(
            target_xyz=args.task_complete_tcp,
            radius_m=args.task_complete_radius_m,
            gripper_open_m=args.task_complete_gripper_open_m,
            hold_steps=args.task_complete_hold_steps,
            min_steps=args.task_complete_min_steps,
            require_grasp=args.task_complete_require_grasp,
            grasp_threshold_m=args.gripper_close_threshold_m,
        )
        print(
            f"[TASK] completion target={completion.target_xyz.tolist()}, radius={completion.radius_m:.3f}m, "
            f"gripper>{completion.gripper_open_m:.3f}m, hold={completion.hold_steps}, "
            f"require_grasp={completion.require_grasp}",
            flush=True,
        )
    calls = steps = 0
    latencies = []
    period = 1.0 / args.fps
    state, images, observation = observe()
    observation["rtc"] = {"num_steps": args.rtc_num_steps}
    print("[RTC] Warming policy before opening the robot command channel", flush=True)
    started = time.monotonic()
    client.infer(observation, args.first_action_timeout_s)
    first_policy_latency = time.monotonic() - started
    # Always obtain the actual initial chunk from a fresh observation after
    # warmup.  This keeps cold and cached runs behaviorally identical.
    state, images, observation = observe()
    observation["rtc"] = {"num_steps": args.rtc_num_steps}
    started = time.monotonic()
    initial_result = client.infer(observation, args.first_action_timeout_s)
    policy_latency = time.monotonic() - started
    print(
        f"[RTC] Policy ready: first={first_policy_latency:.3f}s, steady={policy_latency:.3f}s",
        flush=True,
    )
    actions = quantize_gripper_actions(action_array(initial_result), args)
    latencies.append(policy_latency)
    actions = limits.validate_actions(actions, state)
    horizon = len(actions)
    if horizon < 2:
        raise SafetyAbort(f"RTC requires an action horizon of at least 2; received {horizon}")
    if args.rtc_min_execution_steps >= horizon:
        raise SafetyAbort(
            f"RTC minimum execution steps must be smaller than the checkpoint horizon "
            f"({args.rtc_min_execution_steps} >= {horizon})"
        )
    # executed_steps and delay_steps are static JAX arguments.  Changing either
    # one while the robot is moving causes another compilation and can exhaust
    # the active chunk.  Pick one conservative pair, compile it before opening
    # the command channel, and keep it fixed for the session.
    budget_latency = max(max(latencies), args.rtc_latency_floor_s)
    fixed_delay = math.ceil(budget_latency * args.fps) + 1 + args.rtc_latency_margin_steps
    request_steps = max(args.rtc_min_execution_steps, fixed_delay + 2)
    for warm_attempt in range(1, 4):
        overlap = horizon - request_steps
        if request_steps >= horizon or fixed_delay > overlap:
            raise SafetyAbort(
                f"RTC needs request_step={request_steps}, delay={fixed_delay} for a {horizon}-step "
                "horizon; reduce controller frequency"
            )
        _, _, warm_observation = observe()
        warm_observation["rtc"] = {
            "previous_actions": actions[request_steps:].copy(),
            "executed_steps": request_steps,
            "delay_steps": fixed_delay,
            "num_steps": args.rtc_num_steps,
            "guidance_beta": args.rtc_guidance_beta,
        }
        print(
            f"[RTC] Warming guided sampler before opening the robot command channel "
            f"(executed={request_steps}, delay={fixed_delay}, pass={warm_attempt})",
            flush=True,
        )
        compile_started = time.monotonic()
        warm_actions = action_array(client.infer(warm_observation, args.first_action_timeout_s))
        compile_latency = time.monotonic() - compile_started
        if warm_actions.shape != actions.shape:
            raise SafetyAbort(f"RTC warmup action shape changed from {actions.shape} to {warm_actions.shape}")
        # Always measure the compiled combination once more.  The first call
        # may include XLA compilation; the second is the stable control budget.
        steady_started = time.monotonic()
        steady_actions = action_array(client.infer(warm_observation, args.first_action_timeout_s))
        steady_latency = time.monotonic() - steady_started
        if steady_actions.shape != actions.shape:
            raise SafetyAbort(f"RTC steady warmup action shape changed from {actions.shape} to {steady_actions.shape}")
        print(
            f"[RTC] Guided sampler ready: first={compile_latency:.3f}s, "
            f"steady={steady_latency:.3f}s",
            flush=True,
        )
        latencies.append(steady_latency)
        budget_latency = max(max(latencies), args.rtc_latency_floor_s)
        required_delay = math.ceil(budget_latency * args.fps) + 1 + args.rtc_latency_margin_steps
        required_request_steps = max(args.rtc_min_execution_steps, required_delay + 2)
        if required_delay == fixed_delay and required_request_steps == request_steps:
            break
        fixed_delay = required_delay
        request_steps = required_request_steps
    else:
        raise SafetyAbort("RTC warmup latency did not stabilize after 3 passes")
    latencies = latencies[-10:]
    print(
        f"[RTC] Fixed session parameters: request_step={request_steps}, "
        f"delay_budget={fixed_delay}, horizon={horizon}",
        flush=True,
    )
    if args.rtc_warmup_only:
        print(
            f"[OK] RTC precompiled: request_step={request_steps}, "
            f"delay_budget={fixed_delay}, horizon={horizon}",
            flush=True,
        )
        return 0
    calls += 1
    index = 0
    future = None
    sent_during_inference = 0
    completion_grace_steps = 0
    next_tick = time.monotonic()
    initial_motion = ""
    if args.action_space == "ee_pose":
        initial_motion = (
            f"; state_z={state[2]:.4f}, first_z={actions[0, 2]:.4f}, "
            f"end_z={actions[-1, 2]:.4f}"
        )
    print(
        f"[RTC] Initial chunk ready in {latencies[-1]:.3f}s; {horizon} actions "
        f"at {args.fps:g} Hz{initial_motion}",
        flush=True,
    )
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-infer")
    try:
        while True:
            if future is not None and future.done():
                latency = time.monotonic() - inference_started
                replacement = quantize_gripper_actions(action_array(future.result()), args)
                future = None
                if replacement.shape != actions.shape:
                    raise SafetyAbort(f"RTC action shape changed from {actions.shape} to {replacement.shape}")
                if sent_during_inference >= horizon:
                    raise SafetyAbort("RTC inference arrived after the entire active chunk was executed")
                if sent_during_inference > request_steps:
                    raise SafetyAbort(
                        f"RTC inference took {sent_during_inference} control steps, exceeding the "
                        f"precompiled request step {request_steps}; reduce FPS"
                    )
                current_state = _live_or_offline_state(args, feedback, limits)
                raw_tail = replacement[sent_during_inference:].copy()
                validated_tail = limits.validate_actions(raw_tail, current_state)
                replacement[sent_during_inference:] = validated_tail
                rotation_limited = 0
                max_rotation_correction = 0.0
                if args.action_space == "ee_pose":
                    raw_quat = raw_tail[:, 3:7]
                    raw_quat /= np.linalg.norm(raw_quat, axis=1, keepdims=True)
                    dots = np.abs(np.sum(raw_quat * validated_tail[:, 3:7], axis=1))
                    corrections = 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))
                    # Ignore sub-degree differences caused only by float32
                    # normalization; report actual safety limiting.
                    rotation_limited = int(np.count_nonzero(corrections > 1e-2))
                    max_rotation_correction = float(np.max(corrections, initial=0.0))
                actions = replacement
                index = sent_during_inference
                latencies.append(latency)
                latencies = latencies[-10:]
                motion = ""
                if args.action_space == "ee_pose":
                    motion = (
                        f"; state_z={current_state[2]:.4f}, next_z={actions[index, 2]:.4f}, "
                        f"end_z={actions[-1, 2]:.4f}, rotation_limited={rotation_limited}, "
                        f"max_correction={max_rotation_correction:.4f}rad"
                    )
                print(
                    f"[RTC] chunk {calls}: {latency:.3f}s; skipped {index} elapsed actions{motion}",
                    flush=True,
                )

            step_state = None
            if completion is not None:
                step_state = _live_or_offline_state(args, feedback, limits)
                complete, tcp_distance = completion.update(step_state, steps)
                if completion.consecutive or complete:
                    print(
                        f"[TASK] completion candidate: tcp_distance={tcp_distance:.4f}m, "
                        f"gripper={step_state[-1]:.4f}m, hold={completion.consecutive}/{completion.hold_steps}, "
                        f"grasp_seen={completion.grasp_seen}",
                        flush=True,
                    )
                if complete:
                    task_completed = True
                    print(
                        f"[TASK] COMPLETE after {calls} inference calls and {steps} control steps",
                        flush=True,
                    )
                    break

            more_calls = not args.max_inference_calls or calls < args.max_inference_calls
            if future is None and index > request_steps and more_calls:
                raise SafetyAbort(
                    f"RTC passed its fixed request step ({index} > {request_steps}); reduce FPS"
                )
            if future is None and index == request_steps and more_calls:
                overlap = horizon - request_steps
                _, images, observation = observe()
                observation["rtc"] = {
                    "previous_actions": actions[request_steps:].copy(),
                    "executed_steps": request_steps,
                    "delay_steps": fixed_delay,
                    "num_steps": args.rtc_num_steps,
                    "guidance_beta": args.rtc_guidance_beta,
                }
                inference_started = time.monotonic()
                future = pool.submit(client.infer, observation, args.action_timeout_s)
                sent_during_inference = 0
                calls += 1
                print(f"[RTC] Requested chunk {calls}, overlap={overlap}, delay_budget={fixed_delay}", flush=True)

            if index >= horizon:
                if future is not None:
                    raise SafetyAbort("RTC active chunk exhausted before the next inference completed")
                if completion is not None and completion_grace_steps < args.task_complete_grace_steps:
                    completion_grace_steps += 1
                    next_tick += period
                    time.sleep(max(0.0, next_tick - time.monotonic()))
                    continue
                break

            target = actions[index]
            if step_state is None:
                step_state = _live_or_offline_state(args, feedback, limits)
            target = limits.validate_actions(target[None, :], step_state)[0]
            actions[index] = target
            grip_mm = limits.gripper_mm(target)
            kwargs = dict(max_distance=args.gripper_max_distance, min_distance=args.gripper_min_distance)
            if args.dry_run:
                if args.print_rpc:
                    command = formatter.build_cartesian_stream_cmd(
                        target[:3], target[3:7], grip_mm, int(steps > 0), **kwargs
                    ) if args.action_space == "ee_pose" else formatter.build_subloop1_stream_cmd(
                        target[:6], grip_mm, int(steps > 0), **kwargs
                    )
                    print(f"[RTC][RPC] dry-run would send: {command}", flush=True)
            else:
                if formatter is None:
                    raise SafetyAbort("RTC hardware control requires the robot SDK")
                if arm is None:
                    arm = formatter
                    arm.connect()
                    limits.validate_actions(target[None, :], feedback.read())
                    # Connecting and initializing the robot takes much longer
                    # than one control period.  Start the control clock here so
                    # the loop does not burst through stale ticks to catch up.
                    next_tick = time.monotonic()
                gripper_delta = args.binary_gripper_resend_delta_mm if args.binary_gripper else 0
                sent = (arm.set_cartesian_target_with_gripper(target[:3], target[3:7], grip_mm, cmd_delta=gripper_delta, **kwargs)
                        if args.action_space == "ee_pose" else
                        arm.set_joint_positions_with_gripper(target[:6], grip_mm, cmd_delta=gripper_delta, **kwargs))
                if not sent:
                    raise SafetyAbort("Robot rejected RTC motion command")
                if completion is not None and target[-1] <= args.gripper_close_command_m + 1e-6:
                    # A commanded close is a more reliable grasp-cycle marker
                    # than measured finger spacing, which depends on object width.
                    completion.grasp_seen = True
            if recorder is not None and recorder.is_active():
                step_images = cameras.capture() if cameras else images
                _record_step(recorder, limits, step_images, step_state, target, args.task)
            steps += 1
            index += 1
            if future is not None:
                sent_during_inference += 1
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        try:
            if arm is not None:
                arm.disconnect(disable=not (args.keep_robot_enabled and task_completed))
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    if completion is not None and not task_completed:
        raise SafetyAbort(
            f"Task completion was not reached before the RTC call limit: "
            f"target={completion.target_xyz.tolist()}, radius={completion.radius_m:.3f}m, "
            f"gripper>{completion.gripper_open_m:.3f}m"
        )
    print(
        f"[OK] RTC completed {calls} inference calls and {steps} control steps"
        + (" (task completion detected)" if task_completed else ""),
        flush=True,
    )
    return 0


def run(args):
    validate_args(args)
    needs_robot_sdk = args.local_check or args.print_rpc or not args.offline_dry_run
    if needs_robot_sdk:
        from tb6r5_policy_infer.hardware.tb6r5 import TB6R5Interface, validate_robot_sdk
    else:
        TB6R5Interface = validate_robot_sdk = None
    if args.smoke_test:
        print("[OPENPI] Smoke-test: dry-run with black images and checkpoint mean state; no robot commands", flush=True)
    limits = SafetyLimits.load(
        args.norm_stats, action_space=args.action_space,
        joint_step=args.joint_step_max_rad, tcp_step=args.ee_step_max_m,
        rotation_step=args.rotation_step_max_rad,
        z_margin=args.ee_z_envelope_margin,
        gripper_min_mm=args.gripper_min_distance, gripper_max_mm=args.gripper_max_distance,
    )
    if args.local_check:
        validate_robot_sdk(require_topic=True)
        print(f"[OK] Host dependencies, RPC/Topic SDK and {limits.dimension}D normalization statistics")
        return 0
    client = feedback = cameras = arm = recorder = None
    task_completed = False
    try:
        client = PolicyClient(args.policy_host, args.policy_port, args.first_action_timeout_s)
        validate_policy_server_metadata(client.metadata, dry_run=args.dry_run)
        if args.rtc and not client.metadata.get("rtc_supported"):
            raise SafetyAbort("Policy server does not advertise RTC support")
        print(f"[OPENPI] Server metadata: {client.metadata}", flush=True)
        if not args.no_camera:
            cameras = Cameras(args)
            cameras.start()
        if not (args.smoke_test or args.offline_dry_run):
            feedback = RobotFeedback(args)
            feedback.start()
        recorder_module = _load_sibling_recorder()
        if args.record_dataset:
            recorder = recorder_module.InferenceRecorder(
                args.record_dataset, args.recorder_python,
                args.record_fps, args.record_episode_frames,
            )
            print(f"[RECORD] LeRobotDataset: {recorder.root}", flush=True)
        else:
            recorder = recorder_module.SidecarRecorder(fps=args.fps)
        formatter = None
        if TB6R5Interface is not None:
            formatter = TB6R5Interface(
                args.robot_ip, args.rpc_port, enable_topic=False, print_rpc=args.print_rpc,
                g_model=args.g_model, cd_version=args.cd_version, subloop=args.subloop,
                reuse_enabled=args.reuse_robot_enabled,
            )
        if args.rtc:
            result = run_rtc_loop(args, client, feedback, cameras, formatter, limits, recorder)
            # run_rtc_loop owns and has already disconnected this command
            # channel, possibly leaving the controller enabled for handoff.
            arm = None
            return result
        calls, steps = 0, 0
        completion = None
        if args.task_complete_tcp is not None and not args.dry_run:
            completion = TaskCompletionDetector(
                target_xyz=args.task_complete_tcp,
                radius_m=args.task_complete_radius_m,
                gripper_open_m=args.task_complete_gripper_open_m,
                hold_steps=args.task_complete_hold_steps,
                min_steps=args.task_complete_min_steps,
                require_grasp=args.task_complete_require_grasp,
                grasp_threshold_m=args.gripper_close_threshold_m,
            )
            print(
                f"[TASK] completion target={completion.target_xyz.tolist()}, "
                f"radius={completion.radius_m:.3f}m, gripper>{completion.gripper_open_m:.3f}m, "
                f"hold={completion.hold_steps}, require_grasp={completion.require_grasp}",
                flush=True,
            )
        max_calls = args.max_inference_calls or (1 if args.smoke_test else 0)
        while not max_calls or calls < max_calls:
            images = cameras.capture() if cameras else {name: np.zeros((224, 224, 3), dtype=np.uint8) for name in ("realsense_0", "realsense_1")}
            if feedback:
                state = limits.validate_state(feedback.read())
            else:
                stats = json.loads(args.norm_stats.read_text())["norm_stats"]
                state = np.asarray(stats["state"]["mean"], dtype=np.float32)
            print(f"[OPENPI] state ({args.action_space}): {state.tolist()}", flush=True)
            if args.action_space == "ee_pose":
                print(f"[OPENPI] robottarget={state[:7].tolist()} gripper_mm={state[7] * 1000:.3f} gripper_joint_m={state[7]:.6f}", flush=True)
            observation = prepare_observation(state, images, args.task, allow_constant_images=args.no_camera)
            start = time.monotonic()
            result = client.infer(observation, args.first_action_timeout_s if calls == 0 else args.action_timeout_s)
            actions = action_array(result)
            calls += 1
            print(f"[OPENPI] inference {calls}: {time.monotonic() - start:.3f}s, actions={actions.shape}\n{np.array2string(actions, threshold=np.inf, precision=6)}", flush=True)
            if args.smoke_test:
                if actions.shape[1] != limits.dimension:
                    print(f"[WARNING] Pipeline only: server returns {actions.shape[1]}D actions, checkpoint requires {limits.dimension}D. Hardware validation would reject this output.", flush=True)
                continue
            actions = limits.validate_actions(actions, state)
            for target in actions[:args.n_action_steps]:
                step_start = time.monotonic()
                # Recheck live feedback after a possibly long model call, and before every command.
                step_state = limits.validate_state(feedback.read()) if feedback is not None else state
                if completion is not None:
                    complete, tcp_distance = completion.update(step_state, steps)
                    if completion.consecutive or complete:
                        print(
                            f"[TASK] completion candidate: tcp_distance={tcp_distance:.4f}m, "
                            f"gripper={step_state[-1]:.4f}m, "
                            f"hold={completion.consecutive}/{completion.hold_steps}, "
                            f"grasp_seen={completion.grasp_seen}",
                            flush=True,
                        )
                    if complete:
                        task_completed = True
                        print(
                            f"[TASK] COMPLETE after {calls} inference calls and {steps} control steps",
                            flush=True,
                        )
                        break
                limits.validate_actions(target[None, :], step_state)
                grip_mm = limits.gripper_mm(target)
                kwargs = dict(max_distance=args.gripper_max_distance, min_distance=args.gripper_min_distance)
                if args.dry_run:
                    if args.print_rpc:
                        if args.action_space == "ee_pose":
                            command = formatter.build_cartesian_stream_cmd(target[:3], target[3:7], grip_mm, int(steps > 0), **kwargs)
                        else:
                            command = formatter.build_subloop1_stream_cmd(target[:6], grip_mm, int(steps > 0), **kwargs)
                        print(f"[OPENPI][RPC] dry-run would send: {command}", flush=True)
                else:
                    if arm is None:
                        arm = formatter
                        arm.connect()
                        limits.validate_actions(target[None, :], feedback.read())
                    if args.action_space == "ee_pose":
                        sent = arm.set_cartesian_target_with_gripper(target[:3], target[3:7], grip_mm, cmd_delta=0, **kwargs)
                    else:
                        sent = arm.set_joint_positions_with_gripper(target[:6], grip_mm, cmd_delta=0, **kwargs)
                    if not sent:
                        raise SafetyAbort("Robot rejected motion command")
                if recorder is not None and recorder.is_active():
                    _record_step(recorder, limits, images, step_state, target, args.task)
                steps += 1
                time.sleep(max(0, 1 / args.fps - (time.monotonic() - step_start)))
            if task_completed:
                break
        if completion is not None and not task_completed:
            for _ in range(args.task_complete_grace_steps):
                started = time.monotonic()
                step_state = limits.validate_state(feedback.read())
                complete, tcp_distance = completion.update(step_state, steps)
                if completion.consecutive or complete:
                    print(
                        f"[TASK] completion candidate: tcp_distance={tcp_distance:.4f}m, "
                        f"gripper={step_state[-1]:.4f}m, "
                        f"hold={completion.consecutive}/{completion.hold_steps}, "
                        f"grasp_seen={completion.grasp_seen}",
                        flush=True,
                    )
                if complete:
                    task_completed = True
                    print(
                        f"[TASK] COMPLETE after {calls} inference calls and {steps} control steps",
                        flush=True,
                    )
                    break
                time.sleep(max(0, 1 / args.fps - (time.monotonic() - started)))
        if completion is not None and not task_completed:
            raise SafetyAbort(
                f"Task completion was not reached before the non-RTC call limit: "
                f"target={completion.target_xyz.tolist()}, radius={completion.radius_m:.3f}m, "
                f"gripper>{completion.gripper_open_m:.3f}m"
            )
        mode = "pipeline smoke-test" if args.smoke_test else ("dry-run" if args.dry_run else "hardware")
        print(
            f"[OK] Completed {calls} inference calls ({mode})"
            + (" (task completion detected)" if task_completed else ""),
            flush=True,
        )
        return 0
    finally:
        # Disable the robot even if another resource's cleanup raises.
        try:
            if arm is not None:
                arm.disconnect(disable=not (args.keep_robot_enabled and task_completed))
        finally:
            try:
                if feedback is not None:
                    feedback.close()
            finally:
                try:
                    if cameras is not None:
                        cameras.close()
                finally:
                    try:
                        if client is not None:
                            client.close()
                    finally:
                        if recorder is not None:
                            recorder.close()


def main(argv=None):
    try:
        return run(build_parser().parse_args(argv))
    except KeyboardInterrupt:
        print("[OPENPI] Interrupted; robot command channel closed", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[ABORT] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
