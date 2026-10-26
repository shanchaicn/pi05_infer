"""Host-side OpenPI websocket inference and TB6-R5 control.

The learned model stays in Docker. Dry-run opens only cameras and the read-only
Topic subscriber; it never connects the robot RPC command channel.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import http.client
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
    ROOT / "app/packages/openpi-client/src",
    ROOT / "docker_source/packages/openpi-client/src",
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


def finite_vector(value, dim, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (dim,) or not np.all(np.isfinite(array)):
        raise SafetyAbort(f"{name}: expected finite shape ({dim},), got {array.shape}")
    return array.copy()


def checked_quaternion(value):
    quat = finite_vector(value, 4, "quaternion")
    norm = np.linalg.norm(quat)
    if abs(norm - 1) > .1:
        raise SafetyAbort(f"Quaternion norm must be near 1, got {norm:.4f}")
    quat /= norm
    # Keep the dataset quaternion sign; forcing w >= 0 can leave its training bounds.
    return quat


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
        image = resize_with_pad(raw)
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
    action_space: str = "ee_pose"
    joint_step: float = .03
    tcp_step: float = .03
    rotation_step: float = .15
    gripper_min_mm: float = 30
    gripper_max_mm: float = 70

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
        outside = (value < low - padding) | (value > high + padding)
        if self.action_space == "ee_pose":
            # Quaternion components are coupled and q/-q are equivalent, so
            # per-component quantile bounds can reject a valid nearby pose.
            # Quaternion norm and per-step angular distance remain checked.
            outside[3:7] = False
        if np.any(outside):
            raise SafetyAbort(f"{kind} outside checkpoint training envelope: {value.tolist()}")
        return value

    def validate_actions(self, actions, state):
        actions = action_array({"actions": actions})
        if actions.shape[1] != self.dimension:
            raise SafetyAbort(f"Action dimension {actions.shape[1]} != checkpoint dimension {self.dimension}; no truncation or padding allowed")
        previous = self.validate_state(state)
        for index, raw in enumerate(actions):
            target = self.validate_state(raw, "actions")
            if self.action_space == "ee_pose":
                distance = float(np.linalg.norm(target[:3] - previous[:3]))
                angle = 2 * math.acos(float(np.clip(abs(np.dot(target[3:7], previous[3:7])), 0, 1)))
                if distance > self.tcp_step or angle > self.rotation_step:
                    raise SafetyAbort(f"Unsafe TCP jump at action {index}: {distance:.4f}m, {angle:.4f}rad")
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
    """Capture new frames for each observation; never reuse a cached failed frame."""
    def __init__(self, args):
        self.args = args
        self.sources = {}
        self.caps = {}
        self.realsense = None

    def start(self):
        from tb6r5_policy_infer.camera import parse_camera_urls, parse_camera_devices, parse_camera_serials

        args = self.args
        if args.camera_serials:
            from tb6r5_policy_infer.hardware.realsense import RealSenseCameraInterface

            self.sources = parse_camera_serials(args.camera_serials, {})
            self.realsense = RealSenseCameraInterface(serial_numbers=list(self.sources.values()))
            self.realsense.start()
        elif args.camera_devices:
            import cv2

            self.sources = parse_camera_devices(args.camera_devices)
            for name, device in self.sources.items():
                cap = cv2.VideoCapture(int(device) if device.isdigit() else device)
                self.caps[name] = cap
                if not cap.isOpened():
                    raise SafetyAbort(f"Cannot open camera {name}: {device}")
        else:
            self.sources = parse_camera_urls(args.camera_urls) if args.camera_urls else {
                "realsense_0": f"http://{args.robot_ip}:6888/RsCameraSensor/0/0/color",
                "realsense_1": f"http://{args.robot_ip}:6888/RsCameraSensor/1/0/color",
            }
        if set(self.sources) != {"realsense_0", "realsense_1"}:
            raise SafetyAbort("Camera names must be realsense_0 and realsense_1")
        print(f"[OPENPI] Cameras: {self.sources}", flush=True)

    def _http_image(self, url):
        import cv2
        from tb6r5_policy_infer.camera import HttpCameraStream

        last_error = "no complete JPEG frame"
        for attempt in range(1, self.args.camera_retries + 1):
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                deadline = time.monotonic() + self.args.camera_timeout_s
                with opener.open(url, timeout=self.args.camera_timeout_s) as response:
                    buffer = b""
                    while time.monotonic() < deadline and len(buffer) < 4 * 1024 * 1024:
                        block = response.read1(8192)
                        if not block:
                            break
                        buffer += block
                        jpeg = HttpCameraStream._extract_jpeg_bytes(buffer)
                        if jpeg:
                            bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if bgr is not None:
                                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                last_error = "stream returned no complete JPEG frame"
            except (TimeoutError, urllib.error.URLError, http.client.HTTPException, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.args.camera_retries:
                print(
                    f"[CAMERA] {url} attempt {attempt}/{self.args.camera_retries} failed: "
                    f"{last_error}; retrying",
                    flush=True,
                )
        raise SafetyAbort(
            f"Camera HTTP failed after {self.args.camera_retries} attempts: {url}: {last_error}"
        )

    def capture(self):
        import cv2

        if self.realsense is not None:
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
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {name: pool.submit(self._http_image, url) for name, url in self.sources.items()}
                images = {name: future.result() for name, future in futures.items()}
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
        if self.args.show_camera:
            import cv2
            cv2.destroyAllWindows()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.2.151")
    parser.add_argument("--rpc-port", type=int, default=5868)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8002)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offline-dry-run", action="store_true", help="Synthetic cameras/state; no robot connection (requires --dry-run --no-camera)")
    parser.add_argument("--record-dataset", type=Path, help="New local LeRobot v3 dataset directory")
    parser.add_argument("--recorder-python", default="/home/ai/condaenv/tb6r5/bin/python")
    parser.add_argument("--record-fps", type=int, default=10)
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
    camera_group = parser.add_mutually_exclusive_group()
    for option in ("--camera-urls", "--camera-devices", "--camera-serials"):
        camera_group.add_argument(option)
    parser.add_argument("--norm-stats", type=Path, default=DEFAULT_POLICY_DIR / "assets/norm_stats.json")
    parser.add_argument("--action-space", choices=("ee_pose", "joints"), default="ee_pose")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--n-action-steps", type=int, default=4)
    parser.add_argument("--rtc", action="store_true", help="Asynchronous guided Real-Time Chunking (JAX or PyTorch server)")
    parser.add_argument("--rtc-num-steps", type=int, default=5)
    parser.add_argument("--rtc-min-execution-steps", type=int, default=2)
    parser.add_argument("--rtc-guidance-beta", type=float, default=5.0)
    parser.add_argument("--joint-step-max-rad", type=float, default=.03)
    parser.add_argument("--ee-step-max-m", type=float, default=.03)
    parser.add_argument("--rotation-step-max-rad", type=float, default=.15)
    parser.add_argument("--gripper-min-distance", type=float, default=30)
    parser.add_argument("--gripper-max-distance", type=float, default=70)
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
    if args.record_fps < 1 or args.record_episode_frames < 1:
        raise SafetyAbort("Recording FPS and episode length must be positive")
    if args.rtc and (args.rtc_num_steps < 1 or not 1 <= args.rtc_min_execution_steps < 10):
        raise SafetyAbort("RTC requires positive denoising steps and min execution steps in 1..9")
    if args.rtc and (not math.isfinite(args.rtc_guidance_beta) or args.rtc_guidance_beta <= 0):
        raise SafetyAbort("RTC guidance beta must be positive and finite")
    if args.rtc and args.smoke_test:
        raise SafetyAbort("RTC needs a real-time observation and feedback stream")
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
    for name in ("fps", "first_action_timeout_s", "action_timeout_s", "feedback_timeout_s", "feedback_max_age_s", "camera_timeout_s", "joint_step_max_rad", "ee_step_max_m", "rotation_step_max_rad"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise SafetyAbort(f"{name} must be positive and finite")
    if not (math.isfinite(args.gripper_min_distance) and math.isfinite(args.gripper_max_distance)
            and 0 <= args.gripper_min_distance < args.gripper_max_distance <= 80):
        raise SafetyAbort("Gripper limits must satisfy 0 <= min < max <= 80 mm")


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
    # The dataset stores gripper opening in metres.  gripper_mm clips the command.
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
    calls = steps = 0
    latencies = []
    period = 1.0 / args.fps
    state, images, observation = observe()
    observation["rtc"] = {"num_steps": args.rtc_num_steps}
    print("[RTC] Warming policy before opening the robot command channel", flush=True)
    client.infer(observation, args.first_action_timeout_s)
    state, images, observation = observe()
    observation["rtc"] = {"num_steps": args.rtc_num_steps}
    started = time.monotonic()
    actions = action_array(client.infer(observation, args.first_action_timeout_s))
    latencies.append(time.monotonic() - started)
    actions = limits.validate_actions(actions, state)
    horizon = len(actions)
    if horizon != 10:
        raise SafetyAbort(f"RTC expected this checkpoint's 10-step horizon; received {horizon}")
    calls += 1
    index = 0
    future = None
    sent_during_inference = 0
    next_tick = time.monotonic()
    print(f"[RTC] Initial chunk ready in {latencies[-1]:.3f}s; {horizon} actions at {args.fps:g} Hz", flush=True)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-infer")
    try:
        while True:
            if future is not None and future.done():
                latency = time.monotonic() - inference_started
                replacement = action_array(future.result())
                future = None
                if replacement.shape != actions.shape:
                    raise SafetyAbort(f"RTC action shape changed from {actions.shape} to {replacement.shape}")
                if sent_during_inference >= horizon:
                    raise SafetyAbort("RTC inference arrived after the entire active chunk was executed")
                current_state = _live_or_offline_state(args, feedback, limits)
                limits.validate_actions(replacement[sent_during_inference:], current_state)
                actions = replacement
                index = sent_during_inference
                latencies.append(latency)
                latencies = latencies[-10:]
                print(f"[RTC] chunk {calls}: {latency:.3f}s; skipped {index} elapsed actions", flush=True)

            if (future is None and index >= args.rtc_min_execution_steps
                    and (not args.max_inference_calls or calls < args.max_inference_calls)):
                delay_steps = math.ceil(max(latencies) * args.fps) + 1
                overlap = horizon - index
                if delay_steps > overlap:
                    raise SafetyAbort(
                        f"RTC latency needs {delay_steps} steps, but only {overlap} remain; "
                        "reduce inference latency or controller frequency"
                    )
                _, images, observation = observe()
                observation["rtc"] = {
                    "previous_actions": actions[index:].copy(),
                    "executed_steps": index,
                    "delay_steps": delay_steps,
                    "num_steps": args.rtc_num_steps,
                    "guidance_beta": args.rtc_guidance_beta,
                }
                inference_started = time.monotonic()
                future = pool.submit(client.infer, observation, args.action_timeout_s)
                sent_during_inference = 0
                calls += 1
                print(f"[RTC] Requested chunk {calls}, overlap={overlap}, delay_budget={delay_steps}", flush=True)

            if index >= horizon:
                if future is not None:
                    raise SafetyAbort("RTC active chunk exhausted before the next inference completed")
                break

            target = actions[index]
            step_state = _live_or_offline_state(args, feedback, limits)
            limits.validate_actions(target[None, :], step_state)
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
                sent = (arm.set_cartesian_target_with_gripper(target[:3], target[3:7], grip_mm, cmd_delta=0, **kwargs)
                        if args.action_space == "ee_pose" else
                        arm.set_joint_positions_with_gripper(target[:6], grip_mm, cmd_delta=0, **kwargs))
                if not sent:
                    raise SafetyAbort("Robot rejected RTC motion command")
            if recorder is not None:
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
                arm.disconnect()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    print(f"[OK] RTC completed {calls} inference calls and {steps} control steps", flush=True)
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
        gripper_min_mm=args.gripper_min_distance, gripper_max_mm=args.gripper_max_distance,
    )
    if args.local_check:
        validate_robot_sdk(require_topic=True)
        print(f"[OK] Host dependencies, RPC/Topic SDK and {limits.dimension}D normalization statistics")
        return 0
    client = feedback = cameras = arm = recorder = None
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
        if args.record_dataset:
            from inference_recorder import InferenceRecorder
            recorder = InferenceRecorder(args.record_dataset, args.recorder_python,
                                         args.record_fps, args.record_episode_frames)
            print(f"[RECORD] LeRobotDataset: {recorder.root}", flush=True)
        formatter = None
        if TB6R5Interface is not None:
            formatter = TB6R5Interface(
                args.robot_ip, args.rpc_port, enable_topic=False, print_rpc=args.print_rpc,
                g_model=args.g_model, cd_version=args.cd_version, subloop=args.subloop,
            )
        if args.rtc:
            return run_rtc_loop(args, client, feedback, cameras, formatter, limits, recorder)
        calls, steps = 0, 0
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
                # Pair each recorded action with fresh pre-command camera/state samples.
                record_images = cameras.capture() if recorder is not None and cameras else images
                step_state = limits.validate_state(feedback.read()) if feedback else state
                if recorder is not None:
                    prepare_observation(step_state, record_images, args.task, allow_constant_images=args.no_camera)
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
                if recorder is not None:
                    recorded_target = target.copy()
                    recorded_target[-1] = grip_mm / 1000
                    recorder.record(record_images, step_state, recorded_target, args.task)
                steps += 1
                time.sleep(max(0, 1 / args.fps - (time.monotonic() - step_start)))
        mode = "pipeline smoke-test" if args.smoke_test else ("dry-run" if args.dry_run else "hardware")
        print(f"[OK] Completed {calls} inference calls ({mode})", flush=True)
        return 0
    finally:
        # Disable the robot even if another resource's cleanup raises.
        try:
            if arm is not None:
                arm.disconnect()
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
