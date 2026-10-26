from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_pi05_tb6r5.py"
SPEC = importlib.util.spec_from_file_location("run_pi05_tb6r5", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stats_path = Path(self.temp.name) / 'norm_stats.json'
        low = [-.5, -1, 0, -1, -1, -1, -1, 0]
        high = [.5, 0, 1, 1, 1, 1, 1, .08]
        self.stats_path.write_text(json.dumps({'norm_stats': {
            key: {'q01': low, 'q99': high, 'mean': runner.DEFAULT_DRY_STATE.tolist()}
            for key in ('state', 'actions')
        }}))
        self.limits = runner.SafetyLimits.load(self.stats_path)

    def test_compatibility_script_uses_local_backend(self):
        self.assertEqual(
            runner.PolicyClient.__module__,
            "openpi_runner",
        )

    def test_hardware_requires_confirmation_and_real_cameras(self):
        for argv in ([], ["--confirm-hardware", "wrong"],
                     ["--confirm-hardware", "TB6R5_REAL_INFERENCE", "--no-camera"]):
            with self.subTest(argv=argv), self.assertRaises(runner.SafetyAbort):
                runner.validate_args(runner.build_parser().parse_args(argv))

    def test_smoke_test_requires_no_camera_and_cannot_print_rpc(self):
        for argv in (["--smoke-test"], ["--smoke-test", "--dry-run"],
                     ["--smoke-test", "--no-camera", "--print-rpc"],
                     ["--smoke-test", "--dry-run", "--no-camera", "--print-rpc"]):
            with self.subTest(argv=argv), self.assertRaises(runner.SafetyAbort):
                runner.validate_args(runner.build_parser().parse_args(argv))

    def test_smoke_test_implies_dry_run(self):
        for extra in ([], ["--dry-run"], ["--confirm-hardware", "TB6R5_REAL_INFERENCE"]):
            with self.subTest(extra=extra):
                args = runner.build_parser().parse_args(["--smoke-test", "--no-camera", *extra])
                runner.validate_args(args)
                self.assertTrue(args.dry_run)

    def test_rejects_invalid_control_parameters(self):
        for argv in (["--fps", "nan"], ["--ee-step-max-m", "0"],
                     ["--max-inference-calls", "-1"], ["--gripper-min-distance", "75"]):
            with self.subTest(argv=argv), self.assertRaises(runner.SafetyAbort):
                runner.validate_args(runner.build_parser().parse_args(["--dry-run", *argv]))

    def test_placeholder_server_is_dry_run_only(self):
        metadata = {"placeholder": True, "warning": "test placeholder"}
        runner.validate_policy_server_metadata(metadata, dry_run=True)
        with self.assertRaises(runner.SafetyAbort):
            runner.validate_policy_server_metadata(metadata, dry_run=False)

    def test_msgpack_numpy_round_trip(self):
        payload = {"state": np.arange(7, dtype=np.float32), "prompt": "test"}
        decoded = runner.unpack_message(runner.pack_message(payload))
        np.testing.assert_array_equal(decoded["state"], payload["state"])
        self.assertEqual(decoded["prompt"], "test")

    def test_resize_with_pad(self):
        image = np.full((480, 640, 3), 127, dtype=np.uint8)
        resized = runner.resize_with_pad(image)
        self.assertEqual(resized.shape, (224, 224, 3))
        self.assertEqual(resized.dtype, np.uint8)

    def test_safe_action_chunk(self):
        state = runner.DEFAULT_DRY_STATE.copy()
        actions = np.repeat(state[None, :], 10, axis=0)
        validated = self.limits.validate_actions(actions, state)
        self.assertEqual(validated.shape, (10, 8))

    def test_rejects_non_finite_action(self):
        state = runner.DEFAULT_DRY_STATE.copy()
        actions = np.repeat(state[None, :], 10, axis=0)
        actions[0, 0] = np.nan
        with self.assertRaises(runner.SafetyAbort):
            self.limits.validate_actions(actions, state)

    def test_rejects_dimension_mismatch_without_padding(self):
        with self.assertRaises(runner.SafetyAbort):
            self.limits.validate_actions(np.zeros((10, 7)), runner.DEFAULT_DRY_STATE)

    def test_rejects_bad_quaternion_and_tcp_jump(self):
        for change in (lambda a: a.__setitem__(slice(3, 7), 0),
                       lambda a: a.__setitem__(0, a[0] + .1)):
            action = runner.DEFAULT_DRY_STATE.copy()
            change(action)
            with self.assertRaises(runner.SafetyAbort):
                self.limits.validate_actions(action[None, :], runner.DEFAULT_DRY_STATE)

    def test_rejects_later_unsafe_actions_in_chunk(self):
        actions = np.repeat(runner.DEFAULT_DRY_STATE[None, :], 10, axis=0)
        actions[-1, 2] += .1
        with self.assertRaises(runner.SafetyAbort):
            self.limits.validate_actions(actions, runner.DEFAULT_DRY_STATE)

    def test_gripper_metres_converted_and_limited_to_mm(self):
        action = runner.DEFAULT_DRY_STATE.copy()
        action[-1] = .04
        self.assertAlmostEqual(self.limits.gripper_mm(action), 40, places=4)
        action[-1] = .005
        self.assertEqual(self.limits.gripper_mm(action), 30)

    def test_missing_norm_stats_fails_closed(self):
        with self.assertRaises(FileNotFoundError):
            runner.SafetyLimits.load(Path('/does/not/exist'))

    def test_feedback_rejects_unchanged_controller_timestamp(self):
        feedback, _ = self.make_feedback()
        with mock.patch('openpi_runner.time.monotonic', side_effect=[1., 2.]):
            np.testing.assert_allclose(feedback.read(), runner.DEFAULT_DRY_STATE)
            with self.assertRaisesRegex(runner.SafetyAbort, 'Stale'):
                feedback.read()

    def make_feedback(self):
        args = runner.build_parser().parse_args(['--dry-run'])
        feedback = runner.RobotFeedback(args)
        feedback.reader._topic = mock.Mock()
        snapshot = feedback.reader._topic.get_system_state.return_value
        snapshot.has_rt.return_value = True
        state = runner.DEFAULT_DRY_STATE
        rt = SimpleNamespace(
            header_timestamp=123,
            models=[SimpleNamespace(joint_start_idx=0, joint_count=6),
                    SimpleNamespace(joint_start_idx=6, joint_count=1)],
            models_joints=[SimpleNamespace(position=0.) for _ in range(6)] + [SimpleNamespace(position=.04)],
            models_current_points=[SimpleNamespace(robottarget=state[:7].tolist())],
        )
        snapshot.get_rt.return_value = rt
        return feedback, rt

    def test_feedback_uses_one_snapshot(self):
        feedback, _ = self.make_feedback()
        feedback.read()
        feedback.reader._topic.get_system_state.assert_called_once()

    def test_missing_gripper_model_is_identified(self):
        feedback, rt = self.make_feedback()
        rt.models.pop()
        with self.assertRaisesRegex(runner.SafetyAbort, 'Missing gripper model'):
            feedback.read()

    def test_missing_gripper_joint_is_identified(self):
        feedback, rt = self.make_feedback()
        rt.models[1].joint_start_idx = 7
        with self.assertRaisesRegex(runner.SafetyAbort, 'Missing gripper joint'):
            feedback.read()

    def test_gripper_rejects_non_finite_or_out_of_range_feedback(self):
        for position in (float('nan'), float('inf'), -.01, .1):
            feedback, rt = self.make_feedback()
            rt.models_joints[6].position = position
            with self.subTest(position=position), self.assertRaisesRegex(runner.SafetyAbort, 'Invalid gripper'):
                feedback.read()

    def test_gripper_endpoint_roundoff_is_tolerated(self):
        for position, expected in ((-2e-18, 0.), (.08 + 1e-10, .08)):
            feedback, rt = self.make_feedback()
            rt.models_joints[6].position = position
            with self.subTest(position=position):
                self.assertAlmostEqual(float(feedback.read()[-1]), expected, places=6)

    def test_tcp_mode_does_not_require_joint_velocity_fields(self):
        feedback, rt = self.make_feedback()
        rt.models[0].joint_count = 0
        np.testing.assert_allclose(feedback.read(), runner.DEFAULT_DRY_STATE)

    def test_joint_mode_requires_all_six_joint_positions(self):
        feedback, rt = self.make_feedback()
        feedback.args.action_space = 'joints'
        np.testing.assert_allclose(feedback.read(), [0, 0, 0, 0, 0, 0, 40])
        rt.models[0].joint_count = 5
        with self.assertRaisesRegex(runner.SafetyAbort, 'Missing arm joint'):
            feedback.read()

    def test_hardware_rejects_bad_output_before_connecting_rpc(self):
        from tb6r5_policy_infer.hardware import tb6r5
        args = runner.build_parser().parse_args([
            '--confirm-hardware', 'TB6R5_REAL_INFERENCE', '--no-show-camera',
            '--max-inference-calls', '1', '--norm-stats', str(self.stats_path),
        ])
        images = {name: np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
                  for name in ('realsense_0', 'realsense_1')}
        with mock.patch('openpi_runner.PolicyClient') as client, \
             mock.patch('openpi_runner.RobotFeedback') as feedback, \
             mock.patch('openpi_runner.Cameras') as cameras, \
             mock.patch.object(tb6r5.TB6R5Interface, 'connect') as connect:
            client.return_value.metadata = {}
            client.return_value.infer.return_value = {'actions': np.zeros((10, 7))}
            feedback.return_value.read.return_value = runner.DEFAULT_DRY_STATE.copy()
            cameras.return_value.capture.return_value = images
            with self.assertRaisesRegex(runner.SafetyAbort, 'dimension'):
                runner.run(args)
            connect.assert_not_called()
            client.return_value.close.assert_called_once()

    def test_rejects_missing_and_constant_camera_images(self):
        black = np.zeros((224, 224, 3), dtype=np.uint8)
        for images in ({}, {"realsense_0": black, "realsense_1": black}):
            with self.assertRaises(runner.SafetyAbort):
                runner.prepare_observation(runner.DEFAULT_DRY_STATE, images, runner.DEFAULT_TASK)

    def test_smoke_test_never_opens_robot_or_feedback(self):
        from tb6r5_policy_infer.hardware import tb6r5
        args = runner.build_parser().parse_args(["--smoke-test", "--no-camera", "--no-show-camera", "--max-inference-calls", "10", "--norm-stats", str(self.stats_path)])
        response = {"actions": np.zeros((10, 8), dtype=np.float32)}
        with mock.patch('openpi_runner.PolicyClient') as client, \
             mock.patch('openpi_runner.RobotFeedback') as feedback, \
             mock.patch.object(tb6r5.TB6R5Interface, 'connect') as connect:
            client.return_value.metadata = {}
            client.return_value.infer.return_value = response
            self.assertEqual(runner.run(args), 0)
            self.assertEqual(client.return_value.infer.call_count, 10)
            feedback.assert_not_called()
            connect.assert_not_called()
            client.return_value.close.assert_called_once()

    def test_dry_run_prints_rpc_without_opening_command_channel(self):
        from tb6r5_policy_infer.hardware import tb6r5
        args = runner.build_parser().parse_args(["--dry-run", "--print-rpc", "--no-show-camera", "--max-inference-calls", "2", "--norm-stats", str(self.stats_path)])
        response = {"actions": np.repeat(runner.DEFAULT_DRY_STATE[None, :], 10, axis=0)}
        images = {name: np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
                  for name in ('realsense_0', 'realsense_1')}
        with mock.patch('openpi_runner.PolicyClient') as client, \
             mock.patch('openpi_runner.RobotFeedback') as feedback, \
             mock.patch('openpi_runner.Cameras') as cameras, \
             mock.patch.object(tb6r5.TB6R5Interface, 'connect') as connect, \
             mock.patch.object(tb6r5.TB6R5Interface, 'set_cartesian_target_with_gripper') as send:
            client.return_value.metadata = {}
            client.return_value.infer.return_value = response
            feedback.return_value.read.return_value = runner.DEFAULT_DRY_STATE.copy()
            cameras.return_value.capture.return_value = images
            self.assertEqual(runner.run(args), 0)
            self.assertEqual(client.return_value.infer.call_count, 2)
            connect.assert_not_called()
            send.assert_not_called()
            feedback.return_value.close.assert_called_once()
            cameras.return_value.close.assert_called_once()

    def test_hardware_error_disconnects_and_closes_resources(self):
        from tb6r5_policy_infer.hardware import tb6r5
        args = runner.build_parser().parse_args(["--confirm-hardware", "TB6R5_REAL_INFERENCE", "--no-show-camera", "--max-inference-calls", "1", "--norm-stats", str(self.stats_path)])
        images = {name: np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
                  for name in ('realsense_0', 'realsense_1')}
        with mock.patch('openpi_runner.PolicyClient') as client, \
             mock.patch('openpi_runner.RobotFeedback') as feedback, \
             mock.patch('openpi_runner.Cameras') as cameras, \
             mock.patch.object(tb6r5, 'TB6R5Interface') as arm:
            client.return_value.metadata = {}
            client.return_value.infer.return_value = {'actions': runner.DEFAULT_DRY_STATE[None, :]}
            feedback.return_value.read.return_value = runner.DEFAULT_DRY_STATE.copy()
            cameras.return_value.capture.return_value = images
            arm.return_value.set_cartesian_target_with_gripper.return_value = False
            with self.assertRaises(runner.SafetyAbort):
                runner.run(args)
            arm.return_value.disconnect.assert_called_once()
            client.return_value.close.assert_called_once()
            feedback.return_value.close.assert_called_once()
            cameras.return_value.close.assert_called_once()

    @unittest.skipUnless(
        os.environ.get("PI05_RUN_LIVE_POLICY_DIMENSION_TEST") == "1",
        "Set PI05_RUN_LIVE_POLICY_DIMENSION_TEST=1 to call the local policy service.",
    )
    def test_live_policy_returns_raw_eight_dimensional_actions(self):
        """Calls only the local policy WebSocket; it never opens robot RPC."""
        checkpoint_dir = Path(
            os.environ.get("CHECKPOINT_HOST_DIR", "/home/ai/pi0.5/model/pytorch/pp_white_red/19999")
        )
        stats_path = checkpoint_dir / "assets" / "norm_stats.json"
        with stats_path.open(encoding="utf-8") as handle:
            stats = json.load(handle)
        declared_action_dim = len(stats["norm_stats"]["actions"]["q01"])
        self.assertEqual(declared_action_dim, 8)

        black = np.zeros((640, 640, 3), dtype=np.uint8)
        observation = runner.prepare_observation(
            runner.DEFAULT_DRY_STATE,
            {"realsense_0": black, "realsense_1": black},
            runner.DEFAULT_TASK,
            allow_constant_images=True,
        )
        client = runner.PolicyClient("127.0.0.1", 8002, connect_timeout_s=120.0)
        try:
            response = client.infer(observation, timeout_s=120.0)
        finally:
            client.close()

        self.assertIn("actions", response)
        actions = np.asarray(response["actions"])
        self.assertEqual(actions.ndim, 2)
        self.assertGreater(actions.shape[0], 0)
        self.assertTrue(np.all(np.isfinite(actions)))
        self.assertEqual(actions.shape[1], 8)
        self.assertEqual(actions.shape[1], declared_action_dim)
        print("\n--- Live policy action-dimension result ---")
        print(f"Server metadata: {client.metadata}")
        print(f"Raw action chunk shape: {actions.shape}")
        print(f"Service action dimension: {actions.shape[1]}")
        print(f"Checkpoint norm_stats action dimension: {declared_action_dim}")
        print("Service action dimension matches checkpoint norm_stats.")
        print("Raw returned actions:")
        print(np.array2string(actions, precision=6, suppress_small=False))
        print("--- End live policy result ---")


if __name__ == "__main__":
    unittest.main()
