"""LeRobot recording and RTC inference on the JAX and PyTorch backends.

No robot, camera, checkpoint, or GPU.  Recording goes through the host runner
against a mock websocket policy.  RTC sampling uses the real JAX Pi0 (tiny
random initialization) and the real PyTorch ``PI0Pytorch.sample_actions_rtc``
body, and checks that the two guided integrators agree.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest

import numpy as np


os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docker_source/packages/openpi-client/src"))
sys.path.insert(0, str(ROOT / "rtc_stage"))
sys.path.insert(0, str(ROOT))

import openpi_runner as runner  # noqa: E402


def _write_stats(path: Path) -> None:
    state = runner.DEFAULT_DRY_STATE.tolist()
    low = [-.5, -1, 0, -1, -1, -1, -1, 0]
    high = [.5, 0, 1, 1, 1, 1, 1, .08]
    path.write_text(json.dumps({"norm_stats": {
        key: {"q01": low, "q99": high, "mean": state, "std": [1.0] * 8}
        for key in ("state", "actions")
    }}))


def _load_dataset(root: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/pi05_inference", root=root)
    if len(dataset) < 1:
        raise AssertionError(f"dataset at {root} reloaded empty")
    row = dataset[0]
    state = np.asarray(row["observation.state"], dtype=np.float32)
    action = np.asarray(row["action"], dtype=np.float32)
    if state.shape[-1] != 8 or action.shape[-1] != 8:
        raise AssertionError(f"expected 8D state/action, got {state.shape} {action.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise AssertionError("reloaded state or action is non-finite")
    return dataset, state, action


class GuidanceAgreementTests(unittest.TestCase):
    def test_soft_mask_and_guided_euler_match_on_both_backends(self):
        import jax.numpy as jnp
        import torch

        from openpi.models.rtc_guidance import guidance_scale_value
        from openpi.models.rtc_guidance import soft_mask_weights
        from openpi.models.rtc_jax import guided_euler_step as jax_step
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.models_pytorch.rtc import guided_euler_step as torch_step
        from openpi.models_pytorch.rtc import guidance_scale
        from openpi.models_pytorch.rtc import soft_mask

        horizon, executed, delay, width, steps, beta = 6, 2, 2, 4, 4, 5.0
        weights = np.asarray(soft_mask_weights(horizon, executed, delay), dtype=np.float32)
        torch_weights = soft_mask(horizon, executed, delay, device="cpu").numpy()
        np.testing.assert_allclose(weights, torch_weights)
        self.assertEqual(weights[:delay].tolist(), [1.0] * delay)
        self.assertTrue(np.all(weights[horizon - executed:] == 0))
        self.assertAlmostEqual(guidance_scale(0.4, beta), guidance_scale_value(0.4, beta))
        with self.assertRaises(ValueError):
            soft_mask_weights(horizon, 0, 0)

        noise = np.linspace(-0.5, 0.5, horizon * width, dtype=np.float32).reshape(1, horizon, width)
        previous = np.linspace(0.1, 0.4, (horizon - executed) * width, dtype=np.float32).reshape(1, horizon - executed, width)
        target = np.zeros((1, horizon, width), dtype=np.float32)
        target[:, : horizon - executed] = previous
        weight_b = weights[None, :, None]

        def integrate(step_fn, array):
            state = array(noise)
            target_array = array(target)
            weight_array = array(weight_b)
            for index in range(steps):
                time_value = 1.0 - index / steps
                state = step_fn(
                    state, lambda current: 0.5 * current, time_value, steps, target_array, weight_array, beta
                )
            return np.asarray(state, dtype=np.float32)

        jax_actions = integrate(jax_step, jnp.asarray)
        torch_actions = integrate(torch_step, torch.tensor)
        np.testing.assert_allclose(jax_actions, torch_actions, rtol=1e-5, atol=1e-5)

        class _Config:
            action_horizon = horizon
            action_dim = width

        class _Stub:
            config = _Config()

            def __init__(self):
                language = type("LM", (), {})()
                language.config = type("C", (), {"_attn_implementation": "eager"})()
                paligemma = type("P", (), {"language_model": language})()

                def forward(**kwargs):
                    return None, None

                self.paligemma_with_expert = type("E", (), {"paligemma": paligemma, "forward": staticmethod(forward)})()

            def _preprocess_observation(self, observation, train=False):
                return None, None, None, None, observation.state

            def embed_prefix(self, images, masks, tokens, token_masks):
                pad = torch.ones(1, 2, dtype=torch.bool)
                attention = torch.zeros(1, 2, dtype=torch.int64)
                return torch.zeros(1, 2, 4), pad, attention

            def _prepare_attention_masks_4d(self, mask):
                return mask

            def denoise_step(self, state, prefix_pad, cache, x_t, timestep):
                return 0.5 * x_t

        observation = type("Obs", (), {"state": torch.zeros(1, width)})()
        produced = PI0Pytorch.sample_actions_rtc(
            _Stub(),
            "cpu",
            observation,
            torch.tensor(previous),
            executed,
            delay,
            noise=torch.tensor(noise),
            num_steps=steps,
            guidance_beta=beta,
        )
        self.assertEqual(tuple(produced.shape), (1, horizon, width))
        np.testing.assert_allclose(produced.numpy(), torch_actions, rtol=1e-5, atol=1e-5)


class JaxRtcModelTests(unittest.TestCase):
    def test_tiny_pi05_rtc_chunk_matches_plain_sampler_shape(self):
        import jax
        import jax.numpy as jnp

        from openpi.models.pi0_config import Pi0Config

        config = Pi0Config(
            pi05=True,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_horizon=4,
            action_dim=8,
            max_token_len=8,
            vision_variant="mu/32",
            dtype="float32",
            pytorch_compile_mode=None,
        )
        model = config.create(jax.random.key(0))
        observation = config.fake_obs(batch_size=1)
        noise = jax.random.normal(jax.random.key(1), (1, config.action_horizon, config.action_dim))
        plain = np.asarray(model.sample_actions(jax.random.key(2), observation, num_steps=2, noise=noise))
        previous = jnp.asarray(np.linspace(0.0, 0.2, 2 * config.action_dim, dtype=np.float32).reshape(1, 2, config.action_dim))
        guided = np.asarray(model.sample_actions_rtc(
            jax.random.key(2), observation, previous, 2, 1, num_steps=2, noise=noise, guidance_beta=5.0
        ))
        self.assertEqual(plain.shape, (1, config.action_horizon, config.action_dim))
        self.assertEqual(guided.shape, plain.shape)
        self.assertTrue(np.isfinite(guided).all())
        self.assertFalse(np.allclose(plain, guided, atol=1e-5, rtol=1e-5))

        # The websocket policy JIT-compiles this method, including the VJP guidance step.
        from openpi.shared import nnx_utils

        compiled = nnx_utils.module_jit(
            model.sample_actions_rtc,
            static_argnames=("executed_steps", "delay_steps", "num_steps", "guidance_beta"),
        )
        compiled_actions = np.asarray(compiled(
            jax.random.key(2),
            observation,
            previous_actions=previous,
            executed_steps=2,
            delay_steps=1,
            num_steps=2,
            noise=noise,
            guidance_beta=5.0,
        ))
        self.assertEqual(compiled_actions.shape, guided.shape)
        self.assertTrue(np.isfinite(compiled_actions).all())
        np.testing.assert_allclose(compiled_actions, guided, rtol=1e-4, atol=1e-4)


class PolicyRtcWiringTests(unittest.TestCase):
    def _observation(self, rtc=None):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        observation = {
            "state": np.zeros((32,), dtype=np.float32),
            "image": {key: image for key in keys},
            "image_mask": {key: np.ones((), dtype=bool) for key in keys},
            "tokenized_prompt": np.zeros((4,), dtype=np.int32),
            "tokenized_prompt_mask": np.ones((4,), dtype=bool),
        }
        if rtc is not None:
            observation["rtc"] = rtc
        return observation

    def _policy(self, backend):
        import flax.nnx as nnx
        import jax.numpy as jnp
        import torch
        from openpi.policies.policy import Policy
        from openpi.policies.tb6r5_policy import TB6R5Outputs
        from openpi.shared.normalize import NormStats
        from openpi import transforms

        class TinyJax(nnx.Module):
            def __init__(self):
                self.gain = nnx.Param(jnp.zeros((), dtype=jnp.float32))
                self.action_horizon = 10
                self.action_dim = 32

            def sample_actions(self, rng, observation, num_steps=10, noise=None):
                return jnp.ones((observation.state.shape[0], 10, 32))

            def sample_actions_rtc(self, rng, observation, previous_actions, executed_steps, delay_steps, *, num_steps=5, noise=None, guidance_beta=5.0):
                output = jnp.zeros((previous_actions.shape[0], 10, 32))
                return output.at[:, : previous_actions.shape[1]].set(previous_actions)

        class TinyTorch(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(()))
                self.config = type("Cfg", (), {"action_horizon": 10, "action_dim": 32})()

            def sample_actions(self, device, observation, noise=None, num_steps=10):
                return torch.ones(observation.state.shape[0], 10, 32, device=device)

            def sample_actions_rtc(self, device, observation, previous_actions, executed_steps, delay_steps, noise=None, num_steps=5, guidance_beta=5.0):
                output = torch.zeros(previous_actions.shape[0], 10, 32, device=device)
                output[:, : previous_actions.shape[1]] = previous_actions
                return output

        normalizer = transforms.Normalize({"actions": NormStats(mean=np.zeros(8, np.float32), std=np.ones(8, np.float32))})
        return Policy(
            TinyTorch() if backend == "pytorch" else TinyJax(),
            transforms=[normalizer],
            output_transforms=[TB6R5Outputs(action_dim=8)],
            is_pytorch=backend == "pytorch",
            pytorch_device="cpu",
        )

    def test_both_backends_accept_initial_and_guided_rtc(self):
        previous = np.arange(8 * 8, dtype=np.float32).reshape(8, 8) * 0.01
        guided = {
            "previous_actions": previous,
            "executed_steps": 2,
            "delay_steps": 3,
            "num_steps": 4,
            "guidance_beta": 5.0,
        }
        for backend in ("jax", "pytorch"):
            with self.subTest(backend=backend):
                policy = self._policy(backend)
                self.assertTrue(policy.metadata["rtc_supported"])
                self.assertIs(policy.metadata["hardware_allowed"], False)
                initial = policy.infer(self._observation({"num_steps": 4}))
                self.assertEqual(initial["actions"].shape, (10, 8))
                np.testing.assert_allclose(initial["actions"], 1.0)
                result = policy.infer(self._observation(guided))
                self.assertEqual(result["actions"].shape, (10, 8))
                np.testing.assert_allclose(result["actions"][:8], previous, atol=1e-5)


class RecordingTests(unittest.TestCase):
    def test_staged_sync_writer_round_trips(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "rtc_stage_inference_recorder_test", ROOT / "rtc_stage" / "inference_recorder.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            recorder = module.InferenceRecorder(root, sys.executable, 5, 10)
            images = {
                name: np.zeros((32, 48, 3), dtype=np.uint8) for name in ("realsense_0", "realsense_1")
            }
            state = np.linspace(0.0, 0.07, 8, dtype=np.float32)
            try:
                recorder.record(images, state, state, "pick")
                recorder.record(images, state, state + 0.001, "pick")
            finally:
                recorder.close()
            dataset, loaded_state, loaded_action = _load_dataset(root)
            self.assertEqual(len(dataset), 2)
            np.testing.assert_allclose(loaded_state, state, atol=1e-5)
            np.testing.assert_allclose(loaded_action, state, atol=1e-5)

    def _serve(self, backend, seen):
        from websockets.sync.server import serve

        def handler(connection):
            connection.send(runner.pack_message({
                "backend": backend,
                "rtc_supported": True,
                "hardware_allowed": False,
            }))
            while True:
                try:
                    payload = connection.recv(timeout=30)
                except Exception:
                    return
                observation = runner.unpack_message(payload)
                seen.append(observation.get("rtc"))
                state = np.asarray(observation["observation/state"], dtype=np.float32)
                connection.send(runner.pack_message({"actions": np.repeat(state[None, :], 10, axis=0)}))

        server = serve(handler, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def _run(self, backend, rtc, root):
        seen = []
        server, thread = self._serve(backend, seen)
        port = server.socket.getsockname()[1]
        try:
            with tempfile.TemporaryDirectory() as directory:
                stats = Path(directory) / "norm_stats.json"
                _write_stats(stats)
                argv = [
                    "--offline-dry-run", "--dry-run", "--no-camera", "--no-show-camera",
                    "--policy-host", "127.0.0.1", "--policy-port", str(port),
                    "--norm-stats", str(stats),
                    "--record-dataset", str(root),
                    "--recorder-python", sys.executable,
                    "--record-fps", "5", "--record-episode-frames", "50",
                    "--fps", "4",
                    "--first-action-timeout-s", "30", "--action-timeout-s", "30",
                    "--max-inference-calls", "2" if rtc else "1",
                    "--n-action-steps", "4",
                ]
                if rtc:
                    argv += ["--rtc", "--rtc-num-steps", "2", "--rtc-min-execution-steps", "2"]
                args = runner.build_parser().parse_args(argv)
                self.assertEqual(runner.run(args), 0)
        finally:
            server.shutdown()
            thread.join(timeout=5)
        return seen

    def test_recording_and_rtc_against_jax_and_pytorch_servers(self):
        for backend in ("jax", "pytorch"):
            for rtc in (False, True):
                with self.subTest(backend=backend, rtc=rtc):
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory) / "dataset"
                        seen = self._run(backend, rtc, root)
                        dataset, state, action = _load_dataset(root)
                        self.assertGreaterEqual(len(dataset), 4)
                        self.assertEqual(state.shape[-1], 8)
                        self.assertEqual(action.shape[-1], 8)
                        if rtc:
                            guided = [request for request in seen if request and "previous_actions" in request]
                            self.assertGreaterEqual(len(guided), 1)
                            previous = np.asarray(guided[0]["previous_actions"])
                            self.assertEqual(previous.shape[1], 8)
                            self.assertGreaterEqual(int(guided[0]["executed_steps"]), 1)
                            self.assertGreaterEqual(int(guided[0]["delay_steps"]), 1)


if __name__ == "__main__":
    unittest.main()
