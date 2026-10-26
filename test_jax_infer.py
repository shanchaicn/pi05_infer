"""Offline checks; PI05_TEST_JAX=1 also exercises Orbax -> OpenPI -> JAX JIT."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest

import numpy as np

import jax_infer as runtime


def write_stats(root, dimension=8, asset_id="."):
    stats = {
        key: {"mean": [0.0] * dimension, "std": [1.0] * dimension}
        for key in ("state", "actions")
    }
    assets = root / "assets" / asset_id
    assets.mkdir(parents=True)
    (assets / "norm_stats.json").write_text(json.dumps({"norm_stats": stats}))


def smoke_args(**kwargs):
    return SimpleNamespace(
        smoke_test=True, base_image=None, wrist_image=None, observation=None, state=None,
        task="pick up the ring", **kwargs
    )


class RuntimeTests(unittest.TestCase):
    def test_pytorch_only_checkpoint_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").touch()
            with self.assertRaisesRegex(ValueError, "JAX params"):
                runtime.validate_checkpoint(root)

    def test_dimensions_and_nested_assets_come_from_checkpoint(self):
        for dimension in (7, 8, 12):
            with self.subTest(dimension=dimension), TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "params").mkdir()
                (root / "params/_METADATA").touch()
                write_stats(root, dimension, "new_dataset")
                stats = runtime.validate_checkpoint(root)
                self.assertEqual(len(stats["state"]["mean"]), dimension)
                self.assertEqual(runtime.validate_checkpoint(root, "new_dataset"), stats)

    def test_inconsistent_stats_dimensions_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "params").mkdir()
            (root / "params/_METADATA").touch()
            write_stats(root)
            path = root / "assets/norm_stats.json"
            payload = json.loads(path.read_text())
            payload["norm_stats"]["actions"]["std"].pop()
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "inconsistent actions"):
                runtime.validate_checkpoint(root)

    def test_observation_rejects_nan_and_wrong_state_dimension(self):
        for state in ("0,0,0,0,0,0,1,nan", "0,0,0,0,0,0,0"):
            args = smoke_args()
            args.state = state
            with self.assertRaises(ValueError):
                runtime.build_observation(args, state_dim=8)

    def test_raw_observation_keeps_custom_keys_quaternion_order_and_units(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "observation.npz"
            state = np.array([0.3, -0.1, 0.4, 0, 0, 0, 1, 0.04], dtype=np.float32)
            np.savez(path, custom_state=state, custom_camera=np.zeros((8, 12, 3), dtype=np.uint8), prompt="pick")
            args = smoke_args()
            args.observation, args.smoke_test, args.task = path, False, None
            obs = runtime.build_observation(args, state_dim=8)
            np.testing.assert_array_equal(obs["custom_state"], state)
            self.assertEqual(obs["prompt"], "pick")
            self.assertEqual(set(obs), {"custom_state", "custom_camera", "prompt"})

    def test_smoke_state_uses_checkpoint_reference_without_assuming_semantics(self):
        reference = [0.3, -0.1, 0.4, 0, 0, 0, 1, 0.04]
        obs = runtime.build_observation(smoke_args(), state_dim=8, smoke_state=reference)
        np.testing.assert_array_equal(obs["observation/state"], np.asarray(reference, dtype=np.float32))

    def test_images_are_loaded_as_rgb_without_changing_camera_mapping(self):
        from PIL import Image

        with TemporaryDirectory() as directory:
            base = Path(directory) / "base.png"
            wrist = Path(directory) / "wrist.png"
            Image.new("RGB", (12, 8), (255, 0, 0)).save(base)
            Image.new("RGB", (12, 8), (0, 255, 0)).save(wrist)
            args = smoke_args()
            args.smoke_test, args.base_image, args.wrist_image = False, base, wrist
            args.state = "0.3,-0.1,0.4,0,0,0,1,0.04"
            obs = runtime.build_observation(args, state_dim=8)
            np.testing.assert_array_equal(obs["observation/image"][0, 0], [255, 0, 0])
            np.testing.assert_array_equal(obs["observation/wrist_image"][0, 0], [0, 255, 0])

    def test_nonfinite_and_wrong_action_dimensions_are_rejected(self):
        for dimension in (7, 8, 12):
            runtime.validate_actions({"actions": np.zeros((2, dimension))}, action_dim=dimension)
        for actions in (np.zeros((2, 7)), np.zeros((0, 8)), np.full((2, 8), np.inf)):
            with self.assertRaises(ValueError):
                runtime.validate_actions({"actions": actions}, action_dim=8)

    def test_websocket_client_metadata_and_error_contract(self):
        from openpi_client import msgpack_numpy
        from websockets.sync.server import serve

        for backend in ("jax", "pytorch", "error"):
            def handler(connection):
                connection.send(msgpack_numpy.packb({
                    "backend": "jax" if backend == "error" else backend,
                    "state_dim": 8, "action_dim": 8,
                }))
                if backend == "pytorch":
                    return
                received = msgpack_numpy.unpackb(connection.recv())
                if backend == "error":
                    connection.send("deliberate model failure")
                else:
                    connection.send(msgpack_numpy.packb({"actions": np.repeat(received["observation/state"][None], 3, axis=0)}))

            with serve(handler, "127.0.0.1", 0) as server:
                thread = threading.Thread(target=server.serve_forever)
                thread.start()
                try:
                    args = smoke_args(policy_host="127.0.0.1", policy_port=server.socket.getsockname()[1], timeout=5)
                    if backend == "jax":
                        result = runtime.infer_remote(args)
                        np.testing.assert_array_equal(result["actions"], np.zeros((3, 8)))
                    else:
                        with self.assertRaises((ValueError, RuntimeError)):
                            runtime.infer_remote(args)
                finally:
                    server.shutdown()
                    thread.join(timeout=5)


@unittest.skipUnless(os.environ.get("PI05_TEST_JAX") == "1", "Set PI05_TEST_JAX=1 in an OpenPI runtime")
class JaxIntegrationTests(unittest.TestCase):
    def test_orbax_restore_transforms_and_jit_even_with_pytorch_file_present(self):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        import orbax.checkpoint as ocp
        from openpi import transforms
        from openpi.models.model import ModelType
        from openpi.policies.tb6r5_policy import TB6R5Inputs, TB6R5Outputs

        class TinyModel(nnx.Module):
            def __init__(self, params):
                self.gain = nnx.Param(params["gain"])

            def sample_actions(self, rng, observation, num_steps=10):
                # Exercise the actual OpenPI module_jit and Policy.infer path.
                step = observation.state * self.gain.value / num_steps
                actions = jax.lax.fori_loop(
                    0, num_steps, lambda _, current: current + step, jnp.zeros_like(observation.state)
                )
                return jnp.repeat(actions[:, None, :], 3, axis=1)

        data = SimpleNamespace(
            asset_id="new_dataset", use_quantile_norm=False,
            data_transforms=transforms.Group(
                inputs=[
                    lambda data: {**data, "observation/state": data["raw_pose"]},
                    TB6R5Inputs(ModelType.PI05),
                ],
                outputs=[TB6R5Outputs(action_dim=8)],
            ),
            model_transforms=transforms.Group(inputs=[lambda data: {k: v for k, v in data.items() if k != "prompt"}]),
        )
        config = SimpleNamespace(
            data=SimpleNamespace(create=lambda *_: data), assets_dirs=(),
            model=SimpleNamespace(load=TinyModel), policy_metadata={"test_model": True},
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_stats(root, 8, "new_dataset")
            with ocp.PyTreeCheckpointer() as checkpointer:
                checkpointer.save(root / "params", {"params": {"gain": jnp.full((8,), 2.0)}})
            # This invalid PyTorch file must never be opened by the JAX loader.
            (root / "model.safetensors").write_text("not a PyTorch model")
            runtime.validate_checkpoint(root)
            policy = runtime.create_jax_policy(config, root, num_steps=3)
            args = smoke_args()
            args.state = "0.1,0.2,0.3,0,0,0,1,0.04"
            observation = runtime.build_observation(args, policy.state_dim)
            observation["raw_pose"] = observation.pop("observation/state")
            for _ in range(2):
                result = policy.infer(observation)
                expected = np.repeat((observation["raw_pose"] * 2)[None], 3, axis=0)
                np.testing.assert_allclose(result["actions"], expected, rtol=1e-4, atol=1e-5)
                self.assertGreater(result["jax_timing"]["wall_ms"], 0)
            devices = policy.policy._model.gain.value.devices()
            self.assertEqual({device.platform for device in devices}, {"cpu"})
            self.assertEqual(policy.policy._model.gain.value.dtype, jnp.bfloat16)


if __name__ == "__main__":
    unittest.main()
