#!/usr/bin/env python3
"""Standalone OpenPI JAX inference using the checkpoint's training data configuration."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np


def _install_import_paths() -> None:
    """Put the in-repo OpenPI tree and the websocket client on ``sys.path``.

    ``rtc_stage`` is the tree that implements Real-Time Chunking for both JAX
    and PyTorch.  ``OPENPI_ROOT`` still overrides it (``src/openpi`` or a
    directory that itself contains ``openpi``).  ``app/`` remains an optional
    full checkout for ``openpi_client``.
    """
    root = Path(__file__).resolve().parent
    explicit = os.environ.get("OPENPI_ROOT")
    if explicit:
        base = Path(explicit)
        entries = [base / "src", base, base / "packages/openpi-client/src"]
    else:
        entries = [
            root / "docker_source" / "src",
            root / "app" / "src",
            root / "docker_source" / "packages/openpi-client/src",
            root / "app" / "packages/openpi-client/src",
            root / "rtc_stage",
        ]
    for source in entries:
        if not source.is_dir():
            continue
        path = str(source)
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


_install_import_paths()


def validate_checkpoint(checkpoint: Path, asset_id: str | None = None) -> dict:
    """Check local JAX checkpoint layout before importing or allocating a model."""
    params = checkpoint / "params"
    if not params.is_dir() or not any(params.iterdir()):
        raise ValueError(
            f"Missing/non-populated JAX params directory: {params}. "
            "Use an OpenPI JAX training checkpoint; model.safetensors cannot be loaded as JAX params."
        )
    if asset_id is not None:
        stats_path = checkpoint / "assets" / asset_id / "norm_stats.json"
    else:
        candidates = list((checkpoint / "assets").rglob("norm_stats.json"))
        if len(candidates) != 1:
            raise ValueError("Expected one assets/**/norm_stats.json; select the training asset with --asset-id")
        stats_path = candidates[0]
    if not stats_path.is_file():
        raise ValueError(f"Missing normalization statistics: {stats_path}")
    stats = json.loads(stats_path.read_text())["norm_stats"]
    for key in ("state", "actions"):
        dimension = None
        for field in ("mean", "std", "q01", "q99"):
            value = stats[key].get(field)
            if value is None and field in {"q01", "q99"}:
                continue
            values = np.asarray(value, dtype=np.float32)
            if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{stats_path}: {key}.{field} must be a non-empty finite vector"
                )
            if dimension is not None and values.size != dimension:
                raise ValueError(f"{stats_path}: inconsistent {key} statistics dimensions")
            dimension = values.size
        if np.any(np.asarray(stats[key]["std"]) < 0):
            raise ValueError(f"{stats_path}: {key}.std contains negative values")
    return stats


def initialize_jax(device: str):
    # Must happen before importing OpenPI, which imports JAX transitively.
    os.environ["JAX_PLATFORMS"] = "cuda" if device == "gpu" else device
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    try:
        import jax

        platform = "cuda" if device == "gpu" else device
        devices = jax.devices(platform)
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            f"Cannot initialize JAX on {device}: {exc}. "
            "The existing Orin PyTorch image has CPU-only JAX. For GPU inference use a "
            "JAX CUDA runtime compatible with your JetPack/CUDA; use --device cpu for CPU validation."
        ) from exc
    if not devices:
        raise RuntimeError(f"No JAX {device} device available")
    logging.info("JAX %s devices: %s", jax.__version__, devices)
    return jax


def validate_observation(observation: dict, state_dim: int | None = None) -> None:
    # Custom raw observation keys are handled by the selected training transforms.
    if "observation/state" in observation:
        state = np.asarray(observation["observation/state"])
        if (state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state))
                or (state_dim is not None and state.size != state_dim)):
            raise ValueError(f"Expected finite observation/state vector of dimension {state_dim}, got {state.shape}")
    for key in ("observation/image", "observation/wrist_image"):
        if key not in observation:
            continue
        image = np.asarray(observation[key])
        if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1 or image.dtype != np.uint8:
            raise ValueError(f"{key} must be an RGB uint8 HWC image")
    if "prompt" in observation and (not isinstance(observation["prompt"], str) or not observation["prompt"].strip()):
        raise ValueError("A non-empty prompt is required")


def validate_actions(result: dict, action_dim: int | None = None) -> np.ndarray:
    actions = np.asarray(result["actions"])
    if (actions.ndim != 2 or min(actions.shape) == 0 or not np.all(np.isfinite(actions))
            or (action_dim is not None and actions.shape[1] != action_dim)):
        raise ValueError(
            f"Expected finite action chunk (T, {action_dim or 'D'}), got {actions.shape}. "
            "Check the training config's output transforms against the checkpoint statistics."
        )
    return actions


class InferencePolicy:
    """Check dimensions from checkpoint statistics without assuming coordinate semantics."""

    def __init__(self, policy, stats: dict):
        self.policy = policy
        self.state_dim = len(stats["state"]["mean"])
        self.action_dim = len(stats["actions"]["mean"])
        self.smoke_state = stats["state"]["mean"]

    def infer(self, observation: dict) -> dict:
        # Raw input layout belongs to the training transforms. Validate state after them.
        start = time.monotonic()
        result = self.policy.infer(observation)
        validate_actions(result, self.action_dim)
        # Policy.infer returns NumPy arrays, so this includes device synchronization.
        result["jax_timing"] = {"wall_ms": (time.monotonic() - start) * 1000}
        return result


def create_jax_policy(train_config, checkpoint: Path, num_steps: int, asset_id: str | None = None):
    """Use OpenPI's JAX loader and transform pipeline explicitly, without auto-detection."""
    import jax.numpy as jnp
    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.policies.policy import Policy
    from openpi.training import checkpoints

    data = train_config.data.create(train_config.assets_dirs, train_config.model)
    asset_id = asset_id if asset_id is not None else data.asset_id
    if asset_id is None:
        raise ValueError("Training config has no asset_id; select checkpoint statistics with --asset-id")
    raw_stats = validate_checkpoint(checkpoint, asset_id)
    stats = checkpoints.load_norm_stats(checkpoint / "assets", asset_id)
    # Loading only params/ prevents model.safetensors from selecting PyTorch.
    model = train_config.model.load(model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    class ValidateState:
        def __call__(self, inputs):
            state = np.asarray(inputs["state"])
            expected = len(raw_stats["state"]["mean"])
            if state.shape != (expected,) or not np.all(np.isfinite(state)):
                raise ValueError(f"Training input transforms must produce state ({expected},), got {state.shape}")
            return inputs

    policy = Policy(
        model,
        transforms=[
            *data.data_transforms.inputs,
            ValidateState(),
            transforms.Normalize(stats, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ],
        output_transforms=[
            *data.model_transforms.outputs,
            transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm),
            *data.data_transforms.outputs,
        ],
        sample_kwargs={"num_steps": num_steps},
        metadata=train_config.policy_metadata,
        is_pytorch=False,
    )
    return InferencePolicy(policy, raw_stats)


def load_policy(args):
    # Check the params layout before initializing devices; the training config selects assets below.
    if not (args.checkpoint / "params").is_dir():
        raise ValueError(f"Missing JAX params directory: {args.checkpoint / 'params'}")
    jax = initialize_jax(args.device)
    from openpi.training import config

    train_config = config.get_config(args.config)
    start = time.monotonic()
    with jax.default_device(jax.devices("cuda" if args.device == "gpu" else args.device)[0]):
        policy = create_jax_policy(train_config, args.checkpoint, args.num_steps, args.asset_id)
    logging.info("Loaded JAX policy in %.1fs; first inference includes JIT compilation", time.monotonic() - start)
    return policy


def build_observation(args, state_dim: int | None = None, smoke_state=None) -> dict:
    from PIL import Image

    if args.observation:
        if args.state or args.base_image or args.wrist_image or args.smoke_test:
            raise ValueError("--observation cannot be combined with state/image/smoke-test options")
        with np.load(args.observation, allow_pickle=False) as payload:
            observation = {key: payload[key] for key in payload.files}
        if "prompt" in observation and observation["prompt"].ndim == 0:
            observation["prompt"] = str(observation["prompt"].item())
        if args.task is not None:
            observation["prompt"] = args.task
        return observation

    if not args.task:
        raise ValueError("Provide --task (or supply prompt in --observation)")
    if args.smoke_test:
        if args.base_image or args.wrist_image:
            raise ValueError("--smoke-test cannot be combined with real image files")
        base = wrist = np.zeros((224, 224, 3), dtype=np.uint8)
        logging.warning("Black-frame smoke test: output is only for pipeline validation")
    else:
        if not args.base_image or not args.wrist_image or not args.state:
            raise ValueError("Provide --state, --base-image and --wrist-image, or use --smoke-test")
        with Image.open(args.base_image) as image:
            base = np.asarray(image.convert("RGB"))
        with Image.open(args.wrist_image) as image:
            wrist = np.asarray(image.convert("RGB"))
    if args.state:
        state = np.asarray([float(value) for value in args.state.split(",")], dtype=np.float32)
    elif smoke_state is not None:
        state = np.asarray(smoke_state, dtype=np.float32)
    elif state_dim is not None:
        state = np.zeros(state_dim, dtype=np.float32)
    else:
        raise ValueError("State dimension is unknown; provide --state or a checkpoint/server with state metadata")
    observation = {
        "observation/state": state,
        "observation/image": base,
        "observation/wrist_image": wrist,
        "prompt": args.task,
    }
    validate_observation(observation, state_dim)
    return observation


def infer_remote(args, observation: dict | None = None) -> dict:
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect

    with connect(
        f"ws://{args.policy_host}:{args.policy_port}",
        open_timeout=args.timeout,
        compression=None,
        max_size=None,
        ping_interval=None,
    ) as connection:
        metadata = msgpack_numpy.unpackb(connection.recv(timeout=args.timeout))
        logging.info("Server metadata: %s", metadata)
        if metadata.get("backend") != "jax":
            raise ValueError("Server does not declare backend=jax; connect to jax_infer.py serve")
        state_dim, action_dim = metadata.get("state_dim"), metadata.get("action_dim")
        if observation is None:
            observation = build_observation(args, state_dim, metadata.get("smoke_state"))
        connection.send(msgpack_numpy.packb(observation))
        response = connection.recv(timeout=args.timeout)
        if isinstance(response, str):
            raise RuntimeError(f"Policy service error:\n{response}")
        result = msgpack_numpy.unpackb(response)
    validate_actions(result, action_dim)
    return result


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "infer", "serve", "client"):
        command = commands.add_parser(name)
        if name != "client":
            command.add_argument("--checkpoint", type=Path, required=True, help="JAX checkpoint containing params/ and assets/")
            command.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
            command.add_argument("--asset-id", help="Normalization asset under checkpoint/assets; defaults to training config")
            if name != "check":
                command.add_argument("--config", required=True, help="Exact training config for the JAX checkpoint")
                command.add_argument("--num-steps", type=positive_int, default=10)
        if name in {"infer", "client"}:
            command.add_argument("--task", help="Language prompt; overrides prompt from --observation")
            command.add_argument("--observation", type=Path, help="NPZ with raw observation keys/arrays expected by training transforms")
            command.add_argument("--state", help="Comma-separated state in training order and units; dimension comes from checkpoint")
            command.add_argument("--base-image", type=Path, help="realsense_0 RGB image file")
            command.add_argument("--wrist-image", type=Path, help="realsense_1 RGB image file")
            command.add_argument("--smoke-test", action="store_true")
            command.add_argument("--output", type=Path, help="Save returned action chunk as an .npz file")
        if name == "serve":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8003)
            command.add_argument("--allow-hardware", action="store_true", help="Permit explicitly confirmed TB6-R5 hardware runs")
        if name == "client":
            command.add_argument("--policy-host", default="127.0.0.1")
            command.add_argument("--policy-port", type=int, default=8003)
            command.add_argument("--timeout", type=float, default=300.0, help="Connect/receive timeout including first JIT (seconds)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.command == "check":
            stats = validate_checkpoint(args.checkpoint, args.asset_id)
            initialize_jax(args.device)
            print(f"state_dim={len(stats['state']['mean'])}, action_dim={len(stats['actions']['mean'])}")
            print("JAX device and checkpoint layout OK (parameter compatibility is checked when loading)")
            return 0
        if args.command == "serve":
            policy = load_policy(args)
            from openpi.serving.websocket_policy_server import WebsocketPolicyServer

            metadata = {
                **policy.policy.metadata,
                "backend": "jax", "device": args.device, "config": args.config,
                "state_dim": policy.state_dim, "action_dim": policy.action_dim,
                "smoke_state": list(policy.smoke_state), "hardware_allowed": args.allow_hardware,
            }
            WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()
            return 0
        start = time.monotonic()
        if args.command == "client":
            result = infer_remote(args)
        else:
            policy = load_policy(args)
            observation = build_observation(args, policy.state_dim, policy.smoke_state)
            result = policy.infer(observation)
        actions = validate_actions(result)
        print(f"actions shape={actions.shape}; total elapsed={time.monotonic() - start:.3f}s")
        print(np.array2string(actions, precision=6))
        if args.output:
            with args.output.open("wb") as output:
                np.savez_compressed(output, actions=actions)
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TimeoutError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
