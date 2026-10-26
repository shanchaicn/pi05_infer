import os
import pathlib
import sys
import time

import numpy as np


def _configure_xla_cuda() -> None:
    """Prefer the ptxas bundled with the current Python env when available."""
    site_packages = pathlib.Path(__file__).resolve().parents[1] / ".venv"
    candidates = [
        pathlib.Path(os.environ.get("CONDA_PREFIX", ""))
        / "lib/python3.12/site-packages/triton/backends/nvidia",
        pathlib.Path(sys.prefix) / "lib/python3.12/site-packages/triton/backends/nvidia",
        site_packages / "lib/python3.12/site-packages/triton/backends/nvidia",
    ]
    for cuda_data_dir in candidates:
        if (cuda_data_dir / "bin/ptxas").exists():
            os.environ.setdefault("XLA_FLAGS", f"--xla_gpu_cuda_data_dir={cuda_data_dir}")
            os.environ["PATH"] = f"{cuda_data_dir / 'bin'}:{os.environ.get('PATH', '')}"
            return


def _random_droid_observation() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "pick up the object",
    }


def main() -> None:
    _configure_xla_cuda()

    import jax

    from openpi.policies import policy_config
    from openpi.training import config as _config

    config = _config.get_config("pi05_droid")
    checkpoint_dir = "gs://openpi-assets/checkpoints/pi05_droid"

    print(f"JAX devices: {jax.devices()}")
    print(f"Loading {config.name} from {checkpoint_dir}")
    start = time.monotonic()
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    print(f"Loaded policy in {time.monotonic() - start:.1f}s")

    obs = _random_droid_observation()
    start = time.monotonic()
    result = policy.infer(obs)
    elapsed_ms = (time.monotonic() - start) * 1000

    actions = result["actions"]
    print(f"Action shape: {actions.shape}")
    print(f"Inference time: {elapsed_ms:.1f} ms")
    print("First action:", np.array2string(actions[0], precision=4, suppress_small=True))


if __name__ == "__main__":
    main()
