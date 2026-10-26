from collections.abc import Sequence
import logging
import os
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._action_normalizer = next(
            (transform for transform in transforms if isinstance(transform, _transforms.Normalize)), None
        )

        self._sample_actions_rtc = None
        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            if callable(getattr(self._model, "sample_actions_rtc", None)):
                self._sample_actions_rtc = self._model.sample_actions_rtc
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            if callable(getattr(model, "sample_actions_rtc", None)):
                # Mask sizes and the guidance cap must stay concrete inside the trace.
                self._sample_actions_rtc = nnx_utils.module_jit(
                    model.sample_actions_rtc,
                    static_argnames=("executed_steps", "delay_steps", "num_steps", "guidance_beta"),
                )
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        rtc_request = inputs.pop("rtc", None)
        if rtc_request is not None and self._sample_actions_rtc is None:
            raise ValueError("RTC requests require a policy whose model implements sample_actions_rtc")
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if rtc_request is not None:
            if not isinstance(rtc_request, dict):
                raise ValueError("rtc must be a dictionary")
            sample_kwargs["num_steps"] = int(rtc_request.get("num_steps", 5))
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        sampler = self._sample_actions
        if rtc_request is not None and "previous_actions" in rtc_request:
            if self._action_normalizer is None:
                raise ValueError("RTC requires checkpoint action normalization statistics")
            executed = int(rtc_request["executed_steps"])
            delay = int(rtc_request["delay_steps"])
            previous = np.asarray(rtc_request["previous_actions"], dtype=np.float32)
            model_config = getattr(self._model, "config", None)
            if model_config is not None and hasattr(model_config, "action_horizon"):
                horizon = int(model_config.action_horizon)
                model_action_dim = int(model_config.action_dim)
            else:
                # JAX Pi0 stores the sizes on the module; it has no PyTorch config object.
                horizon = int(self._model.action_horizon)
                model_action_dim = int(self._model.action_dim)
            actual_dim = len(self._action_normalizer.norm_stats["actions"].mean)
            if previous.shape != (horizon - executed, actual_dim) or not np.isfinite(previous).all():
                raise ValueError("RTC previous_actions has invalid shape or values")
            normalized = self._action_normalizer({"actions": previous})["actions"]
            padded = np.zeros((horizon - executed, model_action_dim), dtype=np.float32)
            padded[:, :actual_dim] = normalized
            if self._is_pytorch_model:
                previous_batch = torch.from_numpy(padded)[None].to(self._pytorch_device)
            else:
                previous_batch = jnp.asarray(padded)[None, ...]
            sample_kwargs.update(
                previous_actions=previous_batch,
                executed_steps=executed,
                delay_steps=delay,
                guidance_beta=float(rtc_request.get("guidance_beta", 5.0)),
            )
            sampler = self._sample_actions_rtc
        outputs = {
            "state": inputs["state"],
            "actions": sampler(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        supported = self._sample_actions_rtc is not None
        metadata = {**self._metadata, "rtc_supported": supported}
        # Same hardware gate for JAX and PyTorch.  jax_infer.py may override this
        # with --allow-hardware after copying the metadata.
        if supported and os.environ.get("OPENPI_RTC_HARDWARE_ALLOWED") != "1":
            metadata["hardware_allowed"] = False
        return metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
