"""
Policy transforms for UR5e single-arm robot with Isaac Sim.

This module defines the input/output transforms for fine-tuning π0.5
on UR5e data collected from Isaac Sim (scripted or teleoperation).

The data pipeline is:
    LeRobot dataset -> repack_transforms -> data_transforms (this file) -> normalize -> model_transforms

For your own robot, copy this file and modify:
1. UR5eInputs: map your dataset keys to the model's expected format
2. UR5eOutputs: map model outputs back to your robot's action space
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_ur5e_example() -> dict:
    """Creates a random input example for the UR5e policy (used for inference)."""
    return {
        "observation/state": np.random.rand(7),   # 6 joints + gripper
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the block and place it on the other side",
    }


def _parse_image(image) -> np.ndarray:
    """Convert image to uint8 (H,W,C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR5eInputs(transforms.DataTransformFn):
    """
    Convert UR5e dataset inputs to the format expected by π0/π0.5.

    Key mapping:
        Dataset key                -> Model key
        ─────────────────────────────────────────────
        observation/state          -> state
        observation/image          -> image.base_0_rgb
        observation/left_wrist_0_rgb -> image.left_wrist_0_rgb
        observation/base_0_rgb     -> image.base_0_rgb  (alternative key)
        actions                    -> actions

    π0/π0.5 supports up to 3 image inputs:
        - base_0_rgb:      third-person camera (required)
        - left_wrist_0_rgb: left wrist camera (optional)
        - right_wrist_0_rgb: right wrist camera (optional, not used for single arm)

    If your dataset has different camera names, modify the key lookups below.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images from various possible keys
        # Try common key patterns from LeRobot datasets
        base_image = None
        wrist_image = None

        # Try different key patterns for third-person camera
        for key in ["observation/image", "observation/base_0_rgb", "observation/images.base_0_rgb"]:
            if key in data:
                base_image = _parse_image(data[key])
                break

        # Try different key patterns for wrist camera
        for key in ["observation/wrist_image", "observation/left_wrist_0_rgb", "observation/images.left_wrist_0_rgb"]:
            if key in data:
                wrist_image = _parse_image(data[key])
                break

        if base_image is None:
            raise ValueError(
                f"No base image found in data. Available keys: {list(data.keys())}. "
                "Expected one of: observation/image, observation/base_0_rgb, observation/images.base_0_rgb"
            )

        # Build model inputs
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image if wrist_image is not None else np.zeros_like(base_image),
                # Right wrist is not used for single arm — mask it out
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if wrist_image is not None else np.False_,
                # For π0: True masks padding; for π0-FAST: doesn't matter
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only available during training
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Prompt / language instruction
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5eOutputs(transforms.DataTransformFn):
    """
    Convert model outputs back to UR5e action space.

    Since π0.5 uses action_dim=32 (padded), we need to extract
    only the first 7 dimensions (6 joints + gripper) from the output.
    """

    # Number of action dimensions for UR5e single arm
    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
