"""Inference-time guidance for Real-Time Chunking (RTC).

The controller supplies the unexecuted tail of its current chunk.  These
helpers implement the soft mask and the flow-matching guidance term from
Black et al., *Real-Time Execution of Action Chunking Flow Policies*.
The scalar formulas live in ``openpi.models.rtc_guidance`` and are shared
with the JAX sampler.
"""

import torch

from openpi.models.rtc_guidance import guidance_scale_value
from openpi.models.rtc_guidance import soft_mask_weights


def soft_mask(horizon: int, executed_steps: int, delay_steps: int, *, device) -> torch.Tensor:
    return torch.tensor(soft_mask_weights(horizon, executed_steps, delay_steps), dtype=torch.float32, device=device)


def guidance_scale(time: float, beta: float = 5.0) -> float:
    return guidance_scale_value(time, beta)


def guided_euler_step(x_t, velocity_fn, time: float, num_steps: int, target, weights, beta: float = 5.0):
    """One guided flow step, differentiating only through the current noise variable."""
    with torch.enable_grad():
        x_in = x_t.detach().requires_grad_(True)
        velocity = velocity_fn(x_in)
        projected = x_in - time * velocity
        error = (target - projected).detach() * weights
        correction = torch.autograd.grad(projected, x_in, grad_outputs=error)[0]
    return (x_in + (-velocity + guidance_scale(time, beta) * correction) / num_steps).detach()
