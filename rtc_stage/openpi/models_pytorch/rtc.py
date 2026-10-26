"""Inference-time guidance for Real-Time Chunking (RTC).

The controller supplies the unexecuted tail of its current chunk.  These
helpers implement the soft mask and the flow-matching guidance term from
Black et al., *Real-Time Execution of Action Chunking Flow Policies*.
"""

import math

import torch


def soft_mask(horizon: int, executed_steps: int, delay_steps: int, *, device) -> torch.Tensor:
    if not 0 < executed_steps < horizon:
        raise ValueError("RTC executed_steps must be between 1 and horizon - 1")
    overlap = horizon - executed_steps
    if not 0 <= delay_steps <= overlap:
        raise ValueError("RTC delay_steps exceeds the remaining action overlap")
    weights = torch.zeros(horizon, dtype=torch.float32, device=device)
    weights[:delay_steps] = 1.0
    for i in range(delay_steps, overlap):
        c = (overlap - i) / (overlap - delay_steps + 1)
        weights[i] = math.expm1(c) / math.expm1(1.0)
    return weights


def guidance_scale(time: float, beta: float = 5.0) -> float:
    if not 0.0 <= time <= 1.0 or beta <= 0:
        raise ValueError("Invalid RTC time or guidance limit")
    if time == 1.0 or time == 0.0:
        return beta
    # The paper's flow time runs noise -> data; OpenPI runs data -> noise.
    paper_time = 1.0 - time
    residual_variance = time**2 / (paper_time**2 + time**2)
    return min(beta, time / (paper_time * residual_variance))


def guided_euler_step(x_t, velocity_fn, time: float, num_steps: int, target, weights, beta: float = 5.0):
    """One guided flow step, differentiating only through the current noise variable."""
    with torch.enable_grad():
        x_in = x_t.detach().requires_grad_(True)
        velocity = velocity_fn(x_in)
        projected = x_in - time * velocity
        error = (target - projected).detach() * weights
        correction = torch.autograd.grad(projected, x_in, grad_outputs=error)[0]
    return (x_in + (-velocity + guidance_scale(time, beta) * correction) / num_steps).detach()
