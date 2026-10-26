"""Backend-neutral Real-Time Chunking scalars.

Black et al. soft-mask the overlap between the action chunk that is still
executing and the chunk being denoised, then scale a flow-matching guidance
term.  The JAX and PyTorch samplers both call these formulas so the mask and
the guidance weight stay the same number on either backend.
"""

from __future__ import annotations

import math


def soft_mask_weights(horizon: int, executed_steps: int, delay_steps: int) -> list[float]:
    """Per-timestep guidance weight. Index 0 is the next control tick."""
    if type(horizon) is not int or type(executed_steps) is not int or type(delay_steps) is not int:
        raise ValueError("RTC mask sizes must be Python integers")
    if horizon < 2:
        raise ValueError("RTC horizon must be at least 2")
    if not 0 < executed_steps < horizon:
        raise ValueError("RTC executed_steps must be between 1 and horizon - 1")
    overlap = horizon - executed_steps
    if not 0 <= delay_steps <= overlap:
        raise ValueError("RTC delay_steps exceeds the remaining action overlap")
    weights = [0.0] * horizon
    for index in range(delay_steps):
        weights[index] = 1.0
    denominator = overlap - delay_steps + 1
    for index in range(delay_steps, overlap):
        # expm1(c) / expm1(1) is the paper's soft step on the unfrozen overlap.
        coefficient = (overlap - index) / denominator
        weights[index] = math.expm1(coefficient) / math.expm1(1.0)
    return weights


def guidance_scale_value(time: float, beta: float = 5.0) -> float:
    """Inference-time guidance weight at one flow step.

    OpenPI integrates from noise (t=1) toward data (t=0), which is the reverse
    of the paper's clock.  ``beta`` caps the weight.
    """
    if not 0.0 <= time <= 1.0 or beta <= 0 or not math.isfinite(time) or not math.isfinite(beta):
        raise ValueError("Invalid RTC time or guidance limit")
    if time == 1.0 or time == 0.0:
        return float(beta)
    paper_time = 1.0 - time
    residual_variance = time**2 / (paper_time**2 + time**2)
    return float(min(beta, time / (paper_time * residual_variance)))
