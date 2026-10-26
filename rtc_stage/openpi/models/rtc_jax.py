"""JAX flow step for Real-Time Chunking.

Mirrors ``openpi.models_pytorch.rtc.guided_euler_step``: one Euler step from
noise toward data, plus a VJP correction that pulls the clean-action estimate
toward the prefix still committed to the controller.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from openpi.models.rtc_guidance import guidance_scale_value


def guided_euler_step(x_t, velocity_fn, time: float, num_steps: int, target, weights, beta: float = 5.0):
    """One guided step. ``weights`` broadcasts with ``target`` over (batch, time, dim)."""

    def project(x):
        velocity = velocity_fn(x)
        return x - time * velocity, velocity

    (projected, velocity), pullback = jax.vjp(project, x_t)
    error = jax.lax.stop_gradient(target - projected) * weights
    (correction,) = pullback((error, jnp.zeros_like(velocity)))
    scale = guidance_scale_value(float(time), float(beta))
    return x_t + (-velocity + scale * correction) / num_steps
