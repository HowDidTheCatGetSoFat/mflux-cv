import math

import mlx.core as mx


def base_shaped_sigmas(
    num_steps: int,
    schedule: str,
    sigma_min: float,
    sigma_max: float = 1.0,
) -> mx.array | None:
    """Base sigmas for the non-linear schedules, shared by the schedulers so their formulas cannot drift.

    Returns an mx.array for "cosine" / "karras" / "exponential", or None for "linear" (and any unknown
    value) so each caller applies its own linear ramp. Parameterized by sigma_min because the schedulers
    use different conventions (LinearScheduler: 1/num_steps; FlowMatchEulerDiscreteScheduler: 1/1000).
    """
    if schedule == "cosine":
        t = mx.linspace(0, 1, num_steps)
        return ((1.0 + mx.cos(t * math.pi)) / 2.0).astype(mx.float32)
    if schedule == "karras":
        rho = 7.0
        ramp = mx.linspace(0, 1, num_steps)
        min_inv_rho = sigma_min ** (1.0 / rho)
        max_inv_rho = sigma_max ** (1.0 / rho)
        return ((max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho).astype(mx.float32)
    if schedule == "exponential":
        return mx.exp(mx.linspace(math.log(sigma_max), math.log(sigma_min), num_steps)).astype(mx.float32)
    return None
