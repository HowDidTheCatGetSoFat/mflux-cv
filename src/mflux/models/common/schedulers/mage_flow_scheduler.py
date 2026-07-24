import math
from functools import partial
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from mflux.models.common.config.config import Config

from mflux.models.common.schedulers.base_scheduler import BaseScheduler


@partial(mx.compile, shapeless=True)
def _mage_flow_euler_step(
    noise: mx.array,
    latents: mx.array,
    sigma: mx.array,
    sigma_next: mx.array,
) -> mx.array:
    """Match Diffusers' FP32 Euler update before restoring the latent dtype."""

    updated = latents.astype(mx.float32)
    updated = updated + (sigma_next - sigma).astype(mx.float32) * noise.astype(mx.float32)
    return updated.astype(latents.dtype)


class MageFlowScheduler(BaseScheduler):
    """Static-shift FlowMatch Euler schedule released with Mage Flow."""

    def __init__(self, config: "Config"):
        self.config = config
        self.num_train_timesteps = 1000
        self._sigmas = self._get_sigmas()
        self._timesteps = self._sigmas[:-1] * self.num_train_timesteps

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    def _get_sigmas(self) -> mx.array:
        num_steps = self.config.num_inference_steps
        base_sigmas = mx.linspace(1.0, 1.0 / num_steps, num_steps, dtype=mx.float32)
        model_config = self.config.model_config
        if not math.isclose(model_config.sigma_base_shift, model_config.sigma_max_shift):
            raise ValueError("Mage Flow requires one resolution-independent static sigma shift")

        # MFLUX stores the exponential time-shift parameter (mu), whereas the
        # Microsoft/Diffusers config stores the resulting multiplicative shift.
        shift = math.exp(model_config.sigma_base_shift)
        shifted = shift * base_sigmas / (1.0 + (shift - 1.0) * base_sigmas)
        return mx.concatenate([shifted, mx.zeros((1,), dtype=mx.float32)])

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        sigmas = kwargs.get("sigmas", self._sigmas)
        return _mage_flow_euler_step(noise, latents, sigmas[timestep], sigmas[timestep + 1])
