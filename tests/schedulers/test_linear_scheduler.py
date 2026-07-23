import copy
import io
import math

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.schedulers import try_import_external_scheduler
from mflux.models.common.schedulers.linear_scheduler import LinearScheduler
from mflux.models.common.schedulers.mage_flow_scheduler import MageFlowScheduler


@pytest.mark.fast
def test_linear_scheduler_import_by_name():
    assert (
        try_import_external_scheduler("mflux.models.common.schedulers.linear_scheduler.LinearScheduler")
        == LinearScheduler
    )


@pytest.fixture
def test_config():
    return Config(
        model_config=ModelConfig.dev(),  # requires_sigma_shift=True
        num_inference_steps=14,
        width=1024,
        height=1024,
        scheduler="linear",
    )


@pytest.mark.fast
def test_linear_scheduler_initialization(test_config):
    scheduler = LinearScheduler(config=test_config)
    assert scheduler.sigmas is not None
    assert isinstance(scheduler.sigmas, mx.array)
    assert len(scheduler.sigmas) > 0


@pytest.mark.fast
def test_linear_scheduler_sigmas_property_no_shift(test_config):
    scheduler = LinearScheduler(config=test_config)
    expected_sigmas_from_mflux_0_9_0 = mx.array(
        np.load(
            io.BytesIO(
                bytes.fromhex(
                    # see: https://gist.github.com/anthonywu/2832147ff5f5f50c81df4d13152d2bed
                    "934e554d505901003e007b276465736372273a20273c6634272c2027666f727472616e5f6f72646572273a20547275652c20277368617065273a202831352c20297d20202020200a0000803fb7e9793fd92a733f7aa66b3fab38633f30b4593f59df4e3f4f6f423f4b01343f1b10233fc3e30e3f5cedec3e0a90b03e4025483e00000000"
                )
            )
        )
    )

    assert mx.allclose(scheduler.sigmas, expected_sigmas_from_mflux_0_9_0)
    assert scheduler.sigmas.shape == (test_config.num_inference_steps + 1,)


@pytest.mark.fast
def test_linear_scheduler_sigmas_property_with_shift(test_config):
    test_config.model_config = ModelConfig.schnell()  # requires_sigma_shift=True
    scheduler = LinearScheduler(config=test_config)
    expected_sigmas_from_mflux_0_9_0 = mx.array(
        np.load(
            io.BytesIO(
                bytes.fromhex(
                    # see: https://gist.github.com/anthonywu/2832147ff5f5f50c81df4d13152d2bed
                    "934e554d505901003e007b276465736372273a20273c6634272c2027666f727472616e5f6f72646572273a20547275652c20277368617065273a202831352c20297d20202020200a0000803fdbb66d3fb76d5b3f9224493f6edb363f4992243f2549123f0000003fb76ddb3e6edbb63e2549923eb76d5b3e2549123e2549923d00000000"
                )
            )
        )
    )
    assert mx.allclose(scheduler.sigmas, expected_sigmas_from_mflux_0_9_0)
    assert scheduler.sigmas.shape == (test_config.num_inference_steps + 1,)


@pytest.mark.fast
def test_mage_flow_scheduler_matches_static_shift_six_and_fp32_euler():
    static_mu = math.log(6.0)
    model_config = copy.deepcopy(ModelConfig.dev())
    model_config.sigma_base_shift = static_mu
    model_config.sigma_max_shift = static_mu
    model_config.requires_sigma_shift = True
    config = Config(
        model_config=model_config,
        num_inference_steps=4,
        height=512,
        width=512,
        scheduler="mage_flow",
    )
    scheduler = config.scheduler

    assert isinstance(scheduler, MageFlowScheduler)
    np.testing.assert_allclose(
        np.asarray(scheduler.sigmas),
        np.array([1.0, 0.9473684, 0.8571429, 0.6666667, 0.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )

    latents = mx.array([1.0, -2.0], dtype=mx.bfloat16)
    noise = mx.array([0.125, -0.375], dtype=mx.bfloat16)
    stepped = scheduler.step(noise=noise, timestep=3, latents=latents)
    expected = (
        latents.astype(mx.float32) + (scheduler.sigmas[4] - scheduler.sigmas[3]) * noise.astype(mx.float32)
    ).astype(mx.bfloat16)
    mx.eval(stepped)

    assert stepped.dtype == mx.bfloat16
    np.testing.assert_array_equal(
        np.asarray(stepped.astype(mx.float32)),
        np.asarray(expected.astype(mx.float32)),
    )
