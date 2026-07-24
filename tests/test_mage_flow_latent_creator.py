import mlx.core as mx
import numpy as np

from mflux.models.mage_flow.latent_creator import MageFlowLatentCreator


def test_mage_flow_latent_pack_round_trip() -> None:
    source = mx.arange(1 * 128 * 2 * 3, dtype=mx.float32).reshape(1, 128, 2, 3)
    packed = MageFlowLatentCreator.pack_latents(source)
    restored = MageFlowLatentCreator.unpack_latents(packed, height=32, width=48)

    assert packed.shape == (1, 6, 128)
    np.testing.assert_array_equal(np.asarray(restored), np.asarray(source))


def test_mage_flow_gaussian_shading_is_deterministic_and_detectable() -> None:
    first = MageFlowLatentCreator.create_noise(
        seed=42,
        height=64,
        width=64,
        gaussian_shading=True,
        gaussian_shading_key=20260720,
    )
    second = MageFlowLatentCreator.create_noise(
        seed=42,
        height=64,
        width=64,
        gaussian_shading=True,
        gaussian_shading_key=20260720,
    )

    np.testing.assert_array_equal(
        np.asarray(first.astype(mx.float32)),
        np.asarray(second.astype(mx.float32)),
    )
    report = MageFlowLatentCreator.decode_gaussian_shading(first, key=20260720)
    assert report["present"] is True
    assert report["raw_acc"] == 1.0
    assert report["msg_acc"] == 1.0


def test_mage_flow_gaussian_shading_keeps_distinct_large_seeds() -> None:
    first = MageFlowLatentCreator.create_noise(
        seed=1,
        height=64,
        width=64,
        gaussian_shading=True,
        gaussian_shading_key=20260720,
        dtype=mx.float32,
    )
    second = MageFlowLatentCreator.create_noise(
        seed=2147483649,  # 2**31 + 1; formerly collapsed to seed 1 via 31-bit mask
        height=64,
        width=64,
        gaussian_shading=True,
        gaussian_shading_key=20260720,
        dtype=mx.float32,
    )

    assert not bool(mx.all(first == second).item())


def test_mage_flow_plain_noise_can_disable_watermark() -> None:
    noise = MageFlowLatentCreator.create_noise(
        seed=9,
        height=64,
        width=80,
        gaussian_shading=False,
    )

    assert noise.shape == (1, 20, 128)
    assert noise.dtype == mx.bfloat16
    assert bool(mx.all(mx.isfinite(noise)).item())
