import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.mage_flow.model.mage_flow_vae.vae import MageVAE, _DConvDenoiser, _DConvEncoder
from mflux.models.mage_flow.variants.edit.util import MageFlowEditUtil


def test_mage_flow_edit_target_size_uses_primary_aspect_and_multiples_of_16() -> None:
    primary = Image.new("RGB", (1200, 800))

    assert MageFlowEditUtil.resolve_target_size(primary, width=1024, height=512) == (1024, 512)
    assert MageFlowEditUtil.resolve_target_size(primary, width=None, height=None, max_size=1024) == (1024, 672)
    assert MageFlowEditUtil.resolve_target_size(primary, width=None, height=None) == (1200, 800)


def test_mage_flow_edit_target_size_matches_official_minimum_floor() -> None:
    tiny = Image.new("RGB", (8, 4))
    extreme = Image.new("RGB", (1000, 1))

    assert MageFlowEditUtil.resolve_target_size(tiny, width=8, height=8) == (16, 16)
    assert MageFlowEditUtil.resolve_target_size(tiny, width=None, height=None, max_size=8) == (16, 16)
    assert MageFlowEditUtil.resolve_target_size(extreme, width=None, height=None, max_size=512) == (512, 16)


def test_mage_flow_edit_vl_copy_caps_long_edge_with_bicubic_aspect() -> None:
    image = Image.new("RGB", (1200, 800))
    resized = MageFlowEditUtil.resize_long_edge(image)

    assert resized.size == (384, 256)
    assert MageFlowEditUtil.resize_long_edge(Image.new("RGB", (320, 240))).size == (320, 240)


def test_mage_flow_edit_prepares_vae_batch_in_nchw_minus_one_to_one() -> None:
    black = Image.new("RGB", (10, 20), color=(0, 0, 0))
    white = Image.new("RGB", (20, 10), color=(255, 255, 255))
    batch = MageFlowEditUtil.prepare_vae_images([black, white], width=32, height=16, dtype=mx.float32)

    assert batch.shape == (2, 3, 16, 32)
    np.testing.assert_allclose(np.asarray(batch[0]), -1)
    np.testing.assert_allclose(np.asarray(batch[1]), 1)


def test_mage_flow_edit_encodes_multiple_references_as_one_clean_sequence() -> None:
    encoder = _DConvEncoder(
        z_ch=4,
        hidden_size=32,
        num_blocks=1,
        patch_size=4,
        mlp_ratio=2.0,
        head_size=32,
        num_head_blocks=1,
    )
    decoder = _DConvDenoiser(
        patch_size=4,
        hidden_size=32,
        hidden_size_x=8,
        mlp_ratio=2.0,
        num_blocks=1,
        num_cond_blocks=1,
        bottleneck_dim=4,
        attention_patch_size=2,
    )
    vae = MageVAE(sample_posterior=False, encoder=encoder, decoder_model=decoder)
    images = [Image.new("RGB", (8, 8), color="red"), Image.new("RGB", (8, 8), color="blue")]

    packed = MageFlowEditUtil.encode_references(vae, images, width=8, height=8, seed=3)
    mx.eval(packed)

    assert packed.shape == (1, 8, 4)
    assert bool(mx.all(mx.isfinite(packed)).item())
