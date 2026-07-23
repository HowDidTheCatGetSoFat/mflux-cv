import math

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mflux.models.mage_flow.model.mage_flow_vae.vae import (
    AttnBlock,
    MageVAE,
    _DConvDenoiser,
    _DConvEncoder,
)


def _tiny_vae(sample_posterior: bool) -> MageVAE:
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
        num_blocks=2,
        num_cond_blocks=1,
        bottleneck_dim=4,
        attention_patch_size=2,
    )
    return MageVAE(
        sample_posterior=sample_posterior,
        encoder=encoder,
        decoder_model=decoder,
    )


def test_mage_vae_mean_encode_and_decode_shapes():
    vae = _tiny_vae(sample_posterior=False)
    image = mx.zeros((1, 3, 8, 12), dtype=mx.float32)

    latent = vae.encode(image)
    decoded = vae.decode(latent)
    mx.eval(latent, decoded)

    assert latent.shape == (1, 4, 2, 3)
    assert decoded.shape == image.shape


def test_mage_vae_sampling_requires_an_explicit_key():
    vae = _tiny_vae(sample_posterior=True)
    image = mx.zeros((1, 3, 8, 8), dtype=mx.float32)

    with pytest.raises(ValueError, match="explicit MLX random key"):
        vae.encode(image)

    latent = vae.encode(image, key=mx.random.key(7))
    mx.eval(latent)
    assert latent.shape == (1, 4, 2, 2)


def test_mage_vae_preserves_converted_checkpoint_paths():
    vae = _tiny_vae(sample_posterior=False)
    parameter_paths = {name for name, _ in tree_flatten(vae.parameters())}

    assert "encoder.blocks.0.adaLN_modulation.1.weight" in parameter_paths
    assert "encoder.head_blocks.0.ca.1.weight" in parameter_paths
    assert "decoder_model.y_embedder.decoder.block.1.q.weight" in parameter_paths
    assert "decoder_model.dec_net.res_blocks.0.mlp.2.weight" in parameter_paths
    assert not any(".layers." in name for name in parameter_paths)


def test_mage_vae_default_model_matches_live_checkpoint_inventory():
    parameters = dict(tree_flatten(MageVAE().parameters()))

    assert len(parameters) == 728
    assert sum(math.prod(parameter.shape) for parameter in parameters.values()) == 138_052_035
    assert parameters["encoder.patch_cond_embed.weight"].shape == (768, 16, 16, 3)
    assert parameters["decoder_model.y_embedder.decoder.conv_in.weight"].shape == (384, 3, 3, 128)


def test_mage_vae_fixed_timestep_cache_only_folds_dico_blocks():
    vae = _tiny_vae(sample_posterior=False)

    assert vae.freeze_adaln_cache() == 2
    assert vae.freeze_adaln_cache() == 0
    parameter_paths = {name for name, _ in tree_flatten(vae.parameters())}

    assert "encoder.blocks.0.adaLN_modulation.modulation" in parameter_paths
    assert "decoder_model.blocks.0.adaLN_modulation.modulation" in parameter_paths
    assert "decoder_model.dec_net.res_blocks.0.adaLN_modulation.1.weight" in parameter_paths


def test_mage_vae_patch_attention_preserves_unaligned_spatial_shape():
    attention = AttnBlock(in_channels=32, patch_size=2)
    hidden_states = mx.zeros((1, 3, 5, 32), dtype=mx.float32)

    output = attention(hidden_states)
    mx.eval(output)

    assert output.shape == hidden_states.shape
