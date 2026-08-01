import mlx.core as mx
import pytest

from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE


@pytest.mark.fast
def test_unpack_passes_through_already_unpacked_latents():
    # Ideogram 4's latent creator denorms + unpatchifies itself, so its 32-channel
    # spatial latents must come back unchanged: unpack_packed_latents promises the
    # plain VAE latent, and returning decoded pixels here makes decode_packed_latents
    # decode twice and crash the stepwise preview.
    vae = Flux2VAE()
    latents = mx.random.normal((1, 32, 16, 16))
    out = vae.unpack_packed_latents(latents)
    assert out.shape == latents.shape
    assert mx.array_equal(out, latents)


@pytest.mark.fast
def test_decode_packed_latents_decodes_unpacked_latents_once():
    vae = Flux2VAE()
    latents = mx.random.normal((1, 32, 16, 16))
    decoded = vae.decode_packed_latents(latents)
    assert decoded.shape[1] == 3
    assert decoded.shape[2] == 16 * 8 and decoded.shape[3] == 16 * 8
