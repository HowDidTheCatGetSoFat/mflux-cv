import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_vae.qwen_image_causal_conv_3d import QwenImageCausalConv3D
from mflux.models.qwen.model.qwen_vae.qwen_image_decoder_3d import QwenImageDecoder3D
from mflux.models.qwen.model.qwen_vae.qwen_image_encoder_3d import QwenImageEncoder3D


class QwenVAE(nn.Module):
    LATENTS_MEAN = mx.array([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]).reshape(1, 16, 1, 1, 1)  # fmt: off
    LATENTS_STD = mx.array([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916]).reshape(1, 16, 1, 1, 1)  # fmt: off
    spatial_scale = 8
    latent_channels = 16

    def __init__(self):
        super().__init__()
        self.decoder = QwenImageDecoder3D()
        self.encoder = QwenImageEncoder3D()
        self.post_quant_conv = QwenImageCausalConv3D(16, 16, 1, 1, 0)
        self.quant_conv = QwenImageCausalConv3D(32, 32, 1, 1, 0)  # Keep 32 channels like diffusers

    def _validate_convs(self) -> None:
        # Some Qwen-Image revisions have shipped all-zero quant_conv / post_quant_conv weights, which
        # silently zeroes the latent path (encode -> ~noise, decode -> a flat field). Fail loudly on
        # first use instead of wasting a training run against garbage. Checked once.
        if getattr(self, "_convs_validated", False):
            return
        for name in ("quant_conv", "post_quant_conv"):
            weight = getattr(self, name).conv3d.weight
            if bool(mx.all(weight == 0).item()):
                raise ValueError(
                    f"QwenVAE.{name} weights are all zero — the loaded VAE is broken (some Qwen-Image "
                    "revisions shipped zeroed quant convs, which silently corrupts encode/decode). "
                    "Re-download or re-save the model."
                )
        # Only mark validated after the checks pass, so a caught error is retried rather than skipped.
        self._convs_validated = True

    def decode(self, latents: mx.array) -> mx.array:
        self._validate_convs()
        if len(latents.shape) == 4:
            latents = latents.reshape(latents.shape[0], latents.shape[1], 1, latents.shape[2], latents.shape[3])
        latents = latents * QwenVAE.LATENTS_STD + QwenVAE.LATENTS_MEAN
        latents = self.post_quant_conv(latents)
        decoded = self.decoder(latents)
        return decoded

    def encode(self, latents: mx.array) -> mx.array:
        self._validate_convs()
        if len(latents.shape) == 4:
            latents = latents.reshape(latents.shape[0], latents.shape[1], 1, latents.shape[2], latents.shape[3])
        latents = self.encoder(latents)
        latents = self.quant_conv(latents)
        latents = latents[:, :16, :, :, :]
        return (latents - QwenVAE.LATENTS_MEAN) / QwenVAE.LATENTS_STD


# Materialize the latent constants at import time.
#
# `mx.array([...]).reshape(...)` records a Reshape primitive instead of evaluating, so these stay
# lazy. MLX >= 0.31 keeps a Metal command encoder per THREAD and pins an unevaluated array to the
# stream of the thread that recorded it, so evaluating one from another thread raises
# "There is no Stream(gpu, N) in current thread" (ml-explore/mlx#3529, open as of 0.32).
#
# That bites any host that imports mflux on one thread and generates on another. ComfyUI is the
# concrete case: it imports custom nodes on the main thread but runs every node on a single worker
# thread, so the first eval downstream of this VAE's encode/decode died. Evaluating here, on the
# importing thread, strips the stream affinity and makes the constants usable from any thread.
mx.eval(QwenVAE.LATENTS_MEAN, QwenVAE.LATENTS_STD)
