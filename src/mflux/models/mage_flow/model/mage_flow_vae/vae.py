import math
from functools import lru_cache

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention


def _apply_layers(layers: list[nn.Module], x: mx.array) -> mx.array:
    for layer in layers:
        x = layer(x)
    return x


def _group_norm(norm: nn.GroupNorm, x: mx.array) -> mx.array:
    dtype = x.dtype
    return norm(x.astype(mx.float32)).astype(dtype)


def _nonlinearity(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    if x.ndim == 4:
        shift = shift[:, None, None, :]
        scale = scale[:, None, None, :]
    else:
        shift = shift[:, None, :]
        scale = scale[:, None, :]
    return x * (1 + scale) + shift


def _normalize(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(
        num_groups=32,
        dims=channels,
        eps=1e-6,
        affine=True,
        pytorch_compatible=True,
    )


@lru_cache(maxsize=None)
def _nerf_position_embedding(patch_size: int, max_freqs: int, dtype: mx.Dtype) -> mx.array:
    position = mx.linspace(0, 1, patch_size, dtype=dtype)
    pos_y, pos_x = mx.meshgrid(position, position, indexing="ij")
    pos_x = pos_x.reshape(-1, 1, 1)
    pos_y = pos_y.reshape(-1, 1, 1)
    frequencies = mx.linspace(0, max_freqs, max_freqs, dtype=dtype)
    frequency_x = frequencies[None, :, None]
    frequency_y = frequencies[None, None, :]
    coefficients = (1 + frequency_x * frequency_y) ** -1
    dct_x = mx.cos(pos_x * frequency_x * math.pi)
    dct_y = mx.cos(pos_y * frequency_y * math.pi)
    embedding = (dct_x * dct_y * coefficients).reshape(1, -1, max_freqs**2)
    mx.eval(embedding)
    return embedding


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__(num_channels, eps=eps, affine=affine, bias=affine)


class _EncoderLayerNorm2d(LayerNorm2d):
    pass


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x_float = x.astype(mx.float32)
        variance = mx.mean(mx.square(x_float), axis=-1, keepdims=True)
        normalized = x_float * mx.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.astype(dtype)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = [
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        ]
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: mx.array, dim: int, max_period: int = 10_000) -> mx.array:
        half = dim // 2
        frequencies = mx.exp(
            -math.log(max_period) * mx.arange(half, dtype=mx.float32) / half,
        )
        arguments = t.reshape(-1, 1).astype(mx.float32) * frequencies[None, :]
        embedding = mx.concatenate([mx.cos(arguments), mx.sin(arguments)], axis=-1)
        if dim % 2:
            embedding = mx.concatenate([embedding, mx.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def __call__(self, t: mx.array) -> mx.array:
        embedding = self.timestep_embedding(t, self.frequency_embedding_size)
        return _apply_layers(self.mlp, embedding.astype(self.mlp[0].weight.dtype))


class BottleneckPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        pca_dim: int = 128,
        embed_dim: int = 384,
        bias: bool = True,
    ):
        super().__init__()
        self.proj1 = nn.Conv2d(
            in_chans,
            pca_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.proj2 = nn.Conv2d(pca_dim + embed_dim, embed_dim, kernel_size=1, bias=bias)

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        return self.proj2(mx.concatenate([self.proj1(x), cond], axis=-1))


class _GlobalAveragePool2d(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return mx.mean(x, axis=(1, 2), keepdims=True)


class _ConstAdaLN(nn.Module):
    def __init__(self, modulation: mx.array):
        super().__init__()
        self.modulation = modulation

    def __call__(self, c: mx.array) -> mx.array:
        if self.modulation.shape[0] == c.shape[0]:
            return self.modulation
        return mx.broadcast_to(self.modulation, (c.shape[0], *self.modulation.shape[1:]))


class DiCoBlock(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.conv2 = nn.Conv2d(
            hidden_size,
            hidden_size,
            3,
            padding=1,
            groups=hidden_size,
            bias=True,
        )
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.ca = [
            _GlobalAveragePool2d(),
            nn.Conv2d(hidden_size, hidden_size, 1, bias=True),
            nn.Sigmoid(),
        ]

        feed_forward_size = int(mlp_ratio * hidden_size)
        self.conv4 = nn.Conv2d(hidden_size, feed_forward_size, 1, bias=True)
        self.conv5 = nn.Conv2d(feed_forward_size, hidden_size, 1, bias=True)
        self.norm1 = LayerNorm2d(hidden_size, affine=False)
        self.norm2 = LayerNorm2d(hidden_size, affine=False)
        self.adaLN_modulation: list[nn.Module] | _ConstAdaLN = [
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        ]

    def __call__(self, inp: mx.array, c: mx.array) -> mx.array:
        if isinstance(self.adaLN_modulation, list):
            modulation = _apply_layers(self.adaLN_modulation, c)
        else:
            modulation = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(
            modulation,
            6,
            axis=-1,
        )

        x = _modulate(self.norm1(inp), shift_msa, scale_msa)
        x = nn.gelu(self.conv2(self.conv1(x)))
        x = x * _apply_layers(self.ca, x)
        x = self.conv3(x)
        x = inp + gate_msa[:, None, None, :] * x
        x = x + gate_mlp[:, None, None, :] * self.conv5(
            nn.gelu(self.conv4(_modulate(self.norm2(x), shift_mlp, scale_mlp))),
        )
        return x


class _EncoderDiCoBlock(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.conv2 = nn.Conv2d(
            hidden_size,
            hidden_size,
            3,
            padding=1,
            groups=hidden_size,
            bias=True,
        )
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.ca = [
            _GlobalAveragePool2d(),
            nn.Conv2d(hidden_size, hidden_size, 1, bias=True),
            nn.Sigmoid(),
        ]

        feed_forward_size = int(mlp_ratio * hidden_size)
        self.conv4 = nn.Conv2d(hidden_size, feed_forward_size, 1, bias=True)
        self.conv5 = nn.Conv2d(feed_forward_size, hidden_size, 1, bias=True)
        self.norm1 = _EncoderLayerNorm2d(hidden_size)
        self.norm2 = _EncoderLayerNorm2d(hidden_size)

    def __call__(self, inp: mx.array) -> mx.array:
        x = self.norm1(inp)
        x = nn.gelu(self.conv2(self.conv1(x)))
        x = x * _apply_layers(self.ca, x)
        x = self.conv3(x)
        x = inp + x
        return x + self.conv5(nn.gelu(self.conv4(self.norm2(x))))


class NerfEmbedder(nn.Module):
    def __init__(self, in_channels: int, hidden_size_input: int, max_freqs: int = 8):
        super().__init__()
        self.max_freqs = max_freqs
        self.embedder = [
            nn.Linear(in_channels + max_freqs**2, hidden_size_input, bias=True),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, patch_area, _ = x.shape
        patch_size = math.isqrt(patch_area)
        if patch_size * patch_size != patch_area:
            raise ValueError(f"Expected a square patch area, got {patch_area}")
        dct = mx.broadcast_to(
            _nerf_position_embedding(patch_size, self.max_freqs, x.dtype),
            (batch_size, patch_area, self.max_freqs**2),
        )
        return _apply_layers(self.embedder, mx.concatenate([x, dct], axis=-1))


class NerfFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(self.norm(x))


class _MLPResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = [
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        ]
        self.adaLN_modulation = [
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        ]

    def __call__(self, x: mx.array, y: mx.array) -> mx.array:
        shift, scale, gate = mx.split(_apply_layers(self.adaLN_modulation, y), 3, axis=-1)
        hidden_states = self.in_ln(x) * (1 + scale) + shift
        return x + gate * _apply_layers(self.mlp, hidden_states)


class SimpleMLPAdaLN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_res_blocks: int,
        patch_size: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.patch_size = patch_size
        self.cond_embed = nn.Linear(z_channels, patch_size**2 * model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = [_MLPResBlock(model_channels) for _ in range(num_res_blocks)]

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        x = self.input_proj(x)
        c = self.cond_embed(c).reshape(c.shape[0], self.patch_size**2, -1)
        for block in self.res_blocks:
            x = block(x, c)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = _normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = _normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        hidden_states = self.conv1(_nonlinearity(_group_norm(self.norm1, x)))
        hidden_states = self.conv2(self.dropout(_nonlinearity(_group_norm(self.norm2, hidden_states))))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + hidden_states


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int, patch_size: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.norm = _normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    @staticmethod
    def _to_patches(x: mx.array, patch_size: int) -> tuple[mx.array, int, int]:
        batch_size, height, width, channels = x.shape
        num_patch_rows = height // patch_size
        num_patch_columns = width // patch_size
        x = x.reshape(
            batch_size,
            num_patch_rows,
            patch_size,
            num_patch_columns,
            patch_size,
            channels,
        )
        x = mx.transpose(x, (0, 1, 3, 2, 4, 5))
        return x.reshape(-1, patch_size * patch_size, channels), num_patch_rows, num_patch_columns

    def __call__(self, x: mx.array) -> mx.array:
        hidden_states = _group_norm(self.norm, x)
        query = self.q(hidden_states)
        key = self.k(hidden_states)
        value = self.v(hidden_states)

        batch_size, height, width, channels = query.shape
        patch_size = self.patch_size
        pad_height = (patch_size - height % patch_size) % patch_size
        pad_width = (patch_size - width % patch_size) % patch_size
        if pad_height or pad_width:
            padding = ((0, 0), (0, pad_height), (0, pad_width), (0, 0))
            query = mx.pad(query, padding, mode="edge")
            key = mx.pad(key, padding, mode="edge")
            value = mx.pad(value, padding, mode="edge")

        query, num_patch_rows, num_patch_columns = self._to_patches(query, patch_size)
        key, _, _ = self._to_patches(key, patch_size)
        value, _, _ = self._to_patches(value, patch_size)
        attended = scaled_dot_product_attention(
            query[:, None, :, :],
            key[:, None, :, :],
            value[:, None, :, :],
            scale=channels**-0.5,
        )[:, 0]

        padded_height = num_patch_rows * patch_size
        padded_width = num_patch_columns * patch_size
        attended = attended.reshape(
            batch_size,
            num_patch_rows,
            num_patch_columns,
            patch_size,
            patch_size,
            channels,
        )
        attended = mx.transpose(attended, (0, 1, 3, 2, 4, 5)).reshape(
            batch_size,
            padded_height,
            padded_width,
            channels,
        )
        if pad_height or pad_width:
            attended = attended[:, :height, :width, :]
        return x + self.proj_out(attended)


class _Decoder(nn.Module):
    def __init__(self, out_ch: int = 384, z_ch: int = 128, attention_patch_size: int = 32):
        super().__init__()
        self.conv_in = nn.Conv2d(z_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.block = [
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
            AttnBlock(out_ch, patch_size=attention_patch_size),
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
            AttnBlock(out_ch, patch_size=attention_patch_size),
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
        ]
        self.norm_out = _normalize(out_ch)
        self.conv_out = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.ada = nn.Identity()

    def __call__(self, z: mx.array) -> mx.array:
        hidden_states = _apply_layers(self.block, self.conv_in(z))
        hidden_states = self.conv_out(_nonlinearity(_group_norm(self.norm_out, hidden_states)))
        return self.ada(hidden_states)


class _DConvEncoder(nn.Module):
    def __init__(
        self,
        z_ch: int = 128,
        hidden_size: int = 384,
        num_blocks: int = 21,
        patch_size: int = 16,
        mlp_ratio: float = 4.0,
        head_size: int = 768,
        num_head_blocks: int = 2,
        out_ch_mult: int = 2,
    ):
        super().__init__()
        self.z_ch = z_ch
        self.patch_size = patch_size
        self.patch_cond_embed = nn.Conv2d(
            3,
            head_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.head_blocks = [_EncoderDiCoBlock(head_size, mlp_ratio=mlp_ratio) for _ in range(num_head_blocks)]
        self.proj_down = nn.Conv2d(head_size, hidden_size, kernel_size=1, bias=True)
        self.z_proj = nn.Conv2d(z_ch, hidden_size, kernel_size=1, bias=True)
        self.fuse_proj = nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=1, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = [DiCoBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(num_blocks)]
        self.norm_out = LayerNorm2d(hidden_size)
        self.proj_out = nn.Conv2d(hidden_size, z_ch * out_ch_mult, kernel_size=1, bias=True)

    def forward_pred(self, z_t: mx.array, t: mx.array, y: mx.array) -> mx.array:
        cond = self.patch_cond_embed(y)
        for block in self.head_blocks:
            cond = block(cond)
        cond = self.proj_down(cond)

        hidden_states = self.fuse_proj(mx.concatenate([cond, self.z_proj(z_t)], axis=-1))
        timestep = self.t_embedder(t.reshape(-1))
        for block in self.blocks:
            hidden_states = block(hidden_states, timestep)
        return self.proj_out(self.norm_out(hidden_states))


class _YEmbedder(nn.Module):
    def __init__(self, ch: int = 384, z_ch: int = 128, attention_patch_size: int = 32):
        super().__init__()
        self.decoder = _Decoder(
            out_ch=ch,
            z_ch=z_ch,
            attention_patch_size=attention_patch_size,
        )


class _DConvDenoiser(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 384,
        hidden_size_x: int = 32,
        mlp_ratio: float = 4.0,
        num_blocks: int = 24,
        num_cond_blocks: int = 21,
        bottleneck_dim: int = 128,
        attention_patch_size: int = 32,
    ):
        super().__init__()
        if num_blocks < num_cond_blocks:
            raise ValueError("num_blocks must be greater than or equal to num_cond_blocks")
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.hidden_size_x = hidden_size_x
        self.num_cond_blocks = num_cond_blocks
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder_x = nn.Conv2d(
            hidden_size,
            hidden_size_x * patch_size**2,
            1,
            1,
            0,
        )
        self.x_embedder = NerfEmbedder(in_channels + hidden_size_x, hidden_size_x, max_freqs=8)
        self.s_embedder = BottleneckPatchEmbed(
            patch_size,
            in_channels,
            bottleneck_dim,
            hidden_size,
            bias=True,
        )
        self.blocks = [DiCoBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(num_cond_blocks)]
        self.dec_net = SimpleMLPAdaLN(
            in_channels=hidden_size_x,
            model_channels=hidden_size_x,
            out_channels=in_channels,
            z_channels=hidden_size,
            num_res_blocks=num_blocks - num_cond_blocks,
            patch_size=patch_size,
        )
        self.final_layer = NerfFinalLayer(hidden_size_x, in_channels)
        self.y_embedder = _YEmbedder(
            ch=hidden_size,
            z_ch=bottleneck_dim,
            attention_patch_size=attention_patch_size,
        )

    @staticmethod
    def _patchify(x: mx.array, patch_size: int) -> mx.array:
        batch_size, height, width, channels = x.shape
        grid_height = height // patch_size
        grid_width = width // patch_size
        x = x.reshape(
            batch_size,
            grid_height,
            patch_size,
            grid_width,
            patch_size,
            channels,
        )
        return mx.transpose(x, (0, 1, 3, 2, 4, 5)).reshape(
            batch_size,
            grid_height * grid_width,
            patch_size**2,
            channels,
        )

    def __call__(self, x: mx.array, t: mx.array, cond: mx.array) -> mx.array:
        batch_size, height, width, _ = x.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Height and width must be multiples of {self.patch_size}, got ({height}, {width})",
            )
        grid_height = height // self.patch_size
        grid_width = width // self.patch_size
        if cond.shape[1:3] != (grid_height, grid_width):
            raise ValueError(
                f"Conditioning grid must be ({grid_height}, {grid_width}), got {cond.shape[1:3]}",
            )

        timestep = self.t_embedder(t.reshape(-1))
        conditioning = self.s_embedder(x, cond)
        for block in self.blocks:
            conditioning = block(conditioning, timestep)

        length = grid_height * grid_width
        conditioning = conditioning.reshape(batch_size * length, self.hidden_size)
        image_patches = self._patchify(x, self.patch_size)
        embedded_cond = self.y_embedder_x(cond).reshape(
            batch_size,
            length,
            self.hidden_size_x,
            self.patch_size**2,
        )
        embedded_cond = mx.transpose(embedded_cond, (0, 1, 3, 2))
        pixel_features = mx.concatenate([image_patches, embedded_cond], axis=-1).reshape(
            batch_size * length,
            self.patch_size**2,
            -1,
        )
        pixel_features = self.x_embedder(pixel_features)
        pixel_features = self.dec_net(pixel_features, conditioning)
        pixels = self.final_layer(pixel_features)
        pixels = pixels.reshape(
            batch_size,
            grid_height,
            grid_width,
            self.patch_size,
            self.patch_size,
            self.in_channels,
        )
        return mx.transpose(pixels, (0, 1, 3, 2, 4, 5)).reshape(
            batch_size,
            height,
            width,
            self.in_channels,
        )


def _replace_adaln_with_const(blocks: list[DiCoBlock], c: mx.array) -> int:
    replaced = 0
    for block in blocks:
        if isinstance(block.adaLN_modulation, _ConstAdaLN):
            continue
        modulation = _apply_layers(block.adaLN_modulation, c)
        mx.eval(modulation)
        block.adaLN_modulation = _ConstAdaLN(modulation)
        replaced += 1
    return replaced


class MageVAE(nn.Module):
    latent_channels = 128
    downsample_factor = 16
    spatial_scale = 16

    def __init__(
        self,
        sample_posterior: bool = True,
        *,
        encoder: _DConvEncoder | None = None,
        decoder_model: _DConvDenoiser | None = None,
    ):
        super().__init__()
        self.sample_posterior = sample_posterior
        self.encoder = encoder or _DConvEncoder()
        self.decoder_model = decoder_model or _DConvDenoiser()
        if self.encoder.patch_size != self.decoder_model.patch_size:
            raise ValueError("Encoder and decoder patch sizes must match")
        if self.encoder.z_ch != self.decoder_model.y_embedder.decoder.conv_in.weight.shape[-1]:
            raise ValueError("Encoder and decoder latent channel counts must match")
        self.latent_channels = self.encoder.z_ch
        self.downsample_factor = self.encoder.patch_size
        self.spatial_scale = self.downsample_factor

    @property
    def dconv_encoder(self) -> _DConvEncoder:
        return self.encoder

    @staticmethod
    def _validate_nchw(x: mx.array, channels: int, name: str) -> None:
        if x.ndim != 4:
            raise ValueError(f"{name} must be rank 4 NCHW, got rank {x.ndim}")
        if x.shape[1] != channels:
            raise ValueError(f"{name} must have {channels} channels, got {x.shape[1]}")

    def _moments(self, x: mx.array) -> tuple[mx.array, mx.array]:
        batch_size, height, width, _ = x.shape
        patch_size = self.encoder.patch_size
        latent_state = mx.zeros(
            (
                batch_size,
                height // patch_size,
                width // patch_size,
                self.encoder.z_ch,
            ),
            dtype=x.dtype,
        )
        timestep = mx.zeros((batch_size,), dtype=x.dtype)
        moments = self.encoder.forward_pred(latent_state, timestep, x)
        mean, log_variance = mx.split(moments, 2, axis=-1)
        return mean, mx.clip(log_variance, -20.0, 10.0)

    def encode_moments(self, x: mx.array) -> tuple[mx.array, mx.array]:
        self._validate_nchw(x, 3, "Image")
        height, width = x.shape[-2:]
        patch_size = self.encoder.patch_size
        if height % patch_size or width % patch_size:
            raise ValueError(
                f"Height and width must be multiples of {patch_size}, got ({height}, {width})",
            )
        mean, log_variance = self._moments(mx.transpose(x, (0, 2, 3, 1)))
        return (
            mx.transpose(mean, (0, 3, 1, 2)),
            mx.transpose(log_variance, (0, 3, 1, 2)),
        )

    def encode(self, x: mx.array, *, key: mx.array | None = None) -> mx.array:
        if self.sample_posterior and key is None:
            raise ValueError("An explicit MLX random key is required when sampling the posterior")
        mean, log_variance = self.encode_moments(x)
        if not self.sample_posterior:
            return mean
        noise = mx.random.normal(shape=mean.shape, dtype=mean.dtype, key=key)
        return mean + mx.exp(0.5 * log_variance) * noise

    def decode(self, z: mx.array) -> mx.array:
        self._validate_nchw(z, self.latent_channels, "Latent")
        latent = mx.transpose(z, (0, 2, 3, 1))
        cond = self.decoder_model.y_embedder.decoder(latent)
        batch_size, latent_height, latent_width, _ = latent.shape
        height = latent_height * self.downsample_factor
        width = latent_width * self.downsample_factor
        noise = mx.zeros(
            (batch_size, height, width, self.decoder_model.in_channels),
            dtype=z.dtype,
        )
        timestep = mx.zeros((batch_size,), dtype=z.dtype)
        decoded = self.decoder_model(noise, timestep, cond)
        return mx.transpose(decoded, (0, 3, 1, 2))

    def freeze_adaln_cache(self) -> int:
        """Fold fixed-timestep DiCo modulation only after checkpoint weights are loaded."""
        encoder_dtype = self.encoder.patch_cond_embed.weight.dtype
        encoder_timestep = self.encoder.t_embedder(mx.zeros((1,), dtype=encoder_dtype))
        replaced = _replace_adaln_with_const(self.encoder.blocks, encoder_timestep)

        decoder_dtype = self.decoder_model.s_embedder.proj1.weight.dtype
        decoder_timestep = self.decoder_model.t_embedder(mx.zeros((1,), dtype=decoder_dtype))
        replaced += _replace_adaln_with_const(self.decoder_model.blocks, decoder_timestep)
        return replaced

    def _freeze_adaln_cache(self) -> int:
        return self.freeze_adaln_cache()
