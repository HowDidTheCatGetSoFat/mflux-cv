import math

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as torch_functional
from mlx.utils import tree_flatten

from mflux.models.mage_flow.model.mage_flow_transformer.attention import MageFlowJointAttention
from mflux.models.mage_flow.model.mage_flow_transformer.rope_embedder import MageFlowEmbedRope
from mflux.models.mage_flow.model.mage_flow_transformer.timestep_embedder import MageFlowTimesteps
from mflux.models.mage_flow.model.mage_flow_transformer.transformer import MageFlowTransformer
from mflux.models.mage_flow.model.mage_flow_transformer.transformer_block import MageFlowTransformerBlock


def test_mage_flow_timestep_projection_matches_reference_bfloat16_rounding() -> None:
    timesteps = mx.array([1.0, 0.75, 0.125], dtype=mx.bfloat16)
    projection = MageFlowTimesteps(
        num_channels=8,
        flip_sin_to_cos=True,
        downscale_freq_shift=0,
        scale=1000,
    )(timesteps)

    torch_timesteps = torch.tensor([1.0, 0.75, 0.125], dtype=torch.bfloat16)
    half_dim = 4
    exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32)
    frequencies = torch.exp(exponent / half_dim).to(torch_timesteps.dtype)
    angles = 1000 * torch_timesteps[:, None].float() * frequencies[None, :]
    reference = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    np.testing.assert_allclose(
        np.asarray(projection),
        reference.numpy(),
        rtol=2e-5,
        atol=2e-5,
    )


def test_mage_flow_rope_uses_centered_spatial_and_per_image_frame_positions() -> None:
    rope = MageFlowEmbedRope(theta=10000, axes_dim=(2, 2, 2))
    cos, sin = rope((1, 3, 2))
    angles = mx.arctan2(sin, cos)

    expected = np.array(
        [
            [0, -2, -1],
            [0, -2, 0],
            [0, -1, -1],
            [0, -1, 0],
            [0, 0, -1],
            [0, 0, 0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.asarray(angles), expected, rtol=1e-6, atol=1e-6)

    edit_cos, edit_sin = rope([(1, 1, 1), (1, 1, 1)])
    edit_angles = mx.arctan2(edit_sin, edit_cos)
    expected_edit = np.array([[0, -1, -1], [1, -1, -1]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(edit_angles), expected_edit, rtol=1e-6, atol=1e-6)


def test_mage_flow_rope_rotates_adjacent_pairs_in_float32() -> None:
    hidden_states = mx.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=mx.bfloat16)
    angle = mx.array([[math.pi / 2, math.pi]], dtype=mx.float32)
    rotated = MageFlowJointAttention.apply_rotary_emb(hidden_states, (mx.cos(angle), mx.sin(angle)))

    expected = np.array([[[[-2.0, 1.0, -3.0, -4.0]]]], dtype=np.float32)
    np.testing.assert_allclose(
        np.asarray(rotated.astype(mx.float32)),
        expected,
        rtol=2e-2,
        atol=2e-2,
    )


def test_mage_flow_transformer_preserves_exact_checkpoint_paths() -> None:
    model = MageFlowTransformer()
    parameters = dict(tree_flatten(model.parameters()))

    assert len(parameters) == 397
    assert sum(math.prod(parameter.shape) for parameter in parameters.values()) == 4_115_745_408
    assert parameters["img_in.weight"].shape == (3072, 128)
    assert parameters["txt_in.weight"].shape == (3072, 2560)
    assert parameters["transformer_blocks.0.attn.to_q.weight"].shape == (3072, 3072)
    assert parameters["transformer_blocks.0.img_mod.1.weight"].shape == (18432, 3072)
    assert parameters["proj_out.weight"].shape == (128, 3072)
    assert "img_in.weight" in parameters
    assert "time_text_embed.timestep_embedder.linear_1.weight" in parameters
    assert "transformer_blocks.11.attn.add_q_proj.weight" in parameters
    assert "transformer_blocks.11.img_mlp.net.0.proj.weight" in parameters
    assert "transformer_blocks.11.txt_mlp.net.2.bias" in parameters
    assert "norm_out.linear.weight" in parameters
    assert "proj_out.bias" in parameters


def test_mage_flow_transformer_runs_batched_cfg_with_padded_text_mask() -> None:
    mx.random.seed(7)
    model = MageFlowTransformer(
        in_channels=4,
        out_channels=4,
        context_in_dim=6,
        hidden_size=16,
        num_attention_heads=2,
        depth=1,
        axes_dim=(2, 2, 4),
    )
    image = mx.random.normal((1, 4, 4))
    text = mx.random.normal((2, 3, 6))
    text_mask = mx.array([[1, 1, 1], [1, 1, 0]])

    output = model(
        img=image,
        txt=text,
        timesteps=mx.array(1.0),
        img_shapes=(1, 2, 2),
        text_attention_mask=text_mask,
    )
    mx.eval(output)

    assert output.shape == (2, 4, 4)
    assert bool(mx.all(mx.isfinite(output)).item())


def test_mage_flow_transformer_block_matches_torch_reference() -> None:
    rng = np.random.default_rng(123)
    block = MageFlowTransformerBlock(
        dim=8,
        num_attention_heads=2,
        attention_head_dim=4,
    )
    torch_weights: dict[str, torch.Tensor] = {}
    mlx_weights = []
    for name, parameter in tree_flatten(block.parameters()):
        values = (rng.standard_normal(parameter.shape) * 0.08).astype(np.float32)
        torch_weights[name] = torch.from_numpy(values)
        mlx_weights.append((name, mx.array(values)))
    block.load_weights(mlx_weights)

    image_np = (rng.standard_normal((1, 3, 8)) * 0.2).astype(np.float32)
    text_np = (rng.standard_normal((1, 2, 8)) * 0.2).astype(np.float32)
    timestep_np = (rng.standard_normal((1, 8)) * 0.2).astype(np.float32)
    angles_np = (rng.standard_normal((3, 2)) * 0.5).astype(np.float32)
    rotary = (mx.array(np.cos(angles_np)), mx.array(np.sin(angles_np)))

    mlx_text, mlx_image = block(
        hidden_states=mx.array(image_np),
        encoder_hidden_states=mx.array(text_np),
        temb=mx.array(timestep_np),
        image_rotary_emb=rotary,
    )
    mx.eval(mlx_text, mlx_image)

    image = torch.from_numpy(image_np)
    text = torch.from_numpy(text_np)
    timestep = torch.from_numpy(timestep_np)
    cos = torch.from_numpy(np.cos(angles_np))[:, None, :]
    sin = torch.from_numpy(np.sin(angles_np))[:, None, :]

    def linear(value: torch.Tensor, path: str) -> torch.Tensor:
        return torch_functional.linear(
            value,
            torch_weights[f"{path}.weight"],
            torch_weights.get(f"{path}.bias"),
        )

    def layer_norm(value: torch.Tensor) -> torch.Tensor:
        return torch_functional.layer_norm(value, (value.shape[-1],), eps=1e-6)

    def rms_norm(value: torch.Tensor, path: str) -> torch.Tensor:
        variance = torch.mean(torch.square(value.float()), dim=-1, keepdim=True)
        return (value.float() * torch.rsqrt(variance + 1e-6) * torch_weights[f"{path}.weight"]).to(value.dtype)

    def modulation(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk = value.shape[-1] // 3
        shift, scale, gate = value[..., :chunk], value[..., chunk : 2 * chunk], value[..., 2 * chunk :]
        return shift, scale, gate

    image_params = linear(torch_functional.silu(timestep), "img_mod.1")
    text_params = linear(torch_functional.silu(timestep), "txt_mod.1")
    image_mod_1, image_mod_2 = image_params.chunk(2, dim=-1)
    text_mod_1, text_mod_2 = text_params.chunk(2, dim=-1)
    image_shift_1, image_scale_1, image_gate_1 = modulation(image_mod_1)
    text_shift_1, text_scale_1, text_gate_1 = modulation(text_mod_1)
    image_input = layer_norm(image) * (1 + image_scale_1[:, None]) + image_shift_1[:, None]
    text_input = layer_norm(text) * (1 + text_scale_1[:, None]) + text_shift_1[:, None]

    def split_heads(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(value.shape[0], value.shape[1], 2, 4)

    image_query = rms_norm(split_heads(linear(image_input, "attn.to_q")), "attn.norm_q")
    image_key = rms_norm(split_heads(linear(image_input, "attn.to_k")), "attn.norm_k")
    image_value = split_heads(linear(image_input, "attn.to_v"))
    text_query = rms_norm(split_heads(linear(text_input, "attn.add_q_proj")), "attn.norm_added_q")
    text_key = rms_norm(split_heads(linear(text_input, "attn.add_k_proj")), "attn.norm_added_k")
    text_value = split_heads(linear(text_input, "attn.add_v_proj"))

    def rotate(value: torch.Tensor) -> torch.Tensor:
        pairs = value.float().reshape(*value.shape[:-1], -1, 2)
        real, imaginary = pairs[..., 0], pairs[..., 1]
        return torch.stack(
            [real * cos - imaginary * sin, real * sin + imaginary * cos],
            dim=-1,
        ).reshape_as(value)

    image_query = rotate(image_query)
    image_key = rotate(image_key)
    query = torch.cat([text_query, image_query], dim=1).transpose(1, 2)
    key = torch.cat([text_key, image_key], dim=1).transpose(1, 2)
    value = torch.cat([text_value, image_value], dim=1).transpose(1, 2)
    attended = torch_functional.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        scale=1 / math.sqrt(4),
    )
    attended = attended.transpose(1, 2).reshape(1, 5, 8)
    text_attention = linear(attended[:, :2], "attn.to_add_out")
    image_attention = linear(attended[:, 2:], "attn.to_out.0")
    image = image + image_gate_1[:, None] * image_attention
    text = text + text_gate_1[:, None] * text_attention

    image_shift_2, image_scale_2, image_gate_2 = modulation(image_mod_2)
    text_shift_2, text_scale_2, text_gate_2 = modulation(text_mod_2)
    image_input = layer_norm(image) * (1 + image_scale_2[:, None]) + image_shift_2[:, None]
    text_input = layer_norm(text) * (1 + text_scale_2[:, None]) + text_shift_2[:, None]

    def mlp(value: torch.Tensor, path: str) -> torch.Tensor:
        value = torch_functional.gelu(linear(value, f"{path}.net.0.proj"), approximate="tanh")
        return linear(value, f"{path}.net.2")

    image = image + image_gate_2[:, None] * mlp(image_input, "img_mlp")
    text = text + text_gate_2[:, None] * mlp(text_input, "txt_mlp")

    np.testing.assert_allclose(np.asarray(mlx_image), image.detach().numpy(), rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(np.asarray(mlx_text), text.detach().numpy(), rtol=2e-4, atol=2e-4)
