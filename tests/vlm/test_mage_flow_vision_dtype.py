import mlx.core as mx
from mlx import nn

from mflux.models.mage_flow.model.mage_flow_text_encoder import MageFlowTextEncoder


class _PatchEmbedWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 1, bias=False)
        self.proj.weight = self.proj.weight.astype(mx.bfloat16)


class _VisionDtypeSpy(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.spatial_merge_size = 1
        self.patch_embed = _PatchEmbedWeights()
        self.received_dtype = None

    def __call__(
        self,
        pixel_values: mx.array,
        image_grid_thw: mx.array,
        return_deepstack: bool = False,
    ) -> tuple[mx.array, list[mx.array] | None]:
        del image_grid_thw
        self.received_dtype = pixel_values.dtype
        image_embeds = mx.ones((1, self.hidden_size), dtype=pixel_values.dtype)
        return image_embeds, [] if return_deepstack else None


def test_mage_flow_casts_processor_pixels_to_vision_checkpoint_dtype() -> None:
    visual = _VisionDtypeSpy(hidden_size=8)
    model = MageFlowTextEncoder(
        vocab_size=8,
        hidden_size=8,
        num_hidden_layers=0,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=16,
        max_position_embeddings=32,
        head_dim=8,
        mrope_section=(2, 1, 1),
        image_token_id=3,
        vision_start_token_id=2,
        visual=visual,
    )

    model(
        mx.array([[2, 3]], dtype=mx.int32),
        pixel_values=mx.ones((1, 1), dtype=mx.float32),
        image_grid_thw=mx.array([[1, 1, 1]], dtype=mx.int32),
    )

    assert visual.patch_embed.proj.weight.dtype == mx.bfloat16
    assert visual.received_dtype == mx.bfloat16
