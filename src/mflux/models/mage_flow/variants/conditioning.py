from collections.abc import Sequence

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.mage_flow.model.mage_flow_text_encoder import (
    MageFlowPromptProcessor,
    MageFlowTextEncoder,
)


class MageFlowConditioning:
    """Build the released Qwen3-VL conditioning batches for Mage Flow."""

    @staticmethod
    def encode_text_to_image(
        *,
        prompts: Sequence[str],
        tokenizer: Tokenizer,
        text_encoder: MageFlowTextEncoder,
        max_sequence_length: int = MageFlowPromptProcessor.MAX_CONDITION_TOKENS,
    ) -> tuple[mx.array, mx.array]:
        if not prompts:
            raise ValueError("at least one prompt is required")
        if not hasattr(tokenizer, "tokenizer"):
            raise TypeError("Mage Flow requires a tokenizer wrapper exposing its raw tokenizer")

        formatted = [MageFlowPromptProcessor.format_text_to_image(prompt) for prompt in prompts]
        max_input_length = max_sequence_length + MageFlowPromptProcessor.TEXT_TO_IMAGE_DROP_TOKENS
        tokens = tokenizer.tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=max_input_length,
            return_tensors="np",
        )
        input_ids = mx.array(np.asarray(tokens["input_ids"]))
        attention_mask = mx.array(np.asarray(tokens["attention_mask"]))
        hidden_states = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return MageFlowPromptProcessor.process_text_to_image_hidden_states(
            hidden_states,
            attention_mask,
        )

    @staticmethod
    def encode_edit(
        *,
        prompts: Sequence[str],
        images_per_prompt: Sequence[Sequence[Image.Image]],
        tokenizer: Tokenizer,
        text_encoder: MageFlowTextEncoder,
        max_sequence_length: int = MageFlowPromptProcessor.MAX_CONDITION_TOKENS,
    ) -> tuple[mx.array, mx.array]:
        if not prompts:
            raise ValueError("at least one edit prompt is required")
        if len(prompts) != len(images_per_prompt):
            raise ValueError("prompts and image groups must have the same length")
        if not hasattr(tokenizer, "processor"):
            raise TypeError("Mage Flow edit requires a tokenizer wrapper exposing its vision processor")

        formatted: list[str] = []
        flat_images: list[Image.Image] = []
        for prompt, images in zip(prompts, images_per_prompt, strict=True):
            if not images:
                raise ValueError("every edit prompt requires at least one reference image")
            formatted.append(MageFlowPromptProcessor.format_edit(prompt, num_images=len(images)))
            flat_images.extend(images)

        max_input_length = max_sequence_length + MageFlowPromptProcessor.EDIT_DROP_TOKENS
        inputs = tokenizer.processor(
            text=formatted,
            images=flat_images,
            padding=True,
            truncation=True,
            max_length=max_input_length,
            return_tensors=None,
        )
        batch_size, sequence_length = inputs["input_ids"].shape
        position_ids = mx.broadcast_to(
            mx.arange(sequence_length, dtype=mx.int32)[None, :],
            (batch_size, sequence_length),
        )
        hidden_states = text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            position_ids=position_ids,
        )
        return MageFlowPromptProcessor.process_edit_hidden_states(
            hidden_states,
            inputs["attention_mask"],
        )
