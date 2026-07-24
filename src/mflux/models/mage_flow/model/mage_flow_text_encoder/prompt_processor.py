from collections.abc import Sequence

import mlx.core as mx
import numpy as np


class MageFlowPromptProcessor:
    MAX_CONDITION_TOKENS = 2048
    TEXT_TO_IMAGE_DROP_TOKENS = 34
    EDIT_DROP_TOKENS = 64
    IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

    TEXT_TO_IMAGE_TEMPLATE = (
        "<|im_start|>system\n"
        "Describe the image by detailing the color, shape, size, texture, quantity, "
        "text, spatial relationships of the objects and background:"
        "<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    EDIT_TEMPLATE = (
        "<|im_start|>system\n"
        "Describe the key features of the input image (color, shape, size, texture,"
        " objects, background), then explain how the user's text instruction should alter or modify the image. "
        "Generate a new image that meets the user's requirements while maintaining consistency with the original "
        "input where appropriate.<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    @classmethod
    def format_text_to_image(cls, prompt: str) -> str:
        return cls.TEXT_TO_IMAGE_TEMPLATE.format(prompt)

    @classmethod
    def format_edit(cls, instruction: str, num_images: int = 1) -> str:
        if num_images < 1:
            raise ValueError("an edit prompt requires at least one reference image")
        image_prefix = "".join(
            f"Image {image_index}: {cls.IMAGE_PLACEHOLDER}" for image_index in range(1, num_images + 1)
        )
        return cls.EDIT_TEMPLATE.format(image_prefix + instruction)

    @classmethod
    def process_text_to_image_hidden_states(
        cls,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        return cls.trim_and_pad_hidden_states(
            hidden_states,
            attention_mask,
            drop_tokens=cls.TEXT_TO_IMAGE_DROP_TOKENS,
        )

    @classmethod
    def process_edit_hidden_states(
        cls,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        return cls.trim_and_pad_hidden_states(
            hidden_states,
            attention_mask,
            drop_tokens=cls.EDIT_DROP_TOKENS,
        )

    @staticmethod
    def trim_and_pad_hidden_states(
        hidden_states: mx.array,
        attention_mask: mx.array,
        *,
        drop_tokens: int,
        max_length: int = MAX_CONDITION_TOKENS,
    ) -> tuple[mx.array, mx.array]:
        """Drop template tokens per sample and return a right-padded MLX batch."""

        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, channels]")
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("attention_mask must match the hidden-state batch and sequence dimensions")
        if drop_tokens < 0:
            raise ValueError("drop_tokens must be non-negative")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        # Token masks are small host-side metadata and selecting their active
        # indices once avoids synchronizing any model activations.
        mask = np.asarray(attention_mask).astype(bool, copy=False)
        trimmed: list[mx.array] = []
        for batch_index, sample_mask in enumerate(mask):
            active_indices = np.flatnonzero(sample_mask)
            active = hidden_states[batch_index, mx.array(active_indices, dtype=mx.int32)]
            trimmed.append(active[drop_tokens : drop_tokens + max_length])

        max_length = max((sample.shape[0] for sample in trimmed), default=0)
        hidden_size = hidden_states.shape[-1]
        padded_hidden_states = []
        padded_masks = []
        for sample in trimmed:
            sample_length = sample.shape[0]
            padding_length = max_length - sample_length
            padded_hidden_states.append(
                mx.concatenate(
                    [sample, mx.zeros((padding_length, hidden_size), dtype=hidden_states.dtype)],
                    axis=0,
                )
            )
            padded_masks.append(
                mx.concatenate(
                    [
                        mx.ones((sample_length,), dtype=mx.int32),
                        mx.zeros((padding_length,), dtype=mx.int32),
                    ],
                    axis=0,
                )
            )

        if not padded_hidden_states:
            return hidden_states[:, :0], attention_mask[:, :0].astype(mx.int32)
        return mx.stack(padded_hidden_states), mx.stack(padded_masks)

    @classmethod
    def format_edits(cls, instructions: Sequence[str], image_counts: Sequence[int]) -> list[str]:
        if len(instructions) != len(image_counts):
            raise ValueError("instructions and image_counts must have the same length")
        return [
            cls.format_edit(instruction, num_images=image_count)
            for instruction, image_count in zip(instructions, image_counts, strict=True)
        ]
