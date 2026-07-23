from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.qwen.tokenizer.qwen_image_processor import QwenImageProcessor
from mflux.models.qwen.tokenizer.qwen_vision_language_processor import QwenVisionLanguageProcessor


class MageFlowQwen3VLImageProcessor(QwenImageProcessor):
    """Image preprocessing parameters embedded in the Mage-Flow checkpoint."""

    def __init__(self, max_long_edge: int | None = 384):
        super().__init__(
            min_pixels=65_536,
            max_pixels=16_777_216,
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )
        self.max_long_edge = max_long_edge

    def _preprocess(
        self,
        image: Image.Image,
        resized_height: int | None = None,
        resized_width: int | None = None,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        if resized_height is None and resized_width is None:
            image = self._resize_long_edge(image)
        return super()._preprocess(
            image,
            resized_height=resized_height,
            resized_width=resized_width,
        )

    def _resize_long_edge(self, image: Image.Image) -> Image.Image:
        if self.max_long_edge is None or self.max_long_edge <= 0:
            return image
        width, height = image.size
        long_edge = max(width, height)
        if long_edge <= self.max_long_edge:
            return image
        scale = self.max_long_edge / long_edge
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        return image.resize((resized_width, resized_height), Image.BICUBIC)


class MageFlowQwen3VLProcessor(QwenVisionLanguageProcessor):
    """Qwen3-VL tokenizer/image processor used by Mage-Flow edit prompts.

    The base processor expands each ``<|image_pad|>`` placeholder to
    ``prod(image_grid_thw) / merge_size**2`` tokens before tokenization.
    """

    def __init__(self, tokenizer, max_long_edge: int | None = 384):
        super().__init__(
            tokenizer=tokenizer,
            image_processor=MageFlowQwen3VLImageProcessor(max_long_edge=max_long_edge),
            image_token="<|image_pad|>",
            video_token="<|video_pad|>",
        )

    def __call__(
        self,
        images: Image.Image | list[Image.Image] | None = None,
        text: str | list[str] | None = None,
        padding: bool = True,
        return_tensors: str | None = None,
        max_length: int | None = 2112,
        truncation: bool = True,
    ) -> dict[str, Any]:
        image_inputs: dict[str, Any] = {}
        image_grid_thw = None
        if images is not None:
            image_list = images if isinstance(images, list) else [images]
            pixel_values, image_grid_thw = self.image_processor.preprocess(image_list)
            image_inputs = {
                "pixel_values": mx.array(pixel_values),
                "image_grid_thw": mx.array(image_grid_thw),
            }
        else:
            image_list = []

        if text is None:
            return image_inputs
        texts = [text] if isinstance(text, str) else text.copy()
        placeholder_count = sum(sample.count(self.image_token) for sample in texts)
        if placeholder_count != len(image_list):
            raise ValueError(f"found {placeholder_count} image placeholders for {len(image_list)} images")

        if image_grid_thw is not None:
            placeholder = "<|mage_flow_image_placeholder|>"
            image_index = 0
            for text_index, sample in enumerate(texts):
                while self.image_token in sample:
                    token_count = int(np.prod(image_grid_thw[image_index])) // self.image_processor.merge_size**2
                    sample = sample.replace(self.image_token, placeholder * token_count, 1)
                    image_index += 1
                texts[text_index] = sample.replace(placeholder, self.image_token)

        tokenizer_kwargs: dict[str, Any] = {
            "padding": padding,
            "return_tensors": "pt" if return_tensors == "pt" else "np",
        }
        if max_length is not None:
            tokenizer_kwargs["max_length"] = max_length
        if truncation:
            tokenizer_kwargs["truncation"] = True
        text_inputs = self.tokenizer(texts, **tokenizer_kwargs)
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs.get("attention_mask")
        if return_tensors == "pt" and hasattr(input_ids, "numpy"):
            input_ids = input_ids.numpy()
            attention_mask = attention_mask.numpy() if attention_mask is not None else None

        result = {
            **image_inputs,
            "input_ids": mx.array(np.asarray(input_ids)),
        }
        if attention_mask is not None:
            result["attention_mask"] = mx.array(np.asarray(attention_mask))
        return result
