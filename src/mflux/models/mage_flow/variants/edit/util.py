from pathlib import Path

import mlx.core as mx
from PIL import Image

from mflux.models.mage_flow.latent_creator import MageFlowLatentCreator
from mflux.models.mage_flow.model.mage_flow_vae import MageVAE
from mflux.models.mage_flow.variants.pipeline_helpers import normalize_image_dimension
from mflux.utils.image_util import ImageUtil


class MageFlowEditUtil:
    @staticmethod
    def load_references(
        image_paths: Path | str | Image.Image | list[Path | str | Image.Image],
    ) -> list[Image.Image]:
        references = image_paths if isinstance(image_paths, list) else [image_paths]
        if not references:
            raise ValueError("Mage Flow edit requires at least one reference image")
        return [ImageUtil.load_image(reference) for reference in references]

    @staticmethod
    def resolve_target_size(
        primary_image: Image.Image,
        *,
        width: int | None,
        height: int | None,
        max_size: int | None = None,
    ) -> tuple[int, int]:
        source_width, source_height = primary_image.size
        if width is not None and height is not None:
            target_width, target_height = width, height
        elif max_size:
            scale = max_size / max(source_width, source_height)
            target_width = round(source_width * scale)
            target_height = round(source_height * scale)
        elif width is not None:
            target_width = width
            target_height = round(source_height * width / source_width)
        elif height is not None:
            target_height = height
            target_width = round(source_width * height / source_height)
        else:
            target_width, target_height = source_width, source_height

        target_width = normalize_image_dimension(target_width)
        target_height = normalize_image_dimension(target_height)
        return target_width, target_height

    @staticmethod
    def resize_long_edge(image: Image.Image, max_long_edge: int | None = 384) -> Image.Image:
        if max_long_edge is None or max_long_edge <= 0:
            return image
        long_edge = max(image.size)
        if long_edge <= max_long_edge:
            return image
        scale = max_long_edge / long_edge
        new_width = max(1, round(image.width * scale))
        new_height = max(1, round(image.height * scale))
        return image.resize((new_width, new_height), Image.Resampling.BICUBIC)

    @staticmethod
    def prepare_vae_images(
        images: list[Image.Image],
        *,
        width: int,
        height: int,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> mx.array:
        arrays = []
        for image in images:
            resized = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
            arrays.append(ImageUtil.to_array(resized)[0])
        return mx.stack(arrays, axis=0).astype(dtype)

    @staticmethod
    def encode_references(
        vae: MageVAE,
        images: list[Image.Image],
        *,
        width: int,
        height: int,
        seed: int,
    ) -> mx.array:
        image_batch = MageFlowEditUtil.prepare_vae_images(
            images,
            width=width,
            height=height,
            dtype=vae.encoder.patch_cond_embed.weight.dtype,
        )
        key = mx.random.key(seed) if vae.sample_posterior else None
        reference_latents = vae.encode(image_batch, key=key)
        packed = MageFlowLatentCreator.pack_latents(reference_latents)
        return packed.reshape(1, packed.shape[0] * packed.shape[1], packed.shape[2])
