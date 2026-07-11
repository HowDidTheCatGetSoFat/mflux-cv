import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.depth_pro.model.depth_pro import DepthPro
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.utils.image_util import ImageUtil


class Krea2DepthUtil:
    """Turn a source image (or a supplied depth map) into a Krea 2 depth-control latent.

    The reference depth-ControlNet was trained on Depth-Anything-V2 inverse-depth maps (near = white,
    far = black), normalized to [0, 1]. mflux ships Depth Pro instead (native MLX, metric depth, near =
    black), so its output is inverted here to match the training convention. Supplying a ready-made
    Depth-Anything map via ``depth_image_path`` skips estimation and reproduces the trained input most
    faithfully.
    """

    @staticmethod
    def encode_depth_control(
        vae: QwenVAE,
        depth_pro: DepthPro | None,
        width: int,
        height: int,
        image_path: str | Path | None = None,
        depth_image_path: str | Path | None = None,
    ) -> tuple[mx.array, PIL.Image.Image]:
        depth_image = Krea2DepthUtil._get_or_create_depth_image(
            depth_pro=depth_pro,
            image_path=image_path,
            depth_image_path=depth_image_path,
        )

        scaled = ImageUtil.scale_to_dimensions(
            image=depth_image.convert("RGB"),
            target_width=width,
            target_height=height,
        )
        depth_array = ImageUtil.to_array(scaled)
        control_latent = VAEUtil.encode(vae=vae, image=depth_array)
        return control_latent, depth_image

    @staticmethod
    def _get_or_create_depth_image(
        depth_pro: DepthPro | None,
        image_path: str | Path | None,
        depth_image_path: str | Path | None,
    ) -> PIL.Image.Image:
        # 1. A supplied depth map is used as-is (assumed near = white, per the training convention).
        if depth_image_path:
            if not os.path.exists(depth_image_path):
                raise FileNotFoundError(f"Depth map file not found: {depth_image_path}")
            return ImageUtil.load_image(depth_image_path)

        # 2. Otherwise estimate from the source image with Depth Pro and invert to near = white.
        if not image_path:
            raise ValueError("Either --depth-image-path or --image-path must be provided.")
        if depth_pro is None:
            raise ValueError("Depth Pro is required to estimate depth when no --depth-image-path is given.")
        result = depth_pro.create_depth_map(image_path=image_path)
        inverted = 255 - np.array(result.depth_image.convert("L")).astype(np.uint8)
        return PIL.Image.fromarray(inverted)
