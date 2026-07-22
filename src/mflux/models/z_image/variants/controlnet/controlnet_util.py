from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_vae.vae import VAE
from mflux.models.z_image.variants.controlnet.control_types import ControlSpec, ControlType
from mflux.utils.image_util import ImageUtil

log = logging.getLogger(__name__)

# DepthPro is a separate ~heavy model; load it once, and only if a depth control is actually used.
_DEPTH_PRO = None


def _get_depth_pro():
    global _DEPTH_PRO
    if _DEPTH_PRO is None:
        from mflux.models.depth_pro.model.depth_pro import DepthPro

        _DEPTH_PRO = DepthPro()
    return _DEPTH_PRO


@dataclass(frozen=True)
class EncodedControls:
    control_latents: list[mx.array]
    strengths: list[float]
    types: list[ControlType]
    images: list[PIL.Image.Image]


class ZImageControlnetUtil:
    @staticmethod
    def encode_controls(
        *,
        vae: VAE,
        width: int,
        height: int,
        controls: list[ControlSpec],
    ) -> EncodedControls:
        if len(controls) == 0:
            raise ValueError("At least one control must be provided.")

        control_latents: list[mx.array] = []
        strengths: list[float] = []
        types: list[ControlType] = []
        images: list[PIL.Image.Image] = []

        for control in controls:
            img = ImageUtil.load_image(control.image_path)
            img = ZImageControlnetUtil._scale_image(img=img, width=width, height=height)
            img = ZImageControlnetUtil._preprocess(img, control.type)

            arr = ImageUtil.to_array(img)
            latent = vae.encode(arr)  # (1, 16, 1, H/8, W/8)
            latent = ZImageLatentCreator.pack_latents(latent, height=height, width=width)  # (16, 1, H/8, W/8)

            # Diffusers ZImageControlNetPipeline behavior:
            # If base latents are 16ch but ControlNet expects 33ch, pad the control latents with zeros to 33ch.
            if latent.shape[0] < 33:
                padding = mx.zeros((33 - latent.shape[0], *latent.shape[1:]), dtype=latent.dtype)
                latent = mx.concatenate([latent, padding], axis=0)

            control_latents.append(latent)
            strengths.append(float(control.strength))
            types.append(control.type)
            images.append(img)

        return EncodedControls(control_latents=control_latents, strengths=strengths, types=types, images=images)

    @staticmethod
    def _preprocess(img: PIL.Image.Image, control_type: ControlType) -> PIL.Image.Image:
        # Union checkpoints accept any modality as an already-preprocessed hint. We compute the hint
        # for canny, mlsd and depth locally; hed and pose need neural estimators that are not part of
        # the MLX stack, so they stay pass-through (supply a pre-made hint for those).
        if control_type == ControlType.canny:
            # OpenCV Canny expects an 8-bit single-channel image.
            gray_u8 = np.array(img.convert("L"), dtype=np.uint8)
            edges_u8 = cv2.Canny(gray_u8, 100, 200)
            edges_rgb = np.repeat(edges_u8[:, :, None], 3, axis=2)
            return PIL.Image.fromarray(edges_rgb)

        if control_type == ControlType.mlsd:
            return ZImageControlnetUtil._mlsd(img)

        if control_type == ControlType.depth:
            return ZImageControlnetUtil._depth(img)

        return img

    @staticmethod
    def _mlsd(img: PIL.Image.Image) -> PIL.Image.Image:
        # Straight line segments as white strokes on black, approximating the MLSD hint with OpenCV's
        # LSD (no neural model). Good for architecture and interiors where the strong cues are edges.
        gray_u8 = np.array(img.convert("L"), dtype=np.uint8)
        lines = cv2.createLineSegmentDetector().detect(gray_u8)[0]
        canvas = np.zeros((gray_u8.shape[0], gray_u8.shape[1], 3), dtype=np.uint8)
        if lines is not None:
            for line in lines:
                x0, y0, x1, y1 = (int(round(v)) for v in line[0])
                cv2.line(canvas, (x0, y0), (x1, y1), (255, 255, 255), 1)
        return PIL.Image.fromarray(canvas)

    @staticmethod
    def _depth(img: PIL.Image.Image) -> PIL.Image.Image:
        # DepthPro (native MLX) reads a file path, so round-trip the scaled control image through a
        # temp file and hand back its depth map.
        import os
        import tempfile

        depth_pro = _get_depth_pro()
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            img.convert("RGB").save(tmp_path)
            depth_image = depth_pro.create_depth_map(image_path=tmp_path).depth_image
        finally:
            os.unlink(tmp_path)
        return depth_image.convert("RGB")

    @staticmethod
    def _scale_image(*, img: PIL.Image.Image, width: int, height: int) -> PIL.Image.Image:
        if height != img.height or width != img.width:
            log.warning(
                f"Control image {img.width}x{img.height} has different dimensions than requested. Resizing to {width}x{height}"
            )
            img = img.resize((width, height), PIL.Image.LANCZOS)
        return img
