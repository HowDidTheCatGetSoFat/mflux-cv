from __future__ import annotations

import hashlib
import os
from math import erfc, sqrt
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

DEFAULT_GAUSSIAN_SHADING_PAYLOAD = "MageFlow"
DEFAULT_GAUSSIAN_SHADING_KEY = 20260720
_MESSAGE_BITS = 256


class MageFlowLatentCreator:
    """Create Mage's 128-channel latents, including its released watermark noise."""

    @staticmethod
    def create_noise(
        seed: int,
        height: int,
        width: int,
        *,
        gaussian_shading: bool = True,
        gaussian_shading_key: int | str | None = None,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> mx.array:
        latent_shape = (128, height // 16, width // 16)
        if gaussian_shading:
            key = MageFlowLatentCreator.resolve_gaussian_shading_key(gaussian_shading_key)
            latents = MageFlowLatentCreator._encode_gaussian_shading_noise(
                shape=latent_shape,
                key=key,
                seed=seed,
                dtype=dtype,
            )
        else:
            latents = mx.random.normal((1, *latent_shape), key=mx.random.key(seed)).astype(dtype)
        return MageFlowLatentCreator.pack_latents(latents)

    @staticmethod
    def pack_latents(latents: mx.array) -> mx.array:
        if latents.ndim != 4:
            raise ValueError("latents must have NCHW shape")
        batch, channels, height, width = latents.shape
        return latents.transpose(0, 2, 3, 1).reshape(batch, height * width, channels)

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        if latents.ndim != 3:
            raise ValueError("packed latents must have [batch, sequence, channels] shape")
        batch, sequence, channels = latents.shape
        latent_height = height // 16
        latent_width = width // 16
        if sequence != latent_height * latent_width:
            raise ValueError("packed latent sequence does not match the requested dimensions")
        return latents.reshape(batch, latent_height, latent_width, channels).transpose(0, 3, 1, 2)

    @staticmethod
    def resolve_gaussian_shading_key(explicit: int | str | None = None) -> int:
        if explicit is not None:
            return MageFlowLatentCreator._key_to_int(explicit)

        environment_key = os.environ.get("MAGEFLOW_GS_KEY")
        if environment_key and environment_key.strip():
            return MageFlowLatentCreator._key_to_int(environment_key)

        key_path = Path(os.environ.get("MAGEFLOW_GS_KEY_FILE", "~/.mageflow/gs_key")).expanduser()
        try:
            key_contents = key_path.read_text().strip()
            if key_contents:
                return MageFlowLatentCreator._key_to_int(key_contents)
        except OSError:
            pass
        return DEFAULT_GAUSSIAN_SHADING_KEY

    @staticmethod
    def decode_gaussian_shading(noise: mx.array, key: int | str) -> dict:
        if noise.ndim == 3:
            # Packed Mage latents are [B, HW, C]; restore the original NCHW
            # flattening order used when the keyed pad and positions were made.
            noise = noise.transpose(0, 2, 1)
        values = np.asarray(noise.astype(mx.float32)).reshape(-1)
        count = values.size
        message = MageFlowLatentCreator._payload_to_bits(DEFAULT_GAUSSIAN_SHADING_PAYLOAD)
        pad, positions = MageFlowLatentCreator._pad_and_positions(count, key)

        observed_half = (values > 0).astype(np.int64)
        expected_half = message[positions] ^ pad
        matches = int((observed_half == expected_half).sum())
        raw_accuracy = matches / count

        implied = observed_half ^ pad
        votes = np.zeros((_MESSAGE_BITS, 2), dtype=np.int64)
        np.add.at(votes, (positions, implied), 1)
        recovered = votes.argmax(axis=1)
        message_accuracy = float((recovered == message).mean())

        z_score = (matches - 0.5 * count) / (0.5 * sqrt(count))
        p_value = 0.5 * erfc(z_score / sqrt(2))
        return {
            "raw_acc": raw_accuracy,
            "msg_acc": message_accuracy,
            "matches": matches,
            "n": count,
            "z_score": z_score,
            "pvalue": p_value,
            "present": p_value < 1e-6,
            "msg_hat": recovered,
            "msg": message,
        }

    @staticmethod
    def _encode_gaussian_shading_noise(
        shape: tuple[int, int, int],
        key: int | str,
        seed: int,
        dtype: mx.Dtype,
    ) -> mx.array:
        channels, height, width = shape
        count = channels * height * width
        message = MageFlowLatentCreator._payload_to_bits(DEFAULT_GAUSSIAN_SHADING_PAYLOAD)
        pad, positions = MageFlowLatentCreator._pad_and_positions(count, key)
        target_half = (message[positions] ^ pad).astype(np.float64)

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        uniform = torch.rand(count, generator=generator, dtype=torch.float64)
        half = torch.from_numpy(target_half)
        probability = ((half + uniform) / 2.0).clamp(1e-6, 1.0 - 1e-6)
        gaussian = torch.special.ndtri(probability).reshape(1, channels, height, width)
        return mx.array(gaussian.to(torch.float32).numpy()).astype(dtype)

    @staticmethod
    def _key_to_int(value: int | str) -> int:
        if isinstance(value, int):
            return abs(value)
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("empty Gaussian-Shading key")
        if normalized.lstrip("-").isdigit():
            return abs(int(normalized))
        return int.from_bytes(hashlib.sha256(normalized.encode()).digest(), "big")

    @staticmethod
    def _payload_to_bits(payload: str, bit_count: int = _MESSAGE_BITS) -> np.ndarray:
        output: list[int] = []
        counter = 0
        while len(output) < bit_count:
            digest = hashlib.sha256(f"{payload}:{counter}".encode()).digest()
            for byte in digest:
                output.extend((byte >> bit_index) & 1 for bit_index in range(8))
            counter += 1
        return np.asarray(output[:bit_count], dtype=np.int64)

    @staticmethod
    def _pad_and_positions(
        count: int,
        key: int | str,
        bit_count: int = _MESSAGE_BITS,
    ) -> tuple[np.ndarray, np.ndarray]:
        generator = np.random.default_rng(MageFlowLatentCreator._key_to_int(key))
        pad = generator.integers(0, 2, size=count).astype(np.int64)
        positions = generator.integers(0, bit_count, size=count).astype(np.int64)
        return pad, positions
