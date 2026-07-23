from collections.abc import Sequence

import mlx.core as mx
from mlx import nn

ImageShape = tuple[int, int, int]


class MageFlowEmbedRope(nn.Module):
    """Centered three-axis image RoPE used by Mage's NR-MMDiT."""

    def __init__(self, theta: float = 10000.0, axes_dim: Sequence[int] = (16, 56, 56)):
        super().__init__()
        if any(axis_dim % 2 for axis_dim in axes_dim):
            raise ValueError("every RoPE axis dimension must be even")
        self.theta = theta
        self.axes_dim = tuple(axes_dim)

    def __call__(
        self,
        image_shapes: ImageShape | Sequence[ImageShape] | Sequence[Sequence[ImageShape]],
    ) -> tuple[mx.array, mx.array]:
        shapes = self._normalize_shapes(image_shapes)
        angle_groups: list[mx.array] = []

        for image_index, (frames, height, width) in enumerate(shapes):
            if frames < 1 or height < 1 or width < 1:
                raise ValueError(f"invalid image shape {(frames, height, width)}")

            frame_positions = mx.arange(image_index, image_index + frames, dtype=mx.float32)
            height_positions = mx.arange(-(height - height // 2), height // 2, dtype=mx.float32)
            width_positions = mx.arange(-(width - width // 2), width // 2, dtype=mx.float32)

            frame_angles = self._axis_angles(frame_positions, self.axes_dim[0])
            height_angles = self._axis_angles(height_positions, self.axes_dim[1])
            width_angles = self._axis_angles(width_positions, self.axes_dim[2])

            frame_angles = mx.broadcast_to(
                frame_angles[:, None, None, :], (frames, height, width, frame_angles.shape[-1])
            )
            height_angles = mx.broadcast_to(
                height_angles[None, :, None, :],
                (frames, height, width, height_angles.shape[-1]),
            )
            width_angles = mx.broadcast_to(
                width_angles[None, None, :, :], (frames, height, width, width_angles.shape[-1])
            )
            angles = mx.concatenate([frame_angles, height_angles, width_angles], axis=-1)
            angle_groups.append(angles.reshape(frames * height * width, -1))

        all_angles = mx.concatenate(angle_groups, axis=0)
        return mx.cos(all_angles), mx.sin(all_angles)

    def _axis_angles(self, positions: mx.array, dim: int) -> mx.array:
        frequencies = 1.0 / (self.theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        return positions[:, None] * frequencies[None, :]

    @staticmethod
    def _normalize_shapes(
        image_shapes: ImageShape | Sequence[ImageShape] | Sequence[Sequence[ImageShape]],
    ) -> list[ImageShape]:
        if isinstance(image_shapes, tuple) and len(image_shapes) == 3:
            return [image_shapes]

        shapes = list(image_shapes)
        if len(shapes) == 1 and not isinstance(shapes[0], tuple):
            shapes = list(shapes[0])
        if not shapes or not all(isinstance(shape, tuple) and len(shape) == 3 for shape in shapes):
            raise ValueError("image_shapes must contain (frames, height, width) tuples")
        return shapes
