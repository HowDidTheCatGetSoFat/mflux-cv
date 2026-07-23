import numpy as np
import PIL.Image
import pytest

from mflux.models.z_image.variants.controlnet.control_types import ControlType
from mflux.models.z_image.variants.controlnet.controlnet_util import ZImageControlnetUtil


def _photo() -> PIL.Image.Image:
    # A single bright rectangle on black gives the edge/line detectors something to find.
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    a[16:48, 16:48] = 255
    return PIL.Image.fromarray(a)


@pytest.mark.fast
def test_canny_and_mlsd_produce_a_hint_not_the_original():
    img = _photo()
    canny = ZImageControlnetUtil._preprocess(img, ControlType.canny)
    mlsd = ZImageControlnetUtil._preprocess(img, ControlType.mlsd)
    for hint in (canny, mlsd):
        arr = np.array(hint)
        assert hint.size == img.size
        assert hint.mode == "RGB"
        # a real hint differs from the input photo
        assert arr.tobytes() != np.array(img).tobytes()
        # and actually contains detected strokes: an all-black output (detector disabled) must fail
        assert arr.max() == 255
        assert (arr == 255).any()
    # the mlsd hint is line strokes on black, so most of it is black
    assert (np.array(mlsd) == 0).mean() > 0.5


@pytest.mark.fast
def test_pose_is_pass_through():
    # Pose needs a detector + keypoint pipeline with no MLX path; the Union checkpoint takes a pre-made
    # hint, so preprocessing returns the image unchanged rather than a wrong or crashing result.
    img = _photo()
    out = ZImageControlnetUtil._preprocess(img, ControlType.pose)
    assert out is img
