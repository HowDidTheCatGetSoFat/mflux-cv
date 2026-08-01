import piexif
import PIL.Image
import pytest

from mflux.utils.image_util import ImageUtil


def _save_with_orientation(path, orientation: int) -> None:
    # 4x2 landscape with a red left column so the rotation is observable in pixels.
    image = PIL.Image.new("RGB", (4, 2), color=(0, 0, 255))
    for y in range(2):
        image.putpixel((0, y), (255, 0, 0))
    exif = piexif.dump({"0th": {piexif.ImageIFD.Orientation: orientation}})
    image.save(path, format="JPEG", exif=exif, quality=100, subsampling=0)


@pytest.mark.fast
def test_load_image_applies_exif_orientation(tmp_path):
    path = tmp_path / "rotated.jpg"
    _save_with_orientation(path, orientation=6)

    loaded = ImageUtil.load_image(path)

    # Orientation 6 means the stored pixels need a 270-degree rotation to display upright,
    # so the 4x2 file must come back as 2x4 with the red column now the top row.
    assert loaded.size == (2, 4)
    assert loaded.getpixel((0, 0))[0] > 128
    assert loaded.getpixel((1, 0))[0] > 128
    assert loaded.getpixel((0, 3))[2] > 128


@pytest.mark.fast
def test_load_image_without_orientation_is_untouched(tmp_path):
    path = tmp_path / "plain.jpg"
    PIL.Image.new("RGB", (4, 2), color=(0, 0, 255)).save(path, format="JPEG")

    loaded = ImageUtil.load_image(path)

    assert loaded.size == (4, 2)


@pytest.mark.fast
def test_load_image_applies_orientation_on_in_memory_images(tmp_path):
    path = tmp_path / "rotated.jpg"
    _save_with_orientation(path, orientation=6)

    loaded = ImageUtil.load_image(PIL.Image.open(path))

    assert loaded.size == (2, 4)
