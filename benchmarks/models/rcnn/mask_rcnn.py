import pytest

from vision.models import MaskRCNN
from vision.utils import ImageConverter


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 24])
def test_mask_rcnn(benchmark, batch_size: int):
    img = ImageConverter(from_file="tests/assets/person.jpg").to_pil()
    model = MaskRCNN(device="cuda")
    assert model.device == "cuda"
    for _ in range(5):
        _ = model(img)

    @benchmark
    def run():
        _ = model([img] * batch_size)
