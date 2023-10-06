import pytest

from models.utils import ImageConverter
from models.vision import MaskRCNN


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 20])
def test_mask_rcnn(benchmark, batch_size: int):
    img = ImageConverter(from_file="tests/assets/person.jpg").to_pil()
    model = MaskRCNN(device="cuda")
    assert model.device == "cuda"
    for _ in range(5):
        _ = model(img)

    @benchmark
    def run():
        _ = model([img] * batch_size)
