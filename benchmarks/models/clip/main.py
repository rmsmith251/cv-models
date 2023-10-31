import pytest
from PIL import Image

from models.vision.clip import CLIP


@pytest.mark.parametrize(
    "model_type",
    [
        "RN50",
        "RN101",
        "RN50x4",
        "RN50x16",
        "RN50x64",
        "ViT-B/32",
        "ViT-B/16",
        "ViT-L/14",
        "ViT-L/14@336px",
    ],
)
def test_clip(benchmark, model_type: str):
    model = CLIP(model_type=model_type, device="cuda")
    image = Image.open("tests/assets/person.jpg")
    text = ["a diagram", "a dog", "a cat", "a person"]
    for _ in range(5):
        _ = model.predict([(image, text)])

    benchmark(model.predict, [(image, text)])
