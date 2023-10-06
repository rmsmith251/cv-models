from typing import Sequence

import numpy as np
import pytest
from PIL import Image

from models.utils import images

np_image = np.ones((300, 300, 3), dtype=np.uint8)
pil_image = Image.new("RGB", (300, 300))
NUM_IMAGES = 5


def test_validate_type():
    pass


@pytest.mark.parametrize(
    "input_type,input",
    [
        ("np", np_image),
        ("np", [np_image] * NUM_IMAGES),
        ("np", np.array([np_image] * NUM_IMAGES)),
        ("pil", pil_image),
        ("pil", [pil_image] * NUM_IMAGES),
        ("none", None),
    ],
)
def test_ImageConverter(input_type: str, input: images.NON_TENSOR_IMAGES):
    if input_type == "np":
        out = images.ImageConverter(from_numpy=input).to_pil()
        check = out
        check_type = Image.Image
    elif input_type == "pil":
        out = images.ImageConverter(from_pil=input).to_numpy()
        check = out
        check_type = np.ndarray
    elif input_type == "none":
        try:
            images.ImageConverter()
        except Exception:
            check = []
            check_type = list

    if input is not None:
        if isinstance(input, Sequence):
            assert isinstance(
                out, Sequence
            ), "The output should've been a sequence and it isn't"
            assert len(out) == len(
                input
            ), f"Sequence lengths have changed: in={len(input)}, out={len(out)}"
            check = out[0]
        elif isinstance(input, np.ndarray) and len(input.shape) == 4:
            assert isinstance(
                out, Sequence
            ), "4D numpy array wasn't converted to a list"
            assert (
                len(out) == NUM_IMAGES
            ), f"Sequence lengths have changed: in={input.shape[0]}, out={len(out)}"
            check = out[0]

    assert isinstance(
        check, check_type
    ), f"{type(check)} is not expected type {check_type}"
