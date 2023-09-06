from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import cv2
import numpy as np
import torch
from PIL import Image

NUMPY_ARRAYS = Union[np.ndarray, Sequence[np.ndarray]]
PIL_IMAGES = Union[Image.Image, Sequence[Image.Image]]
NON_TENSOR_IMAGES = Union[PIL_IMAGES, NUMPY_ARRAYS]

IMAGE_TYPES = Union[NUMPY_ARRAYS, PIL_IMAGES, torch.Tensor, Sequence[torch.Tensor]]


def validate_type(object: NON_TENSOR_IMAGES, expected_type: NON_TENSOR_IMAGES) -> None:
    if isinstance(object, list):
        check = object[0]
    else:
        check = object

    assert isinstance(
        check, expected_type
    ), f"Expected {expected_type}, got {type(check)}"


@dataclass
class ImageConverter:
    from_numpy: NUMPY_ARRAYS | None = None
    from_pil: PIL_IMAGES | None = None
    mode: str = "RGB"

    _images: NON_TENSOR_IMAGES | None = None

    def __post_init__(self):
        if self.from_numpy is not None:
            validate_type(self.from_numpy, np.ndarray)
            self._images = self.from_numpy
        elif self.from_pil is not None:
            validate_type(self.from_pil, Image.Image)
            self._images = self.from_pil
        else:
            raise ValueError("No images provided")

    def to_pil(self) -> PIL_IMAGES:
        if self.from_pil is None:
            if isinstance(self.from_numpy, Sequence) or len(self.from_numpy.shape) == 4:
                self.from_pil = [
                    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), self.mode)
                    for img in self.from_numpy
                ]
            else:
                self.from_pil = Image.fromarray(
                    cv2.cvtColor(self.from_numpy, cv2.COLOR_BGR2RGB), self.mode
                )

        return self.from_pil

    def to_numpy(self) -> NUMPY_ARRAYS:
        if self.from_numpy is None:
            self.from_numpy = np.array(self.from_pil)
            if isinstance(self.from_pil, Sequence):
                self.from_numpy = list(self.from_numpy)

        return self.from_numpy
