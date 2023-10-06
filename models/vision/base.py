from __future__ import annotations

from abc import abstractmethod
from typing import Callable, List, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from models.utils import ImageConverter

IMAGE_TYPES = Union[
    np.ndarray, List[np.ndarray], List[Image.Image], torch.Tensor, List[torch.Tensor]
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseTrainer(nn.Module):
    def __init__(self):
        super().__init__()


class BaseImageInferenceModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        max_image_size: int = 800,
        device: str | torch.device | None = DEVICE,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_image_size = max_image_size
        self.device = device
        self.model: Callable | None = None

    def preprocess(self, inputs: IMAGE_TYPES) -> torch.Tensor:
        tensors = ImageConverter.from_any(inputs).to_tensor(normalize=True)
        return tensors.to(self.device)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise NotImplementedError(f"{self.model_name} is not implemented")

        images = self.preprocess(inputs).to(dtype=torch.float16)
        with torch.no_grad(), torch.autocast(
            device_type=self.device, dtype=torch.float16
        ):
            out = self.model(images)
            if len(out) == 1:
                return out[0]
            return out

    def __call__(self, inputs: IMAGE_TYPES) -> torch.Tensor:
        return self.predict(inputs)
