from __future__ import annotations

from abc import abstractmethod
from typing import List, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

IMAGE_TYPES = Union[
    np.ndarray, List[np.ndarray], List[Image.Image], torch.Tensor, List[torch.Tensor]
]


class BaseTrainer(nn.Module):
    def __init__(self):
        super().__init__()


class BaseInferenceModel(nn.Module):
    def __init__(self, max_image_size: int = 800, device: str | torch.device = "cpu"):
        super().__init__()
        self.device = device

    def preprocess_pil_images(self, inputs: IMAGE_TYPES) -> torch.Tensor:
        pass

    def preprocess_numpy_array(self, inputs: IMAGE_TYPES) -> torch.Tensor:
        torch.Tensor(cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB))

    def preprocess_tensors(self, inputs: IMAGE_TYPES) -> torch.Tensor:
        if isinstance(inputs, list):
            out = torch.Tensor(inputs)
        else:
            out = inputs
        return out

    def preprocess(self, inputs: IMAGE_TYPES) -> torch.Tensor:
        if isinstance(inputs, list):
            sample = inputs[0]
        else:
            sample = inputs

        if isinstance(sample, np.ndarray):
            return self.preprocess_numpy_array(inputs)
        elif isinstance(sample, Image.Image):
            return self.preprocess_pil_images(inputs)
        elif isinstance(sample, torch.Tensor):
            return self.preprocess_tensors(inputs)
        else:
            raise ValueError(f"Object of type {type(sample)} is not supported")

    @abstractmethod
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        ...

    def __call__(self, inputs: IMAGE_TYPES) -> torch.Tensor:
        return self.predict(self.preprocess(inputs))
