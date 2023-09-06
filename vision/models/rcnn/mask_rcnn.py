from __future__ import annotations

import torch
from torch import Tensor
from torchvision.models import detection

from vision.models.base import IMAGE_TYPES, BaseInferenceModel

TORCHVISION_ALIASES = {
    "mask-rcnn": detection.maskrcnn_resnet50_fpn,
    "mask-rcnn-v2": detection.maskrcnn_resnet50_fpn_v2,
}


class MaskRCNN(BaseInferenceModel):
    """
    This is just a wrapper around the Torchvision models for now until I implement Mask-RCNN for different
    backbones.
    """

    def __init__(self, model_name: str = "mask-rcnn", max_image_size: int = 800):
        super().__init__()
        self.model_name = model_name

        assert (
            self.model_name in TORCHVISION_ALIASES
        ), f"{self.model_name} is not a valid model name"
        self.model = TORCHVISION_ALIASES[self.model_name]

    @classmethod
    def aliases() -> list[str]:
        return list(TORCHVISION_ALIASES.keys())

    def preprocess(self, inputs: IMAGE_TYPES) -> torch.Tensor:
        pass

    def predict(self, inputs: Tensor) -> Tensor:
        pass
