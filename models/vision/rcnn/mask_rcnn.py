from __future__ import annotations

import torch
import torchvision
from torchvision.models import detection

from models.vision.base import BaseImageInferenceModel

TORCHVISION_ALIASES = {
    "mask-rcnn": detection.maskrcnn_resnet50_fpn,
    "mask-rcnn-v2": detection.maskrcnn_resnet50_fpn_v2,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MaskRCNN(BaseImageInferenceModel):
    """
    This is just a wrapper around the Torchvision models for now until I implement Mask-RCNN for different
    backbones.
    """

    def __init__(
        self,
        model_name: str = "mask-rcnn",
        max_image_size: int = 800,
        weights: str = "DEFAULT",
        device: torch.device | str | None = None,
    ):
        super().__init__(
            model_name=model_name, max_image_size=max_image_size, device=device
        )
        self.model_name = model_name
        self.weights = weights
        self.device = device if device is not None else DEVICE

        assert (
            self.model_name in TORCHVISION_ALIASES
        ), f"{self.model_name} is not a valid model name"
        self.model = TORCHVISION_ALIASES[self.model_name](weights=self.weights).to(
            self.device
        )
        self.model.eval()

    @classmethod
    def aliases() -> list[str]:
        return list(TORCHVISION_ALIASES.keys())
