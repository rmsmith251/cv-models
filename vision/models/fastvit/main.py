from __future__ import annotations

import torch

from vision.models.base import BaseTrainer
from vision.models.fastvit.modules import Stage, Stem


class FastViT(BaseTrainer):
    """
    My personal implementation of Apple's FastViT model

    https://arxiv.org/pdf/2303.14189v2.pdf
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        training: bool = False,
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        self.stem = Stem(in_channels=in_channels, out_channels=out_channels)
        self.stage1 = Stage(training=training)
        self.stage2 = Stage(training=training)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.stem(inputs)
        x = self.stage1(x)

    def train(self):
        pass
