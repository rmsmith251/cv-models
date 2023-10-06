from __future__ import annotations

import torch
import torch.nn as nn

from models.vision.mobileone.block import MobileOneBlock


class PatchEmbed(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, device: str | torch.device = "cpu"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.block1 = MobileOneBlock(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=7,
            stride=2,
            padding=1,
            groups=out_channels,
            num_conv_branches=1,
            device=self.device,
        )
        self.block2 = MobileOneBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            num_conv_branches=1,
            ignore_scale=True,
            device=self.device,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.block1(inputs)
        x = self.block2(x)
        return x
