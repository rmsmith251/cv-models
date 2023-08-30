from __future__ import annotations

import torch
import torch.nn as nn


class ConvFFN(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, device: str | torch.device = "cpu"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.conv_7x7 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=7,
            stride=1,
            padding=0,
            groups=self.out_channels,
            device=self.device,
        )
        self.bn = nn.BatchNorm2d(num_features=self.out_channels, device=self.device)
        self.conv_1x1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            device=self.device,
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv_7x7(inputs)
        x = self.bn(x)
        x = self.conv_1x1(x)
        x = self.activation(x)
        x = self.conv_1x1(x)
        return inputs + x
