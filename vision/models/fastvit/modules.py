from __future__ import annotations

import torch
import torch.nn as nn

from vision.models.components.conv_ffn import ConvFFN
from vision.models.mobileone.block import MobileOneBlock, fuse_kernel_bn


class Stem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, training: bool = False):
        super().__init__()
        self.block1 = MobileOneBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            num_conv_branches=1,
        )
        self.block2 = MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=out_channels,
            num_conv_branches=1,
        )
        self.block3 = MobileOneBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            num_conv_branches=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        return x


class RepMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        num_features: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        device: str | torch.device = "cpu",
        training: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device
        self.training = training

        self.bn = nn.BatchNorm2d(num_features=self.num_features, device=self.device)
        self.conv = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        self.reparam_conv: nn.Conv2d | None = None

    def training_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.bn(inputs)
        x = self.conv(x)
        return inputs + x

    def inference_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.reparam_conv is None:
            self.reparametrize()
        return self.reparam_conv(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.training_forward(inputs)
        else:
            return self.inference_forward(inputs)

    def reparametrize(self) -> None:
        if self.reparam_conv is None:
            kernel, bias = fuse_kernel_bn(self.conv, self.bn)
            self.reparam_conv = nn.Conv2d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
            self.reparam_conv.weight.data = kernel
            self.reparam_conv.bias.data = bias


class RepMixerStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        training: bool = False,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.training = training
        self.device = device

        self.rep_block = RepMixer(
            channels=self.out_channels, num_features=self.in_channels
        )
        self.ffn = ConvFFN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            self.rep_block.reparametrize()
        x = self.rep_block(inputs)
        x = self.ffn(x)
        return x


class AttentionStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        training: bool = False,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.training = training
        self.device = device
