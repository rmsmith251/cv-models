from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
from abc import abstractmethod

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseRepBranch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_features: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        groups: int = 1,
        dilation: int = 1,
        bias: bool = False,
        ignore: bool = False,
        device: str | torch.device = "cpu",
    ):
        super(BaseRepBranch, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self.bias = bias
        self._id_tensor = None
        self.ignore = ignore
        self.device = device

        self.batchnorm = nn.BatchNorm2d(num_features=self.num_features, device=device)
        self.conv: Optional[nn.Conv2d] = None

    def id_tensor(self):
        if self._id_tensor is None:
            if self.conv is not None:
                self._id_tensor = self.conv.weight
            else:
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.out_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype=self.batchnorm.weight.dtype,
                    device=self.batchnorm.weight.device,
                )
                for idx in range(self.in_channels):
                    kernel_value[
                        idx,
                        idx % input_dim,
                        self.kernel_size // 2,
                        self.kernel_size // 2,
                    ] = 1
                self._id_tensor = kernel_value

        return self._id_tensor

    def fuse_bn(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The following implementation is adapted from Apple's MobileOne implementation
        https://github.com/apple/ml-mobileone/blob/main/mobileone.py#L219
        """
        if self.ignore:
            return 0, 0
        kernel = self.id_tensor()
        running_mean = self.batchnorm.running_mean
        running_var = self.batchnorm.running_var
        gamma = self.batchnorm.weight
        beta = self.batchnorm.bias
        eps = self.batchnorm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ...


class SkipBranch(BaseRepBranch):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_features: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        groups: int = 1,
        dilation: int = 1,
        bias: bool = False,
        ignore: bool = False,
        device: str | torch.device = "cpu",
    ):
        super().__init__(
            in_channels,
            out_channels,
            num_features,
            kernel_size,
            stride,
            padding,
            groups,
            dilation,
            bias,
            ignore,
            device,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.ignore:
            return 0
        return self.batchnorm(inputs)


class ConvBranch(BaseRepBranch):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_features: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        groups: int = 1,
        dilation: int = 1,
        bias: bool = False,
        ignore: bool = False,
        device: str | torch.device = "cpu",
    ):
        super().__init__(
            in_channels,
            out_channels,
            num_features,
            kernel_size,
            stride,
            padding,
            groups,
            dilation,
            bias,
            ignore,
            device,
        )
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias=self.bias,
            device=self.device,
        )

    def update_weights(self, kernel: torch.Tensor, bias: torch.Tensor) -> None:
        self.conv.weight.data = kernel
        if self.bias:
            self.conv.bias.data = bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.ignore:
            return 0
        x = self.conv(inputs)
        return self.batchnorm(x)


class MobileOneBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        num_conv_branches: int = 1,
        device: str | torch.device = "cpu",
    ):
        super(MobileOneBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self.bias = bias
        self.num_conv_branches = num_conv_branches
        self.device = device

        self.skip = SkipBranch(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_features=self.in_channels,
            ignore=not (self.out_channels == self.in_channels and self.stride == 1),
            device=device,
        )
        self.scale = ConvBranch(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_features=self.out_channels,
            kernel_size=1,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
            ignore=self.kernel_size <= 1,
            device=device,
        )
        convs = [
            ConvBranch(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                num_features=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
                ignore=False,
                device=device,
            )
            for _ in range(self.num_conv_branches)
        ]
        self.convs = nn.ModuleList(convs)
        self.reparam_conv = None
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.autocast(DEVICE, dtype=torch.float16):
            skip = self.skip(inputs)
            scale = self.scale(inputs)
            out = skip + scale
            for conv in self.convs:
                out += conv(inputs)

            return self.activation(out)

    def fuse_bn(self) -> tuple[torch.Tensor, torch.Tensor]:
        scale_kernel, scale_bias = self.scale.fuse_bn()
        if isinstance(scale_kernel, torch.Tensor):
            pad = self.kernel_size // 2
            scale_kernel = nn.functional.pad(scale_kernel, [pad, pad, pad, pad])
        skip_kernel, skip_bias = self.skip.fuse_bn()
        conv_kernel, conv_bias = 0, 0
        for conv in self.convs:
            kernel, bias = conv.fuse_bn()
            conv_kernel += kernel
            conv_bias += bias
        return (
            skip_kernel + scale_kernel + conv_kernel,
            skip_bias + scale_bias + conv_bias,
        )

    def reparametrize(self) -> None:
        if self.reparam_conv is None:
            kernel, bias = self.fuse_bn()
            self.reparam_conv = ConvBranch(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                num_features=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
                ignore=False,
                device=self.device,
            )
            self.reparam_conv.update_weights(kernel, bias)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.reparam_conv is None:
            self.reparametrize()
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.float16):
            return self.activation(self.reparam_conv(inputs))
