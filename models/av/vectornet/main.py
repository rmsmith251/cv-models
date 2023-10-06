from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DataPoint:
    start_pos: torch.Tensor
    end_pos: torch.Tensor
    attribute: int | float
    index: int

    @classmethod
    def from_list(self, data: list) -> DataPoint:
        return DataPoint(
            start_pos=torch.tensor(data[0]),
            end_pos=torch.tensor(data[1]),
            attribute=data[2],
            index=data[3],
        )


class NodeEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.device = device

        self.in_layer = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias,
            device=self.device,
        )
        self.norm = nn.LayerNorm(normalized_shape=self.out_features, device=self.device)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.in_layer(inputs)
        x = self.norm(x)
        breakpoint()
        return self.relu(x)


class SubGraph(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class GraphNeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class VectorNet(nn.Module):
    def __init__(
        self,
        in_features: int = 64,
        out_features: int = 64,
        kernel_size: int = 1024,
        stride: int | None = None,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.device = device

        self.node_encoder = NodeEncoder(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias,
            device=self.device,
        )
        self.agg = nn.MaxPool3d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    def forward(self, inputs: list[DataPoint]) -> torch.Tensor:
        x = self.node_encoder(inputs)
        y = self.agg(x)
        return torch.cat((x, y), 1)
