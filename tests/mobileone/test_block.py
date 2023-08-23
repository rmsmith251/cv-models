from __future__ import annotations

import pytest
import torch

from models.mobileone.block import MobileOneBlock

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Values taken from Apple's MobileOne example
# https://github.com/apple/ml-mobileone/blob/main/mobileone.py#L310
@pytest.mark.parametrize(
    "inputs,expected_size",
    [
        [[2, 512, 3, 48, 3, 2, 1, 1, 4], 256],  # Stage 0 in Apple example s0
        [[2, 512, 3, 48, 1, 2, 1, 1, 1], 257],  # Remove scale convolution
        [[2, 512, 3, 96, 1, 1, 0, 1, 1], 512],  # Stage 1 in Apple example s1 (PW)
        [[2, 512, 3, 256, 1, 1, 0, 1, 1], 512],  # Stage 1 in Apple example s2 (PW)
        [[1, 512, 3, 768, 1, 1, 0, 1, 1], 512],  # Stage 3 in Apple example s3 (PW)
        [
            [2, 512, 3, 2048, 1, 1, 0, 1, 1],
            512,
        ],  # Stage 4 in Apple example s4 (PW)
        [[2, 512, 64, 64, 3, 2, 1, 64, 1], 256],  # Depthwise convolution
    ],
)
def test_mobileone_block(inputs: list[int], expected_size: int):
    (
        batch_size,
        size,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups,
        branches,
    ) = inputs
    x = torch.randn(
        (batch_size, in_channels, size, size), dtype=torch.float32, device=DEVICE
    )
    model = MobileOneBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        num_conv_branches=branches,
        device=DEVICE,
    ).to(DEVICE)
    y = model.forward(x)
    assert list(y.shape) == [batch_size, out_channels, expected_size, expected_size]
    y = model(x)
    assert list(y.shape) == [batch_size, out_channels, expected_size, expected_size]
    # When calling the model, we reparametrize the layers into a single layer for inference
    assert model.reparam_conv is not None
