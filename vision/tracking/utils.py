from __future__ import annotations

import numpy as np
import torch

COORDINATE_PAIR = tuple[float, float]
POINTS = tuple[float, float, float, float]


def vectorized_bbox_iou(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    x11 = arr1[:, 0][..., None]
    y11 = arr1[:, 1][..., None]
    x12 = arr1[:, 2][..., None]
    y12 = arr1[:, 3][..., None]

    x21 = arr2[:, 0]
    y21 = arr2[:, 1]
    x22 = arr2[:, 2]
    y22 = arr2[:, 3]

    x_a = np.maximum(x11, x21)
    y_a = np.maximum(y11, y21)
    x_b = np.minimum(x12, x22)
    y_b = np.minimum(y12, y22)

    intersection = np.maximum((x_b - x_a), 0) * np.maximum((y_b - y_a), 0)
    box_1_area = (x12 - x11) * (y12 - y11)
    box_2_area = (x22 - x21) * (y22 - y21)
    union = box_1_area + box_2_area - intersection
    iou = intersection / (union + 1e-8)
    return iou


def vectorized_bbox_iou_torch(arr1: torch.Tensor, arr2: torch.Tensor) -> torch.Tensor:
    x11 = torch.unsqueeze(arr1[:, 0], -1)
    y11 = torch.unsqueeze(arr1[:, 1], -1)
    x12 = torch.unsqueeze(arr1[:, 2], -1)
    y12 = torch.unsqueeze(arr1[:, 3], -1)

    x21 = arr2[:, 0]
    y21 = arr2[:, 1]
    x22 = arr2[:, 2]
    y22 = arr2[:, 3]

    x_a = torch.maximum(x11, x21)
    y_a = torch.maximum(y11, y21)
    x_b = torch.minimum(x12, x22)
    y_b = torch.minimum(y12, y22)

    zero_tensor = torch.Tensor([0]).to(arr1.device())
    intersection = torch.maximum((x_b - x_a), zero_tensor) * torch.maximum(
        (y_b - y_a), zero_tensor
    )
    box_1_area = (x12 - x11) * (y12 - y11)
    box_2_area = (x22 - x21) * (y22 - y21)
    union = box_1_area + box_2_area - intersection
    iou = intersection / (union + 1e-8)
    return iou


def vectorized_mask_iou(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    pass
