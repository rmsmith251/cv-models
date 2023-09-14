from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from vision.tracking.utils import vectorized_bbox_iou, vectorized_bbox_iou_torch

COORDINATE_PAIR = tuple[float, float]
POINTS = tuple[float, float, float, float]
NUMPY_TYPES = np.ndarray | list[np.ndarray]
TENSOR_TYPES = torch.Tensor | list[torch.Tensor]


@dataclass
class Box:
    idx: int
    points: POINTS

    _top_left: COORDINATE_PAIR | None = None
    _bottom_right: COORDINATE_PAIR | None = None

    @property
    def top_left(self) -> COORDINATE_PAIR:
        if self._top_left is None:
            self._top_left = (self.points[0], self.points[1])

        return self._top_left

    @property
    def bottom_right(self) -> COORDINATE_PAIR:
        if self._bottom_right is None:
            self._bottom_right = (self.points[2], self.points[3])

        return self._bottom_right

    @property
    def tensor_box(self) -> torch.Tensor:
        if isinstance(self.box, torch.Tensor):
            return self.box
        else:
            return torch.from_array(self.box)

    @property
    def np_box(self) -> np.ndarray:
        if isinstance(self.box, np.ndarray):
            return self.box
        else:
            return self.box.detach().cpu().numpy()

    def iou(self, others: NUMPY_TYPES | TENSOR_TYPES) -> np.ndarray:
        if isinstance(others, (torch.Tensor, list[torch.Tensor])):
            _others = others
            if isinstance(others, list):
                _others = torch.Tensor(others)
            return vectorized_bbox_iou_torch(
                self.tensor_box.to(_others.device()), _others
            )
        else:
            _others = np.asarray(others)
            return vectorized_bbox_iou(self.np_box, _others)


@dataclass
class Mask:
    idx: int
    mask: np.ndarray | torch.Tensor

    @property
    def tensor_mask(self) -> torch.Tensor:
        if isinstance(self.mask, torch.Tensor):
            return self.mask
        else:
            return torch.from_array(self.mask)

    @property
    def np_mask(self) -> np.ndarray:
        if isinstance(self.mask, np.ndarray):
            return self.mask
        else:
            return self.mask.detach().cpu().numpy()

    def iou(self, others: NUMPY_TYPES | TENSOR_TYPES) -> np.ndarray:
        if isinstance(others, (torch.Tensor, list[torch.Tensor])):
            _others = others
            if isinstance(others, list):
                _others = torch.Tensor(others)
            return vectorized_bbox_iou_torch(
                self.tensor_mask.to(_others.device()), _others
            )
        else:
            _others = np.asarray(others)
            return vectorized_bbox_iou(self.np_mask, _others)

    def to_box(self) -> Box:
        pass


@dataclass
class TrackedObject:
    id: int
    box: Box | None = None
    mask: Mask | None = None

    def iou(self, others: NUMPY_TYPES | TENSOR_TYPES) -> np.ndarray:
        """
        Takes in either a 4D array of masks to compare against or a list of masks and
        returns an array of IoU values for each mask/bounding box.
        """
        if self.mask is not None:
            self.mask.iou(others)
        elif self.box is not None:
            self.box.iou(others)
        else:
            raise ValueError("No bounding boxes or masks found")

    def best_iou(self, others: NUMPY_TYPES | TENSOR_TYPES) -> tuple[float, int]:
        arr = self.iou(others)
        return max(arr), np.argmax(arr)
