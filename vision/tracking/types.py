from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision.tracking.utils import vectorized_bbox_iou

COORDINATE_PAIR = tuple[float, float]
POINTS = tuple[float, float, float, float]


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

    def iou(self, others: np.ndarray | list[np.ndarray]) -> np.ndarray:
        _others = np.asarray(others)
        return vectorized_bbox_iou(self.points[None, ...], others)


@dataclass
class Mask:
    idx: int
    mask: np.ndarray

    def iou(self, others: np.ndarray | list[np.ndarray]) -> np.ndarray:
        _others = np.asarray(others)

    def to_box(self) -> Box:
        pass


@dataclass
class TrackedObject:
    id: int
    box: Box | None = None
    mask: Mask | None = None

    def iou(self, others: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """
        Takes in either a 4D array of masks to compare against or a list of masks and
        returns an array of IoU values for each mask/bounding box.
        """
        pass

    def best_iou(self, others: np.ndarray | list[np.ndarray]) -> tuple[float, int]:
        arr = self.iou(others)
        return max(arr), np.argmax(arr)
