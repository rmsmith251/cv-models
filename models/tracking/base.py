from __future__ import annotations

from abc import abstractmethod

import numpy as np
import torch

from models.tracking.types import NUMPY_TYPES, TENSOR_TYPES, TrackedObject
from models.tracking.utils import vectorized_bbox_iou, vectorized_bbox_iou_torch


class BaseTracker:
    name: str
    patience: int = 1
    threshold: float = 0.5

    _objects: list[TrackedObject] = []
    _current_state: dict[str, NUMPY_TYPES | TENSOR_TYPES] | None = None
    _current_track_id: int = 1

    @property
    def object_boxes(self) -> NUMPY_TYPES | TENSOR_TYPES:
        return [obj.box for obj in self._objects]

    @property
    def object_masks(self) -> NUMPY_TYPES | TENSOR_TYPES:
        return [obj.mask for obj in self._objects]

    def bbox_ious(
        self, predictions: dict[str, NUMPY_TYPES | TENSOR_TYPES]
    ) -> np.ndarray | None:
        boxes = predictions.get("boxes", [])
        ious: np.ndarray | None = None
        if isinstance(boxes, torch.Tensor):
            torch_boxes = torch.Tensor(
                [box.tensor_box for box in self.object_boxes]
            ).to(boxes.device())
            ious = vectorized_bbox_iou_torch(boxes, torch_boxes).detach().cpu().numpy()
        elif isinstance(boxes, np.ndarray):
            np_boxes = np.asarray([box.np_box for box in self.object_boxes])
            ious = vectorized_bbox_iou(boxes, np_boxes)

        return ious

    def mask_ious(
        self, predictions: dict[str, NUMPY_TYPES | TENSOR_TYPES]
    ) -> np.ndarray | None:
        pass

    @abstractmethod
    def update(
        self, predictions: dict[str, NUMPY_TYPES | TENSOR_TYPES]
    ) -> dict[str, NUMPY_TYPES | TENSOR_TYPES]:
        ...
