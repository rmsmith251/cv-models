from __future__ import annotations

import numpy as np

from models.tracking.base import BaseTracker
from models.tracking.types import NUMPY_TYPES, TENSOR_TYPES


class IoU(BaseTracker):
    def match_ious(self, ious: np.ndarray):
        pass

    def update_boxes(
        self, predictions: dict[str, NUMPY_TYPES | TENSOR_TYPES]
    ) -> dict[str, NUMPY_TYPES | TENSOR_TYPES]:
        ious = self.bbox_ious(predictions)

        if ious is not None:
            self.match_ious(ious)

    def update_masks(
        self, predictions: dict[str, NUMPY_TYPES | TENSOR_TYPES]
    ) -> dict[str, NUMPY_TYPES | TENSOR_TYPES]:
        _ = self.mask_ious(predictions)

    def update(
        self, predictions: dict[str, NUMPY_TYPES | TENSOR_TYPES]
    ) -> dict[str, NUMPY_TYPES | TENSOR_TYPES]:
        if "masks" in predictions:
            self.update_masks(predictions)
        elif "boxes" in predictions:
            self.update_boxes(predictions)

        return self._current_state
