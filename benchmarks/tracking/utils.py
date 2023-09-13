import numpy as np
import pytest

from vision.tracking.utils import vectorized_bbox_iou

test_runs = [
    (1, 1),
    (1, 100),
    (1, 1000),
    (10, 1),
    (10, 100),
    (10, 1000),
    (20, 1),
    (20, 100),
    (20, 1000),
    (40, 1),
    (40, 100),
    (40, 1000),
    (100, 100),
    (200, 200),
    (400, 400),
    (600, 600),
    (800, 800),
    (1000, 1000),
]


@pytest.mark.parametrize(
    "arr1_mult,arr2_mult",
    test_runs,
)
def test_benchmark_bbox_iou(benchmark, arr1_mult, arr2_mult):
    arr1 = np.asarray([[100, 150, 200, 250]] * arr1_mult)
    arr2 = np.asarray([[90, 160, 190, 260]] * arr2_mult)

    benchmark(vectorized_bbox_iou, arr1, arr2)
