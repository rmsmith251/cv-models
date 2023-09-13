import numpy as np
import pytest

from vision.tracking.utils import vectorized_bbox_iou


@pytest.mark.parametrize(
    "arr1,arr2",
    [
        (
            [[100, 150, 200, 250], [100, 150, 200, 250]],
            [
                [90, 160, 190, 260],
                [0, 0, 10, 10],
                [100, 150, 101, 151],
            ],
        )
    ],
)
def test_vectorized_bbox_iou(arr1, arr2):
    _arr1 = np.asarray(arr1)
    _arr2 = np.asarray(arr2)
    iou = vectorized_bbox_iou(_arr1, _arr2)
    assert iou.shape == (len(arr1), len(arr2))
    assert (iou <= 1.0).all() and (iou >= 0.0).all()
    assert (iou == 0.0).any()


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
def test_benchmark_bbox_iou_np(benchmark, arr1_mult, arr2_mult):
    arr1 = np.asarray([[100, 150, 200, 250]] * arr1_mult)
    arr2 = np.asarray([[90, 160, 190, 260]] * arr2_mult)

    benchmark(vectorized_bbox_iou, arr1, arr2)
