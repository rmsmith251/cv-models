import numpy as np
import pytest
from shapely.geometry import Polygon

from models.tracking.utils import vectorized_bbox_iou

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


def bbox_to_shapely(arr):
    return ((arr[0], arr[1]), (arr[0], arr[3]), (arr[2], arr[3]), (arr[2], arr[1]))


def shapely_iou(poly1: Polygon, polys2: list[Polygon]) -> np.ndarray:
    intersections = [poly1.intersection(poly).area for poly in polys2]
    unions = [
        poly1.area + poly.area - intersections[idx] for idx, poly in enumerate(polys2)
    ]
    return np.array(intersections) / (np.array(unions) + 1e-8)


@pytest.mark.parametrize("arr1_mult,arr2_mult", test_runs[:10])
def test_shapely_bbox_iou(benchmark, arr1_mult, arr2_mult):
    """
    This only serves as an example for how much faster the vectorized version is. This converts the
    bounding boxes into Shapely Polygons and then computes the IoU for each polygon against the other
    array.
    """
    arr1 = np.asarray([[100, 150, 200, 250]] * arr1_mult)
    arr2 = np.asarray([[90, 160, 190, 260]] * arr2_mult)

    @benchmark
    def run():
        polys_1 = [Polygon(bbox_to_shapely(arr)) for arr in arr1]
        polys_2 = [Polygon(bbox_to_shapely(arr)) for arr in arr2]
        _ = [shapely_iou(poly, polys_2) for poly in polys_1]
