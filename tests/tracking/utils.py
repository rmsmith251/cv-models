import numpy as np
import pytest
from shapely.geometry import Polygon

from models.tracking.utils import vectorized_bbox_iou


def bbox_to_shapely(arr):
    return ((arr[0], arr[1]), (arr[0], arr[3]), (arr[2], arr[3]), (arr[2], arr[1]))


def shapely_iou(poly1: Polygon, polys2: list[Polygon]) -> np.ndarray:
    intersections = [poly1.intersection(poly).area for poly in polys2]
    unions = [
        poly1.area + poly.area - intersections[idx] for idx, poly in enumerate(polys2)
    ]
    return np.array(intersections) / (np.array(unions) + 1e-8)


@pytest.mark.parametrize(
    "arr1,arr2",
    [
        (
            [[100, 150, 200, 250]],
            [
                [90, 160, 190, 260],
                [0, 0, 10, 10],
                [100, 150, 101, 151],
            ],
        )
    ],
)
def test_vectorized_bbox_iou(arr1, arr2):
    # Use Shapely to ensure that the vectorized values are correct
    polys_1 = [Polygon(bbox_to_shapely(arr)) for arr in arr1]
    polys_2 = [Polygon(bbox_to_shapely(arr)) for arr in arr2]
    ious = [shapely_iou(poly, polys_2) for poly in polys_1]

    _arr1 = np.asarray(arr1)
    _arr2 = np.asarray(arr2)
    iou = vectorized_bbox_iou(_arr1, _arr2)
    assert iou.shape == (
        len(arr1),
        len(arr2),
    ), "For some reason we're returning the wrong number of items"
    assert (iou <= 1.0).all() and (
        iou >= 0.0
    ).all(), "This should only return a ratio between 0 and 1"
    for idx, gt_iou in enumerate(ious):
        np.testing.assert_almost_equal(gt_iou, iou[idx])
