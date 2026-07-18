"""Tests for the pure geometry helpers (NMS, polygons, diameter)."""

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import box

from crown_detector import geometry

_CRS = "EPSG:25830"


def test_iou_of_overlapping_squares() -> None:
    a = box(0, 0, 2, 2)
    b = box(1, 1, 3, 3)
    # intersection = 1, union = 7
    assert abs(geometry.iou(a, b) - 1 / 7) < 1e-9


def test_nms_suppresses_the_smaller_duplicate() -> None:
    gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 10, 10), box(1, 1, 9, 9)]}, crs=_CRS)
    result = geometry.apply_spatial_nms(gdf, iou_threshold=0.5)
    assert len(result) == 1  # the smaller, nested crown is dropped


def test_nms_keeps_disjoint_detections() -> None:
    gdf = gpd.GeoDataFrame(
        {"geometry": [box(0, 0, 5, 5), box(100, 100, 105, 105)]}, crs=_CRS
    )
    result = geometry.apply_spatial_nms(gdf, iou_threshold=0.5)
    assert len(result) == 2


def test_add_diameter_columns_filters_small_crowns() -> None:
    gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 10, 10), box(0, 0, 1, 1)]}, crs=_CRS)
    result = geometry.add_diameter_columns(gdf, min_diameter_m=3.0)
    assert len(result) == 1  # the 1x1 crown (diam ~1.13 m) is filtered out
    assert result["area_sqm"].iloc[0] == 100.0
    # equivalent-circle diameter of a 100 m2 crown
    assert abs(result["crown_diam_m"].iloc[0] - 2 * np.sqrt(100 / np.pi)) < 0.1


def test_masks_to_polygons_with_identity_transform() -> None:
    mask = np.array([[0, 0], [0, 4], [4, 4], [4, 0]], dtype=float)
    polys = geometry.masks_to_polygons(
        [mask], col_off=0, row_off=0, affine_transform=Affine.identity()
    )
    assert len(polys) == 1
    assert abs(polys[0].area - 16.0) < 1e-6
