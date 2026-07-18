"""Pure geometry helpers: mask-to-polygon, spatial NMS and diameter attributes.

These functions carry no model or GPU state, so they are unit-testable on their
own without loading YOLO or opening a raster.
"""

import logging

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.errors import TopologicalError
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


def masks_to_polygons(
    masks: list[np.ndarray],
    col_off: int,
    row_off: int,
    affine_transform: Affine,
) -> list[Polygon]:
    """Convert tile-local pixel masks into georeferenced polygons.

    Two-step transform: tile-local pixel -> global image pixel (add the tile
    offset) -> CRS coordinate (apply the raster affine transform).
    """
    polygons: list[Polygon] = []
    for mask in masks:
        global_pixels = [(p[0] + col_off, p[1] + row_off) for p in mask]
        geo_coords = [affine_transform * p for p in global_pixels]
        try:
            poly = Polygon(geo_coords).buffer(0)  # buffer(0) repairs self-crossings
        except (TopologicalError, ValueError):
            continue
        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)
    return polygons


def iou(geom_left: Polygon, geom_right: Polygon) -> float:
    """Intersection-over-Union of two polygons (0.0 on failure)."""
    try:
        inter = geom_left.intersection(geom_right).area
        union = geom_left.union(geom_right).area
        return float(inter / union) if union > 0 else 0.0
    except (TopologicalError, ValueError):
        return 0.0


def apply_spatial_nms(gdf: gpd.GeoDataFrame, iou_threshold: float) -> gpd.GeoDataFrame:
    """Suppress duplicate detections from overlapping tiles via vectorized NMS.

    Larger crowns win ties; a smaller detection is dropped when it overlaps a
    larger one with IoU above ``iou_threshold``.
    """
    if len(gdf) == 0:
        return gdf

    logger.info("Running vectorized NMS on %d raw detections...", len(gdf))
    gdf = gdf.copy()
    gdf["_area"] = gdf.geometry.area
    gdf = gdf.sort_values("_area", ascending=False).reset_index(drop=True)
    gdf["_id"] = gdf.index

    joined = gpd.sjoin(
        gdf[["_id", "_area", "geometry"]],
        gdf[["_id", "_area", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    pairs = joined[joined["_id_left"] < joined["_id_right"]].copy()
    if len(pairs) == 0:
        return gdf.drop(columns=["_area", "_id"])

    pairs["_iou"] = pairs.apply(
        lambda row: iou(
            gdf.loc[row["_id_left"], "geometry"],
            gdf.loc[row["_id_right"], "geometry"],
        ),
        axis=1,
    )
    suppress_ids = set(pairs.loc[pairs["_iou"] > iou_threshold, "_id_right"].tolist())
    gdf_clean = (
        gdf[~gdf["_id"].isin(suppress_ids)]
        .drop(columns=["_area", "_id"])
        .reset_index(drop=True)
    )
    logger.info(
        "NMS complete: %d raw -> %d unique (removed %d duplicates)",
        len(gdf),
        len(gdf_clean),
        len(gdf) - len(gdf_clean),
    )
    return gdf_clean


def add_diameter_columns(
    gdf: gpd.GeoDataFrame, min_diameter_m: float
) -> gpd.GeoDataFrame:
    """Add ``area_sqm`` and ``crown_diam_m`` columns and filter tiny crowns.

    Crown diameter is the equivalent-circle diameter derived from polygon area.
    """
    if len(gdf) == 0:
        return gdf
    gdf = gdf.copy()
    gdf["area_sqm"] = gdf.geometry.area.round(2)
    gdf["crown_diam_m"] = (2.0 * np.sqrt(gdf["area_sqm"] / np.pi)).round(2)
    filtered: gpd.GeoDataFrame = (
        gdf[gdf["crown_diam_m"] >= min_diameter_m].copy().reset_index(drop=True)
    )
    return filtered
