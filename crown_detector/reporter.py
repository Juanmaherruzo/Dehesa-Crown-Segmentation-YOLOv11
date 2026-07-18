"""Dasometric statistics and GeoPackage export for detected crowns."""

import logging
from pathlib import Path
from typing import cast

import geopandas as gpd

logger = logging.getLogger(__name__)

# Dedicated stdout logger for the human-readable inventory summary.
report = logging.getLogger("crown_detector.report")
if not report.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    report.addHandler(_handler)
    report.setLevel(logging.INFO)
    report.propagate = False


class InventoryReporter:
    """Compute forest statistics and export the crown inventory to GeoPackage."""

    @staticmethod
    def compute_statistics(
        gdf: gpd.GeoDataFrame,
        raster_meta: dict[str, object],
        output_path: Path,
    ) -> gpd.GeoDataFrame:
        """Add ID and centroid columns, log the inventory summary and export."""
        if len(gdf) == 0:
            logger.warning("Empty GeoDataFrame - nothing to report.")
            return gdf

        gdf = gdf.copy().reset_index(drop=True)
        gdf["id"] = range(1, len(gdf) + 1)
        centroids = gdf.geometry.centroid
        gdf["coord_x"] = centroids.x.round(2)
        gdf["coord_y"] = centroids.y.round(2)

        total_area_ha = cast(float, raster_meta["total_area_ha"])
        res_m = cast(float, raster_meta["res_m"])
        diameters = gdf["crown_diam_m"]
        canopy_cover = gdf["area_sqm"].sum() / (total_area_ha * 10_000) * 100
        density = len(gdf) / total_area_ha

        report.info("\n%s", "=" * 50)
        report.info("       FOREST INVENTORY STATISTICS")
        report.info("=" * 50)
        report.info("  GSD                  : %.2f cm/px", res_m * 100)
        report.info("  Valid area analyzed  : %.4f ha", total_area_ha)
        report.info("  Detected stems       : %d trees", len(gdf))
        report.info("  Density              : %.2f stems/ha", density)
        report.info("  Canopy Cover (FCC)   : %.2f %%", canopy_cover)
        report.info("-" * 50)
        report.info("  Mean crown diameter  : %.2f m", diameters.mean())
        report.info("  Std deviation (diam) : %.2f m", diameters.std())
        report.info("  Min diameter         : %.2f m", diameters.min())
        report.info("  Max diameter         : %.2f m", diameters.max())
        report.info("  Median diameter      : %.2f m", diameters.median())
        report.info("=" * 50)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["id", "area_sqm", "crown_diam_m", "coord_x", "coord_y", "geometry"]
        gdf[cols].to_file(str(output_path), driver="GPKG")
        logger.info("GeoPackage exported: %s", output_path.name)
        return gdf
