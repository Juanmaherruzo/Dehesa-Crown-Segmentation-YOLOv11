"""Sample one multispectral index value per tree crown.

For every index raster (NDVI, EVI, ...) found in a directory, this adds one
column to the crown GeoDataFrame holding the pixel value at each crown centroid.
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)


def _centroid_xy(geom: BaseGeometry | None) -> tuple[float, float]:
    """Return the ``(x, y)`` centroid of a geometry, or NaNs if empty/missing."""
    if geom is None or geom.is_empty:
        return (float("nan"), float("nan"))
    centroid = geom.centroid
    return (centroid.x, centroid.y)


def extract_indices(
    gpkg_path: Path,
    index_dir: Path,
    out_gpkg: Path,
    out_csv: Path,
) -> gpd.GeoDataFrame:
    """Attach one column per index raster (sampled at crown centroids) and export."""
    trees = gpd.read_file(gpkg_path)
    rasters = sorted(index_dir.glob("*.tif"))
    logger.info(
        "%d trees | %d indices: %s",
        len(trees),
        len(rasters),
        [r.stem for r in rasters],
    )
    if not rasters:
        raise FileNotFoundError(f"No index rasters (*.tif) found in {index_dir}")

    bad = trees.geometry.is_empty | trees.geometry.isna()
    if bad.any():
        logger.warning("%d trees with empty/null geometry -> NaN", int(bad.sum()))

    # Reproject crowns to the raster CRS before sampling.
    with rasterio.open(rasters[0]) as src0:
        raster_crs = src0.crs
    trees_r = trees.to_crs(raster_crs)
    centroids = [_centroid_xy(g) for g in trees_r.geometry]

    for rpath in rasters:
        name = rpath.stem
        with rasterio.open(rpath) as src:
            nodata = src.nodata
            values = np.full(len(trees_r), np.nan, dtype="float64")
            valid = [i for i, (x, _) in enumerate(centroids) if np.isfinite(x)]
            if valid:
                sampled = np.array(
                    [v[0] for v in src.sample([centroids[i] for i in valid])],
                    dtype="float64",
                )
                if nodata is not None:
                    sampled[sampled == nodata] = np.nan
                values[valid] = sampled
        trees[name] = values
        logger.info(
            "  %s: %d/%d trees with a value",
            name,
            int(np.isfinite(values).sum()),
            len(trees),
        )

    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    trees.to_file(out_gpkg, driver="GPKG")
    trees.drop(columns=trees.geometry.name).to_csv(out_csv, index=False)
    logger.info("Saved: %s | %s", out_gpkg, out_csv)
    return trees


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample multispectral index values per tree crown."
    )
    parser.add_argument("--gpkg", type=Path, required=True, help="Crown GeoPackage")
    parser.add_argument(
        "--index-dir", type=Path, required=True, help="Directory of index .tif rasters"
    )
    parser.add_argument(
        "--out-gpkg",
        type=Path,
        default=None,
        help="Output GeoPackage (default: <gpkg>_indices.gpkg)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV (default: <gpkg>_indices.csv)",
    )
    return parser


def main() -> None:
    """Console-script entry point (``crown-indices``)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_arg_parser().parse_args()
    gpkg: Path = args.gpkg
    out_gpkg = args.out_gpkg or gpkg.with_name(f"{gpkg.stem}_indices.gpkg")
    out_csv = args.out_csv or gpkg.with_name(f"{gpkg.stem}_indices.csv")
    extract_indices(gpkg, args.index_dir, out_gpkg, out_csv)


if __name__ == "__main__":
    main()
