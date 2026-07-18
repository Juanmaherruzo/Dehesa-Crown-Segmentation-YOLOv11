"""Parallel tile-based tree crown detector using a YOLOv11 segmentation model."""

import logging
import multiprocessing
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.windows import Window
from shapely.geometry import Polygon
from ultralytics import YOLO  # type: ignore[attr-defined]  # not re-exported in __all__

from crown_detector import geometry
from crown_detector.tiling import TilingSampler

try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:  # pragma: no cover - torch is expected in practice
    CUDA_AVAILABLE = False
    warnings.warn("torch not found. Inference will run on CPU.", stacklevel=2)

logger = logging.getLogger(__name__)


class CrownDetectionEngine:
    """Detect tree crowns over a large orthophoto, tile by tile.

    The raster is read out-of-core via rasterio windows and inferred in a thread
    pool (I/O-bound reads + GPU inference favour threads over processes). Masks
    are converted to georeferenced polygons, de-duplicated with spatial NMS and
    filtered by minimum crown diameter.
    """

    def __init__(
        self,
        model_path: Path,
        tile_size: int = 960,
        overlap: int = 200,
        conf_threshold: float = 0.25,
        nms_iou_thresh: float = 0.50,
        min_diameter_m: float = 3.0,
        n_workers: int = -1,
        flush_every: int = 50,
    ) -> None:
        self.model_path = Path(model_path)
        self.tile_size = tile_size
        self.overlap = overlap
        self.conf_threshold = conf_threshold
        self.nms_iou_thresh = nms_iou_thresh
        self.min_diameter_m = min_diameter_m
        self.n_workers = (
            max(1, multiprocessing.cpu_count())
            if n_workers == -1
            else max(1, n_workers)
        )
        self.flush_every = flush_every

        self.device = "cuda:0" if CUDA_AVAILABLE else "cpu"
        logger.info("Inference device: %s", self.device)
        logger.info("Loading YOLO model: %s...", self.model_path.name)
        # ultralytics ships incomplete type info; treat the model as untyped.
        self.model: Any = YOLO(str(self.model_path))

    def _read_tile(
        self,
        src: rasterio.DatasetReader,
        col_off: int,
        row_off: int,
        tile_w: int,
        tile_h: int,
    ) -> np.ndarray:
        """Read RGB bands for a tile window as an HWC uint8 array."""
        window = Window(col_off, row_off, tile_w, tile_h)
        return np.moveaxis(src.read([1, 2, 3], window=window), 0, -1)

    def _infer_tile(self, tile_img: np.ndarray) -> list[np.ndarray] | None:
        """Run YOLO segmentation on one tile; return pixel-space mask arrays."""
        results = self.model.predict(
            tile_img,
            imgsz=self.tile_size,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        if results[0].masks is None:
            return None
        return [mask for mask in results[0].masks.xy if len(mask) >= 3]

    def _process_single_tile(
        self,
        image_path: Path,
        col_off: int,
        row_off: int,
        tile_w: int,
        tile_h: int,
        affine_transform: Affine,
    ) -> list[Polygon]:
        """Read, infer and convert one tile to polygons; never raises."""
        try:
            with rasterio.open(image_path) as src:
                tile_img = self._read_tile(src, col_off, row_off, tile_w, tile_h)
            masks = self._infer_tile(tile_img)
            if masks is None:
                return []
            return geometry.masks_to_polygons(masks, col_off, row_off, affine_transform)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.debug("Tile (%d,%d) failed: %s", col_off, row_off, exc)
            return []

    def detect(self, image_path: Path) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
        """Run the full detection pipeline on a GeoTIFF orthophoto."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        logger.info("Opening raster: %s", path.name)
        with rasterio.open(str(path)) as src:
            affine = src.transform
            crs = src.crs
            width = src.width
            height = src.height
            res_x, res_y = src.res

            # Valid-pixel area from a subsampled data mask (not the bounding box).
            scale = 10
            data_mask = src.dataset_mask(
                out_shape=(height // scale + 1, width // scale + 1)
            )
            valid_frac = (data_mask > 0).mean()
            total_area_ha = width * height * abs(res_x * res_y) * valid_frac / 10000

            raster_meta: dict[str, object] = {
                "width": width,
                "height": height,
                "res_m": abs(res_x),
                "crs": crs,
                "total_area_ha": total_area_ha,
            }
            logger.info(
                "Raster: %dx%d px | GSD=%.2f cm/px | valid area=%.4f ha",
                width,
                height,
                abs(res_x) * 100,
                total_area_ha,
            )

            all_tiles = list(TilingSampler(width, height, self.tile_size, self.overlap))
            n_tiles = len(all_tiles)
            logger.info(
                "Tiling: %d tiles | tile_size=%d | overlap=%d | workers=%d",
                n_tiles,
                self.tile_size,
                self.overlap,
                self.n_workers,
            )

            polygon_batches: list[gpd.GeoDataFrame] = []
            current_batch: list[Polygon] = []
            batch_lock = Lock()
            tiles_done = 0

            def process_tile_task(
                tile_args: tuple[int, int, int, int],
            ) -> list[Polygon]:
                col_off, row_off, tile_w, tile_h = tile_args
                return self._process_single_tile(
                    path, col_off, row_off, tile_w, tile_h, affine
                )

            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(process_tile_task, tile): tile for tile in all_tiles
                }
                for future in as_completed(futures):
                    polys = future.result()
                    tiles_done += 1
                    with batch_lock:
                        current_batch.extend(polys)
                        if current_batch and tiles_done % self.flush_every == 0:
                            polygon_batches.append(
                                gpd.GeoDataFrame({"geometry": current_batch}, crs=crs)
                            )
                            current_batch = []
                    if tiles_done % 100 == 0 or tiles_done == n_tiles:
                        logger.info("  Tiles: %d/%d", tiles_done, n_tiles)

                if current_batch:
                    polygon_batches.append(
                        gpd.GeoDataFrame({"geometry": current_batch}, crs=crs)
                    )

        if not polygon_batches:
            logger.warning("No tree crowns detected.")
            return gpd.GeoDataFrame({"geometry": []}, crs=crs), raster_meta

        gdf_raw = gpd.GeoDataFrame(
            pd.concat(polygon_batches, ignore_index=True), geometry="geometry", crs=crs
        )
        logger.info("Total raw detections: %d", len(gdf_raw))

        gdf_clean = geometry.apply_spatial_nms(gdf_raw, self.nms_iou_thresh)
        gdf_clean = geometry.add_diameter_columns(gdf_clean, self.min_diameter_m)
        logger.info(
            "After diameter filter (>= %.1fm): %d crowns retained",
            self.min_diameter_m,
            len(gdf_clean),
        )
        return gdf_clean, raster_meta
