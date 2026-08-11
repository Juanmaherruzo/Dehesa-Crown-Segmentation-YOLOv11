"""Parallel tile-based tree crown detector using a YOLOv11 segmentation model."""

import logging
import multiprocessing
import threading
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

    Threading. An Ultralytics ``YOLO`` predictor carries mutable per-call state,
    so a single instance shared across worker threads can interleave and return
    mixed results. Each worker therefore builds its own model, held in
    thread-local storage. On CUDA that means ``n_workers`` copies of the weights
    resident at once, so the default worker count is capped for GPU runs — see
    :meth:`_default_workers`.

    Failures. A tile that raises is counted, not silently dropped: the count is
    logged at ``WARNING`` and returned in the metadata as ``tiles_failed`` so a
    caller can tell a complete inventory from a partial one.
    """

    # Concurrent CUDA inferences each hold a full copy of the weights. More than
    # this on one GPU is how a 4 GB laptop card runs out of VRAM.
    MAX_CUDA_WORKERS = 2

    def __init__(
        self,
        model_path: Path,
        tile_size: int = 960,
        overlap: int = 200,
        conf_threshold: float = 0.50,
        nms_iou_thresh: float = 0.40,
        min_diameter_m: float = 3.0,
        n_workers: int = -1,
        flush_every: int = 20,
    ) -> None:
        self.model_path = Path(model_path)
        self.tile_size = tile_size
        self.overlap = overlap
        self.conf_threshold = conf_threshold
        self.nms_iou_thresh = nms_iou_thresh
        self.min_diameter_m = min_diameter_m
        self.flush_every = flush_every

        self.device = "cuda:0" if CUDA_AVAILABLE else "cpu"
        self.n_workers = self._default_workers(n_workers)
        logger.info("Inference device: %s | workers: %d", self.device, self.n_workers)

        # One predictor per worker thread. Ultralytics models are not safe to
        # call concurrently from several threads on a single instance.
        self._local = threading.local()
        self._tiles_failed = 0
        self._fail_lock = Lock()

        logger.info("Loading YOLO model: %s...", self.model_path.name)
        # Load once up front so a bad path fails here rather than inside a worker.
        self._build_model()

    def _default_workers(self, requested: int) -> int:
        """Resolve the worker count, capping GPU runs to bound VRAM use."""
        resolved = (
            max(1, multiprocessing.cpu_count())
            if requested == -1
            else max(1, requested)
        )
        if self.device.startswith("cuda") and resolved > self.MAX_CUDA_WORKERS:
            logger.info(
                "Capping workers %d -> %d: each CUDA worker holds its own copy of "
                "the weights. Pass --workers explicitly to override.",
                resolved,
                self.MAX_CUDA_WORKERS,
            )
            return self.MAX_CUDA_WORKERS
        return resolved

    def _build_model(self) -> Any:
        """Return this thread's model, constructing it on first use."""
        model = getattr(self._local, "model", None)
        if model is None:
            # ultralytics ships incomplete type info; treat the model as untyped.
            model = YOLO(str(self.model_path))
            self._local.model = model
        return model

    @property
    def model(self) -> Any:
        """The calling thread's YOLO predictor."""
        return self._build_model()

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
        """Read, infer and convert one tile to polygons; never raises.

        A failure here is real data loss — the crowns in that tile are missing
        from the inventory — so it is counted and logged at ``WARNING`` rather
        than swallowed. CUDA out-of-memory is a ``RuntimeError`` subclass and is
        singled out, because on a small GPU it is the likely cause and the fix
        (fewer workers, smaller tiles) is actionable.
        """
        try:
            with rasterio.open(image_path) as src:
                tile_img = self._read_tile(src, col_off, row_off, tile_w, tile_h)
            masks = self._infer_tile(tile_img)
            if masks is None:
                return []
            return geometry.masks_to_polygons(masks, col_off, row_off, affine_transform)
        except (OSError, ValueError, RuntimeError) as exc:
            with self._fail_lock:
                self._tiles_failed += 1
                failed_so_far = self._tiles_failed
            oom = CUDA_AVAILABLE and isinstance(exc, torch.cuda.OutOfMemoryError)
            hint = (
                " (CUDA out of memory — retry with fewer --workers "
                "or a smaller --tile-size)"
                if oom
                else ""
            )
            # Log the first few in full, then only every 50th, to keep a badly
            # broken run readable without hiding the failure.
            if failed_so_far <= 5 or failed_so_far % 50 == 0:
                logger.warning(
                    "Tile (%d,%d) failed [%d so far]: %s%s",
                    col_off,
                    row_off,
                    failed_so_far,
                    exc,
                    hint,
                )
            return []

    def detect(self, image_path: Path) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
        """Run the full detection pipeline on a GeoTIFF orthophoto."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        logger.info("Opening raster: %s", path.name)
        with self._fail_lock:
            self._tiles_failed = 0
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
                # Filled in after the tile loop; a non-zero value means the
                # inventory below is incomplete.
                "n_tiles": 0,
                "tiles_failed": 0,
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

            raster_meta["n_tiles"] = n_tiles
            raster_meta["tiles_failed"] = self._tiles_failed
            if self._tiles_failed:
                logger.warning(
                    "INCOMPLETE INVENTORY: %d of %d tiles (%.1f%%) failed and their "
                    "crowns are missing from the result. Re-run with fewer --workers "
                    "or a smaller --tile-size before quoting these figures.",
                    self._tiles_failed,
                    n_tiles,
                    100.0 * self._tiles_failed / n_tiles,
                )
            else:
                logger.info("All %d tiles processed successfully.", n_tiles)

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
