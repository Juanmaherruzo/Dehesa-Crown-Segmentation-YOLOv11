"""Tests for the detector's threading and failure-accounting behaviour.

These are regression tests for two defects that made a run report success while
silently returning an incomplete inventory:

- every worker thread shared one Ultralytics predictor, which is not safe to
  call concurrently;
- a tile that raised was swallowed at ``debug`` level, so a CUDA out-of-memory
  on a small GPU removed crowns from the result with no visible signal.

They exercise the engine without loading real weights: ``YOLO`` is patched out,
so the tests stay fast and run on CI without a GPU or a checkpoint.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, ClassVar

import pytest

from crown_detector import detector as detector_module
from crown_detector.detector import CrownDetectionEngine


class _FakeYOLO:
    """Stand-in for ultralytics.YOLO that records which thread built it."""

    instances: ClassVar[list[_FakeYOLO]] = []
    lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, path: str) -> None:
        self.path = path
        self.built_in_thread = threading.get_ident()
        with _FakeYOLO.lock:
            _FakeYOLO.instances.append(self)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("predict() should not be reached in these tests")


@pytest.fixture(autouse=True)
def _patch_yolo(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeYOLO.instances = []
    monkeypatch.setattr(detector_module, "YOLO", _FakeYOLO)


def _engine(**kwargs: Any) -> CrownDetectionEngine:
    return CrownDetectionEngine(model_path=Path("models/best.pt"), **kwargs)


def test_each_thread_builds_its_own_model() -> None:
    """A shared Ultralytics predictor can interleave; one per thread cannot."""
    engine = _engine(n_workers=2)
    main_model = engine.model
    assert engine.model is main_model  # cached per thread

    from_worker: list[object] = []

    def grab() -> None:
        from_worker.append(engine.model)

    threads = [threading.Thread(target=grab) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(from_worker) == 3
    for model in from_worker:
        assert model is not main_model
    # Three worker threads plus the one built in __init__.
    assert len({id(m) for m in from_worker}) == 3


def test_failed_tiles_are_counted_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising tile increments the counter instead of vanishing."""
    engine = _engine(n_workers=1)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(engine, "_read_tile", boom)

    result = engine._process_single_tile(
        Path("nonexistent.tif"), 0, 0, 64, 64, detector_module.Affine.identity()
    )
    assert result == []
    assert engine._tiles_failed == 1


def _tiny_raster(path: Path) -> Path:
    """Write a 200x200 three-band GeoTIFF with a real affine transform.

    Deliberately CRS-less. The tiling and failure-accounting logic under test is
    independent of the projection, and requiring an EPSG lookup would make the
    suite depend on whichever PROJ database happens to be first on PATH.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    profile = {
        "driver": "GTiff",
        "width": 200,
        "height": 200,
        "count": 3,
        "dtype": "uint8",
        "transform": from_origin(300000.0, 4100000.0, 0.25, 0.25),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((3, 200, 200), 128, dtype=np.uint8))
    return path


def test_failure_counter_is_reset_between_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counts must not leak from one orthophoto into the next."""
    raster = _tiny_raster(tmp_path / "ortho.tif")
    engine = _engine(n_workers=1, tile_size=100, overlap=0)
    engine._tiles_failed = 7  # residue from an earlier run

    # Every tile fails, so the count must equal the tile total, not 7 + total.
    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated tile failure")

    monkeypatch.setattr(engine, "_read_tile", boom)
    gdf, meta = engine.detect(raster)

    assert meta["n_tiles"] == 4  # 200x200 raster, 100px tiles, no overlap
    assert meta["tiles_failed"] == 4
    assert engine._tiles_failed == 4
    assert len(gdf) == 0


def test_metadata_reports_a_clean_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no failures the metadata must say so explicitly."""
    raster = _tiny_raster(tmp_path / "ortho.tif")
    engine = _engine(n_workers=1, tile_size=100, overlap=0)
    monkeypatch.setattr(engine, "_infer_tile", lambda tile: None)

    _, meta = engine.detect(raster)
    assert meta["n_tiles"] == 4
    assert meta["tiles_failed"] == 0


def test_cuda_worker_count_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unbounded concurrent CUDA inference is what exhausts a 4 GB GPU."""
    monkeypatch.setattr(detector_module, "CUDA_AVAILABLE", True)
    engine = _engine(n_workers=32)
    assert engine.device.startswith("cuda")
    assert engine.n_workers == CrownDetectionEngine.MAX_CUDA_WORKERS


def test_cpu_worker_count_is_not_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap is about VRAM, so it must not apply on CPU."""
    monkeypatch.setattr(detector_module, "CUDA_AVAILABLE", False)
    engine = _engine(n_workers=8)
    assert engine.device == "cpu"
    assert engine.n_workers == 8


def test_cli_and_library_defaults_agree() -> None:
    """A library user must get the inventory the README documents."""
    from crown_detector.cli import _build_arg_parser

    defaults = {a.dest: a.default for a in _build_arg_parser()._actions}
    engine = _engine()
    assert engine.conf_threshold == defaults["conf"]
    assert engine.nms_iou_thresh == defaults["nms_iou"]
    assert engine.min_diameter_m == defaults["min_diameter"]
    assert engine.tile_size == defaults["tile_size"]
    assert engine.overlap == defaults["overlap"]
    assert engine.flush_every == defaults["flush_every"]
