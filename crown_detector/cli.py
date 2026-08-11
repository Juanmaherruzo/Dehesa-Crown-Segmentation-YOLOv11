"""Command-line entry point for the crown detection pipeline."""

import argparse
import inspect
import logging
from pathlib import Path

from crown_detector.detector import CrownDetectionEngine
from crown_detector.reporter import InventoryReporter


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect tree crowns in an orthophoto with a YOLOv11 model."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input GeoTIFF")
    parser.add_argument(
        "--model", type=Path, default=Path("models/best.pt"), help="YOLOv11 weights"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output GeoPackage (.gpkg)"
    )
    # Defaults are taken from the engine so the CLI and the library cannot drift
    # apart and hand users different inventories from the same orthophoto.
    defaults = inspect.signature(CrownDetectionEngine.__init__).parameters
    parser.add_argument("--tile-size", type=int, default=defaults["tile_size"].default)
    parser.add_argument("--overlap", type=int, default=defaults["overlap"].default)
    parser.add_argument(
        "--conf", type=float, default=defaults["conf_threshold"].default
    )
    parser.add_argument(
        "--nms-iou", type=float, default=defaults["nms_iou_thresh"].default
    )
    parser.add_argument(
        "--min-diameter", type=float, default=defaults["min_diameter_m"].default
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=defaults["n_workers"].default,
        help=(
            "-1 uses one worker per CPU core, capped on CUDA because each worker "
            "holds its own copy of the weights."
        ),
    )
    parser.add_argument(
        "--flush-every", type=int, default=defaults["flush_every"].default
    )
    return parser


def main() -> int:
    """Console-script entry point (``crown-detect``).

    Returns:
        ``0`` on a complete run, ``1`` if any tile failed. A partial inventory
        should not look like a success to a calling script.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_arg_parser().parse_args()

    engine = CrownDetectionEngine(
        model_path=args.model,
        tile_size=args.tile_size,
        overlap=args.overlap,
        conf_threshold=args.conf,
        nms_iou_thresh=args.nms_iou,
        min_diameter_m=args.min_diameter,
        n_workers=args.workers,
        flush_every=args.flush_every,
    )
    gdf_crowns, meta = engine.detect(args.image)
    InventoryReporter.compute_statistics(gdf_crowns, meta, args.output)
    return 1 if meta.get("tiles_failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
