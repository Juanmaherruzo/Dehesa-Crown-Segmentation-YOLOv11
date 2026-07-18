"""Command-line entry point for the crown detection pipeline."""

import argparse
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
    parser.add_argument("--tile-size", type=int, default=960)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.40)
    parser.add_argument("--min-diameter", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--flush-every", type=int, default=20)
    return parser


def main() -> None:
    """Console-script entry point (``crown-detect``)."""
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


if __name__ == "__main__":
    main()
