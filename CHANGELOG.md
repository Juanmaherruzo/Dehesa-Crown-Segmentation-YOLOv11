# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-11

### Fixed
- **Inference was not thread-safe.** Every worker thread called `predict()` on a
  single shared Ultralytics `YOLO` instance, which carries mutable per-call
  state; concurrent tiles could interleave and return mixed results. Each worker
  now builds its own predictor in thread-local storage. Because that means one
  copy of the weights per worker on CUDA, the default worker count is capped at
  2 for GPU runs, which also keeps a 4 GB laptop card inside its budget.
- **Failed tiles were dropped silently.** `_process_single_tile` swallowed
  `OSError`, `ValueError` and `RuntimeError` at `debug` level and returned no
  polygons. `torch.cuda.OutOfMemoryError` subclasses `RuntimeError`, so on a
  small GPU crowns simply disappeared from the inventory with no visible signal
  and the run still reported success. Failures are now counted, logged at
  `WARNING` with an actionable hint, returned in the metadata as `tiles_failed`,
  reported in the inventory summary as a coverage percentage, and
  `crown-detect` exits non-zero when any tile was lost.
- CLI and library defaults had diverged (`conf` 0.5 vs 0.25, `nms-iou` 0.40 vs
  0.50, `flush-every` 20 vs 50), so library users got materially different
  inventories from the documented ones. The CLI now reads its defaults from the
  engine signature and a test asserts they agree.
- `.gitignore` excluded `*.pt` while `models/best.pt` was tracked, so a retrained
  model would have been silently left out of `git status`.
- Broken quickstart: the README told users to `cd` into a directory name that
  does not exist.

### Added
- `tests/test_detector.py`: regression tests for the threading and
  failure-accounting behaviour, plus CLI/library default agreement. `YOLO` is
  patched out, so they run on CI without a GPU or checkpoint.

### Changed
- The README no longer presents the 357,185-stem inventory as a validation. It
  is an inference output with no ground truth, stated as a lower bound given a
  mask recall of 70.9%, and the accuracy table is labelled as validation-split
  figures with no independent test set. Two limitations added: no independent
  test set, and no field validation of the inventory.

## [0.1.0] - 2026-07-18

### Added
- Initial release: tiled YOLOv11 tree crown detection over large orthophotos.
- `crown_detector` package: tiling, geometry (NMS / polygons), detection engine,
  inventory reporter, per-crown index sampling and a CLI.
- Console entry points: `crown-detect`, `crown-indices`.
- `pyproject.toml` packaging, CI pipeline (ruff, black, mypy, pytest) and tests.

### Changed
- Converted the single-cell detection notebook into importable, typed modules.
- Replaced hard-coded absolute paths with command-line arguments.
