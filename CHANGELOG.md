# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
