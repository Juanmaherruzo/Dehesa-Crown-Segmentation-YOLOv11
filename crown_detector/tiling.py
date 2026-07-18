"""Sliding-window tile sampler — pure geometric window arithmetic, no I/O."""

from collections.abc import Iterator

# Tiles smaller than this (pixels) are skipped as too small for inference.
_MIN_TILE_PX = 32


class TilingSampler:
    """Generate ``(col_off, row_off, width, height)`` tile descriptors for a raster.

    Uses a sliding window with configurable overlap. No image data is read here.
    """

    def __init__(
        self,
        raster_width: int,
        raster_height: int,
        tile_size: int,
        overlap: int,
    ) -> None:
        if overlap >= tile_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than tile_size ({tile_size})."
            )
        self.raster_width = raster_width
        self.raster_height = raster_height
        self.tile_size = tile_size
        self.stride = tile_size - overlap  # effective step between tiles

    def __iter__(self) -> Iterator[tuple[int, int, int, int]]:
        """Yield ``(col_off, row_off, tile_w, tile_h)`` for every valid tile."""
        for row_off in range(0, self.raster_height, self.stride):
            for col_off in range(0, self.raster_width, self.stride):
                tile_w = min(self.tile_size, self.raster_width - col_off)
                tile_h = min(self.tile_size, self.raster_height - row_off)
                if tile_w >= _MIN_TILE_PX and tile_h >= _MIN_TILE_PX:
                    yield col_off, row_off, tile_w, tile_h

    def __len__(self) -> int:
        """Total number of tiles along the grid (for progress reporting)."""
        cols = len(range(0, self.raster_width, self.stride))
        rows = len(range(0, self.raster_height, self.stride))
        return cols * rows
