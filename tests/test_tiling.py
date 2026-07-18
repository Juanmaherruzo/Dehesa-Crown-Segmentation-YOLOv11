"""Tests for the sliding-window tile sampler."""

import pytest

from crown_detector.tiling import TilingSampler


def test_overlap_must_be_smaller_than_tile_size() -> None:
    with pytest.raises(ValueError, match="must be less than tile_size"):
        TilingSampler(1000, 1000, tile_size=100, overlap=100)


def test_tiles_cover_the_raster_with_expected_stride() -> None:
    sampler = TilingSampler(200, 200, tile_size=100, overlap=20)
    tiles = list(sampler)
    # stride = 80 -> offsets 0, 80, 160 in each axis = 3 x 3 = 9 tiles
    assert len(tiles) == 9
    assert (0, 0, 100, 100) in tiles


def test_border_tiles_are_clipped_to_the_raster() -> None:
    sampler = TilingSampler(150, 150, tile_size=100, overlap=0)
    tiles = list(sampler)
    # Second column/row start at 100 and are clipped to width 50.
    assert (100, 0, 50, 100) in tiles
    assert all(w >= 32 and h >= 32 for _, _, w, h in tiles)


def test_len_matches_grid_dimensions() -> None:
    sampler = TilingSampler(200, 200, tile_size=100, overlap=20)
    assert len(sampler) == 9
