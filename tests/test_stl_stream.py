"""
Tests for ``stl_stream.py`` (bounded-memory binary-STL slicer).

Coverage spans the pure-geometry helpers (header validation, batch splitting,
ring stitching, polygonization) and an end-to-end slice of a real binary box
STL, whose interior slices are the 10x10 square.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import trimesh
from shapely.geometry import MultiPolygon, Polygon

from ampm.stl_stream import (
    _batch_bounds,
    _polygonize_layer,
    _read_binary_stl_header,
    _stitch_rings,
    slice_stl_streaming,
)

THICKNESS = 0.03


@pytest.fixture
def box_stl(tmp_path):
    """Binary STL box: x,y in [0,10], z in [0,3]."""
    box = trimesh.creation.box(extents=[10.0, 10.0, 3.0])
    box.apply_translation([5.0, 5.0, 1.5])
    path = tmp_path / "box.stl"
    box.export(path)  # trimesh writes binary STL by default
    return path


def square_segments():
    """Four CCW directed segments forming a [0,10]^2 square, [x1,y1,x2,y2]."""
    return np.array(
        [
            [0.0, 0.0, 10.0, 0.0],
            [10.0, 0.0, 10.0, 10.0],
            [10.0, 10.0, 0.0, 10.0],
            [0.0, 10.0, 0.0, 0.0],
        ]
    )


class TestHeader:
    def test_valid_header_returns_triangle_count(self, box_stl):
        n = _read_binary_stl_header(box_stl)
        assert n == 12  # a box is 12 triangles

    def test_too_small_raises(self, tmp_path):
        p = tmp_path / "tiny.stl"
        p.write_bytes(b"\x00" * 10)
        with pytest.raises(ValueError, match="too small"):
            _read_binary_stl_header(p)

    def test_ascii_stl_rejected(self, tmp_path):
        p = tmp_path / "ascii.stl"
        # Contains no null bytes -> looks ASCII.
        p.write_bytes(b"solid mything" + b" " * 80)
        with pytest.raises(ValueError, match="binary STL"):
            _read_binary_stl_header(p)

    def test_size_inconsistent_raises(self, tmp_path):
        p = tmp_path / "bad.stl"
        # 80-byte header + count says 5 triangles, but no triangle data follows.
        p.write_bytes(b"\x00" * 80 + struct.pack("<I", 5))
        with pytest.raises(ValueError, match="but file is"):
            _read_binary_stl_header(p)

    def test_zero_triangles_raises(self, tmp_path):
        p = tmp_path / "empty.stl"
        # Header + count=0, file is exactly 84 bytes -> size check passes, count fails.
        p.write_bytes(b"\x00" * 80 + struct.pack("<I", 0))
        with pytest.raises(ValueError, match="zero triangles"):
            _read_binary_stl_header(p)


class TestBatchBounds:
    def test_covers_full_range_contiguously(self):
        cum = np.array([2, 4, 7, 10])
        bounds = _batch_bounds(cum, cap=5)
        assert bounds[0][0] == 0
        assert bounds[-1][1] == 4
        # contiguous, non-overlapping
        for (a, b), (c, _) in zip(bounds, bounds[1:]):
            assert b == c

    def test_single_oversized_triangle_gets_its_own_slice(self):
        # First triangle alone expands to 100 rows, far above the cap.
        cum = np.array([100, 101, 102])
        bounds = _batch_bounds(cum, cap=10)
        assert bounds[0] == (0, 1)  # forced progress despite exceeding cap

    def test_cap_respected_when_possible(self):
        cum = np.array([1, 2, 3, 4, 5, 6])
        bounds = _batch_bounds(cum, cap=2)
        # Each slice should expand to <= cap rows (here 2 triangles each).
        for a, b in bounds:
            base = cum[a - 1] if a > 0 else 0
            assert cum[b - 1] - base <= 2 or b - a == 1


class TestStitchRings:
    def test_closed_square_makes_one_ring(self):
        rings = _stitch_rings(square_segments())
        assert len(rings) == 1
        assert rings[0].shape == (4, 2)

    def test_open_chain_discarded(self):
        # Three segments that never close back to the start.
        segs = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 2.0, 0.0],
                [2.0, 0.0, 3.0, 0.0],
            ]
        )
        assert _stitch_rings(segs) == []


class TestPolygonize:
    def test_square_becomes_polygon_area_100(self):
        geom = _polygonize_layer(square_segments())
        assert isinstance(geom, (Polygon, MultiPolygon))
        assert geom.area == pytest.approx(100.0)

    def test_no_rings_returns_none(self):
        segs = np.array([[0.0, 0.0, 1.0, 0.0]])  # single open segment
        assert _polygonize_layer(segs) is None

    def test_square_with_hole(self):
        # Outer CCW 10x10 square + inner CW 2x2 hole around (4..6).
        outer = square_segments()
        inner = np.array(
            [
                [4.0, 4.0, 4.0, 6.0],
                [4.0, 6.0, 6.0, 6.0],
                [6.0, 6.0, 6.0, 4.0],
                [6.0, 4.0, 4.0, 4.0],
            ]
        )
        geom = _polygonize_layer(np.vstack([outer, inner]))
        assert geom is not None
        assert geom.area == pytest.approx(100.0 - 4.0)


class TestSliceStreaming:
    def test_no_layers_raises(self, box_stl):
        with pytest.raises(ValueError, match="No layers"):
            slice_stl_streaming(box_stl, layers=[], layer_thickness=THICKNESS)

    def test_ascii_input_rejected(self, tmp_path):
        p = tmp_path / "ascii.stl"
        p.write_bytes(b"solid x" + b" " * 100)
        with pytest.raises(ValueError, match="binary STL"):
            slice_stl_streaming(p, layers=[10], layer_thickness=THICKNESS)

    def test_interior_layers_produce_squares(self, box_stl):
        mask = slice_stl_streaming(
            box_stl, layers=[10, 50, 90], layer_thickness=THICKNESS, verbose=False
        )
        assert set(mask) == {10, 50, 90}
        for geom in mask.values():
            assert isinstance(geom, (Polygon, MultiPolygon))
            assert geom.area == pytest.approx(100.0, abs=1.0)

    def test_layers_above_box_absent(self, box_stl):
        mask = slice_stl_streaming(
            box_stl, layers=[50, 200], layer_thickness=THICKNESS, verbose=False
        )
        assert set(mask) == {50}

    def test_chunking_matches_single_pass(self, box_stl):
        big = slice_stl_streaming(
            box_stl,
            layers=[50],
            layer_thickness=THICKNESS,
            chunk_triangles=10_000,
            verbose=False,
        )
        small = slice_stl_streaming(
            box_stl,
            layers=[50],
            layer_thickness=THICKNESS,
            chunk_triangles=2,
            verbose=False,
        )
        assert big[50].area == pytest.approx(small[50].area)

    def test_matches_trimesh_path(self, box_stl):
        import ampm.masking

        trimesh_mask = ampm.masking._slice_trimesh(box_stl, [50], THICKNESS)
        stream_mask = slice_stl_streaming(box_stl, [50], THICKNESS, verbose=False)
        assert stream_mask[50].area == pytest.approx(trimesh_mask[50].area, abs=1.0)
