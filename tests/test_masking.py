"""
Tests for ``masking.py`` — STL slicing into per-layer polygons and point-in-mask
filtering.

A small axis-aligned box STL (x,y in [0,10], z in [0,3]) is generated with
trimesh so the expected slice polygon at any layer inside the box is the
10x10 square (area 100).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import trimesh
from shapely.geometry import Polygon

import ampm.masking as masking
from ampm.masking import (
    apply_mask,
    apply_mask_keep,
    build_mask,
    stl_hash,
)

THICKNESS = 0.03


@pytest.fixture
def box_stl(tmp_path):
    """Box spanning x,y in [0,10], z in [0,3], exported as a binary STL."""
    box = trimesh.creation.box(extents=[10.0, 10.0, 3.0])
    box.apply_translation([5.0, 5.0, 1.5])
    path = tmp_path / "box.stl"
    box.export(path)
    return path


def points_df(rows):
    """rows: list of (x, y, layer)."""
    return pl.DataFrame(
        {
            "Demand X": pl.Series([float(r[0]) for r in rows], dtype=pl.Float32),
            "Demand Y": pl.Series([float(r[1]) for r in rows], dtype=pl.Float32),
            "layer": pl.Series([int(r[2]) for r in rows], dtype=pl.Int16),
        }
    )


@pytest.fixture
def square_mask():
    """A single layer (10) masked to the unit-ish square [0,10]^2."""
    return {10: Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])}


class TestBuildMask:
    def test_missing_stl_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_mask(tmp_path / "nope.stl", layers=[10])

    def test_no_layers_raises(self, box_stl):
        with pytest.raises(ValueError, match="No layers"):
            build_mask(box_stl, layers=[])

    def test_layers_inside_box_produce_square(self, box_stl):
        mask = build_mask(box_stl, layers=[10, 50, 90], layer_thickness=THICKNESS)
        assert set(mask) == {10, 50, 90}
        for geom in mask.values():
            assert geom.area == pytest.approx(100.0, abs=1.0)

    def test_layers_above_box_are_absent(self, box_stl):
        # layer 200 -> z = 6.0, well above the z<=3 box.
        mask = build_mask(box_stl, layers=[50, 200], layer_thickness=THICKNESS)
        assert set(mask) == {50}

    def test_positive_buffer_grows_polygon(self, box_stl):
        base = build_mask(box_stl, layers=[50], layer_thickness=THICKNESS)
        grown = build_mask(
            box_stl, layers=[50], layer_thickness=THICKNESS, buffer_mm=1.0
        )
        assert grown[50].area > base[50].area

    def test_negative_buffer_shrinks_polygon(self, box_stl):
        base = build_mask(box_stl, layers=[50], layer_thickness=THICKNESS)
        shrunk = build_mask(
            box_stl, layers=[50], layer_thickness=THICKNESS, buffer_mm=-1.0
        )
        assert shrunk[50].area < base[50].area

    def test_buffer_that_empties_geometry_drops_layer(self, box_stl):
        # Eroding a 10x10 square by 10mm removes it entirely.
        mask = build_mask(
            box_stl, layers=[50], layer_thickness=THICKNESS, buffer_mm=-10.0
        )
        assert mask == {}

    def test_cache_hit_avoids_reslice(self, box_stl, tmp_path, monkeypatch):
        cache = tmp_path / "mask.pkl"
        first = build_mask(
            box_stl, layers=[10, 50], layer_thickness=THICKNESS, cache_path=cache
        )
        assert cache.is_file()

        def boom(*a, **k):
            raise AssertionError("cache hit should not re-slice")

        monkeypatch.setattr(masking, "_slice_trimesh", boom)
        second = build_mask(
            box_stl, layers=[10, 50], layer_thickness=THICKNESS, cache_path=cache
        )
        assert set(second) == set(first)

    def test_force_ignores_cache(self, box_stl, tmp_path, monkeypatch):
        cache = tmp_path / "mask.pkl"
        build_mask(box_stl, layers=[10], layer_thickness=THICKNESS, cache_path=cache)
        monkeypatch.setattr(
            masking,
            "_slice_trimesh",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("resliced")),
        )
        with pytest.raises(AssertionError, match="resliced"):
            build_mask(
                box_stl,
                layers=[10],
                layer_thickness=THICKNESS,
                cache_path=cache,
                force=True,
            )

    def test_changed_params_invalidate_cache(self, box_stl, tmp_path, monkeypatch):
        cache = tmp_path / "mask.pkl"
        build_mask(
            box_stl,
            layers=[10],
            layer_thickness=THICKNESS,
            buffer_mm=0.0,
            cache_path=cache,
        )
        # Different buffer -> different cache key -> must re-slice.
        monkeypatch.setattr(
            masking,
            "_slice_trimesh",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("resliced")),
        )
        with pytest.raises(AssertionError, match="resliced"):
            build_mask(
                box_stl,
                layers=[10],
                layer_thickness=THICKNESS,
                buffer_mm=0.5,
                cache_path=cache,
            )

    def test_non_mesh_load_raises_typeerror(self, box_stl, monkeypatch):
        monkeypatch.setattr(
            masking.trimesh, "load_mesh", lambda *a, **k: trimesh.Scene()
        )
        with pytest.raises(TypeError):
            build_mask(box_stl, layers=[10], layer_thickness=THICKNESS)

    def test_empty_mesh_raises_valueerror(self, box_stl, monkeypatch):
        empty = trimesh.Trimesh(
            vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64)
        )
        monkeypatch.setattr(masking.trimesh, "load_mesh", lambda *a, **k: empty)
        with pytest.raises(ValueError, match="EMPTY"):
            build_mask(box_stl, layers=[10], layer_thickness=THICKNESS)


class TestApplyMask:
    def test_keeps_only_points_inside_their_layer(self, square_mask):
        df = points_df([(5, 5, 10), (15, 15, 10), (5, 5, 99)])
        out = apply_mask(df, square_mask)
        # (5,5,10) inside; (15,15,10) outside; (5,5,99) layer not in mask.
        assert out.height == 1
        assert out["Demand X"].to_list() == [5.0]

    def test_keep_array_values(self, square_mask):
        df = points_df([(5, 5, 10), (15, 15, 10)])
        keep = apply_mask_keep(df, square_mask)
        assert keep is not None
        assert keep.tolist() == [True, False]

    def test_empty_df(self, square_mask):
        df = points_df([])
        assert apply_mask_keep(df, square_mask) is None
        out = apply_mask(df, square_mask)
        assert out.height == 0

    def test_missing_column_raises(self, square_mask):
        df = pl.DataFrame({"Demand X": [1.0], "Demand Y": [2.0]})  # no 'layer'
        with pytest.raises(KeyError, match="layer"):
            apply_mask_keep(df, square_mask)

    def test_chunking_matches_single_pass(self, square_mask):
        df = points_df([(5, 5, 10), (15, 15, 10), (1, 1, 10), (9, 9, 10), (20, 0, 10)])
        full = apply_mask_keep(df, square_mask, chunk_rows=8_000_000)
        chunked = apply_mask_keep(df, square_mask, chunk_rows=2)
        assert full is not None and chunked is not None
        assert np.array_equal(full, chunked)
        assert full.tolist() == [True, False, True, True, False]

    def test_layer_not_in_mask_is_dropped(self, square_mask):
        df = points_df([(5, 5, 11), (5, 5, 12)])  # neither layer present in mask
        assert apply_mask(df, square_mask).height == 0

    def test_end_to_end_build_then_apply(self, box_stl):
        mask = build_mask(box_stl, layers=[10], layer_thickness=THICKNESS)
        df = points_df([(5, 5, 10), (12, 12, 10), (0.5, 0.5, 10)])
        out = apply_mask(df, mask)
        kept = sorted(out["Demand X"].to_list())
        assert kept == pytest.approx([0.5, 5.0])


class TestHashingAndCache:
    def test_stl_hash_is_deterministic(self, box_stl):
        assert stl_hash(box_stl) == stl_hash(box_stl)

    def test_stl_hash_changes_with_content(self, tmp_path):
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"x" * 256)
        b.write_bytes(b"y" * 256)
        assert stl_hash(a) != stl_hash(b)

    def test_cache_key_varies_with_each_input(self, box_stl):
        p = Path(box_stl)
        base = masking._cache_key(p, [10], THICKNESS, 0.0)
        assert base != masking._cache_key(p, [10, 11], THICKNESS, 0.0)
        assert base != masking._cache_key(p, [10], 0.05, 0.0)
        assert base != masking._cache_key(p, [10], THICKNESS, 1.0)

    def test_save_load_cache_roundtrip(self, tmp_path):
        path = tmp_path / "c.pkl"
        masking._save_cache(path, {"key": "abc", "mask": {1: "geom"}})
        loaded = masking._load_cache(path)
        assert loaded is not None
        assert loaded["key"] == "abc"

    def test_load_cache_corrupt_returns_none(self, tmp_path):
        path = tmp_path / "c.pkl"
        path.write_bytes(b"not a pickle at all")
        assert masking._load_cache(path) is None

    def test_load_cache_missing_returns_none(self, tmp_path):
        assert masking._load_cache(tmp_path / "missing.pkl") is None
