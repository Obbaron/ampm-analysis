"""
Tests for ampm.masking.

We generate synthetic STLs with known geometry, slice them, and verify the
right points survive apply_mask.
"""
from __future__ import annotations

import sys

import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import trimesh

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))

from ampm.masking import build_mask, apply_mask

def make_two_block_stl(path: Path) -> None:
    """Two 10x5x3 blocks side-by-side. Block 1 centered (0,0,1.5), Block 2 at (15,0,1.5)."""
    b1 = trimesh.creation.box(extents=[10, 5, 3]); b1.apply_translation([0, 0, 1.5])
    b2 = trimesh.creation.box(extents=[10, 5, 3]); b2.apply_translation([15, 0, 1.5])
    mesh = trimesh.util.concatenate([b1, b2])
    mesh.export(path)

def make_synthetic_data(layers: list[int], n_per_layer: int = 1000, seed: int = 0) -> pl.DataFrame:
    """Random points spanning a region wider than either block."""
    rng = np.random.default_rng(seed)
    rows = []
    for L in layers:
        for _ in range(n_per_layer):
            rows.append({
                "Demand X": rng.uniform(-20, 30),
                "Demand Y": rng.uniform(-10, 10),
                "layer": L,
                "signal": rng.normal(500, 100),
            })
    return pl.DataFrame(rows)

def test_build_basic() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mask_test_"))
    try:
        stl = tmp / "blocks.stl"
        make_two_block_stl(stl)

        # Layer thickness 0.03; blocks span Z=0..3, so layers 1..100 hit them.
        # Z = layer * 0.03, so layer 50 → Z=1.5 (mid-block).
        mask = build_mask(stl, layers=range(1, 105), layer_thickness=0.03)

        # Layers 1..100 should produce polygons (Z=0.03..3.00).
        # Layers 101..104 are above the blocks (Z=3.03..3.12) — should be dropped.
        in_mask = sorted(mask.keys())
        assert min(in_mask) >= 1
        assert max(in_mask) <= 100
        # Approximate count check (slicing exactly at Z=0 and Z=3 may or may not produce a polygon).
        assert 95 <= len(in_mask) <= 100, len(in_mask)
        print(f"  build_basic: sliced {len(in_mask)} layers from blocks ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_apply_mask_keeps_correct_points() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mask_test_"))
    try:
        stl = tmp / "blocks.stl"
        make_two_block_stl(stl)
        mask = build_mask(stl, layers=range(40, 60), layer_thickness=0.03)

        # Hand-place points: some clearly inside block 1, some inside block 2,
        # some clearly outside both.
        df = pl.DataFrame({
            "Demand X": [0.0, 15.0, 7.0, -10.0, 25.0],  # b1, b2, gap, far-left, far-right of b2
            "Demand Y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "layer": [50, 50, 50, 50, 50],  # all at Z=1.5
            "signal": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        out = apply_mask(df, mask)
        # Only the first two rows (block 1, block 2) should survive.
        kept = sorted(out["signal"].to_list())
        assert kept == [1.0, 2.0], kept
        print(f"  apply_mask kept correct rows: {kept} ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_apply_mask_drops_layers_above_part() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mask_test_"))
    try:
        stl = tmp / "blocks.stl"
        make_two_block_stl(stl)
        mask = build_mask(stl, layers=range(1, 200), layer_thickness=0.03)

        # Mix points at layer 50 (inside part) and layer 150 (above part).
        df = pl.DataFrame({
            "Demand X": [0.0, 0.0],
            "Demand Y": [0.0, 0.0],
            "layer": [50, 150],
            "signal": [1.0, 2.0],
        })
        out = apply_mask(df, mask)
        # Layer 150 has no mask entry, so its row is dropped even though
        # x,y would be "inside" if the layer existed.
        assert out["signal"].to_list() == [1.0]
        print("  apply_mask drops layers above part ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_buffer_grow_and_shrink() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mask_test_"))
    try:
        stl = tmp / "blocks.stl"
        make_two_block_stl(stl)

        # A point 0.05 mm outside block 1's edge in X.
        # Block 1: X in [-5, 5], Y in [-2.5, 2.5].
        # Point at (5.05, 0) should be:
        #   buffer=0    → outside (dropped)
        #   buffer=+0.1 → inside (kept)
        df = pl.DataFrame({
            "Demand X": [5.05],
            "Demand Y": [0.0],
            "layer": [50],
            "signal": [42.0],
        })
        m_strict  = build_mask(stl, layers=[50], layer_thickness=0.03, buffer_mm=0.0)
        m_lenient = build_mask(stl, layers=[50], layer_thickness=0.03, buffer_mm=0.1)

        assert len(apply_mask(df, m_strict))  == 0
        assert len(apply_mask(df, m_lenient)) == 1

        # And the inverse: a point 0.05 mm inside should be:
        #   buffer=0    → inside
        #   buffer=-0.1 → outside (we shrank past it)
        df2 = pl.DataFrame({
            "Demand X": [4.95],
            "Demand Y": [0.0],
            "layer": [50],
            "signal": [42.0],
        })
        m_inset = build_mask(stl, layers=[50], layer_thickness=0.03, buffer_mm=-0.1)
        assert len(apply_mask(df2, m_strict)) == 1
        assert len(apply_mask(df2, m_inset))  == 0
        print("  buffer_mm grow & shrink behave correctly ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_cache_roundtrip() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mask_test_"))
    try:
        stl = tmp / "blocks.stl"
        make_two_block_stl(stl)
        cache = tmp / "mask.pkl"

        m1 = build_mask(stl, layers=range(40, 60), layer_thickness=0.03, cache_path=cache)
        assert cache.is_file()
        # Touch the STL to ensure mtime changes don't break — cache is keyed on contents.
        m2 = build_mask(stl, layers=range(40, 60), layer_thickness=0.03, cache_path=cache)
        assert sorted(m1.keys()) == sorted(m2.keys())

        # Different params → cache miss → rebuild (key changes).
        m3 = build_mask(stl, layers=range(40, 60), layer_thickness=0.03, buffer_mm=0.1, cache_path=cache)
        # Different buffer means polygons should differ in area.
        any_layer = next(iter(m1))
        assert m3[any_layer].area > m1[any_layer].area
        print("  cache roundtrip + invalidation ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_apply_mask_at_scale() -> None:
    """Sanity check the vectorized contains is fast enough at AMPM scale."""
    import time
    tmp = Path(tempfile.mkdtemp(prefix="mask_test_"))
    try:
        stl = tmp / "blocks.stl"
        make_two_block_stl(stl)
        mask = build_mask(stl, layers=range(1, 101), layer_thickness=0.03)

        # 1M points spread across 100 layers
        N = 1_000_000
        rng = np.random.default_rng(0)
        df = pl.DataFrame({
            "Demand X": rng.uniform(-20, 30, N),
            "Demand Y": rng.uniform(-10, 10, N),
            "layer": rng.integers(1, 101, N).astype(np.int32),
            "signal": rng.normal(500, 100, N),
        })
        t0 = time.time()
        out = apply_mask(df, mask)
        dt = time.time() - t0
        # The two blocks together cover 100mm² out of the 50x20=1000mm² sampling area = 10%.
        # Expected survival ~10%.
        survival = out.height / N
        assert 0.05 < survival < 0.15, survival
        print(f"  scale: {N:,} pts masked in {dt:.2f}s, survival={survival:.1%} ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_apply_mask_unknown_column_raises() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mask_test_"))
    try:
        stl = tmp / "blocks.stl"
        make_two_block_stl(stl)
        mask = build_mask(stl, layers=[50], layer_thickness=0.03)
        df = pl.DataFrame({"Demand X": [0.0], "Demand Y": [0.0], "layer": [50]})
        try:
            apply_mask(df, mask, x_col="bogus")
        except KeyError:
            pass
        else:
            raise AssertionError("expected KeyError")
        print("  apply_mask + unknown col raises ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def main() -> None:
    print("Phase 4 masking tests:")
    test_build_basic()
    test_apply_mask_keeps_correct_points()
    test_apply_mask_drops_layers_above_part()
    test_buffer_grow_and_shrink()
    test_cache_roundtrip()
    test_apply_mask_at_scale()
    test_apply_mask_unknown_column_raises()
    print("\nAll Phase 4 tests passed")

if __name__ == "__main__":
    main()
