"""
Test the DataStore against synthetic data that mimics the real AMPM format.

This generates a temp directory with a handful of 'Packet data for layer N, laser 4.txt'
files using the real column names and tab-separated format (with trailing tab),
then exercises:
  - discovery
  - cache build
  - cache reuse (mtime-based skip)
  - cache invalidation (touch a source file)
  - query with various filter combinations
  - summary
"""
from __future__ import annotations

import sys

import os
import random
import shutil
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ampm import DataStore
from ampm.datastore import EXPECTED_COLUMNS

def make_fake_layer_file(path: Path, n_rows: int, layer: int, seed: int = 0) -> None:
    """Write a tab-separated file with the real column names and a trailing tab."""
    rng = random.Random(seed + layer)
    header = "\t".join(EXPECTED_COLUMNS) + "\t\n"  # trailing tab like the real files
    lines = [header]
    t = 19000 + layer * 1000
    for i in range(n_rows):
        t += 70
        # Spread X,Y across a plausible build-plate region per layer.
        x = -107 + rng.uniform(-5, 5)
        y = 73 + rng.uniform(-5, 5)
        focus = 0.0
        dlp_mean = 1576.0
        mvp_mean = rng.uniform(300, 1500)
        mvm_mean = rng.uniform(100, 300)
        lv_mean = rng.uniform(400, 800)
        lbr_mean = rng.uniform(13000, 17000)
        lop_mean = rng.uniform(13000, 17000)
        dlp_med = 1576.0
        mvp_med = mvp_mean + rng.uniform(-50, 50)
        mvm_med = mvm_mean + rng.uniform(-20, 20)
        lv_med = lv_mean + rng.uniform(-50, 50)
        lbr_med = lbr_mean + rng.uniform(-100, 100)
        lop_med = lop_mean + rng.uniform(-100, 100)
        vals = [
            t, 60, x, y, focus, dlp_mean,
            mvp_mean, mvm_mean, lv_mean, lbr_mean, lop_mean,
            dlp_med, mvp_med, mvm_med, lv_med, lbr_med, lop_med,
        ]
        # Match the real format: 3-decimal floats for spatial/signal columns.
        formatted = []
        for j, v in enumerate(vals):
            if j in (0, 1):  # Start time, Duration → integers
                formatted.append(str(int(v)))
            else:
                formatted.append(f"{v:.3f}")
        lines.append("\t".join(formatted) + "\t\n")  # trailing tab
    path.write_text("".join(lines))

def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ampm_test_"))
    print(f"Test directory: {tmp}\n")
    try:
        # 1. Generate synthetic source files
        layers_to_make = [1, 2, 3, 100, 526, 999]
        rows_per_layer = 500
        for L in layers_to_make:
            fname = f"Packet data for layer {L}, laser 4.txt"
            make_fake_layer_file(tmp / fname, rows_per_layer, L)
        print(f"[1] Wrote {len(layers_to_make)} synthetic layer files.\n")

        # 2. Discovery
        store = DataStore(tmp, layer_thickness=0.03)
        print(f"[2] {store!r}")
        assert store.layers == sorted(layers_to_make), store.layers
        print(f"    Discovered layers: {store.layers}")
        print(f"    Columns: {store.columns}\n")

        # 3. Partial cache build (triggered by first query, only requested layers)
        print("[3] First query of layers 1-3 → should ONLY build those layers:")
        df = store.query(layers=(1, 3), columns=["Demand X", "Demand Y"])
        print(f"    Got DataFrame shape={df.shape}, cols={df.columns}")
        assert df.shape[0] == 3 * rows_per_layer
        assert set(df.columns) == {"Demand X", "Demand Y", "layer", "Z"}
        # Verify only layers 1-3 got cached, not 100/526/999.
        cached_layers = sorted(
            int(p.stem.split("=")[1]) for p in (tmp / ".cache").glob("layer=*.parquet")
        )
        assert cached_layers == [1, 2, 3], f"Expected [1,2,3], got {cached_layers}"
        print(f"    Only layers {cached_layers} were cached ✓")
        # Z should equal layer * 0.03
        # Z values are stored as float32 — tolerance reflects that.
        z_check = df.group_by("layer").agg(pl.col("Z").first().alias("Z"))
        for row in z_check.iter_rows(named=True):
            assert abs(row["Z"] - row["layer"] * 0.03) < 1e-5
        print("    Z values match layer * thickness ✓")

        # Verify compact dtypes survive the round-trip through Parquet.
        full = store.query(layers=[1])
        schema = dict(zip(full.columns, full.dtypes))
        assert schema["layer"] == pl.Int16, f"layer is {schema['layer']}"
        assert schema["Z"] == pl.Float32, f"Z is {schema['Z']}"
        assert schema["Start time"] == pl.Int32
        assert schema["Duration"] == pl.Int16
        assert schema["Demand X"] == pl.Float32
        assert schema["MeltVIEW plasma (mean)"] == pl.Float32
        print("    Dtypes: int16/int32/float32 as expected ✓\n")

        # 3b. Overlapping query — should reuse 1-3, build only 4-not-existing
        #     and the new ones (100), skipping the rest.
        print("[3b] Overlapping query (layers 1-3 + 100) → builds only 100:")
        df_overlap = store.query(layers=[1, 2, 3, 100])
        cached_layers = sorted(
            int(p.stem.split("=")[1]) for p in (tmp / ".cache").glob("layer=*.parquet")
        )
        assert cached_layers == [1, 2, 3, 100], cached_layers
        assert df_overlap["layer"].unique().sort().to_list() == [1, 2, 3, 100]
        print(f"    Cache now contains {cached_layers} ✓\n")

        # 4. Full build_cache() — should build remaining layers (526, 999),
        #    skip already-cached ones (1, 2, 3, 100).
        print("[4] build_cache() with no args → builds remaining layers only:")
        store.build_cache()
        cached_layers = sorted(
            int(p.stem.split("=")[1]) for p in (tmp / ".cache").glob("layer=*.parquet")
        )
        assert cached_layers == sorted(layers_to_make), cached_layers
        print(f"    Cache complete: {cached_layers}\n")

        # 5. Cache invalidation by touching a source file
        print("[5] Touching source for layer 100 (newer mtime → should rebuild only it):")
        time.sleep(1.1)  # ensure mtime resolution clears
        target = tmp / "Packet data for layer 100, laser 4.txt"
        os.utime(target, None)
        store.build_cache()
        print()

        # 5b. Old cache format (float64 Z) → schema-version check rebuilds it.
        print("[5b] Old-format cache (float64 Z, int32 layer) → should rebuild:")
        # Overwrite layer 1's cache with a pretend "v1" Parquet file.
        zero_row = {c: [0.0] for c in EXPECTED_COLUMNS}
        zero_row["Start time"] = [0]
        zero_row["Duration"] = [0]
        old_format_df = pl.DataFrame(zero_row).with_columns(
            pl.lit(1, dtype=pl.Int32).alias("layer"),     # old wide dtype
            pl.lit(0.03, dtype=pl.Float64).alias("Z"),    # old wide dtype
        )
        cache_path = tmp / ".cache" / "layer=00001.parquet"
        old_format_df.write_parquet(cache_path)
        # Make the stale cache newer than the source so mtime *alone* would NOT
        # trigger rebuild — the schema check is what must catch it.
        os.utime(cache_path, None)
        store.build_cache(layers=[1])
        rebuilt = store.query(layers=[1])
        rebuilt_schema = dict(zip(rebuilt.columns, rebuilt.dtypes))
        assert rebuilt_schema["Z"] == pl.Float32, rebuilt_schema["Z"]
        assert rebuilt_schema["layer"] == pl.Int16, rebuilt_schema["layer"]
        # And it should have the real row count (not just 1 stale row).
        assert rebuilt.height == rows_per_layer
        print("    Old format auto-detected and rebuilt ✓\n")

        # 6. Query variations
        print("[6] Query variations:")
        # All layers, all columns
        df_all = store.query()
        print(f"    All:           shape={df_all.shape}, "
              f"columns count={len(df_all.columns)}")
        assert df_all.shape == (len(layers_to_make) * rows_per_layer, 19)

        # Single layer via list
        df_one = store.query(layers=[526])
        print(f"    layers=[526]:  shape={df_one.shape}")
        assert df_one.shape[0] == rows_per_layer
        assert df_one["layer"].unique().to_list() == [526]

        # Range tuple
        df_rng = store.query(layers=(1, 3))
        assert df_rng["layer"].unique().sort().to_list() == [1, 2, 3]
        print(f"    layers=(1,3):  shape={df_rng.shape}")

        # Spatial filter
        df_sp = store.query(
            layers=[526],
            x_range=(-107, -106),
            y_range=(73, 74),
        )
        print(f"    spatial:       shape={df_sp.shape} "
              f"(subset of {rows_per_layer})")
        assert df_sp.shape[0] < rows_per_layer
        # Verify filter actually held.
        assert df_sp["Demand X"].min() >= -107
        assert df_sp["Demand X"].max() <= -106

        # Generic filter on signal column
        df_sig = store.query(
            layers=[526],
            filters={"MeltVIEW plasma (mean)": (1000.0, 1500.0)},
        )
        print(f"    signal filter: shape={df_sig.shape}")
        assert df_sig["MeltVIEW plasma (mean)"].min() >= 1000.0
        assert df_sig["MeltVIEW plasma (mean)"].max() <= 1500.0

        # Column projection
        df_proj = store.query(layers=[526], columns=["Demand X"])
        print(f"    projection:    cols={df_proj.columns}")
        assert set(df_proj.columns) == {"Demand X", "layer", "Z"}
        print()

        # 7. Summary
        print("[7] Summary:")
        s = store.summary()
        print(s)
        assert s.shape == (len(layers_to_make), 6)  # layer, n_rows, x_min/max, y_min/max
        print()

        # 8. Error paths
        print("[8] Error handling:")
        try:
            store.query(columns=["Bogus column"])
        except KeyError as e:
            print(f"    Bad column → KeyError ✓  ({e})")

        try:
            store.query(layers=[99999])
        except ValueError as e:
            print(f"    No matching layers → ValueError ✓  ({e})")

        print("\nAll tests passed ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    import polars as pl  # used by Z-check inside main
    main()
