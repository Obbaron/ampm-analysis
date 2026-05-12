"""
01_load_and_mask.py

Loads source .txt files into per-layer parquet caches, then masks the
data to part regions using the STL geometry. Both steps are cached on
disk so subsequent runs are fast.

What this script writes to disk
-------------------------------
- Per-layer Parquet files under <SOURCE>/.cache/layer=NNNNN.parquet
  (one per source file; rebuilds only if source mtime is newer)
- Mask polygons under <SOURCE>/.cache/fullplate_mask.pkl
  (rebuilds only if the STL contents change)
- Surviving-row keys under <SOURCE>/.cache/mask_keep.pq
  (rebuilds only if mask params change)

Subsequent scripts or re-runs use these caches without rebuilding them.
"""

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parent.parent.parent)
)  # make config.py at the project root importable

import polars as pl

from ampm import DataStore
from ampm.mask_cache import mask_or_load
from ampm.masking import apply_mask, build_mask
from config import (
    LAYER_THICKNESS,
    MASK_CACHE,
    MASK_KEEP_CACHE,
    SOURCE,
    STL,
)


def main() -> None:
    store = DataStore(SOURCE, layer_thickness=LAYER_THICKNESS)

    needs_rebuild = [L for L in store.layers if store._needs_rebuild(L)]
    if needs_rebuild:
        print(
            f"[Layer cache] {len(needs_rebuild)} of {len(store.layers)} layers "
            f"need (re)conversion to Parquet."
        )
    else:
        print(
            f"[Layer cache] All {len(store.layers)} layer Parquet files are up-to-date."
        )

    df = store.query()  # Load the full dataset
    print(f"Loaded {df.height:,} rows across {len(store.layers)} layers.")

    mask_params = {
        "layers": (min(store.layers), max(store.layers)),
        "stl": str(STL),
        "buffer_mm": 0.0,
        "layer_thickness": LAYER_THICKNESS,
    }

    def masking_wrapper(d: pl.DataFrame) -> pl.DataFrame:  # wrapper to mask_or_load
        mask = build_mask(
            STL,
            layers=store.layers,
            layer_thickness=LAYER_THICKNESS,
            buffer_mm=0.0,
            cache_path=MASK_CACHE,
        )
        print(f"Mask covers {len(mask)} of {len(store.layers)} layers.")
        return apply_mask(d, mask)

    if Path(MASK_KEEP_CACHE).exists():
        print("[Mask cache] mask_keep.pq exists.\nChecking parameters...")
    else:
        print("[Mask cache] no cached survivor rows.\nBuilding from scratch.")

    df_masked = mask_or_load(
        df,
        cache_path=MASK_KEEP_CACHE,
        mask_fn=masking_wrapper,
        params=mask_params,
        strict=True,
    )

    survival = df_masked.height / df.height
    print(
        f"After mask: {df_masked.height:,} rows kept "
        f"({survival:.1%} of original {df.height:,})."
    )
    del df  # IMPORTANT: delete the unmasked dataframe (memory is precious)

    print("\nDone.")


if __name__ == "__main__":
    main()
