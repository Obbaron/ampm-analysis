"""
02a_assign_parts_direct.py

Assigns each masked row to its nearest part by 2D Euclidean distance in
the (Demand X, Demand Y) plane, skipping DBSCAN entirely. This is the
appropriate approach when parts are large and well-spaced.

Pipeline branching
------------------
- Use direct assignment when the build has a small number of large,
  well-separated parts.
- Use DBSCAN instead when parts are small and closely-spaced.

Both methods produce the same downstream output: a DataFrame with a
``part_id`` string column.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl

from ampm import DataStore
from ampm.mask_cache import mask_or_load
from ampm.masking import apply_mask, build_mask
from ampm.parts import QuantAMParts, assign_nearest_part
from config import (
    LAYER_THICKNESS,
    MASK_CACHE,
    MASK_KEEP_CACHE,
    PARTS_CSV,
    SOURCE,
    STL,
)

MAX_DISTANCE_MM = (
    None  # rows farther than this from any part become "noise"; None = assign every row
)


def main() -> None:
    store = DataStore(SOURCE, layer_thickness=LAYER_THICKNESS)

    df = store.query()
    print(f"Loaded {df.height:,} rows across {len(store.layers)} layers.")

    mask_params = {
        "layers": (min(store.layers), max(store.layers)),
        "stl": str(STL),
        "buffer_mm": 0.0,
        "layer_thickness": LAYER_THICKNESS,
    }

    def masking_wrapper(d: pl.DataFrame) -> pl.DataFrame:
        mask = build_mask(
            STL,
            layers=store.layers,
            layer_thickness=LAYER_THICKNESS,
            buffer_mm=0.0,
            cache_path=MASK_CACHE,
        )
        return apply_mask(d, mask)

    df_masked = mask_or_load(
        df,
        cache_path=MASK_KEEP_CACHE,
        mask_fn=masking_wrapper,
        params=mask_params,
        strict=True,
    )
    print(f"After mask: {df_masked.height:,} rows.")

    del df

    quantam = QuantAMParts.from_path(PARTS_CSV)
    parts_table = quantam.parent_parts()
    print(f"Loaded {parts_table.height} parts from {Path(PARTS_CSV).name}.")

    print("\nAssigning each row to its nearest part...")
    assigned = assign_nearest_part(
        df_masked,
        parts_table,
        max_distance_mm=MAX_DISTANCE_MM,
        noise_label="noise",
    )

    print(
        f"\nResult: {assigned.height:,} rows, "
        f"{assigned['part_id'].n_unique()} unique part_id values."
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
