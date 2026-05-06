# Caching

Three caches sit on disk under `<SOURCE>/.cache/`:

```
.cache/
├── layer=00001.parquet         # ... × 434, the per-layer source cache
├── fullplate_mask.pkl          # STL → per-layer polygons (small)
├── mask_keep.pq                # Survivor row keys after applying mask
└── cluster_labels.pq           # (layer, Start time, cluster) triples
```

Each is independent. Invalidating one doesn't force a rebuild of the others.

## Layer Parquet cache

Built once per source file by `DataStore`. Each `Packet data for layer N, laser 4.txt` becomes a `layer=NNNNN.parquet` file. Total size is roughly **half the source `.txt` size** thanks to numeric column compression.

Invalidation: per-file, by source `.txt` mtime. If you rerun `cov.py` after editing one source file, only that file's Parquet rebuilds. The rest are unchanged.

Force a full rebuild:

```python
store = DataStore(SOURCE)
store.build_cache(force=True)
```

Or simply delete the cache manually.

The cache also auto-rebuilds if the schema changes (e.g., column dtypes change). The DataStore tracks a `CACHE_FORMAT_VERSION` and rebuilds when it doesn't match.

## Mask cache

Built once per (STL, layers, buffer_mm) combination. Two pieces:

**`fullplate_mask.pkl`** — the per-layer 2D shapely MultiPolygons. Cache key is SHA256 of the STL contents (not mtime), so editing the STL in place invalidates correctly. ~1-10 MB depending on geometry complexity.

**`mask_keep.pq`** — the surviving `(layer, Start time)` keys after applying the polygon mask to source data. About 250 MB JR299 vs ~6 GB for the full filtered DataFrame.

The mask cache is consulted via `mask_or_load`:

```python
df_masked = mask_or_load(
    df,
    cache_path=MASK_KEEP_CACHE,
    mask_fn=do_masking,
    params={"stl": str(STL), "buffer_mm": 0.0, ...},
    strict=True,
)
```

`params` is stored in the Parquet file metadata. On reload, the params must match exactly (`strict=True`) or the cache invalidates. Pass `strict=False` to fall through silently to recomputation.

## Cluster cache

Built once per (layers, mask, eps_xy, eps_z, min_samples, mode, layers_per_chunk, overlap_layers) combination. Stores `(layer, Start time, cluster)` triples — about 250 MB for 80M rows.

Invalidation:

- Change any clustering parameter → `cluster_or_load` raises with a clear "expected vs cached" diff
- Change the mask cache → cluster cache no longer makes sense (different rows in the masked DataFrame)
- Delete the file to force recomputation

The cluster cache is consulted via `cluster_or_load`:

```python
clustered = cluster_or_load(
    df_masked,
    cache_path=CLUSTER_CACHE,
    cluster_fn=do_clustering,
    params=cluster_params,
    strict=True,
)
```

Same `strict` semantics as the mask cache.

## Why both mask and cluster caches use a key-based format

The cluster cache could in principle store the entire labeled DataFrame, but that would be ~7 GB. Instead it stores just `(layer, Start time, cluster)` keys (~250 MB) and joins them back onto the full DataFrame on load. Same for the mask cache.

The trade-off: load is slightly slower (a join must happen) but disk space is 25× smaller, and the cache can be reused across DataFrames that have different column subsets.

## When a cache mismatch happens

```python
ValueError: Cache params don't match expected:
  eps_xy: cache=0.3, expected=0.4
```

This is `strict=True` doing its job — refusing to silently use a cache that no longer matches your code. To fix:

1. **Change params back** to match the cache, OR
2. **Delete the cache file** to force recomputation, OR
3. **Pass `strict=False`** to fall back to recomputation

Option 2 is the cleanest. The cache is small and rebuilds quickly.

## Disk space inventory

Example files sizes for JR299:

| Cache | Size |
|-------|------|
| Layer Parquet cache (×434 files) | ~6 GB |
| `fullplate_mask.pkl` | ~5 MB |
| `mask_keep.pq` | ~250 MB |
| `cluster_labels.pq` | ~280 MB |
| **Total** | **~6.5 GB** |

The layer cache dominates. If disk space is tight, you can delete it after running once and accept that the next run will be slow (the Parquet rebuild takes ~1-2 minutes per 100 source files).

## Cache file format details

All Parquet caches use **zstd compression** (best size/speed tradeoff for numeric columns). The mask `.pkl` is a Python pickle — small enough that a more sophisticated format isn't worth it.

Parquet metadata is the actual home for params. We use file-level metadata keys:

- `ampm_mask_cache_version`, `ampm_mask_cache_params`
- `ampm_cluster_cache_version`, `ampm_cluster_cache_params`

Inspecting a cache from the command line:

```python
import pyarrow.parquet as pq
import json

pf = pq.ParquetFile("path/to/cluster_labels.pq")
metadata = pf.schema_arrow.metadata
params = json.loads(metadata[b"ampm_cluster_cache_params"])
print(params)
```

## Why caches use Polars `glob=False`

Windows paths often contain `[N]` characters from QuantAM's export naming convention. Polars treats `[` and `]` as glob metacharacters by default, so `pl.read_parquet("[3] Export Packets/foo.pq")` looks for a file matching the character class `[3]` instead of literally `[3]`.

We pass `glob=False` everywhere we read Parquet files. There's a regression test for this in each cache module.

## Updating params in a backward-incompatible way

If you change a default in the package code (e.g., bump `MeltPoolCorrection`'s polynomial), bump the corresponding `CACHE_FORMAT_VERSION` so old caches invalidate cleanly. Then everyone gets a fresh build instead of silently using stale data.
