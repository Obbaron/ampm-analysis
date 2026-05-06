# Pipeline

End-to-end walkthrough of how data flows from raw `.txt` files to final plots. Each stage is a separate module and most are cacheable — see [CACHING.md](CACHING.md) for invalidation rules.

## Stage 1: Data loading via DataStore

`ampm.datastore.DataStore` finds every `Packet data for layer N, laser 4.txt` file in your `SOURCE` directory and converts each to a Parquet file under `<SOURCE>/.cache/layer=NNNNN.parquet`. The conversion is lazy: only files whose source mtime is newer than the cache file get rewritten.

Why per-layer Parquet rather than one giant file:

- Easy to add layers later without rewriting the whole archive
- Allows partial loading (`store.query(layers=range(100, 200))`)
- Mtime-based invalidation works per-file
- Polars' multi-file scan handles the rest

The cache uses **float32** for spatial and signal columns, **Int16** for the layer index and Duration, **Int32** for Start time. Total cache size for an 89M-row build is roughly 6 GB.

```python
from ampm import DataStore
store = DataStore(SOURCE, layer_thickness=0.03)
df = store.query()                              # all layers, all columns
df = store.query(columns=["Demand X", "Demand Y", "MeltVIEW melt pool (mean)"])
df = store.query(layers=range(100, 200))
df = store.query(filters={"Demand X": (-10, 10)})
```

A `Z` column equal to `layer * layer_thickness` is added on read so downstream code has a true Cartesian Z position.

## Stage 2: Masking to part region

Raw monitoring data covers the whole build plate including support rafts, contour scans, and inter-part gaps. `ampm.masking.build_mask` slices the parts STL into per-layer 2D polygons; `apply_mask` filters rows to those inside the polygons.

```python
from ampm.masking import build_mask, apply_mask
mask = build_mask(STL, layers=store.layers, layer_thickness=0.03)
df_masked = apply_mask(df, mask)
```

The mask is cached by SHA256 of the STL content, not by file mtime, so editing the STL in-place still invalidates correctly. `mask_or_load` from `ampm.mask_cache` wraps both steps with a Parquet-backed cache of the survivor row keys (see [CACHING.md](CACHING.md)).

The 8% of rows dropped by the mask are mostly on contour scans and inter-part rapids — useful to know if you're investigating a "missing region" that turns out to be intentionally excluded.

## Stage 3: Clustering parts

`ampm.clustering.cluster_dbscan_chunked` runs DBSCAN with anisotropic Z-scaling: `eps_xy` controls the in-plane neighbor radius, `eps_z` controls the through-thickness radius. Points within both thresholds of a neighbor join the same cluster.

For an 80M-row build, naïve DBSCAN runs out of memory (the pairwise neighbor matrix doesn't fit). The chunked variant processes overlapping layer ranges, then uses union-find to merge labels across chunk boundaries. Memory peak is bounded by `LAYERS_PER_CHUNK` × ~250k rows.

```python
clustered = cluster_dbscan_chunked(
    df_masked,
    eps_xy=0.3, eps_z=0.06,
    min_samples=10,
    layers_per_chunk=11, overlap_layers=2,
)
```

Tuning is build-specific. See [CLUSTERING.md](CLUSTERING.md).

## Stage 4: Linking clusters to parts

DBSCAN gives you 20 clusters with integer IDs. The QuantAM parts CSV gives you 20 parts with names like `Part(1)` and known XY positions. `compute_part_id_map` matches them by nearest centroid:

```python
from ampm.parts import QuantAMParts, compute_part_id_map, apply_part_id_map

quantam = QuantAMParts.from_path(PARTS_CSV)
parts_table = quantam.parent_parts()
mapping = compute_part_id_map(clustered, parts_table)
clustered = apply_part_id_map(clustered, mapping, noise_label="noise")
```

For a clean build the max centroid-to-part distance is sub-millimeter. Larger distances trigger warnings — usually a sign of a misaligned mask or wrong parts file. See [PARTS.md](PARTS.md).

## Stage 5: Optional XY-bias correction

The MAIN machine's MeltVIEW sensor has a smooth XY-dependent bias in melt-pool intensity. `MeltPoolCorrection` divides this out with a pre-fitted polynomial:

```python
from ampm.correction import MeltPoolCorrection
clustered = MeltPoolCorrection().apply(clustered)
# Adds 'MeltVIEW melt pool (mean) corrected' column.
```

This is **not** for the RBV machine and **not** for other sensors. See [CORRECTION.md](CORRECTION.md) before applying to non-MAIN data.

## Stage 6: Statistics — coefficient of variation

`ampm.stats.compute_cov` aggregates per-part variability. Three modes:

- **`overall`**: total variability across all rows in each part (intra-layer + drift + outliers)
- **`per_layer_mean`**: average within-layer CoV (filters out drift)
- **`across_layers`**: CoV of per-layer means (only drift)

```python
from ampm.stats import compute_cov
cov = compute_cov(
    clustered,
    columns=["MeltVIEW melt pool (mean)", "MeltVIEW plasma (mean)"],
    group_by="part_id",
    mode="overall",
)
```

Output is one row per part with `cov_<column>` columns. Combine three modes to diagnose drift vs noise: a part with low `per_layer_mean` but high `across_layers` is one where each layer was clean but the process slowly walked off-target.

## Stage 7: Linking with laser parameters

For parametric studies, join the per-part CoV with the laser parameters from the QuantAM CSV's Tab 10:

```python
from ampm.parts import join_parts_with_stats

parts_with_speed = quantam.volume_parameters_with_speed()
joined = join_parts_with_stats(cov_overall, parts_with_speed)
# One row per part with both CoV and Hatches Power / Hatch Speed.
```

`Hatch Speed` is computed as `(Hatches Point Distance / Hatches Exposure Time) * 1000` in mm/s.

## Stage 8: Visualization

All plotters live in `ampm.plotting`:

- `scatter3d` — 3D point cloud, colored by any column
- `scatter2d` — top-down 2D view with optional equal aspect
- `scatter2d_layered` — slider-controlled per-layer viewer with signal dropdown
- `bar` — bar chart with sort and orientation control
- `contour` — filled contour over a (x, y) grid for parametric process maps
- `kde` — overlaid kernel density curves for distribution comparison

See [PLOTTING.md](PLOTTING.md) for parameters, file size implications, and which plot to reach for when.

## Where this all comes together

`cov.py` runs stages 1–8 with the JR299 settings as a reference:

```bash
python examples/cov.py
```

Produces three plots: a 3D scatter colored by per-part CoV, a parametric process map (CoV vs Hatches Power × Hatch Speed), and a KDE comparison of the most-stable and least-stable parts.

`explore.py` is the same pipeline with all the extra prints, mode comparisons, and intermediate plots — useful for development and tuning. Treat `cov.py` as the polished version, `explore.py` as the working scratch.

`view_layers.py` is a separate single-purpose tool for scrubbing through layers visually. It loads the mask cache directly without re-clustering, so it's fast to start.
