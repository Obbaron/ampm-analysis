# Clustering

DBSCAN identifies the physical parts on the build plate without knowing how many there are or where they sit. The challenge is making it work at AMPM scale (~80M points) without melting your laptop.

## What DBSCAN does

A point is a "core point" if at least `min_samples` other points lie within distance `eps`. Two core points within `eps` of each other join the same cluster. Points within `eps` of a core point but with fewer than `min_samples` neighbors of their own become "border points" of that cluster. Everything else is "noise" (cluster ID `-1`).

Why DBSCAN rather than k-means: we don't know the number of clusters in advance, parts can have arbitrary shapes (fork tines, hollow geometries), and noise rejection comes for free.

## Anisotropic Z-scaling

Layers are 30 µm thick — much smaller than the typical XY hatch spacing of 0.085 mm. If we used a single isotropic `eps`, points within one layer would have to share a tiny radius with points 5+ layers away.

We use two parameters:

- `eps_xy` — neighbor radius in the XY plane (mm)
- `eps_z` — neighbor radius through Z (mm)

Internally, Z is rescaled by `eps_xy / eps_z` so isotropic DBSCAN with `eps = eps_xy` does the right thing.

## Tuning eps_xy

Start by running the k-distance curve to find a sensible value:

```python
from ampm.clustering import k_distance_curve
curve = k_distance_curve(df_masked, k=10, sample_size=50_000, mode="3d",
                         eps_xy=0.5, eps_z=0.05)  # rough initial guess
# Plot 'Rank' vs 'k-distance (mm)'; eps_xy lives at the elbow.
```

For the JR299 Sterling build the elbow is at ~0.3 mm, which matches the hatch spacing. Larger parts with sparser scan tracks need larger `eps_xy`.

Common pitfalls:

- **Too small** → clusters fragment (you get 50 clusters for 20 parts) and noise inflates
- **Too large** → adjacent parts merge into one cluster
- The right value is roughly **2-3× the hatch spacing**

## Tuning eps_z

`eps_z` should be a small multiple of layer thickness. With `LAYER_THICKNESS = 0.03`, an `eps_z = 0.06` requires neighbors within 2 layers. That's robust to single missing-data layers without bridging across larger gaps.

Common values:

- `eps_z = 0.06` — connects across 2 layers (most common)
- `eps_z = 0.15` — connects across 5 layers, useful if your data has gaps
- `eps_z = layer_thickness` — connects only within-layer (rarely useful — you'd just use 2D mode)

## Choosing min_samples

`min_samples` is the density threshold. Each cluster must have at least one core point with this many neighbors within `eps`.

- `min_samples = 10` — works for the JR299 data
- For lower-density data (smaller parts, sparser hatches), reduce to 5
- Above 20, you start losing edges of clusters as noise

## The chunked algorithm

The single-pass DBSCAN we'd love to run can't fit 80M points × 80M neighbor matrix in memory. Two options:

**Option A: downsample-and-propagate.** Run DBSCAN on a representative sample (say 200k points), then assign every other point to its nearest sample's cluster via a BallTree. Fast but fragile — at low representative density, clusters fragment because the sample doesn't capture them all.

**Option B: chunked DBSCAN.** Process overlapping layer ranges. Within a chunk, DBSCAN runs on the full data for that chunk. Overlapping rows in adjacent chunks get matched and labels are merged via union-find.

We default to **Option B** because it's robust at full data density.

```python
from ampm.clustering import cluster_dbscan_chunked

clustered = cluster_dbscan_chunked(
    df_masked,
    eps_xy=0.3, eps_z=0.06,
    min_samples=10,
    mode="3d",
    layers_per_chunk=11,        # smaller = lower memory, more chunks
    overlap_layers=2,           # at least ceil(eps_z / layer_thickness)
    verbose=True,
)
```

Auto-overlap: pass `overlap_layers=None` to let the function pick `max(10, ceil(eps_z/layer_thickness)*2)`.

## Memory considerations

DBSCAN's memory peak is during the pairwise neighbor query within each chunk. For ~250k rows per layer × `layers_per_chunk` × ~3 neighbor-list slots per point, that's roughly:

| layers_per_chunk | rows per chunk | peak RAM (approx) |
|------------------|----------------|-------------------|
| 11               | ~2.7M          | ~3 GB             |
| 20               | ~5M            | ~6 GB             |
| 50               | ~12M           | ~15 GB            |

If your machine has 32 GB total and you're running everything else (Plotly, polars copies of the data), `layers_per_chunk=11-20` is the safe range. Below 11 you start spending more time on chunk boundaries than on the actual clustering.

The chunked algorithm is parallelized via DBSCAN's `n_jobs=-1`, so all CPU cores are used. Expect ~99% CPU utilization throughout.

## Checking the result

After clustering, look at:

```python
from ampm.clustering import cluster_summary
print(cluster_summary(clustered))
```

You want:

- **Number of clusters** matching your known number of parts
- **n_rows per cluster** roughly equal (within 2× of each other for parts of similar size)
- **noise points** under 1% — higher means `min_samples` is too high or `eps` is too small
- **z_min and z_max per cluster** spanning the full build height (3.0 to 12.99 mm for JR299)

If z_min/z_max for each cluster is just a few layers, `eps_z` is too small and DBSCAN is splitting parts vertically into slabs. Increase `eps_z` or `overlap_layers`.

## Stable labels across runs

By default `stable_labels=True`: clusters are renumbered so that cluster 0 has the lowest centroid X, cluster 1 the next lowest, etc. (with Y as a tiebreaker, then Z). This means:

- Reruns produce the same labels
- The label happens to correspond loosely to physical position
- Caching works correctly across reruns

Disable only if you have a downstream system that expects DBSCAN's natural label ordering.

## When clustering goes wrong

Symptoms and fixes:

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Way more clusters than parts | `eps_xy` too small | Increase `eps_xy` |
| Way fewer clusters than parts | `eps_xy` too large or parts physically close | Decrease `eps_xy`; check part spacing |
| 100% noise | `min_samples` too high | Decrease to 5 |
| Clusters span only a few layers | `eps_z` too small | Increase `eps_z`; increase `overlap_layers` |
| OOM during DBSCAN | `layers_per_chunk` too large | Decrease to 11 |
| Took forever, single core | `n_jobs` not set | We always pass `n_jobs=-1`; check sklearn version |
| Different cluster IDs each run | `stable_labels=False` | Set `stable_labels=True` (default) |

## Reference: tested settings for JR299

```python
EPS_XY = 0.3
EPS_Z = 0.06
MIN_SAMPLES = 10
LAYERS_PER_CHUNK = 11
OVERLAP_LAYERS = 2
```

Result: 20 clusters, 0 noise points, full Z range per cluster, max centroid-to-part distance 0.00 mm.
