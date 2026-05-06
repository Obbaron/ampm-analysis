# Plotting

All plot builders live in `ampm.plotting`. They share a "dumb plotter" philosophy: the function takes a DataFrame plus column names and returns a Plotly Figure. No data manipulation inside the plotters themselves — if you need to downsample or aggregate, do it before calling.

## Available functions

| Function | Use case | Underlying Plotly trace | Typical points |
|----------|----------|-------------------------|----------------|
| `scatter3d` | 3D point cloud, any column for color | `go.Scatter3d` (Three.js) | up to ~100k |
| `scatter2d` | Top-down 2D view | `go.Scattergl` (WebGL) | up to ~1M |
| `scatter2d_layered` | Per-layer slider + signal dropdown | `go.Scattergl` × M layers | varies |
| `bar` | Sorted bars over a categorical axis | `go.Bar` | dozens |
| `contour` | Filled contour for parametric maps | `go.Contour` | small grid |
| `kde` | Overlaid distribution curves | `go.Scatter` × N groups | dozens of curves |

## Common parameters

Every plotter accepts at least:

```python
fig = scatter3d(
    df,
    x="Demand X", y="Demand Y", z="Z",
    color="part_id",                      # any column; numeric or string
    size=2,                               # constant or column name
    colorscale="Turbo",
    color_range=(0.1, 0.3),               # cmin/cmax clip; None for auto
    title="...",
    xaxis_title="...", yaxis_title="...", zaxis_title="...",
    colorbar_title="...",
    hover_columns=["part_id"],            # extra columns in hover tooltip
)
```

Numeric color columns produce a continuous colorscale. String columns produce a discrete categorical colorbar with the original labels visible — `_resolve_color` encodes strings to integer codes internally and lays out the colorbar with `tickmode="array"`.

## scatter3d

3D point cloud rendered with Three.js. Comfortable up to ~100,000 points; beyond that, frames drop and rotation gets sluggish.

```python
sample = prepare_for_plot(clustered, target_points=80_000, method="random", seed=0)
fig = scatter3d(
    sample,
    x="Demand X", y="Demand Y", z="Z",
    color="cov_MeltVIEW melt pool (mean)",
    size=2,
    colorscale="Turbo",
)
```

For a build that's ~80M rows, **always downsample before calling**. The `prepare_for_plot` helper in `ampm.sampling` provides random / stride / grid downsamplers.

## scatter2d

Top-down 2D view rendered with WebGL (much faster than the 3D backend at scale). Comfortable up to ~1M points. Default `equal_aspect=True` locks the aspect ratio at 1:1, which is what you want for spatial XY data.

For non-spatial 2D plots — like a process map plotting CoV vs (speed, power) — pass `equal_aspect=False` so the axes scale independently.

## scatter2d_layered

Per-layer interactive viewer with a slider and (optional) signal dropdown. Each layer is a separate Plotly trace; the slider toggles which is visible. The dropdown rewrites `marker.color` across all traces using a `restyle` update.

```python
fig = scatter2d_layered(
    df_masked,
    x="Demand X", y="Demand Y",
    color_columns=["MeltVIEW melt pool (mean)", "MeltVIEW plasma (mean)"],
    points_per_layer=10_000,
    size=4.0,
)
```

**File size scales linearly with both layers and signals.** Each layer × signal pair stores its own color array. With 334 layers × 4 signals × 10,000 points, the rendered HTML is ~600 MB. Practical recommendations:

- **Single signal** → drop the dropdown, use `color_columns="meltpool"` (string, not list)
- **Many signals** → keep the list short (2-3); each adds ~25% to file size
- **Lots of layers** → reduce `points_per_layer` from 10k to 5k

The function downsamples randomly per layer. **This means single-point anomalies have only `points_per_layer / total` chance of being kept.** If you specifically need outlier preservation, downsample with `downsample_grid` (method="max") before calling — see `view_layers.py` for an alternative implementation.

## bar

Bar chart with sort and orientation control:

```python
fig = bar(
    cov_overall,
    x="part_id", y="cov_MeltVIEW melt pool (mean)",
    sort_by="y", sort_descending=False,
    orientation="h",           # horizontal — better for long part labels
)
```

Optional `color` column works the same way as the scatters (numeric → continuous; string → categorical).

## contour

Filled contour for parametric process maps. Pivots long-form `(x, y, z)` rows to a 2D grid, then renders `go.Contour`:

```python
fig = contour(
    joined,
    x="Hatch Speed", y="Hatches Power", z="cov_MeltVIEW melt pool (mean)",
    colorscale="Turbo",
    show_points=True,           # overlay the actually-measured points
)
```

Default `show_points=True` puts white-outlined markers at each measured combination so the visualization stays honest about which points are real and which are interpolated.

Works best for **regular grids** of speed × power parameters. Irregular sample patterns produce noticeable gaps in the contour. No line smoothing; the shape reflects the actual data.

## kde

Overlaid kernel density curves, one per group. Computes `scipy.stats.gaussian_kde` for each group's values:

```python
fig = kde(
    clustered,
    column="MeltVIEW melt pool (mean)",
    group_by="part_id",
    groups=["Part(1)", "Part(6)", "Part(11)"],   # filter to specific parts
    fill=True, opacity=0.5,
    colorscale="Turbo",
    max_points_per_group=80_000,                  # samples large groups
)
```

**Defaults that matter for memory:**

- `max_points_per_group=80_000` — caps sampling size per group. KDE is a smoothed estimate; the curve shape is virtually identical between 80k-sample and full-data versions (tested at <3% relative L2 difference). Set to None to use every row.
- Prints `[kde] Part(1): sampled 80,000/4,103,593 points` when sampling kicks in
- Set `verbose=False` to suppress

**Group count:** kde() warns when more than 12 groups are plotted. With 20 parts overlay, curves overlap badly and you can't read the plot. Filter to specific parts of interest (e.g., the most/least stable from CoV ranking).

## File size implications

Approximate file sizes for the JR299 build (334 layers, 4 signals, ~80M rows):

| Plot | Typical size |
|------|--------------|
| `scatter3d` (80k points sampled) | 5-10 MB |
| `scatter2d` (filtered to one layer) | 1-2 MB |
| `scatter2d_layered` (10k/layer, 4 signals) | ~600 MB |
| `scatter2d_layered` (5k/layer, 1 signal) | ~50 MB |
| `bar` (20 parts) | <100 KB |
| `contour` (20 points + interpolation) | <100 KB |
| `kde` (6 groups × 200 eval points) | <100 KB |

The big one is `scatter2d_layered`. Everything else is small. If your machine struggles, the layered viewer is almost always the cause.

## Browser performance

Plotly figures open in your default browser as standalone HTML files. The Three.js (3D) and WebGL (2D scattergl) backends both run on the GPU.

- **3D scatter rotation feels slow** → downsample to <50k points
- **2D layered viewer slider is jumpy** → reduce `points_per_layer` or limit signals
- **Browser hangs while loading** → file is >500 MB; reduce data
- **Hover tooltip is laggy** → too many `hover_columns` × points; drop columns

For very large builds, consider opening the figure once and then `fig.write_html("path.html")` instead of `fig.show()` so it doesn't try to launch a temp file each time.

## When you don't see what you expect

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Single dot for all data | Forgot to downsample; rendered 80M overlapping points | Use `prepare_for_plot` |
| Categorical colorbar shows ints not labels | Old `scatter3d` predates `_resolve_color` | Update `plotting.py` |
| Contour has weird stripes | Speed/power grid is irregular | Use `scatter2d` with `size=20` instead |
| KDE curves look spiky | `bandwidth` too small | Increase `bandwidth` or pass `"silverman"` |
| KDE curves look flat | `bandwidth` too large | Decrease bandwidth or use Scott's default |
| Layered viewer doesn't show colorbar updates | Plotly version too old | Upgrade plotly to >= 5.0 |

## Why the dumb-plotter pattern

Every plot function takes a fully-prepared DataFrame and renders it. No filtering, aggregation, or computation inside the plotter (with the exception of KDE, which has to run scipy by definition).

The trade-off: each plot call needs an explicit prep step. The benefit: the plotters are tiny, predictable, and composable. You can use the same `scatter2d` for spatial data, parametric data, time-series, and anything else — just pass different columns. No special modes to learn.

The plotters are also what you'd call directly when you want to visualize something from your scratch script — the `compute_X` and `prepare_X` functions stay separate so you can reuse them in scripts that don't plot at all.
