# Changelog

All notable changes to AMPM Analyzer are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.3] - 2026-06-13

Dense-build support: the full load → mask → assign → statistics pipeline now
runs in bounded memory, letting builds of ~750M+ rows complete on a 32 GB
machine where they previously exhausted memory. Roughly a 10x reduction in
peak memory across the pipeline.

### Added

- **Columns to load.** A *Columns* field on the Config tab loads only the
  signal columns you name (comma-separated; `all` for everything), pruned at
  Parquet-scan time. `Demand X`/`Demand Y`/`Start time` and `layer`/`Z` are
  always included. Remembered per build in `.ampm-ui.json`.
- **X / Y spatial range.** Optional inclusive `Demand X` / `Demand Y` bounds
  on the Config tab, applied at load time (one-sided bounds allowed; blank
  loads the full plate). Remembered per build, and included in the mask-cache
  key so changing the extent invalidates correctly.
- Optional memory profiler (`ampm/memprof.py`): wrap pipeline stages in
  `phase(...)` to log working-set and commit-charge readings per stage, with
  a marker when the process peak grows inside a stage. Off by default; enable
  with the `AMPM_MEMPROF` environment variable. Reads OS counters directly so
  it sees native (polars/numpy/shapely) allocations.

### Changed

- **Streaming mask application.** `apply_mask` now tests point-in-polygon in
  bounded chunks via `shapely.contains_xy` on raw coordinate arrays (no
  per-row `Point` objects), with peak memory independent of build size. New
  `apply_mask_keep` returns just the boolean keep-array for callers that
  don't need the filtered frame materialized.
- **Streaming mask cache.** `mask_cache` writes keys incrementally with a
  `ParquetWriter` (uniqueness checked per layer-run), and cached loads apply
  the keys with a sequential merge-walk over the cache file instead of a
  whole-build hash semi-join — bounding memory and running markedly faster.
  `mask_or_load` gains a `keep_fn` path that writes the cache straight from
  the keep-array.
- **Compact part assignment.** `assign_nearest_part` now emits `part_id` as a
  `pl.Enum` (4-byte codes over the part-name categories) built directly from
  the index buffer, and accumulates per-part distance statistics in a single
  pass rather than per-part full-length masks. `noise` is always a category
  when a noise label is given. The downstream power/speed attach uses a
  direct lookup instead of a left join, avoiding a full second copy of the
  frame.
- **Streaming CoV.** `compute_cov` projects to only the columns it needs
  before filtering and runs the group-by lazily with a streaming collect, so
  derived-column statistics no longer copy the full-width frame.

### Fixed

- Mask-cache writes no longer fail on Windows with `PermissionError`
  (WinError 5) when a load is immediately followed by a recompute: the cache
  file handle is now released promptly after reading metadata, and the
  atomic replace retries transient locks (antivirus, file indexer, cloud
  sync, Explorer preview) before falling back, with an actionable message if
  the file is genuinely locked.

## [1.1.2] - 2026-06-11

### Fixed

- 3D scatter no longer fails with a Plotly `ValueError` ("Invalid element(s)
  received for the 'color' property") when the selected color column contains
  nulls — e.g. coloring by a per-part statistic while some rows are
  unassigned (`noise`). Null-colored points are now excluded from the plot
  with a logged count, and an explicit error is raised if *every* sampled
  point has a null color.
- Plotting no longer requires `part_id` to exist. Hover columns are now
  best-effort across all views (3D scatter, 2D scatter, contour, and the
  layered viewer): columns missing from the data — such as `part_id` when
  part assignment is skipped at import — are omitted from the hover tooltip
  with a logged note instead of raising `KeyError`. Missing axis or color
  columns still fail loudly.

### Changed

- `stl_stream.py` docstrings reformatted to the numpy convention, with full
  `Parameters`/`Returns`/`Raises` sections added to all helpers (no behavior
  change; verified against a known build).

## [1.1.1] - 2026-06-04

### Changed

- Reverted distribution from the single self-contained executable (one-file
  PyInstaller build, introduced in 1.1.0) back to a folder build: a smaller
  executable alongside an `_internal` folder.

## [1.1.0] - 2026-06-04

### Added

- Drop-in plot views. Additional views are loaded at runtime from three
  locations, in increasing precedence: a per-user views folder, a build's
  `<project_root>/views/` folder, and the `AMPM_VIEWS_PATH` environment
  variable. Any external view may override a built-in of the same name. Works
  in the compiled executable without rebuilding. Documented in `docs/APP.md`.
- **Reload Views** button (next to the plot *Type* selector) to re-scan the
  view folders without restarting.
- The per-user views folder is created automatically on first launch.

### Changed

- Distribution is now a single self-contained executable (PyInstaller one-file
  build) instead of an executable plus a companion folder.
- Cleaned up the PyInstaller spec: removed stale hidden imports and corrected
  the bundled `ampm` module list.

## [1.0.0] - 2026-06-03

Initial release.

### Added

- Desktop GUI (PyQt6) with a Config tab (build selection, paths, parameters)
  and an Analysis tab (derived columns, plot view/axes/settings).
- End-to-end pipeline: per-layer Parquet cache, STL-based masking, part
  assignment (direct nearest-part or DBSCAN clustering), and per-part
  coefficient-of-variation statistics. Each stage is cached under
  `<source>/.cache/`.
- Pluggable, auto-discovered plot views (scatter 2D/3D, contour, KDE, bar,
  layer and single-layer viewers, CoV summary, k-distance).
- Layer range selection. Load a *From* / *To* subset instead of the whole
  build, bounded to the range detected in the source.
- Chunked direct part assignment with bounded memory for full builds (tens of
  millions of rows).
- Per-range cache files so switching between layer ranges reuses earlier work;
  parameter changes recompute only what's affected instead of erroring.
- Per-build session memory: pipeline parameters, derived-column recipes, and
  the selected plot view/axes/settings are saved beside each build in
  `.ampm-ui.json` and restored on reopen. `config.toml` is never modified.
- Last project-root folder remembered between launches.
- Progress feedback: phase-by-phase load progress and a plotting busy
  indicator.
- Input validation before loading, with **Load Data** disabled until required
  inputs are present.
- Collapsible data-source paths; **Plot** button positioned at the bottom of
  the Analysis tab.
- Dropdowns ignore the mouse wheel (scrolls the page instead of changing the
  selection).
- CLI launcher with startup retry and graceful `Ctrl+C` handling (second
  `Ctrl+C` forces quit).
- Documentation: GUI user guide (`docs/APP.md`), README, and pipeline docs.

[Unreleased]: https://github.com/Obbaron/ampm-analysis/compare/v1.1.3...HEAD
[1.1.3]: https://github.com/Obbaron/ampm-analysis/compare/v1.1.2...v1.1.3
[1.1.2]: https://github.com/Obbaron/ampm-analysis/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/Obbaron/ampm-analysis/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/Obbaron/ampm-analysis/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Obbaron/ampm-analysis/releases/tag/v1.0.0