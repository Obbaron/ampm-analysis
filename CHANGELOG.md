# Changelog

All notable changes to AMPM Analyzer are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.3.1] - 2026-06-26

Mainly a structural release: AMPM Analyzer was
extracted from a single package project into the **OHPAL** (OHP Analytics Library)
workspace: the analysis pipeline is now a standalone library (`ohpal.ampm`) that the GUI
*depends on* rather than *contains*, packaging moved to a uv workspace with a src-layout
namespace, the import system was cleaned up (no more `sys.path`), and the build, asset
loading, and developer tooling were reworked to match. The QtWebEngine dependency was
dropped. Alongside the repackaging, this release also makes project loading fast and
non-blocking, aligns the Config and Analysis form layouts, and fixes a false validation
error on the auto-discovered BuildStarted DHXML.

### Added

- **OHPAL monorepo (uv workspace).** AMPM Analyzer is now one member of the OHPAL
  workspace, alongside the pipeline and shared libraries under `apps/`, `packages/`, and
  `plugins/`. A single `uv.lock` covers the whole workspace, members install editable, and
  they resolve each other locally via `[tool.uv.sources]`. First-time setup is a single
  `uv sync --all-packages`.
- **`ohpal-ampm` as a standalone pipeline library.** The end-to-end pipeline — load, mask,
  part assignment, clustering, correction, and statistics — is split into its own
  distribution, importable as `ohpal.ampm`, with no dependency on PyQt or any GUI. It can
  now be driven headless from scripts, notebooks, and CI; the GUI is just one consumer.
- **Task runner over uv.** A `justfile` (and an updated `makefile`) expose `setup`,
  `verify`, `run`, `test`, `build`, `rebuild`, and `clean`, each wrapping the corresponding
  uv command. `verify` smoke-tests the interpreter version, the cross-package imports, and
  that the `ohpal` namespace hasn't been collapsed by a stray `__init__.py`.
- **Pinned interpreter.** A `.python-version` pins the workspace to Python 3.11, keeping
  every environment on the `cp311` wheel ABI the offline build targets.
- **READMEs** for the workspace root, the application, and the pipeline package.

### Changed

- **Faster, non-blocking project loading.** Selecting a project root now discovers its
  config, layer range, columns, and views on a background worker, streaming each step to
  the log with a progress bar, so the window stays responsive instead of freezing while a
  build is scanned.
- **Schema read without building a DataStore.** Listing the loadable columns and the
  available layer range no longer constructs a `DataStore` (which parsed the derived
  Parquet) just to configure a load. Columns come from the fixed packet-header schema;
  trusted when the saved setup matches the current version, otherwise read from a single
  `.txt` header, and the layer range is parsed straight from the packet filenames. This
  removes the main delay after picking a project.
- **Aligned config and analysis forms.** Labelled controls on the Config tab and the
  Analysis tab's *Axes* and *Settings* groups now share one aligned label/field column
  instead of ragged per-row layouts, and the X / Y spatial range inputs each take their
  own row.
- **DHXML auto-discovery prefers the newest file.** When a folder holds several
  BuildStarted `.dhxml` files, the most-recently-modified one is selected rather than the
  alphabetically-first; date-stamped names aren't zero-padded, so name order could
  otherwise pick an older build.
- **src-layout namespace packaging.** `ohpal` is now a PEP 420 native namespace package;
  each distribution ships its own subtree under `src/ohpal/` with no `__init__.py` at the
  namespace level, so libraries can share the `ohpal.*` namespace. The application moved to
  `src/ampm_analyzer/`.
- **Import system.** All `sys.path` manipulation was removed. Modules resolve through the
  installed packages instead. Pipeline imports moved from `ampm.*` to `ohpal.ampm.*`, and
  the app's view imports were corrected to package-relative form.
- **Unified entry point.** Launch funnels through a single `main()` shared by
  `python -m ampm_analyzer`, the `ampm-analyzer` script, and a thin top-level `run.py` used
  as the PyInstaller entry point. `run.py` lives outside the package so the package's
  relative imports resolve when frozen.
- **Bundled-asset loading.** The window icon now loads via `importlib.resources` from
  package data, resolving identically in an editable install, a wheel, and the frozen
  executable, replacing the previous `__file__` / `sys._MEIPASS` path logic.
- **View discovery.** Built-in views are imported relative to their package
  (`{__package__}.<name>`) rather than via a hardcoded module path so discovery survives
  both the package rename and the frozen build.
- **PyInstaller spec.** The entry script is `run.py`; built-in views and the
  lazily-imported pipeline are gathered with `collect_submodules`; assets are bundled at
  their package-relative location to match the `importlib.resources` lookup; and the dead
  bytecode-cipher lines were dropped.
- **Developer tooling moved to dependency groups.** `pytest` and `pyinstaller` moved from
  `[project.optional-dependencies]` to PEP 735 `[dependency-groups]`, keeping them out of
  the app's published metadata while `uv sync --all-packages` still installs them for
  development and builds.

### Removed

- **QtWebEngine.** The `PyQt6.QtWebEngineWidgets` / `QtWebEngineCore` / `QtWebChannel`
  stack is no longer used; dropping bundled Chromium runtime to shrink the binary.

### Fixed

- **False "BuildStarted DHXML does not exist" on load.** Config validation could reject
  a DHXML path that had been correctly auto-discovered, because a stale path saved in the
  config or the `.ampm-ui.json` sidecar was applied over the discovered one — BuildStarted
  files are date-stamped, so a path from an earlier export goes out of date. Both the load
  scan and the saved-setup restore now keep a path only if it still exists, falling back
  to the freshly-discovered file, so a valid build no longer has to be re-selected by
  hand.

## [1.3.0] - 2026-06-22

A part-assignment method driven by the machine's own part bounding boxes, two
new ways to scope what gets loaded and analyzed (a part filter and a column
picker), and looser path requirements so builds without an STL or parts CSV
can still be loaded. The Load button and input validation were reworked so
nothing is silently gated, and the window now restores its layout on launch. Under
the hood, STL slicing and direct part assignment were reworked to cut peak
memory sharply and run faster.

### Added

- **DHXML part assignment.** A new `dhxml` assignment method reads the per-part
  3D bounding boxes from the Renishaw *BuildStarted* `.dhxml` file the 500S
  writes alongside the data, and assigns each point to the box that contains
  it. Selectable from the *Method* dropdown, with a *BuildStarted DHXML* path
  on the Config tab (auto-discovered from the source/project folder). Parts
  that share a name are made unique with a `name#n` suffix, and points falling
  outside every box are labelled `noise` so the direct/DBSCAN methods can still
  catch them. Aimed at tightly-packed builds where nearest-part assignment
  bleeds across part boundaries.
- **Part filter.** A collapsible *Part filter* on the Config tab lists the
  build's parts (from the parts CSV, or the DHXML for the dhxml method) with
  tick boxes; unticking a part drops its rows. Applied *after* assignment,
  so excluding a part can't pull its rows onto a neighbor, and `noise`
  rows are always kept. Remembered per build in `.ampm-ui.json`.
- **Window state.** Window size, position, maximized/fullscreen state, and the
  config/log splitter position are saved on close and restored on next launch.

### Changed

- **Columns picker.** The *Columns* field is now a collapsible checklist
  populated from the dataset schema (probed when the source is set) instead of
  a comma-separated text box. Untick signals to skip loading them; `Demand X`/
  `Demand Y`, `Start time`, and `layer`/`Z` remain always-loaded. Still
  remembered per build.
- **STL and parts CSV are now optional.** Config generation and loading no
  longer require an STL or a QuantAM parts CSV. A build with neither — e.g. one
  using the dhxml method with masking off — now loads, with those paths left
  empty and the layer thickness entered in the GUI rather than read from the
  CSV. An *ambiguous* parts-CSV match still errors; only "none found" became
  non-fatal.
- **Load button and validation.** *Load Data* is no longer disabled based on
  inputs (reversing the 1.0.0 behavior). It stays available, runs validation
  when pressed, and reports any problems in the log — including a missing or
  failed config. Options that can't be completed are greyed out with a tooltip
  naming the file to add: masking needs an STL, `direct`/`dbscan` need a parts
  CSV, `dhxml` needs a BuildStarted DHXML, and *Assign parts* is disabled when
  no assignment file is present.
- **Constant-memory direct assignment.** `assign_nearest_part` (the direct
  method) now finds each point's nearest part with a `scipy.spatial.cKDTree`
  (`query(..., distance_upper_bound=...)`) instead of materializing a
  `(chunk x n_parts)` distance matrix. Peak memory drops from multiple GB to
  roughly constant — the per-build profile line fell from ~1.8 GB to a few
  hundred MB — and it runs faster, while preserving the per-part count and
  distance statistics and the noise / max-distance behavior. Points are
  queried in 2M-row blocks (a tunable memory/speed constant that doesn't
  affect results); `workers=1` keeps profiling deterministic.
- **Vectorized STL slicing.** The in-house `stl_stream.py` slicer is now the
  preferred slicing path (fastest), with its shapely geometry constructions
  vectorized.
- **Linear-time ring stitching.** The slice-contour stitcher replaced its
  per-segment Python tuple-to-dict with a packed-`int64` key (`(a << 32) | b`)
  plus `np.unique(..., return_inverse=True)` — the same trick the clustering
  code already used — turning the hot loop into a pure-NumPy operation and
  cutting a ~3.7 GB allocation spike by roughly an order of magnitude. A stable
  sort reproduces the old dict's per-group segment ordering, so the greedy walk
  picks exactly the same rings as before (equivalent output).
- **Cheaper cross-section union.** Building the per-layer mask polygon no longer
  calls `shapely.union_all` on every slice. Distinct part cross-sections are
  disjoint islands, yet `union_all` runs a full planar overlay regardless
  (~841 ms on a 2000-part slice just to confirm no overlaps). The mask now does
  an `STRtree` self-query for touching/overlapping parts and, in the common
  disjoint case, returns a plain `MultiPolygon` — geometrically identical and
  ~20-24x cheaper — falling back to `union_all` only for genuinely touching or
  overlapping slices.

## [1.2.0] - 2026-06-18

Build and dependency hardening, plus one user-facing addition: the MeltVIEW
melt-pool signal correction is now exposed as a toggleable parameter. The
development and offline build/runtime environments are also pinned to a single
Python version with capped, reproducible dependencies and a smaller offline
footprint.

### Added

- **MeltVIEW melt-pool correction.** The `correction` parameter is now exposed
  as a user-configurable setting, applying a correction to the MeltVIEW
  melt-pool (mean) signal from the MAIN machine.

### Changed

- **Standardized on Python 3.11.9 (64-bit).** Development, offline source
  install, and the compiled binary now target the same interpreter version as
  the offline deployment machine, so version-specific wheels
  (`cp311`/`win_amd64`) match across every environment. `requires-python`
  remains `>=3.11`.
- **Capped dependency versions.** All runtime and dev dependencies now carry
  upper bounds at the next major (e.g. `numpy>=1.24,<3`, `polars>=1.0,<2`), so
  resolves are reproducible and a future breaking major can't be pulled in
  silently. `pyarrow` is held to its current major (`<25`), as it bumps major
  frequently.
- **Slimmed trimesh to base `trimesh`.** Replaced `trimesh[easy]` with plain
  `trimesh` after confirming — by import-tracing a full app session plus the
  test suite — that the slicing path uses only mesh loading and sectioning.
  The unused native extras (`embreex`, `manifold3d`, `mapbox_earcut`,
  `vhacdx`, `pycollada`, `lxml`, `pillow`, and others) are no longer bundled,
  shrinking both the offline wheel set and the compiled binary. Slicing's real
  dependencies — `shapely`, `networkx`, `rtree` — are retained as explicit
  requirements.
- Reworded the missing-dependency guard in `masking.py` to name the packages
  the slicing path actually needs (`shapely`/`networkx`/`rtree`) and point at
  the bundled offline wheel set, instead of a `trimesh[easy]` network install
  that would dead-end on an air-gapped machine.

### Removed

- Stale trimesh-backend hidden imports (`embreex`, `manifold3d`,
  `mapbox_earcut`, `lxml`, `svg.path`, `pycollada`) from the PyInstaller spec,
  now that those packages are no longer installed.
- Unused `etc/` scratch scripts, the project's only reference to `matplotlib`
  (which was never a declared dependency).

### Fixed

- **Offline source install.** The bundled wheel set now includes the build
  backend (`setuptools`, `wheel`), so installing the project from source on an
  offline machine (`pip install --no-index --find-links`) no longer fails
  trying to fetch the backend from PyPI.

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

[Unreleased]: https://github.com/Obbaron/ampm-analysis/compare/v1.3.1...HEAD
[1.3.1]: https://github.com/Obbaron/ampm-analysis/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/Obbaron/ampm-analysis/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/Obbaron/ampm-analysis/compare/v1.1.3...v1.2.0
[1.1.3]: https://github.com/Obbaron/ampm-analysis/compare/v1.1.2...v1.1.3
[1.1.2]: https://github.com/Obbaron/ampm-analysis/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/Obbaron/ampm-analysis/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/Obbaron/ampm-analysis/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Obbaron/ampm-analysis/releases/tag/v1.0.0