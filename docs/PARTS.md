# Parts module

`ampm.parts` parses QuantAM parts CSVs and links DBSCAN cluster IDs to physical Part IDs. This is what turns `cluster=0` into `Part(1)`.

## What QuantAM exports

The Renishaw 500S exports build metadata as a multi-section CSV. Each section is what the QuantAM UI calls a "tab":

```
Tab -1   Parent Parts        Top-level XY position, layer count, layer thickness
Tab  1   General             Per-(part, variant) general properties
Tab  2   Strategy            Scan strategy choices
Tab  3   Control             ...
...
Tab 10   Scan Volume         Laser parameters: power, point distance, exposure time
Tab 11   Scan Upskin         Same for upskin layers
...
Tab 14   Scan Shell and Core
```

The file has each section as a `#,Tab - N,SectionName` header followed by:

1. A column-name row (`#,"col1","col2",...`)
2. A machine-code row (`ID.,"[T0C1]","[T0C2]",...`) — we ignore this
3. Data rows (start with a comma; first field is empty)
4. A blank line ending the section

Section sizes vary because not every part has every variant. The old `get_parts` function in our codebase used "skip N rows" arithmetic to navigate this — fragile and easy to break. The current parser splits by section structure (blank lines + Tab markers), which is robust.

## QuantAMParts class

Single-pass parser that loads everything once and exposes sections by name or tab number:

```python
from ampm.parts import QuantAMParts
quantam = QuantAMParts.from_path(PARTS_CSV)

quantam.section_names                  # ['Parent Parts', 'General', ...]
quantam["Scan Volume"]                 # DataFrame for any section
quantam.tab(10)                        # Same, by tab number
quantam.parent_parts()                 # Normalized 5-col table
quantam.volume_parameters()            # Tab 10 + parent metadata
quantam.volume_parameters_with_speed() # Same + derived 'Hatch Speed' column
```

`parent_parts()` returns the layout-relevant subset:

```
Part ID     Layer Thickness   X Position   Y Position   Layers Count
Part(1)     0.03              -26.787      -11.585      333
Part(2)     0.03              -13.823      -10.59       333
...
```

`volume_parameters()` is `parent_parts()` left-joined with Tab 10 (Scan Volume), filtered to one variant. The default `variant="1"` picks the part body; pass `variant="s"` for the supports.

`volume_parameters_with_speed()` adds a derived `Hatch Speed` column:

```python
Hatch Speed = (Hatches Point Distance / Hatches Exposure Time) * 1000  # mm/s
```

Note the column names use **`Hatches`** (plural) — that's what QuantAM exports, not `Hatch`.

## Linking clusters to parts

DBSCAN gives integer cluster IDs. The parts file gives string Part IDs and known XY positions. `compute_part_id_map` matches each cluster's centroid to the nearest part:

```python
from ampm.parts import compute_part_id_map, apply_part_id_map

mapping = compute_part_id_map(clustered, parts_table)
# {0: 'Part(1)', 1: 'Part(6)', 2: 'Part(10)', ...}

clustered = apply_part_id_map(clustered, mapping, noise_label="noise")
```

`compute_part_id_map` prints diagnostics by default:

- **Collisions** — multiple clusters claiming the same part. Usually means DBSCAN fragmented a part. Fix by tuning `EPS_XY` (see [CLUSTERING.md](CLUSTERING.md)).
- **Far matches** — any cluster centroid more than `max_distance_mm` (default 5 mm) from its nearest part. Smell test for misaligned mask or wrong parts file.
- **Unmatched parts** — parts in the CSV that no cluster claimed. Often a part wasn't actually printed, or supports-only parts.

For a clean build, expect to see:

```
[part_id_map] 20 clusters mapped to 20 unique parts. Max distance: 0.00 mm.
```

Sub-millimeter max distance is a strong signal that everything is aligned correctly — the cluster centroids match the QuantAM-recorded part positions essentially exactly.

## Joining stats with laser parameters

For parametric studies you want per-part CoV alongside laser parameters. `join_parts_with_stats` bridges the column-name mismatch (`part_id` from `compute_cov` vs `Part ID` from QuantAM):

```python
from ampm.parts import join_parts_with_stats

parts_with_speed = quantam.volume_parameters_with_speed()
joined = join_parts_with_stats(cov_overall, parts_with_speed)
# Left join: every CoV row gets the matching parts-table columns added.
```

Verbose by default. Warns if any parts in the stats table have no parameter data, and notes any parts in the parts table that had no stats (these are silently dropped by the left join, but you're told).

This is the workflow for the parametric process map:

```python
from ampm.plotting import contour
fig = contour(joined, x="Hatch Speed", y="Hatches Power", z="cov_MeltVIEW melt pool (mean)")
```

## When parts go wrong

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| KeyError on `Part ID` | Wrong section accessed; `Source Index` is the QuantAM column | Use `parent_parts()` which renames it |
| `Hatch Speed` not in DataFrame | Used `volume_parameters()` not `volume_parameters_with_speed()` | Switch methods |
| `'Hatch Power'` not found | Column is `'Hatches Power'` (plural) in QuantAM exports | Use `Hatches Power` |
| Far-match warning at 100+ mm | Wrong parts CSV for the build, or build origin offset | Verify the CSV matches the source data |
| Multiple clusters → same part | DBSCAN fragmented a part | Tune `EPS_XY` (see [CLUSTERING.md](CLUSTERING.md)) |
| Mapping looks scrambled (cluster 0 → Part(7), etc.) | Normal — cluster IDs are sorted by centroid X position, which doesn't match QuantAM's part numbering | Not a bug; just use the mapping |

## Variants in QuantAM exports

A single physical part has multiple rows in Tab 10 — one per scan variant:

- `1.1` — the part body (default)
- `1.s` — the support structure
- `1.1.b` — sometimes seen for sub-features

`volume_parameters(variant="1")` keeps only `*.1` rows. To analyze supports separately, call with `variant="s"`. To see everything, access `quantam["Scan Volume"]` directly — that's the raw DataFrame.

## Why this is a clean rewrite

The old code (`get_parts` in the legacy codebase) used:

```python
skipped_rows = 9 * (len(full_parts_list) + 4) + (len(parts_idx) + 10)
```

to navigate the file. This worked for the specific export structure that existed when it was written, but breaks if QuantAM adds a tab or any section grows. The new parser walks rows once, splits by `Tab -` headers and blank lines, and never hard-codes line counts.

If you ever encounter a parts file the parser can't read, the most likely cause is a non-standard line-ending or unusual column naming in a Tab. Check `quantam.section_names` first; if the section list is wrong, the issue is in the section-finding regex (look at `_iter_sections` in `parts.py`).
