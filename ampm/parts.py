"""
Parser for QuantAM "parts" CSV exports.

QuantAM exports build-plate part metadata as a multi-section CSV:

    #,Renishaw,Material,Development
    ,Version,0.6.1
    <blank>
    #,Tab - -1,Parent Parts
    #,"Sr. No.","Source Index","Layer Thickness","X Position", ...
    ID.,"[T0C1]","[T0C2]", ...
    ,"1","Part(1)","0.03","-26.787",...
    ,"2","Part(2)","0.03","-13.823",...
    ...
    <blank>
    #,Tab - 1,General
    #,"Sr. No.",...
    ...

Each section is a tab in the QuantAM UI:
  Tab -1 : Parent Parts        (positions, layer counts — top-level metadata)
  Tab  1 : General             (one row per (part, variant): N, N.1, N.s)
  Tab  2 : Strategy
  ...
  Tab 10 : Scan Volume         (laser parameters per (part, variant))
  Tab 11 : Scan Upskin
  ...

The shape of each section is identical:
  - line 1: section header  (#,Tab - N,Name)
  - line 2: human-readable column names (starts with #,)
  - line 3: machine column codes — ignored (starts with ID.,)
  - lines 4..M: data rows (start with a leading comma — first field is empty)
  - blank line terminates the section

This module reads the file once, splits into sections, and exposes them as a
``QuantAMParts`` object holding {section_name: pl.DataFrame}.  Helper methods
combine sections (e.g. parent metadata + scan volume parameters) into the
assembled tables typically wanted downstream.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import polars as pl

SECTION_PARENT = "Parent Parts"
SECTION_GENERAL = "General"
SECTION_SCAN_VOLUME = "Scan Volume"


def _iter_sections(text: str) -> Iterator[tuple[str, int, list[list[str]]]]:
    """
    Walk the file and yield ``(section_name, tab_number, rows)`` for each
    Tab section. ``rows`` are raw CSV-parsed rows (lists of strings) including
    the human-readable header row but excluding the section header and code row.
    """
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)

    i = 0
    while i < len(rows):
        row = rows[i]
        if len(row) >= 2 and row[0] == "#" and row[1].startswith("Tab - "):
            tab_num = int(row[1].removeprefix("Tab - ").strip())
            section_name = row[2] if len(row) >= 3 else ""

            if i + 1 >= len(rows):
                break
            header_row = rows[i + 1]
            if not header_row or header_row[0] != "#":
                raise ValueError(
                    f"Section 'Tab - {tab_num}' at line {i + 1}: expected "
                    f"'#'-prefixed header row but got {header_row}"
                )

            data_start = i + 3
            data_rows = []
            j = data_start
            while j < len(rows):
                r = rows[j]
                if not r or all(cell == "" for cell in r):
                    break
                data_rows.append(r)
                j += 1

            yield section_name, tab_num, [header_row, *data_rows]
            i = j
        else:
            i += 1


def _section_to_dataframe(
    header_row: list[str],
    data_rows: list[list[str]],
) -> pl.DataFrame:
    """
    Convert one section's raw rows to a polars DataFrame.

    Header row layout: ``["#", "Sr. No.", "Source Index", <real cols...>, ""]``
    Data row layout:    ``["", "1.1", "Part(1)", <values...>, ""]``

    The leading and trailing empty fields come from the file's literal format
    (every line starts with a comma and ends with a comma). We strip both.
    """
    headers = [h for h in header_row[1:] if h != ""]

    cleaned: list[list[str]] = []
    for r in data_rows:
        body = r[1:] if r and r[0] == "" else r
        while body and body[-1] == "":
            body.pop()
        if len(body) < len(headers):
            body = body + [""] * (len(headers) - len(body))
        elif len(body) > len(headers):
            body = body[: len(headers)]
        cleaned.append(body)

    if not cleaned:
        return pl.DataFrame({h: pl.Series([], dtype=pl.String) for h in headers})

    columns = {h: [] for h in headers}
    for row in cleaned:
        for h, val in zip(headers, row):
            columns[h].append(val)

    df = pl.DataFrame(columns)

    out_cols = []
    for c in df.columns:
        s = df[c]
        coerced = _try_numeric(s)
        out_cols.append(coerced if coerced is not None else s)
    return pl.DataFrame(out_cols)


def _try_numeric(s: pl.Series) -> pl.Series | None:
    """Return a numeric version of ``s`` if every non-empty value parses; else None."""
    if s.dtype != pl.String:
        return None  # already numeric or non-string
    cast = s.cast(pl.Float64, strict=False)
    original_non_empty = s.str.len_chars() > 0
    new_nulls = cast.is_null() & original_non_empty
    if new_nulls.any():
        return None
    if (
        cast.drop_nulls().is_finite().all()
        and (cast.drop_nulls() == cast.drop_nulls().cast(pl.Int64)).all()
    ):
        return cast.cast(pl.Int64).alias(s.name)
    return cast.alias(s.name)


class QuantAMParts:
    """
    Parsed representation of a QuantAM parts CSV.

    Sections are accessible by name via ``parts["Scan Volume"]`` or by tab
    number via ``parts.tab(10)``. Two convenience methods —
    ``parent_parts()`` and ``volume_parameters()`` — return assembled tables
    in the format usually wanted downstream.
    """

    def __init__(
        self,
        sections: dict[str, pl.DataFrame],
        tab_numbers: dict[str, int],
        path: Path | None = None,
    ) -> None:
        self._sections = sections
        self._tab_numbers = tab_numbers
        self.path = path

    @classmethod
    def from_path(cls, path: str | Path) -> "QuantAMParts":
        """Load and parse a QuantAM parts CSV."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"QuantAM parts file not found: {path}")
        text = path.read_text(encoding="utf-8-sig")  # handles BOM if present

        sections: dict[str, pl.DataFrame] = {}
        tab_numbers: dict[str, int] = {}
        for name, tab_num, rows in _iter_sections(text):
            df = _section_to_dataframe(rows[0], rows[1:])
            sections[name] = df
            tab_numbers[name] = tab_num

        if not sections:
            raise ValueError(
                f"No 'Tab - N' sections found in {path}. "
                f"Is this a QuantAM parts CSV?"
            )
        return cls(sections, tab_numbers, path=path)

    def __getitem__(self, section_name: str) -> pl.DataFrame:
        if section_name not in self._sections:
            available = ", ".join(self._sections)
            raise KeyError(
                f"Section {section_name!r} not in file. Available: {available}"
            )
        return self._sections[section_name]

    def __contains__(self, section_name: str) -> bool:
        return section_name in self._sections

    @property
    def section_names(self) -> list[str]:
        """All section names in the order they appear in the file."""
        return list(self._sections.keys())

    def tab(self, tab_number: int) -> pl.DataFrame:
        """Look up a section by its Tab number (e.g. 10 for Scan Volume)."""
        for name, num in self._tab_numbers.items():
            if num == tab_number:
                return self._sections[name]
        raise KeyError(f"No section with tab number {tab_number}")

    def __repr__(self) -> str:
        path = f", path={self.path!s}" if self.path else ""
        return (
            f"QuantAMParts(sections={len(self._sections)}{path}, "
            f"names={self.section_names!r})"
        )

    def parent_parts(self) -> pl.DataFrame:
        """
        Return the top-level "Parent Parts" table with normalized column
        names matching the AMPM analyzer's convention:

        Columns
        -------
        Part ID         : str  ("Part(1)", "Part(2)", ...)
        Layer Thickness : float
        X Position      : float
        Y Position      : float
        Layers Count    : int

        Each row is one physical part on the build plate.
        """
        if SECTION_PARENT not in self._sections:
            raise ValueError(f"No {SECTION_PARENT!r} section found in {self.path}")
        df = self._sections[SECTION_PARENT]

        wanted = {
            "Source Index": "Part ID",
            "Layer Thickness": "Layer Thickness",
            "X Position": "X Position",
            "Y Position": "Y Position",
            "Layers Count": "Layers Count",
        }
        missing = [c for c in wanted if c not in df.columns]
        if missing:
            raise ValueError(
                f"{SECTION_PARENT!r} section missing columns {missing}. "
                f"Found: {df.columns}"
            )
        return df.select([pl.col(src).alias(dst) for src, dst in wanted.items()])

    def volume_parameters(self, *, variant: str = "1") -> pl.DataFrame:
        """
        Return the per-part scan-volume laser parameters joined with the
        parent metadata.

        Parameters
        ----------
        variant
            Which variant to extract. Each part has multiple rows in the
            volume section (e.g. "1.1" for the part body and "1.s" for the
            supports). Default ``"1"`` selects the body. Pass ``"s"`` for
            supports, or any custom suffix QuantAM exports.

        Returns
        -------
        DataFrame with one row per part, joining parent metadata with the
        scan-volume parameters. The "Part ID" column is the key.
        """
        if SECTION_SCAN_VOLUME not in self._sections:
            raise ValueError(f"No {SECTION_SCAN_VOLUME!r} section found in {self.path}")
        parent = self.parent_parts()
        volume = self._sections[SECTION_SCAN_VOLUME]

        if "Sr. No." not in volume.columns or "Source Index" not in volume.columns:
            raise ValueError(
                f"{SECTION_SCAN_VOLUME!r} missing 'Sr. No.' or 'Source Index'. "
                f"Found: {volume.columns}"
            )

        sr_str = volume["Sr. No."].cast(pl.String)
        variant_suffix = f".{variant}"
        keep = sr_str.str.ends_with(variant_suffix)
        volume_v = volume.filter(keep).rename({"Source Index": "Part ID"})

        if "Sr. No." in volume_v.columns:
            volume_v = volume_v.drop("Sr. No.")
        return parent.join(volume_v, on="Part ID", how="left")

    def volume_parameters_with_speed(self, *, variant: str = "1") -> pl.DataFrame:
        """
        Same as ``volume_parameters`` but adds a derived ``Hatch Speed``
        column computed as:

            Hatch Speed = (Hatches Point Distance / Hatches Exposure Time) * 1000

        Units: mm/s, given Point Distance in mm and Exposure Time in
        microseconds (the QuantAM convention).

        Returns the same DataFrame as ``volume_parameters`` plus the new
        ``Hatch Speed`` column at the end. The original parameters
        (``Hatches Point Distance``, ``Hatches Exposure Time``) are preserved.
        """
        df = self.volume_parameters(variant=variant)
        for c in ("Hatches Point Distance", "Hatches Exposure Time"):
            if c not in df.columns:
                raise ValueError(
                    f"Cannot derive Hatch Speed: column {c!r} missing from "
                    f"{SECTION_SCAN_VOLUME!r}. Found: {df.columns}"
                )
        return df.with_columns(
            (
                pl.col("Hatches Point Distance")
                / pl.col("Hatches Exposure Time")
                * 1000.0
            ).alias("Hatch Speed")
        )


def compute_part_id_map(
    clustered: pl.DataFrame,
    parts_table: pl.DataFrame,
    *,
    cluster_col: str = "cluster",
    cluster_x_col: str = "Demand X",
    cluster_y_col: str = "Demand Y",
    parts_id_col: str = "Part ID",
    parts_x_col: str = "X Position",
    parts_y_col: str = "Y Position",
    max_distance_mm: float = 5.0,
    verbose: bool = True,
) -> dict[int, str]:
    """
    Build a {cluster_id: part_id} mapping by matching each cluster's centroid
    to the nearest part position.

    Each cluster is matched independently to its nearest part. This means
    multiple clusters can map to the same part (a "collision") — useful when
    a single physical part fragments into multiple clusters, but worth
    flagging so the caller knows.

    Parameters
    ----------
    clustered
        DataFrame containing at least cluster_col, cluster_x_col, cluster_y_col.
        Noise rows (cluster=-1) are ignored — they are not in the returned map.
    parts_table
        DataFrame with at least parts_id_col, parts_x_col, parts_y_col.
        Typically the output of ``QuantAMParts.parent_parts()``.
    cluster_col, cluster_x_col, cluster_y_col
        Column names in ``clustered``. Defaults match DataStore output.
    parts_id_col, parts_x_col, parts_y_col
        Column names in ``parts_table``. Defaults match parent_parts().
    max_distance_mm
        Warn if any cluster's centroid is farther than this from the nearest
        part — usually a sign of a misaligned mask or wrong parts file.
    verbose
        Print summary information including any collisions or far matches.

    Returns
    -------
    dict mapping cluster_id (int) -> part_id (str).
    """
    for c in (cluster_col, cluster_x_col, cluster_y_col):
        if c not in clustered.columns:
            raise KeyError(f"Column {c!r} not in clustered DataFrame")
    for c in (parts_id_col, parts_x_col, parts_y_col):
        if c not in parts_table.columns:
            raise KeyError(f"Column {c!r} not in parts_table")

    centroids = (
        clustered.lazy()
        .filter(pl.col(cluster_col) >= 0)
        .group_by(cluster_col)
        .agg(
            pl.col(cluster_x_col).mean().alias("_cx"),
            pl.col(cluster_y_col).mean().alias("_cy"),
        )
        .sort(cluster_col)
        .collect()
    )

    if centroids.is_empty():
        if verbose:
            print("[part_id_map] No non-noise clusters to map.")
        return {}

    cluster_ids = centroids[cluster_col].to_numpy()
    cx = centroids["_cx"].to_numpy()
    cy = centroids["_cy"].to_numpy()
    px = parts_table[parts_x_col].to_numpy()
    py = parts_table[parts_y_col].to_numpy()
    part_ids = parts_table[parts_id_col].to_list()

    dx = cx[:, None] - px[None, :]
    dy = cy[:, None] - py[None, :]
    dist2 = dx * dx + dy * dy

    nearest_idx = dist2.argmin(axis=1)
    nearest_dist = np.sqrt(dist2[np.arange(len(cluster_ids)), nearest_idx])

    mapping: dict[int, str] = {}
    for i, cid in enumerate(cluster_ids):
        mapping[int(cid)] = part_ids[int(nearest_idx[i])]

    if verbose:
        counts = Counter(mapping.values())
        collisions = {p: n for p, n in counts.items() if n > 1}
        if collisions:
            print(
                f"[part_id_map] Warning: {len(collisions)} part(s) claimed by "
                f"multiple clusters:"
            )
            for p, n in collisions.items():
                claiming = [c for c, mapped in mapping.items() if mapped == p]
                print(f"  {p}: claimed by clusters {claiming}")

        far = nearest_dist > max_distance_mm
        if far.any():
            print(
                f"[part_id_map] Warning: {int(far.sum())} cluster(s) more than "
                f"{max_distance_mm} mm from nearest part:"
            )
            for i in np.flatnonzero(far):
                print(
                    f"  cluster {int(cluster_ids[i])} -> {part_ids[int(nearest_idx[i])]}: "
                    f"{nearest_dist[i]:.2f} mm"
                )

        unmatched_parts = set(part_ids) - set(mapping.values())
        if unmatched_parts:
            print(
                f"[part_id_map] Note: {len(unmatched_parts)} part(s) had no "
                f"matching cluster: {sorted(unmatched_parts)}"
            )

        print(
            f"[part_id_map] {len(mapping)} clusters mapped to "
            f"{len(set(mapping.values()))} unique parts. "
            f"Max distance: {nearest_dist.max():.2f} mm."
        )

    return mapping


def apply_part_id_map(
    clustered: pl.DataFrame,
    mapping: dict[int, str],
    *,
    cluster_col: str = "cluster",
    part_id_col: str = "part_id",
    noise_label: str | None = None,
) -> pl.DataFrame:
    """
    Add a ``part_id_col`` to ``clustered`` based on a cluster->part_id mapping.

    Parameters
    ----------
    clustered
        DataFrame with a cluster column.
    mapping
        Dict produced by ``compute_part_id_map`` (or hand-edited).
    cluster_col
        Name of the source cluster column. Default ``"cluster"``.
    part_id_col
        Name of the new column to add. Default ``"part_id"``.
    noise_label
        What to assign to rows whose cluster id isn't in the mapping
        (typically noise points with cluster=-1). Default ``None``, which
        produces a null entry. Pass e.g. ``"noise"`` for a string sentinel.

    Returns
    -------
    Original DataFrame with ``part_id_col`` added (String dtype).
    """
    if cluster_col not in clustered.columns:
        raise KeyError(f"Column {cluster_col!r} not in DataFrame")

    all_cluster_ids = clustered[cluster_col].unique().to_list()
    full_mapping: dict[int, str | None] = {int(k): v for k, v in mapping.items()}
    for cid in all_cluster_ids:
        if int(cid) not in full_mapping:
            full_mapping[int(cid)] = noise_label

    return clustered.with_columns(
        pl.col(cluster_col)
        .replace_strict(full_mapping, return_dtype=pl.String)
        .alias(part_id_col)
    )


def join_parts_with_stats(
    stats_table: pl.DataFrame,
    parts_table: pl.DataFrame,
    *,
    stats_part_col: str = "part_id",
    parts_part_col: str = "Part ID",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Left-join a per-part stats table (e.g. CoV) onto a parts-parameter table.

    Each input has one row per part. They use different column conventions:
    ``compute_cov`` outputs ``part_id`` (lowercase, underscored), QuantAM
    parts tables use ``Part ID`` (mixed case, the spelling QuantAM exports).
    This function bridges the two.

    Parameters
    ----------
    stats_table
        DataFrame from e.g. ``compute_cov``. Must have ``stats_part_col``.
    parts_table
        DataFrame from e.g. ``QuantAMParts.volume_parameters_with_speed``.
        Must have ``parts_part_col``.
    stats_part_col
        Column name of the part identifier in ``stats_table``.
        Default ``"part_id"``.
    parts_part_col
        Column name of the part identifier in ``parts_table``.
        Default ``"Part ID"``.
    verbose
        If True, warn about parts that have stats but no parameters
        (typically a sign of missing or misnamed part data).

    Returns
    -------
    DataFrame: ``stats_table`` augmented with all columns from ``parts_table``.
    Number of rows equals ``stats_table.height``. Parts present in stats but
    not in parts get null values for the parameter columns.
    """
    if stats_part_col not in stats_table.columns:
        raise KeyError(f"stats_part_col {stats_part_col!r} not in stats_table")
    if parts_part_col not in parts_table.columns:
        raise KeyError(f"parts_part_col {parts_part_col!r} not in parts_table")

    left = stats_table.with_columns(pl.col(stats_part_col).cast(pl.String))
    right = parts_table.with_columns(
        pl.col(parts_part_col).cast(pl.String).alias(stats_part_col)
    )
    if parts_part_col != stats_part_col:
        right = right.drop(parts_part_col)

    out = left.join(right, on=stats_part_col, how="left")

    if verbose:
        parts_only_cols = [c for c in right.columns if c != stats_part_col]
        if parts_only_cols:
            probe = parts_only_cols[0]
            missing_rows = out.filter(pl.col(probe).is_null())
            if missing_rows.height > 0:
                missing_ids = missing_rows[stats_part_col].to_list()
                print(
                    f"[join_parts_with_stats] Warning: "
                    f"{missing_rows.height} part(s) in stats table have no "
                    f"matching row in parts_table:"
                )
                for pid in missing_ids:
                    print(f"  {pid}")

        stats_part_set = set(left[stats_part_col].to_list())
        parts_part_set = set(right[stats_part_col].to_list())
        parts_only = parts_part_set - stats_part_set
        if parts_only:
            print(
                f"[join_parts_with_stats] Note: "
                f"{len(parts_only)} part(s) in parts_table had no stats "
                f"and were not included: {sorted(parts_only)}"
            )

    return out


def assign_nearest_part(
    masked: pl.DataFrame,
    parts_table: pl.DataFrame,
    *,
    x_col: str = "Demand X",
    y_col: str = "Demand Y",
    parts_id_col: str = "Part ID",
    parts_x_col: str = "X Position",
    parts_y_col: str = "Y Position",
    part_id_col: str = "part_id",
    max_distance_mm: float | None = None,
    noise_label: str | None = "noise",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Assign every row in ``masked`` to its nearest part by XY position.

    A simpler alternative to DBSCAN clustering + ``compute_part_id_map`` when
    parts are well-separated (typical for builds with few, large parts).
    Each row gets the Part ID of the closest part by 2D Euclidean distance
    in the (x_col, y_col) plane. Z is ignored.

    Memory note
    -----------
    Builds a distance matrix of shape (n_rows, n_parts) in NumPy. For 80M
    rows and 6 parts this peaks at ~2 GB. For builds with 50+ parts the
    matrix grows and a KDTree approach would be more efficient — but this
    function is intentionally simple for the small-parts-count case it's
    designed for.

    Parameters
    ----------
    masked
        Mask-filtered DataFrame; must contain ``x_col`` and ``y_col``.
    parts_table
        DataFrame with ``parts_id_col``, ``parts_x_col``, ``parts_y_col``.
        Typically the output of ``QuantAMParts.parent_parts()``.
    x_col, y_col
        Spatial columns in ``masked``. Defaults match DataStore output.
    parts_id_col, parts_x_col, parts_y_col
        Column names in ``parts_table``. Defaults match parent_parts().
    part_id_col
        Name of the new column to add. Default ``"part_id"``.
    max_distance_mm
        Optional cap on assignment distance. Rows whose nearest part is
        farther than this are assigned ``noise_label`` instead. ``None``
        (the default) means assign every row regardless of distance —
        appropriate when parts are large and well-separated. Set to a
        sensible cap (e.g., 1.5× expected part radius) if you want to
        flag rows that are unambiguously outside any part.
    noise_label
        What to assign to rows beyond ``max_distance_mm``. Default
        ``"noise"`` matches the convention used by ``apply_part_id_map``.
    verbose
        Print per-part row counts and distance statistics.

    Returns
    -------
    DataFrame with a new ``part_id_col`` (String dtype) added.

    See also
    --------
    compute_part_id_map : Use when clustering has produced cluster IDs and
        you want to label each cluster with a part. Use ``assign_nearest_part``
        directly when clustering isn't appropriate (well-separated parts).
    """
    for c in (x_col, y_col):
        if c not in masked.columns:
            raise KeyError(f"Column {c!r} not in masked DataFrame")
    for c in (parts_id_col, parts_x_col, parts_y_col):
        if c not in parts_table.columns:
            raise KeyError(f"Column {c!r} not in parts_table")
    if parts_table.is_empty():
        raise ValueError("parts_table is empty")

    import numpy as np

    n_parts = parts_table.height
    part_ids = parts_table[parts_id_col].to_list()
    px = parts_table[parts_x_col].to_numpy().astype(np.float32)
    py = parts_table[parts_y_col].to_numpy().astype(np.float32)

    x = masked[x_col].to_numpy()
    y = masked[y_col].to_numpy()

    dx = x[:, None] - px[None, :]
    dy = y[:, None] - py[None, :]
    dist2 = dx * dx + dy * dy
    del dx, dy

    nearest_idx = dist2.argmin(axis=1)
    nearest_dist2 = np.take_along_axis(dist2, nearest_idx[:, None], axis=1).ravel()
    del dist2
    nearest_dist = np.sqrt(nearest_dist2)
    del nearest_dist2

    idx_to_id = {i: part_ids[i] for i in range(n_parts)}
    assigned = np.array([idx_to_id[int(i)] for i in nearest_idx], dtype=object)

    if max_distance_mm is not None:
        too_far = nearest_dist > max_distance_mm
        n_too_far = int(too_far.sum())
        if n_too_far > 0:
            assigned[too_far] = noise_label  # may be None
            if verbose:
                pct = n_too_far / len(assigned)
                print(
                    f"[assign_nearest_part] {n_too_far:,} row(s) "
                    f"({pct:.1%}) farther than {max_distance_mm} mm from any "
                    f"part → labelled {noise_label!r}"
                )

    out = masked.with_columns(pl.Series(part_id_col, assigned, dtype=pl.String))

    if verbose:
        print(
            f"[assign_nearest_part] Assigned {len(assigned):,} rows to "
            f"{n_parts} part(s):"
        )
        for i, pid in enumerate(part_ids):
            mask = nearest_idx == i
            if max_distance_mm is not None:
                mask = mask & ~too_far
            n = int(mask.sum())
            if n == 0:
                print(f"  {pid}: 0 rows assigned")
                continue
            d = nearest_dist[mask]
            print(
                f"  {pid}: {n:>9,} rows, "
                f"distance mean={d.mean():.2f} mm, "
                f"max={d.max():.2f} mm"
            )

    return out
