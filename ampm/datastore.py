"""
DataStore: lazy-loading, Parquet-cached access to Renishaw 500S AMPM data.

A directory of 'Packet data for layer N, laser M.txt' files is converted on first
use to a partitioned Parquet dataset (one file per layer) under <source_dir>/.cache/.
Queries use Polars' lazy API so only the requested slice is materialized in memory.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import polars as pl

CACHE_FORMAT_VERSION = 2

EXPECTED_COLUMNS: list[str] = [
    "Start time",
    "Duration",
    "Demand X",
    "Demand Y",
    "Demand focus",
    "Demand laser power (mean)",
    "MeltVIEW plasma (mean)",
    "MeltVIEW melt pool (mean)",
    "LaserVIEW (mean)",
    "Laser back reflection (mean)",
    "Laser output power (mean)",
    "Demand laser power (median)",
    "MeltVIEW plasma (median)",
    "MeltVIEW melt pool (median)",
    "LaserVIEW (median)",
    "Laser back reflection (median)",
    "Laser output power (median)",
]


_CSV_SCHEMA: dict[str, type[pl.DataType]] = {
    "Start time": pl.Int32,
    "Duration": pl.Int16,
    "Demand X": pl.Float32,
    "Demand Y": pl.Float32,
    "Demand focus": pl.Float32,
    "Demand laser power (mean)": pl.Float32,
    "MeltVIEW plasma (mean)": pl.Float32,
    "MeltVIEW melt pool (mean)": pl.Float32,
    "LaserVIEW (mean)": pl.Float32,
    "Laser back reflection (mean)": pl.Float32,
    "Laser output power (mean)": pl.Float32,
    "Demand laser power (median)": pl.Float32,
    "MeltVIEW plasma (median)": pl.Float32,
    "MeltVIEW melt pool (median)": pl.Float32,
    "LaserVIEW (median)": pl.Float32,
    "Laser back reflection (median)": pl.Float32,
    "Laser output power (median)": pl.Float32,
}

_FILENAME_RE = re.compile(
    r"^Packet data for layer (\d+), laser \d+\.txt$",
    re.IGNORECASE,
)


class DataStore:
    """
    Lazy data store over a directory of AMPM Packet data text files.

    Parameters
    ----------
    source_dir : str | Path
        Directory containing the 'Packet data for layer N, laser M.txt' files.
    layer_thickness : float, default 0.03
        Layer thickness in mm. Used to compute the Z column as
        ``layer * layer_thickness``.
    cache_dir : str | Path | None, default None
        Directory for the Parquet cache. If None, uses ``source_dir / ".cache"``.
    """

    def __init__(
        self,
        source_dir: str | Path,
        layer_thickness: float = 0.03,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.source_dir = Path(source_dir).resolve()
        if not self.source_dir.is_dir():
            raise FileNotFoundError(f"source_dir does not exist: {self.source_dir}")

        self.layer_thickness = float(layer_thickness)
        self.cache_dir = (
            Path(cache_dir).resolve()
            if cache_dir is not None
            else self.source_dir / ".cache"
        )

        self._source_files: dict[int, Path] = self._discover_source_files()

    def _discover_source_files(self) -> dict[int, Path]:
        """Find all matching .txt files in source_dir and map layer -> path."""
        out: dict[int, Path] = {}
        for p in self.source_dir.iterdir():
            if not p.is_file():
                continue
            m = _FILENAME_RE.match(p.name)
            if m is None:
                continue
            layer = int(m.group(1))
            if layer in out:
                raise ValueError(
                    f"Duplicate layer {layer}: {out[layer].name} and {p.name}"
                )
            out[layer] = p
        if not out:
            raise FileNotFoundError(
                f"No 'Packet data for layer N, laser M.txt' files found in "
                f"{self.source_dir}"
            )
        return out

    def _cache_path(self, layer: int) -> Path:
        """Return the Parquet path for a given layer."""
        return self.cache_dir / f"layer={layer:05d}.parquet"

    @property
    def layers(self) -> list[int]:
        """Sorted list of layer numbers discovered in source_dir."""
        return sorted(self._source_files.keys())

    @property
    def columns(self) -> list[str]:
        """Available data columns: the 17 source columns + 'layer' + 'Z'."""
        return [*EXPECTED_COLUMNS, "layer", "Z"]

    def _needs_rebuild(self, layer: int) -> bool:
        """
        True if the Parquet for this layer is missing, older than its source,
        or written with an old cache format version.
        """
        src = self._source_files[layer]
        cache = self._cache_path(layer)
        if not cache.exists():
            return True
        if src.stat().st_mtime > cache.stat().st_mtime:
            return True
        try:
            schema = pl.scan_parquet(str(cache), glob=False).collect_schema()
            if schema.get("Z") != pl.Float32:
                return True
        except Exception:
            return True
        return False

    def _convert_one(self, layer: int) -> None:
        """Read one source .txt and write its Parquet sibling."""
        src = self._source_files[layer]
        df = pl.read_csv(
            src,
            separator="\t",
            has_header=True,
            truncate_ragged_lines=True,
            glob=False,  # path may contain '[3]' etc. that look like glob patterns
            schema_overrides=_CSV_SCHEMA,
        )

        drop_cols = [c for c in df.columns if c.strip() == ""]
        if drop_cols:
            df = df.drop(drop_cols)

        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Layer {layer} ({src.name}) is missing expected columns: {missing}\n"
                f"Found columns: {df.columns}"
            )
        df = df.select(EXPECTED_COLUMNS)

        df = df.with_columns(
            pl.lit(layer, dtype=pl.Int16).alias("layer"),
            pl.lit(layer * self.layer_thickness, dtype=pl.Float32).alias("Z"),
        )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(
            self._cache_path(layer),
            compression="zstd",
        )

    def build_cache(
        self,
        layers: Iterable[int] | None = None,
        force: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Convert source .txt files to Parquet. Skips files whose Parquet is
        already up-to-date (by mtime) unless ``force=True``.

        Parameters
        ----------
        layers
            If given, only build cache for these layer numbers. If None,
            consider every discovered layer.
        force
            If True, rebuild even up-to-date cache files.
        verbose
            Print progress messages.
        """
        if layers is None:
            candidates = self.layers
        else:
            candidates = sorted(set(int(L) for L in layers) & set(self._source_files))

        to_build = [L for L in candidates if force or self._needs_rebuild(L)]

        if verbose and to_build:
            print(
                f"Building Parquet cache: {len(to_build)} of {len(candidates)} "
                f"requested layers need (re)conversion."
            )

        for i, layer in enumerate(to_build, start=1):
            self._convert_one(layer)
            if verbose and (i % 25 == 0 or i == len(to_build)):
                print(f"  [{i}/{len(to_build)}] layer {layer}")

    def _ensure_cache(self, layers: Iterable[int]) -> None:
        """Build cache only for the specific layers requested. Cheap when up-to-date."""
        self.build_cache(layers=layers, verbose=True)

    def query(
        self,
        layers: range | Iterable[int] | tuple[int, int] | None = None,
        columns: list[str] | None = None,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        filters: dict[str, tuple[float, float]] | None = None,
    ) -> pl.DataFrame:
        """
        Query a slice of the dataset.

        Parameters
        ----------
        layers
            Layers to include. Accepts:
              - None: all layers
              - range: e.g. ``range(100, 201)``
              - tuple of length 2: inclusive ``(start, end)`` range
              - any other iterable of ints: an explicit set of layers
        columns
            Columns to return. ``"layer"`` and ``"Z"`` are always included
            even if not listed. If None, returns all columns.
        x_range, y_range
            Inclusive ``(min, max)`` filters on Demand X / Demand Y.
        filters
            Additional ``{column_name: (min, max)}`` inclusive filters.

        Returns
        -------
        polars.DataFrame
        """
        layer_set = self._resolve_layers(layers)
        if not layer_set:
            raise ValueError("No layers selected by the given 'layers' argument.")

        self._ensure_cache(layer_set)

        paths = [str(self._cache_path(L)) for L in sorted(layer_set)]
        lf = pl.scan_parquet(paths, glob=False)

        predicates: list[pl.Expr] = []
        if x_range is not None:
            lo, hi = x_range
            predicates.append(pl.col("Demand X").is_between(lo, hi))
        if y_range is not None:
            lo, hi = y_range
            predicates.append(pl.col("Demand Y").is_between(lo, hi))
        if filters:
            for col, (lo, hi) in filters.items():
                if col not in self.columns:
                    raise KeyError(f"Unknown column in filters: {col!r}")
                predicates.append(pl.col(col).is_between(lo, hi))

        if predicates:
            combined = predicates[0]
            for p in predicates[1:]:
                combined = combined & p
            lf = lf.filter(combined)

        if columns is not None:
            unknown = [c for c in columns if c not in self.columns]
            if unknown:
                raise KeyError(f"Unknown column(s): {unknown}")
            keep = list(
                dict.fromkeys([*columns, "layer", "Z"])
            )  # de-dupe, preserve order
            lf = lf.select(keep)

        return lf.collect()

    def _resolve_layers(
        self,
        layers: range | Iterable[int] | tuple[int, int] | None,
    ) -> set[int]:
        """Normalize the various accepted forms into a set of available layers."""
        all_layers = set(self._source_files.keys())
        if layers is None:
            return all_layers
        if isinstance(layers, range):
            requested = set(layers)
        elif (
            isinstance(layers, tuple)
            and len(layers) == 2
            and all(isinstance(x, int) for x in layers)
        ):
            lo, hi = layers
            requested = set(range(lo, hi + 1))
        else:
            requested = set(int(x) for x in layers)

        missing = requested - all_layers
        if missing and len(missing) == len(requested):
            raise ValueError(
                f"None of the requested layers exist. "
                f"Available range: {min(all_layers)}–{max(all_layers)}"
            )
        return requested & all_layers

    def summary(self) -> pl.DataFrame:
        """
        Per-layer summary: row count and basic ranges for spatial + key signals.
        Uses lazy scan so it doesn't load full data into memory.
        """
        self._ensure_cache(self.layers)

        paths = [str(self._cache_path(L)) for L in self.layers]
        lf = pl.scan_parquet(paths, glob=False)
        return (
            lf.group_by("layer")
            .agg(
                pl.len().alias("n_rows"),
                pl.col("Demand X").min().alias("x_min"),
                pl.col("Demand X").max().alias("x_max"),
                pl.col("Demand Y").min().alias("y_min"),
                pl.col("Demand Y").max().alias("y_max"),
            )
            .sort("layer")
            .collect()
        )

    def __repr__(self) -> str:
        L = self.layers
        return (
            f"DataStore(source_dir={self.source_dir!s}, "
            f"layers={len(L)} ({min(L)}–{max(L)}), "
            f"layer_thickness={self.layer_thickness} mm)"
        )
