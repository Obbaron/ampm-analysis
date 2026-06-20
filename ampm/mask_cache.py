"""
Persistence for the keep-mask produced by apply_mask().

The mask filter step is deterministic given (source data, STL, buffer_mm),
and slow at scale (~80M points x hundreds of polygons via shapely.contains).
We cache the result by storing the ``(layer, Start time)`` keys of the rows
that survived masking. On load, we filter the input DataFrame down to just
those keys.

Usage
-----
    params = {
        "layers": (1, 434),
        "stl": str(STL),
        "buffer_mm": 0.0,
        "layer_thickness": 0.03,
    }
    df_masked = mask_or_load(
        df_full,
        cache_path="mask_keep.pq",
        mask_fn=lambda d: apply_mask(d, mask),
        keep_fn=lambda d: apply_mask_keep(d, mask),  # bounded-memory path
        params=params,
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import pyarrow.parquet as pq

try:
    from .memprof import phase
except ImportError:
    try:
        from memprof import phase
    except ImportError:
        from contextlib import nullcontext as phase

CACHE_FORMAT_VERSION = 1
_KEY_COLUMNS: tuple[str, str] = ("layer", "Start time")

_META_VERSION_KEY = b"ampm_mask_cache_version"
_META_PARAMS_KEY = b"ampm_mask_cache_params"

CHUNK_ROWS = 8_000_000  # Rows per chunk for streaming cache writer.


def save_mask_keep(
    df_masked: pl.DataFrame,
    cache_path: str | Path,
    *,
    params: dict | None = None,
    verbose: bool = True,
    chunk_rows: int = CHUNK_ROWS,
) -> None:
    """
    Persist the keys of the rows that survived masking.

    The cache file is written incrementally in ``chunk_rows`` slices, so
    saving never loads the full table into memory.

    Parameters
    ----------
    df_masked
        The DataFrame returned by ``apply_mask`` — i.e., only the kept rows.
        Must contain ``layer`` and ``Start time``.
    cache_path
        Where to write the cache (a single Parquet file).
    params
        Optional dict of parameters used to produce ``df_masked``. Must be
        JSON-serializable (Path objects fine — we use ``default=str``).
    verbose
        Print a one-line confirmation. Default True.
    chunk_rows
        Rows written per chunk. Default ``CHUNK_ROWS``.
    """
    if df_masked.is_empty():
        raise ValueError("ERROR: Mask empty.")
    _write_keys_streaming(
        df_masked, None, Path(cache_path), params, verbose, chunk_rows
    )


def save_mask_keep_from_keep(
    df_full: pl.DataFrame,
    keep: np.ndarray,
    cache_path: str | Path,
    *,
    params: dict | None = None,
    verbose: bool = True,
    chunk_rows: int = CHUNK_ROWS,
) -> None:
    """
    Persist mask-keep keys directly from a boolean keep-array.

    Parameters
    ----------
    df_full
        The pre-mask DataFrame. Must contain ``layer`` and ``Start time``.
    keep
        Boolean array of length ``df_full.height``; True = row survived.
    cache_path
        Where to write the cache (a single Parquet file).
    params
        Optional dict of parameters used to produce ``keep``. Must be
        JSON-serializable.
    verbose
        Print a one-line confirmation. Default True.
    chunk_rows
        Rows scanned per chunk. Default ``CHUNK_ROWS``.
    """
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (df_full.height,):
        raise ValueError(f"keep has shape {keep.shape}, expected ({df_full.height},).")
    if not keep.any():
        raise ValueError("ERROR: Mask empty.")
    _write_keys_streaming(df_full, keep, Path(cache_path), params, verbose, chunk_rows)


def _atomic_replace(tmp_path: Path, dest: Path, *, attempts: int = 10) -> None:
    """
    Replace ``dest`` with ``tmp_path`` as atomically.
    """
    import os
    import time

    last_err: Exception | None = None
    for i in range(attempts):
        try:
            os.replace(tmp_path, dest)
            return
        except PermissionError as e:  # WinError 5 (access) / 32 (sharing)
            last_err = e
            time.sleep(0.1 * (i + 1))
    try:
        if dest.exists():
            dest.unlink()
        os.replace(tmp_path, dest)
        return
    except OSError as e:
        last_err = e
    raise PermissionError(
        f"Could not replace mask cache {dest}\n"
        "It may be locked by another program.\n"
        f"{last_err!r}"
    ) from last_err


def _write_keys_streaming(
    df: pl.DataFrame,
    keep: np.ndarray | None,
    cache_path: Path,
    params: dict | None,
    verbose: bool,
    chunk_rows: int,
) -> None:
    """
    Stream ``(layer, Start time)`` keys to ``cache_path`` chunk by chunk.
    """
    for c in _KEY_COLUMNS:
        if c not in df.columns:
            raise KeyError(f"Column {c!r} required for mask cache; missing from input.")

    metadata: dict[bytes, bytes] = {
        _META_VERSION_KEY: str(CACHE_FORMAT_VERSION).encode(),
    }
    if params is not None:
        try:
            metadata[_META_PARAMS_KEY] = json.dumps(
                params,
                sort_keys=True,
                default=str,
            ).encode()
        except TypeError as e:
            raise TypeError(
                f"params must be JSON-serializable (got error: {e}). "
                f"Convert paths and other objects to strings first."
            ) from e

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")

    checker = _KeyUniquenessChecker()
    writer: pq.ParquetWriter | None = None
    n_written = 0
    try:
        for start in range(0, df.height, chunk_rows):
            sub = df.slice(start, chunk_rows).select(list(_KEY_COLUMNS))
            if keep is not None:
                k = keep[start : start + sub.height]
                if not k.any():
                    continue
                sub = sub.filter(pl.Series(k))
            if sub.is_empty():  # pragma: no cover
                continue
            checker.update(sub)
            table = sub.to_arrow()
            if writer is None:
                schema = table.schema.with_metadata(
                    {**(table.schema.metadata or {}), **metadata}
                )
                writer = pq.ParquetWriter(tmp_path, schema, compression="zstd")
            writer.write_table(table)
            n_written += sub.height
        if writer is not None:
            writer.close()
            writer = None
        if n_written == 0:  # pragma: no cover
            raise ValueError("ERROR: Mask empty.")
        checker.finalize(n_written, tmp_path)
        _atomic_replace(tmp_path, cache_path)
    except BaseException:  # pragma: no cover
        if writer is not None:
            writer.close()
        tmp_path.unlink(missing_ok=True)
        raise

    if verbose:
        print(f"Saved {n_written:,} mask-keep keys to {cache_path}")


class _KeyUniquenessChecker:
    """
    Streaming uniqueness check for ``(layer, Start time)`` keys.

    Fast path: AMPM data arrives layer-contiguous, so each layer forms one
    run of consecutive rows. Uniqueness then only needs a per-run
    ``n_unique`` over that run's ``Start time`` values — memory bounded by
    the largest single layer, not the whole build.
    """

    def __init__(self) -> None:
        self._completed: set[int] = set()
        self._cur_layer: int | None = None
        self._cur_pieces: list[pl.Series] = []
        self._needs_global = False

    def update(self, sub: pl.DataFrame) -> None:
        if self._needs_global:
            return
        layers = sub[_KEY_COLUMNS[0]].to_numpy()
        starts = sub[_KEY_COLUMNS[1]]
        boundaries = np.flatnonzero(layers[1:] != layers[:-1]) + 1
        bounds = np.concatenate([[0], boundaries, [len(layers)]])
        for a, b in zip(bounds[:-1], bounds[1:]):
            layer_n = int(layers[a])
            if self._cur_layer is not None and layer_n == self._cur_layer:
                self._cur_pieces.append(starts.slice(int(a), int(b - a)))
                continue
            self._close_run()
            if layer_n in self._completed:
                self._needs_global = True
                self._cur_pieces = []
                self._cur_layer = None
                return
            self._cur_layer = layer_n
            self._cur_pieces = [starts.slice(int(a), int(b - a))]

    def _close_run(self) -> None:
        if self._cur_layer is None:
            return
        run = (
            pl.concat(self._cur_pieces)
            if len(self._cur_pieces) > 1
            else self._cur_pieces[0]
        )
        n_unique = run.n_unique()
        if n_unique != run.len():
            raise ValueError(
                f"(layer, Start time) is not unique across rows: layer "
                f"{self._cur_layer} has {run.len():,} rows but only "
                f"{n_unique:,} distinct 'Start time' values. This cache "
                f"format requires unique row keys."
            )
        self._completed.add(self._cur_layer)
        self._cur_layer = None
        self._cur_pieces = []

    def finalize(self, n_total: int, written_path: Path) -> None:
        """Close the last run; run the global fallback check if needed."""
        self._close_run()
        if not self._needs_global:
            return
        n_unique = (
            pl.scan_parquet(written_path, glob=False)
            .select(pl.struct(list(_KEY_COLUMNS)).n_unique().alias("n"))
            .collect(engine="streaming")
            .item()
        )
        if n_unique != n_total:
            raise ValueError(
                f"(layer, Start time) is not unique across rows: "
                f"{n_total:,} rows but only {n_unique:,} distinct keys. "
                f"This cache format requires unique row keys."
            )


def load_mask_keep(
    df_full: pl.DataFrame,
    cache_path: str | Path,
    *,
    expect_params: dict | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Load the mask-keep keys and return ``df_full`` filtered to just the
    rows whose ``(layer, Start time)`` is present in the cache.

    Parameters
    ----------
    df_full
        The pre-mask DataFrame. Must contain ``layer`` and ``Start time``.
    cache_path
        Path to a cache previously written by ``save_mask_keep``.
    expect_params
        If given, the cache's stored params must match exactly.
    strict
        If True (default), any mismatch raises (version, params, missing).
        If False, mismatches raise ``FileNotFoundError`` so the caller can
        fall back to recomputing.
    verbose
        Print a one-line confirmation when load succeeds.

    Returns
    -------
    Filtered DataFrame.
    """
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        msg = f"Mask cache not found:\n{cache_path}"
        if strict:
            raise FileNotFoundError(msg)
        if verbose:
            print(f"  [mask_cache] {msg}")
        raise FileNotFoundError(msg)

    for c in _KEY_COLUMNS:
        if c not in df_full.columns:
            raise KeyError(f"Column {c!r} required to load mask cache.")

    pf = pq.ParquetFile(cache_path)
    raw_meta = pf.schema_arrow.metadata or {}
    _num_rows = pf.metadata.num_rows
    _empty_schema = pf.schema_arrow.empty_table()
    del pf  # stops WinError 5

    version_bytes = raw_meta.get(_META_VERSION_KEY)
    if version_bytes is None:
        msg = (
            f"Cache file {cache_path} has no version metadata; refusing to "
            f"load (likely produced by a different tool)."
        )
        if strict:
            raise ValueError(msg)
        if verbose:
            print(f"  [mask_cache] {msg}")
        raise FileNotFoundError(msg)
    version = int(version_bytes.decode())
    if version != CACHE_FORMAT_VERSION:
        msg = (
            f"Cache format version {version} != expected "
            f"{CACHE_FORMAT_VERSION}: rebuild required."
        )
        if strict:
            raise ValueError(msg)
        if verbose:
            print(f"  [mask_cache] {msg}")
        raise FileNotFoundError(msg)

    if expect_params is not None:
        params_bytes = raw_meta.get(_META_PARAMS_KEY)
        cached_params = json.loads(params_bytes.decode()) if params_bytes else None
        normalized_expected = json.loads(
            json.dumps(expect_params, sort_keys=True, default=str)
        )
        if cached_params != normalized_expected:
            diff = _format_param_diff(cached_params, expect_params)
            msg = f"Mask cache params don't match expected:\n{diff}"
            if strict:
                raise ValueError(msg)
            if verbose:
                print(f"  [mask_cache] {msg}")
            raise FileNotFoundError(msg)

    if _num_rows == 0:
        msg = f"Mask cache contains 0 keys:\n{cache_path}"
        if strict:
            raise ValueError(msg)
        if verbose:
            print(f"  [mask_cache] {msg}")
        raise FileNotFoundError(msg)

    # Align dtypes to cache schema without loading data
    cached_schema = pl.from_arrow(_empty_schema)
    cast_exprs = []
    for k in _KEY_COLUMNS:
        target_dtype = cached_schema[k].dtype
        if df_full[k].dtype != target_dtype:
            cast_exprs.append(pl.col(k).cast(target_dtype))
    if cast_exprs:
        df_to_filter = df_full.with_columns(cast_exprs)
    else:
        df_to_filter = df_full

    with phase("mask: keep-array from cached keys (merge-walk)"):
        keep = _keep_from_cached_keys(df_to_filter, cache_path)
    with phase("mask: materialize filtered DataFrame (cache hit)"):
        out = df_to_filter.filter(pl.Series(keep))

    if out.is_empty() and not df_full.is_empty():
        msg = f"Mask cache matched 0 of {df_full.height:,} rows.\n" f"{cache_path}"
        if strict:
            raise ValueError(msg)
        if verbose:
            print(f"  [mask_cache] {msg}")
        raise FileNotFoundError(msg)

    if verbose:
        kept = out.height
        total = df_full.height
        print(
            f"Loaded mask-keep from {cache_path}: "
            f"{kept:,}/{total:,} rows kept ({kept/total:.1%})"
        )
    return out


def _keep_from_cached_keys(df: pl.DataFrame, cache_path: Path) -> np.ndarray:
    """
    Compute the boolean keep-array for ``df`` against the cached keys
    without building any whole-build hash structure.

    Parameters
    ----------
    df
        Pre-mask DataFrame, key columns already cast to the cache dtypes.
    cache_path
        Path to the keys Parquet file.

    Returns
    -------
    ndarray of bool, shape (df.height,)
    """
    layer_col, time_col = _KEY_COLUMNS
    n = df.height
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep

    layers = df[layer_col].to_numpy()
    times = df[time_col]

    boundaries = np.flatnonzero(layers[1:] != layers[:-1]) + 1
    bounds = np.concatenate([[0], boundaries, [n]])
    runs_by_layer: dict[int, list[tuple[int, int]]] = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        runs_by_layer.setdefault(int(layers[a]), []).append((int(a), int(b)))

    def _apply(layer_n: int, pieces: list[pl.Series]) -> None:
        runs = runs_by_layer.get(layer_n)
        if not runs:
            return
        keys = pl.concat(pieces) if len(pieces) > 1 else pieces[0]
        for a, b in runs:
            keep[a:b] = times.slice(a, b - a).is_in(keys.implode()).to_numpy()

    pf = pq.ParquetFile(cache_path)
    cur_layer: int | None = None
    pieces: list[pl.Series] = []
    ascending = True
    for batch in pf.iter_batches(columns=list(_KEY_COLUMNS), batch_size=2_000_000):
        b_layers = batch.column(0).to_numpy(zero_copy_only=False)
        b_times = pl.Series(time_col, batch.column(1))
        b_bounds = np.concatenate(
            [[0], np.flatnonzero(b_layers[1:] != b_layers[:-1]) + 1, [len(b_layers)]]
        )
        for a, b in zip(b_bounds[:-1], b_bounds[1:]):
            layer_n = int(b_layers[a])
            if cur_layer is not None and layer_n < cur_layer:
                ascending = False
                break
            if layer_n != cur_layer:
                if cur_layer is not None:
                    _apply(cur_layer, pieces)
                cur_layer = layer_n
                pieces = []
            pieces.append(b_times.slice(int(a), int(b - a)))
        if not ascending:
            break
    if ascending:
        if cur_layer is not None:
            _apply(cur_layer, pieces)
        return keep

    keep[:] = False
    lazy_cache = pl.scan_parquet(cache_path, glob=False)
    for layer_n in runs_by_layer:
        keys = (
            lazy_cache.filter(pl.col(layer_col) == layer_n)
            .select(time_col)
            .collect()[time_col]
        )
        if not keys.is_empty():
            _apply(layer_n, [keys])
    return keep


def _format_param_diff(cached: dict | None, expected: dict) -> str:
    """Pretty-print which params differ between cache and expected."""
    if cached is None:
        return f"  (cache has no params metadata; expected {expected})"
    lines = []
    expected_norm = json.loads(json.dumps(expected, sort_keys=True, default=str))
    all_keys = sorted(set(cached) | set(expected_norm))
    for k in all_keys:
        cache = cached.get(k, "<missing>")
        exp = expected_norm.get(k, "<missing>")
        if cache != exp:
            lines.append(f"  {k}: cache={cache!r}, expected={exp!r}")
    return "\n".join(lines) if lines else "  (no field-level differences detected)"


def mask_or_load(
    df_full: pl.DataFrame,
    cache_path: str | Path,
    mask_fn: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
    *,
    keep_fn: Callable[[pl.DataFrame], np.ndarray | None] | None = None,
    params: dict,
    strict: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Load the masked DataFrame from cache if available and matching ``params``;
    otherwise compute, save the result, and return it.

    Parameters
    ----------
    df_full
        Pre-mask DataFrame.
    cache_path
        Where the cache lives.
    mask_fn
        Callable taking the DataFrame and returning the masked DataFrame;
        ignored when ``keep_fn`` is given.
    keep_fn
        Preferred, bounded-memory path: callable taking the DataFrame and
        returning the boolean keep-array (or None for an empty input), e.g.
        ``lambda d: apply_mask_keep(d, mask)``.
    params
        Parameters describing this masking run. Stored on save, checked on
        load. Must be JSON-serializable.
    strict
        Forwarded to ``load_mask_keep``. If True (default), version or
        params mismatch raises. If False, mismatches fall through to
        recomputation.
    verbose
        Print loading / computing / saving messages.
    """
    if mask_fn is None and keep_fn is None:
        raise TypeError("mask_or_load requires mask_fn or keep_fn.")

    try:
        return load_mask_keep(
            df_full,
            cache_path,
            expect_params=params,
            strict=strict,
            verbose=verbose,
        )
    except FileNotFoundError:
        if verbose:
            print(f"  [mask_cache] computing fresh mask:\n{cache_path}")

        if keep_fn is not None:
            with phase("mask: compute keep array (apply_mask_keep)"):
                keep = keep_fn(df_full)
            if keep is None:  # empty input
                return df_full
            keep = np.asarray(keep, dtype=bool)
            if not keep.any() and not df_full.is_empty():
                raise RuntimeError(f"Masking kept 0 of {df_full.height:,} rows.")
            with phase("mask: save keep cache (streaming write)"):
                save_mask_keep_from_keep(
                    df_full, keep, cache_path, params=params, verbose=verbose
                )
            with phase("mask: materialize filtered DataFrame"):
                return df_full.filter(pl.Series(keep))

        with phase("mask: legacy mask_fn (filter inside)"):
            masked = mask_fn(df_full)
        if masked.is_empty() and not df_full.is_empty():
            raise RuntimeError(f"Masking kept 0 of {df_full.height:,} rows.")
        save_mask_keep(masked, cache_path, params=params, verbose=verbose)
        return masked
