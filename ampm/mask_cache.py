"""
Persistence for the keep-mask produced by apply_mask().

The mask filter step is deterministic given (source data, STL, buffer_mm),
and slow at scale (~80M points × hundreds of polygons via shapely.contains).
We cache the result by storing the ``(layer, Start time)`` keys of the rows
that survived masking. On load, we filter the input DataFrame down to just
those keys.

Format mirrors ``ampm.cluster_cache``: a single Parquet file with two key
columns + JSON-serializable params metadata. Strict params comparison by
default — change any parameter and the cache invalidates loudly.

Typical usage
-------------
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
        params=params,
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import polars as pl
import pyarrow.parquet as pq

CACHE_FORMAT_VERSION = 1
_KEY_COLUMNS: tuple[str, str] = ("layer", "Start time")

_META_VERSION_KEY = b"ampm_mask_cache_version"
_META_PARAMS_KEY = b"ampm_mask_cache_params"


def save_mask_keep(
    df_masked: pl.DataFrame,
    cache_path: str | Path,
    *,
    params: dict | None = None,
    verbose: bool = True,
) -> None:
    """
    Persist the keys of the rows that survived masking.

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
    """
    cache_path = Path(cache_path)

    for c in _KEY_COLUMNS:
        if c not in df_masked.columns:
            raise KeyError(f"Column {c!r} required for mask cache; missing from input.")

    keys = df_masked.select(list(_KEY_COLUMNS))

    n_unique = keys.n_unique()
    if n_unique != keys.height:
        raise ValueError(
            f"(layer, Start time) is not unique across rows: "
            f"{keys.height:,} rows but only {n_unique:,} distinct keys. "
            f"This cache format requires unique row keys."
        )

    table = keys.to_arrow()
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
    existing = table.schema.metadata or {}
    table = table.replace_schema_metadata({**existing, **metadata})

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, cache_path, compression="zstd")

    if verbose:
        print(f"Saved {df_masked.height:,} mask-keep keys to {cache_path}")


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
        msg = f"Mask cache not found: {cache_path}"
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

    cached_keys = pl.read_parquet(cache_path, glob=False)

    cast_exprs = []
    for k in _KEY_COLUMNS:
        target_dtype = cached_keys[k].dtype
        if df_full[k].dtype != target_dtype:
            cast_exprs.append(pl.col(k).cast(target_dtype))
    if cast_exprs:
        df_to_filter = df_full.with_columns(cast_exprs)
    else:
        df_to_filter = df_full

    # Inner join semantics: keep rows whose (layer, Start time) is in the cache.
    # Using join with how="semi" is the fastest way to do this in polars as
    # it returns only matching left rows, no duplicate columns.
    out = df_to_filter.join(cached_keys, on=list(_KEY_COLUMNS), how="semi")

    if verbose:
        kept = out.height
        total = df_full.height
        print(
            f"Loaded mask-keep from {cache_path}: "
            f"{kept:,}/{total:,} rows kept ({kept/total:.1%})"
        )
    return out


def _format_param_diff(cached: dict | None, expected: dict) -> str:
    """Pretty-print which params differ between cache and expected."""
    if cached is None:
        return f"  (cache has no params metadata; expected {expected})"
    lines = []
    expected_norm = json.loads(json.dumps(expected, sort_keys=True, default=str))
    all_keys = sorted(set(cached) | set(expected_norm))
    for k in all_keys:
        c = cached.get(k, "<missing>")
        e = expected_norm.get(k, "<missing>")
        if c != e:
            lines.append(f"  {k}: cache={c!r}, expected={e!r}")
    return "\n".join(lines) if lines else "  (no field-level differences detected)"


def mask_or_load(
    df_full: pl.DataFrame,
    cache_path: str | Path,
    mask_fn: Callable[[pl.DataFrame], pl.DataFrame],
    *,
    params: dict,
    strict: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Load the masked DataFrame from cache if available and matching ``params``;
    otherwise call ``mask_fn(df_full)`` to compute, save the result, and
    return it.

    Parameters
    ----------
    df_full
        Pre-mask DataFrame.
    cache_path
        Where the cache lives.
    mask_fn
        Callable taking the DataFrame and returning the masked DataFrame.
        E.g. ``lambda d: apply_mask(d, mask)``.
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
            print(f"  [mask_cache] computing fresh mask → {cache_path}")
        masked = mask_fn(df_full)
        save_mask_keep(masked, cache_path, params=params, verbose=verbose)
        return masked
