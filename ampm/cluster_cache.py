"""
Persistence for cluster labels so expensive clustering runs aren't wasted.

The cache stores three columns — ``(layer, Start time, cluster)`` — alongside
JSON-serializable parameter metadata. On load, we look up each row in the
fresh DataFrame by ``(layer, Start time)`` and merge in the cluster label.

Why store keys instead of the full DataFrame?
- It's smaller (~3 columns instead of 19+).
- It's robust: if you query a different layer range or change the mask, the
  load step only labels rows it recognizes and leaves the rest as -1, instead
  of silently giving you the wrong labels.

Why store params metadata?
- So you don't accidentally load labels computed under different DBSCAN
  parameters than what you're about to use. Strict comparison is the default.

Typical usage
-------------
    params = {
        "layers": (1, 434),
        "stl": str(STL),
        "buffer_mm": 0.0,
        "eps_xy": 0.3, "eps_z": 0.06,
        "min_samples": 10, "mode": "3d",
        "layers_per_chunk": 20, "overlap_layers": None,
    }
    clustered = cluster_or_load(
        df_masked,
        cache_path="clusters.pq",
        cluster_fn=lambda d: cluster_dbscan_chunked(d, eps_xy=0.3, eps_z=0.06, ...),
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
_LABEL_COLUMN = "cluster"

# Parquet file-level metadata keys (must be bytes for pyarrow).
_META_VERSION_KEY = b"ampm_cluster_cache_version"
_META_PARAMS_KEY = b"ampm_cluster_cache_params"


def save_cluster_labels(
    clustered: pl.DataFrame,
    cache_path: str | Path,
    *,
    params: dict | None = None,
    verbose: bool = True,
) -> None:
    """
    Persist cluster labels to ``cache_path`` keyed by ``(layer, Start time)``.

    Parameters
    ----------
    clustered
        Output of ``cluster_dbscan`` or ``cluster_dbscan_chunked`` — must
        contain ``layer``, ``Start time``, and ``cluster`` columns.
    cache_path
        Where to write the cache (a single Parquet file).
    params
        Optional dict of parameters used to produce ``clustered``. Must be
        JSON-serializable. Stored in Parquet file metadata so ``load`` can
        verify the cache is still relevant.
    verbose
        Print a one-line confirmation. Default True.
    """
    cache_path = Path(cache_path)

    for c in (*_KEY_COLUMNS, _LABEL_COLUMN):
        if c not in clustered.columns:
            raise KeyError(
                f"Column {c!r} required for cluster cache; missing from input."
            )

    keys = clustered.select([*_KEY_COLUMNS, _LABEL_COLUMN])

    n_unique = keys.select(_KEY_COLUMNS).n_unique()
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
                params, sort_keys=True, default=str
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
        n = clustered.height
        n_clusters = (
            clustered.filter(pl.col(_LABEL_COLUMN) >= 0)
            .select(pl.col(_LABEL_COLUMN).n_unique())
            .item()
        )
        print(f"Saved {n:,} cluster labels ({n_clusters} clusters) " f"to {cache_path}")


def load_cluster_labels(
    df: pl.DataFrame,
    cache_path: str | Path,
    *,
    expect_params: dict | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Load cluster labels from ``cache_path`` and join them into ``df`` on
    ``(layer, Start time)``.

    Rows in ``df`` whose key isn't in the cache get ``cluster = -1``.

    Parameters
    ----------
    df
        DataFrame to label. Must contain ``layer`` and ``Start time``.
    cache_path
        Path to a cache previously written by ``save_cluster_labels``.
    expect_params
        If given, the cache's stored params dict must match this exactly.
        Behavior on mismatch is controlled by ``strict``.
    strict
        If True (default), raise on any mismatch (version, params, missing
        file). If False, mismatches print a warning and the function raises
        ``FileNotFoundError`` so the caller can fall back to recomputing.
    verbose
        Print a one-line confirmation when a load succeeds.

    Returns
    -------
    DataFrame: ``df`` plus a ``cluster`` Int32 column.
    """
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        msg = f"Cluster cache not found: {cache_path}"
        if strict:
            raise FileNotFoundError(msg)
        if verbose:
            print(f"  [cache] {msg}")
        raise FileNotFoundError(msg)

    for c in _KEY_COLUMNS:
        if c not in df.columns:
            raise KeyError(f"Column {c!r} required to load cluster cache.")

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
            print(f"  [cache] {msg}")
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
            print(f"  [cache] {msg}")
        raise FileNotFoundError(msg)

    if expect_params is not None:
        params_bytes = raw_meta.get(_META_PARAMS_KEY)
        cached_params = json.loads(params_bytes.decode()) if params_bytes else None
        if cached_params != json.loads(
            json.dumps(expect_params, sort_keys=True, default=str)
        ):
            diff = _format_param_diff(cached_params, expect_params)
            msg = f"Cache params don't match expected:\n{diff}"
            if strict:
                raise ValueError(msg)
            if verbose:
                print(f"  [cache] {msg}")
            raise FileNotFoundError(msg)

    cached = pl.read_parquet(cache_path, glob=False)

    cast_exprs = []
    for k in _KEY_COLUMNS:
        target_dtype = cached[k].dtype
        if df[k].dtype != target_dtype:
            cast_exprs.append(pl.col(k).cast(target_dtype))
    if cast_exprs:
        df_to_join = df.with_columns(cast_exprs)
    else:
        df_to_join = df

    out = df_to_join.join(cached, on=list(_KEY_COLUMNS), how="left").with_columns(
        pl.col(_LABEL_COLUMN).fill_null(-1).cast(pl.Int32)
    )

    if verbose:
        labelled = (out[_LABEL_COLUMN] != -1).sum()
        n_clusters = (
            out.filter(pl.col(_LABEL_COLUMN) >= 0)
            .select(pl.col(_LABEL_COLUMN).n_unique())
            .item()
        )
        print(
            f"Loaded cluster labels from {cache_path}: "
            f"{labelled:,}/{out.height:,} rows labelled, "
            f"{n_clusters} clusters"
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


def cluster_or_load(
    df: pl.DataFrame,
    cache_path: str | Path,
    cluster_fn: Callable[[pl.DataFrame], pl.DataFrame],
    *,
    params: dict,
    strict: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Load cluster labels from ``cache_path`` if present and matching ``params``;
    otherwise call ``cluster_fn(df)`` to compute, save the result, and return it.

    Parameters
    ----------
    df
        DataFrame to label.
    cache_path
        Where the cache lives.
    cluster_fn
        Callable taking the DataFrame and returning a DataFrame with a new
        ``cluster`` column. E.g. ``lambda d: cluster_dbscan_chunked(d, ...)``.
    params
        Parameters describing this clustering run. Stored on save, checked
        on load. Must be JSON-serializable.
    strict
        Forwarded to ``load_cluster_labels``. If True (default), any version
        or params mismatch raises. If False, mismatches fall through to
        recomputation.
    verbose
        Print loading / computing / saving messages.
    """
    try:
        return load_cluster_labels(
            df,
            cache_path,
            expect_params=params,
            strict=strict,
            verbose=verbose,
        )
    except FileNotFoundError:
        if verbose:
            print(f"  [cache] computing fresh clusters -> {cache_path}")
        clustered = cluster_fn(df)
        save_cluster_labels(clustered, cache_path, params=params, verbose=verbose)
        return clustered
