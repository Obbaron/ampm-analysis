"""
Shared pytest fixtures for the AMPM test suite.

The only cross-module fixture is :func:`keyed_df`, a builder for small Polars
frames with the ``(layer, Start time)`` key schema that both ``cluster_cache``
and ``mask_cache`` are built around. Module-specific fixtures live in their own
test files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def keyed_df():
    """
    Build a Polars frame keyed by ``(layer, Start time)``.

    Parameters of the returned builder
    ----------------------------------
    keys : list[tuple[int, int]]
        ``(layer, start_time)`` pairs, one per row, in row order.
    cluster : list[int] | None
        Optional per-row cluster labels (Int32 column ``"cluster"``).
    layer_dtype, time_dtype : polars dtypes
        Dtypes for the key columns. Defaults match DataStore output
        (layer Int16, Start time Int32).
    extra : dict[str, list] | None
        Any additional columns to attach.
    """

    def _build(
        keys, *, cluster=None, layer_dtype=pl.Int16, time_dtype=pl.Int32, extra=None
    ):
        data = {
            "layer": pl.Series("layer", [int(k[0]) for k in keys], dtype=layer_dtype),
            "Start time": pl.Series(
                "Start time", [int(k[1]) for k in keys], dtype=time_dtype
            ),
        }
        if cluster is not None:
            data["cluster"] = pl.Series("cluster", list(cluster), dtype=pl.Int32)
        if extra:
            for name, values in extra.items():
                data[name] = pl.Series(name, list(values))
        return pl.DataFrame(data)

    return _build
