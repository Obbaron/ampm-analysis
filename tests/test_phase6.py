"""
Tests for cluster_dbscan_chunked.

We synthesize known cluster structure (parts as columns spanning many layers)
and verify chunked DBSCAN recovers it without fragmentation across chunk
boundaries.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from ampm.clustering import (
    cluster_dbscan,
    cluster_dbscan_chunked,
    cluster_summary,
)


def make_columnar_parts(
    n_parts: int = 4,
    points_per_layer_per_part: int = 200,
    layers: int = 100,
    layer_thickness: float = 0.03,
    part_xy_radius: float = 1.5,
    part_separation: float = 8.0,
    seed: int = 0,
) -> tuple[pl.DataFrame, np.ndarray]:
    """
    n_parts vertical columns at distinct XY centers, each spanning all layers.
    Returns (df, ground_truth_part_ids).
    """
    rng = np.random.default_rng(seed)
    rows = []
    truth = []
    for part_id in range(n_parts):
        cx = part_id * part_separation
        cy = 0.0
        for layer_n in range(1, layers + 1):
            for _ in range(points_per_layer_per_part):
                x = cx + rng.uniform(-part_xy_radius, part_xy_radius)
                y = cy + rng.uniform(-part_xy_radius, part_xy_radius)
                z = layer_n * layer_thickness
                rows.append(
                    {
                        "Demand X": x,
                        "Demand Y": y,
                        "Z": z,
                        "layer": layer_n,
                    }
                )
                truth.append(part_id)
    return pl.DataFrame(rows), np.array(truth, dtype=np.int32)


def test_basic_recovery_3d() -> None:
    """No chunk boundary needed (single chunk covers everything)."""
    df, truth = make_columnar_parts(n_parts=3, layers=20)
    out = cluster_dbscan_chunked(
        df,
        eps_xy=1.0,
        eps_z=0.1,
        min_samples=10,
        mode="3d",
        layers_per_chunk=50,
        overlap_layers=10,
        layer_thickness=0.03,
        verbose=False,
    )
    labels = out["cluster"].to_numpy()
    n_clusters = len({L for L in labels if L >= 0})
    assert n_clusters == 3, f"Expected 3 clusters, got {n_clusters}"
    # No fragmentation: each ground-truth part should map to exactly one cluster.
    for part_id in range(3):
        cluster_ids = set(labels[truth == part_id])
        cluster_ids.discard(-1)
        assert len(cluster_ids) == 1, f"Part {part_id} fragmented across {cluster_ids}"
    print("  basic 3d recovery (single chunk) OK")


def test_multi_chunk_no_fragmentation() -> None:
    """
    Force multiple chunks with overlap. Verify each part remains a single
    cluster despite spanning chunk boundaries.
    """
    df, truth = make_columnar_parts(n_parts=3, layers=120)
    # 120 layers / (chunk 50 - overlap 10) = (120 - 50) / 40 + 1 = ~3 chunks
    out = cluster_dbscan_chunked(
        df,
        eps_xy=1.0,
        eps_z=0.1,
        min_samples=10,
        mode="3d",
        layers_per_chunk=50,
        overlap_layers=10,
        layer_thickness=0.03,
        verbose=False,
    )
    labels = out["cluster"].to_numpy()
    n_clusters = len({L for L in labels if L >= 0})
    assert n_clusters == 3, f"Expected 3 clusters across chunks, got {n_clusters}"
    for part_id in range(3):
        cluster_ids = set(labels[truth == part_id])
        cluster_ids.discard(-1)
        assert (
            len(cluster_ids) == 1
        ), f"Part {part_id} fragmented across chunks: {cluster_ids}"
    print("  multi-chunk no fragmentation OK")


def test_default_overlap_calculation() -> None:
    """If overlap_layers=None, it should be auto-computed."""
    df, _ = make_columnar_parts(n_parts=2, layers=80)
    out = cluster_dbscan_chunked(
        df,
        eps_xy=1.0,
        eps_z=0.15,
        min_samples=10,
        mode="3d",
        layers_per_chunk=40,
        overlap_layers=None,  # auto
        layer_thickness=0.03,
        verbose=False,
    )
    labels = out["cluster"].to_numpy()
    n_clusters = len({L for L in labels if L >= 0})
    assert n_clusters == 2, n_clusters
    print("  default overlap calc OK")


def test_overlap_clamping() -> None:
    """overlap_layers=2 with eps_z=0.15 needs clamping up to ~10."""
    df, _ = make_columnar_parts(n_parts=2, layers=80)
    # Run with too-small overlap — should warn and clamp.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = cluster_dbscan_chunked(
            df,
            eps_xy=1.0,
            eps_z=0.15,
            min_samples=10,
            mode="3d",
            layers_per_chunk=40,
            overlap_layers=2,  # too small
            layer_thickness=0.03,
            verbose=True,
        )
    output = buf.getvalue()
    assert "WARNING" in output and "Clamping" in output, output
    # And it should still produce sensible clusters.
    labels = out["cluster"].to_numpy()
    assert len({L for L in labels if L >= 0}) == 2
    print("  overlap clamping warns and works OK")


def test_overlap_too_large_raises() -> None:
    df, _ = make_columnar_parts(n_parts=2, layers=80)
    try:
        cluster_dbscan_chunked(
            df,
            eps_xy=1.0,
            eps_z=0.1,
            min_samples=10,
            mode="3d",
            layers_per_chunk=20,
            overlap_layers=20,  # >= chunk size
            layer_thickness=0.03,
            verbose=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    print("  overlap >= chunk size raises OK")


def test_2d_mode() -> None:
    df, truth = make_columnar_parts(n_parts=4, layers=50)
    out = cluster_dbscan_chunked(
        df,
        eps_xy=2.0,
        min_samples=10,
        mode="2d",
        layers_per_chunk=30,
        overlap_layers=5,  # need overlap to merge across chunks
        verbose=False,
    )
    labels = out["cluster"].to_numpy()
    n_clusters = len({L for L in labels if L >= 0})
    assert n_clusters == 4, n_clusters
    print("  2d mode OK")


def test_stable_labels_centroid_order() -> None:
    df, _ = make_columnar_parts(n_parts=3, layers=30)
    out = cluster_dbscan_chunked(
        df,
        eps_xy=1.0,
        eps_z=0.1,
        min_samples=10,
        mode="3d",
        layers_per_chunk=40,
        verbose=False,
    )
    summary = cluster_summary(out).filter(pl.col("cluster") >= 0)
    x_means = summary["x_mean"].to_list()
    assert x_means == sorted(x_means), x_means
    print("  stable_labels centroid order OK")


def test_deterministic() -> None:
    df, _ = make_columnar_parts(n_parts=3, layers=80)
    a = cluster_dbscan_chunked(
        df,
        eps_xy=1.0,
        eps_z=0.1,
        min_samples=10,
        mode="3d",
        layers_per_chunk=40,
        overlap_layers=10,
        verbose=False,
    )
    b = cluster_dbscan_chunked(
        df,
        eps_xy=1.0,
        eps_z=0.1,
        min_samples=10,
        mode="3d",
        layers_per_chunk=40,
        overlap_layers=10,
        verbose=False,
    )
    assert a["cluster"].to_list() == b["cluster"].to_list()
    print("  deterministic OK")


def test_handles_layer_gaps() -> None:
    """Skip some layers — chunked clustering shouldn't crash."""
    df, _ = make_columnar_parts(n_parts=2, layers=60)
    # Drop layers 30-40 to simulate missing data
    df = df.filter((pl.col("layer") < 30) | (pl.col("layer") > 40))
    out = cluster_dbscan_chunked(
        df,
        eps_xy=1.0,
        eps_z=0.5,  # generous eps_z to bridge gap
        min_samples=10,
        mode="3d",
        layers_per_chunk=80,
        overlap_layers=40,  # large overlap to fit min_overlap
        verbose=False,
    )
    labels = out["cluster"].to_numpy()
    n_clusters = len({L for L in labels if L >= 0})
    # With a 10-layer gap and eps_z=0.5, parts may or may not bridge —
    # we just want no crash and reasonable output.
    assert n_clusters >= 2
    print(f"  layer gaps handled (got {n_clusters} clusters) OK")


def test_3d_requires_eps_z() -> None:
    df, _ = make_columnar_parts(n_parts=2, layers=20)
    try:
        cluster_dbscan_chunked(df, eps_xy=1.0, mode="3d", verbose=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    print("  3d requires eps_z OK")


def test_unknown_layer_col_raises() -> None:
    df, _ = make_columnar_parts(n_parts=2, layers=20)
    try:
        cluster_dbscan_chunked(
            df,
            eps_xy=1.0,
            eps_z=0.1,
            mode="3d",
            layer_col="bogus",
            verbose=False,
        )
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    print("  unknown layer_col raises OK")


def test_match_with_non_chunked() -> None:
    """
    Sanity check: when both methods get full data and the data is small enough
    to not need chunking, results should match in cluster *count* (label
    values may differ but stable_labels makes them comparable).
    """
    df, truth = make_columnar_parts(n_parts=4, layers=30)
    chunked = cluster_dbscan_chunked(
        df,
        eps_xy=1.0,
        eps_z=0.1,
        min_samples=10,
        mode="3d",
        layers_per_chunk=100,  # one big chunk
        verbose=False,
    )
    non_chunked = cluster_dbscan(
        df,
        eps_xy=1.0,
        eps_z=0.1,
        min_samples=10,
        mode="3d",
        representative_size=df.height,  # no downsampling
        seed=0,
    )
    n_chunked = len({L for L in chunked["cluster"].to_list() if L >= 0})
    n_full = len({L for L in non_chunked["cluster"].to_list() if L >= 0})
    assert n_chunked == n_full == 4, (n_chunked, n_full)
    # And both should perfectly recover ground truth.
    for part_id in range(4):
        sets = []
        for lbls in (chunked["cluster"].to_numpy(), non_chunked["cluster"].to_numpy()):
            ids = set(lbls[truth == part_id])
            ids.discard(-1)
            sets.append(ids)
        assert len(sets[0]) == 1 and len(sets[1]) == 1
    print("  agrees with non-chunked on small data OK")


def test_dtype() -> None:
    df, _ = make_columnar_parts(n_parts=2, layers=20)
    out = cluster_dbscan_chunked(
        df,
        eps_xy=1.0,
        eps_z=0.1,
        mode="3d",
        layers_per_chunk=30,
        verbose=False,
    )
    assert out["cluster"].dtype == pl.Int32
    print("  cluster column dtype Int32 OK")


def main() -> None:
    print("Phase 6 chunked-clustering tests:")
    test_basic_recovery_3d()
    test_multi_chunk_no_fragmentation()
    test_default_overlap_calculation()
    test_overlap_clamping()
    test_overlap_too_large_raises()
    test_2d_mode()
    test_stable_labels_centroid_order()
    test_deterministic()
    test_handles_layer_gaps()
    test_3d_requires_eps_z()
    test_unknown_layer_col_raises()
    test_match_with_non_chunked()
    test_dtype()
    print("\nAll Phase 6 tests passed")


if __name__ == "__main__":
    main()
