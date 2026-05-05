"""
Tests for ampm.clustering.

Synthesize known cluster structure (LPBF-shaped: parts separated in XY,
densely packed in Z) and verify DBSCAN recovers it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import time

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from ampm.clustering import (
    cluster_dbscan,
    cluster_summary,
    k_distance_curve,
)


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
def make_three_part_data(
    points_per_part: int = 5_000,
    layers: int = 50,
    layer_thickness: float = 0.03,
    seed: int = 0,
) -> tuple[pl.DataFrame, np.ndarray]:
    """
    Three "parts" arranged as columns at different XY positions, each spanning
    `layers` layers in Z. Within each layer, points are uniformly spread in a
    small XY area. Returns (df, ground_truth_labels).
    """
    rng = np.random.default_rng(seed)
    centers = [(-15, -15), (0, 0), (15, 15)]
    half_extent = 3.0  # each part is ~6mm x 6mm

    rows = []
    truth = []
    for part_id, (cx, cy) in enumerate(centers):
        for layer_n in range(1, layers + 1):
            for _ in range(points_per_part // layers):
                x = cx + rng.uniform(-half_extent, half_extent)
                y = cy + rng.uniform(-half_extent, half_extent)
                z = layer_n * layer_thickness
                rows.append({"Demand X": x, "Demand Y": y, "Z": z, "layer": layer_n})
                truth.append(part_id)
    return pl.DataFrame(rows), np.array(truth)


def make_well_separated_2d(seed: int = 0) -> pl.DataFrame:
    """Three obvious clusters in the XY plane at a single Z."""
    rng = np.random.default_rng(seed)
    parts = []
    for cx, cy in [(-10, -10), (10, -10), (0, 10)]:
        x = cx + rng.normal(0, 0.5, 1000)
        y = cy + rng.normal(0, 0.5, 1000)
        parts.append(pl.DataFrame({
            "Demand X": x, "Demand Y": y, "Z": np.full(1000, 1.5),
        }))
    return pl.concat(parts)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_k_distance_curve_shape() -> None:
    df, _ = make_three_part_data()
    out = k_distance_curve(df, k=10, sample_size=2000, mode="3d",
                           eps_xy=1.0, eps_z=0.1, seed=0)
    assert out.columns == ["Rank", "k-distance (mm)"]
    assert out.height == 2000
    # Curve must be monotonically non-decreasing.
    d = out["k-distance (mm)"].to_numpy()
    assert np.all(np.diff(d) >= 0), "k-distance curve must be sorted"
    # And there should be a clear range — not all zeros, not all huge.
    assert d.min() >= 0
    assert d.max() > d[100]  # distinct distribution
    print(f"  k_distance_curve: monotonic, min={d.min():.3f}, max={d.max():.3f} OK")


def test_k_distance_2d_mode() -> None:
    df = make_well_separated_2d()
    out = k_distance_curve(df, k=10, sample_size=1000, mode="2d", eps_xy=1.0)
    d = out["k-distance (mm)"].to_numpy()
    assert np.all(np.diff(d) >= 0)
    print("  k_distance_curve 2D mode OK")


def test_dbscan_recovers_three_parts_3d() -> None:
    df, truth = make_three_part_data(points_per_part=3_000, layers=20)
    out = cluster_dbscan(
        df,
        eps_xy=1.0, eps_z=0.06,  # 2x layer thickness
        min_samples=10, mode="3d",
        representative_size=10_000,
        seed=0,
    )
    labels = out["cluster"].to_numpy()
    n_clusters = len({L for L in labels if L >= 0})
    assert n_clusters == 3, f"Expected 3 clusters, got {n_clusters}"

    # Permutation-invariant accuracy: each ground-truth part should map to a
    # single cluster id (no part should fragment, no merging).
    for part_id in range(3):
        cluster_ids = set(labels[truth == part_id])
        cluster_ids.discard(-1)  # noise allowed at edges
        assert len(cluster_ids) == 1, (
            f"Part {part_id} fragmented across clusters {cluster_ids}"
        )
    print(f"  DBSCAN 3D recovers 3 separated parts OK")


def test_dbscan_2d_mode() -> None:
    df = make_well_separated_2d()
    out = cluster_dbscan(
        df, eps_xy=1.5, min_samples=10, mode="2d",
        representative_size=2000, seed=0,
    )
    labels = out["cluster"].to_numpy()
    n_clusters = len({L for L in labels if L >= 0})
    assert n_clusters == 3, n_clusters
    print("  DBSCAN 2D recovers 3 clusters OK")


def test_anisotropic_z_scaling_matters() -> None:
    """
    Build data where parts are 5mm apart in XY but only 0.5mm apart in Z.
    With isotropic eps=1.0, parts would merge across Z. With eps_xy=1.0,
    eps_z=0.1, Z is scaled up by 10x and parts stay separate.
    """
    rng = np.random.default_rng(0)
    rows = []
    truth = []
    # Two thin slabs at different Z, same XY footprint
    for part_id, z_center in enumerate([1.0, 1.5]):
        for _ in range(2000):
            rows.append({
                "Demand X": rng.uniform(-3, 3),
                "Demand Y": rng.uniform(-3, 3),
                "Z": z_center + rng.uniform(-0.05, 0.05),
            })
            truth.append(part_id)
    df = pl.DataFrame(rows)
    truth = np.array(truth)

    # With small eps_z the layers are pushed apart.
    out = cluster_dbscan(
        df, eps_xy=1.0, eps_z=0.1, min_samples=10, mode="3d",
        representative_size=4000, seed=0,
    )
    labels = out["cluster"].to_numpy()
    n_clusters = len({L for L in labels if L >= 0})
    assert n_clusters == 2, f"Anisotropy failed: got {n_clusters} clusters"

    # With large eps_z the layers should merge.
    out2 = cluster_dbscan(
        df, eps_xy=1.0, eps_z=2.0, min_samples=10, mode="3d",
        representative_size=4000, seed=0,
    )
    labels2 = out2["cluster"].to_numpy()
    n2 = len({L for L in labels2 if L >= 0})
    assert n2 == 1, f"Lenient eps_z should merge slabs, got {n2}"
    print("  anisotropic eps_xy/eps_z controls Z separability OK")


def test_stable_labels_deterministic() -> None:
    df, _ = make_three_part_data(points_per_part=2_000, layers=10)
    out1 = cluster_dbscan(df, eps_xy=1.0, eps_z=0.06, mode="3d",
                          representative_size=5000, seed=0,
                          stable_labels=True)
    out2 = cluster_dbscan(df, eps_xy=1.0, eps_z=0.06, mode="3d",
                          representative_size=5000, seed=0,
                          stable_labels=True)
    assert out1["cluster"].to_list() == out2["cluster"].to_list()

    # Stable label semantics: cluster 0's centroid X should be < cluster 1's, etc.
    summary = cluster_summary(out1)
    summary = summary.filter(pl.col("cluster") >= 0)
    x_means = summary["x_mean"].to_list()
    assert x_means == sorted(x_means), f"stable labels not centroid-sorted: {x_means}"
    print("  stable_labels deterministic + sorted by centroid OK")


def test_cluster_summary() -> None:
    df, _ = make_three_part_data(points_per_part=2_000, layers=10)
    out = cluster_dbscan(df, eps_xy=1.0, eps_z=0.06, mode="3d",
                        representative_size=5000, seed=0)
    s = cluster_summary(out)
    assert "cluster" in s.columns
    assert "n_rows" in s.columns
    assert "x_mean" in s.columns
    assert "z_mean" in s.columns
    # All parts together should sum to df.height
    assert s["n_rows"].sum() == df.height
    print(f"  cluster_summary OK: {s.height} clusters")


def test_2d_mode_no_eps_z() -> None:
    df = make_well_separated_2d()
    # 2D mode shouldn't require eps_z at all.
    out = cluster_dbscan(df, eps_xy=1.5, mode="2d",
                        representative_size=1000, seed=0)
    assert "cluster" in out.columns
    print("  2D mode without eps_z OK")


def test_3d_mode_requires_eps_z() -> None:
    df, _ = make_three_part_data()
    try:
        cluster_dbscan(df, eps_xy=1.0, mode="3d")  # missing eps_z
    except ValueError:
        pass
    else:
        raise AssertionError("3D mode should require eps_z")
    print("  3D mode requires eps_z OK")


def test_unknown_column_raises() -> None:
    df = make_well_separated_2d()
    try:
        cluster_dbscan(df, eps_xy=1.0, mode="2d",
                      columns=("bogus", "Demand Y", "Z"))
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    print("  unknown column raises OK")


def test_label_propagation_at_scale() -> None:
    """Build 500k points across 3 well-separated parts; ensure propagation
    correctly labels nearly all of them."""
    df, truth = make_three_part_data(points_per_part=170_000, layers=30)
    t0 = time.time()
    out = cluster_dbscan(
        df, eps_xy=1.5, eps_z=0.1, mode="3d",
        representative_size=20_000, seed=0,
    )
    dt = time.time() - t0

    labels = out["cluster"].to_numpy()
    # For each ground-truth part, the dominant cluster id should hold the
    # vast majority of its rows.
    correct = 0
    for part_id in range(3):
        mask = truth == part_id
        unique, counts = np.unique(labels[mask], return_counts=True)
        # Strip noise from majority calc
        keep = unique != -1
        if keep.any():
            correct += counts[keep].max()
    accuracy = correct / df.height
    assert accuracy > 0.97, f"propagation accuracy only {accuracy:.2%}"
    print(f"  label propagation: {df.height:,} rows in {dt:.2f}s, "
          f"accuracy={accuracy:.1%} OK")


def main() -> None:
    print("Phase 5 clustering tests:")
    test_k_distance_curve_shape()
    test_k_distance_2d_mode()
    test_dbscan_recovers_three_parts_3d()
    test_dbscan_2d_mode()
    test_anisotropic_z_scaling_matters()
    test_stable_labels_deterministic()
    test_cluster_summary()
    test_2d_mode_no_eps_z()
    test_3d_mode_requires_eps_z()
    test_unknown_column_raises()
    test_label_propagation_at_scale()
    print("\nAll Phase 5 tests passed")


if __name__ == "__main__":
    main()
