"""
Tests for ampm.parts.

We synthesize multi-section QuantAM-style files with controlled content and
verify the parser extracts the right sections, columns, and values.
"""
from __future__ import annotations

import sys

import shutil
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from ampm.parts import QuantAMParts, apply_part_id_map, compute_part_id_map, join_parts_with_stats


# --------------------------------------------------------------------------- #
# Synthetic file builders
# --------------------------------------------------------------------------- #
def _make_minimal_file() -> str:
    """Two sections — Parent Parts and Scan Volume — with three parts each."""
    return (
        "#,Renishaw,Material,Development\n"
        ",Version,0.6.1\n"
        "\n"
        "#,Tab - -1,Parent Parts\n"
        '#,"Sr. No.","Source Index","Layer Thickness","X Position","Y Position","Layers Count","Part Type",\n'
        'ID.,"[T0C1]","[T0C2]","[T-1C1]","[T-1C2]","[T-1C3]","[T-1C4]","[T-1C5]",\n'
        ',"1","Part(1)","0.03","-26.787","-11.585","333","Geometry Part",\n'
        ',"2","Part(2)","0.03","-13.823","-10.59","333","Geometry Part",\n'
        ',"3","Part(3)","0.03","-0.505","-10.328","333","Geometry Part",\n'
        "\n"
        "#,Tab - 10,Scan Volume\n"
        '#,"Sr. No.","Source Index","Border Power","Hatches Power","Hatches Exposure Time",\n'
        'ID.,"[T0C1]","[T0C2]","[T10C1]","[T10C5]","[T10C8]",\n'
        ',"1.1","Part(1)","150","150","180",\n'
        ',"2.1","Part(2)","150","150","90",\n'
        ',"3.1","Part(3)","150","150","60",\n'
        ',"1.s","Part(1)","150","150","180",\n'
        ',"2.s","Part(2)","150","150","90",\n'
        ',"3.s","Part(3)","150","150","60",\n'
    )


def test_section_discovery() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="parts_test_"))
    try:
        f = tmp / "parts.csv"
        f.write_text(_make_minimal_file())
        parts = QuantAMParts.from_path(f)
        assert parts.section_names == ["Parent Parts", "Scan Volume"]
        assert "Parent Parts" in parts
        assert "Scan Volume" in parts
        assert "Bogus" not in parts
        print("  section discovery OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_parent_parts_table() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="parts_test_"))
    try:
        f = tmp / "parts.csv"
        f.write_text(_make_minimal_file())
        parts = QuantAMParts.from_path(f)
        pp = parts.parent_parts()
        assert pp.shape == (3, 5)
        assert pp.columns == ["Part ID", "Layer Thickness", "X Position",
                              "Y Position", "Layers Count"]
        # Check actual values — the first row should be Part(1) at (-26.787, -11.585)
        first = pp.row(0, named=True)
        assert first["Part ID"] == "Part(1)"
        assert abs(first["X Position"] - (-26.787)) < 1e-6
        assert first["Layers Count"] == 333
        print("  parent_parts table OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_volume_parameters_default_variant() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="parts_test_"))
    try:
        f = tmp / "parts.csv"
        f.write_text(_make_minimal_file())
        parts = QuantAMParts.from_path(f)
        vp = parts.volume_parameters()  # default variant="1"
        # Should have one row per part (3), joined with parent metadata.
        assert vp.height == 3
        assert "Hatches Exposure Time" in vp.columns
        assert "X Position" in vp.columns
        # Check the join worked — Part(1) should have its metadata + scan volume
        row = vp.filter(pl.col("Part ID") == "Part(1)").row(0, named=True)
        assert row["Hatches Exposure Time"] == 180
        assert abs(row["X Position"] - (-26.787)) < 1e-6
        print("  volume_parameters (variant=1) OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_volume_parameters_support_variant() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="parts_test_"))
    try:
        f = tmp / "parts.csv"
        f.write_text(_make_minimal_file())
        parts = QuantAMParts.from_path(f)
        sup = parts.volume_parameters(variant="s")
        assert sup.height == 3
        # Same exposures expected because we put identical numbers in the file.
        row = sup.filter(pl.col("Part ID") == "Part(2)").row(0, named=True)
        assert row["Hatches Exposure Time"] == 90
        print("  volume_parameters (variant=s) OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_tab_access_by_number() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="parts_test_"))
    try:
        f = tmp / "parts.csv"
        f.write_text(_make_minimal_file())
        parts = QuantAMParts.from_path(f)
        # Tab -1 is Parent Parts.
        assert parts.tab(-1).height == 3
        # Tab 10 is Scan Volume.
        assert parts.tab(10).height == 6  # 3 parts × 2 variants
        try:
            parts.tab(99)
        except KeyError:
            pass
        else:
            raise AssertionError("expected KeyError for missing tab")
        print("  tab() lookup by number OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_unknown_section_raises() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="parts_test_"))
    try:
        f = tmp / "parts.csv"
        f.write_text(_make_minimal_file())
        parts = QuantAMParts.from_path(f)
        try:
            _ = parts["Bogus Section"]
        except KeyError:
            pass
        else:
            raise AssertionError("expected KeyError")
        print("  unknown section raises OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_file_raises() -> None:
    try:
        QuantAMParts.from_path("/no/such/file.csv")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
    print("  missing file raises OK")


def test_no_sections_raises() -> None:
    """A file with no Tab- markers should raise a helpful error."""
    tmp = Path(tempfile.mkdtemp(prefix="parts_test_"))
    try:
        f = tmp / "empty.csv"
        f.write_text("just,some,arbitrary,content\nno,sections,here,\n")
        try:
            QuantAMParts.from_path(f)
        except ValueError as e:
            assert "No 'Tab - N' sections" in str(e)
        else:
            raise AssertionError("expected ValueError")
        print("  no sections raises clearly OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_non_numeric_columns_stay_string() -> None:
    """A column with 'N\\A' values should not be coerced to numeric."""
    tmp = Path(tempfile.mkdtemp(prefix="parts_test_"))
    try:
        f = tmp / "parts.csv"
        # General section has 'N\A' values in X/Y positions.
        f.write_text(
            "#,Tab - 1,General\n"
            '#,"Sr. No.","Source Index","Layer Thickness","X Position","Y Position","Layers Count",\n'
            'ID.,"[T0C1]","[T0C2]","[T1C1]","[T1C2]","[T1C3]","[T1C4]",\n'
            ',"1.1","Part(1)","0.03","N\\A","N\\A","0",\n'
            ',"2.1","Part(2)","0.03","N\\A","N\\A","0",\n'
        )
        parts = QuantAMParts.from_path(f)
        df = parts["General"]
        # X Position should stay string because of the 'N\A' content.
        assert df["X Position"].dtype == pl.String
        # Layer Thickness and Layers Count should still numeric.
        assert df["Layer Thickness"].dtype.is_numeric()
        assert df["Layers Count"].dtype.is_numeric()
        print("  mixed-content columns stay string OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_real_file_smoke() -> None:
    """Smoke test against the real sample file if it's available."""
    real = Path("/mnt/user-data/uploads/1777618604995_JR299_sterling_parts.csv")
    if not real.is_file():
        print("  real file smoke (skipped, file not present)")
        return
    parts = QuantAMParts.from_path(real)
    assert len(parts.section_names) == 15
    pp = parts.parent_parts()
    assert pp.height == 20
    vp = parts.volume_parameters()
    assert vp.height == 20
    # Sanity-check one value from the real data.
    p1 = pp.filter(pl.col("Part ID") == "Part(1)").row(0, named=True)
    assert abs(p1["X Position"] - (-26.787)) < 1e-6
    assert abs(p1["Y Position"] - (-11.585)) < 1e-6
    print(f"  real file smoke ({pp.height} parts, {len(parts.section_names)} sections) OK")


# --------------------------------------------------------------------------- #
# Cluster -> Part ID assignment
# --------------------------------------------------------------------------- #
def _make_three_part_clustering() -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Return (clustered_df, parts_table) where:
      - 3 clusters with centroids near (0, 0), (10, 0), (20, 0)
      - parts table has 3 parts with positions matching those centroids
    """
    rng = np.random.default_rng(0)
    rows = []
    for cluster_id, (cx, cy) in enumerate([(0, 0), (10, 0), (20, 0)]):
        for _ in range(100):
            rows.append({
                "Demand X": cx + rng.uniform(-0.5, 0.5),
                "Demand Y": cy + rng.uniform(-0.5, 0.5),
                "cluster": cluster_id,
            })
    # Add some noise points
    for _ in range(50):
        rows.append({
            "Demand X": rng.uniform(-50, 50),
            "Demand Y": rng.uniform(-50, 50),
            "cluster": -1,
        })
    clustered = pl.DataFrame(rows).with_columns(
        pl.col("cluster").cast(pl.Int32)
    )
    parts_table = pl.DataFrame({
        "Part ID": ["Part(1)", "Part(2)", "Part(3)"],
        "X Position": [0.0, 10.0, 20.0],
        "Y Position": [0.0, 0.0, 0.0],
    })
    return clustered, parts_table


def test_compute_part_id_map_basic() -> None:
    clustered, parts_table = _make_three_part_clustering()
    mapping = compute_part_id_map(clustered, parts_table, verbose=False)
    assert mapping == {0: "Part(1)", 1: "Part(2)", 2: "Part(3)"}
    print("  compute_part_id_map basic OK")


def test_apply_part_id_map_basic() -> None:
    clustered, parts_table = _make_three_part_clustering()
    mapping = compute_part_id_map(clustered, parts_table, verbose=False)
    out = apply_part_id_map(clustered, mapping)
    assert "part_id" in out.columns
    assert out["part_id"].dtype == pl.String
    # Cluster 0 rows should have part_id = "Part(1)"
    p0 = out.filter(pl.col("cluster") == 0)
    assert (p0["part_id"] == "Part(1)").all()
    # Noise rows should have null part_id by default.
    n = out.filter(pl.col("cluster") == -1)
    assert n["part_id"].is_null().all()
    print("  apply_part_id_map default (null for noise) OK")


def test_apply_part_id_map_noise_label() -> None:
    clustered, parts_table = _make_three_part_clustering()
    mapping = compute_part_id_map(clustered, parts_table, verbose=False)
    out = apply_part_id_map(clustered, mapping, noise_label="noise")
    n = out.filter(pl.col("cluster") == -1)
    assert (n["part_id"] == "noise").all()
    print("  apply_part_id_map with noise_label OK")


def test_compute_part_id_map_collision_warning(capsys=None) -> None:
    """If 4 clusters but 3 parts, warn that two clusters map to the same part."""
    clustered, parts_table = _make_three_part_clustering()
    # Add a 4th cluster very close to part 1's position — should collide.
    rng = np.random.default_rng(1)
    extra_rows = []
    for _ in range(100):
        extra_rows.append({
            "Demand X": 0.1 + rng.uniform(-0.5, 0.5),
            "Demand Y": 0.1 + rng.uniform(-0.5, 0.5),
            "cluster": 3,
        })
    extra = pl.DataFrame(extra_rows).with_columns(pl.col("cluster").cast(pl.Int32))
    bigger = pl.concat([clustered, extra])

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mapping = compute_part_id_map(bigger, parts_table, verbose=True)
    assert "claimed by multiple clusters" in buf.getvalue()
    # Both 0 and 3 should map to Part(1)
    assert mapping[0] == "Part(1)"
    assert mapping[3] == "Part(1)"
    print("  collision warning OK")


def test_compute_part_id_map_far_warning() -> None:
    clustered, parts_table = _make_three_part_clustering()
    # Add a cluster at (1000, 1000) — far from all parts.
    rng = np.random.default_rng(2)
    extra_rows = [
        {"Demand X": 1000.0 + rng.uniform(-0.5, 0.5),
         "Demand Y": 1000.0 + rng.uniform(-0.5, 0.5),
         "cluster": 99}
        for _ in range(50)
    ]
    extra = pl.DataFrame(extra_rows).with_columns(pl.col("cluster").cast(pl.Int32))
    bigger = pl.concat([clustered, extra])

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        compute_part_id_map(bigger, parts_table, max_distance_mm=5.0, verbose=True)
    assert "more than 5.0 mm" in buf.getvalue()
    print("  far cluster warning OK")


def test_compute_part_id_map_no_clusters() -> None:
    """All rows are noise — should return an empty dict without crashing."""
    df = pl.DataFrame({
        "Demand X": [0.0, 1.0],
        "Demand Y": [0.0, 1.0],
        "cluster": pl.Series([-1, -1], dtype=pl.Int32),
    })
    parts_table = pl.DataFrame({
        "Part ID": ["Part(1)"], "X Position": [0.0], "Y Position": [0.0],
    })
    mapping = compute_part_id_map(df, parts_table, verbose=False)
    assert mapping == {}
    print("  no clusters returns empty map OK")


def test_compute_part_id_map_unknown_column_raises() -> None:
    clustered, parts_table = _make_three_part_clustering()
    try:
        compute_part_id_map(clustered, parts_table, cluster_col="bogus", verbose=False)
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    try:
        compute_part_id_map(
            clustered, parts_table, parts_x_col="bogus", verbose=False,
        )
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    print("  unknown column raises OK")


def test_apply_part_id_map_unknown_cluster_id_passes_through() -> None:
    """Cluster ids in the data not present in the mapping should fall back
    to noise_label without raising."""
    df = pl.DataFrame({
        "Demand X": [0.0, 1.0, 2.0],
        "Demand Y": [0.0, 1.0, 2.0],
        "cluster": pl.Series([0, 99, -1], dtype=pl.Int32),
    })
    mapping = {0: "Part(1)"}  # 99 and -1 missing
    out = apply_part_id_map(df, mapping, noise_label="unmapped")
    assert out["part_id"].to_list() == ["Part(1)", "unmapped", "unmapped"]
    print("  unknown cluster id falls back to noise_label OK")


def test_real_file_with_synthetic_clusters() -> None:
    """End-to-end: parse the real parts CSV, place synthetic clusters at
    each part's position, verify the mapping recovers the right names."""
    real = Path("/mnt/user-data/uploads/1777618604995_JR299_sterling_parts.csv")
    if not real.is_file():
        print("  real file end-to-end (skipped, not present)")
        return
    parts = QuantAMParts.from_path(real)
    pp = parts.parent_parts().sort("X Position").with_row_index("cluster_truth")

    # Build synthetic clusters: 1 cluster per part, 100 rows each, placed at
    # the real part positions with a small jitter.
    rng = np.random.default_rng(0)
    rows = []
    for r in pp.iter_rows(named=True):
        for _ in range(100):
            rows.append({
                "Demand X": r["X Position"] + rng.uniform(-0.2, 0.2),
                "Demand Y": r["Y Position"] + rng.uniform(-0.2, 0.2),
                "cluster": r["cluster_truth"],
            })
    clustered = pl.DataFrame(rows).with_columns(pl.col("cluster").cast(pl.Int32))

    mapping = compute_part_id_map(clustered, pp, verbose=False)
    out = apply_part_id_map(clustered, mapping)

    # Every cluster's rows should map to the same part_id, matching truth.
    for r in pp.iter_rows(named=True):
        cid = r["cluster_truth"]
        sub = out.filter(pl.col("cluster") == cid)
        unique_pids = sub["part_id"].unique().to_list()
        assert unique_pids == [r["Part ID"]], (
            f"Cluster {cid} expected {r['Part ID']}, got {unique_pids}"
        )
    print(f"  real file end-to-end ({pp.height} parts mapped 1:1) OK")


# --------------------------------------------------------------------------- #
# volume_parameters_with_speed
# --------------------------------------------------------------------------- #
def _make_parametric_file() -> str:
    """Two-section file with parametric Hatches params for 3 parts."""
    return (
        "#,Tab - -1,Parent Parts\n"
        '#,"Sr. No.","Source Index","Layer Thickness","X Position","Y Position","Layers Count",\n'
        'ID.,"[T0C1]","[T0C2]","[T-1C1]","[T-1C2]","[T-1C3]","[T-1C4]",\n'
        ',"1","Part(1)","0.03","-26.787","-11.585","333",\n'
        ',"2","Part(2)","0.03","-13.823","-10.59","333",\n'
        ',"3","Part(3)","0.03","-0.505","-10.328","333",\n'
        "\n"
        "#,Tab - 10,Scan Volume\n"
        '#,"Sr. No.","Source Index","Hatches Power","Hatches Point Distance","Hatches Exposure Time",\n'
        'ID.,"[T0C1]","[T0C2]","[T10C5]","[T10C6]","[T10C7]",\n'
        ',"1.1","Part(1)","150","0.085","60",\n'
        ',"2.1","Part(2)","200","0.085","75",\n'
        ',"3.1","Part(3)","250","0.085","90",\n'
    )


def test_volume_parameters_with_speed_basic() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="parts_test_"))
    try:
        f = tmp / "parts.csv"
        f.write_text(_make_parametric_file())
        parts = QuantAMParts.from_path(f)
        vps = parts.volume_parameters_with_speed()
        assert "Hatch Speed" in vps.columns
        # Original cols still present (we said keep them).
        assert "Hatches Point Distance" in vps.columns
        assert "Hatches Exposure Time" in vps.columns
        # (0.085 / 60) * 1000 = 1.4166...
        row1 = vps.filter(pl.col("Part ID") == "Part(1)").row(0, named=True)
        assert abs(row1["Hatch Speed"] - (0.085 / 60 * 1000)) < 1e-6
        row3 = vps.filter(pl.col("Part ID") == "Part(3)").row(0, named=True)
        assert abs(row3["Hatch Speed"] - (0.085 / 90 * 1000)) < 1e-6
        print("  volume_parameters_with_speed math OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_volume_parameters_with_speed_missing_columns_raises() -> None:
    """If the CSV lacks Point Distance or Exposure Time, raise clearly."""
    tmp = Path(tempfile.mkdtemp(prefix="parts_test_"))
    try:
        # Only Hatches Power, no other columns — should fail to derive speed.
        f = tmp / "parts.csv"
        f.write_text(
            "#,Tab - -1,Parent Parts\n"
            '#,"Sr. No.","Source Index","Layer Thickness","X Position","Y Position","Layers Count",\n'
            'ID.,"[T0C1]","[T0C2]","[T-1C1]","[T-1C2]","[T-1C3]","[T-1C4]",\n'
            ',"1","Part(1)","0.03","0","0","100",\n'
            "\n"
            "#,Tab - 10,Scan Volume\n"
            '#,"Sr. No.","Source Index","Hatches Power",\n'
            'ID.,"[T0C1]","[T0C2]","[T10C5]",\n'
            ',"1.1","Part(1)","150",\n'
        )
        parts = QuantAMParts.from_path(f)
        try:
            parts.volume_parameters_with_speed()
        except ValueError as e:
            assert "Hatches Point Distance" in str(e) or "Hatches Exposure Time" in str(e)
        else:
            raise AssertionError("expected ValueError")
        print("  volume_parameters_with_speed missing cols raises OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_volume_parameters_with_speed_real_file() -> None:
    """Smoke-test against the real CSV."""
    real = Path("/mnt/user-data/uploads/1777618604995_JR299_sterling_parts.csv")
    if not real.is_file():
        print("  volume_parameters_with_speed real file (skipped)")
        return
    parts = QuantAMParts.from_path(real)
    vps = parts.volume_parameters_with_speed()
    assert vps.height == 20
    assert "Hatch Speed" in vps.columns
    # Spot-check: speed should be positive and finite for all parts.
    speeds = vps["Hatch Speed"].to_numpy()
    assert (speeds > 0).all()
    assert np.isfinite(speeds).all()
    print(f"  volume_parameters_with_speed real file ({vps.height} parts, "
          f"speeds {speeds.min():.0f}-{speeds.max():.0f} mm/s) OK")


# --------------------------------------------------------------------------- #
# join_parts_with_stats
# --------------------------------------------------------------------------- #
def test_join_parts_with_stats_basic() -> None:
    stats = pl.DataFrame({
        "part_id": ["Part(1)", "Part(2)", "Part(3)"],
        "cov_signal": [0.1, 0.2, 0.3],
    })
    parts = pl.DataFrame({
        "Part ID": ["Part(1)", "Part(2)", "Part(3)"],
        "Hatch Power": [150.0, 200.0, 250.0],
        "Hatch Speed": [1417.0, 1133.3, 944.4],
    })
    out = join_parts_with_stats(stats, parts, verbose=False)
    assert out.height == 3
    assert "cov_signal" in out.columns
    assert "Hatch Power" in out.columns
    assert "Hatch Speed" in out.columns
    # Check actual join values — Part(2) should have power=200 and cov=0.2.
    row2 = out.filter(pl.col("part_id") == "Part(2)").row(0, named=True)
    assert row2["Hatch Power"] == 200.0
    assert abs(row2["cov_signal"] - 0.2) < 1e-9
    print("  join_parts_with_stats basic OK")


def test_join_parts_with_stats_missing_in_parts_warns() -> None:
    stats = pl.DataFrame({
        "part_id": ["Part(1)", "Part(2)", "Part(99)"],  # 99 not in parts
        "cov_signal": [0.1, 0.2, 0.3],
    })
    parts = pl.DataFrame({
        "Part ID": ["Part(1)", "Part(2)"],
        "Hatch Power": [150.0, 200.0],
    })
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = join_parts_with_stats(stats, parts, verbose=True)
    assert "Part(99)" in buf.getvalue()
    assert "Warning" in buf.getvalue()
    # Output keeps all stats rows (left join) — Part(99) has null Hatch Power.
    assert out.height == 3
    row99 = out.filter(pl.col("part_id") == "Part(99)").row(0, named=True)
    assert row99["Hatch Power"] is None
    print("  join warns on missing parts in parts_table OK")


def test_join_parts_with_stats_extras_in_parts_noted() -> None:
    """If parts_table has parts not in stats_table, note them."""
    stats = pl.DataFrame({
        "part_id": ["Part(1)"],
        "cov_signal": [0.1],
    })
    parts = pl.DataFrame({
        "Part ID": ["Part(1)", "Part(2)", "Part(3)"],
        "Hatch Power": [150.0, 200.0, 250.0],
    })
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = join_parts_with_stats(stats, parts, verbose=True)
    output = buf.getvalue()
    assert "Part(2)" in output and "Part(3)" in output
    assert "Note" in output
    # Output is just stats rows (left join) — only 1 row.
    assert out.height == 1
    print("  join notes extras in parts_table OK")


def test_join_parts_with_stats_unknown_column_raises() -> None:
    stats = pl.DataFrame({"part_id": ["Part(1)"]})
    parts = pl.DataFrame({"Part ID": ["Part(1)"]})
    try:
        join_parts_with_stats(
            stats, parts, stats_part_col="bogus", verbose=False,
        )
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    print("  join_parts_with_stats unknown col raises OK")


def test_join_parts_with_stats_real_pipeline() -> None:
    """End-to-end: real parts file + synthetic CoV table + plot-ready join."""
    real = Path("/mnt/user-data/uploads/1777618604995_JR299_sterling_parts.csv")
    if not real.is_file():
        print("  join real pipeline (skipped)")
        return
    parts = QuantAMParts.from_path(real)
    parts_table = parts.volume_parameters_with_speed()
    # Fake stats table with one CoV per part, taking part IDs from real file.
    rng = np.random.default_rng(0)
    fake_stats = parts_table.select("Part ID").rename({"Part ID": "part_id"}).with_columns(
        pl.Series("cov_signal", rng.uniform(0.1, 0.3, parts_table.height))
    )
    out = join_parts_with_stats(fake_stats, parts_table, verbose=False)
    assert out.height == parts_table.height
    assert "Hatch Speed" in out.columns
    assert "Hatches Power" in out.columns
    assert "cov_signal" in out.columns
    # All Hatches Power values should be non-null (every part has parameters).
    assert out["Hatches Power"].null_count() == 0
    print(f"  join real pipeline ({out.height} parts joined) OK")


def main() -> None:
    print("Phase 8 parts tests:")
    test_section_discovery()
    test_parent_parts_table()
    test_volume_parameters_default_variant()
    test_volume_parameters_support_variant()
    test_tab_access_by_number()
    test_unknown_section_raises()
    test_missing_file_raises()
    test_no_sections_raises()
    test_non_numeric_columns_stay_string()
    test_real_file_smoke()
    test_compute_part_id_map_basic()
    test_apply_part_id_map_basic()
    test_apply_part_id_map_noise_label()
    test_compute_part_id_map_collision_warning()
    test_compute_part_id_map_far_warning()
    test_compute_part_id_map_no_clusters()
    test_compute_part_id_map_unknown_column_raises()
    test_apply_part_id_map_unknown_cluster_id_passes_through()
    test_real_file_with_synthetic_clusters()
    test_volume_parameters_with_speed_basic()
    test_volume_parameters_with_speed_missing_columns_raises()
    test_volume_parameters_with_speed_real_file()
    test_join_parts_with_stats_basic()
    test_join_parts_with_stats_missing_in_parts_warns()
    test_join_parts_with_stats_extras_in_parts_noted()
    test_join_parts_with_stats_unknown_column_raises()
    test_join_parts_with_stats_real_pipeline()
    print("\nAll Phase 8 tests passed")


if __name__ == "__main__":
    main()
