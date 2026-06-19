"""
Tests for ``parts.py`` — QuantAM parts-CSV parsing, BuildStarted DHXML parsing,
and the cluster/row -> part assignment helpers.
"""

from __future__ import annotations

import csv
import io
import json

import polars as pl
import pytest

from ampm.parts import (
    BuildStartedDHXML,
    QuantAMParts,
    _parse_bounding_box,
    _suffix_duplicate_names,
    _try_numeric,
    apply_part_id_map,
    assign_bounding_box_part,
    assign_nearest_part,
    compute_part_id_map,
    join_parts_with_stats,
)


def make_quantam_csv(sections):
    """sections: list of (tab_num, name, headers, data_rows)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["#", "Renishaw", "Material", "Development"])
    w.writerow(["", "Version", "0.6.1"])
    w.writerow([])
    for tab_num, name, headers, rows in sections:
        w.writerow(["#", f"Tab - {tab_num}", name])
        w.writerow(["#", *headers])
        w.writerow(["ID.", *[f"[T0C{i}]" for i in range(len(headers))]])
        for r in rows:
            w.writerow(["", *r])
        w.writerow([])
    return buf.getvalue()


PARENT_HEADERS = [
    "Sr. No.",
    "Source Index",
    "Layer Thickness",
    "X Position",
    "Y Position",
    "Layers Count",
]
VOLUME_HEADERS = [
    "Sr. No.",
    "Source Index",
    "Hatches Point Distance",
    "Hatches Exposure Time",
]


def standard_csv():
    return make_quantam_csv(
        [
            (
                -1,
                "Parent Parts",
                PARENT_HEADERS,
                [
                    ["1", "Part(1)", "0.03", "-26.787", "10.0", "100"],
                    ["2", "Part(2)", "0.03", "-13.823", "12.0", "120"],
                ],
            ),
            (
                10,
                "Scan Volume",
                VOLUME_HEADERS,
                [
                    ["1.1", "Part(1)", "0.06", "60"],
                    ["2.1", "Part(2)", "0.07", "70"],
                ],
            ),
        ]
    )


def write_csv(tmp_path, text, name="parts.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def dhxml_payload(parts_list):
    return {"version1": {"build": {"parts": parts_list}}}


class TestPureHelpers:
    def test_try_numeric_integers(self):
        out = _try_numeric(pl.Series("x", ["1", "2", "3"]))
        assert out is not None
        assert out.dtype == pl.Int64 and out.to_list() == [1, 2, 3]

    def test_try_numeric_floats(self):
        out = _try_numeric(pl.Series("x", ["1.5", "2.0"]))
        assert out is not None
        assert out.dtype == pl.Float64

    def test_try_numeric_non_numeric_returns_none(self):
        assert _try_numeric(pl.Series("x", ["Part(1)", "Part(2)"])) is None

    def test_try_numeric_mixed_returns_none(self):
        assert _try_numeric(pl.Series("x", ["1", "abc"])) is None

    def test_parse_bounding_box_basic(self):
        assert _parse_bounding_box("0,1,2,3,4,5") == (0, 1, 2, 3, 4, 5)

    def test_parse_bounding_box_normalizes_min_max(self):
        # min/max swapped on each axis -> normalized.
        assert _parse_bounding_box("3,4,5,0,1,2") == (0, 1, 2, 3, 4, 5)

    def test_parse_bounding_box_wrong_count_raises(self):
        with pytest.raises(ValueError, match="6 comma-separated"):
            _parse_bounding_box("1,2,3")

    def test_suffix_duplicate_names(self):
        out = _suffix_duplicate_names([{"name": "A"}, {"name": "B"}, {"name": "A"}])
        ids = [p["part_id"] for p in out]
        assert ids == ["A#1", "B", "A#2"]


class TestQuantAMParts:
    def test_from_path_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            QuantAMParts.from_path(tmp_path / "nope.csv")

    def test_from_path_no_sections(self, tmp_path):
        p = write_csv(tmp_path, "#,Renishaw,Material\n,Version,0.6.1\n")
        with pytest.raises(ValueError, match="No 'Tab - N' sections"):
            QuantAMParts.from_path(p)

    def test_section_access_and_membership(self, tmp_path):
        q = QuantAMParts.from_path(write_csv(tmp_path, standard_csv()))
        assert "Parent Parts" in q
        assert "Scan Volume" in q
        assert set(q.section_names) == {"Parent Parts", "Scan Volume"}
        assert q["Parent Parts"].height == 2

    def test_unknown_section_raises(self, tmp_path):
        q = QuantAMParts.from_path(write_csv(tmp_path, standard_csv()))
        with pytest.raises(KeyError, match="Strategy"):
            q["Strategy"]

    def test_tab_lookup(self, tmp_path):
        q = QuantAMParts.from_path(write_csv(tmp_path, standard_csv()))
        assert q.tab(10).height == 2
        with pytest.raises(KeyError, match="tab number 99"):
            q.tab(99)

    def test_parent_parts_columns_and_types(self, tmp_path):
        q = QuantAMParts.from_path(write_csv(tmp_path, standard_csv()))
        pp = q.parent_parts()
        assert pp.columns == [
            "Part ID",
            "Layer Thickness",
            "X Position",
            "Y Position",
            "Layers Count",
        ]
        assert pp["Part ID"].to_list() == ["Part(1)", "Part(2)"]
        assert pp["X Position"].to_list() == pytest.approx([-26.787, -13.823])

    def test_parent_parts_suffixes_duplicates(self, tmp_path):
        csv_text = make_quantam_csv(
            [
                (
                    -1,
                    "Parent Parts",
                    PARENT_HEADERS,
                    [
                        ["1", "Part(1)", "0.03", "0.0", "0.0", "100"],
                        ["4", "Part(1)", "0.03", "5.0", "5.0", "100"],
                    ],
                )
            ]
        )
        q = QuantAMParts.from_path(write_csv(tmp_path, csv_text))
        ids = q.parent_parts()["Part ID"].to_list()
        assert ids == ["Part(1)#1", "Part(1)#4"]

    def test_volume_parameters_joins_on_instance(self, tmp_path):
        q = QuantAMParts.from_path(write_csv(tmp_path, standard_csv()))
        vp = q.volume_parameters(variant="1")
        assert vp["Part ID"].to_list() == ["Part(1)", "Part(2)"]
        assert "Hatches Point Distance" in vp.columns
        assert "Sr. No." not in vp.columns

    def test_volume_parameters_with_speed(self, tmp_path):
        q = QuantAMParts.from_path(write_csv(tmp_path, standard_csv()))
        vp = q.volume_parameters_with_speed(variant="1")
        # PD/ET*1000: 0.06/60*1000 = 1.0 and 0.07/70*1000 = 1.0
        assert vp["Hatch Speed"].to_list() == pytest.approx([1.0, 1.0])

    def test_volume_parameters_with_speed_missing_column_raises(self, tmp_path):
        csv_text = make_quantam_csv(
            [
                (
                    -1,
                    "Parent Parts",
                    PARENT_HEADERS,
                    [["1", "Part(1)", "0.03", "0", "0", "10"]],
                ),
                (10, "Scan Volume", ["Sr. No.", "Source Index"], [["1.1", "Part(1)"]]),
            ]
        )
        q = QuantAMParts.from_path(write_csv(tmp_path, csv_text))
        with pytest.raises(ValueError, match="Hatch Speed"):
            q.volume_parameters_with_speed()


class TestBuildStartedDHXML:
    def _write(self, tmp_path, payload, name="build.dhxml"):
        p = tmp_path / name
        p.write_text(json.dumps(payload), encoding="utf-8")
        return p

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BuildStartedDHXML.from_path(tmp_path / "nope.dhxml")

    def test_invalid_json(self, tmp_path):
        p = tmp_path / "bad.dhxml"
        p.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            BuildStartedDHXML.from_path(p)

    def test_missing_parts_path(self, tmp_path):
        p = self._write(tmp_path, {"version1": {"build": {}}})
        with pytest.raises(ValueError, match="not a recognized"):
            BuildStartedDHXML.from_path(p)

    def test_empty_parts_list(self, tmp_path):
        p = self._write(tmp_path, dhxml_payload([]))
        with pytest.raises(ValueError, match="No parts"):
            BuildStartedDHXML.from_path(p)

    def test_malformed_entry(self, tmp_path):
        p = self._write(tmp_path, dhxml_payload([{"name": "X"}]))  # no boundingBox
        with pytest.raises(ValueError, match="Malformed part entry"):
            BuildStartedDHXML.from_path(p)

    def test_parts_table_and_centres(self, tmp_path):
        p = self._write(
            tmp_path,
            dhxml_payload(
                [
                    {"name": "Part(1)", "boundingBox": "0,0,0,10,10,5"},
                    {"name": "Part(2)", "boundingBox": "20,0,0,30,10,5"},
                ]
            ),
        )
        build = BuildStartedDHXML.from_path(p)
        assert len(build) == 2
        assert build.part_names == ["Part(1)", "Part(2)"]
        table = build.parts_table()
        assert table["Part ID"].to_list() == ["Part(1)", "Part(2)"]
        assert table["X Position"].to_list() == pytest.approx([5.0, 25.0])
        assert table["Y Position"].to_list() == pytest.approx([5.0, 5.0])

    def test_duplicate_names_suffixed(self, tmp_path):
        p = self._write(
            tmp_path,
            dhxml_payload(
                [
                    {"name": "P", "boundingBox": "0,0,0,1,1,1"},
                    {"name": "P", "boundingBox": "2,2,2,3,3,3"},
                ]
            ),
        )
        build = BuildStartedDHXML.from_path(p)
        assert build.parts_table()["Part ID"].to_list() == ["P#1", "P#2"]
        assert build.part_names == ["P", "P"]  # original names preserved


def parts_positions(ids, xs, ys):
    return pl.DataFrame(
        {
            "Part ID": ids,
            "X Position": pl.Series(xs, dtype=pl.Float64),
            "Y Position": pl.Series(ys, dtype=pl.Float64),
        }
    )


class TestComputePartIdMap:
    def test_maps_clusters_to_nearest_part(self):
        clustered = pl.DataFrame(
            {
                "cluster": [0, 0, 1, 1],
                "Demand X": [5.0, 5.2, 25.0, 24.8],
                "Demand Y": [5.0, 4.9, 5.0, 5.1],
            }
        )
        ptable = parts_positions(["A", "B"], [5.0, 25.0], [5.0, 5.0])
        mapping = compute_part_id_map(clustered, ptable, verbose=False)
        assert mapping == {0: "A", 1: "B"}

    def test_noise_clusters_ignored(self):
        clustered = pl.DataFrame(
            {"cluster": [-1, -1], "Demand X": [5.0, 6.0], "Demand Y": [5.0, 6.0]}
        )
        ptable = parts_positions(["A"], [5.0], [5.0])
        assert compute_part_id_map(clustered, ptable, verbose=False) == {}

    def test_missing_column_raises(self):
        clustered = pl.DataFrame({"cluster": [0], "Demand X": [1.0]})  # no Demand Y
        ptable = parts_positions(["A"], [0.0], [0.0])
        with pytest.raises(KeyError):
            compute_part_id_map(clustered, ptable, verbose=False)

    def test_collision_warning_printed(self, capsys):
        # Two clusters, one part -> both map to it -> collision warning.
        clustered = pl.DataFrame(
            {"cluster": [0, 1], "Demand X": [0.0, 0.1], "Demand Y": [0.0, 0.1]}
        )
        ptable = parts_positions(["A"], [0.0], [0.0])
        compute_part_id_map(clustered, ptable, verbose=True)
        assert "claimed by multiple clusters" in capsys.readouterr().out


class TestApplyPartIdMap:
    def test_adds_part_id_column(self):
        clustered = pl.DataFrame({"cluster": [0, 0, 1, -1]})
        out = apply_part_id_map(clustered, {0: "A", 1: "B"})
        assert out["part_id"].to_list() == ["A", "A", "B", None]

    def test_custom_noise_label(self):
        clustered = pl.DataFrame({"cluster": [0, -1]})
        out = apply_part_id_map(clustered, {0: "A"}, noise_label="noise")
        assert out["part_id"].to_list() == ["A", "noise"]

    def test_missing_cluster_column_raises(self):
        with pytest.raises(KeyError):
            apply_part_id_map(pl.DataFrame({"x": [1]}), {0: "A"})


class TestJoinPartsWithStats:
    def test_left_join_preserves_stats_rows(self):
        stats = pl.DataFrame({"part_id": ["A", "B"], "cov": [0.1, 0.2]})
        ptable = pl.DataFrame({"Part ID": ["A"], "power": [200.0]})
        out = join_parts_with_stats(stats, ptable, verbose=False)
        assert out.height == 2
        # B has no parameter row -> null power.
        b_row = out.filter(pl.col("part_id") == "B").row(0, named=True)
        assert b_row["power"] is None

    def test_missing_columns_raise(self):
        with pytest.raises(KeyError):
            join_parts_with_stats(
                pl.DataFrame({"x": [1]}),
                pl.DataFrame({"Part ID": ["A"]}),
                verbose=False,
            )
        with pytest.raises(KeyError):
            join_parts_with_stats(
                pl.DataFrame({"part_id": ["A"]}),
                pl.DataFrame({"y": [1]}),
                verbose=False,
            )


def parts_box_table(ids, boxes):
    """boxes: list of (xmin,ymin,zmin,xmax,ymax,zmax)."""
    cols = {"Part ID": ids}
    for i, key in enumerate(["X min", "Y min", "Z min", "X max", "Y max", "Z max"]):
        cols[key] = pl.Series([b[i] for b in boxes], dtype=pl.Float64)
    return pl.DataFrame(cols)


class TestAssignNearestPart:
    def test_assigns_each_row_to_nearest(self):
        masked = pl.DataFrame({"Demand X": [1.0, 101.0], "Demand Y": [0.0, 0.0]})
        ptable = parts_positions(["A", "B"], [0.0, 100.0], [0.0, 0.0])
        out = assign_nearest_part(masked, ptable, verbose=False)
        assert out["part_id"].to_list() == ["A", "B"]
        assert isinstance(out["part_id"].dtype, pl.Enum)

    def test_empty_parts_table_raises(self):
        masked = pl.DataFrame({"Demand X": [1.0], "Demand Y": [0.0]})
        with pytest.raises(ValueError, match="empty"):
            assign_nearest_part(masked, parts_positions([], [], []), verbose=False)

    def test_missing_column_raises(self):
        masked = pl.DataFrame({"Demand X": [1.0]})  # no Demand Y
        ptable = parts_positions(["A"], [0.0], [0.0])
        with pytest.raises(KeyError):
            assign_nearest_part(masked, ptable, verbose=False)

    def test_far_rows_labelled_noise(self):
        masked = pl.DataFrame({"Demand X": [1.0, 1000.0], "Demand Y": [0.0, 0.0]})
        ptable = parts_positions(["A", "B"], [0.0, 100.0], [0.0, 0.0])
        out = assign_nearest_part(
            masked, ptable, max_distance_mm=5.0, noise_label="noise", verbose=False
        )
        assert out["part_id"].to_list() == ["A", "noise"]

    def test_far_rows_null_when_no_label(self):
        masked = pl.DataFrame({"Demand X": [1.0, 1000.0], "Demand Y": [0.0, 0.0]})
        ptable = parts_positions(["A", "B"], [0.0, 100.0], [0.0, 0.0])
        out = assign_nearest_part(
            masked, ptable, max_distance_mm=5.0, noise_label=None, verbose=False
        )
        assert out["part_id"].to_list() == ["A", None]


class TestAssignBoundingBoxPart:
    def test_assigns_by_containment(self):
        masked = pl.DataFrame(
            {
                "Demand X": [5.0, 25.0, 15.0],
                "Demand Y": [5.0, 5.0, 5.0],
                "Z": [1.0, 1.0, 1.0],
            }
        )
        ptable = parts_box_table(
            ["A", "B"],
            [(0, 0, 0, 10, 10, 5), (20, 0, 0, 30, 10, 5)],
        )
        out = assign_bounding_box_part(masked, ptable, verbose=False)
        # (5,5) in A, (25,5) in B, (15,5) in neither -> noise.
        assert out["part_id"].to_list() == ["A", "B", "noise"]

    def test_outside_all_boxes_null_when_no_label(self):
        masked = pl.DataFrame({"Demand X": [15.0], "Demand Y": [5.0], "Z": [1.0]})
        ptable = parts_box_table(["A"], [(0, 0, 0, 10, 10, 5)])
        out = assign_bounding_box_part(masked, ptable, noise_label=None, verbose=False)
        assert out["part_id"].to_list() == [None]

    def test_use_z_excludes_out_of_range_height(self):
        masked = pl.DataFrame({"Demand X": [5.0], "Demand Y": [5.0], "Z": [99.0]})
        ptable = parts_box_table(["A"], [(0, 0, 0, 10, 10, 5)])
        out = assign_bounding_box_part(masked, ptable, use_z=True, verbose=False)
        # Z=99 is outside [0,5] -> not contained -> noise.
        assert out["part_id"].to_list() == ["noise"]

    def test_empty_parts_table_raises(self):
        masked = pl.DataFrame({"Demand X": [1.0], "Demand Y": [0.0]})
        empty = parts_box_table([], [])
        with pytest.raises(ValueError, match="empty"):
            assign_bounding_box_part(masked, empty, verbose=False)

    def test_missing_column_raises(self):
        masked = pl.DataFrame({"Demand X": [1.0]})  # no Demand Y
        ptable = parts_box_table(["A"], [(0, 0, 0, 10, 10, 5)])
        with pytest.raises(KeyError):
            assign_bounding_box_part(masked, ptable, verbose=False)
