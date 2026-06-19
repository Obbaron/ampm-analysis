"""
Tests for ``setup_build.py`` — build-file auto-detection and config.toml
generation.
"""

from __future__ import annotations

import pytest

from ampm.setup_build import (
    _extract_layer_thickness,
    _find_parts_csv,
    _find_source_dir,
    _find_stl,
    _is_quantam_csv,
    _stl_depth,
    _stl_has_keyword,
    _stl_is_support,
    create_config,
)

QUANTAM_HEADER = "#,Renishaw,Material,Development"


def quantam_csv_text(layer_thickness="0.03"):
    """Minimal QuantAM parts CSV with a Parent Parts section."""
    return (
        f"{QUANTAM_HEADER}\n"
        ",Version,0.6.1\n"
        "\n"
        "#,Tab - -1,Parent Parts\n"
        '#,"Sr. No.","Source Index","Layer Thickness","X Position"\n'
        'ID.,"[T0C1]","[T0C2]","[T0C3]","[T0C4]"\n'
        f',"1","Part(1)","{layer_thickness}","-26.787"\n'
        "\n"
    )


def make_packet(directory, layer=1, laser=1):
    p = directory / f"Packet data for layer {layer}, laser {laser}.txt"
    p.write_text("Start time\tDuration\n0\t1\n")
    return p


class TestFindSourceDir:
    def test_finds_packet_directory(self, tmp_path):
        data = tmp_path / "data"
        data.mkdir()
        make_packet(data)
        assert _find_source_dir(tmp_path) == data

    def test_no_packets_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Packet data"):
            _find_source_dir(tmp_path)

    def test_multiple_directories_raises(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        make_packet(a)
        make_packet(b)
        with pytest.raises(ValueError, match="multiple directories"):
            _find_source_dir(tmp_path)


class TestFindStl:
    def test_no_stl_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No .stl"):
            _find_stl(tmp_path)

    def test_single_stl_returned(self, tmp_path):
        p = tmp_path / "part.stl"
        p.write_bytes(b"x")
        assert _find_stl(tmp_path) == p

    def test_shallower_depth_wins(self, tmp_path):
        nested = tmp_path / "sub"
        nested.mkdir()
        shallow = tmp_path / "part.stl"
        deep = nested / "plate.stl"
        shallow.write_bytes(b"x")
        deep.write_bytes(b"x")
        # Even though 'plate' has a keyword, the shallower file wins first.
        assert _find_stl(tmp_path) == shallow

    def test_keyword_preferred_at_same_depth(self, tmp_path):
        plain = tmp_path / "aaa.stl"
        keyworded = tmp_path / "fullplate.stl"
        plain.write_bytes(b"x")
        keyworded.write_bytes(b"x")
        assert _find_stl(tmp_path) == keyworded

    def test_non_support_preferred_over_support(self, tmp_path):
        body = tmp_path / "plate.stl"
        support = tmp_path / "plate_s.stl"
        body.write_bytes(b"x")
        support.write_bytes(b"x")
        assert _find_stl(tmp_path) == body

    def test_helpers(self, tmp_path):
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        deep = sub / "x.stl"
        deep.write_bytes(b"x")
        assert _stl_depth(deep, tmp_path) == 2
        assert _stl_has_keyword(tmp_path / "fullplate.stl") is True
        assert _stl_has_keyword(tmp_path / "widget.stl") is False
        assert _stl_is_support(tmp_path / "x_s.stl") is True
        assert _stl_is_support(tmp_path / "x.stl") is False


class TestFindPartsCsv:
    def test_is_quantam_csv(self, tmp_path):
        good = tmp_path / "good.csv"
        good.write_text(quantam_csv_text())
        bad = tmp_path / "bad.csv"
        bad.write_text("col1,col2\n1,2\n")
        assert _is_quantam_csv(good) is True
        assert _is_quantam_csv(bad) is False
        assert _is_quantam_csv(tmp_path / "missing.csv") is False

    def test_single_csv_returned(self, tmp_path):
        p = tmp_path / "parts.csv"
        p.write_text(quantam_csv_text())
        assert _find_parts_csv(tmp_path) == p

    def test_no_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No .csv"):
            _find_parts_csv(tmp_path)

    def test_multiple_csvs_selects_quantam(self, tmp_path):
        quantam = tmp_path / "parts.csv"
        quantam.write_text(quantam_csv_text())
        other = tmp_path / "notes.csv"
        other.write_text("a,b\n1,2\n")
        assert _find_parts_csv(tmp_path) == quantam


class TestExtractLayerThickness:
    def test_reads_first_parent_row(self, tmp_path):
        p = tmp_path / "parts.csv"
        p.write_text(quantam_csv_text("0.04"))
        assert _extract_layer_thickness(p) == pytest.approx(0.04)

    def test_missing_layer_thickness_column_raises(self, tmp_path):
        p = tmp_path / "parts.csv"
        p.write_text(
            f"{QUANTAM_HEADER}\n\n"
            "#,Tab - -1,Parent Parts\n"
            '#,"Sr. No.","Source Index"\n'
            'ID.,"[T0C1]","[T0C2]"\n'
            ',"1","Part(1)"\n\n'
        )
        with pytest.raises(ValueError, match="Layer Thickness"):
            _extract_layer_thickness(p)

    def test_no_data_rows_raises(self, tmp_path):
        p = tmp_path / "parts.csv"
        p.write_text(
            f"{QUANTAM_HEADER}\n\n"
            "#,Tab - -1,Parent Parts\n"
            '#,"Sr. No.","Layer Thickness"\n'
            'ID.,"[T0C1]","[T0C2]"\n\n'
        )
        with pytest.raises(ValueError, match="Could not extract"):
            _extract_layer_thickness(p)


class TestCreateConfig:
    def _populate(self, root):
        data = root / "data"
        data.mkdir()
        make_packet(data)
        (root / "plate.stl").write_bytes(b"x")
        (root / "parts.csv").write_text(quantam_csv_text("0.03"))

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            create_config(tmp_path / "nope")

    def test_writes_toml_with_relative_paths(self, tmp_path):
        self._populate(tmp_path)
        toml_path = create_config(tmp_path)
        assert toml_path == tmp_path / "config.toml"
        text = toml_path.read_text()
        assert "source    = 'data'" in text
        assert "stl       = 'plate.stl'" in text
        assert "parts_csv = 'parts.csv'" in text
        assert "layer_thickness = 0.03" in text

    def test_overrides_skip_autodetection(self, tmp_path):
        # Only provide the CSV for layer thickness; override the rest.
        (tmp_path / "parts.csv").write_text(quantam_csv_text("0.05"))
        src = tmp_path / "src"
        src.mkdir()
        stl = tmp_path / "explicit.stl"
        stl.write_bytes(b"x")
        toml_path = create_config(
            tmp_path, source=src, stl=stl, parts_csv=tmp_path / "parts.csv"
        )
        text = toml_path.read_text()
        assert "explicit.stl" in text
        assert "layer_thickness = 0.05" in text
