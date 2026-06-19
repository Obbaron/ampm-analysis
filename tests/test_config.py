"""
Tests for ``config.py`` — loading and validating ``config.toml``.

``load_config`` reports errors via ``sys.exit`` (raising ``SystemExit``), so
the failure-path tests assert on that rather than a normal exception.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import ampm.config as config

MINIMAL_TOML = """\
[paths]
source    = 'data'
stl       = 'part.stl'
parts_csv = 'parts.csv'

[build]
layer_thickness = 0.03
"""


def write_toml(directory, text, name="config.toml"):
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadConfigErrors:
    def test_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            config.load_config(tmp_path)

    def test_invalid_toml_exits(self, tmp_path):
        write_toml(tmp_path, "this is = = not valid toml [[[")
        with pytest.raises(SystemExit):
            config.load_config(tmp_path)

    def test_missing_required_key_exits(self, tmp_path):
        write_toml(tmp_path, "[paths]\nsource = 'data'\n")  # no stl/parts_csv/build
        with pytest.raises(SystemExit):
            config.load_config(tmp_path)


class TestLoadConfigSuccess:
    def test_relative_paths_resolved_against_build_dir(self, tmp_path):
        write_toml(tmp_path, MINIMAL_TOML)
        cfg = config.load_config(tmp_path)
        assert cfg["SOURCE"] == str(tmp_path / "data")
        assert cfg["STL"] == str(tmp_path / "part.stl")
        assert cfg["PARTS_CSV"] == str(tmp_path / "parts.csv")
        assert cfg["LAYER_THICKNESS"] == 0.03

    def test_absolute_path_preserved(self, tmp_path):
        abs_stl = (tmp_path / "elsewhere" / "x.stl").resolve()
        toml = MINIMAL_TOML.replace(
            "stl       = 'part.stl'", f"stl       = '{abs_stl}'"
        )
        write_toml(tmp_path, toml)
        cfg = config.load_config(tmp_path)
        assert cfg["STL"] == str(abs_stl)

    def test_derived_cache_paths(self, tmp_path):
        write_toml(tmp_path, MINIMAL_TOML)
        cfg = config.load_config(tmp_path)
        source = Path(cfg["SOURCE"])
        assert cfg["MASK_CACHE"] == str(source / ".cache" / "fullplate_mask.pkl")
        assert cfg["MASK_KEEP_CACHE"] == str(source / ".cache" / "mask_keep.pq")
        assert cfg["CLUSTER_CACHE"] == str(source / ".cache" / "cluster_labels.pq")

    def test_defaults_applied(self, tmp_path):
        write_toml(tmp_path, MINIMAL_TOML)
        cfg = config.load_config(tmp_path)
        assert cfg["METHOD"] == "direct"
        assert cfg["MAX_DISTANCE_MM"] is None  # 'none' default -> None
        assert cfg["EPS_XY"] == 0.3
        assert cfg["EPS_Z"] == 0.06
        assert cfg["MIN_SAMPLES"] == 10
        assert cfg["LAYERS_PER_CHUNK"] == 11
        assert cfg["OVERLAP_LAYERS"] is None  # 'auto' -> None
        assert cfg["SIGNALS"] == [
            "MeltVIEW melt pool (mean)",
            "Laser output power (mean)",
        ]

    def test_none_and_auto_sentinels(self, tmp_path):
        toml = MINIMAL_TOML + (
            "\n[assignment]\nmethod = 'dbscan'\nmax_distance_mm = 2.5\n"
            "\n[clustering]\noverlap_layers = 4\n"
        )
        write_toml(tmp_path, toml)
        cfg = config.load_config(tmp_path)
        assert cfg["METHOD"] == "dbscan"
        assert cfg["MAX_DISTANCE_MM"] == 2.5
        assert cfg["OVERLAP_LAYERS"] == 4

    def test_custom_signals(self, tmp_path):
        toml = MINIMAL_TOML + "\n[signals]\ncolumns = ['A', 'B', 'C']\n"
        write_toml(tmp_path, toml)
        assert config.load_config(tmp_path)["SIGNALS"] == ["A", "B", "C"]


class TestCreateOrLoad:
    def test_existing_toml_just_loads(self, tmp_path):
        write_toml(tmp_path, MINIMAL_TOML)
        cfg = config.create_or_load_config(tmp_path)
        assert cfg["LAYER_THICKNESS"] == 0.03

    def test_autocreate_then_load(self, tmp_path, monkeypatch):
        import ampm.setup_build as setup_build

        ampm_pkg = types.ModuleType("ampm")
        monkeypatch.setitem(sys.modules, "ampm", ampm_pkg)
        monkeypatch.setitem(sys.modules, "ampm.setup_build", setup_build)

        # Populate a build dir so auto-detection succeeds.
        data = tmp_path / "data"
        data.mkdir()
        (data / "Packet data for layer 1, laser 1.txt").write_text("Start time\n0\n")
        (tmp_path / "plate.stl").write_bytes(b"x")
        (tmp_path / "parts.csv").write_text(
            "#,Renishaw,Material,Development\n\n"
            "#,Tab - -1,Parent Parts\n"
            '#,"Sr. No.","Source Index","Layer Thickness"\n'
            'ID.,"[T0C1]","[T0C2]","[T0C3]"\n'
            ',"1","Part(1)","0.03"\n\n'
        )

        assert not (tmp_path / "config.toml").exists()
        cfg = config.create_or_load_config(tmp_path)
        assert (tmp_path / "config.toml").exists()
        assert cfg["LAYER_THICKNESS"] == 0.03
        assert cfg["SOURCE"] == str(data)
