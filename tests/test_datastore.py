"""
Tests for ``datastore.py`` (the :class:`DataStore` lazy Parquet-cached reader).

These tests build small synthetic "Packet data for layer N, laser M.txt" files
in a temporary directory, so nothing here touches real Renishaw data or any
pre-existing cache.
"""

from __future__ import annotations

import os
import time

import polars as pl
import pytest

from ampm.datastore import (
    CACHE_FORMAT_VERSION,
    EXPECTED_COLUMNS,
    DataStore,
)


def _record(i: int, x: float, y: float) -> dict[str, str]:
    """A full row keyed by column name, as the raw .txt would store strings."""
    rec = {c: f"{(i % 10) + 0.5:.4f}" for c in EXPECTED_COLUMNS}
    rec["Start time"] = str(1000 + i)
    rec["Duration"] = "5"
    rec["Demand X"] = f"{x:.4f}"
    rec["Demand Y"] = f"{y:.4f}"
    rec["Demand focus"] = "0.0000"
    return rec


def write_layer_file(
    directory,
    layer: int,
    *,
    laser: int = 1,
    n_rows: int = 8,
    xs: list[float] | None = None,
    ys: list[float] | None = None,
    trailing_tab: bool = False,
    columns: list[str] | None = None,
):
    """
    Write a synthetic 'Packet data for layer {layer}, laser {laser}.txt' file.

    Parameters mirror the knobs the tests need: a custom column set (to test the
    missing-column path), a trailing tab (to test empty-column dropping), and
    explicit X/Y arrays (to test spatial filters).
    """
    columns = list(columns) if columns is not None else list(EXPECTED_COLUMNS)
    tab = "\t" if trailing_tab else ""

    header = "\t".join(columns) + tab
    lines = [header]
    for i in range(n_rows):
        x = xs[i] if xs is not None else float(i)
        y = ys[i] if ys is not None else float(i)
        rec = _record(i, x, y)
        lines.append("\t".join(rec[c] for c in columns) + tab)

    path = directory / f"Packet data for layer {layer}, laser {laser}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def source_dir(tmp_path):
    """A source directory holding layers 1, 2 and 5 (intentionally non-contiguous)."""
    d = tmp_path / "src"
    d.mkdir()
    for layer in (1, 2, 5):
        write_layer_file(d, layer, n_rows=8)
    return d


@pytest.fixture
def store(source_dir):
    return DataStore(source_dir, layer_thickness=0.03)


def _touch_in_future(path, seconds: float = 5.0) -> None:
    """Bump a file's mtime into the future so mtime comparisons are deterministic."""
    future = time.time() + seconds
    os.utime(path, (future, future))


class TestInit:
    def test_resolves_source_dir_to_absolute(self, source_dir):
        ds = DataStore(source_dir)
        assert ds.source_dir.is_absolute()
        assert ds.source_dir == source_dir.resolve()

    def test_missing_source_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DataStore(tmp_path / "does_not_exist")

    def test_source_dir_that_is_a_file_raises(self, source_dir):
        a_file = source_dir / "Packet data for layer 1, laser 1.txt"
        with pytest.raises(FileNotFoundError):
            DataStore(a_file)

    def test_layer_thickness_coerced_to_float(self, source_dir):
        ds = DataStore(source_dir, layer_thickness=1)
        assert isinstance(ds.layer_thickness, float)
        assert ds.layer_thickness == 1.0

    def test_default_cache_dir(self, source_dir):
        ds = DataStore(source_dir)
        assert ds.cache_dir == (source_dir.resolve() / ".cache")

    def test_custom_cache_dir(self, source_dir, tmp_path):
        custom = tmp_path / "elsewhere"
        ds = DataStore(source_dir, cache_dir=custom)
        assert ds.cache_dir == custom.resolve()

    def test_no_matching_files_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "readme.txt").write_text("not a packet file\n")
        with pytest.raises(FileNotFoundError):
            DataStore(empty)


class TestDiscovery:
    def test_discovers_only_matching_files(self, tmp_path):
        d = tmp_path / "src"
        d.mkdir()
        write_layer_file(d, 1)
        write_layer_file(d, 2)
        (d / "notes.txt").write_text("ignore me\n")
        (d / "Packet data for layer 3.txt").write_text("missing laser part\n")
        (d / "summary.csv").write_text("x,y\n1,2\n")
        ds = DataStore(d)
        assert ds.layers == [1, 2]

    def test_subdirectories_are_ignored(self, tmp_path):
        d = tmp_path / "src"
        d.mkdir()
        write_layer_file(d, 1)
        # A directory whose name would otherwise match the regex.
        (d / "Packet data for layer 9, laser 1.txt").mkdir()
        ds = DataStore(d)
        assert ds.layers == [1]

    def test_filename_matching_is_case_insensitive(self, tmp_path):
        d = tmp_path / "src"
        d.mkdir()
        path = d / "PACKET DATA FOR LAYER 7, LASER 2.TXT"
        # Reuse the writer's content by writing a valid file then renaming.
        valid = write_layer_file(d, 7, laser=2)
        valid.rename(path)
        ds = DataStore(d)
        assert ds.layers == [7]

    def test_duplicate_layer_raises(self, tmp_path):
        """
        Two laser files for the *same* layer both map to that layer number and
        therefore collide. This documents current behavior: layer is the only
        key, the laser index is not part of it.
        """
        d = tmp_path / "src"
        d.mkdir()
        write_layer_file(d, 1, laser=1)
        write_layer_file(d, 1, laser=2)
        with pytest.raises(ValueError, match="Duplicate layer"):
            DataStore(d)


class TestProperties:
    def test_layers_sorted(self, store):
        assert store.layers == [1, 2, 5]

    def test_columns_appends_layer_and_z(self, store):
        assert store.columns == [*EXPECTED_COLUMNS, "layer", "Z"]
        assert len(store.columns) == 19

    def test_cache_path_is_zero_padded(self, store):
        p = store._cache_path(42)
        assert p.name == "layer=00042.parquet"
        assert p.parent == store.cache_dir

    def test_repr_reports_count_and_range(self, store):
        text = repr(store)
        assert "layers=3" in text
        assert "1" in text and "5" in text
        assert "0.03" in text


class TestBuildCache:
    def test_build_cache_creates_parquet_per_layer(self, store):
        store.build_cache(verbose=False)
        for layer in store.layers:
            assert store._cache_path(layer).exists()

    def test_build_cache_adds_layer_and_z_columns(self, store):
        store.build_cache(layers=[5], verbose=False)
        df = pl.read_parquet(store._cache_path(5))
        assert "layer" in df.columns and "Z" in df.columns
        assert df.schema["layer"] == pl.Int16
        assert df.schema["Z"] == pl.Float32
        # Z = layer * layer_thickness
        assert df["Z"].unique().to_list() == pytest.approx([5 * 0.03])
        assert df["layer"].unique().to_list() == [5]

    def test_converted_columns_are_exactly_expected_plus_layer_z(self, store):
        store.build_cache(layers=[1], verbose=False)
        df = pl.read_parquet(store._cache_path(1))
        assert df.columns == [*EXPECTED_COLUMNS, "layer", "Z"]

    def test_build_cache_skips_up_to_date(self, store):
        store.build_cache(verbose=False)
        cache = store._cache_path(1)
        first_mtime = cache.stat().st_mtime
        time.sleep(0.01)
        store.build_cache(verbose=False)
        assert cache.stat().st_mtime == first_mtime

    def test_force_rebuilds_even_when_fresh(self, store):
        store.build_cache(verbose=False)
        cache = store._cache_path(1)
        _touch_in_future(cache, seconds=10)
        forced_before = cache.stat().st_mtime
        time.sleep(0.01)
        store.build_cache(force=True, verbose=False)
        assert cache.stat().st_mtime != forced_before

    def test_build_cache_only_requested_layers(self, store):
        store.build_cache(layers=[2], verbose=False)
        assert store._cache_path(2).exists()
        assert not store._cache_path(1).exists()
        assert not store._cache_path(5).exists()

    def test_build_cache_ignores_unknown_layers(self, store):
        store.build_cache(layers=[1, 999], verbose=False)
        assert store._cache_path(1).exists()
        assert not store._cache_path(999).exists()

    def test_missing_expected_column_raises(self, tmp_path):
        d = tmp_path / "src"
        d.mkdir()
        cols = [c for c in EXPECTED_COLUMNS if c != "LaserVIEW (mean)"]
        write_layer_file(d, 1, columns=cols)
        ds = DataStore(d)
        with pytest.raises(ValueError, match="missing expected columns"):
            ds.build_cache(verbose=False)

    def test_trailing_empty_column_is_dropped(self, tmp_path):
        d = tmp_path / "src"
        d.mkdir()
        write_layer_file(d, 1, trailing_tab=True)
        ds = DataStore(d)
        ds.build_cache(verbose=False)
        df = pl.read_parquet(ds._cache_path(1))
        assert "" not in df.columns
        assert df.columns == [*EXPECTED_COLUMNS, "layer", "Z"]

    def test_z_scales_with_layer_thickness(self, tmp_path):
        d = tmp_path / "src"
        d.mkdir()
        write_layer_file(d, 10)
        ds = DataStore(d, layer_thickness=0.05)
        ds.build_cache(verbose=False)
        df = pl.read_parquet(ds._cache_path(10))
        assert df["Z"].unique().to_list() == pytest.approx([10 * 0.05])


class TestNeedsRebuild:
    def test_true_when_cache_missing(self, store):
        assert store._needs_rebuild(1) is True

    def test_false_when_fresh(self, store):
        store.build_cache(layers=[1], verbose=False)
        assert store._needs_rebuild(1) is False

    def test_true_when_source_newer_than_cache(self, store):
        store.build_cache(layers=[1], verbose=False)
        _touch_in_future(store._source_files[1], seconds=10)
        assert store._needs_rebuild(1) is True

    def test_true_when_z_dtype_is_wrong(self, store):
        store.cache_dir.mkdir(parents=True, exist_ok=True)
        bad = pl.DataFrame({"Z": pl.Series([0.1, 0.2], dtype=pl.Float64)})
        bad.write_parquet(store._cache_path(1))
        _touch_in_future(store._cache_path(1), seconds=10)
        assert store._needs_rebuild(1) is True

    def test_true_when_cache_is_corrupt(self, store):
        store.cache_dir.mkdir(parents=True, exist_ok=True)
        store._cache_path(1).write_bytes(b"not a parquet file")
        _touch_in_future(store._cache_path(1), seconds=10)
        assert store._needs_rebuild(1) is True


class TestResolveLayers:
    def test_none_returns_all(self, store):
        assert store._resolve_layers(None) == {1, 2, 5}

    def test_range(self, store):
        assert store._resolve_layers(range(1, 3)) == {1, 2}

    def test_tuple_is_inclusive(self, store):
        assert store._resolve_layers((1, 5)) == {1, 2, 5}

    def test_explicit_iterable(self, store):
        assert store._resolve_layers([1, 5]) == {1, 5}
        assert store._resolve_layers({2}) == {2}

    def test_filters_out_missing_layers(self, store):
        assert store._resolve_layers([1, 3, 4, 5]) == {1, 5}

    def test_all_missing_raises(self, store):
        with pytest.raises(ValueError, match="None of the requested layers exist"):
            store._resolve_layers([100, 200])


class TestQuery:
    def test_returns_all_layers_by_default(self, store):
        df = store.query()
        assert set(df["layer"].unique().to_list()) == {1, 2, 5}
        assert df.height == 3 * 8  # three layers, eight rows each

    def test_layer_range_filter(self, store):
        df = store.query(layers=range(1, 3))
        assert set(df["layer"].unique().to_list()) == {1, 2}

    def test_layer_tuple_inclusive(self, store):
        df = store.query(layers=(1, 2))
        assert set(df["layer"].unique().to_list()) == {1, 2}

    def test_column_selection_always_includes_layer_and_z(self, store):
        df = store.query(layers=[1], columns=["Demand X"])
        assert df.columns == ["Demand X", "layer", "Z"]

    def test_column_selection_preserves_order_and_dedupes(self, store):
        df = store.query(layers=[1], columns=["Demand Y", "Demand X", "layer"])
        assert df.columns == ["Demand Y", "Demand X", "layer", "Z"]

    def test_unknown_column_raises(self, store):
        with pytest.raises(KeyError, match="Unknown column"):
            store.query(layers=[1], columns=["nope"])

    def test_x_range_filter_is_inclusive(self, tmp_path):
        d = tmp_path / "src"
        d.mkdir()
        write_layer_file(d, 1, n_rows=5, xs=[0, 1, 2, 3, 4], ys=[0, 0, 0, 0, 0])
        ds = DataStore(d)
        df = ds.query(x_range=(1.0, 3.0))
        assert sorted(df["Demand X"].to_list()) == pytest.approx([1.0, 2.0, 3.0])

    def test_y_range_filter_is_inclusive(self, tmp_path):
        d = tmp_path / "src"
        d.mkdir()
        write_layer_file(d, 1, n_rows=5, xs=[0, 0, 0, 0, 0], ys=[0, 1, 2, 3, 4])
        ds = DataStore(d)
        df = ds.query(y_range=(2.0, 4.0))
        assert sorted(df["Demand Y"].to_list()) == pytest.approx([2.0, 3.0, 4.0])

    def test_filters_dict(self, tmp_path):
        d = tmp_path / "src"
        d.mkdir()
        write_layer_file(d, 1, n_rows=5, xs=[10, 11, 12, 13, 14], ys=[0, 1, 2, 3, 4])
        ds = DataStore(d)
        df = ds.query(filters={"Demand X": (11.0, 13.0)})
        assert sorted(df["Demand X"].to_list()) == pytest.approx([11.0, 12.0, 13.0])

    def test_combined_filters_are_anded(self, tmp_path):
        d = tmp_path / "src"
        d.mkdir()
        write_layer_file(d, 1, n_rows=5, xs=[0, 1, 2, 3, 4], ys=[0, 1, 2, 3, 4])
        ds = DataStore(d)
        df = ds.query(x_range=(1.0, 3.0), y_range=(2.0, 4.0))
        assert sorted(df["Demand X"].to_list()) == pytest.approx([2.0, 3.0])

    def test_unknown_filter_column_raises(self, store):
        with pytest.raises(KeyError, match="Unknown column in filters"):
            store.query(layers=[1], filters={"bogus": (0.0, 1.0)})

    def test_empty_layer_selection_raises(self, store):
        with pytest.raises(ValueError, match="No layers selected"):
            store.query(layers=[])

    def test_query_triggers_cache_build(self, store):
        assert not store._cache_path(1).exists()
        store.query(layers=[1])
        assert store._cache_path(1).exists()
        assert not store._cache_path(5).exists()


class TestSummary:
    def test_summary_shape_and_counts(self, store):
        s = store.summary()
        assert s.columns == ["layer", "n_rows", "x_min", "x_max", "y_min", "y_max"]
        assert s["layer"].to_list() == [1, 2, 5]
        assert s["n_rows"].to_list() == [8, 8, 8]

    def test_summary_ranges_match_data(self, tmp_path):
        d = tmp_path / "src"
        d.mkdir()
        write_layer_file(d, 1, n_rows=4, xs=[2, 5, 1, 4], ys=[7, 3, 9, 1])
        ds = DataStore(d)
        s = ds.summary()
        row = s.row(0, named=True)
        assert row["x_min"] == pytest.approx(1.0)
        assert row["x_max"] == pytest.approx(5.0)
        assert row["y_min"] == pytest.approx(1.0)
        assert row["y_max"] == pytest.approx(9.0)


class TestRoundTrip:
    def test_values_survive_conversion(self, tmp_path):
        d = tmp_path / "src"
        d.mkdir()
        write_layer_file(d, 3, n_rows=3, xs=[1.25, 2.5, 3.75], ys=[4.0, 5.0, 6.0])
        ds = DataStore(d)
        df = ds.query(layers=[3], columns=["Start time", "Demand X", "Demand Y"])
        assert df["Demand X"].to_list() == pytest.approx([1.25, 2.5, 3.75])
        assert df["Demand Y"].to_list() == pytest.approx([4.0, 5.0, 6.0])
        assert df["Start time"].to_list() == [1000, 1001, 1002]

    def test_cache_format_version_constant_present(self):
        assert isinstance(CACHE_FORMAT_VERSION, int)
