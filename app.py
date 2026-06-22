"""
app.py - GUI for AMPM analysis

Select a packet directory, load/mask/assign data, add derived columns,
pick a view (plot type), configure axes and settings, and plot.
"""

import builtins
import json
import re
import sys
import traceback
from pathlib import Path
from typing import cast

from PyQt6.QtCore import QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).parent))

UI_STATE_VERSION = 2
UI_STATE_FILENAME = ".ampm-ui.json"

AVAILABLE_CORRECTIONS = {
    ("MAIN", "MeltVIEW melt pool (mean)"): {
        "label": "MAIN / MeltVIEW melt pool (mean)",
    },
}

# Pipeline params the GUI may edit
_OVERLAY_KEYS = (
    "LAYER_THICKNESS",
    "METHOD",
    "MAX_DISTANCE_MM",
    "EPS_XY",
    "EPS_Z",
    "MIN_SAMPLES",
    "LAYERS_PER_CHUNK",
    "OVERLAP_LAYERS",
    "APPLY_MASK",
    "ASSIGN_PARTS",
    "DHXML",
    "PART_EXCLUDE",
    "LAYER_RANGE",
    "LOAD_COLUMNS",
    "X_RANGE",
    "Y_RANGE",
    "APPLY_CORRECTION",
    "CORRECTION_MACHINE",
    "CORRECTION_COLUMN",
)


class NoScrollComboBox(QComboBox):
    """A combo box that ignores wheel events."""

    def wheelEvent(self, e):  # noqa: N802 (Qt naming)
        if e is not None:
            e.ignore()


def build_widget(spec: dict) -> QWidget:
    """Create the appropriate widget for a setting spec."""
    wtype = spec["type"]

    match wtype:
        case "float":
            widget = QDoubleSpinBox()
            widget.setDecimals(4)
            widget.setRange(spec.get("min", 0.0), spec.get("max", 999999.0))
            widget.setValue(spec.get("default", 0.0))
            return widget

        case "int":
            widget = QSpinBox()
            widget.setRange(spec.get("min", 0), spec.get("max", 999999))
            widget.setValue(spec.get("default", 0))
            return widget

        case "choice":
            widget = NoScrollComboBox()
            widget.addItems(spec.get("options", []))
            default = spec.get("default")
            if default is not None:
                widget.setCurrentText(str(default))
            return widget

        case "bool":
            widget = QCheckBox()
            widget.setChecked(spec.get("default", False))
            return widget

        case "float_or_none":
            widget = QLineEdit()
            default = spec.get("default")
            widget.setText("none" if default is None else str(default))

            if spec.get("tooltip"):
                widget.setToolTip(spec["tooltip"])

            return widget

        case "int_or_auto":
            widget = QLineEdit()
            default = spec.get("default")
            widget.setText("auto" if default is None else str(default))

            if spec.get("tooltip"):
                widget.setToolTip(spec["tooltip"])

            return widget

        case _:
            widget = QLineEdit()
            widget.setText(str(spec.get("default", "")))
            return widget


def read_widget(widget, spec):
    """Read the current value from a widget."""
    wtype = spec["type"]

    match wtype:
        case "float" | "int":
            return widget.value()

        case "choice":
            return widget.currentText()

        case "bool":
            return widget.isChecked()

        case "float_or_none":
            text = widget.text().strip().lower()
            return None if text == "none" else float(text)

        case "int_or_auto":
            text = widget.text().strip().lower()
            return None if text == "auto" else int(text)

        case _:
            return widget.text()


def set_widget_value(widget, spec, value):
    """Set a widget's value from a config dict entry."""
    wtype = spec["type"]

    match wtype:
        case "float":
            widget.setValue(float(value))

        case "int":
            widget.setValue(int(value))

        case "choice":
            widget.setCurrentText(str(value))

        case "bool":
            widget.setChecked(bool(value))

        case "float_or_none":
            widget.setText("none" if value is None else str(value))

        case "int_or_auto":
            widget.setText("auto" if value is None else str(value))

        case _:
            widget.setText(str(value))


def _layer_range_tag(layer_range) -> str:
    """Filesystem-safe tag for a layer range, used to name per-range caches."""
    if layer_range is None:
        return "all"
    lo, hi = int(layer_range[0]), int(layer_range[1])
    return f"L{lo:05d}-{hi:05d}"


def _correction_machines():
    """Distinct machines that have at least one calibrated correction."""
    seen = []
    for machine, _ in AVAILABLE_CORRECTIONS:
        if machine not in seen:
            seen.append(machine)
    return seen


def _correction_columns(machine):
    """Columns with a calibrated correction for the given machine."""
    return [col for (m, col) in AVAILABLE_CORRECTIONS if m == machine]


def _ui_state_path(project_root) -> Path:
    return Path(project_root) / UI_STATE_FILENAME


def load_ui_state(project_root) -> dict:
    """Load the sidecar UI state. Returns {} on any problem (tolerant)."""

    path = _ui_state_path(project_root)
    if not path.is_file():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    except (OSError, ValueError):
        return {}

    if not isinstance(data, dict):
        return {}

    # Future versions degrade to defaults rather than mis-loading
    if data.get("version") not in (None, UI_STATE_VERSION):
        return {}

    return data


def save_ui_state(project_root, state: dict) -> None:
    """Write the sidecar UI state atomically."""
    path = _ui_state_path(project_root)
    tmp = path.parent / (path.name + ".tmp")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        tmp.replace(path)

    except OSError:
        pass


class CollapsibleSection(QWidget):
    """A header button that shows/hides a content area below it."""

    def __init__(self, title: str, expanded: bool = False, parent=None) -> None:
        super().__init__(parent)

        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._toggle.toggled.connect(self._on_toggled)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(12, 0, 0, 0)
        self._content.setVisible(expanded)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._toggle)
        outer.addWidget(self._content)

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self._content.setVisible(checked)

    def content_layout(self) -> QVBoxLayout:
        """The layout callers should add their rows/widgets to."""
        return self._content_layout

    def set_title(self, title: str) -> None:
        """Update the header text (e.g. to reflect a summary/count)."""
        self._toggle.setText(title)


class LoadWorker(QThread):
    """Loads, masks, and assigns data in a background thread."""

    log = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(object, object)
    finished_err = pyqtSignal(str)

    _CACHE_LINE = re.compile(r"\[(\d+)/(\d+)\]")

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self._phase = None
        self._load_band = (2, 55)  # (start%, end%) of the cache-build sub-phase

    def _print(self, msg):
        self.log.emit(msg)
        if self._phase == "load":
            m = self._CACHE_LINE.search(msg)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                if total > 0:
                    lo, hi = self._load_band
                    pct = lo + (hi - lo) * done / total
                    self.progress.emit(int(pct), "Loading data")

    def _emit_progress(self, pct: int, label: str) -> None:
        self.progress.emit(pct, label)

    def run(self):
        original_print = builtins.print
        builtins.print = lambda *a, **kw: self._print(" ".join(str(x) for x in a))

        try:
            df = self._load_mask_assign()
            self.finished_ok.emit(df, self.config)
        except Exception:
            self.finished_err.emit(traceback.format_exc())
        finally:
            builtins.print = original_print

    def _load_mask_assign(self):
        import polars as pl

        from ampm import DataStore
        from ampm.cluster_cache import cluster_or_load
        from ampm.clustering import cluster_dbscan_chunked
        from ampm.mask_cache import mask_or_load
        from ampm.masking import apply_mask_keep, build_mask, stl_hash
        from ampm.memprof import phase
        from ampm.parts import (
            BuildStartedDHXML,
            QuantAMParts,
            apply_part_id_map,
            assign_bounding_box_part,
            assign_nearest_part,
            compute_part_id_map,
        )

        config = self.config
        SOURCE = config["SOURCE"]
        STL = config["STL"]
        PARTS_CSV = config["PARTS_CSV"]
        DHXML = config.get("DHXML") or None
        PART_EXCLUDE = config.get("PART_EXCLUDE") or None
        LAYER_THICKNESS = config["LAYER_THICKNESS"]
        MASK_CACHE = config["MASK_CACHE"]
        MASK_KEEP_CACHE = config["MASK_KEEP_CACHE"]
        CLUSTER_CACHE = config["CLUSTER_CACHE"]
        METHOD = config["METHOD"]
        MAX_DISTANCE_MM = config["MAX_DISTANCE_MM"]
        APPLY_MASK = config.get("APPLY_MASK", True)
        ASSIGN_PARTS = config.get("ASSIGN_PARTS", True)
        APPLY_CORRECTION = config.get("APPLY_CORRECTION", False)
        CORRECTION_MACHINE = config.get("CORRECTION_MACHINE")
        CORRECTION_COLUMN = config.get("CORRECTION_COLUMN")
        LAYER_RANGE = config.get("LAYER_RANGE")  # None or (lo, hi)
        LOAD_COLUMNS = config.get("LOAD_COLUMNS")  # None or [cols]
        X_RANGE = config.get("X_RANGE")  # None or (min, max)
        Y_RANGE = config.get("Y_RANGE")  # None or (min, max)

        if APPLY_MASK and ASSIGN_PARTS:
            self._load_band, mask_band, assign_band = (2, 50), (50, 68), (68, 96)
        elif APPLY_MASK:
            self._load_band, mask_band, assign_band = (2, 75), (75, 98), None
        elif ASSIGN_PARTS:
            self._load_band, mask_band, assign_band = (2, 60), None, (60, 96)
        else:
            self._load_band, mask_band, assign_band = (2, 96), None, None

        self._phase = "load"
        self._emit_progress(self._load_band[0], "Loading data")

        store = DataStore(SOURCE, layer_thickness=LAYER_THICKNESS)
        layers_arg = (
            None if LAYER_RANGE is None else (int(LAYER_RANGE[0]), int(LAYER_RANGE[1]))
        )
        if LOAD_COLUMNS is None:
            columns_arg = None
        else:
            required = ["Demand X", "Demand Y", "Start time"]
            columns_arg = list(dict.fromkeys([*required, *LOAD_COLUMNS]))
            print(
                f"Column pruning active: loading {len(columns_arg)} signal "
                f"column(s) + layer, Z."
            )
        if X_RANGE is not None or Y_RANGE is not None:
            bounds = []
            if X_RANGE is not None:
                bounds.append(f"X\u2208[{X_RANGE[0]}, {X_RANGE[1]}]")
            if Y_RANGE is not None:
                bounds.append(f"Y\u2208[{Y_RANGE[0]}, {Y_RANGE[1]}]")
            print("Spatial filter active: " + ", ".join(bounds) + ".")
        with phase("load: store.query"):
            df = store.query(
                layers=layers_arg,
                columns=columns_arg,
                x_range=X_RANGE,
                y_range=Y_RANGE,
            )

        if LAYER_RANGE is None:
            queried_layers = list(store.layers)
        else:
            lo, hi = int(LAYER_RANGE[0]), int(LAYER_RANGE[1])
            queried_layers = [L for L in store.layers if lo <= L <= hi]
        if not queried_layers:
            raise ValueError(
                f"No layers in the selected range {LAYER_RANGE}. "
                f"Available: {min(store.layers)}–{max(store.layers)}."
            )

        layer_span = (min(queried_layers), max(queried_layers))
        print(
            f"Loaded {df.height:,} rows across {len(queried_layers)} layers "
            f"({layer_span[0]}–{layer_span[1]}).\n"
        )
        self._phase = None
        self._emit_progress(self._load_band[1], "Loaded")

        if APPLY_MASK:
            if mask_band is not None:
                self._emit_progress(mask_band[0], "Applying mask")
            mask_params = {
                "layers": layer_span,
                "stl": str(STL),
                "stl_sha256": stl_hash(STL),
                "buffer_mm": 0.0,
                "layer_thickness": LAYER_THICKNESS,
                "x_range": list(X_RANGE) if X_RANGE is not None else None,
                "y_range": list(Y_RANGE) if Y_RANGE is not None else None,
            }

            def keep_wrapper(d):
                mask = build_mask(
                    STL,
                    layers=queried_layers,
                    layer_thickness=LAYER_THICKNESS,
                    buffer_mm=0.0,
                    cache_path=MASK_CACHE,
                )
                return apply_mask_keep(d, mask)

            print("Applying mask...")
            with phase("mask: mask_or_load total"):
                df = mask_or_load(
                    df,
                    cache_path=MASK_KEEP_CACHE,
                    keep_fn=keep_wrapper,
                    params=mask_params,
                    strict=False,
                )
            print(f"After mask: {df.height:,} rows.")
            if mask_band is not None:
                self._emit_progress(mask_band[1], "Masked")
        else:
            print("Skipping mask.")

        if ASSIGN_PARTS:
            if assign_band is not None:
                self._emit_progress(assign_band[0], "Assigning parts")

            quantam = QuantAMParts.from_path(PARTS_CSV) if PARTS_CSV else None

            if METHOD == "dhxml":
                if not DHXML:
                    raise ValueError(
                        "Method 'dhxml' requires a BuildStarted DHXML "
                        "(.dhxml) but none is set."
                    )
                print("Assigning parts (dhxml bounding boxes)...")
                build_file = BuildStartedDHXML.from_path(DHXML)
                parts_table = build_file.parts_table()
                print(f"Loaded {parts_table.height} parts from BuildStarted DHXML.\n")
                with phase("assign: assign_bounding_box_part"):
                    df = assign_bounding_box_part(
                        df,
                        parts_table,
                        noise_label="noise",
                    )
            elif METHOD == "direct":
                if quantam is None:
                    raise ValueError("Method 'direct' requires a Parts CSV.")
                parts_table = quantam.parent_parts()
                print(f"Loaded {parts_table.height} parts.\n")
                print("Assigning parts (direct)...")
                with phase("assign: assign_nearest_part"):
                    df = assign_nearest_part(
                        df,
                        parts_table,
                        max_distance_mm=MAX_DISTANCE_MM,
                        noise_label="noise",
                    )
            else:
                if quantam is None:
                    raise ValueError("Method 'dbscan' requires a Parts CSV.")
                parts_table = quantam.parent_parts()
                print(f"Loaded {parts_table.height} parts.\n")
                EPS_XY = config["EPS_XY"]
                EPS_Z = config["EPS_Z"]
                MIN_SAMPLES = config["MIN_SAMPLES"]
                LAYERS_PER_CHUNK = config["LAYERS_PER_CHUNK"]
                OVERLAP_LAYERS = config["OVERLAP_LAYERS"]

                cluster_params = {
                    "layers": layer_span,
                    "stl": str(STL),
                    "buffer_mm": 0.0,
                    "eps_xy": EPS_XY,
                    "eps_z": EPS_Z,
                    "min_samples": MIN_SAMPLES,
                    "mode": "3d",
                    "layers_per_chunk": LAYERS_PER_CHUNK,
                    "overlap_layers": OVERLAP_LAYERS,
                    "layer_thickness": LAYER_THICKNESS,
                }

                def clustering_wrapper(d):
                    return cluster_dbscan_chunked(
                        d,
                        eps_xy=EPS_XY,
                        eps_z=EPS_Z,
                        min_samples=MIN_SAMPLES,
                        mode="3d",
                        layers_per_chunk=LAYERS_PER_CHUNK,
                        overlap_layers=OVERLAP_LAYERS,
                        layer_thickness=LAYER_THICKNESS,
                        verbose=True,
                    )

                print("Clustering (DBSCAN)...")
                clustered = cluster_or_load(
                    df,
                    cache_path=CLUSTER_CACHE,
                    cluster_fn=clustering_wrapper,
                    params=cluster_params,
                    strict=False,
                )
                mapping = compute_part_id_map(clustered, parts_table)
                df = apply_part_id_map(clustered, mapping, noise_label="noise")

            if quantam is not None:
                try:
                    parts_with_speed = quantam.volume_parameters_with_speed()
                except Exception as e:
                    print(f"Skipping power/speed attachment: {e}")
                    parts_with_speed = None
                if parts_with_speed is not None:
                    with phase("assign: attach power/speed (lookup)"):
                        _ids = parts_with_speed["Part ID"].to_list()
                        _power = dict(
                            zip(_ids, parts_with_speed["Hatches Power"].to_list())
                        )
                        _speed = dict(
                            zip(_ids, parts_with_speed["Hatch Speed"].to_list())
                        )
                        _pid_dtype = df["part_id"].dtype
                        if (
                            isinstance(_pid_dtype, pl.Enum)
                            and df["part_id"].null_count() == 0
                        ):
                            import numpy as np

                            _cats = _pid_dtype.categories.to_list()
                            _phys = df["part_id"].to_physical().to_numpy()
                            _pw = np.array(
                                [_power.get(c, np.nan) for c in _cats],
                                dtype=np.float32,
                            )
                            _sp = np.array(
                                [_speed.get(c, np.nan) for c in _cats],
                                dtype=np.float32,
                            )
                            df = df.with_columns(
                                pl.Series("Hatches Power", _pw[_phys]).fill_nan(None),
                                pl.Series("Hatch Speed", _sp[_phys]).fill_nan(None),
                            )
                        else:  # String part_id (e.g. DBSCAN path)
                            df = df.with_columns(
                                pl.col("part_id")
                                .replace_strict(
                                    _power, default=None, return_dtype=pl.Float64
                                )
                                .alias("Hatches Power"),
                                pl.col("part_id")
                                .replace_strict(
                                    _speed, default=None, return_dtype=pl.Float64
                                )
                                .alias("Hatch Speed"),
                            )
            else:
                print("No Parts CSV provided; skipping power/speed attachment.")

            if PART_EXCLUDE and "part_id" in df.columns:
                exclude = [str(p) for p in PART_EXCLUDE]
                before = df.height
                df = df.filter(~pl.col("part_id").cast(pl.String).is_in(exclude))
                dropped = before - df.height
                print(
                    f"Part filter: excluded {len(exclude)} part(s); "
                    f"{dropped:,} row(s) dropped, {df.height:,} remain."
                )
                if df.is_empty():
                    print(
                        "Warning: the part filter removed every row. Check "
                        "which parts are ticked in the Part filter section."
                    )

            if assign_band is not None:
                self._emit_progress(assign_band[1], "Assigned")
        else:
            print("Skipping part assignment.")

        if APPLY_CORRECTION:
            if CORRECTION_COLUMN is None:
                raise ValueError("CORRECTION_COLUMN not set.")

            key = (CORRECTION_MACHINE, CORRECTION_COLUMN)
            if key not in AVAILABLE_CORRECTIONS:
                print(
                    f"No correction calibrated for machine "
                    f"{CORRECTION_MACHINE!r} / column {CORRECTION_COLUMN!r}; "
                    f"skipping correction."
                )
            else:
                from ampm.correction import MeltPoolCorrection

                print(
                    f"Applying correction: {CORRECTION_MACHINE} / "
                    f"{CORRECTION_COLUMN}..."
                )
                try:
                    df = MeltPoolCorrection().apply(df, meltpool_col=CORRECTION_COLUMN)
                    print(f"Added '{CORRECTION_COLUMN} corrected' column.")
                except KeyError as e:
                    print(
                        f"Correction skipped: required column missing ({e}). "
                        f"The correction needs Demand X/Y, LaserVIEW (mean), "
                        f"and {CORRECTION_COLUMN!r}."
                    )

        print(f"Data ready: {df.height:,} rows, {len(df.columns)} columns.")
        self._emit_progress(100, "Ready")
        return df


class PlotWorker(QThread):
    """Runs a view's run() function in a background thread."""

    log = pyqtSignal(str)
    finished_ok = pyqtSignal()
    finished_err = pyqtSignal(str)

    def __init__(self, view_module, df, config, axes, settings):
        super().__init__()
        self.view_module = view_module
        self.df = df
        self.config = config
        self.axes = axes
        self.settings = settings

    def _print(self, msg):
        self.log.emit(msg)

    def run(self):
        original_print = builtins.print
        builtins.print = lambda *a, **kw: self._print(" ".join(str(x) for x in a))

        try:
            self.view_module.run(self.df, self.config, self.axes, self.settings)
            self.finished_ok.emit()
        except Exception:
            self.finished_err.emit(traceback.format_exc())
        finally:
            builtins.print = original_print


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AMPM Analyzer")
        self.setMinimumSize(1000, 700)

        from PyQt6.QtGui import QIcon

        base = (
            Path(sys._MEIPASS)  # type: ignore[attr-defined]
            if getattr(sys, "frozen", False)
            else Path(__file__).parent
        )
        self.setWindowIcon(QIcon(str(base / "assets" / "ampm.ico")))

        self._load_worker = None
        self._plot_worker = None
        self._config = None
        self._df = None
        self._derived = {}
        self._derived_recipes = {}  # {"stat","signal","mode"}
        self._views = {}
        self._views_loaded = False
        self._axis_combos = {}
        self._setting_widgets = {}
        self._available_layers = None
        self._project_root = None
        self._pending_resume = None  # analysis state
        self._settings = QSettings("AMPM", "AMPM Analyzer")

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)  # left, config/analysis, right, log

        self._tabs = QTabWidget()
        splitter.addWidget(self._tabs)

        ## CONFIG TAB
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_scroll.setWidget(config_widget)
        self._tabs.addTab(config_scroll, "Config")

        # Data Sources
        sources_group = QGroupBox("Data Sources")
        sources_layout = QVBoxLayout(sources_group)

        dir_row = QHBoxLayout()
        dir_lbl = QLabel("Project root:")
        dir_lbl.setFixedWidth(self._LABEL_WIDTH)
        dir_row.addWidget(dir_lbl)
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("Select project root directory...")
        self._dir_edit.setReadOnly(True)
        dir_browse = QPushButton("Browse...")
        dir_browse.clicked.connect(self._browse_build_dir)
        dir_row.addWidget(self._dir_edit, stretch=1)
        dir_row.addWidget(dir_browse)
        sources_layout.addLayout(dir_row)

        self._paths_section = CollapsibleSection("Data source paths", expanded=False)
        sources_layout.addWidget(self._paths_section)
        paths_layout = self._paths_section.content_layout()

        self._source_edit = self._make_path_row(paths_layout, "Source:", is_dir=True)
        self._stl_edit = self._make_path_row(
            paths_layout, "STL:", file_filter="STL files (*.stl)"
        )
        self._csv_edit = self._make_path_row(
            paths_layout, "Parts CSV:", file_filter="CSV files (*.csv)"
        )
        self._dhxml_edit = self._make_path_row(
            paths_layout,
            "BuildStarted DHXML:",
            file_filter="BuildStarted DHXML (*.dhxml)",
        )

        self._source_edit.textChanged.connect(self._probe_layers)

        config_layout.addWidget(sources_group)

        # Settings
        self._assign_group = QWidget()
        assign_layout = QVBoxLayout(self._assign_group)
        assign_layout.setContentsMargins(0, 0, 0, 0)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._method_combo = NoScrollComboBox()
        self._method_combo.addItems(["direct", "dbscan", "dhxml"])
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        method_row.addWidget(self._method_combo, stretch=1)
        assign_layout.addLayout(method_row)

        self._dist_widget = QWidget()
        dist_row = QHBoxLayout(self._dist_widget)
        dist_row.setContentsMargins(0, 0, 0, 0)
        dist_row.addWidget(QLabel("Max distance (mm):"))
        self._max_dist_edit = QLineEdit("none")
        self._max_dist_edit.setToolTip(
            "'none' assigns every row regardless of distance"
        )
        dist_row.addWidget(self._max_dist_edit, stretch=1)
        assign_layout.addWidget(self._dist_widget)

        # DBSCAN params
        self._cluster_widget = QWidget()
        cluster_layout = QVBoxLayout(self._cluster_widget)
        cluster_layout.setContentsMargins(0, 0, 0, 0)

        self._eps_xy_edit = self._make_float_row(cluster_layout, "EPS_XY (mm):", "0.3")
        self._eps_z_edit = self._make_float_row(cluster_layout, "EPS_Z (mm):", "0.06")
        self._min_samples_spin = self._make_int_row(
            cluster_layout, "Min samples:", 10, 1, 1000
        )
        self._chunk_spin = self._make_int_row(
            cluster_layout, "Layers per chunk:", 11, 1, 500
        )

        overlap_row = QHBoxLayout()
        overlap_row.addWidget(QLabel("Overlap layers:"))
        self._overlap_edit = QLineEdit("auto")
        self._overlap_edit.setToolTip("'auto' or an integer")
        overlap_row.addWidget(self._overlap_edit, stretch=1)
        cluster_layout.addLayout(overlap_row)

        self._cluster_widget.setVisible(False)
        assign_layout.addWidget(self._cluster_widget)

        self._dhxml_hint = QLabel(
            "Uses the part bounding boxes in the BuildStarted DHXML "
            "(set 'BuildStarted DHXML' above). Points outside every box are "
            "labelled 'noise' \u2014 switch to 'direct' or 'dbscan' to catch them."
        )
        self._dhxml_hint.setWordWrap(True)
        self._dhxml_hint.setVisible(False)
        assign_layout.addWidget(self._dhxml_hint)

        # Parts filter
        self._part_filter_section = CollapsibleSection("Part filter", expanded=False)
        pf_layout = self._part_filter_section.content_layout()

        pf_hint = QLabel(
            "Untick parts to drop them from the analysis (e.g. parts a "
            "colleague placed on the same plate). Unassigned 'noise' rows are "
            "kept regardless."
        )
        pf_hint.setWordWrap(True)
        pf_layout.addWidget(pf_hint)

        pf_btn_row = QHBoxLayout()
        self._pf_all_btn = QPushButton("Select all")
        self._pf_none_btn = QPushButton("Select none")
        self._pf_all_btn.clicked.connect(lambda: self._set_all_parts_checked(True))
        self._pf_none_btn.clicked.connect(lambda: self._set_all_parts_checked(False))
        pf_btn_row.addWidget(self._pf_all_btn)
        pf_btn_row.addWidget(self._pf_none_btn)
        pf_btn_row.addStretch()
        pf_layout.addLayout(pf_btn_row)

        self._part_list = QListWidget()
        self._part_list.setMaximumHeight(180)
        self._part_list.itemChanged.connect(self._on_part_item_changed)
        pf_layout.addWidget(self._part_list)

        self._pf_status = QLabel("No parts loaded.")
        self._pf_status.setWordWrap(True)
        self._pf_status.setStyleSheet("color: gray;")
        pf_layout.addWidget(self._pf_status)

        assign_layout.addWidget(self._part_filter_section)

        # Correction
        self._correction_group = QWidget()
        correction_layout = QVBoxLayout(self._correction_group)
        correction_layout.setContentsMargins(0, 0, 0, 0)

        machine_row = QHBoxLayout()
        machine_row.addWidget(QLabel("Machine:"))
        self._correction_machine_combo = NoScrollComboBox()
        self._correction_machine_combo.addItems(_correction_machines())
        self._correction_machine_combo.currentTextChanged.connect(
            self._on_correction_machine_changed
        )
        machine_row.addWidget(self._correction_machine_combo, stretch=1)
        correction_layout.addLayout(machine_row)

        column_row = QHBoxLayout()
        column_row.addWidget(QLabel("Column:"))
        self._correction_column_combo = NoScrollComboBox()
        column_row.addWidget(self._correction_column_combo, stretch=1)
        correction_layout.addLayout(column_row)
        self._on_correction_machine_changed(
            self._correction_machine_combo.currentText()
        )

        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)

        lt_row = QHBoxLayout()
        lt_row.addWidget(QLabel("Layer thickness (mm):"))
        self._lt_edit = QLineEdit()
        self._lt_edit.setPlaceholderText("e.g. 0.03")
        lt_row.addWidget(self._lt_edit, stretch=1)
        settings_layout.addLayout(lt_row)

        # Layer range
        self._all_layers_check = QCheckBox("All layers")
        self._all_layers_check.setChecked(True)
        self._all_layers_check.setToolTip(
            "Load every layer in the source. Uncheck to load a sub-range."
        )
        self._all_layers_check.toggled.connect(self._on_all_layers_toggled)
        settings_layout.addWidget(self._all_layers_check)

        self._layer_range_widget = QWidget()
        layer_range_layout = QHBoxLayout(self._layer_range_widget)
        layer_range_layout.setContentsMargins(0, 0, 0, 0)
        layer_range_layout.addWidget(QLabel("From:"))
        self._layer_from_spin = QSpinBox()
        self._layer_from_spin.setRange(0, 999999)
        layer_range_layout.addWidget(self._layer_from_spin)
        layer_range_layout.addWidget(QLabel("To:"))
        self._layer_to_spin = QSpinBox()
        self._layer_to_spin.setRange(0, 999999)
        layer_range_layout.addWidget(self._layer_to_spin)
        self._layer_avail_label = QLabel("Available: \u2014")
        self._layer_avail_label.setStyleSheet("color: gray;")
        layer_range_layout.addWidget(self._layer_avail_label, stretch=1)
        self._layer_range_widget.setEnabled(False)  # follows "All layers" checkbox
        settings_layout.addWidget(self._layer_range_widget)

        # Columns to load
        cols_row = QHBoxLayout()
        cols_row.addWidget(QLabel("Columns:"))
        self._columns_edit = QLineEdit()
        self._columns_edit.setText("all")
        self._columns_edit.setPlaceholderText(
            "all  — or comma-separated signal columns"
        )
        self._columns_edit.setToolTip(
            "Signal columns to load, comma-separated (e.g. 'MeltVIEW melt pool "
            "(mean), Laser output power (mean)'). 'all' loads every column. "
            "Demand X/Y, Start time, layer, and Z are always included. For "
            "dense builds, loading only the columns you analyze cuts memory "
            "roughly in proportion."
        )
        cols_row.addWidget(self._columns_edit, stretch=1)
        settings_layout.addLayout(cols_row)

        # X / Y spatial range (optional). Blank = no bound on that side.
        xr_row = QHBoxLayout()
        xr_row.addWidget(QLabel("X range:"))
        self._x_min_edit = QLineEdit()
        self._x_min_edit.setPlaceholderText("min")
        self._x_min_edit.setValidator(QDoubleValidator())
        xr_row.addWidget(self._x_min_edit)
        xr_row.addWidget(QLabel("to"))
        self._x_max_edit = QLineEdit()
        self._x_max_edit.setPlaceholderText("max")
        self._x_max_edit.setValidator(QDoubleValidator())
        xr_row.addWidget(self._x_max_edit)
        xr_row.addWidget(QLabel("Y range:"))
        self._y_min_edit = QLineEdit()
        self._y_min_edit.setPlaceholderText("min")
        self._y_min_edit.setValidator(QDoubleValidator())
        xr_row.addWidget(self._y_min_edit)
        xr_row.addWidget(QLabel("to"))
        self._y_max_edit = QLineEdit()
        self._y_max_edit.setPlaceholderText("max")
        self._y_max_edit.setValidator(QDoubleValidator())
        xr_row.addWidget(self._y_max_edit)
        _xy_tip = (
            "Optional inclusive bounds on Demand X / Demand Y (mm), applied "
            "at load time. Leave a box blank for no bounds."
        )
        for _w in (
            self._x_min_edit,
            self._x_max_edit,
            self._y_min_edit,
            self._y_max_edit,
        ):
            _w.setToolTip(_xy_tip)
        settings_layout.addLayout(xr_row)

        self._mask_check = QCheckBox("Apply STL mask")
        self._mask_check.setChecked(True)
        self._mask_check.setToolTip(
            "Filter rows to the part region using the STL geometry"
        )
        settings_layout.addWidget(self._mask_check)

        self._assign_check = QCheckBox("Assign parts")
        self._assign_check.setChecked(True)
        self._assign_check.setToolTip("Assign each row to its nearest part ID")
        self._assign_check.toggled.connect(self._assign_group.setVisible)
        settings_layout.addWidget(self._assign_check)

        settings_layout.addWidget(self._assign_group)

        self._correction_check = QCheckBox("Correction")
        self._correction_check.setChecked(False)
        self._correction_check.setToolTip(
            "Apply a pre-fitted spatial correction to a sensor signal, adding "
            "a corrected column alongside the original."
        )
        self._correction_check.toggled.connect(self._correction_group.setVisible)
        settings_layout.addWidget(self._correction_check)

        self._correction_group.setVisible(False)
        settings_layout.addWidget(self._correction_group)

        config_layout.addWidget(settings_group)

        # Load button
        self._load_btn = QPushButton("Load Data")
        self._load_btn.setEnabled(True)
        self._load_btn.setMinimumHeight(40)
        self._load_btn.clicked.connect(self._load_data)
        config_layout.addWidget(self._load_btn)

        self._load_progress = QProgressBar()
        self._load_progress.setRange(0, 100)
        self._load_progress.setTextVisible(True)
        self._load_progress.setVisible(False)
        config_layout.addWidget(self._load_progress)

        config_layout.addStretch()

        ## ANALYSIS
        analysis_scroll = QScrollArea()
        analysis_scroll.setWidgetResizable(True)
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        analysis_scroll.setWidget(analysis_widget)

        # Derived columns
        derived_group = QGroupBox("Derived Columns")
        derived_layout = QVBoxLayout(derived_group)

        self._derived_list = QListWidget()
        self._derived_list.setMaximumHeight(80)
        derived_layout.addWidget(self._derived_list)

        add_row = QHBoxLayout()
        add_row.addWidget(QLabel("Stat:"))
        self._derived_stat = NoScrollComboBox()
        self._derived_stat.addItems(["CoV"])
        add_row.addWidget(self._derived_stat)

        add_row.addWidget(QLabel("Signal:"))
        self._derived_signal = NoScrollComboBox()
        add_row.addWidget(self._derived_signal, stretch=1)

        add_row.addWidget(QLabel("Mode:"))
        self._derived_mode = NoScrollComboBox()
        self._derived_mode.addItems(["overall", "per_layer_mean", "across_layers"])
        add_row.addWidget(self._derived_mode)

        derived_layout.addLayout(add_row)

        btn_row = QHBoxLayout()
        self._add_derived_btn = QPushButton("Add")
        self._add_derived_btn.clicked.connect(self._add_derived)
        btn_row.addWidget(self._add_derived_btn)

        self._remove_derived_btn = QPushButton("Remove")
        self._remove_derived_btn.clicked.connect(self._remove_derived)
        btn_row.addWidget(self._remove_derived_btn)
        btn_row.addStretch()

        derived_layout.addLayout(btn_row)
        analysis_layout.addWidget(derived_group)

        # View selector
        view_group = QGroupBox("Plot")
        view_layout = QVBoxLayout(view_group)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Type:"))
        self._view_combo = NoScrollComboBox()
        self._view_combo.addItems(list(self._views.keys()))
        self._view_combo.currentTextChanged.connect(self._on_view_changed)
        selector_row.addWidget(self._view_combo, stretch=1)
        self._reload_views_btn = QPushButton("Reload Views")
        self._reload_views_btn.setToolTip(
            "Re-scan for view files (built-in, per-build, user, AMPM_VIEWS_PATH)"
        )
        self._reload_views_btn.clicked.connect(self._on_reload_views)
        selector_row.addWidget(self._reload_views_btn)
        view_layout.addLayout(selector_row)

        self._description_label = QLabel()
        self._description_label.setWordWrap(True)
        self._description_label.setStyleSheet("color: gray; font-style: italic;")
        view_layout.addWidget(self._description_label)

        # Axes
        self._axes_group = QGroupBox("Axes")
        self._axes_layout = QVBoxLayout(self._axes_group)
        view_layout.addWidget(self._axes_group)

        # Settings
        self._settings_group = QGroupBox("Settings")
        self._settings_layout = QVBoxLayout(self._settings_group)
        view_layout.addWidget(self._settings_group)

        # Plot button
        self._plot_btn = QPushButton("Plot")
        self._plot_btn.setMinimumHeight(40)
        self._plot_btn.clicked.connect(self._run_plot)

        self._plot_progress = QProgressBar()
        self._plot_progress.setTextVisible(False)
        self._plot_progress.setVisible(False)

        analysis_layout.addWidget(view_group)
        analysis_layout.addWidget(self._plot_btn)
        analysis_layout.addWidget(self._plot_progress)
        analysis_layout.addStretch()

        self._analysis_tab = analysis_scroll

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        log_layout.addWidget(self._log)
        splitter.addWidget(log_group)

        splitter.setSizes([600, 400])  # left, 60%; right, 40%

        for edit in (self._stl_edit, self._csv_edit, self._dhxml_edit):
            edit.textChanged.connect(self._refresh_options)
        self._csv_edit.textChanged.connect(self._repopulate_part_filter)
        self._dhxml_edit.textChanged.connect(self._repopulate_part_filter)
        self._refresh_options()

        if self._views:
            self._on_view_changed(self._view_combo.currentText())

    def closeEvent(self, a0):
        """Save UI state and wait for any running threads before closing."""
        self._save_ui_state()

        for worker in (self._load_worker, self._plot_worker):
            if worker is not None and worker.isRunning():
                worker.quit()
                worker.wait()

        if a0 is not None:
            a0.accept()

    _LABEL_WIDTH = 100

    def _make_path_row(self, parent_layout, label, is_dir=False, file_filter=""):
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setFixedWidth(self._LABEL_WIDTH)
        row.addWidget(lbl)
        edit = QLineEdit()
        edit.setReadOnly(True)
        browse = QPushButton("Browse...")

        def on_browse():
            if is_dir:
                path = QFileDialog.getExistingDirectory(self, f"Select {label}")
            else:
                path, _ = QFileDialog.getOpenFileName(
                    self, f"Select {label}", "", file_filter
                )
            if path:
                edit.setText(path)

        browse.clicked.connect(on_browse)
        row.addWidget(edit, stretch=1)
        row.addWidget(browse)
        parent_layout.addLayout(row)
        return edit

    def _find_dhxml(self, source_dir: str) -> str:
        """First ``*.dhxml`` in the source dir or project root, else ""."""
        candidates: list[Path] = []
        for d in (source_dir, self._project_root):
            if d and Path(d).is_dir():
                candidates.extend(sorted(Path(d).glob("*.dhxml")))
        return str(candidates[0]) if candidates else ""

    def _read_parts_for_filter(self, method: str):
        """``(rows, error)`` with ``rows = [(part_id, x, y), ...]``, parsed from
        the method's parts file. IDs match those assignment produces."""
        try:
            if method == "dhxml":
                path = self._dhxml_edit.text().strip()
                if not path:
                    return [], "Select a BuildStarted DHXML above to list parts."
                if not Path(path).is_file():
                    return [], "BuildStarted DHXML not found."
                from ampm.parts import BuildStartedDHXML

                pt = BuildStartedDHXML.from_path(path).parts_table()
            else:
                path = self._csv_edit.text().strip()
                if not path:
                    return [], "Select a Parts CSV above to list parts."
                if not Path(path).is_file():
                    return [], "Parts CSV not found."
                from ampm.parts import QuantAMParts

                pt = QuantAMParts.from_path(path).parent_parts()
        except Exception as e:  # noqa: BLE001 - surface any parse failure in the UI
            return [], f"Could not read parts: {e}"

        ids = [str(p) for p in pt["Part ID"].to_list()]
        has_xy = "X Position" in pt.columns and "Y Position" in pt.columns
        xs = pt["X Position"].to_list() if has_xy else [None] * len(ids)
        ys = pt["Y Position"].to_list() if has_xy else [None] * len(ids)
        return list(zip(ids, xs, ys)), None

    def _repopulate_part_filter(self) -> None:
        """Rebuild the checklist, keeping prior exclusions for IDs that remain."""
        if not hasattr(self, "_part_list"):
            return

        previously_excluded = self._current_excluded_parts()
        method = self._method_combo.currentText()
        rows, err = self._read_parts_for_filter(method)

        self._part_list.blockSignals(True)
        self._part_list.clear()
        for pid, x, y in rows:
            if x is not None and y is not None:
                label = f"{pid}    @ ({x:.1f}, {y:.1f})"
            else:
                label = pid
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, pid)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked = pid not in previously_excluded
            item.setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
            self._part_list.addItem(item)
        self._part_list.blockSignals(False)

        self._pf_error = err
        self._update_part_filter_status()

    def _current_excluded_parts(self) -> set[str]:
        """The set of part IDs currently unticked in the checklist."""
        out: set[str] = set()
        if not hasattr(self, "_part_list"):
            return out
        for i in range(self._part_list.count()):
            item = cast(QListWidgetItem, self._part_list.item(i))
            if item.checkState() != Qt.CheckState.Checked:
                out.add(str(item.data(Qt.ItemDataRole.UserRole)))
        return out

    def _set_all_parts_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self._part_list.blockSignals(True)
        for i in range(self._part_list.count()):
            cast(QListWidgetItem, self._part_list.item(i)).setCheckState(state)
        self._part_list.blockSignals(False)
        self._update_part_filter_status()

    def _apply_part_exclude(self, excluded) -> None:
        """Tick/untick items to match a saved exclude list."""
        wanted = {str(p) for p in (excluded or [])}
        self._part_list.blockSignals(True)
        for i in range(self._part_list.count()):
            item = cast(QListWidgetItem, self._part_list.item(i))
            pid = str(item.data(Qt.ItemDataRole.UserRole))
            item.setCheckState(
                Qt.CheckState.Unchecked if pid in wanted else Qt.CheckState.Checked
            )
        self._part_list.blockSignals(False)
        self._update_part_filter_status()

    def _on_part_item_changed(self, _item) -> None:
        self._update_part_filter_status()

    def _update_part_filter_status(self) -> None:
        n = self._part_list.count()
        if n == 0:
            self._pf_status.setText(
                getattr(self, "_pf_error", None) or "No parts loaded."
            )
            self._part_filter_section.set_title("Part filter")
            return
        excluded = len(self._current_excluded_parts())
        kept = n - excluded
        if excluded == 0:
            self._pf_status.setText(f"All {n} parts included.")
            self._part_filter_section.set_title("Part filter")
        else:
            self._pf_status.setText(
                f"{kept} of {n} parts included \u2014 {excluded} excluded."
            )
            self._part_filter_section.set_title(
                f"Part filter \u2014 {excluded} excluded"
            )

    def _make_float_row(self, parent_layout, label, default):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        edit = QLineEdit(default)
        row.addWidget(edit, stretch=1)
        parent_layout.addLayout(row)
        return edit

    def _make_int_row(self, parent_layout, label, default, min_val, max_val):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        row.addWidget(spin, stretch=1)
        parent_layout.addLayout(row)
        return spin

    def _sync_method_widgets(self, method: str) -> None:
        self._cluster_widget.setVisible(method == "dbscan")
        self._dist_widget.setVisible(method == "direct")
        self._dhxml_hint.setVisible(method == "dhxml")
        self._repopulate_part_filter()

    def _on_method_changed(self, method):
        self._sync_method_widgets(method)
        self._refresh_options()

    def _on_correction_machine_changed(self, machine):
        current = self._correction_column_combo.currentText()
        self._correction_column_combo.clear()
        columns = _correction_columns(machine)
        self._correction_column_combo.addItems(columns)
        if current in columns:
            self._correction_column_combo.setCurrentText(current)

    def _on_all_layers_toggled(self, all_layers: bool) -> None:
        self._layer_range_widget.setEnabled(not all_layers)

    def _set_layers_unavailable(self) -> None:
        self._available_layers = None
        self._layer_avail_label.setText("Available: \u2014")

    def _probe_layers(self) -> None:
        """Cheaply read the available layer numbers from the source directory.

        DataStore construction only scans filenames (no Parquet build), so this
        is safe to run synchronously whenever the source path changes.
        """
        src = self._source_edit.text().strip()
        if not src:
            self._set_layers_unavailable()
            return
        try:
            from ampm import DataStore

            store = DataStore(src)  # layer_thickness irrelevant for layer discovery
            layers = store.layers
        except Exception:
            self._set_layers_unavailable()
            return

        lo, hi = min(layers), max(layers)
        self._available_layers = (lo, hi)
        for spin in (self._layer_from_spin, self._layer_to_spin):
            spin.setRange(lo, hi)
        self._layer_from_spin.setValue(lo)
        self._layer_to_spin.setValue(hi)
        self._layer_avail_label.setText(
            f"Available: {lo}\u2013{hi} ({len(layers)} layers)"
        )

    def _on_progress(self, pct: int, label: str) -> None:
        self._load_progress.setValue(pct)
        self._load_progress.setFormat(f"{label} \u2014 %p%")

    def _refresh_options(self) -> None:
        """Enable only the options the available files can support.

        STL -> masking; Parts CSV -> direct/dbscan; BuildStarted DHXML -> dhxml.
        Runs live off the path fields, so Browsing a file unlocks its options.
        The Load button is not gated here; validation happens on press.
        """
        if getattr(self, "_in_refresh_options", False):
            return

        self._in_refresh_options = True

        try:
            has_stl = bool(self._stl_edit.text().strip())
            has_csv = bool(self._csv_edit.text().strip())
            has_dhxml = bool(self._dhxml_edit.text().strip())

            self._mask_check.setEnabled(has_stl)

            if getattr(self, "_prev_has_stl", None) != has_stl:
                self._mask_check.setChecked(has_stl)

            self._prev_has_stl = has_stl

            self._mask_check.setToolTip(
                "Filter rows to the part region using the STL geometry"
                if has_stl
                else "No STL set"
            )

            avail = {"direct": has_csv, "dbscan": has_csv, "dhxml": has_dhxml}
            model = self._method_combo.model()

            for i in range(self._method_combo.count()):
                method = self._method_combo.itemText(i)
                ok = avail.get(method, True)
                item = model.item(i)
                item.setEnabled(ok)
                item.setToolTip(
                    ""
                    if ok
                    else (
                        "Needs a Parts CSV"
                        if method in ("direct", "dbscan")
                        else "Needs a BuildStarted DHXML"
                    )
                )

            any_method = any(avail.values())
            if any_method and not avail.get(self._method_combo.currentText(), False):
                for i in range(self._method_combo.count()):
                    if avail.get(self._method_combo.itemText(i), False):
                        self._method_combo.blockSignals(True)
                        self._method_combo.setCurrentIndex(i)
                        self._method_combo.blockSignals(False)
                        self._sync_method_widgets(self._method_combo.currentText())
                        break

            self._assign_check.setEnabled(any_method)

            if getattr(self, "_prev_any_method", None) != any_method:
                self._assign_check.setChecked(any_method)

            self._prev_any_method = any_method

            self._assign_check.setToolTip(
                "Assign each row to its nearest part ID"
                if any_method
                else "No Parts CSV or BuildStarted DHXML set"
            )
        finally:
            self._in_refresh_options = False

    def _validate_inputs(self) -> list[str]:
        """Check inputs before loading. Returns a list of human-readable problems."""
        problems: list[str] = []

        if self._config is None:
            problems.append(
                "No project loaded \u2014 choose a project root first "
                "(its config failed to load)."
            )
            return problems

        source = self._source_edit.text().strip()
        if not source:
            problems.append("Source directory is not set.")
        elif not Path(source).is_dir():
            problems.append(f"Source directory does not exist: {source}")

        lt = self._lt_edit.text().strip()
        try:
            if float(lt) <= 0:
                problems.append("Layer thickness must be greater than 0.")
        except ValueError:
            problems.append(f"Layer thickness is not a number: {lt or '(empty)'}")

        if self._mask_check.isChecked():
            stl = self._stl_edit.text().strip()
            if not stl:
                problems.append("Apply mask is on but no STL file is set.")
            elif not Path(stl).is_file():
                problems.append(f"STL file does not exist: {stl}")

        if self._assign_check.isChecked():
            method = self._method_combo.currentText()

            if method == "dhxml":
                dhxml = self._dhxml_edit.text().strip()
                if not dhxml:
                    problems.append(
                        "Assign parts is set to 'dhxml' but no BuildStarted "
                        "DHXML is set."
                    )
                elif not Path(dhxml).is_file():
                    problems.append(f"BuildStarted DHXML does not exist: {dhxml}")
                csv = self._csv_edit.text().strip()
                if csv and not Path(csv).is_file():
                    problems.append(f"Parts CSV does not exist: {csv}")
            else:
                csv = self._csv_edit.text().strip()
                if not csv:
                    problems.append("Assign parts is on but no Parts CSV is set.")
                elif not Path(csv).is_file():
                    problems.append(f"Parts CSV does not exist: {csv}")

                if method == "direct":
                    md = self._max_dist_edit.text().strip().lower()
                    if md != "none":
                        try:
                            if float(md) <= 0:
                                problems.append(
                                    "Max distance must be greater than 0 "
                                    "(or 'none')."
                                )
                        except ValueError:
                            problems.append(
                                f"Max distance must be a number or 'none': "
                                f"{self._max_dist_edit.text()!r}"
                            )

                if method == "dbscan":
                    for label, widget in (
                        ("EPS_XY", self._eps_xy_edit),
                        ("EPS_Z", self._eps_z_edit),
                    ):
                        txt = widget.text().strip()
                        try:
                            if float(txt) <= 0:
                                problems.append(f"{label} must be greater than 0.")
                        except ValueError:
                            problems.append(
                                f"{label} is not a number: {txt or '(empty)'}"
                            )

                    ov = self._overlap_edit.text().strip().lower()
                    if ov != "auto":
                        try:
                            if int(ov) < 0:
                                problems.append(
                                    "Overlap layers must be 0 or more (or 'auto')."
                                )
                        except ValueError:
                            problems.append(
                                f"Overlap layers must be an integer or 'auto': "
                                f"{self._overlap_edit.text()!r}"
                            )

        if (
            not self._all_layers_check.isChecked()
            and self._available_layers is not None
        ):
            lo = min(self._layer_from_spin.value(), self._layer_to_spin.value())
            hi = max(self._layer_from_spin.value(), self._layer_to_spin.value())
            alo, ahi = self._available_layers
            if hi < alo or lo > ahi:
                problems.append(
                    f"Selected layer range {lo}\u2013{hi} is outside the "
                    f"available range {alo}\u2013{ahi}."
                )

        return problems

    def _browse_build_dir(self):
        start_dir = ""
        last = self._settings.value("last_project_root", "", type=str)
        if last:
            parent = Path(last).parent
            if parent.is_dir():
                start_dir = str(parent)

        path = QFileDialog.getExistingDirectory(
            self, "Select packet directory", start_dir
        )
        if not path:
            return

        self._project_root = path
        self._settings.setValue("last_project_root", path)
        self._dir_edit.setText(path)
        self._log.clear()
        self._log.append(f"Selected: {path}")

        try:
            from ampm.config import create_or_load_config

            config = create_or_load_config(path)
        except Exception as e:
            self._config = None
            self._project_root = None
            self._log.append(f"ERROR loading config: {e}")
            self._refresh_options()
            return

        self._config = config
        self._populate_config(config)
        self._refresh_views(path)
        self._load_resume_state(path)
        self._refresh_options()
        self._log.append(
            "Config loaded. Review paths and click 'Load Data' when ready."
        )

    def _populate_config(self, config):
        self._source_edit.setText(config["SOURCE"])
        self._stl_edit.setText(config["STL"])
        self._csv_edit.setText(config["PARTS_CSV"])
        dhxml = config.get("DHXML") or self._find_dhxml(config.get("SOURCE", ""))
        self._dhxml_edit.setText(dhxml or "")
        self._lt_edit.setText(str(config["LAYER_THICKNESS"]))

        self._method_combo.setCurrentText(config["METHOD"])
        self._on_method_changed(self._method_combo.currentText())
        max_dist = config["MAX_DISTANCE_MM"]
        self._max_dist_edit.setText("none" if max_dist is None else str(max_dist))

        self._eps_xy_edit.setText(str(config["EPS_XY"]))
        self._eps_z_edit.setText(str(config["EPS_Z"]))
        self._min_samples_spin.setValue(config["MIN_SAMPLES"])
        self._chunk_spin.setValue(config["LAYERS_PER_CHUNK"])
        overlap = config["OVERLAP_LAYERS"]
        self._overlap_edit.setText("auto" if overlap is None else str(overlap))

    ## UI STATE
    def _load_resume_state(self, project_root) -> None:
        """Read the sidecar, overlay pipeline params now, stash analysis state."""
        self._pending_resume = None
        state = load_ui_state(project_root)
        if not state:
            return

        overrides = state.get("config_overrides")
        if isinstance(overrides, dict):
            self._apply_config_overrides(overrides)
            if self._config is not None:
                for key in _OVERLAY_KEYS:
                    if key in overrides:
                        self._config[key] = overrides[key]

        self._pending_resume = {
            "derived": state.get("derived_columns") or [],
            "view": state.get("view"),
            "axes": state.get("axes") or {},
            "settings": state.get("settings") or {},
        }
        self._log.append("Found a saved setup; it will be restored after load.")

    def _apply_config_overrides(self, override: dict) -> None:
        """Apply persisted pipeline params to the config-tab widgets."""
        if "LAYER_THICKNESS" in override:
            self._lt_edit.setText(str(override["LAYER_THICKNESS"]))
        if "METHOD" in override:
            self._method_combo.setCurrentText(str(override["METHOD"]))
        if "MAX_DISTANCE_MM" in override:
            md = override["MAX_DISTANCE_MM"]
            self._max_dist_edit.setText("none" if md is None else str(md))
        if "EPS_XY" in override:
            self._eps_xy_edit.setText(str(override["EPS_XY"]))
        if "EPS_Z" in override:
            self._eps_z_edit.setText(str(override["EPS_Z"]))
        if "MIN_SAMPLES" in override:
            try:
                self._min_samples_spin.setValue(int(override["MIN_SAMPLES"]))
            except (TypeError, ValueError):
                pass
        if "LAYERS_PER_CHUNK" in override:
            try:
                self._chunk_spin.setValue(int(override["LAYERS_PER_CHUNK"]))
            except (TypeError, ValueError):
                pass
        if "OVERLAP_LAYERS" in override:
            ov = override["OVERLAP_LAYERS"]
            self._overlap_edit.setText("auto" if ov is None else str(ov))
        if "APPLY_MASK" in override:
            self._mask_check.setChecked(bool(override["APPLY_MASK"]))
        if "ASSIGN_PARTS" in override:
            self._assign_check.setChecked(bool(override["ASSIGN_PARTS"]))
        if "DHXML" in override and override["DHXML"]:
            self._dhxml_edit.setText(str(override["DHXML"]))
        if "PART_EXCLUDE" in override:
            self._apply_part_exclude(override["PART_EXCLUDE"])
        if "CORRECTION_MACHINE" in override:
            machine = override["CORRECTION_MACHINE"]
            if machine in _correction_machines():
                self._correction_machine_combo.setCurrentText(machine)
        if "CORRECTION_COLUMN" in override:
            column = override["CORRECTION_COLUMN"]
            idx = self._correction_column_combo.findText(column)
            if idx >= 0:
                self._correction_column_combo.setCurrentIndex(idx)
        if "APPLY_CORRECTION" in override:
            self._correction_check.setChecked(bool(override["APPLY_CORRECTION"]))
        if "LAYER_RANGE" in override:
            lr = override["LAYER_RANGE"]
            if lr is None:
                self._all_layers_check.setChecked(True)
            elif isinstance(lr, (list, tuple)) and len(lr) == 2:
                self._all_layers_check.setChecked(False)
                try:
                    self._layer_from_spin.setValue(int(lr[0]))
                    self._layer_to_spin.setValue(int(lr[1]))
                except (TypeError, ValueError):
                    pass
        if "LOAD_COLUMNS" in override:
            lc = override["LOAD_COLUMNS"]
            if lc is None:
                self._columns_edit.setText("all")
            elif isinstance(lc, (list, tuple)):
                self._columns_edit.setText(", ".join(str(c) for c in lc))

        def _restore_range(key, lo_edit, hi_edit):
            rng = override.get(key)
            if rng is None:
                lo_edit.setText("")
                hi_edit.setText("")
            elif isinstance(rng, (list, tuple)) and len(rng) == 2:
                lo_edit.setText("" if rng[0] is None else str(rng[0]))
                hi_edit.setText("" if rng[1] is None else str(rng[1]))

        if "X_RANGE" in override:
            _restore_range("X_RANGE", self._x_min_edit, self._x_max_edit)
        if "Y_RANGE" in override:
            _restore_range("Y_RANGE", self._y_min_edit, self._y_max_edit)

    def _current_analysis_state(self) -> dict:
        """Snapshot the live analysis setup (recipes + view/axes/settings)."""
        derived = [
            {"stat": r["stat"], "signal": r["signal"], "mode": r["mode"], "name": name}
            for name, r in self._derived_recipes.items()
        ]
        axes = {k: (c.currentText() or None) for k, c in self._axis_combos.items()}
        settings = {}
        for key, (widget, spec) in self._setting_widgets.items():
            try:
                settings[key] = read_widget(widget, spec)
            except Exception:
                pass
        return {
            "derived": derived,
            "view": self._view_combo.currentText() or None,
            "axes": axes,
            "settings": settings,
        }

    def _gather_ui_state(self) -> dict:
        """Build the full sidecar dict from the current GUI state."""
        try:
            cfg = self._gather_full_config()
            overrides = {}
            for key in _OVERLAY_KEYS:
                if key in cfg:
                    val = cfg[key]
                    overrides[key] = list(val) if isinstance(val, tuple) else val
        except Exception:
            overrides = {
                key: (list(v) if isinstance(v := self._config[key], tuple) else v)
                for key in _OVERLAY_KEYS
                if self._config and key in self._config
            }

        a = self._current_analysis_state()
        return {
            "version": UI_STATE_VERSION,
            "config_overrides": overrides,
            "derived_columns": a["derived"],
            "view": a["view"],
            "axes": a["axes"],
            "settings": a["settings"],
        }

    def _save_ui_state(self) -> None:
        if not self._project_root:
            return
        try:
            save_ui_state(self._project_root, self._gather_ui_state())
        except Exception:
            pass

    def _apply_resume_state(self) -> None:
        """After load: recompute derived recipes, restore view/axes/settings.

        Does not plot. Anything stale (missing signal/view/column) is skipped
        with a log note.
        """
        resume = self._pending_resume
        self._pending_resume = None

        if not resume:
            self._on_view_changed(self._view_combo.currentText())
            return

        for rec in resume.get("derived", []):
            if not isinstance(rec, dict):
                continue
            signal = rec.get("signal")
            if signal:
                self._compute_derived(
                    rec.get("stat", "CoV"), signal, rec.get("mode", "overall")
                )

        view = resume.get("view")
        if view and view in self._views:
            self._view_combo.blockSignals(True)
            self._view_combo.setCurrentText(view)
            self._view_combo.blockSignals(False)
        self._on_view_changed(self._view_combo.currentText())

        for key, val in (resume.get("axes") or {}).items():
            combo = self._axis_combos.get(key)
            if combo is not None and val:
                idx = combo.findText(val)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

        for key, val in (resume.get("settings") or {}).items():
            pair = self._setting_widgets.get(key)
            if pair is not None:
                widget, spec = pair
                try:
                    set_widget_value(widget, spec, val)
                except Exception:
                    pass

        self._log.append("Analysis setup restored. Click Plot when ready.")

    def _gather_full_config(self):
        """Read config from GUI fields, merging with the loaded config."""
        config = dict(self._config) if self._config else {}

        config["SOURCE"] = self._source_edit.text()
        config["STL"] = self._stl_edit.text()
        config["PARTS_CSV"] = self._csv_edit.text()
        config["DHXML"] = self._dhxml_edit.text()
        config["PART_EXCLUDE"] = sorted(self._current_excluded_parts()) or None
        config["LAYER_THICKNESS"] = float(self._lt_edit.text())

        config["METHOD"] = self._method_combo.currentText()
        max_dist_text = self._max_dist_edit.text().strip().lower()
        config["MAX_DISTANCE_MM"] = (
            None if max_dist_text == "none" else float(max_dist_text)
        )

        config["EPS_XY"] = float(self._eps_xy_edit.text())
        config["EPS_Z"] = float(self._eps_z_edit.text())
        config["MIN_SAMPLES"] = self._min_samples_spin.value()
        config["LAYERS_PER_CHUNK"] = self._chunk_spin.value()
        overlap_text = self._overlap_edit.text().strip().lower()
        config["OVERLAP_LAYERS"] = None if overlap_text == "auto" else int(overlap_text)

        config["APPLY_MASK"] = self._mask_check.isChecked()
        config["ASSIGN_PARTS"] = self._assign_check.isChecked()
        config["APPLY_CORRECTION"] = self._correction_check.isChecked()
        config["CORRECTION_MACHINE"] = self._correction_machine_combo.currentText()
        config["CORRECTION_COLUMN"] = self._correction_column_combo.currentText()

        if self._all_layers_check.isChecked():
            config["LAYER_RANGE"] = None
        else:
            lo = self._layer_from_spin.value()
            hi = self._layer_to_spin.value()
            config["LAYER_RANGE"] = (min(lo, hi), max(lo, hi))

        cols_text = self._columns_edit.text().strip()
        if not cols_text or cols_text.lower() == "all":
            config["LOAD_COLUMNS"] = None
        else:
            config["LOAD_COLUMNS"] = [
                col.strip() for col in cols_text.split(",") if col.strip()
            ]

        def _gather_range(lo_edit, hi_edit, label):
            lo_txt = lo_edit.text().strip()
            hi_txt = hi_edit.text().strip()
            if not lo_txt and not hi_txt:
                return None
            lo = float(lo_txt) if lo_txt else float("-inf")
            hi = float(hi_txt) if hi_txt else float("inf")
            if lo > hi:
                raise ValueError(
                    f"{label} range minimum ({lo}) exceeds maximum ({hi})."
                )
            return (lo, hi)

        config["X_RANGE"] = _gather_range(self._x_min_edit, self._x_max_edit, "X")
        config["Y_RANGE"] = _gather_range(self._y_min_edit, self._y_max_edit, "Y")

        source = config["SOURCE"]
        cache_dir = Path(source) / ".cache"
        if config["LAYER_RANGE"] is None:
            config["MASK_CACHE"] = str(cache_dir / "fullplate_mask.pkl")
            config["MASK_KEEP_CACHE"] = str(cache_dir / "mask_keep.pq")
            config["CLUSTER_CACHE"] = str(cache_dir / "cluster_labels.pq")
        else:
            tag = _layer_range_tag(config["LAYER_RANGE"])
            config["MASK_CACHE"] = str(cache_dir / f"mask_geom_{tag}.pkl")
            config["MASK_KEEP_CACHE"] = str(cache_dir / f"mask_keep_{tag}.pq")
            config["CLUSTER_CACHE"] = str(cache_dir / f"cluster_labels_{tag}.pq")

        return config

    def _load_data(self):
        problems = self._validate_inputs()
        if problems:
            self._log.append("")
            self._log.append("Cannot load \u2014 please fix the following:")
            for p in problems:
                self._log.append(f"  \u2022 {p}")
            return

        config = self._gather_full_config()
        self._config = config

        self._load_btn.setEnabled(False)
        self._load_progress.setValue(0)
        self._load_progress.setFormat("Starting \u2014 %p%")
        self._load_progress.setVisible(True)

        idx = self._tabs.indexOf(self._analysis_tab)
        if idx != -1:
            self._tabs.removeTab(idx)

        # On a reload, preserve current Analysis
        # On a first load, pending resume comes from sidecar
        if self._df is not None:
            self._pending_resume = self._current_analysis_state()

        self._df = None
        self._derived.clear()
        self._derived_recipes.clear()
        self._derived_list.clear()

        self._log.append("")
        self._log.append("Loading data...")
        self._log.append("")

        self._load_worker = LoadWorker(config)
        self._load_worker.log.connect(self._on_log)
        self._load_worker.progress.connect(self._on_progress)
        self._load_worker.finished_ok.connect(self._on_load_ok)
        self._load_worker.finished_err.connect(self._on_load_err)
        self._load_worker.start()

    def _on_load_ok(self, df, config):
        self._df = df
        self._config = config
        self._load_progress.setValue(100)
        self._load_progress.setVisible(False)
        self._log.append("")
        self._log.append("Data loaded and ready.")

        self._load_btn.setEnabled(True)

        if self._tabs.indexOf(self._analysis_tab) == -1:  # makes tab appear on load
            self._tabs.addTab(self._analysis_tab, "Analysis")

        self._tabs.setCurrentIndex(self._tabs.indexOf(self._analysis_tab))
        self._refresh_column_combos()
        self._apply_resume_state()
        self._save_ui_state()

        if self._load_worker is not None:
            self._load_worker.wait()
        self._load_worker = None

    def _on_load_err(self, tb):
        self._load_progress.setVisible(False)
        self._log.append("")
        self._log.append("ERROR: Data load failed.")
        self._log.append(tb)
        self._load_btn.setEnabled(True)
        if self._load_worker is not None:
            self._load_worker.wait()
        self._load_worker = None

    def _all_columns(self):
        cols = []
        if self._df is not None:
            cols = list(self._df.columns)
        for name in self._derived:
            if name not in cols:
                cols.append(name)
        return cols

    def _refresh_column_combos(self):
        cols = self._all_columns()
        for _, combo in self._axis_combos.items():
            current = combo.currentText()
            combo.clear()
            combo.addItem("")
            combo.addItems(cols)
            if current in cols:
                combo.setCurrentText(current)

        if self._df is not None:
            numeric_cols = [
                col
                for col, dt in zip(self._df.columns, self._df.dtypes)
                if dt.is_numeric()
            ]
            current_sig = self._derived_signal.currentText()
            self._derived_signal.clear()
            self._derived_signal.addItems(numeric_cols)
            if current_sig in numeric_cols:
                self._derived_signal.setCurrentText(current_sig)

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.layout():
                self._clear_layout(item.layout())
            if item.widget():
                item.widget().deleteLater()

    def _on_view_changed(self, view_name):
        if view_name not in self._views:
            return

        module = self._views[view_name]
        self._description_label.setText(getattr(module, "DESCRIPTION", ""))

        self._clear_layout(self._axes_layout)
        self._axis_combos.clear()
        axes = getattr(module, "AXES", {})
        cols = self._all_columns()

        for key, spec in axes.items():
            row = QHBoxLayout()
            row.addWidget(QLabel(spec.get("label", key) + ":"))
            combo = NoScrollComboBox()
            combo.addItem("")
            combo.addItems(cols)
            default = spec.get("default")
            if default and default in cols:
                combo.setCurrentText(default)
            row.addWidget(combo, stretch=1)
            self._axes_layout.addLayout(row)
            self._axis_combos[key] = combo

        self._axes_group.setVisible(bool(axes))

        self._clear_layout(self._settings_layout)
        self._setting_widgets.clear()
        settings = getattr(module, "SETTINGS", {})

        for key, spec in settings.items():
            row = QHBoxLayout()
            row.addWidget(QLabel(spec.get("label", key) + ":"))
            widget = build_widget(spec)

            if self._config and key in self._config:
                set_widget_value(widget, spec, self._config[key])

            row.addWidget(widget, stretch=1)
            self._settings_layout.addLayout(row)
            self._setting_widgets[key] = (widget, spec)

        self._settings_group.setVisible(bool(settings))

    def _compute_derived(self, stat, signal, mode):
        """Compute and register one derived column. Returns its name or None.

        Used by the Add button and by resume-state restoration. Skips (with a
        log note) if the signal isn't present in the loaded data.
        """
        if self._df is None:
            return None
        if not signal:
            self._log.append("Select a signal for the derived column.")
            return None

        col_name = f"cov_{mode}_{signal}"
        if col_name in self._derived:
            self._log.append(f"'{col_name}' already exists.")
            return col_name
        if signal not in self._df.columns:
            self._log.append(
                f"Skipped derived '{col_name}': signal '{signal}' not in this build."
            )
            return None

        self._log.append(f"Computing {col_name}...")
        try:
            from ampm.stats import CovMode, compute_cov

            cov = compute_cov(
                self._df,
                [signal],
                group_by="part_id",
                mode=cast(CovMode, mode),
                noise_label="noise",
            )
            raw_col = f"cov_{signal}"
            derived_df = cov.select(["part_id", raw_col]).rename({raw_col: col_name})

            self._derived[col_name] = derived_df
            self._derived_recipes[col_name] = {
                "stat": stat,
                "signal": signal,
                "mode": mode,
            }
            self._derived_list.addItem(col_name)
            self._refresh_column_combos()
            self._log.append(f"Added '{col_name}' ({derived_df.height} rows).")
            return col_name
        except Exception as e:
            self._log.append(f"ERROR computing derived column: {e}")
            return None

    def _add_derived(self):
        if self._df is None:
            return
        stat = self._derived_stat.currentText()  # for future stats
        signal = self._derived_signal.currentText()
        mode = self._derived_mode.currentText()
        if self._compute_derived(stat, signal, mode):
            self._save_ui_state()

    def _remove_derived(self):
        current = self._derived_list.currentItem()
        if current is None:
            return
        name = current.text()
        self._derived.pop(name, None)
        self._derived_recipes.pop(name, None)
        self._derived_list.takeItem(self._derived_list.row(current))
        self._refresh_column_combos()
        self._log.append(f"Removed '{name}'.")
        self._save_ui_state()

    def _all_axes_group_level(self, axes):
        if self._df is None:
            return False

        if not any(col for col in axes.values() if col is not None):
            return False

        group_cols = {"part_id", "Hatches Power", "Hatch Speed"}
        group_cols.update(self._derived.keys())

        for col in axes.values():
            if col is None:
                continue
            if col not in group_cols:
                return False
        return True

    def _build_group_df(self, needed_columns):
        import polars as pl

        if self._df is None:
            raise ValueError("No data loaded.")

        base = (
            self._df.filter(pl.col("part_id") != "noise")
            .select(["part_id", "Hatches Power", "Hatch Speed"])
            .unique(subset=["part_id"])
            .sort("part_id")
        )

        for col in needed_columns:
            if col in self._derived:
                base = base.join(self._derived[col], on="part_id", how="left")

        return base

    def _prepare_row_df(self, needed_columns):
        if self._df is None:
            raise ValueError("No data loaded.")
        df = self._df
        for col in needed_columns:
            if col in self._derived and col not in df.columns:
                df = df.join(self._derived[col], on="part_id", how="left")
        return df

    def _run_plot(self):
        view_name = self._view_combo.currentText()
        if view_name not in self._views or self._df is None:
            return

        module = self._views[view_name]

        axes = {}
        for key, combo in self._axis_combos.items():
            text = combo.currentText()
            axes[key] = text if text else None

        settings = {}
        for key, (widget, spec) in self._setting_widgets.items():
            settings[key] = read_widget(widget, spec)

        merged_config = dict(self._config) if self._config else {}
        merged_config.update(settings)

        derived_needed = [col for col in axes.values() if col and col in self._derived]

        if self._all_axes_group_level(axes):
            df = self._build_group_df(derived_needed)
            self._log.append("")
            self._log.append(f"Plotting: {view_name} (group-level, {df.height} rows)")
        else:
            df = self._prepare_row_df(derived_needed)
            self._log.append("")
            self._log.append(f"Plotting: {view_name} (row-level, {df.height:,} rows)")

        self._plot_btn.setEnabled(False)
        self._plot_progress.setRange(0, 0)  # indeterminate / busy
        self._plot_progress.setVisible(True)
        self._plot_worker = PlotWorker(module, df, merged_config, axes, settings)
        self._plot_worker.log.connect(self._on_log)
        self._plot_worker.finished_ok.connect(self._on_plot_ok)
        self._plot_worker.finished_err.connect(self._on_plot_err)
        self._plot_worker.start()

    def _on_plot_ok(self):
        self._plot_progress.setVisible(False)
        self._log.append("Plot complete.")
        self._plot_btn.setEnabled(True)
        self._save_ui_state()
        if self._plot_worker is not None:
            self._plot_worker.wait()
        self._plot_worker = None

    def _on_plot_err(self, tb):
        self._plot_progress.setVisible(False)
        self._log.append("ERROR: Plot failed.")
        self._log.append(tb)
        self._plot_btn.setEnabled(True)
        if self._plot_worker is not None:
            self._plot_worker.wait()
        self._plot_worker = None

    def _on_log(self, msg):
        self._log.append(msg)

    def _load_views(self):
        """Lazily discover views and populate the combo box."""
        if self._views_loaded:
            return
        from ampm.views import ensure_user_views_dir

        ensure_user_views_dir()
        self._refresh_views(None)

    def _on_reload_views(self):
        """Re-scan view folders on demand (built-in + external)."""
        before = set(self._views)
        self._refresh_views(self._project_root)
        after = set(self._views)
        added = sorted(after - before)
        removed = sorted(before - after)
        if added:
            self._log.append(f"Views reloaded. Added: {', '.join(added)}.")
        elif removed:
            self._log.append(f"Views reloaded. Removed: {', '.join(removed)}.")
        else:
            self._log.append(f"Views reloaded ({len(after)} available).")

    def _refresh_views(self, project_root=None):
        """(Re)discover views, including any external/per-build folders, and
        repopulate the combo while preserving the current selection."""
        from ampm.views import discover

        current = self._view_combo.currentText()
        self._views = discover(project_root=project_root, log=self._on_log)
        self._view_combo.blockSignals(True)
        self._view_combo.clear()
        self._view_combo.addItems(list(self._views.keys()))
        if current in self._views:
            self._view_combo.setCurrentText(current)
        self._view_combo.blockSignals(False)
        self._views_loaded = True


def main():
    import os
    import signal

    from PyQt6.QtCore import QTimer

    app = QApplication(sys.argv)
    app.setOrganizationName("AMPM")
    app.setApplicationName("AMPM Analyzer")
    window = MainWindow()
    window.show()
    QTimer.singleShot(0, window._load_views)

    sigint_fired = False

    def _handle_sigint(*_):
        nonlocal sigint_fired
        if sigint_fired:
            os._exit(130)  # second Ctrl+C: force quit immediately
        sigint_fired = True
        print("\nInterrupt received \u2014 closing (Ctrl+C again to force quit)...")
        window.close()  # saves UI state, waits for threads

    signal.signal(signal.SIGINT, _handle_sigint)

    _sigint_timer = QTimer()
    _sigint_timer.timeout.connect(lambda: None)
    _sigint_timer.start(200)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
