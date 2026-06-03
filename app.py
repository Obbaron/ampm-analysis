"""
app.py - GUI for AMPM analysis

Select a packet directory, load/mask/assign data, add derived columns,
pick a view (plot type), configure axes and settings, and plot.
"""

import builtins
import re
import sys
import traceback
from pathlib import Path
from typing import cast

from PyQt6.QtCore import Qt, QThread, pyqtSignal
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


def build_widget(spec: dict) -> QWidget:
    """Create the appropriate widget for a setting spec."""
    wtype = spec["type"]

    if wtype == "float":
        widget = QDoubleSpinBox()
        widget.setDecimals(4)
        widget.setRange(spec.get("min", 0.0), spec.get("max", 999999.0))
        widget.setValue(spec.get("default", 0.0))
        return widget

    if wtype == "int":
        widget = QSpinBox()
        widget.setRange(spec.get("min", 0), spec.get("max", 999999))
        widget.setValue(spec.get("default", 0))
        return widget

    if wtype == "choice":
        widget = QComboBox()
        widget.addItems(spec.get("options", []))
        default = spec.get("default")
        if default is not None:
            widget.setCurrentText(str(default))
        return widget

    if wtype == "bool":
        widget = QCheckBox()
        widget.setChecked(spec.get("default", False))
        return widget

    if wtype in ("float_or_none", "int_or_auto"):
        widget = QLineEdit()
        default = spec.get("default")
        if wtype == "float_or_none":
            widget.setText("none" if default is None else str(default))
        else:
            widget.setText("auto" if default is None else str(default))
        if spec.get("tooltip"):
            widget.setToolTip(spec["tooltip"])
        return widget

    widget = QLineEdit()
    widget.setText(str(spec.get("default", "")))
    return widget


def read_widget(widget, spec):
    """Read the current value from a widget."""
    wtype = spec["type"]
    if wtype == "float":
        return widget.value()
    if wtype == "int":
        return widget.value()
    if wtype == "choice":
        return widget.currentText()
    if wtype == "bool":
        return widget.isChecked()
    if wtype == "float_or_none":
        text = widget.text().strip().lower()
        return None if text == "none" else float(text)
    if wtype == "int_or_auto":
        text = widget.text().strip().lower()
        return None if text == "auto" else int(text)
    return widget.text()


def set_widget_value(widget, spec, value):
    """Set a widget's value from a config dict entry."""
    wtype = spec["type"]
    if wtype == "float":
        widget.setValue(float(value))
    elif wtype == "int":
        widget.setValue(int(value))
    elif wtype == "choice":
        widget.setCurrentText(str(value))
    elif wtype == "bool":
        widget.setChecked(bool(value))
    elif wtype == "float_or_none":
        widget.setText("none" if value is None else str(value))
    elif wtype == "int_or_auto":
        widget.setText("auto" if value is None else str(value))
    else:
        widget.setText(str(value))


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
        from ampm.masking import apply_mask, build_mask
        from ampm.parts import (
            QuantAMParts,
            apply_part_id_map,
            assign_nearest_part,
            compute_part_id_map,
        )

        config = self.config
        SOURCE = config["SOURCE"]
        STL = config["STL"]
        PARTS_CSV = config["PARTS_CSV"]
        LAYER_THICKNESS = config["LAYER_THICKNESS"]
        MASK_CACHE = config["MASK_CACHE"]
        MASK_KEEP_CACHE = config["MASK_KEEP_CACHE"]
        CLUSTER_CACHE = config["CLUSTER_CACHE"]
        METHOD = config["METHOD"]
        MAX_DISTANCE_MM = config["MAX_DISTANCE_MM"]
        APPLY_MASK = config.get("APPLY_MASK", True)
        ASSIGN_PARTS = config.get("ASSIGN_PARTS", True)
        LAYER_RANGE = config.get("LAYER_RANGE")  # None (all) or (lo, hi) inclusive

        # Make bar reach 100% regardless of enabled steps
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
        df = store.query(layers=layers_arg)

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
            f"({layer_span[0]}–{layer_span[1]})."
        )
        self._phase = None
        self._emit_progress(self._load_band[1], "Loaded")

        if APPLY_MASK:
            if mask_band is not None:
                self._emit_progress(mask_band[0], "Applying mask")
            mask_params = {
                "layers": layer_span,
                "stl": str(STL),
                "buffer_mm": 0.0,
                "layer_thickness": LAYER_THICKNESS,
            }

            def masking_wrapper(d):
                mask = build_mask(
                    STL,
                    layers=queried_layers,
                    layer_thickness=LAYER_THICKNESS,
                    buffer_mm=0.0,
                    cache_path=MASK_CACHE,
                )
                return apply_mask(d, mask)

            print("Applying mask...")
            df = mask_or_load(
                df,
                cache_path=MASK_KEEP_CACHE,
                mask_fn=masking_wrapper,
                params=mask_params,
                strict=True,
            )
            print(f"After mask: {df.height:,} rows.")
            if mask_band is not None:
                self._emit_progress(mask_band[1], "Masked")
        else:
            print("Skipping mask.")

        if ASSIGN_PARTS:
            if assign_band is not None:
                self._emit_progress(assign_band[0], "Assigning parts")
            quantam = QuantAMParts.from_path(PARTS_CSV)
            parts_table = quantam.parent_parts()
            print(f"Loaded {parts_table.height} parts.")

            if METHOD == "direct":
                print("Assigning parts (direct)...")
                df = assign_nearest_part(
                    df,
                    parts_table,
                    max_distance_mm=MAX_DISTANCE_MM,
                    noise_label="noise",
                )
            else:
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
                    strict=True,
                )
                mapping = compute_part_id_map(clustered, parts_table)
                df = apply_part_id_map(clustered, mapping, noise_label="noise")

            parts_with_speed = quantam.volume_parameters_with_speed()
            df = df.join(
                parts_with_speed.select(
                    [
                        pl.col("Part ID").alias("part_id"),
                        "Hatches Power",
                        "Hatch Speed",
                    ]
                ),
                on="part_id",
                how="left",
            )
            if assign_band is not None:
                self._emit_progress(assign_band[1], "Assigned")
        else:
            print("Skipping part assignment.")

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
        self._views = {}
        self._views_loaded = False
        self._axis_combos = {}
        self._setting_widgets = {}
        self._available_layers = None

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

        self._source_edit.textChanged.connect(self._probe_layers)

        config_layout.addWidget(sources_group)

        # Settings
        self._assign_group = QWidget()
        assign_layout = QVBoxLayout(self._assign_group)
        assign_layout.setContentsMargins(0, 0, 0, 0)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(["direct", "dbscan"])
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        method_row.addWidget(self._method_combo, stretch=1)
        assign_layout.addLayout(method_row)

        dist_row = QHBoxLayout()
        dist_row.addWidget(QLabel("Max distance (mm):"))
        self._max_dist_edit = QLineEdit("none")
        self._max_dist_edit.setToolTip(
            "'none' assigns every row regardless of distance"
        )
        dist_row.addWidget(self._max_dist_edit, stretch=1)
        assign_layout.addLayout(dist_row)

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

        config_layout.addWidget(settings_group)

        # Load button
        self._load_btn = QPushButton("Load Data")
        self._load_btn.setEnabled(False)
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
        self._derived_stat = QComboBox()
        self._derived_stat.addItems(["CoV"])
        add_row.addWidget(self._derived_stat)

        add_row.addWidget(QLabel("Signal:"))
        self._derived_signal = QComboBox()
        add_row.addWidget(self._derived_signal, stretch=1)

        add_row.addWidget(QLabel("Mode:"))
        self._derived_mode = QComboBox()
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
        self._view_combo = QComboBox()
        self._view_combo.addItems(list(self._views.keys()))
        self._view_combo.currentTextChanged.connect(self._on_view_changed)
        selector_row.addWidget(self._view_combo, stretch=1)
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
        view_layout.addWidget(self._plot_btn)

        self._plot_progress = QProgressBar()
        self._plot_progress.setTextVisible(False)
        self._plot_progress.setVisible(False)
        view_layout.addWidget(self._plot_progress)

        analysis_layout.addWidget(view_group)
        analysis_layout.addStretch()

        self._analysis_tab = analysis_scroll

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        log_layout.addWidget(self._log)
        splitter.addWidget(log_group)

        splitter.setSizes([600, 400])  # left, 60%; right, 40%

        if self._views:
            self._on_view_changed(self._view_combo.currentText())

    def closeEvent(self, a0):
        """Wait for any running threads before closing."""
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

    def _on_method_changed(self, method):
        self._cluster_widget.setVisible(method == "dbscan")

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

    def _browse_build_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select packet directory")
        if not path:
            return

        self._dir_edit.setText(path)
        self._log.clear()
        self._log.append(f"Selected: {path}")

        try:
            from ampm.config import create_or_load_config

            config = create_or_load_config(path)
        except Exception as e:
            self._log.append(f"ERROR loading config: {e}")
            self._load_btn.setEnabled(False)
            return

        self._config = config
        self._populate_config(config)
        self._load_btn.setEnabled(True)
        self._log.append(
            "Config loaded. Review paths and click 'Load Data' when ready."
        )

    def _populate_config(self, config):
        self._source_edit.setText(config["SOURCE"])
        self._stl_edit.setText(config["STL"])
        self._csv_edit.setText(config["PARTS_CSV"])
        self._lt_edit.setText(str(config["LAYER_THICKNESS"]))

        self._method_combo.setCurrentText(config["METHOD"])
        max_dist = config["MAX_DISTANCE_MM"]
        self._max_dist_edit.setText("none" if max_dist is None else str(max_dist))

        self._eps_xy_edit.setText(str(config["EPS_XY"]))
        self._eps_z_edit.setText(str(config["EPS_Z"]))
        self._min_samples_spin.setValue(config["MIN_SAMPLES"])
        self._chunk_spin.setValue(config["LAYERS_PER_CHUNK"])
        overlap = config["OVERLAP_LAYERS"]
        self._overlap_edit.setText("auto" if overlap is None else str(overlap))

    def _gather_full_config(self):
        """Read config from GUI fields, merging with the loaded config."""
        config = dict(self._config) if self._config else {}

        config["SOURCE"] = self._source_edit.text()
        config["STL"] = self._stl_edit.text()
        config["PARTS_CSV"] = self._csv_edit.text()
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

        if self._all_layers_check.isChecked():
            config["LAYER_RANGE"] = None
        else:
            lo = self._layer_from_spin.value()
            hi = self._layer_to_spin.value()
            config["LAYER_RANGE"] = (min(lo, hi), max(lo, hi))

        source = config["SOURCE"]
        config["MASK_CACHE"] = str(Path(source) / ".cache" / "fullplate_mask.pkl")
        config["MASK_KEEP_CACHE"] = str(Path(source) / ".cache" / "mask_keep.pq")
        config["CLUSTER_CACHE"] = str(Path(source) / ".cache" / "cluster_labels.pq")

        return config

    def _load_data(self):
        config = self._gather_full_config()
        self._config = config

        self._load_btn.setEnabled(False)
        self._load_progress.setValue(0)
        self._load_progress.setFormat("Starting \u2014 %p%")
        self._load_progress.setVisible(True)

        idx = self._tabs.indexOf(self._analysis_tab)
        if idx != -1:
            self._tabs.removeTab(idx)

        self._df = None
        self._derived.clear()
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
        self._on_view_changed(self._view_combo.currentText())

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
            combo = QComboBox()
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

    def _add_derived(self):
        if self._df is None:
            return

        stat = self._derived_stat.currentText()  # for future stats
        signal = self._derived_signal.currentText()
        mode = self._derived_mode.currentText()

        if not signal:
            self._log.append("Select a signal for the derived column.")
            return

        col_name = f"cov_{mode}_{signal}"

        if col_name in self._derived:
            self._log.append(f"'{col_name}' already exists.")
            return

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
            self._derived_list.addItem(col_name)
            self._refresh_column_combos()
            self._log.append(f"Added '{col_name}' ({derived_df.height} rows).")
        except Exception as e:
            self._log.append(f"ERROR computing derived column: {e}")

    def _remove_derived(self):
        current = self._derived_list.currentItem()
        if current is None:
            return
        name = current.text()
        del self._derived[name]
        self._derived_list.takeItem(self._derived_list.row(current))
        self._refresh_column_combos()
        self._log.append(f"Removed '{name}'.")

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
        from ampm.views import discover

        self._views = discover()
        self._view_combo.addItems(list(self._views.keys()))
        self._views_loaded = True


def main():
    from PyQt6.QtCore import QTimer

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    QTimer.singleShot(0, window._load_views)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
