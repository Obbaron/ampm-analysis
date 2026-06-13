"""
example_view.py

A fully-commented EXAMPLE VIEW. Copy this file, rename it, and edit the pieces
marked "EDIT ME" to create your own view.

WHAT IS A VIEW?
---------------
A "view" is a single Python module that describes one plot type. The host app:

    1. Discovers this module and reads the module-level attributes below
       (NAME, DESCRIPTION, AXES, SETTINGS) to build the UI: a title, a set of
       axis pickers (dropdowns of dataset columns), and a set of setting
       controls (number boxes, checkboxes, dropdowns).

    2. When the user clicks "run", the app resolves their choices and calls
       run(df, config, axes, settings) with everything filled in.

So a view is really just: some metadata + one run() function. Everything else
(loading the data, building the UI, wiring up the controls) is handled for you.
"""

# ---------------------------------------------------------------------------
# MODULE METADATA
# ---------------------------------------------------------------------------

# NAME: short label shown in the view picker. Keep it human-friendly.
NAME = "Example View"

# DESCRIPTION: one-line summary shown under the name. Tell colleagues what the
# view is for, not how it works.
DESCRIPTION = "Annotated template showing how a view is structured."


# ---------------------------------------------------------------------------
# AXES  -->  the column pickers the user sees
# ---------------------------------------------------------------------------
#
# Each key here becomes an axis selector in the UI. The user picks a dataset
# COLUMN for each one. By the time run() is called, `axes` is a plain dict
# mapping {key: selected_column_name}.
#
# Per-axis fields:
#   "label"   : text shown next to the dropdown.
#   "default" : the column pre-selected when the view opens. Use None to leave
#               it unset (the user then chooses, and the value may be None in
#               run() if they leave it blank -- see the "optional axis" pattern
#               in run() below).
#
# Choose keys that make sense for your plot: x / y / z / color are common, but
# you can use anything (see cov_summary's signal_1/signal_2, or kde's
# column/group_by). A view with no axes at all is fine too -- just use
# AXES = {} (see k_distance.py).

AXES = {
    # Two required axes. We give them sensible defaults so the view works on
    # first open. EDIT ME: point these at columns that exist in your data.
    "x": {"label": "X axis", "default": "Demand X"},
    "y": {"label": "Y axis", "default": "Demand Y"},
    # An OPTIONAL axis. default=None means "nothing selected". We handle the
    # not-selected case in run() with axes.get("color").
    "color": {"label": "Color (optional)", "default": None},
}


# ---------------------------------------------------------------------------
# SETTINGS  -->  the non-column controls the user sees
# ---------------------------------------------------------------------------
#
# Each key becomes a control in the UI. By the time run() is called, `settings`
# is a plain dict mapping {key: value}, already converted to the right type.
#
# Supported "type" values and their extra fields:
#
#   "int"    : whole number. Optional "min"/"max" clamp the input.
#   "float"  : decimal number. Optional "min"/"max".
#   "bool"   : checkbox. Value is True/False.
#   "choice" : dropdown. Requires "options": a list of allowed strings; the
#              value is one of those strings.
#
# Every setting also takes:
#   "default" : the starting value.
#   "label"   : text shown next to the control.
#
# A view with no settings is fine -- use SETTINGS = {} (see cov_summary.py).

SETTINGS = {
    # int with bounds: the UI won't let the user go outside min..max.
    "SAMPLE_SIZE": {
        "type": "int",
        "default": 100_000,
        "min": 1_000,
        "max": 1_000_000,
        "label": "Sample size",
    },
    # float with bounds.
    "POINT_SIZE": {
        "type": "float",
        "default": 4.0,
        "min": 1.0,
        "max": 20.0,
        "label": "Point size",
    },
    # bool -> renders as a checkbox.
    "EQUAL_ASPECT": {
        "type": "bool",
        "default": True,
        "label": "Equal aspect ratio",
    },
    # choice -> renders as a dropdown limited to "options".
    "SORT_BY": {
        "type": "choice",
        "options": ["none", "x", "y"],
        "default": "none",
        "label": "Sort by",
    },
}


# ---------------------------------------------------------------------------
# run()  -->  the entry point the app calls
# ---------------------------------------------------------------------------
#
# Signature is always exactly: run(df, config, axes, settings)
#
#   df       : a Polars DataFrame holding the loaded dataset. Use the Polars
#              API (pl.col(...), df.filter(...), df.group_by(...), df.height,
#              df.sample(...), etc.).
#   config   : dict. The loaded packet/pipeline configuration (the same config
#              the data-loading pipeline used) with THIS view's `settings`
#              merged on top. So it contains file paths (SOURCE, STL,
#              PARTS_CSV) and pipeline parameters (LAYER_THICKNESS, METHOD,
#              MAX_DISTANCE_MM, EPS_XY, EPS_Z, MIN_SAMPLES, LAYERS_PER_CHUNK,
#              OVERLAP_LAYERS, APPLY_MASK, ASSIGN_PARTS, LAYER_RANGE,
#              LOAD_COLUMNS, X_RANGE, Y_RANGE, ...), PLUS every key from your
#              SETTINGS (a setting overrides a config key of the same name).
#              Use this when your view wants a PIPELINE value it didn't declare
#              as a setting e.g. config.get("EPS_XY") rather than forcing
#              the user to re-enter it. It's a shallow copy, so reassigning
#              top-level keys is safe. Most views ignore it: anything in
#              SETTINGS is already in `settings` below.
#   axes     : dict {axis_key: column_name}. Required axes are present; optional
#              ones may be None if the user left them blank.
#   settings : dict {setting_key: value}, already typed per SETTINGS above.
#              (Also folded into `config`, so you can read your settings from
#              either. Prefer `settings` for clarity.)
#
# Conventions worth keeping (so all views feel the same):
#   * Do imports INSIDE run(), not at module top. Views are imported just to
#     read their metadata, so keeping heavy imports lazy keeps startup fast.
#   * Read settings defensively with settings.get(KEY, fallback). The fallback
#     should match the "default" you declared above.
#   * print() short progress messages -- the app redirects print() to its log
#     panel while your view runs, so this is how you report progress.
#   * Build a Plotly figure via the ampm.* helpers, then call .show() on it.
#   * Finish with print("Done.").


def run(df, config, axes, settings):
    # Lazy imports (see note above). Helpers return Plotly figures.
    from ampm.plotting import scatter2d
    from ampm.sampling import prepare_for_plot

    # 1. Read settings, with fallbacks matching the SETTINGS defaults
    sample_size = settings.get("SAMPLE_SIZE", 100_000)
    point_size = settings.get("POINT_SIZE", 4.0)
    equal_aspect = settings.get("EQUAL_ASPECT", True)
    sort_by = settings.get("SORT_BY", "none")

    # A common "choice" idiom: a sentinel string that means "do nothing".
    # Translate it to None for the downstream code (mirrors bar.py).
    if sort_by == "none":
        sort_by = None

    # 2. Read axes. Required vs optional.
    x = axes["x"]  # required: indexing asserts it's there.
    y = axes["y"]  # required.
    color = axes.get("color")  # optional: may be None if left unset.

    #  (optional) Read a PIPELINE value from `config`. Use .get() with a
    #  fallback so the view still works if the key isn't present. This is the
    #  main reason to touch `config` at all: grabbing a pipeline parameter
    #  you didn't declare as a setting, instead of making the user re-enter
    #  it. (We only print it here to show the access pattern.)
    eps_xy = config.get("EPS_XY")
    if eps_xy is not None:
        print(f"(pipeline EPS_XY = {eps_xy} mm)")

    # 3. Validate. Raise ValueError with a clear message on bad input;
    #    the app surfaces it to the user (mirrors single_layer.py).
    if not x or not y:
        raise ValueError("Both X and Y axes must be set.")

    # 4. Do the work. Here: downsample big data before plotting so the
    #    browser stays responsive (mirrors scatter_2d.py).
    print(f"Sampling up to {sample_size:,} points...")
    sample = prepare_for_plot(
        df,
        target_points=sample_size,
        method="random",
        seed=0,  # fixed seed -> reproducible plots.
    )

    # Example of using `sort_by` and the Polars API. Skip if not requested.
    if sort_by is not None and sort_by in sample.columns:
        sample = sample.sort(sort_by)

    # 5. Build the figure and show it. The ampm plotting helpers take the
    #    column names (strings) you resolved from `axes`.
    print("Rendering scatter...")
    scatter2d(
        sample,
        x=x,
        y=y,
        color=color,  # None is acceptable
        size=point_size,
        equal_aspect=equal_aspect,
        colorscale="Turbo",
        title=f"Example View — color: {color or 'none'}",
        xaxis_title=x,
        yaxis_title=y,
        colorbar_title=color or "",
    ).show()

    print("Done.")
