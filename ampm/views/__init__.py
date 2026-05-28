"""
ampm.views

Each .py file in this package that defines NAME, AXES, SETTINGS, and run()
is automatically discovered and made available to the app.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_KNOWN_VIEWS = [  # so the compiler can discover them
    "bar",
    "contour",
    "cov_summary",
    "k_distance",
    "kde",
    "layer_viewer",
    "scatter_2d",
    "scatter_3d",
]


def discover() -> dict[str, object]:
    """
    Scan this package for view modules and return {NAME: module}.

    A valid view module must define:
        NAME: str           # display name for the GUI
        DESCRIPTION: str    # tooltip / help text
        AXES: dict          # column picker definitions
        SETTINGS: dict      # extra widget definitions
        run(df, config, axes, settings): None
    """
    views = {}

    if getattr(sys, "frozen", False):
        stems = _KNOWN_VIEWS
    else:
        package_dir = Path(__file__).parent
        stems = [
            path.stem
            for path in sorted(package_dir.glob("*.py"))
            if not path.name.startswith("_")
        ]

    for stem in stems:
        module = importlib.import_module(f"ampm.views.{stem}")

        required = ("NAME", "AXES", "SETTINGS", "run")
        if not all(hasattr(module, attr) for attr in required):
            continue

        views[module.NAME] = module

    return views
