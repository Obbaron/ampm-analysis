"""
ampm.views

Each .py file in this package that defines NAME, AXES, SETTINGS, and run()
is automatically discovered and made available to the app.
"""

from __future__ import annotations

import importlib
from pathlib import Path


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
    package_dir = Path(__file__).parent

    for path in sorted(package_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue

        module = importlib.import_module(f"ampm.views.{path.stem}")

        required = ("NAME", "AXES", "SETTINGS", "run")
        if not all(hasattr(module, attr) for attr in required):
            continue

        views[module.NAME] = module

    return views
