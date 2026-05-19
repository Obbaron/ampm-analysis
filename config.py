"""
Shared configuration for ampm analysis - user-editable values in config.toml
"""

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # back-port name used by some packages
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # for Python < 3.11
        except ModuleNotFoundError:
            sys.exit(
                "Python < 3.11 requires the 'tomli' package.\n"
                "Install: pip install tomli"
            )


_CONFIG_DIR = Path(__file__).resolve().parent

try:
    with open(_CONFIG_DIR / "config.toml", "rb") as file:
        _config = tomllib.load(file)
except FileNotFoundError:
    try:
        with open(_CONFIG_DIR.parent / "config.toml", "rb") as file:
            _config = tomllib.load(file)
    except FileNotFoundError:
        sys.exit("ERROR: config.toml not found.\n")
    except tomllib.TOMLDecodeError as e:
        sys.exit(f"ERROR: config.toml has invalid syntax:\n{e}")
except tomllib.TOMLDecodeError as e:
    sys.exit(f"ERROR: config.toml has invalid syntax:\n{e}")


try:
    SOURCE = _config["paths"]["source"]
    STL = _config["paths"]["stl"]
    PARTS_CSV = _config["paths"]["parts_csv"]
    LAYER_THICKNESS = _config["build"]["layer_thickness"]
except KeyError as e:
    sys.exit(f"ERROR: Missing required key in config.toml: {e}")

MASK_CACHE = str(Path(SOURCE) / ".cache" / "fullplate_mask.pkl")
MASK_KEEP_CACHE = str(Path(SOURCE) / ".cache" / "mask_keep.pq")
CLUSTER_CACHE = str(Path(SOURCE) / ".cache" / "cluster_labels.pq")
