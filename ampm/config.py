"""
config.py - shared configuration for ampm analysis

Usage:
    from config import load_config
    config = load_config("path/to/project_root")

    Or, to auto-generate config.toml if it doesn't exist:
    from config import create_or_load_config
    config = create_or_load_config("path/to/project_root")

    The project root should contain a `config.toml` file.
    Paths in the TOML can be relative or absolute.
"""

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            sys.exit(
                "Python < 3.11 requires the 'tomli' package.\n"
                "Install: pip install tomli"
            )


def _resolve_path(path_str: str, base_dir: Path) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str(base_dir / path)


def load_config(build_dir: str | Path) -> dict:
    """
    Load configuration from a project root's config.toml

    Parameters
    ----------
    build_dir : str | Path
        Path to the project root containing config.toml

    Returns
    -------
    dict with keys for paths, build parameters, assignment method,
    clustering parameters, signal columns, and derived cache paths.
    """
    build_dir = Path(build_dir).resolve()
    toml_path = build_dir / "config.toml"

    try:
        with open(toml_path, "rb") as file:
            _config = tomllib.load(file)
    except FileNotFoundError:
        sys.exit(f"ERROR: config.toml not found in {build_dir}\n")
    except tomllib.TOMLDecodeError as e:
        sys.exit(f"ERROR: {toml_path} has invalid syntax:\n{e}")

    try:
        source = _resolve_path(_config["paths"]["source"], build_dir)
        stl = _resolve_path(_config["paths"]["stl"], build_dir)
        parts_csv = _resolve_path(_config["paths"]["parts_csv"], build_dir)
        layer_thickness = _config["build"]["layer_thickness"]
    except KeyError as e:
        sys.exit(f"ERROR: Missing required key in {toml_path}: {e}")

    assignment = _config.get("assignment", {})
    method = assignment.get("method", "direct")
    max_distance_mm = assignment.get("max_distance_mm", "none")
    if isinstance(max_distance_mm, str) and max_distance_mm.lower() == "none":
        max_distance_mm = None

    clustering = _config.get("clustering", {})
    eps_xy = clustering.get("eps_xy", 0.3)
    eps_z = clustering.get("eps_z", 0.06)
    min_samples = clustering.get("min_samples", 10)
    layers_per_chunk = clustering.get("layers_per_chunk", 11)
    overlap_layers = clustering.get("overlap_layers", "auto")
    if isinstance(overlap_layers, str) and overlap_layers.lower() == "auto":
        overlap_layers = None

    signals_section = _config.get("signals", {})
    signals = signals_section.get(
        "columns",
        [
            "MeltVIEW melt pool (mean)",
            "Laser output power (mean)",
        ],
    )

    return {
        "SOURCE": source,
        "STL": stl,
        "PARTS_CSV": parts_csv,
        "LAYER_THICKNESS": layer_thickness,
        "MASK_CACHE": str(Path(source) / ".cache" / "fullplate_mask.pkl"),
        "MASK_KEEP_CACHE": str(Path(source) / ".cache" / "mask_keep.pq"),
        "CLUSTER_CACHE": str(Path(source) / ".cache" / "cluster_labels.pq"),
        "METHOD": method,
        "MAX_DISTANCE_MM": max_distance_mm,
        "EPS_XY": eps_xy,
        "EPS_Z": eps_z,
        "MIN_SAMPLES": min_samples,
        "LAYERS_PER_CHUNK": layers_per_chunk,
        "OVERLAP_LAYERS": overlap_layers,
        "SIGNALS": signals,
    }


def create_or_load_config(
    build_dir: str | Path,
    *,
    source: str | Path | None = None,
    stl: str | Path | None = None,
    parts_csv: str | Path | None = None,
) -> dict:
    """
    Load config.toml from the project root, creating it first if absent.

    Parameters
    ----------
    build_dir : str | Path
        Path to the project root.
    source : str, Path, or None
        Override for the packet data directory. Forwarded to
        ``setup_build.create_config`` if the TOML needs to be generated.
    stl : str, Path, or None
        Override for the STL file.
    parts_csv : str, Path, or None
        Override for the QuantAM parts CSV.

    Returns
    -------
    dict with keys for paths, build parameters, assignment method,
    clustering parameters, signal columns, and derived cache paths.
    """
    build_dir = Path(build_dir).resolve()
    toml_path = build_dir / "config.toml"

    if not toml_path.exists():
        from ampm.setup_build import create_config

        create_config(
            build_dir,
            source=source,
            stl=stl,
            parts_csv=parts_csv,
        )

    return load_config(build_dir)
