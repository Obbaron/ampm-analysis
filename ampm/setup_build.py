"""
setup_build.py - auto-detect build files and generate config.toml

Walks a project directory to locate the source data, STL, and QuantAM
parts CSV, then writes a config.toml with relative paths. Each field
can be overridden explicitly if auto-detection picks the wrong file.

Usage from other scripts:
    from setup_build import create_config
    create_config("/path/to/project_root")
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

_PACKET_RE = re.compile(
    r"^Packet data for layer \d+, laser \d+\.txt$",
    re.IGNORECASE,
)

_QUANTAM_HEADER = "#,Renishaw,Material,Development"

_STL_KEYWORDS = {"fullplate", "full", "plate"}


def _find_source_dir(build_dir: Path) -> Path:
    """Walk the tree and find the directory containing packet data files."""

    candidates: set[Path] = set()

    for path in build_dir.rglob("*.txt"):
        if _PACKET_RE.match(path.name):
            candidates.add(path.parent)

    if not candidates:
        raise FileNotFoundError(
            f"No 'Packet data for layer N, laser M.txt' files found "
            f"under {build_dir}"
        )

    if len(candidates) > 1:
        raise ValueError(
            "Packet data files found in multiple directories:\n"
            + "\n".join(f"  {c}" for c in sorted(candidates))
        )

    return candidates.pop()


def _stl_depth(path: Path, build_dir: Path) -> int:
    """Number of directories between the file and the build root."""

    return len(path.relative_to(build_dir).parts) - 1


def _stl_has_keyword(path: Path) -> bool:
    """True if the filename contains 'fullplate', 'full', or 'plate'."""

    name_lower = path.stem.lower()
    return any(kw in name_lower for kw in _STL_KEYWORDS)


def _stl_is_support(path: Path) -> bool:
    """True if the filename ends with '_s' (QuantAM supports export)."""

    return path.stem.lower().endswith("_s")


def _find_stl(build_dir: Path) -> Path:
    """
    Walk the tree and find the best STL file.

    Priority (applied in order):
    1. Depth from root — shallower wins
    2. Name keywords — 'fullplate', 'full', 'plate' preferred
    3. Non-support — '<name>.stl' preferred over '<name>_s.stl'
    """

    stls = sorted(build_dir.rglob("*.stl"))

    if not stls:
        raise FileNotFoundError(f"No .stl files found under {build_dir}")

    if len(stls) == 1:
        return stls[0]

    stls.sort(
        key=lambda p: (
            _stl_depth(p, build_dir),
            0 if _stl_has_keyword(p) else 1,
            1 if _stl_is_support(p) else 0,
        )
    )

    return stls[0]


def _is_quantam_csv(path: Path) -> bool:
    """Check if the first line of a CSV matches the QuantAM header."""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            first_line = f.readline().strip()

        return first_line == _QUANTAM_HEADER

    except (OSError, UnicodeDecodeError):

        return False


def _find_parts_csv(build_dir: Path) -> Path:
    """
    Walk the tree and find the QuantAM parts CSV.

    If multiple CSVs exist, filter by the QuantAM header. Error if
    zero or multiple CSVs match.
    """
    csvs = sorted(build_dir.rglob("*.csv"))

    if not csvs:
        raise FileNotFoundError(f"No .csv files found under {build_dir}")

    if len(csvs) == 1:
        return csvs[0]

    quantam_csvs = [part for part in csvs if _is_quantam_csv(part)]

    if len(quantam_csvs) == 1:
        return quantam_csvs[0]

    if len(quantam_csvs) == 0:
        raise FileNotFoundError(
            f"Found {len(csvs)} CSV files under {build_dir} but none "
            f"have the QuantAM header."
        )

    raise ValueError(
        f"Found {len(quantam_csvs)} CSVs with the QuantAM header:"
        + "\n".join(f"  {path}" for path in quantam_csvs)
        + "\nCannot determine which is the parts file."
    )


def _extract_layer_thickness(parts_csv: Path) -> float:
    """
    Parse the QuantAM parts CSV and return the layer thickness from the
    first data row in the Parent Parts section.
    """
    with open(parts_csv, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    in_parent_parts = False
    header_cols: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_parent_parts and header_cols:
                break  # blank line ends the section
            continue

        if stripped.startswith("#,Tab - -1,Parent Parts"):
            in_parent_parts = True
            continue

        if in_parent_parts and not header_cols and stripped.startswith("#,"):
            reader = csv.reader([stripped])
            header_cols = [c.strip().strip('"') for c in next(reader)]
            continue

        if in_parent_parts and stripped.startswith("ID."):
            continue  # skip machine-code row

        if in_parent_parts and header_cols and stripped.startswith(","):
            reader = csv.reader([stripped])
            row = [c.strip().strip('"') for c in next(reader)]
            try:
                lt_index = header_cols.index("Layer Thickness")
            except ValueError:
                raise ValueError(
                    f"'Layer Thickness' column not found in Parent Parts "
                    f"header: {header_cols}"
                )
            return float(row[lt_index])

    raise ValueError(
        f"Could not extract layer thickness from {parts_csv}. "
        f"No data rows found in the Parent Parts section."
    )


def create_config(
    build_dir: str | Path,
    *,
    source: str | Path | None = None,
    stl: str | Path | None = None,
    parts_csv: str | Path | None = None,
) -> Path:
    """
    Auto-detect build files and write a config.toml into the project root.

    Parameters
    ----------
    build_dir : str or Path
        Root directory of the project.
    source : str, Path, or None
        Path to the packet data directory. Auto-detected if None.
    stl : str, Path, or None
        Path to the STL file. Auto-detected if None.
    parts_csv : str, Path, or None
        Path to the QuantAM parts CSV. Auto-detected if None.

    Returns
    -------
    Path to the written config.toml.
    """
    build_dir = Path(build_dir).resolve()

    if not build_dir.is_dir():
        raise FileNotFoundError(f"Directory not found:\n{build_dir}")

    if source is not None:
        source_path = Path(source).resolve()
    else:
        source_path = _find_source_dir(build_dir)

    if stl is not None:
        stl_path: Path | None = Path(stl).resolve()

    else:
        try:
            stl_path = _find_stl(build_dir)

        except FileNotFoundError:
            stl_path = None

    if parts_csv is not None:
        csv_path: Path | None = Path(parts_csv).resolve()

    else:
        try:
            csv_path = _find_parts_csv(build_dir)

        except FileNotFoundError:
            csv_path = None

    layer_thickness = _extract_layer_thickness(csv_path) if csv_path else None

    def _rel(path: Path | None) -> str:
        if path is None:
            return ""
        try:
            return str(path.relative_to(build_dir))
        except ValueError:
            return str(path)  # absolute if outside build_dir

    source_rel = _rel(source_path)
    stl_rel = _rel(stl_path)
    csv_rel = _rel(csv_path)

    if layer_thickness is None:
        lt_line = (
            "layer_thickness = 0.0  # SET THIS (mm): no parts CSV found to "
            "auto-detect from\n"
        )

    else:
        lt_line = f"layer_thickness = {layer_thickness}  # mm\n"

    toml_path = build_dir / "config.toml"
    toml_content = (
        "# ampm build configuration (auto-generated by setup_build.py)\n"
        "#\n"
        "# Paths are relative to this file's directory.\n"
        "# stl/parts_csv may be empty: STL is only needed for masking, the\n"
        "# parts CSV only for the 'direct'/'dbscan' methods and power/speed.\n"
        "\n"
        "[paths]\n"
        f"source    = '{source_rel}'\n"
        f"stl       = '{stl_rel}'\n"
        f"parts_csv = '{csv_rel}'\n"
        "\n"
        "[build]\n"
        f"{lt_line}"
        "\n"
        "[assignment]\n"
        "method          = 'direct'  # 'direct', 'dbscan', or 'dhxml'\n"
        "max_distance_mm = 'none'    # only used with direct; 'none' = assign all\n"
        "\n"
        "[clustering]\n"
        "eps_xy           = 0.3\n"
        "eps_z            = 0.06\n"
        "min_samples      = 10\n"
        "layers_per_chunk = 11\n"
        "overlap_layers   = 'auto'   # 'auto' = max(2, ceil(eps_z / layer_thickness) * 2)\n"
        "\n"
        "[signals]\n"
        "columns = ['MeltVIEW melt pool (mean)', 'Laser output power (mean)']\n"
    )

    toml_path.write_text(toml_content, encoding="utf-8")

    return toml_path
