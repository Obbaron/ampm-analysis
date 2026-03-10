import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

file_logger = logging.FileHandler("ampm.log", mode="a")
file_logger.setLevel(logging.INFO)
file_logger.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
logger.addHandler(file_logger)

console_logger = logging.StreamHandler()
console_logger.setLevel(logging.INFO)
console_logger.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console_logger)

logger.setLevel(logging.INFO)
logger.propagate = False

# Column indices for numpy arrays
_TIME_COL = 0
_DWELL_COL = 1
_X_COL = 2
_Y_COL = 3
_LV_COL = 4
_PLASMA_COL = 5
_MP_COL = 6

# LaserVIEW XY correction regression model (reg_XYLv), fitted to the main
# Renishaw 500S machine. Do not use on RBV data.
_POWER_MATRIX = np.array(
    [
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [0, 0, 0],
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
    ]
)
_COEFFICIENTS = np.array(
    [
        -3.21229823e-01,
        -1.36372304e-01,
        1.93998884e-03,
        1.81922155e-02,
        -7.99325110e-04,
        -5.58476892e-04,
        1.20919463e02,
        -1.25344185e-03,
        -5.77211144e-04,
        3.72306302e-03,
    ]
)


def _get_cross_section(z: float, part_mesh: trimesh.Trimesh) -> MultiPolygon | None:
    """
    Slice an STL mesh at height z and return the cross-section as a MultiPolygon.

    Parameters
    ----------
    z : float
        Height at which to slice the mesh (mm).
    part_mesh : trimesh.Trimesh
        Loaded STL mesh object.

    Returns
    -------
    MultiPolygon | None
        Cross-sectional geometry at height z, or None if no geometry exists.
    """
    try:
        section = part_mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if section is None:
            return None
        cross_section, _ = section.to_2D()
        polygons = cross_section.polygons_full
        if not polygons:
            return None
        result = unary_union(polygons)
        if result.is_empty:
            return None
        if isinstance(result, Polygon):
            return MultiPolygon([result])
        if isinstance(result, MultiPolygon):
            return result
        return None
    except Exception as e:
        logger.error(f"_get_cross_section error at z={z:.4f}mm: {e}")
        return None


def import_ampm_data(
    filepath: Path | str,
    start_layer: int,
    end_layer: int,
    x_min: float = -125.0,
    x_max: float = 125.0,
    y_min: float = -125.0,
    y_max: float = 125.0,
    laser_number: int = 4,
) -> dict[int, np.ndarray]:
    """
    Import AMPM data from region of interest.

    Parameters
    ----------
    filepath : Path | str
        Directory path containing the data files.
    start_layer : int
        First layer number to process.
    end_layer : int
        Last layer number to process (inclusive).
    x_min : float, optional
        Lower x-coordinate boundary for ROI (mm) (default: -125.0).
    x_max : float, optional
        Upper x-coordinate boundary for ROI (mm) (default: 125.0).
    y_min : float, optional
        Lower y-coordinate boundary for ROI (mm) (default: -125.0).
    y_max : float, optional
        Upper y-coordinate boundary for ROI (mm) (default: 125.0).
    laser_number : int, optional
        Laser number in filename (default: 4).

    Returns
    -------
    roi_data : dict[int, np.ndarray]
        Dict of numpy arrays, one per layer. Key is layer number.
        7 columns: [Time, Dwell, X, Y, LaserVIEW, Plasma, Meltpool].
    """
    filepath = Path(filepath) if isinstance(filepath, str) else filepath
    if not isinstance(start_layer, int) or not isinstance(end_layer, int):
        raise TypeError(
            f"start_layer and end_layer must be integers, got {type(start_layer).__name__} and {type(end_layer).__name__}"
        )
    if not isinstance(laser_number, int):
        raise TypeError(
            f"laser_number must be an integer, got {type(laser_number).__name__}"
        )
    if not filepath.exists():
        raise FileNotFoundError(f"Directory not found: {filepath}")
    if not filepath.is_dir():
        raise NotADirectoryError(f"filepath must be a directory: {filepath}")
    if x_min >= x_max:
        raise ValueError(f"x_min ({x_min}) must be < x_max ({x_max})")
    if y_min >= y_max:
        raise ValueError(f"y_min ({y_min}) must be < y_max ({y_max})")
    if start_layer < 1:
        raise ValueError(f"start_layer must be >= 1, got {start_layer}")
    if start_layer > end_layer:
        raise ValueError(
            f"start_layer ({start_layer}) must be <= end_layer ({end_layer})"
        )
    if laser_number < 1:
        raise ValueError(f"laser_number must be positive, got {laser_number}")

    USECOLS = [
        "Start time",
        "Duration",
        "Demand X",
        "Demand Y",
        "LaserVIEW (mean)",
        "MeltVIEW plasma (mean)",
        "MeltVIEW melt pool (mean)",
    ]
    num_layers = end_layer - start_layer + 1
    roi_data = {}
    logger.info(f"IMPORTING {num_layers} LAYERS...")

    for j in range(start_layer, end_layer + 1):
        filename = f"Packet data for layer {j}, laser {laser_number}.txt"
        full_path = filepath / filename
        if not full_path.exists():
            logger.warning(f"File not found: {filename}, skipping...")
            continue
        layer_data = pd.read_csv(
            full_path,
            sep="\t",
            usecols=USECOLS,
            on_bad_lines="warn",
        )
        layer_data = layer_data.astype(np.float32)
        if np.any(np.isinf(layer_data.to_numpy())):
            logger.warning(
                f"  Layer {j}: inf values detected after float32 cast, possible overflow"
            )
        roi_mask = (
            (layer_data["Demand X"] > x_min)
            & (layer_data["Demand X"] < x_max)
            & (layer_data["Demand Y"] > y_min)
            & (layer_data["Demand Y"] < y_max)
        )
        n_filtered = roi_mask.sum()
        logger.info(
            f"  Layer {j}: found {n_filtered} / {len(layer_data)} points in ROI"
        )
        if n_filtered == 0:
            logger.warning(f"  Layer {j} has no points in ROI, skipping...")
            continue
        roi_data[j] = layer_data.to_numpy()[roi_mask]

    expected = set(range(start_layer, end_layer + 1))
    missing = expected - set(roi_data.keys())
    if missing:
        logger.warning(f"Missing layers after import: {sorted(missing)}")
    logger.info(f"\nSUCCESSFULLY IMPORTED {len(roi_data)} LAYERS\n")

    return roi_data


def mask_ampm_data(
    roi_data: dict[int, np.ndarray],
    stl_path: Path | str,
    layer_thickness: float = 0.03,
) -> dict[int, np.ndarray]:
    """
    Mask AMPM data to only include points within the STL geometry.

    Parameters
    ----------
    roi_data : dict[int, np.ndarray]
        Output from import_ampm_data. Key is layer number, value is numpy
        array with columns [Time, Dwell, X, Y, LaserVIEW, Plasma, Meltpool].
    stl_path : Path | str
        Path to the STL file of the parts. As this is a direct export from the
        Renishaw 500S machine, coordinates are guaranteed to match the AMPM data.
    layer_thickness : float, optional
        Layer thickness in mm (default: 0.03).

    Returns
    -------
    masked_data : dict[int, np.ndarray]
        Same structure as roi_data but with only points inside the STL.
    """
    stl_path = Path(stl_path) if isinstance(stl_path, str) else stl_path
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    if layer_thickness <= 0:
        raise ValueError(f"layer_thickness must be positive, got {layer_thickness}")

    logger.info(f"Loading STL: {stl_path.name}...")
    part_mesh = trimesh.load(str(stl_path), force="mesh")
    if not isinstance(part_mesh, trimesh.Trimesh):
        raise TypeError(
            f"Expected Trimesh after loading STL, got {type(part_mesh).__name__}. "
            f"Ensure the STL file contains a single mesh."
        )

    logger.info("Masking layers to STL cross-sections...")
    masked_data = {}

    for j, data in roi_data.items():
        z = (j - 0.5) * layer_thickness  # midpoint of layer j
        cross_section = _get_cross_section(z, part_mesh)

        if cross_section is None or cross_section.is_empty:
            logger.warning(
                f"  Layer {j}: no STL cross-section at z={z:.4f}mm, skipping..."
            )
            continue

        points_xy = data[:, [_X_COL, _Y_COL]]
        shapely.prepare(cross_section)
        stl_mask = shapely.covers(cross_section, shapely.points(points_xy))

        n_masked = stl_mask.sum()
        logger.info(f"  Layer {j}: {n_masked} / {len(data)} points inside STL")

        if n_masked == 0:
            logger.warning(f"  Layer {j}: no points inside STL, skipping...")
            continue

        masked_data[j] = data[stl_mask]

    logger.info(f"\nSUCCESSFULLY MASKED {len(masked_data)} LAYERS\n")
    return masked_data


def correct_ampm_data(
    roi_data: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """
    Apply LaserVIEW-based XY positional correction to MeltPool signal.

    This correction is only applicable to data collected on the main Renishaw
    500S machine. Do NOT apply this correction to data from the RBV (Reduced
    Build Volume) machine, as the regression model was fitted to the main
    machine's optical response and will produce incorrect results on the RBV.

    Parameters
    ----------
    roi_data : dict[int, np.ndarray]
        Output from import_ampm_data or mask_ampm_data. Key is layer number,
        value is numpy array with columns:
        [Time, Dwell, X, Y, LaserVIEW, Plasma, Meltpool].

    Returns
    -------
    corrected_data : dict[int, np.ndarray]
        Same structure as roi_data with corrected Meltpool column.
    """
    logger.info("APPLYING XY MELTPOOL CORRECTION...")
    corrected_data = {}

    for j, data in roi_data.items():
        corrected = data.copy()
        x_vals = data[:, _X_COL]
        y_vals = data[:, _Y_COL]
        lv_vals = data[:, _LV_COL]
        mp_vals = data[:, _MP_COL]

        point_matrix = np.column_stack([x_vals, y_vals, lv_vals])
        origin_matrix = np.column_stack(
            [
                np.zeros(len(data)),
                np.zeros(len(data)),
                lv_vals,
            ]
        )

        point_scores = np.prod(point_matrix[:, np.newaxis, :] ** _POWER_MATRIX, axis=2)
        origin_scores = np.prod(
            origin_matrix[:, np.newaxis, :] ** _POWER_MATRIX, axis=2
        )

        point_pred = point_scores @ _COEFFICIENTS
        origin_pred = origin_scores @ _COEFFICIENTS

        corrected[:, _MP_COL] = mp_vals * (origin_pred / point_pred)

        logger.info(f"  Layer {j}: correction applied to {len(data)} points")
        corrected_data[j] = corrected

    logger.info(f"\nSUCCESSFULLY CORRECTED {len(corrected_data)} LAYERS\n")
    return corrected_data


if __name__ == "__main__":
    ampm_dir = "path/to/ampm/export/packets"
    fullplate_stl = "path/to/fullplate.stl"

    data = import_ampm_data(ampm_dir, 165, 185)
    data = mask_ampm_data(data, fullplate_stl)
    data = correct_ampm_data(data)
    print(len(data))
