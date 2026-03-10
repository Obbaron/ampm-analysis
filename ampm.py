import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
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

# AMPMData Columns
_TIME_COL = 0
_DWELL_COL = 1
_X_COL = 2
_Y_COL = 3
_LV_COL = 4
_PLASMA_COL = 5
_MP_COL = 6


class AMPMData:
    """
    Container for AMPM process monitoring data from a Renishaw 500S build.

    Data is stored as a dict of numpy arrays, one per layer, each with 7 columns:
    [Time, Dwell, X, Y, LaserVIEW, Plasma, Meltpool].

    Use the factory method AMPMData.from_directory() to load data, then call
    mask() and correct() in-place to process it.

    Note
    ----
    correct() is only applicable to data from the main Renishaw 500S machine.
    Do not use it on RBV machine data.

    Examples
    --------
    >>> data = AMPMData.from_directory("path/to/ampm/export/packets", 1, 100)
    >>> data.mask("path/to/fullplate.stl")
    >>> data.correct()  # main machine only
    >>> print(len(data))
    """

    # Main 500S LaserVIEW XY correction regression model (reg_XYLv)
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

    def __init__(self, data: dict[int, np.ndarray]) -> None:
        self.data = data
        self.parts: pd.DataFrame | None = None

    @staticmethod
    def _get_cross_section(z: float, part_mesh: trimesh.Trimesh) -> MultiPolygon | None:
        """
        Slice an STL mesh at height z and return the cross-section as a MultiPolygon.

        Uses trimesh.intersections.mesh_plane() to get raw line segments, then
        assembles them into polygons with shapely. This bypasses trimesh's Path3D
        to Path2D conversion which is slow due to scipy sparse graph operations.

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
            raw = trimesh.intersections.mesh_plane(
                part_mesh,
                plane_normal=[0, 0, 1],
                plane_origin=[0, 0, z],
            )
            lines: np.ndarray = np.asarray(raw)
            if lines is None or len(lines) == 0:
                return None
            segments_2d = lines[:, :, :2]
            geoms = shapely.linestrings(segments_2d)  # type: ignore[assignment]
            result = shapely.polygonize(geoms.tolist())  # type: ignore[attr-defined]
            polys = list(result.geoms)
            if not polys:
                return None
            merged = unary_union(polys)
            if merged.is_empty:
                return None
            if isinstance(merged, Polygon):
                return MultiPolygon([merged])
            if isinstance(merged, MultiPolygon):
                return merged
            return None
        except Exception:
            return None

    @staticmethod
    def _mask_layer_worker(
        args: tuple[int, np.ndarray, np.ndarray, np.ndarray, float],
    ) -> tuple[int, np.ndarray | None, str]:
        """
        Worker for threaded masking. Reconstructs the mesh from vertices and
        faces to avoid thread-safety issues with trimesh.load().

        Parameters
        ----------
        args : tuple
            (layer_number, data_array, mesh_vertices, mesh_faces, layer_thickness)

        Returns
        -------
        tuple[int, np.ndarray | None, str]
            (layer_number, masked_array_or_None, log_message)
        """
        j, data, vertices, faces, layer_thickness = args
        part_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        z = (j - 0.5) * layer_thickness
        cross_section = AMPMData._get_cross_section(z, part_mesh)

        if cross_section is None or cross_section.is_empty:
            return (
                j,
                None,
                f"  Layer {j}: no STL cross-section at z={z:.4f}mm, skipping...",
            )

        minx, miny, maxx, maxy = cross_section.bounds
        points_xy = data[:, [_X_COL, _Y_COL]]
        bbox_mask = (
            (points_xy[:, 0] >= minx)
            & (points_xy[:, 0] <= maxx)
            & (points_xy[:, 1] >= miny)
            & (points_xy[:, 1] <= maxy)
        )
        candidates = points_xy[bbox_mask]

        if len(candidates) == 0:
            return j, None, f"  Layer {j}: no points inside STL, skipping..."

        shapely.prepare(cross_section)
        stl_mask = np.zeros(len(data), dtype=bool)
        stl_mask[bbox_mask] = shapely.covers(cross_section, shapely.points(candidates))

        n_masked = stl_mask.sum()
        if n_masked == 0:
            return j, None, f"  Layer {j}: no points inside STL, skipping..."

        return (
            j,
            data[stl_mask],
            f"  Layer {j}: {n_masked} / {len(data)} points inside STL",
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, layer: int) -> np.ndarray:
        return self.data[layer]

    def __iter__(self):
        return iter(self.data)

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def import_parts(
        self,
        filepath: Path | str,
        parametric: bool = False,
    ) -> None:
        """
        Import part metadata from a QuantAM exported parts CSV and store as self.parts.

        Parameters
        ----------
        filepath : Path | str
            Path to the QuantAM exported parts CSV file.
        parametric : bool, optional
            Include Hatch Power, Hatch Point Distance, Hatch Exposure Time (default: False).
        """
        filepath = Path(filepath) if isinstance(filepath, str) else filepath
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        parts_data = pd.read_csv(
            filepath,
            usecols=[1, 3, 4, 5, 6],
            names=[
                "Part ID",
                "Layer Thickness",
                "X Position",
                "Y Position",
                "Layers Count",
            ],
            skiprows=6,
            on_bad_lines="skip",
            skip_blank_lines=True,
        )

        parametric_possible = "Tab - 10" in parts_data["Part ID"].values

        parts_idx = []
        for value in parts_data["Part ID"]:
            if pd.isna(value):
                break
            if isinstance(value, str) and value.startswith("Tab - "):
                break
            parts_idx.append(int(value) - 1)
        parts_data = parts_data.loc[parts_idx]

        if parametric and not parametric_possible:
            raise ValueError(f"No parametric data found in file: {filepath}")

        if parametric and parametric_possible:
            skipped_rows = len(parts_idx) + 10
            first_col = pd.read_csv(
                filepath,
                usecols=[1],
                names=["Part ID"],
                skiprows=skipped_rows,
                on_bad_lines="skip",
                skip_blank_lines=True,
            )
            full_parts_list = []
            for value in first_col["Part ID"]:
                if pd.isna(value):
                    break
                if isinstance(value, str) and value.startswith("Tab - "):
                    break
                if isinstance(value, str):
                    base_value = value.replace(".1", "").replace(".s", "")
                    if base_value in parts_data["Part ID"].values:
                        full_parts_list.append(value)
            if "Tab - 10" and "Tab - 11" in first_col["Part ID"].values:
                skipped_rows = 9 * (len(full_parts_list) + 4) + (len(parts_idx) + 10)
            parts_params = pd.read_csv(
                filepath,
                usecols=[7, 9, 10],
                names=["Hatch Power", "Hatch Point Distance", "Hatch Exposure Time"],
                skiprows=skipped_rows,
                nrows=len(parts_idx),
                on_bad_lines="skip",
                skip_blank_lines=True,
            )
            parts_data = pd.concat([parts_data, parts_params], axis=1)

        self.parts = parts_data
        logger.info(f"Imported {len(parts_data)} parts from {filepath.name}")

    @classmethod
    def from_directory(
        cls,
        filepath: Path | str,
        start_layer: int,
        end_layer: int,
        x_min: float = -125.0,
        x_max: float = 125.0,
        y_min: float = -125.0,
        y_max: float = 125.0,
        laser_number: int = 4,
    ) -> "AMPMData":
        """
        Import AMPM data from a directory of layer files.

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
        AMPMData
            Loaded data object.
        """
        filepath = Path(filepath) if isinstance(filepath, str) else filepath
        if not isinstance(start_layer, int) or not isinstance(end_layer, int):
            raise TypeError(
                f"start_layer and end_layer must be integers, got "
                f"{type(start_layer).__name__} and {type(end_layer).__name__}"
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
        logger.info(f"IMPORTING {num_layers} LAYERS...")

        def _read_layer(j: int) -> tuple[int, np.ndarray | None, str, bool]:
            """Read and ROI-filter a single layer file. Returns (layer, array_or_None, msg, is_warning)."""
            filename = f"Packet data for layer {j}, laser {laser_number}.txt"
            full_path = filepath / filename
            if not full_path.exists():
                return j, None, f"File not found: {filename}, skipping...", True
            layer_data = pd.read_csv(
                full_path,
                sep="\t",
                usecols=USECOLS,
                on_bad_lines="warn",
            )
            layer_data = layer_data.astype(np.float32)
            if np.any(np.isinf(layer_data.to_numpy())):
                return (
                    j,
                    None,
                    f"Layer {j}: inf values detected after float32 cast, possible overflow",
                    True,
                )
            roi_mask = (
                (layer_data["Demand X"] > x_min)
                & (layer_data["Demand X"] < x_max)
                & (layer_data["Demand Y"] > y_min)
                & (layer_data["Demand Y"] < y_max)
            )
            n_filtered = roi_mask.sum()
            if n_filtered == 0:
                return j, None, f"Layer {j} has no points in ROI, skipping...", True
            arr = layer_data.to_numpy()[roi_mask]
            return (
                j,
                arr,
                f"  Layer {j}: found {n_filtered} / {len(layer_data)} points in ROI",
                False,
            )

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_read_layer, j): j
                for j in range(start_layer, end_layer + 1)
            }
            results = {}
            for future in as_completed(futures):
                j, arr, msg, is_warning = future.result()
                results[j] = (arr, msg, is_warning)

        data = {}
        for j in range(start_layer, end_layer + 1):
            arr, msg, is_warning = results[j]
            if is_warning:
                logger.warning(msg)
            else:
                logger.info(msg)
            if arr is not None:
                data[j] = arr

        expected = set(range(start_layer, end_layer + 1))
        missing = expected - set(data.keys())
        if missing:
            logger.warning(f"Missing layers after import: {sorted(missing)}")
        logger.info(f"\nSUCCESSFULLY IMPORTED {len(data)} LAYERS\n")

        return cls(data)

    def mask(
        self,
        stl_path: Path | str,
        layer_thickness: float = 0.03,
    ) -> None:
        """
        Mask data in-place to only include points within the STL geometry.

        Parameters
        ----------
        stl_path : Path | str
            Path to the STL file of the parts. As this is a direct export from
            the Renishaw 500S machine, coordinates are guaranteed to match the
            AMPM data.
        layer_thickness : float, optional
            Layer thickness in mm (default: 0.03).
        """
        stl_path = Path(stl_path) if isinstance(stl_path, str) else stl_path
        if not stl_path.exists():
            raise FileNotFoundError(f"STL file not found: {stl_path}")
        if layer_thickness <= 0:
            raise ValueError(f"layer_thickness must be positive, got {layer_thickness}")

        logger.info(f"Loading STL: {stl_path.name}...")
        part_mesh = trimesh.load(str(stl_path), force="mesh", process=False)
        if not isinstance(part_mesh, trimesh.Trimesh):
            raise TypeError(
                f"Expected Trimesh after loading STL, got {type(part_mesh).__name__}. "
                f"Ensure the STL file contains a single mesh."
            )

        vertices = np.array(part_mesh.vertices)
        faces = np.array(part_mesh.faces)

        logger.info("Masking layers to STL cross-sections...")
        worker_args = [
            (j, data, vertices, faces, layer_thickness) for j, data in self.data.items()
        ]

        try:
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self._mask_layer_worker, args): args[0]
                    for args in worker_args
                }
                results = {}
                for future in as_completed(futures):
                    j, masked, msg = future.result()
                    results[j] = (masked, msg)
        except KeyboardInterrupt:
            logger.warning("Masking interrupted by user.")
            raise

        self.data = {}
        for j, (masked, msg) in sorted(results.items()):
            if masked is None:
                logger.warning(msg)
            else:
                logger.info(msg)
                self.data[j] = masked
        logger.info(f"\nSUCCESSFULLY MASKED {len(self.data)} LAYERS\n")

    def correct(self) -> None:
        """
        Apply LaserVIEW-based XY positional correction to MeltPool signal in-place.

        This correction is only applicable to data collected on the main Renishaw
        500S machine. Do NOT apply this correction to data from the RBV (Reduced
        Build Volume) machine, as the regression model was fitted to the main
        machine's optical response and will produce incorrect results on the RBV.
        """
        logger.info("APPLYING XY MELTPOOL CORRECTION...")

        for j, data in self.data.items():
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

            point_scores = np.prod(
                point_matrix[:, np.newaxis, :] ** self._POWER_MATRIX, axis=2
            )
            origin_scores = np.prod(
                origin_matrix[:, np.newaxis, :] ** self._POWER_MATRIX, axis=2
            )

            point_pred = point_scores @ self._COEFFICIENTS
            origin_pred = origin_scores @ self._COEFFICIENTS

            data[:, _MP_COL] = mp_vals * (origin_pred / point_pred)

            logger.info(f"  Layer {j}: correction applied to {len(data)} points")

        logger.info(f"\nSUCCESSFULLY CORRECTED {len(self.data)} LAYERS\n")

    def save(self, path: Path | str) -> None:
        """
        Save the current state of the AMPMData object to an HDF5 file.

        Each layer is stored as a dataset named by its layer number. The file
        can also be read directly in MATLAB using h5read.

        Parameters
        ----------
        path : Path | str
            Output file path. Conventionally use a .h5 extension.
        """
        path = Path(path) if isinstance(path, str) else path
        with h5py.File(path, "w") as f:
            for j, data in self.data.items():
                f.create_dataset(str(j), data=data)
        logger.info(f"Saved {len(self.data)} layers to {path.name}")

    @classmethod
    def load(cls, path: Path | str) -> "AMPMData":
        """
        Load an AMPMData object from an HDF5 file.

        Parameters
        ----------
        path : Path | str
            Path to a .h5 file previously saved with AMPMData.save().

        Returns
        -------
        AMPMData
            Loaded data object.
        """
        path = Path(path) if isinstance(path, str) else path
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        data = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                data[int(key)] = np.array(f[key])
        logger.info(f"Loaded {len(data)} layers from {path.name}")
        return cls(data)


if __name__ == "__main__":
    ampm_dir = "C:/Users/ohp460/Documents/Code/ampm-data/JR306_Fares_plate/JR306_AMPM/[3] Export Packets"
    fullplate_stl = "C:/Users/ohp460/Documents/Code/ampm-data/JR306_Fares_plate/JR306_ElDesF_CranialRepeat_20260127/STL/fullplate/JR306_FULLPLATE_STL.stl"
    save_path = "C:/Users/ohp460/Documents/Code/ampm-data/JR306_Fares_plate/JR309_masked_corrected.h5"
    parts_csv = "C:/Users/ohp460/Documents/Code/ampm-data/JR306_parts.csv"

    # data = AMPMData.from_directory(ampm_dir, 165, 265)
    # data.mask(fullplate_stl)
    # data.correct()
    # data.save(save_path)

    data = AMPMData.load(save_path)
    data.import_parts(parts_csv, parametric=True)

    print(data.parts)
