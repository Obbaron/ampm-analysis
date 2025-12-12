import logging
import os
import tkinter as tk
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, linregress
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

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


def import_ampm_data(
    filepath: Path | str,
    start_layer: int,
    end_layer: int,
    x_min: float = -125.0,
    x_max: float = 125.0,
    y_min: float = -125.0,
    y_max: float = 125.0,
    laser_number: int = 4,
    return_dict: bool = False,
) -> list[np.ndarray] | dict:
    """
    Import AMPM data from region of interest: [Layer, Time, Dwell, X, Y, Plasma, Meltpool].

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
    return_dict : bool, optional
        If True, return dict with layer number as keys (default: False).

    Returns
    -------
    roi_data : list | dict
        List of numpy arrays (dict if return_dict=True), one array per layer.
        7 columns: [layer, time, duration, x, y, plasma, meltpool].
    """

    filepath = Path(filepath) if isinstance(filepath, str) else filepath
    if not filepath.exists():
        raise FileNotFoundError(f"Directory not found: {filepath}")
    if x_min >= x_max:
        raise ValueError(f"x_min ({x_min}) must be < x_max ({x_max})")
    if y_min >= y_max:
        raise ValueError(f"y_min ({y_min}) must be < y_max ({y_max})")
    if start_layer > end_layer:
        raise ValueError(
            f"start_layer ({start_layer}) must be <= end_layer ({end_layer})"
        )
    if start_layer < 0:
        raise ValueError(f"start_layer must be non-negative, got {start_layer}")

    num_layers = end_layer - start_layer + 1
    roi_data = {} if return_dict else []

    logger.info(f"IMPORTING {num_layers} LAYERS...")

    for idx, j in enumerate(range(start_layer, end_layer + 1)):
        logger.info(f"Importing layer {j} ({idx + 1}/{num_layers})")

        filename = f"Packet data for layer {j}, laser {laser_number}.txt"
        full_path = filepath / filename

        if not full_path.exists():
            logger.warning(f"File not found: {filename}, skipping...")
            continue

        try:
            layer_data = pd.read_csv(
                full_path,
                sep="\t",
                usecols=([0, 1, 2, 3, 5, 6]),  # Skips LaserVIEW
                on_bad_lines="warn",
            )

            layer_array = layer_data.to_numpy()

            locations = (
                (layer_array[:, 2] > x_min)
                & (layer_array[:, 2] < x_max)
                & (layer_array[:, 3] > y_min)
                & (layer_array[:, 3] < y_max)
            )

            filtered_data = layer_array[locations, :]

            layer_column = np.full((filtered_data.shape[0], 1), j)
            filtered_data = np.column_stack((layer_column, filtered_data))

            logger.info(f"  Found {locations.sum()} / {len(layer_array)} points in ROI")

            if return_dict:
                roi_data[j] = filtered_data
            else:
                roi_data.append(filtered_data)

        except Exception as e:
            logger.error(f"Error importing layer {j}: {str(e)}\n")
            raise

    logger.info("\n")
    logger.info(f"SUCCESSFULLY IMPORTED {len(roi_data)} LAYERS\n")

    return roi_data


def find_ampm_files(project_directory: Path | str) -> dict[Path | None]:
    """
    Locate all relavant AMPM data files in a given directory and its subdirectories.

    Parameters
    ----------
    project_directory : Path | str
        Path to the directory to search.

    Returns
    -------
    project_info : dict[Path | None]
        Dictionary containing project metadata with the following keys:
        - 'project_directory' : Path
            Path to root project directory containing all relavant files.
        - 'data_directory' : Path
            Path to the directory containing the AMPM packet data files.
        - 'parts_filepath' : Path | None
            Path to the QuantAM parts CSV file, or None if not found.

    Raises
    ------
    FileNotFoundError
        If the project_directory does not exist, or no packet data files are
        found within the project_directory or its subdirectories.
    ValueError
        If multiple valid data directories or parts files are found.

    Notes
    -----
    If project_directory is a subdirectory starting with 'Spectral.ExportPackets.'
    or 'output', the function will recursively navigate up to its parent directories
    to find the project root.
    """
    project_directory = (
        Path(project_directory)
        if isinstance(project_directory, str)
        else project_directory
    )
    if not project_directory.exists():
        raise FileNotFoundError(f"Project directory not found: {project_directory}")

    original_path = project_directory
    while project_directory.name.startswith(
        "Spectral.ExportPackets."
    ) or project_directory.name.startswith("output"):
        project_directory = project_directory.parent

    if project_directory != original_path:
        print(
            f"Detected sub-directory '{original_path.name}'. "
            f"Using parent directory as project root: {project_directory}"
        )

    project_info = {
        "project_directory": project_directory,
        "data_directory": None,
        "parts_filepath": None,
    }

    found_data_dirs = []
    found_parts_files = []

    for root, _, files in os.walk(project_directory):
        root_path = Path(root)

        for file in files:
            if file.endswith(".txt") and file.startswith("Packet data for layer "):
                if root_path not in found_data_dirs:
                    found_data_dirs.append(root_path)
                break

        for file in files:
            if file.lower().startswith("jr") and file.endswith(".csv"):
                parts_path = root_path / file
                if parts_path not in found_parts_files:
                    found_parts_files.append(parts_path)
                break

    if len(found_data_dirs) == 0:
        raise FileNotFoundError(
            f"No packet data files found in {project_directory}.\nExpected .txt files starting with 'Packet data for layer...'"
        )
    elif len(found_data_dirs) > 1:
        dirs_list = "\n  ".join(str(d) for d in found_data_dirs)
        raise ValueError(
            f"Multiple data directories found in {project_directory}:\n  {dirs_list}\n"
            "Specify a more specific project directory."
        )
    else:
        project_info["data_directory"] = found_data_dirs[0]

    if len(found_parts_files) == 0:
        print(
            f"No parts CSV file found in {project_directory}.\nExpected .csv file starting with 'jr...'"  # logger.warn
        )
    elif len(found_parts_files) > 1:
        files_list = "\n  ".join(str(f) for f in found_parts_files)
        raise ValueError(
            f"Multiple parts CSV files found in {project_directory}:\n  {files_list}\n"
            "Specify a more specific project directory or remove duplicate files."
        )
    else:
        project_info["parts_filepath"] = found_parts_files[0]

    return project_info


def get_parts(filepath: Path | str, parametric: bool = False) -> pd.DataFrame:
    """
    Create a pandas DataFrame from parts CSV exported from QuantAM.

    Parameters
    ----------
    filepath : Path | str
        Path to QuantAM exported parts CSV file.
    parametric : bool, optional
        Include Hatch Power, Hatch Point Distance, Hatch Exposure Time (default: False).

    Returns
    -------
    parts_data : pd.DataFrame
        DataFrame of [Part ID, Layer Thickness, X Position, Y Position, Layers Count].
        Optionally include [Hatch Power, Hatch Point Distance, Hatch Exposure Time].
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

    if "Tab - 10" in parts_data["Part ID"].values:
        parametric_possible = True
    else:
        parametric_possible = False

    parts_idx = []
    for value in parts_data["Part ID"]:
        if pd.isna(value):
            break
        if isinstance(value, str) and value.startswith(
            "Tab - "
        ):  # Exports from QuantAM go straight to next tab
            break
        parts_idx.append(int(value) - 1)

    parts_data = parts_data.loc[parts_idx]

    if parametric and not parametric_possible:
        raise ValueError(f"No parametric data found in file: {filepath}")

    # Count number of parts, use as spacer to jump to Tab 10, get parametric data
    elif parametric and parametric_possible:
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
            nrows=(len(parts_idx)),
            on_bad_lines="skip",
            skip_blank_lines=True,
        )

        parts_data = pd.concat([parts_data, parts_params], axis=1)

        return parts_data

    else:
        return parts_data


def cluster_data(
    data: list[np.ndarray],
    eps_xy: float = 0.135,
    eps_z: float = 0.135,
    min_samples: int = 50,
    layers_per_chunk: int = 10,
    overlap_layers: int = 2,
    layer_spacing: float = 0.03,
    visualize: bool = False,
) -> np.ndarray:
    """
    Groups AMPM data into clusters using a chunked 3D DBSCAN.

    Parameters
    ----------
    data : list[np.ndarray]
        Output from import_data().
        7 columns: [Layer, Time, Dwell, X, Y, Plasma, Meltpool].
    eps_xy : float, optional
        Max XY distance (mm) between two samples for one to be considered as in the neighborhood of the other  (default: 0.10).
    eps_z : float, optional
        Effective 3D eps = sqrt(eps_xy**2 + eps_z**2) (default: 0.10).
    min_samples : int, optional
        Number of samples in a neighborhood for a point to be considered a core point (default: 20).
    layers_per_chunk : int, optional
        Layers are processed in chunks to limit memory requirements (default: 10).
    overlap_layers : int, optional
        Number of overlapping layers at chunk boundary to maintain Z continuity (default: 2).
    layer_spacing : float, optional
        Z coords = layer number * layer spacing (default: 0.03).
    visualize : bool, optional
        Set to True to see a decimated 3D scatter plot after cluserting (default: False).

    Returns
    -------
    clustered_data : np.ndarray
        Data with ClusterID labels (noise = -1).
        8 columns: [Layer, Time, Dwell, X, Y, Plasma, Meltpool, ClusterID].
    """
    clustered_data = np.vstack(data)

    coords_3d = np.column_stack(
        [
            clustered_data[:, 3],  # X
            clustered_data[:, 4],  # Y
            clustered_data[:, 0] * layer_spacing,  # Z
        ]
    )

    logger.info("INITIALIZING CLUSTERING...")
    logger.info(f"Total data points: {coords_3d.shape[0]}")
    logger.info("Coordinate ranges:")
    logger.info(f"  X: [{coords_3d[:, 0].min():.2f}, {coords_3d[:, 0].max():.2f}]")
    logger.info(f"  Y: [{coords_3d[:, 1].min():.2f}, {coords_3d[:, 1].max():.2f}]")
    logger.info(f"  Z: [{coords_3d[:, 2].min():.2f}, {coords_3d[:, 2].max():.2f}]\n")
    logger.info("Performing chunked 3D clustering across all layers...")

    eps_3d = np.sqrt(eps_xy**2 + eps_z**2)

    logger.info("Clustering parameters:")
    logger.info(f"  eps (XY): {eps_xy} mm")
    logger.info(f"  eps (Z): {eps_z} mm")
    logger.info(f"  Effective 3D eps: {eps_3d:.3f} mm")
    logger.info(f"  min_samples: {min_samples}")
    logger.info(f"  Layer spacing: layer * {layer_spacing}\n")

    unique_layers = np.unique(clustered_data[:, 0])
    n_layers = len(unique_layers)

    logger.info(
        f"Processing {n_layers} layers in chunks of {layers_per_chunk} with {overlap_layers} layer overlap...\n"
    )
    chunk_num = 0

    labels = np.full(len(clustered_data), -1, dtype=int)
    global_cluster_id = 0

    for chunk_start in range(0, n_layers, layers_per_chunk - overlap_layers):
        chunk_end = min(chunk_start + layers_per_chunk, n_layers)
        chunk_layers = unique_layers[chunk_start:chunk_end]

        if chunk_start > 0 and chunk_end <= chunk_start + overlap_layers:
            logger.info(
                f"Skipping redundant chunk: Layers {int(chunk_layers[0])}-{int(chunk_layers[-1])} (entirely overlap)"
            )
            continue

        chunk_mask = np.isin(clustered_data[:, 0], chunk_layers)
        chunk_coords = coords_3d[chunk_mask]
        chunk_indices = np.where(chunk_mask)[0]
        chunk_num += 1

        logger.info(
            f"Chunk {chunk_num}: Layers {int(chunk_layers[0])}-{int(chunk_layers[-1])} ({len(chunk_coords):,} points)..."
        )

        clustering = DBSCAN(eps=eps_3d, min_samples=min_samples, n_jobs=-1)
        chunk_labels = clustering.fit_predict(chunk_coords)

        n_clusters_chunk = len(set(chunk_labels)) - (1 if -1 in chunk_labels else 0)
        n_noise_chunk = list(chunk_labels).count(-1)

        logger.info(
            f"  Found {n_clusters_chunk} clusters, {n_noise_chunk:,} noise points"
        )

        if chunk_start == 0:
            # First chunk: assign directly
            chunk_labels[chunk_labels >= 0] += global_cluster_id
            labels[chunk_indices] = chunk_labels
            global_cluster_id += n_clusters_chunk
        else:
            # Overlapping chunk: merge with previous clusters
            overlap_layer_start = chunk_layers[0]
            overlap_layer_end = chunk_layers[
                min(overlap_layers - 1, len(chunk_layers) - 1)
            ]

            overlap_mask = (clustered_data[chunk_indices, 0] >= overlap_layer_start) & (
                clustered_data[chunk_indices, 0] <= overlap_layer_end
            )

            logger.info(
                f"  Overlap region: layers {int(overlap_layer_start)}-{int(overlap_layer_end)}"
            )
            logger.info(f"  Points in overlap: {overlap_mask.sum():,}")
            logger.info(f"  Points in non-overlap: {(~overlap_mask).sum():,}")

            # Match clusters in overlap region
            for new_cluster_id in range(n_clusters_chunk):
                new_cluster_mask = chunk_labels == new_cluster_id
                overlap_new_cluster = new_cluster_mask & overlap_mask

                if np.any(overlap_new_cluster):
                    overlap_indices = chunk_indices[overlap_new_cluster]
                    existing_labels = labels[overlap_indices]
                    existing_labels = existing_labels[existing_labels >= 0]

                    if len(existing_labels) > 0:
                        most_common = np.bincount(existing_labels).argmax()
                        chunk_labels[new_cluster_mask] = most_common
                    else:
                        # New cluster
                        chunk_labels[new_cluster_mask] = global_cluster_id
                        global_cluster_id += 1
                else:
                    chunk_labels[new_cluster_mask] = global_cluster_id
                    global_cluster_id += 1

            labels[chunk_indices[~overlap_mask]] = chunk_labels[
                ~overlap_mask
            ]  # Stops chunk boundaries being set to noise

    valid_labels = labels[labels >= 0]
    unique_clusters = np.unique(valid_labels)
    n_clusters = len(unique_clusters)
    n_noise_total = np.sum(labels == -1)

    # REPORT
    logger.info("\n")
    logger.info("CLUSTERING COMPLETE")
    logger.info(f"Total clusters: {n_clusters}")
    logger.info(
        f"Total noise points: {n_noise_total:,} ({100*n_noise_total/len(labels):.2f}%)"
    )
    logger.info(
        f"Total clustered points: {np.sum(labels >= 0):,} ({100*np.sum(labels >= 0)/len(labels):.2f}%)\n"
    )

    clustered_data = np.column_stack([clustered_data, labels])

    unique_layers = np.unique(clustered_data[:, 0])
    for layer in unique_layers:
        layer_mask = clustered_data[:, 0] == layer
        layer_labels = labels[layer_mask]
        n_noise = np.sum(layer_labels == -1)
        n_clusters_in_layer = len(set(layer_labels[layer_labels >= 0]))
        logger.info(
            f"  Layer {int(layer)}: {n_clusters_in_layer} clusters, {n_noise:,} noise points"
        )
        valid_clusters = labels[labels >= 0]

    if len(valid_clusters) > 0:
        unique_clusters, cluster_counts = np.unique(valid_clusters, return_counts=True)

        logger.info(" ")
        logger.info(f"Mean cluster size: {cluster_counts.mean():,} points")
        logger.info(f"Median cluster size: {np.median(cluster_counts):,} points")
        logger.info(f"Largest cluster: {cluster_counts.max():,} points")
        logger.info(f"Smallest cluster: {cluster_counts.min():,} points\n")

    if visualize:
        viz_downsample = max(1, len(coords_3d) // 50000)  # Plot ~50k points
        viz_coords = coords_3d[::viz_downsample]
        viz_labels = labels[::viz_downsample]

        fig = go.Figure()

        noise_mask = viz_labels == -1
        cluster_mask = viz_labels >= 0

        if np.any(noise_mask):
            fig.add_trace(
                go.Scatter3d(
                    x=viz_coords[noise_mask, 0],
                    y=viz_coords[noise_mask, 1],
                    z=viz_coords[noise_mask, 2],
                    mode="markers",
                    marker=dict(size=1, color="black", opacity=0.3),
                    name="Noise",
                    hovertemplate="<b>Noise</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
                )
            )

        if np.any(cluster_mask):
            fig.add_trace(
                go.Scatter3d(
                    x=viz_coords[cluster_mask, 0],
                    y=viz_coords[cluster_mask, 1],
                    z=viz_coords[cluster_mask, 2],
                    mode="markers",
                    marker=dict(
                        size=1,
                        color=viz_labels[cluster_mask],
                        colorscale="Spectral",
                        opacity=0.6,
                        colorbar=dict(title="Cluster ID", thickness=15),
                    ),
                    name=" ",
                    hovertemplate="<b>Cluster %{marker.color}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
                )
            )

        fig.update_layout(
            title=f"Cluster Map of AMPM Data<br>{n_clusters} clusters, {len(coords_3d):,} total points",
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title=f"Z (Layer * {layer_spacing})",
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=1),  # Force 1:1:1
            ),
            width=1280,
            height=720,
            showlegend=True,
        )

        fig.show()

    return clustered_data


def assign_parts(
    clustered_data: np.ndarray,
    parts_df: pd.DataFrame,
    save_file: bool = True,
) -> np.ndarray:
    """
    Use spatial matching to reassign ClusterIDs to associated Part IDs from parts data.

    Parameters
    ----------
    clustered_data : np.ndarray
        Output from cluster_data().
        8 columns: [Layer, Time, Dwell, X, Y, Plasma, Meltpool, ClusterID].
    parts_df : pd.DataFrame
        Output from get_parts()
        Columns: [Part ID, Layer Thickness, X position, Y Position, Layers count].
    save_file : bool, optional
        Set to False to not generate .npy of part-assigned array (default: True).
    verbose : bool, optional
        Set to False to silence console output (default: True).

    Returns
    -------
    relabeled_data : np.ndarray
        Data with ClusterID column replaced by Part ID.
        8 columns: [Layer, Time, Dwell, X, Y, Plasma, Meltpool, Part ID].
    """
    cluster_labels = clustered_data[:, -1].astype(int)

    unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
    n_clusters = len(unique_clusters)
    n_parts = len(parts_df)

    logger.info("ASSIGNING PARTS TO CLUSTERS...")
    logger.info(f"Found {n_clusters} clusters")
    logger.info(f"Found {n_parts} parts in DataFrame")

    if n_clusters != n_parts:
        logger.error(
            f"MISMATCH: Number of clusters ({n_clusters}) does not match number of parts ({n_parts})\n"
        )
        raise ValueError(
            f"Cluster-Part mismatch: {n_clusters} clusters vs {n_parts} parts"
        )

    cluster_to_part_map = {}

    for idx, part_row in parts_df.iterrows():
        part_id = part_row["Part ID"]
        part_x = float(part_row["X Position"])
        part_y = float(part_row["Y Position"])

        # Find which cluster contains this part's (X, Y) position
        matched_cluster = None

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_points = clustered_data[cluster_mask]

            cluster_x = cluster_points[:, 3]
            cluster_y = cluster_points[:, 4]

            x_min, x_max = cluster_x.min(), cluster_x.max()
            y_min, y_max = cluster_y.min(), cluster_y.max()

            if x_min <= part_x <= x_max and y_min <= part_y <= y_max:
                matched_cluster = cluster_id
                break

        if matched_cluster is None:
            logger.warning(
                f"  Part {part_id} at ({part_x:.2f}, {part_y:.2f}) does not fall within any cluster bounds"
            )
        else:
            cluster_to_part_map[matched_cluster] = part_id
            logger.info(f"  Cluster {matched_cluster} → Part {part_id}")

    if len(cluster_to_part_map) != n_clusters:
        logger.error(
            f"MAPPING ERROR: Only {len(cluster_to_part_map)} of {n_clusters} clusters were mapped to parts\n"
        )
        raise ValueError

    new_labels = cluster_labels.copy()
    for cluster_id, part_id in cluster_to_part_map.items():
        new_labels[cluster_labels == cluster_id] = part_id

    relabeled_data = clustered_data.copy()
    relabeled_data[:, -1] = new_labels

    logger.info("\n")
    logger.info("ASSIGNING COMPLETE")
    logger.info(f"Successfully remapped {n_clusters} clusters to Part IDs\n")

    if save_file:
        logger.info("Saving part assigned data...")
        np.save("ampm_part_assigned.npy", relabeled_data)
        logger.info("Saved to: ampm_part_assigned.npy")
        logger.info("  Columns: Layer, Time, Dwell, X, Y, Plasma, Meltpool, PartID\n")

    return relabeled_data


def assign_density(archi_data, parts_data) -> pd.DataFrame:
    """
    Add density column to parts_data by looking up values from archi_data.

    Parameters
    ----------
    archi_data : pd.DataFrame
        Columns: [Speed (mm/s), Power (W), Density AVG (g/cm^3)].
    parts_data : pd.DataFrame
        Output from get_parts().
        Columns: [Part ID, Hatch Power, Hatch Point Distance, Hatch Exposure Time].

    Returns
    -------
    parts_w_density : pd.DataFrame
        parts_data with added columns: [Hatch Speed, Density (g/cm^3)].
    """
    required_cols = ["Hatch Power", "Hatch Point Distance", "Hatch Exposure Time"]
    missing_cols = [col for col in required_cols if col not in parts_data.columns]

    if missing_cols:
        raise ValueError(f"Missing parametric data in parts DataFrame: {missing_cols}")

    parts_w_density = parts_data.copy()

    parts_w_density["Hatch Speed"] = (
        parts_w_density["Hatch Point Distance"].astype(float)
        / parts_w_density["Hatch Exposure Time"].astype(float)
    ) * 1000

    densities = []
    for idx, row in parts_w_density.iterrows():
        hatch_speed = row["Hatch Speed"]
        hatch_power = float(row["Hatch Power"])

        density_row = archi_data[
            (archi_data["Speed (mm/s)"] == hatch_speed)
            & (archi_data["Power (W)"] == hatch_power)
        ]

        if len(density_row) > 0:
            density = density_row["Density AVG (g/cm^3)"].values[0]
        else:
            density = np.nan
            print(
                f"Warning: No density found for Part {row['Part ID']} (Speed={hatch_speed:.2f}, Power={hatch_power:.2f})"
            )

        densities.append(density)

    parts_w_density["Density (g/cm^3)"] = densities

    return parts_w_density


def cov_by_part(data) -> pd.DataFrame:
    """
    Uses ampm part-assigned data to calculate CoV for each part.
    Calculates statistics for both Plasma and Meltpool diodes.

    Parameters
    ----------
    data : np.ndarray
        Output from assign_parts().
        8 columns: Layer, Time, Dwell, X, Y, Plasma, Meltpool, PartID.

    Returns
    -------
    cov_df : pd.DataFrame
        7 columns: [Part ID, Plasma Mean, Plasma Std, Plasma CoV, Meltpool Mean, Meltpool Std, Meltpool CoV].
    """
    unique_parts = np.unique(data[:, -1])

    results = []
    for part_id in unique_parts:
        part_mask = data[:, -1] == part_id

        plasma_values = data[part_mask, 5]  # Plasma
        plasma_mean = np.mean(plasma_values)
        plasma_std = np.std(plasma_values)
        plasma_cov = plasma_std / plasma_mean

        meltpool_values = data[part_mask, 6]  # Meltpool
        meltpool_mean = np.mean(meltpool_values)
        meltpool_std = np.std(meltpool_values)
        meltpool_cov = meltpool_std / meltpool_mean

        results.append(
            {
                "Part ID": int(part_id),
                "Plasma Mean": plasma_mean,
                "Plasma Std": plasma_std,
                "Plasma CoV": plasma_cov,
                "Meltpool Mean": meltpool_mean,
                "Meltpool Std": meltpool_std,
                "Meltpool CoV": meltpool_cov,
            }
        )

    cov_df = pd.DataFrame(results)

    return cov_df


def assign_cov(cov_table, parts_data) -> pd.DataFrame:
    """
    Add CoV columns to parts_data by merging with cov_table.

    Parameters
    ----------
    cov_table : pd.DataFrame
        Output from cov_by_part().
        4 columns: ['Part ID', '<Diode> CoV', '<Diode> Mean', '<Diode> Std'].
    parts_data : pd.DataFrame
        Output from get_parts().
        Columns: [Part ID, Layer Thickness, X Position, Y Position, Layers Count].

    Returns
    -------
    parts_w_cov : pd.DataFrame
        parts_data with added calculated columns.
        Columns: [Part ID, Layer Thickness, X Position, Y Position, Layers Count, '<Diode> CoV', '<Diode> Mean', '<Diode> Std'].
    """
    parts_w_cov = parts_data.copy()  # ADD ERROR CHECK FOR BAD COV TABLE OR PART DATA
    cov_table_copy = cov_table.copy()

    parts_w_cov["Part ID"] = parts_w_cov["Part ID"].astype(str)
    cov_table_copy["Part ID"] = cov_table_copy["Part ID"].astype(str)

    parts_w_cov = parts_w_cov.merge(cov_table_copy, on="Part ID", how="left")

    return parts_w_cov


def find_neighbors(
    data: list[np.ndarray], k: int = 5, layer_spacing: float = 0.03
) -> None:
    """
    Plot k-distance graph to help determine optimal eps parameter for DBSCAN.

    The k-distance graph shows the distance to the k-th nearest neighbor for each point.
    The "knee" in the curve suggests a good eps value.

    Parameters
    ----------
    data : list[np.ndarray]
        Output from import_ampm_data().
        7 columns: [Layer, Time, Dwell, X, Y, Plasma, Meltpool].
    k : int, optional
        Number of nearest neighbors to consider (should match min_samples in DBSCAN) (default: 50).
    layer_spacing : float, optional
        Z coords = layer number * layer spacing (default: 0.03).

    Returns
    -------
    None
        Displays interactive plotly graph of k-distance.
    """
    logger.info("COMPUTING K-DISTANCE GRAPH...")
    logger.info(f"k-neighbors: {k}")

    all_data = np.vstack(data)

    coords_3d = np.column_stack(
        [all_data[:, 3], all_data[:, 4], all_data[:, 0] * layer_spacing]  # X  # Y  # Z
    )

    logger.info(f"Total points: {len(coords_3d):,}")

    logger.info("Fitting NearestNeighbors model...")
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(coords_3d)

    logger.info(
        f"  Computing distances to {k}th nearest neighbor for each point (this may take some time)...\n"
    )
    distances, indices = nbrs.kneighbors(coords_3d)

    k_distances = distances[:, -1]

    k_distances_sorted = np.sort(k_distances)

    logger.info("K-DISTANCE COMPLETE")
    logger.info(
        f"Distance range: [{k_distances_sorted.min():.4f}, {k_distances_sorted.max():.4f}]"
    )
    logger.info(f"Median distance: {np.median(k_distances_sorted):.4f}")
    logger.info(f"Mean distance: {np.mean(k_distances_sorted):.4f}\n")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(k_distances_sorted)),
            y=k_distances_sorted,
            mode="lines",
            line=dict(color="blue", width=1),
            name=f"{k}-distance",
        )
    )

    fig.add_hline(
        y=np.median(k_distances_sorted),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Median: {np.median(k_distances_sorted):.4f}",
        annotation_position="right",
    )

    fig.update_layout(
        title=f"K-Distance Graph (k={k})",
        xaxis_title="Points (sorted by distance)",
        yaxis_title=f"Distance to {k}th Nearest Neighbor",
        width=1280,
        height=720,
        hovermode="closest",
    )

    fig.show()


def analyze_parts_distribution(data, plot_parts=None, parts_data=None) -> None:
    """
    Analyze the distribution of Plasma and Meltpool columns by Part ID.

    Parameters
    ----------
    data : numpy.ndarray
        2D array with shape (n_rows, 8)
        Columns: [Layer, Time, Dwell, X, Y, Plasma, Meltpool, Part ID]
    plot_parts : list[int], optional
        List of Part IDs to plot. If None, plots all parts.
    parts_data : pandas.DataFrame, optional
        DataFrame containing part parameters.
        Must have "Part ID" and parametric data for enhanced legend display.

    Returns
    -------
    None
        Displays interactive plotly graph of density plots.
    """
    PLASMA_COL = 5
    MELTPOOL_COL = 6
    PART_ID_COL = 7

    if parts_data is not None:
        if "Part ID" not in parts_data.columns:
            raise ValueError(f"Error! No column 'Part ID' in parts_data:\n{parts_data}")

        if "Hatch Speed" not in parts_data.columns:
            if (
                "Hatch Point Distance" in parts_data.columns
                and "Hatch Exposure Time" in parts_data.columns
            ):
                parts_data = parts_data.copy()
                parts_data["Hatch Speed"] = (
                    parts_data["Hatch Point Distance"]
                    / parts_data["Hatch Exposure Time"]
                ) * 1000
            else:
                raise ValueError(
                    f"Warning! No parametric data in parts_data:\n{parts_data}"
                )

        if "Hatch Power" not in parts_data.columns:
            raise ValueError(
                f"Warning! No column 'Hatch Power' in parts_data:\n{parts_data}"
            )

    unique_parts = np.unique(data[:, PART_ID_COL])

    if plot_parts is not None:
        plot_parts = np.array(plot_parts, dtype=int)
        unique_parts_int = unique_parts.astype(int)
        mask = np.isin(unique_parts_int, plot_parts)
        unique_parts = unique_parts[mask]
        if len(unique_parts) == 0:
            raise ValueError(
                f"No valid Part IDs found. Requested: {plot_parts}, Available: {unique_parts_int}"
            )

    n_parts = len(unique_parts)

    logger.info(f"PLOTTING {n_parts} PART IDS: {unique_parts}...\n")

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Plasma Distribution",
            "Meltpool Distribution",
            "Joint Distribution",
        ),
        horizontal_spacing=0.1,
    )

    for idx, part_id in enumerate(unique_parts):
        mask = data[:, PART_ID_COL] == part_id
        filtered_data = data[mask]

        if len(filtered_data) == 0:
            continue

        plasma = filtered_data[:, PLASMA_COL]
        meltpool = filtered_data[:, MELTPOOL_COL]

        valid_mask = np.isfinite(plasma) & np.isfinite(meltpool)
        plasma = plasma[valid_mask]
        meltpool = meltpool[valid_mask]

        if len(plasma) == 0:
            continue

        color = colors[idx % len(colors)]

        # Part ID -1 = Noise
        if int(part_id) == -1:
            label = "Noise"
            color = "#000000"
        else:
            label = f"Part {int(part_id)}"

        if parts_data is not None and int(part_id) != -1:
            part_row = parts_data[parts_data["Part ID"] == int(part_id)]
            if len(part_row) == 0 and parts_data["Part ID"].dtype == "object":
                part_row = parts_data[parts_data["Part ID"].astype(int) == int(part_id)]

            if len(part_row) > 0:
                hatch_speed = part_row["Hatch Speed"].iloc[0]
                hatch_power = part_row["Hatch Power"].iloc[0]
                label = f"P{int(part_id)}: {hatch_speed:.0f}mm/s, {hatch_power:.0f}W"

        regression = linregress(plasma, meltpool)
        slope = regression[0]
        r_value = regression[2]
        r_squared = r_value**2

        label_with_fit = f"{label} (m={slope:.2f}, R²={r_squared:.3f})"

        # Plasma density
        kde_plasma = gaussian_kde(plasma)
        x_plasma = np.linspace(plasma.min(), plasma.max(), 200)
        y_plasma = kde_plasma(x_plasma)

        fig.add_trace(
            go.Scatter(
                x=x_plasma,
                y=y_plasma,
                mode="lines",
                name=label_with_fit,
                line=dict(color=color, width=2),
                fill="tozeroy",
                opacity=0.1,
                legendgroup=label,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Meltpool density
        kde_meltpool = gaussian_kde(meltpool)
        x_meltpool = np.linspace(meltpool.min(), meltpool.max(), 200)
        y_meltpool = kde_meltpool(x_meltpool)

        fig.add_trace(
            go.Scatter(
                x=x_meltpool,
                y=y_meltpool,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
                fill="tozeroy",
                opacity=0.1,
                legendgroup=label,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Joint scatter plot
        sample_plasma = plasma
        sample_meltpool = meltpool
        if len(plasma) > 200:
            sample_idx = np.random.choice(len(plasma), 200, replace=False)
            sample_plasma = plasma[sample_idx]
            sample_meltpool = meltpool[sample_idx]

        fig.add_trace(
            go.Scatter(
                x=sample_plasma,
                y=sample_meltpool,
                mode="markers",
                name=label_with_fit,
                marker=dict(color=color, size=4, opacity=0.2),
                legendgroup=label,
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        if int(part_id) == -1:
            logger.info(f"Noise: {int(part_id)} ({len(plasma)} points)")
        else:
            logger.info(f"Part ID: {int(part_id)} ({len(plasma)} points)")

        if parts_data is not None and int(part_id) != -1:
            part_row = parts_data[parts_data["Part ID"] == int(part_id)]
            if len(part_row) == 0 and parts_data["Part ID"].dtype == "object":
                part_row = parts_data[parts_data["Part ID"].astype(int) == int(part_id)]

            if len(part_row) > 0:
                hatch_speed_val = part_row["Hatch Speed"].iloc[0]
                hatch_power_val = part_row["Hatch Power"].iloc[0]
                logger.info(
                    f"  Speed: {hatch_speed_val:.0f}mm/s, Power: {hatch_power_val:.0f}W"
                )

        logger.info(
            f"  Plasma  - Mean: {plasma.mean():.4f}, Std: {plasma.std():.4f}, CoV: {plasma.std()/plasma.mean():.4f}"
        )
        logger.info(
            f"  Meltpool - Mean: {meltpool.mean():.4f}, Std: {meltpool.std():.4f}, CoV: {meltpool.std()/meltpool.mean():.4f}"
        )
        logger.info(f"  Linear fit - Slope: {slope:.4f}, R²: {r_squared:.4f}\n")

    fig.update_xaxes(
        title_text="Plasma", row=1, col=1, showgrid=True, gridcolor="lightgray"
    )
    fig.update_yaxes(
        title_text="Density", row=1, col=1, showgrid=True, gridcolor="lightgray"
    )

    fig.update_xaxes(
        title_text="Meltpool", row=1, col=2, showgrid=True, gridcolor="lightgray"
    )
    fig.update_yaxes(
        title_text="Density", row=1, col=2, showgrid=True, gridcolor="lightgray"
    )

    fig.update_xaxes(
        title_text="Plasma",
        row=1,
        col=3,
        showgrid=True,
        gridcolor="lightgray",
        dtick=None,
        minor=dict(showgrid=True, gridcolor="lightgray", griddash="dot"),
    )
    fig.update_yaxes(
        title_text="Meltpool",
        row=1,
        col=3,
        showgrid=True,
        gridcolor="lightgray",
        dtick=None,
        minor=dict(showgrid=True, gridcolor="lightgray", griddash="dot"),
    )

    try:
        root = tk.Tk()
        screen_width = (
            root.winfo_screenwidth()
        )  # Probably better to make a config at some point
        root.destroy()
        fig_width = min(int(screen_width * 0.9), 1800)  # 90% of screen width
    except Exception:
        fig_width = 1400

    fig.update_layout(
        height=500,
        width=fig_width,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        hovermode="closest",
        template="plotly_white",
    )

    logger.info("\n")
    logger.info(f"SUCCESSFULLY PLOTTED {n_parts} parts\n")

    fig.show()


# Example usage:
if __name__ == "__main__":

    filepath = (
        Path.cwd()
        / "JR265_AMPM"
        / "output"
        / "Spectral.ExportPackets.20251001-123012.386 (579d0101-046b-4142-bce7-5d3f82867967)"
    )

    try:
        roi_data = import_ampm_data(
            filepath=filepath,
            x_min=-125,
            x_max=125,
            y_min=-125,
            y_max=125,
            start_layer=200,
            end_layer=205,
            return_dict=False,
        )

        print(f"Number of layers processed: {len(roi_data)}")
        if roi_data:
            print(f"First layer shape: {roi_data[0].shape}")
            print("Column order: [layer, time, duration, x, y, plasma, meltpool]")

    except Exception as e:
        print(f"Error: {e}")
