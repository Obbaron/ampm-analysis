from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import logging


logger = logging.getLogger(__name__)

console_logger = logging.StreamHandler()
console_logger.setLevel(logging.INFO)
console_logger.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(console_logger)

file_logger = logging.FileHandler('clustering.log', mode='a')
file_logger.setLevel(logging.INFO)
file_logger.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(file_logger)

logger.setLevel(logging.INFO)
logger.propagate = False


def import_ampm_data(filepath: str | Path,
                     start_layer: int,
                     end_layer: int,
                     x_min: float = -125.0,
                     x_max: float = 125.0,
                     y_min: float = -125.0,
                     y_max: float = 125.0,
                     laser_number: int = 4,
                     verbose: bool = True,
                     return_dict: bool = False
                    ) -> list[np.ndarray] | dict:
    """
    Import AMPM data from region of interest: [Layer, Time, Dwell, X, Y, Plasma, Meltpool]
    
    Parameters:
    -----------
    filepath : str or Path
        Directory path containing the data files
    start_layer : int
        First layer number to process   
    end_layer : int
        Last layer number to process (inclusive)
    x_min : float, optional
        Lower x-coordinate boundary for ROI (mm) (default: -125.0)
    x_max : float, optional
        Upper x-coordinate boundary for ROI (mm) (default: 125.0)
    y_min : float, optional
        Lower y-coordinate boundary for ROI (mm) (default: -125.0)
    y_max : float, optional
        Upper y-coordinate boundary for ROI (mm) (default: 125.0)
    laser_number : int, optional
        Laser number in filename (default: 4)
    verbose : bool, optional
        Set to False to silence console output (default: True)
    return_dict : bool, optional
        If True, return dict with layer number as keys (default: False)
    
    Returns:
    --------
    list or dict
        List of numpy arrays (dict if return_dict=True), one array per layer
        Each array has 7 columns: [layer, time, duration, x, y, plasma, meltpool]
    """
    
    console_logger = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)][0]
    if verbose:
        console_logger.setLevel(logging.INFO)
    else:
        console_logger.setLevel(logging.ERROR)
    
    filepath = Path(filepath) if type(filepath) == str else filepath
    if not filepath.exists():
        raise FileNotFoundError(f"Directory not found: {filepath}")
    if x_min >= x_max:
        raise ValueError(f"x_min ({x_min}) must be < x_max ({x_max})")
    if y_min >= y_max:
        raise ValueError(f"y_min ({y_min}) must be < y_max ({y_max})")
    if start_layer > end_layer:
        raise ValueError(f"start_layer ({start_layer}) must be <= end_layer ({end_layer})")
    if start_layer < 0:
        raise ValueError(f"start_layer must be non-negative, got {start_layer}")
    
    
    num_layers = end_layer - start_layer + 1
    roi_data = {} if return_dict else []
    
    logger.info(f"IMPORTING {num_layers} LAYERS...")
    
    for idx, j in enumerate(range(start_layer, end_layer + 1)):
        logger.info(f"Importing layer {j} ({idx + 1}/{num_layers})")
        
        filename = f'Packet data for layer {j}, laser {laser_number}.txt'
        full_path = filepath / filename
        
        if not full_path.exists():
            logger.warning(f"File not found: {filename}, skipping...")
            continue
        
        try:
            layer_data = pd.read_csv(
                full_path,
                sep='\t',
                usecols=([0, 1, 2, 3, 5, 6]), # Skips LaserVIEW
                on_bad_lines='warn'
            )
            
            layer_array = layer_data.to_numpy()
            
            locations = (
                (layer_array[:, 2] > x_min) & 
                (layer_array[:, 2] < x_max) &
                (layer_array[:, 3] > y_min) & 
                (layer_array[:, 3] < y_max)
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


def get_parts(filepath : str | Path, parametric : bool = False) -> pd.DataFrame:
    """
    Create a pandas DataFrame from parts data exported from QuantAM (.csv)

    Parameters:
    -----------
    filepath : str or Path
        Filepath to parts data file (.csv)
    parametric : bool, optional
        Include Hatch Power, Hatch Point Distance, Hatch Exposure Time (default: False)

    Returns:
    --------
    pd.DataFrame
        DataFrame of [Part ID, Layer Thickness, X Position, Y Position, Layers Count]
        Optionally include [Hatch Power, Hatch Point Distance, Hatch Exposure Time]
    """

    filepath = Path(filepath) if type(filepath) == str else filepath
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    parts_data = pd.read_csv(filepath,
                    usecols=[1,3,4,5,6],
                    names=['Part ID','Layer Thickness','X Position','Y Position','Layers Count'],
                    skiprows=6,
                    on_bad_lines='skip',
                    skip_blank_lines=True
                    )
    
    if 'Tab - 10' in parts_data['Part ID'].values:
        parametric_possible = True
    else:
        parametric_possible = False
        
    parts_idx = []
    for value in parts_data['Part ID']:
        if pd.isna(value):
            break
        if isinstance(value, str) and value.startswith('Tab - '): # Exports from QuantAM go straight to next tab
            break
        parts_idx.append(int(value)-1)

    parts_data = parts_data.loc[parts_idx]
    
    if parametric and not parametric_possible:
        raise ValueError(f'No parametric data found in file: {filepath}')
        
    # Count number of parts, use as spacer to jump to Tab 10, get parametric data
    elif parametric and parametric_possible:
        skipped_rows = len(parts_idx) + 10

        first_col = pd.read_csv(filepath,
                            usecols=[1],
                            names=['Part ID'],
                            skiprows=skipped_rows,
                            on_bad_lines='skip',
                            skip_blank_lines=True
                            )

        full_parts_list = []
        for value in first_col['Part ID']:
            if pd.isna(value):
                break
            if isinstance(value, str) and value.startswith('Tab - '):
                break
            if isinstance(value, str):
                base_value = value.replace('.1', '').replace('.s', '')
                if base_value in parts_data['Part ID'].values:
                    full_parts_list.append(value)

        if 'Tab - 10' and 'Tab - 11' in first_col['Part ID'].values:
            skipped_rows = 9 * (len(full_parts_list) + 4) + (len(parts_idx) + 10) 

        parts_params = pd.read_csv(filepath,
                            usecols=[7,9,10],
                            names=['Power','Point Distance','Exposure Time'],
                            skiprows=skipped_rows,
                            nrows=(len(parts_idx)),
                            on_bad_lines='skip',
                            skip_blank_lines=True
                            )
        
        parts_data = pd.concat([parts_data,parts_params],axis=1)
        
        return parts_data
    
    else:
        return parts_data


def cluster_data(data : list[np.ndarray],
                 eps_xy : float = 0.135,
                 eps_z : float = 0.135,
                 min_samples : int = 50,
                 layers_per_chunk : int = 10,
                 overlap_layers : int = 2,
                 layer_spacing : float = 0.03,
                 verbose : bool = True,
                 visualize : bool = False
                 ) -> np.ndarray:
    """
    Groups AMPM data into clusters using a chunked 3D DBSCAN
    
    Parameters:
    -----------
    data : list[np.ndarray]
        Output from import_data with columns: Layer, Time, Dwell, X, Y, Plasma, Meltpool, ClusterID
    eps_xy : float, optional
        Max XY distance (mm) between two samples for one to be considered as in the neighborhood of the other  (default: 0.10)
    eps_z : float, optional
        Effective 3D eps = sqrt(eps_xy**2 + eps_z**2) (default: 0.10)
    min_samples : int, optional
        Number of samples in a neighborhood for a point to be considered a core point (default: 20)
    layers_per_chunk : int, optional
        Layers are processed in chunks to limit memory requirements (default: 10)
    overlap_layers : int, optional
        Number of overlapping layers at chunk boundary to maintain Z continuity (default: 2)
    layer_spacing : float, optional
        Z coords = layer number * layer spacing (default: 0.03)
    verbose : bool, optional
        Set to False to silence console output (default: True)
    visualize : bool, optional
        Set to True to see a decimated 3D scatter plot after cluserting (default: False)
        
    Returns:
    --------
    np.ndarray
        Data with ClusterID labels (noise = -1)
    """
    
    console_logger = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)][0]
    if verbose:
        console_logger.setLevel(logging.INFO)
    else:
        console_logger.setLevel(logging.ERROR)
    
    all_data = np.vstack(data)

    coords_3d = np.column_stack([
        all_data[:, 3],  # X
        all_data[:, 4],  # Y
        all_data[:, 0] * layer_spacing # Z
    ])

    logger.info("INITIALIZING CLUSTERING...")
    logger.info(f"Total data points: {coords_3d.shape[0]}")
    logger.info(f"Coordinate ranges:")
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

    unique_layers = np.unique(all_data[:, 0])
    n_layers = len(unique_layers)

    logger.info(f"Processing {n_layers} layers in chunks of {layers_per_chunk} with {overlap_layers} layer overlap...\n")
    chunk_num = 0

    labels = np.full(len(all_data), -1, dtype=int)
    global_cluster_id = 0

    for chunk_start in range(0, n_layers, layers_per_chunk - overlap_layers):
        chunk_end = min(chunk_start + layers_per_chunk, n_layers)
        chunk_layers = unique_layers[chunk_start:chunk_end]
        
        if chunk_start > 0 and chunk_end <= chunk_start + overlap_layers:
            logger.info(f"Skipping redundant chunk: Layers {int(chunk_layers[0])}-{int(chunk_layers[-1])} (entirely overlap)")
            continue
        
        chunk_mask = np.isin(all_data[:, 0], chunk_layers)
        chunk_coords = coords_3d[chunk_mask]
        chunk_indices = np.where(chunk_mask)[0]
        chunk_num += 1
        
        logger.info(f"Chunk {chunk_num}: Layers {int(chunk_layers[0])}-{int(chunk_layers[-1])} ({len(chunk_coords):,} points)...")
        
        clustering = DBSCAN(eps=eps_3d, min_samples=min_samples, n_jobs=-1)
        chunk_labels = clustering.fit_predict(chunk_coords)
        
        n_clusters_chunk = len(set(chunk_labels)) - (1 if -1 in chunk_labels else 0)
        n_noise_chunk = list(chunk_labels).count(-1)
        
        logger.info(f"  Found {n_clusters_chunk} clusters, {n_noise_chunk:,} noise points")
        
        if chunk_start == 0:
            # First chunk: assign directly
            chunk_labels[chunk_labels >= 0] += global_cluster_id
            labels[chunk_indices] = chunk_labels
            global_cluster_id += n_clusters_chunk
        else:
            # Overlapping chunk: merge with previous clusters
            overlap_layer_start = chunk_layers[0]
            overlap_layer_end = chunk_layers[min(overlap_layers - 1, len(chunk_layers) - 1)]
            
            overlap_mask = (all_data[chunk_indices, 0] >= overlap_layer_start) & \
                        (all_data[chunk_indices, 0] <= overlap_layer_end)
            
            logger.info(f"  Overlap region: layers {int(overlap_layer_start)}-{int(overlap_layer_end)}")
            logger.info(f"  Points in overlap: {overlap_mask.sum():,}")
            logger.info(f"  Points in non-overlap: {(~overlap_mask).sum():,}")
            
            # Match clusters in overlap region
            for new_cluster_id in range(n_clusters_chunk):
                new_cluster_mask = (chunk_labels == new_cluster_id)
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
            
            labels[chunk_indices[~overlap_mask]] = chunk_labels[~overlap_mask] # Stops chunk boundaries being set to noise

    valid_labels = labels[labels >= 0]
    unique_clusters = np.unique(valid_labels)
    n_clusters = len(unique_clusters)
    n_noise_total = np.sum(labels == -1)

    # REPORT
    logger.info("\n")
    logger.info("CLUSTERING COMPLETE")
    logger.info(f"Total clusters: {n_clusters}")
    logger.info(f"Total noise points: {n_noise_total:,} ({100*n_noise_total/len(labels):.2f}%)")
    logger.info(f"Total clustered points: {np.sum(labels >= 0):,} ({100*np.sum(labels >= 0)/len(labels):.2f}%)\n")

    all_data = np.column_stack([all_data, labels])
    
    unique_layers = np.unique(all_data[:,0])
    for layer in unique_layers:
        layer_mask = all_data[:,0] == layer
        layer_labels = labels[layer_mask]
        n_noise = np.sum(layer_labels == -1)
        n_clusters_in_layer = len(set(layer_labels[layer_labels >= 0]))
        logger.info(f"  Layer {int(layer)}: {n_clusters_in_layer} clusters, {n_noise:,} noise points")
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
            fig.add_trace(go.Scatter3d(
                x=viz_coords[noise_mask, 0],
                y=viz_coords[noise_mask, 1],
                z=viz_coords[noise_mask, 2],
                mode='markers',
                marker=dict(size=1, color='black', opacity=0.3),
                name='Noise',
                hovertemplate='<b>Noise</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            ))

        if np.any(cluster_mask):
            fig.add_trace(go.Scatter3d(
                x=viz_coords[cluster_mask, 0],
                y=viz_coords[cluster_mask, 1],
                z=viz_coords[cluster_mask, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color=viz_labels[cluster_mask],
                    colorscale='Spectral',
                    opacity=0.6,
                    colorbar=dict(title="Cluster ID", thickness=15)
                ),
                name=' ',
                hovertemplate='<b>Cluster %{marker.color}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            ))

        fig.update_layout(
            title=f'Cluster Map of AMPM Data<br>{n_clusters} clusters, {len(coords_3d):,} total points',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title=f'Z (Layer * {layer_spacing})',
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1)  # Force 1:1:1
            ),
            width=1280,
            height=720,
            showlegend=True
        )

        fig.show()
    
    return all_data


def assign_parts(clustered_data : np.ndarray,
                 parts_df : pd.DataFrame,
                 save_file : bool = True,
                 verbose : bool = True) -> np.ndarray:
    """
    Reassign ClusterIDs to PartIDs based on spatial matching.
    
    Parameters:
    -----------
    clustered_data : np.ndarray
        Output from cluster_data() with columns: Layer, Time, Dwell, X, Y, Plasma, Meltpool, ClusterID
    parts_df : pd.DataFrame
        Output from get_parts() with columns: Part ID, Layer Thickness, X position, Y Position, Layers count
    save_file : bool, optional
        Set to False to not generate .npy of part-assigned array (default: True)
    verbose : bool, optional
        Set to False to silence console output (default: True)
        
    Returns:
    --------
    np.ndarray
        Data with ClusterID column replaced by Part ID
    """
    
    console_logger = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)][0]
    if verbose:
        console_logger.setLevel(logging.INFO)
    else:
        console_logger.setLevel(logging.ERROR)
    
    cluster_labels = clustered_data[:, -1].astype(int)
    
    unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
    n_clusters = len(unique_clusters)
    n_parts = len(parts_df)
    
    logger.info(f"ASSIGNING PARTS TO CLUSTERS...")
    logger.info(f"Found {n_clusters} clusters")
    logger.info(f"Found {n_parts} parts in DataFrame")
    
    if n_clusters != n_parts:
        logger.error(f"MISMATCH: Number of clusters ({n_clusters}) does not match number of parts ({n_parts})")
        raise ValueError(f"Cluster-Part mismatch: {n_clusters} clusters vs {n_parts} parts")
    
    cluster_to_part_map = {}
    
    for idx, part_row in parts_df.iterrows():
        part_id = part_row['Part ID']
        part_x = float(part_row['X Position'])
        part_y = float(part_row['Y Position'])
        
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
            logger.warning(f"  Part {part_id} at ({part_x:.2f}, {part_y:.2f}) does not fall within any cluster bounds")
        else:
            cluster_to_part_map[matched_cluster] = part_id
            logger.info(f"  Cluster {matched_cluster} → Part {part_id}")
    
    if len(cluster_to_part_map) != n_clusters:
        logger.error(f"MAPPING ERROR: Only {len(cluster_to_part_map)} of {n_clusters} clusters were mapped to parts\n")
        raise ValueError(f"Could not map all clusters to parts")
    
    new_labels = cluster_labels.copy()
    for cluster_id, part_id in cluster_to_part_map.items():
        new_labels[cluster_labels == cluster_id] = part_id
    
    result_data = clustered_data.copy()
    result_data[:, -1] = new_labels
    
    logger.info("\n")
    logger.info("ASSIGNING COMPLETE")
    logger.info(f"Successfully remapped {n_clusters} clusters to Part IDs\n")
    
    if save_file:
        logger.info("Saving part assigned data...")
        np.save('ampm_part_assigned.npy', result_data)
        logger.info("Saved to: ampm_part_assigned.npy")
        logger.info("  Columns: Layer, Time, Dwell, X, Y, Plasma, Meltpool, PartID\n")
    
    return result_data


def find_neighbors(data: list[np.ndarray], k: int = 5, layer_spacing: float = 0.03) -> None:
    """
    Plot k-distance graph to help determine optimal eps parameter for DBSCAN
    
    The k-distance graph shows the distance to the k-th nearest neighbor for each point
    The "knee" in the curve suggests a good eps value
    
    Parameters:
    -----------
    data : list[np.ndarray]
        Output from import_ampm_data with columns: Layer, Time, Dwell, X, Y, Plasma, Meltpool
    k : int, optional
        Number of nearest neighbors to consider (should match min_samples in DBSCAN) (default: 50)
    layer_spacing : float, optional
        Z coords = layer number * layer spacing (default: 0.03)
        
    Returns:
    --------
    None
        Displays interactive plotly graph
    """
    
    logger.info("COMPUTING K-DISTANCE GRAPH...")
    logger.info(f"k-neighbors: {k}")
    
    all_data = np.vstack(data)
    
    coords_3d = np.column_stack([
        all_data[:, 3],  # X
        all_data[:, 4],  # Y
        all_data[:, 0] * layer_spacing  # Z
    ])
    
    logger.info(f"Total points: {len(coords_3d):,}")
    
    logger.info("Fitting NearestNeighbors model...")
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(coords_3d)
    
    logger.info(f"  Computing distances to {k}th nearest neighbor for each point (this may take some time)...\n")
    distances, indices = nbrs.kneighbors(coords_3d)
    
    k_distances = distances[:, -1]
    
    k_distances_sorted = np.sort(k_distances)
    
    logger.info("K-DISTANCE COMPLETE")
    logger.info(f"Distance range: [{k_distances_sorted.min():.4f}, {k_distances_sorted.max():.4f}]")
    logger.info(f"Median distance: {np.median(k_distances_sorted):.4f}")
    logger.info(f"Mean distance: {np.mean(k_distances_sorted):.4f}\n")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=np.arange(len(k_distances_sorted)),
        y=k_distances_sorted,
        mode='lines',
        line=dict(color='blue', width=1),
        name=f'{k}-distance'
    ))
    
    fig.add_hline(
        y=np.median(k_distances_sorted),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Median: {np.median(k_distances_sorted):.4f}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=f'K-Distance Graph (k={k})',
        xaxis_title='Points (sorted by distance)',
        yaxis_title=f'Distance to {k}th Nearest Neighbor',
        width=1280,
        height=720,
        hovermode='closest'
    )
    
    fig.show()


# Example usage:
if __name__ == "__main__":
    
    filepath = Path.cwd()/'JR265_AMPM'/'Spectral.ExportPackets.20251001-123012.386 (579d0101-046b-4142-bce7-5d3f82867967)'

    try:
        roi_data = import_ampm_data(
            filepath=filepath,
            x_min=-125,
            x_max=125,
            y_min=-125,
            y_max=125,
            start_layer=200,
            end_layer=205,
            return_dict=False
        )
        
        print(f"Number of layers processed: {len(roi_data)}")
        if roi_data:
            print(f"First layer shape: {roi_data[0].shape}")
            print(f"Column order: [layer, time, duration, x, y, plasma, meltpool]")
            
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
