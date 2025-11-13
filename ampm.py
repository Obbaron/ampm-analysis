import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def import_ampm_data(
    filepath: str | Path,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    start_layer: int,
    end_layer: int,
    laser_number: int = 4,
    verbose: bool = True,
    return_dict: bool = False
) -> list[np.ndarray] | dict:
    """
    Import data from ROI: Layer, Time, Dwell, X, Y, Plasma, Meltpool
    
    Parameters:
    -----------
    filepath : str or Path
        Directory path containing the data files
    x_min : float
        Lower x-coordinate boundary for ROI (mm)
    x_max : float
        Upper x-coordinate boundary for ROI (mm)
    y_min : float
        Lower y-coordinate boundary for ROI (mm)
    y_max : float
        Upper y-coordinate boundary for ROI (mm)
    start_layer : int
        First layer number to process
    end_layer : int
        Last layer number to process (inclusive)
    laser_number : int, optional
        Laser number in filename (default: 4)
    verbose : bool, optional
        Print progress information (default: True)
    return_dict : bool, optional
        If True, return dict with layer # as keys (default: False)
    
    Returns:
    --------
    list or dict
        List of numpy arrays (dict if return_dict=True), one array per layer
        Each array has 7 columns: [layer, time, duration, x, y, plasma, meltpool]
    """
    
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
    
    
    USECOLS = [0, 1, 2, 3, 5, 6] # Columns to use (skip LaserVIEW)
    num_layers = end_layer - start_layer + 1
    roi_data = {} if return_dict else []
    
    for idx, j in enumerate(range(start_layer, end_layer + 1)):
        if verbose:
            logger.info(f"Processing layer {j} ({idx + 1}/{num_layers})")
        
        filename = f'Packet data for layer {j}, laser {laser_number}.txt'
        full_path = filepath / filename
        
        if not full_path.exists():
            logger.warning(f"File not found: {filename}, skipping...")
            continue
        
        try:
            layer_data = pd.read_csv(
                full_path,
                sep='\t',
                usecols=USECOLS,
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
            
            if verbose:
                logger.info(f"  Found {locations.sum()} / {len(layer_array)} points in ROI")
            
            if return_dict:
                roi_data[j] = filtered_data
            else:
                roi_data.append(filtered_data)
                
        except Exception as e:
            logger.error(f"Error processing layer {j}: {str(e)}")
            raise
    
    if verbose:
        logger.info(f"Successfully processed {len(roi_data)} layers")
    
    return roi_data


def cluster_xy(data, layer=None, eps=0.25, min_samples=10, verbose=True):
    """
    Group nearby points using DBSCAN clustering algorithm.
    Adds cluster labels as 8th column of numpy array (noise: -1)
    
    Parameters:
    -----------
    data : dict
        Dict where key is layer # and value is numpy array w/ [n][7] dims
        Modified in-place to have [n][8] dims with cluster labels in column 7
    layer : int or None
        Layer number to cluster. If None, clusters all layers
    eps : float
        Maximum distance between two points to be considered in same cluster (default: 0.25)
    min_samples : int
        Minimum number of points to form a dense region (default: 10)
    verbose : bool
        Print progress information (default: True)
    
    Returns:
    --------
    results : dict
        Dictionary with layer numbers as keys and tuples of (n_clusters, n_noise) as values
    """
    results = {}
    
    if layer is not None:
        layers_to_process = [layer]
    else:
        layers_to_process = sorted(data.keys())
    
    logger.info(f"Processing {len(layers_to_process)} layer(s)...")
    
    for layer_num in layers_to_process:
        try:
            # X and Y are now in columns 3 and 4 (due to layer being column 0)
            cluster_labels = DBSCAN(eps=eps, min_samples=min_samples).fit(data[layer_num][:, 3:5]).labels_
            
            data[layer_num] = np.column_stack((data[layer_num], cluster_labels))
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            results[layer_num] = (n_clusters, n_noise)
        
            if verbose:
                logger.info(f"Layer {layer_num}: {n_clusters} clusters, {n_noise} noise points")
                
        except Exception as e:
            logger.error(f"Error processing layer {layer_num}: {str(e)}")
            raise
    
    logger.info(f"\nCompleted! Cluster labels added as column 7 (8th column)")
    
    return results


# Example usage:
if __name__ == "__main__":
    
    filepath = '/Users/guggoo/Documents/Programming/Matlab/Spectral.ExportPackets.20251001-123012.386 (579d0101-046b-4142-bce7-5d3f82867967)'

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
