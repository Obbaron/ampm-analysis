from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from ampm import import_ampm_data


data_directory = Path.cwd()/'JR265_AMPM'/'Spectral.ExportPackets.20251001-123012.386 (579d0101-046b-4142-bce7-5d3f82867967)'
parts_file = Path.cwd()/'JR265_AMPM'/'JR265_AMPM_parameters_all(fresh).csv'


def cluster_data(data : list[np.ndarray],
                 eps_xy : float = 0.25,
                 eps_z : float = 0.05,
                 min_samples : int = 50,
                 layers_per_chunk : int = 10,
                 overlap_layers : int = 2,
                 ):
    
    all_data = np.vstack(data)

    coords_3d = np.column_stack([
        all_data[:, 3],  # X
        all_data[:, 4],  # Y
        all_data[:, 0] * 0.01  # Layer * 0.01
    ])

    print(f"Total data points: {coords_3d.shape[0]}")
    print(f"Coordinate ranges:")
    print(f"  X: [{coords_3d[:, 0].min():.2f}, {coords_3d[:, 0].max():.2f}]")
    print(f"  Y: [{coords_3d[:, 1].min():.2f}, {coords_3d[:, 1].max():.2f}]")
    print(f"  Z: [{coords_3d[:, 2].min():.2f}, {coords_3d[:, 2].max():.2f}]")

    # CHUNKED 3D CLUSTERING
    print("\n" + "="*60)
    print("Performing chunked 3D clustering across all layers...")
    print("="*60)

    # DBSCAN parameters
    eps_xy = 0.25  # Distance threshold in XY plane (mm)
    eps_z = 0.05  # Distance threshold in Z direction (layer * 0.01)
    min_samples = 50  # Minimum points to form a cluster
    layers_per_chunk = 10
    overlap_layers = 2

    eps_3d = np.sqrt(eps_xy**2 + eps_z**2)

    print(f"\nClustering parameters:")
    print(f"  eps (XY): {eps_xy} mm")
    print(f"  eps (Z): {eps_z} (layer * 0.01)")
    print(f"  Effective 3D eps: {eps_3d:.3f}")
    print(f"  min_samples: {min_samples}")

    unique_layers = np.unique(all_data[:, 0])
    n_layers = len(unique_layers)

    print(f"\nProcessing {n_layers} layers in chunks of {layers_per_chunk} with {overlap_layers} layer overlap...")

    labels = np.full(len(all_data), -1, dtype=int)
    global_cluster_id = 0

    for chunk_start in range(0, n_layers, layers_per_chunk - overlap_layers):
        chunk_end = min(chunk_start + layers_per_chunk, n_layers)
        chunk_layers = unique_layers[chunk_start:chunk_end]
        
        if chunk_start > 0 and chunk_end <= chunk_start + overlap_layers:
            print(f"\nSkipping redundant chunk: Layers {int(chunk_layers[0])}-{int(chunk_layers[-1])} (entirely overlap)")
            continue
        
        chunk_mask = np.isin(all_data[:, 0], chunk_layers)
        chunk_coords = coords_3d[chunk_mask]
        chunk_indices = np.where(chunk_mask)[0]
        
        print(f"\nChunk: Layers {int(chunk_layers[0])}-{int(chunk_layers[-1])} ({len(chunk_coords):,} points)")
        
        # CLUSTER CHUNK
        clustering = DBSCAN(eps=eps_3d, min_samples=min_samples)
        chunk_labels = clustering.fit_predict(chunk_coords)
        
        n_clusters_chunk = len(set(chunk_labels)) - (1 if -1 in chunk_labels else 0)
        n_noise_chunk = list(chunk_labels).count(-1)
        
        print(f"  Found {n_clusters_chunk} clusters, {n_noise_chunk:,} noise points")
        
        # For non-overlapping region, assign new global cluster IDs
        if chunk_start == 0:
            # First chunk: assign directly
            chunk_labels[chunk_labels >= 0] += global_cluster_id
            labels[chunk_indices] = chunk_labels
            global_cluster_id += n_clusters_chunk
        else:
            # Overlapping chunk: need to merge with previous clusters
            overlap_layer_start = chunk_layers[0]
            overlap_layer_end = chunk_layers[min(overlap_layers - 1, len(chunk_layers) - 1)]
            
            overlap_mask = (all_data[chunk_indices, 0] >= overlap_layer_start) & \
                        (all_data[chunk_indices, 0] <= overlap_layer_end)
            
            print(f"  Overlap region: layers {int(overlap_layer_start)}-{int(overlap_layer_end)}")
            print(f"  Points in overlap: {overlap_mask.sum():,}")
            print(f"  Points in non-overlap: {(~overlap_mask).sum():,}")
            
            # Match clusters in overlap region
            for new_cluster_id in range(n_clusters_chunk):
                new_cluster_mask = (chunk_labels == new_cluster_id)
                overlap_new_cluster = new_cluster_mask & overlap_mask
                
                if np.any(overlap_new_cluster):
                    # Check which existing clusters overlap with this new cluster
                    overlap_indices = chunk_indices[overlap_new_cluster]
                    existing_labels = labels[overlap_indices]
                    existing_labels = existing_labels[existing_labels >= 0]
                    
                    if len(existing_labels) > 0:
                        # Merge with most common existing cluster
                        most_common = np.bincount(existing_labels).argmax()
                        chunk_labels[new_cluster_mask] = most_common
                        print(f"    Cluster {new_cluster_id}: merged with existing cluster {most_common}")
                    else:
                        # New cluster
                        chunk_labels[new_cluster_mask] = global_cluster_id
                        print(f"    Cluster {new_cluster_id}: assigned new ID {global_cluster_id} (in overlap but no existing labels)")
                        global_cluster_id += 1
                else:
                    # Cluster not in overlap, assign new ID
                    chunk_labels[new_cluster_mask] = global_cluster_id
                    print(f"    Cluster {new_cluster_id}: assigned new ID {global_cluster_id} (not in overlap)")
                    global_cluster_id += 1
            
            # Check for noise points (-1) in chunk_labels before update
            noise_before = np.sum(chunk_labels == -1)
            print(f"  Noise points in chunk after relabeling: {noise_before:,}")
            
            # Update labels for non-overlap region only
            non_overlap_mask = ~overlap_mask
            labels[chunk_indices[non_overlap_mask]] = chunk_labels[non_overlap_mask]
            
            non_overlap_labels = chunk_labels[non_overlap_mask]
            noise_in_non_overlap = np.sum(non_overlap_labels == -1)
            print(f"  Noise points written to non-overlap region: {noise_in_non_overlap:,}")

    # Renumber clusters to be consecutive
    valid_labels = labels[labels >= 0]
    unique_clusters = np.unique(valid_labels)
    cluster_map = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
    cluster_map[-1] = -1  # Keep noise as -1

    labels_renumbered = np.array([cluster_map[label] for label in labels])
    labels = labels_renumbered

    n_clusters = len(unique_clusters)
    n_noise_total = np.sum(labels == -1)

    print("\n" + "="*60)
    print(f"CLUSTERING COMPLETE")
    print("="*60)
    print(f"Total clusters: {n_clusters}")
    print(f"Total noise points: {n_noise_total:,} ({100*n_noise_total/len(labels):.2f}%)")
    print(f"Total clustered points: {np.sum(labels >= 0):,} ({100*np.sum(labels >= 0)/len(labels):.2f}%)")

    print("\nPreparing visualization...")
    viz_downsample = max(1, len(coords_3d) // 50000)  # Max 50k points
    viz_coords = coords_3d[::viz_downsample]
    viz_labels = labels[::viz_downsample]

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    print(f"Plotting {len(viz_coords):,} points...")

    noise_mask = viz_labels == -1
    cluster_mask = viz_labels >= 0

    if np.any(noise_mask):
        ax.scatter(viz_coords[noise_mask, 0], 
                viz_coords[noise_mask, 1], 
                viz_coords[noise_mask, 2],
                c='black', s=0.5, alpha=0.3, label='Noise')

    if np.any(cluster_mask):
        scatter = ax.scatter(viz_coords[cluster_mask, 0], 
                            viz_coords[cluster_mask, 1], 
                            viz_coords[cluster_mask, 2],
                            c=viz_labels[cluster_mask],
                            cmap='Spectral',
                            s=1,
                            alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Cluster ID', shrink=0.5)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (Layer * 0.01)')
    ax.set_title(f'DBSCAN of AMPM Data\n{n_clusters} clusters, {len(coords_3d):,} total points')

    plt.tight_layout()
    plt.show()

    # Save results
    print("\nSaving cluster assignments...")
    clustered_data = np.column_stack([all_data, labels])
    np.save('ampm_clustered_3d.npy', clustered_data)
    print("Saved to: ampm_clustered_3d.npy")
    print("Columns: Layer, Time, Dwell, X, Y, Plasma, Meltpool, ClusterID")

    # Load and analyze the saved file
    print("\n" + "="*60)
    print("ANALYZING SAVED FILE")
    print("="*60)
    loaded_data = np.load('ampm_clustered_3d.npy')
    loaded_labels = loaded_data[:, -1].astype(int)  # Last column is ClusterID

    print(f"Loaded {len(loaded_labels):,} total points")
    print(f"Noise points in loaded file: {np.sum(loaded_labels == -1):,}")
    print(f"\nUnique cluster IDs in file: {sorted(np.unique(loaded_labels))}")

    # Check noise by layer in loaded file
    print("\nNoise distribution by layer in saved file:")
    for layer in unique_layers:  # Check ALL layers
        layer_mask = loaded_data[:, 0] == layer
        layer_labels = loaded_labels[layer_mask]
        n_noise = np.sum(layer_labels == -1)
        if n_noise > 0:
            pct = 100 * n_noise / len(layer_labels)
            print(f"  Layer {int(layer)}: {n_noise:,} noise ({pct:.2f}%)")
        else:
            clusters_present = len(np.unique(layer_labels))
            print(f"  Layer {int(layer)}: 0 noise, {clusters_present} clusters")

    # Cluster statistics by layer
    print("\nCluster distribution across layers:")
    unique_layers = np.unique(all_data[:, 0])
    for layer in unique_layers[:5]:  # Show first 5 layers
        layer_mask = all_data[:, 0] == layer
        layer_labels = labels[layer_mask]
        n_clusters_in_layer = len(set(layer_labels[layer_labels >= 0]))
        print(f"  Layer {int(layer)}: {n_clusters_in_layer} clusters present")

    # Overall cluster size distribution
    print("\nCluster size distribution:")
    valid_clusters = labels[labels >= 0]
    if len(valid_clusters) > 0:
        unique_clusters, cluster_counts = np.unique(valid_clusters, return_counts=True)
        
        print(f"  Mean cluster size: {cluster_counts.mean():.0f} points")
        print(f"  Median cluster size: {np.median(cluster_counts):.0f} points")
        print(f"  Largest cluster: {cluster_counts.max():,} points")
        print(f"  Smallest cluster: {cluster_counts.min():,} points")
        
        print("\nTop 10 largest clusters:")
        top_indices = np.argsort(cluster_counts)[-10:][::-1]
        for idx in top_indices:
            cluster_id = unique_clusters[idx]
            count = cluster_counts[idx]
            cluster_mask = labels == cluster_id
            cluster_layers = all_data[cluster_mask, 0]
            layer_span = f"{int(cluster_layers.min())}-{int(cluster_layers.max())}"
            print(f"  Cluster {cluster_id}: {count:,} points, layers {layer_span}")


data_list = import_ampm_data(
    filepath=data_directory,
    start_layer=101,
    end_layer=150  # CHANGE TO 400 LATER
)
            
            
cluster_data(data_list)
