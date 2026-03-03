from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from ampm import import_ampm_data


data_directory = (
    Path.cwd()
    / "Spectral.ExportPackets.20251119-122056.059 (4915479c-081b-423c-9764-21950405961b)"
)
parts_file = Path.cwd() / "JR293_Ti7Cu.csv"


data_list = import_ampm_data(
    filepath=data_directory,
    x_min=-125,
    x_max=125,
    y_min=-125,
    y_max=125,
    start_layer=101,
    end_layer=150,
    return_dict=False,
)

all_data = np.vstack(data_list)

coords_3d = np.column_stack(
    [all_data[:, 3], all_data[:, 4], all_data[:, 0] * 0.01]  # X  # Y  # Layer * 0.01
)

print(f"Total data points: {coords_3d.shape[0]}")
print("Coordinate ranges:")
print(f"  X: [{coords_3d[:, 0].min():.2f}, {coords_3d[:, 0].max():.2f}]")
print(f"  Y: [{coords_3d[:, 1].min():.2f}, {coords_3d[:, 1].max():.2f}]")
print(f"  Z: [{coords_3d[:, 2].min():.2f}, {coords_3d[:, 2].max():.2f}]")


n_clusters = 15  # Adjust based on expected number of features/regions
downsample_factor = 50  # Adjust based on memory: 10-100

coords_sampled = coords_3d[::downsample_factor]
print(f"\nDownsampled to: {coords_sampled.shape[0]} points for clustering")
print("\nPerforming MiniBatchKMeans clustering...")

clustering = MiniBatchKMeans(
    n_clusters=n_clusters, batch_size=5000, random_state=42, verbose=1, max_iter=100
)
sample_labels = clustering.fit_predict(coords_sampled)

print(f"\nClustering complete: {n_clusters} clusters")

print("\nPredicting labels for all points (in batches)...")
batch_size = 100000
labels = np.zeros(len(coords_3d), dtype=int)

for i in range(0, len(coords_3d), batch_size):
    end_idx = min(i + batch_size, len(coords_3d))
    labels[i:end_idx] = clustering.predict(coords_3d[i:end_idx])
    if (i // batch_size + 1) % 10 == 0:
        print(f"  Processed {end_idx}/{len(coords_3d)} points")

print("Label prediction complete")

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

viz_downsample = max(1, len(coords_3d) // 50000)  # Max 50k points for vizualization
viz_coords = coords_3d[::viz_downsample]
viz_labels = labels[::viz_downsample]

print(f"\nVisualizing {len(viz_coords)} points...")

colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters))

scatter = ax.scatter(
    viz_coords[:, 0],
    viz_coords[:, 1],
    viz_coords[:, 2],
    c=viz_labels,
    cmap="Spectral",
    s=1,
    alpha=0.6,
)

ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (Layer)")
ax.set_title(
    f"Mini K Means Clustering of AMPM Data\n{n_clusters} clusters, {len(coords_3d):,} total points"
)

plt.colorbar(scatter, ax=ax, label="Cluster ID", shrink=0.5)
plt.tight_layout()
plt.show()

print("\nSaving cluster assignments...")
clustered_data = np.column_stack([all_data, labels])
np.save("ampm_clustered_data.npy", clustered_data)
print("Saved to: ampm_clustered_data.npy")

print("\nCluster statistics:")
unique, counts = np.unique(labels, return_counts=True)
for cluster_id, count in zip(unique[:10], counts[:10]):  # Show first 10
    print(f"  Cluster {cluster_id}: {count:,} points ({100*count/len(labels):.2f}%)")
if len(unique) > 10:
    print(f"  ... and {len(unique)-10} more clusters")
