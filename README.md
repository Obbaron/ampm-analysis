# ampm-data

A comprehensive tool for analyzing additive manufacturing process monitoring (AMPM) data with advanced clustering, visualization, and statistical analysis capabilities.

## Overview

`ampm-data` provides a robust framework for processing and analyzing large-scale additive manufacturing sensor data. The toolkit enables spatial clustering of process monitoring measurements, automatic part identification, and multi-faceted statistical analysis to support quality assessment and process optimization.

## Features

- **Automated Spatial Clustering**: Memory-efficient chunked processing for large datasets with automatic cluster merging across layer boundaries
- **Interactive 3D Visualization**: Plotly-based interactive scatter plots for exploring spatial distributions of clusters and measurements
- **Part Assignment**: Automatic mapping of spatial clusters to CAD part definitions based on geometric positioning
- **Parameter Optimization**: K-distance analysis tools to determine optimal clustering parameters
- **Statistical Analysis**: Calculate statistics (mean, standard deviation, coefficient of variation) for each part or cluster
- **Comprehensive Logging**: Detailed process logging to both console and file for reproducibility and debugging

## Installation

### Prerequisites

Ensure you have Python 3.8+ installed. Install required dependencies:

```bash
pip install scikit-learn numpy pandas plotly
```

### Setup

Clone the repository and ensure the `ampm.py` module is in your Python path:

```bash
git clone https://github.com/yourusername/ampm-data.git
cd ampm-data
```

## Usage

### Basic Workflow

```python
from pathlib import Path
from ampm import import_ampm_data, get_parts, cluster_data, assign_parts

# Define data paths
data_directory = Path('path/to/ampm/data')
parts_file = Path('path/to/parts.csv')

# Import AMPM data
data = import_ampm_data(
    filepath=data_directory,
    start_layer=101,
    end_layer=400
)

# Cluster the data
clustered_data = cluster_data(
    data,
    eps_xy=0.10,
    eps_z=0.10,
    min_samples=20,
    visualize=True
)

# Assign parts to clusters
parts_df = get_parts(parts_file)
final_data = assign_parts(clustered_data, parts_df)
```

### Finding Optimal Clustering Parameters

Use k-distance analysis to determine appropriate `eps` values:

```python
from ampm import find_neighbors

# Analyze k-distance graph
find_neighbors(data, k=20)

# Look for the "elbow" in the plot to determine optimal eps
```

### Analyzing Results

```python
import numpy as np
import pandas as pd

# Convert to DataFrame for analysis
df = pd.DataFrame(
    final_data,
    columns=['Layer', 'Time', 'Dwell', 'X', 'Y', 'Plasma', 'Meltpool', 'PartID']
)

# Calculate statistics per part
stats = df.groupby('PartID')['Meltpool'].agg(['mean', 'std', 'count'])
print(stats)
```

## Core Functions

### `cluster_data()`
Groups AMPM data points into spatial clusters using 3D clustering with memory-efficient chunked processing.

**Parameters:**
- `data`: List of numpy arrays from `import_ampm_data()`
- `eps_xy`: XY distance threshold (mm)
- `eps_z`: Z distance threshold (mm)
- `min_samples`: Minimum points per cluster
- `layer_spacing`: Z-axis scaling factor
- `verbose`: Enable/disable console logging
- `visualize`: Display interactive 3D plot

### `assign_parts()`
Maps spatial clusters to part IDs based on CAD part positions.

**Parameters:**
- `clustered_data`: Output from `cluster_data()`
- `parts_df`: DataFrame with part definitions (Part ID, X Position, Y Position, etc.)

### `find_neighbors()`
Generates k-distance graph for determining optimal clustering parameters.

**Parameters:**
- `data`: List of numpy arrays from `import_ampm_data()`
- `k`: Number of nearest neighbors (typically equals `min_samples`)
- `layer_spacing`: Z-axis scaling factor

## Data Format

### Input Data
AMPM data arrays should contain columns: `[Layer, Time, Dwell, X, Y, Plasma, Meltpool]`

### Parts Definition
Parts CSV should contain: `[Part ID, Layer Thickness, X Position, Y Position, Layers count]`

### Output Data
Clustered data includes: `[Layer, Time, Dwell, X, Y, Plasma, Meltpool, ClusterID/PartID]`

## Logging

All operations are logged to both console and `clustering.log` with timestamps. Control console verbosity with the `verbose` parameter:

```python
# Silent operation (logs only to file)
clustered_data = cluster_data(data, verbose=False)
```

## Future Development

- Coefficient of variation analysis
- Additional statistical metrics
- Anomaly detection capabilities
- Export functionality for analysis results

## Contributing

This is an internal research tool. For questions or suggestions, please contact the development team.

## License

Internal use only.
