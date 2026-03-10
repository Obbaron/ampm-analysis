# ampmdata

Python tools for importing, masking, and correcting AMPM process monitoring data from the Renishaw 500S metal 3D printer.

## Installation

Clone the repository and install the required dependencies:

```bash
pip install numpy pandas shapely trimesh networkx scipy h5py
```

## Usage

### 1. Import

Import AMPM layer data from a directory of exported packet files. A rectangular region of interest (ROI) can be specified to limit the data to a specific area of the build plate.

```python
from ampmdata import AMPMData

data = AMPMData.from_directory(
    filepath="path/to/ampm/export/packets",
    start_layer=1,
    end_layer=100,
    x_min=-125.0,   # optional, default ±125mm
    x_max=125.0,
    y_min=-125.0,
    y_max=125.0,
    laser_number=4, # optional, default 4
)
```

Each layer is stored as a numpy array with 7 columns: `[Time, Dwell, X, Y, LaserVIEW, Plasma, Meltpool]`.

### 2. Mask

Mask the data in-place to only retain points that fall within the part geometry, using the full plate STL file exported directly from the Renishaw machine.

```python
data.mask(
    stl_path="path/to/fullplate.stl",
    layer_thickness=0.03,  # optional, default 0.03mm
)
```

### 3. Correct

Apply a LaserVIEW-based XY positional correction to the MeltPool signal in-place, to account for optical sensitivity variation across the build plate.

> **Note:** This correction is only applicable to data from the **main Renishaw 500S machine**. Do not apply it to RBV machine data.

```python
data.correct()
```

### Full pipeline

```python
from ampmdata import AMPMData

data = AMPMData.from_directory("path/to/ampm/export/packets", start_layer=1, end_layer=100)
data.mask("path/to/fullplate.stl")
data.correct()  # main machine only

print(f"Imported {len(data)} layers")
```

### Saving and loading

Masking is the most expensive step. Save the masked state to an HDF5 file to skip it on subsequent runs:

```python
# First run — mask and save
data = AMPMData.from_directory("path/to/ampm/export/packets", start_layer=1, end_layer=100)
data.mask("path/to/fullplate.stl")
data.save("masked_data.h5")

# Subsequent runs — load directly
data = AMPMData.load("masked_data.h5")
data.correct()
```

The `.h5` file can also be read directly in MATLAB using `h5read`.