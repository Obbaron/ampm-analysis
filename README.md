# ampm-replacement

Python tools for importing, masking, and correcting AMPM process monitoring data from the Renishaw 500S metal 3D printer.

## Installation

Clone the repository and install the required dependencies:

```bash
pip install numpy pandas shapely trimesh networkx scipy
```

## Usage

### 1. Import

Import AMPM layer data from a directory of exported packet files. A rectangular region of interest (ROI) can be specified to limit the data to a specific area of the build plate.

```python
from ampmdata import import_ampm_data

data = import_ampm_data(
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

Mask the data to only retain points that fall within the part geometry, using the full plate STL file exported from QuantAM.

```python
from ampmdata import mask_ampm_data

data = mask_ampm_data(
    roi_data=data,
    stl_path="path/to/fullplate.stl",
    layer_thickness=0.03,  # optional, default 0.03mm
)
```

### 3. Correct

Apply a LaserVIEW-based XY positional correction to the MeltPool signal to account for optical sensitivity variation across the build plate.

> **Note:** This correction is only applicable to data from the **main Renishaw 500S machine**. Do NOT apply it to RBV machine data!

```python
from ampmdata import correct_ampm_data

data = correct_ampm_data(data)
```

### Full pipeline

```python
from ampmdata import import_ampm_data, mask_ampm_data, correct_ampm_data

data = import_ampm_data("path/to/ampm/export/packets", start_layer=1, end_layer=100)
data = mask_ampm_data(data, "path/to/fullplate.stl")
data = correct_ampm_data(data)  # main machine only

print(f"Imported {len(data)} layers")
```
