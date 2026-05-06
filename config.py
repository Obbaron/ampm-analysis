"""
Shared configuration for ampm scripts

Paths and physical parameters that apply across the analysis pipeline live
here so all scripts stay in sync. Per-script tuning (cluster eps, signal
choices, plot settings) stays in the script that uses it.
"""

from pathlib import Path

SOURCE = r"C:\Users\ohp460\Documents\Code\ampm-data\AMPM_data\JR314_14GHz_model_Fi_bool_Supp_cube\[3] Export Packets"
STL = r"C:\Users\ohp460\Documents\Code\ampm-data\AMPM_data\JR314_14GHz_model_Fi_bool_Supp_cube\JR314_14GHz_fullplate.stl"
PARTS_CSV = r"C:\Users\ohp460\Documents\Code\ampm-data\AMPM_data\JR314_14GHz_model_Fi_bool_Supp_cube\JR314_14GHz_cube_parts.csv"

MASK_CACHE = str(Path(SOURCE) / ".cache" / "fullplate_mask.pkl")
MASK_KEEP_CACHE = str(Path(SOURCE) / ".cache" / "mask_keep.pq")
CLUSTER_CACHE = str(Path(SOURCE) / ".cache" / "cluster_labels.pq")

LAYER_THICKNESS = 0.03  # mm
