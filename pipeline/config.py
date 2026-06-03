from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("data")
    out_dir: Path = Path("output")

    # process all Avizo files in data_dir
    input_glob: str = "*.am"

    gas_label: int = 2
    n_keep: int = 5
    slab_depth: int = 128
    connectivities: tuple[int, ...] = (3,)

    memmap_raw: bool = False
    parse_spacing: bool = True
    crop_margin: int = 10

    # Save full-volume label array (.npy) — only needed for debugging
    # Not a simulation input — off by default
    save_labels_npy: bool = False

    # Inlet/outlet crop: minimum slice area as fraction of peak Z-slice voxel count
    inlet_outlet_threshold: float = 0.10

    # Cluster tracking across timesteps
    track_clusters: bool = True

    # Regime detection
    # "transition" — includes displacement + transition phases (default, matches paper)
    # "displacement" — conservative, displacement phase only
    regime_cutoff: str = "transition"

    # Minimum scans required for three-segment fit; below this falls back to two segments
    min_scans_three_segment: int = 6

    # Crop mode
    # "fixed"   — primary mode: full CC only on timestep X, frozen box applied to all earlier
    # "dynamic" — full CC pipeline every qualifying timestep, box recomputed each time
    crop_mode: str = "fixed"

    # Terminal UI (Rich)
    enable_ui: bool = True

    # 3D PyVista viewer
    enable_pyvista: bool = True
    pyvista_downsample: int = 8

    # Dash analytics dashboard
    enable_dash: bool = True
    dash_port: int = 8050

    # Reference saturation file (CSV or Excel)
    # Leave empty to skip reference comparison
    saturation_file: str = "data/18_Sg_3d.xlsx"

    # Column indices (0-based) in the reference file:
    #   name_col  — scan name / identifier
    #   sg_col    — gas saturation Sg (pipeline computes Sw = 1 - Sg)
    #   time_col  — elapsed time in minutes from experiment start
    # Adjust these to match your file layout.
    saturation_name_col: int = 0
    saturation_sg_col: int = 4
    saturation_time_col: int = 8


CC3D_CONNECTIVITY = {1: 6, 2: 18, 3: 26}
CONNECTIVITY_NAME = {1: "6N", 2: "18N", 3: "26N"}