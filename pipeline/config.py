from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("data")
    out_dir: Path = Path("output")
    input_glob: str = "*.am"

    gas_label: int = 2
    n_keep: int = 6
    slab_depth: int = 128
    connectivities: tuple[int, ...] = (2,)

    crop_margin: int = 10
    inlet_outlet_threshold: float = 0.20  # min slice area as fraction of peak Z-slice
    track_clusters: bool = True

    regime_cutoff: str = "transition"      # "transition" (default) | "displacement"
    min_scans_three_segment: int = 6

    crop_mode: str = "fixed"

    enable_ui: bool = True
    enable_pyvista: bool = True
    pyvista_downsample: int = 4
    enable_dash: bool = False
    dash_port: int = 8050

    saturation_file: str = "data/18_Sg_3d.xlsx"  # empty to skip reference comparison
    saturation_name_col: int = 0
    saturation_sg_col: int = 4
    saturation_time_col: int = 8


CC3D_CONNECTIVITY = {1: 6, 2: 18, 3: 26}
CONNECTIVITY_NAME = {1: "6N", 2: "18N", 3: "26N"}
