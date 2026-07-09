from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("data")
    out_dir: Path = Path("output")
    input_glob: str = "*.am"

    gas_label: int = 2
    n_keep: int = 8
    slab_depth: int = 0      # 0 = auto-size from available RAM (see __post_init__)
    connectivities: tuple[int, ...] = (2,)

    prepass_workers: int = 8  # 1 = serial (laptop-safe); >1 = parallel decode in pre-pass

    crop_margin: int = 10
    inlet_outlet_threshold: float = 0.20  # min slice area as fraction of peak Z-slice
    track_clusters: bool = True

    regime_cutoff: str = "displacement"      # "transition" (default) | "displacement"
    min_scans_three_segment: int = 3
    interactive_regime: bool = True        # prompt user to confirm/override the
                                           # regime cutoff; False skips the prompt

    crop_mode: str = "fixed"

    enable_ui: bool = True
    enable_pyvista: bool = True
    pyvista_downsample: int = 2
    enable_dash: bool = False
    dash_port: int = 8050

    saturation_file: str = "data/Sg_3d_H2_19.xlsx"  # empty to skip reference comparison
    saturation_name_col: int = 0
    saturation_sg_col: int = 8
    saturation_time_col: int = 4

    def __post_init__(self):
        # Auto-size the CC slab depth from available RAM when left at 0.
        # Bigger slabs = fewer passes = faster, but need more RAM for the int32
        # label array. The meaningful win is reaching one pass (slab >= full Z),
        # which removes boundary-stitching entirely; mid tiers use RAM smoothly.
        #   < 24 GB free  -> 128   (laptop floor, proven safe)
        #   24-48 GB      -> 256
        #   48-96 GB      -> 512
        #   96-192 GB     -> 1024
        #   >= 192 GB     -> 4096  (>= full Z of 3780, single-pass cc3d)
        if self.slab_depth == 0:
            depth = 128
            try:
                import psutil
                free_gb = psutil.virtual_memory().available / (1024 ** 3)
                if free_gb >= 192:
                    depth = 4096
                elif free_gb >= 96:
                    depth = 1024
                elif free_gb >= 48:
                    depth = 512
                elif free_gb >= 24:
                    depth = 256
                else:
                    depth = 128
                print(f"[config] auto slab_depth={depth} (free RAM ~{free_gb:.0f} GB)")
            except Exception:
                print("[config] psutil not available — using default slab_depth=128")
                depth = 128
            object.__setattr__(self, "slab_depth", depth)


CC3D_CONNECTIVITY = {1: 6, 2: 18, 3: 26}
CONNECTIVITY_NAME = {1: "6N", 2: "18N", 3: "26N"}