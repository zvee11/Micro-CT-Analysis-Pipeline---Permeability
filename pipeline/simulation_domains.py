from __future__ import annotations

import numpy as np


DOMAIN_FLUID_VALUE = np.uint8(0)    # 0 = fluid/open
DOMAIN_BLOCKED_VALUE = np.uint8(1)  # 1 = solid/blocked


def domain_filename(dtype: str, track_id: int, scan_index: int,
                    spacing, shape_zyx) -> str:
    """Build the GeoDict-batch domain filename, identical for X and Step B so the
    downstream orchestrator can group the three domains of a (track, scan).

        domain_{dtype}_{track:02d}-{scan}_{voxel}um_8bu_{NX}x{NY}x{NZ}.raw
    e.g. domain_gas_03-7_4.99684um_8bu_750x750x523.raw

    dtype     : 'absolute' | 'gas' | 'water'
    spacing   : (dz, dy, dx) in metres from the .am header; voxel token is dx in
                micrometres (the in-plane resolution, constant across domains).
    shape_zyx : (NZ, NY, NX) of the written domain volume.
    Bits are hard-coded 8bu (domains are uint8 binary masks).
    """
    nz, ny, nx = shape_zyx
    vox_um = (spacing[2] * 1e6) if spacing else 5.0   # dx -> micrometres
    vox_tok = f"{vox_um:g}"                            # trims trailing zeros
    return (f"domain_{dtype}_{track_id:02d}-{scan_index}_"
            f"{vox_tok}um_8bu_{nx}x{ny}x{nz}.raw")


def _require_3d(name: str, arr: np.ndarray) -> None:
    if arr.ndim != 3:
        raise ValueError(f"{name} must be a 3D array, got shape {arr.shape}")


def make_absolute_domain(volume_crop: np.ndarray) -> np.ndarray:
    """Open where label is 1 (brine) or 2 (gas); blocked otherwise."""
    _require_3d("volume_crop", volume_crop)
    return np.where((volume_crop == 1) | (volume_crop == 2), DOMAIN_FLUID_VALUE, DOMAIN_BLOCKED_VALUE).astype(np.uint8)


def make_gas_domain(volume_crop: np.ndarray, cluster_mask: np.ndarray) -> np.ndarray:
    """Open where cluster_mask > 0; blocked otherwise."""
    _require_3d("volume_crop", volume_crop)
    _require_3d("cluster_mask", cluster_mask)
    if volume_crop.shape != cluster_mask.shape:
        raise ValueError(f"Shape mismatch: volume_crop {volume_crop.shape}, cluster_mask {cluster_mask.shape}")
    return np.where(cluster_mask > 0, DOMAIN_FLUID_VALUE, DOMAIN_BLOCKED_VALUE).astype(np.uint8)


def make_water_domain(volume_crop: np.ndarray) -> np.ndarray:
    """Open where label is 1 (brine); blocked otherwise."""
    _require_3d("volume_crop", volume_crop)
    return np.where(volume_crop == 1, DOMAIN_FLUID_VALUE, DOMAIN_BLOCKED_VALUE).astype(np.uint8)
