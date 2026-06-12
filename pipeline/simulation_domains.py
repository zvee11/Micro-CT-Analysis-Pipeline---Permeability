from __future__ import annotations

import numpy as np


DOMAIN_FLUID_VALUE = np.uint8(0)    # 0 = fluid/open
DOMAIN_BLOCKED_VALUE = np.uint8(1)  # 1 = solid/blocked


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
