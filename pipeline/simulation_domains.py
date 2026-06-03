from __future__ import annotations

import numpy as np


DOMAIN_FLUID_VALUE = np.uint8(0)
DOMAIN_BLOCKED_VALUE = np.uint8(1)


def _require_3d(name: str, arr: np.ndarray) -> None:
    if arr.ndim != 3:
        raise ValueError(f"{name} must be a 3D array, got shape {arr.shape}")


def make_absolute_domain(volume_crop: np.ndarray) -> np.ndarray:
    """
    Build an absolute-permeability domain from a cropped labelled volume.

    Output convention:
        0 = fluid/open
        1 = solid/blocked

    Open labels:
        1 = water/brine
        2 = gas

    Blocked labels:
        0 = rock/background
        10, 64, and any other non-1/non-2 value = artifact/blocked
    """
    _require_3d("volume_crop", volume_crop)
    return np.where((volume_crop == 1) | (volume_crop == 2), DOMAIN_FLUID_VALUE, DOMAIN_BLOCKED_VALUE).astype(np.uint8)


def make_gas_domain(volume_crop: np.ndarray, cluster_mask: np.ndarray) -> np.ndarray:
    """
    Build a gas effective-permeability domain for one selected connected gas cluster.

    Output convention:
        0 = fluid/open
        1 = solid/blocked

    Open voxels:
        cluster_mask > 0

    Blocked voxels:
        water, rock/background, artifacts, and gas voxels outside the selected cluster.

    `volume_crop` is accepted and shape-checked to prevent accidentally pairing
    a mask with the wrong cropped labelled volume.
    """
    _require_3d("volume_crop", volume_crop)
    _require_3d("cluster_mask", cluster_mask)
    if volume_crop.shape != cluster_mask.shape:
        raise ValueError(f"Shape mismatch: volume_crop {volume_crop.shape}, cluster_mask {cluster_mask.shape}")

    return np.where(cluster_mask > 0, DOMAIN_FLUID_VALUE, DOMAIN_BLOCKED_VALUE).astype(np.uint8)


def make_water_domain(volume_crop: np.ndarray) -> np.ndarray:
    """
    Build a water effective-permeability domain from a cropped labelled volume.

    Output convention:
        0 = fluid/open
        1 = solid/blocked

    Open labels:
        1 = water/brine

    Blocked labels:
        0 = rock/background
        2 = gas
        10, 64, and any other non-1 value = artifact/blocked
    """
    _require_3d("volume_crop", volume_crop)
    return np.where(volume_crop == 1, DOMAIN_FLUID_VALUE, DOMAIN_BLOCKED_VALUE).astype(np.uint8)
