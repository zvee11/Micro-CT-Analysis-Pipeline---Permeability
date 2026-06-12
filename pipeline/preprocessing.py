from __future__ import annotations

import logging

import numpy as np


def find_flow_crop_z(mask: np.ndarray, threshold_fraction: float = 0.10) -> tuple[int, int]:
    """Half-open Z bounds where per-slice area >= threshold_fraction of peak slice."""
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}")
    slice_counts = (mask > 0).sum(axis=(1, 2)).astype(np.int64)
    peak = int(slice_counts.max())
    if peak == 0:
        raise ValueError("mask contains no non-zero voxels")
    qualifying = slice_counts >= peak * threshold_fraction
    if not qualifying.any():
        raise ValueError(f"No Z-slice meets threshold fraction {threshold_fraction} of peak {peak}")
    z_in = int(np.argmax(qualifying))
    z_out = int(mask.shape[0]) - int(np.argmax(qualifying[::-1]))
    return z_in, z_out


def compute_cluster_cog(mask: np.ndarray) -> tuple[float, float, float]:
    """Centre of gravity (z, y, x) of a binary cluster mask, scanned slice-by-slice."""
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}")
    sum_z = sum_y = sum_x = 0.0
    total = 0
    for z in range(mask.shape[0]):
        sl = mask[z] > 0
        n = int(sl.sum())
        if n == 0:
            continue
        ys, xs = np.where(sl)
        sum_z += z * n
        sum_y += float(ys.sum())
        sum_x += float(xs.sum())
        total += n
    if total == 0:
        raise ValueError("mask contains no non-zero voxels")
    return (sum_z / total, sum_y / total, sum_x / total)


def detect_and_fix_x_wraparound(vol: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """Detect X-axis wrap-around (offset extraction window) and roll it back in-place."""
    logger.info("Detecting X wrap-around...")
    x_density = np.zeros(vol.shape[2], dtype=np.int64)
    for z in range(vol.shape[0]):
        x_density += (vol[z] > 0).sum(axis=0)

    threshold = x_density.max() * 0.05
    above = x_density > threshold
    sliver_end = int(np.argmax(~above))
    if sliver_end == 0:
        logger.info("No wrap-around detected — skipping correction.")
        return vol

    gap_end = sliver_end + int(np.argmax(above[sliver_end:]))
    shift = gap_end
    logger.info("Wrap-around detected: sliver x=0..%d, gap x=%d..%d, shift=%d px",
                sliver_end - 1, sliver_end, gap_end - 1, shift)

    # Roll left by `shift` along X in Z-slabs so peak extra memory stays ~one slab.
    nz = vol.shape[0]
    slab = 256
    saved = vol[:, :, :shift].copy()
    for z0 in range(0, nz, slab):
        z1 = min(z0 + slab, nz)
        block = vol[z0:z1, :, shift:].copy()
        vol[z0:z1, :, :-shift] = block
        vol[z0:z1, :, -shift:] = saved[z0:z1]
    del saved
    logger.info("X wrap-around corrected (chunked in-place, slab=%d).", slab)
    return vol


def log_volume_info(vol: np.ndarray, spacing, info: dict, logger: logging.Logger,
                    hist: dict[int, int] | None = None) -> None:
    logger.info("shape (Z,Y,X): %s", vol.shape)
    logger.info("dtype: %s | codec: %s", vol.dtype, info["codec"])
    logger.info("value range: %s - %s", vol.min(), vol.max())
    if spacing is not None:
        logger.info("spacing (µm): %.3f, %.3f, %.3f",
                    spacing[0] * 1e6, spacing[1] * 1e6, spacing[2] * 1e6)
    if hist is not None:
        for label, count in hist.items():
            logger.info("label %3d: %d voxels", int(label), int(count))
