from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np


class Bounds3D(NamedTuple):
    """Half-open bounds in NumPy order: z0:z1, y0:y1, x0:x1."""

    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int


def remove_ignored_labels(
    vol: np.ndarray,
    ignored_labels: tuple[int, ...] = (10, 64),
    replacement: int = 0,
) -> np.ndarray:
    """
    Return a volume where ignored labels are replaced by `replacement`.

    This function does not mutate the input volume. That is intentional because
    Avizo volumes may be read-only memmaps or decoded buffers.
    """
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {vol.shape}")
    if not ignored_labels:
        return np.array(vol, copy=True)

    cleaned = np.array(vol, copy=True)
    for label in ignored_labels:
        cleaned[cleaned == label] = replacement
    return cleaned


def compute_valid_phase_bbox(
    vol: np.ndarray,
    phase_labels: tuple[int, ...] = (1, 2),
    margin: int = 10,
) -> Bounds3D:
    """
    Compute a half-open bounding box around selected phase labels.

    The scan is slice-wise to avoid constructing a full-volume coordinate list
    with np.where(valid), which can become very large for 3D µCT volumes.
    """
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {vol.shape}")
    if margin < 0:
        raise ValueError("margin must be >= 0")
    if not phase_labels:
        raise ValueError("phase_labels must contain at least one label")

    z_dim, y_dim, x_dim = vol.shape
    z0 = y0 = x0 = None
    z1 = y1 = x1 = None

    for z in range(z_dim):
        sl = vol[z]
        mask = np.zeros(sl.shape, dtype=bool)
        for label in phase_labels:
            mask |= sl == label

        if not mask.any():
            continue

        ys, xs = np.where(mask)
        z0 = z if z0 is None else z0
        z1 = z + 1
        y_min = int(ys.min())
        y_max = int(ys.max()) + 1
        x_min = int(xs.min())
        x_max = int(xs.max()) + 1

        y0 = y_min if y0 is None else min(y0, y_min)
        y1 = y_max if y1 is None else max(y1, y_max)
        x0 = x_min if x0 is None else min(x0, x_min)
        x1 = x_max if x1 is None else max(x1, x_max)

    if z0 is None or y0 is None or x0 is None:
        raise ValueError(f"No voxels found for phase labels {phase_labels}")

    return Bounds3D(
        max(int(z0) - margin, 0),
        min(int(z1) + margin, z_dim),
        max(int(y0) - margin, 0),
        min(int(y1) + margin, y_dim),
        max(int(x0) - margin, 0),
        min(int(x1) + margin, x_dim),
    )


def crop_volume(vol: np.ndarray, bounds: Bounds3D | tuple[int, int, int, int, int, int]) -> np.ndarray:
    """Crop a 3D volume using half-open Z/Y/X bounds."""
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {vol.shape}")

    b = Bounds3D(*bounds)
    z_dim, y_dim, x_dim = vol.shape
    if not (0 <= b.z0 < b.z1 <= z_dim and 0 <= b.y0 < b.y1 <= y_dim and 0 <= b.x0 < b.x1 <= x_dim):
        raise ValueError(f"Invalid bounds {b} for volume shape {vol.shape}")

    return np.array(vol[b.z0:b.z1, b.y0:b.y1, b.x0:b.x1], copy=True)


def find_flow_crop_z(
    mask: np.ndarray,
    threshold_fraction: float = 0.10,
) -> tuple[int, int]:
    """
    Find the inlet and outlet Z-slices of a binary cluster mask where the
    cross-sectional area is at least `threshold_fraction` of the peak slice area.

    This ensures the crop starts and ends at a Z-plane with a meaningful opening
    for flow simulation, rather than at a near-zero tip of the cluster.

    Parameters
    ----------
    mask:
        3D binary array (non-zero = cluster voxels). Shape (Z, Y, X).
    threshold_fraction:
        Minimum slice voxel count as a fraction of the peak slice count.
        Default 0.10 = 10% of peak.

    Returns
    -------
    (z_in, z_out):
        Half-open Z indices such that mask[z_in:z_out] contains only slices
        meeting the threshold. z_in is the first qualifying slice from the top,
        z_out is one past the last qualifying slice from the bottom.

    Raises
    ------
    ValueError
        If no slice meets the threshold (empty or degenerate mask).
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}")

    # Count non-zero voxels per Z-slice — vectorised over Y and X axes
    slice_counts = (mask > 0).sum(axis=(1, 2)).astype(np.int64)

    peak = int(slice_counts.max())
    if peak == 0:
        raise ValueError("mask contains no non-zero voxels")

    threshold = peak * threshold_fraction
    qualifying = slice_counts >= threshold

    if not qualifying.any():
        raise ValueError(f"No Z-slice meets threshold fraction {threshold_fraction} of peak {peak}")

    z_in = int(np.argmax(qualifying))                          # first True from top
    z_out = int(mask.shape[0]) - int(np.argmax(qualifying[::-1]))  # one past last True from bottom

    return z_in, z_out


def compute_cluster_cog(mask: np.ndarray) -> tuple[float, float, float]:
    """
    Compute the centre of gravity (centroid) of a binary cluster mask.

    Scanning is slice-by-slice to avoid constructing a full coordinate list
    with np.where, which can be large for µCT volumes.

    Parameters
    ----------
    mask:
        3D binary array (non-zero = cluster voxels). Shape (Z, Y, X).

    Returns
    -------
    (cog_z, cog_y, cog_x) in voxel coordinates.
    """
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
    """
    Detect and correct X-axis wrap-around caused by the extraction window
    being offset from the cylinder centre.

    The wrap-around signature in the X-column density profile is:
      - A spike at x=0..k  (sliver of circle that wrapped around)
      - A near-zero gap at x=k..gap_end
      - The main circle body from gap_end onwards

    We detect the gap, measure its width, and roll the volume along X
    so the sliver moves to its correct position after x=749.
    """
    logger.info("Detecting X wrap-around...")

    # Compute foreground voxel count per X column across all Z and Y
    # Do it slice by slice to avoid loading the full volume twice
    x_density = np.zeros(vol.shape[2], dtype=np.int64)
    for z in range(vol.shape[0]):
        x_density += (vol[z] > 0).sum(axis=0)

    # Find the gap: region of near-zero density after the initial spike
    # Threshold at 5% of the maximum density
    threshold = x_density.max() * 0.05

    above = x_density > threshold
    # Find where it first drops below threshold (end of sliver)
    sliver_end = int(np.argmax(~above))        # first False after start
    if sliver_end == 0:
        logger.info("No wrap-around detected — skipping correction.")
        return vol

    # Find where it rises back above threshold (start of main circle)
    gap_end = sliver_end + int(np.argmax(above[sliver_end:]))

    shift = gap_end  # roll left by this many pixels
    logger.info(
        "Wrap-around detected: sliver x=0..%d, gap x=%d..%d, shift=%d px",
        sliver_end - 1, sliver_end, gap_end - 1, shift
    )

    # np.roll shifts the sliver to the right end where it belongs
    vol = np.roll(vol, -shift, axis=2)
    logger.info("X wrap-around corrected.")
    return vol


def log_volume_info(
    vol: np.ndarray,
    spacing,
    info: dict,
    logger: logging.Logger,
    hist: dict[int, int] | None = None,
) -> None:
    """
    Log volume metadata. If `hist` is provided (pre-computed in the pre-pass),
    label counts are logged at no extra cost — no second histogram computation.
    `hist` should always be passed; it is optional only for backwards compatibility.
    """
    logger.info("shape (Z,Y,X): %s", vol.shape)
    logger.info("dtype: %s | codec: %s", vol.dtype, info["codec"])
    logger.info("value range: %s - %s", vol.min(), vol.max())

    if spacing is not None:
        logger.info(
            "spacing (µm): %.3f, %.3f, %.3f",
            spacing[0] * 1e6,
            spacing[1] * 1e6,
            spacing[2] * 1e6,
        )

    if hist is not None:
        for label, count in hist.items():
            logger.info("label %3d: %d voxels", int(label), int(count))
