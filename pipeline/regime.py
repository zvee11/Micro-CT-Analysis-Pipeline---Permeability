from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Callable

import numpy as np

from .config import Config
from .io import label_histogram, read_avizo


def compute_sw_series(
    input_files: list[Path],
    cfg: Config,
    logger: logging.Logger,
    on_file_start: "Callable[[int, str], None] | None" = None,
    on_file_done: "Callable[[int, str, float], None] | None" = None,
) -> tuple[list[float], list[dict[int, int]]]:
    """
    Pre-pass: load each volume once, compute both brine saturation (Sw) and
    the full label histogram for every file.

    Parameters
    ----------
    on_file_start:
        Optional callback called when a file starts loading.
        Signature: (scan_index: int, file_name: str) -> None
    on_file_done:
        Optional callback called when a file finishes.
        Signature: (scan_index: int, file_name: str, sw: float) -> None

    Returns
    -------
    sw_series:
        Sw = brine_voxels / (brine_voxels + gas_voxels) per file, in input order.
        NaN when a file contains no brine or gas voxels.
    hist_series:
        Full label histogram (label -> voxel_count) per file, in input order.
        Passed into log_volume_info in the main pass so the histogram is never
        recomputed — it was already paid for here.
    """
    sw_series: list[float] = []
    hist_series: list[dict[int, int]] = []

    for idx, path in enumerate(input_files, start=1):
        logger.info("pre-pass %d/%d: %s", idx, len(input_files), path.name)

        if on_file_start:
            on_file_start(idx - 1, path.name)

        vol, _, _ = read_avizo(path, parse_spacing=False, memmap_raw=cfg.memmap_raw)

        hist = label_histogram(vol)
        hist_series.append(hist)

        brine = hist.get(1, 0)
        gas = hist.get(cfg.gas_label, 0)
        total = brine + gas

        if total == 0:
            logger.warning(
                "no brine or gas voxels found in %s — Sw set to NaN", path.name
            )
            sw = float("nan")
        else:
            sw = brine / total
        sw_series.append(sw)

        del vol
        gc.collect()

        logger.info("pre-pass %d/%d done | Sw=%.4f | %s", idx, len(input_files), sw if not np.isnan(sw) else -1, path.name)

        if on_file_done:
            on_file_done(idx - 1, path.name, sw)

    logger.info(
        "Sw series: %s",
        ", ".join(f"{sw:.4f}" if not np.isnan(sw) else "NaN" for sw in sw_series),
    )
    return sw_series, hist_series


def _sse(xs: np.ndarray, ys: np.ndarray) -> float:
    """Sum of squared errors for a linear fit to (xs, ys)."""
    if len(xs) < 2:
        return np.inf
    coeffs = np.polyfit(xs, ys, 1)
    return float(np.sum((np.polyval(coeffs, xs) - ys) ** 2))


def _piecewise_linear_3seg(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, list[float]]:
    """
    Fit three line segments (two breakpoints) to (x, y) by exhaustive search.
    Returns (bp1_x, bp2_x, [slope1, slope2, slope3]).
    """
    best_sse = np.inf
    best_i, best_j = 1, 2

    for i in range(1, len(x) - 2):
        sse_left = _sse(x[:i + 1], y[:i + 1])
        if sse_left >= best_sse:
            continue  # prune: left segment alone already worse than best total
        for j in range(i + 1, len(x) - 1):
            sse = sse_left + _sse(x[i:j + 1], y[i:j + 1]) + _sse(x[j:], y[j:])
            if sse < best_sse:
                best_sse = sse
                best_i, best_j = i, j

    bp1 = float(x[best_i])
    bp2 = float(x[best_j])
    s1 = float(np.polyfit(x[:best_i + 1], y[:best_i + 1], 1)[0])
    s2 = float(np.polyfit(x[best_i:best_j + 1], y[best_i:best_j + 1], 1)[0])
    s3 = float(np.polyfit(x[best_j:], y[best_j:], 1)[0])
    return bp1, bp2, [s1, s2, s3]


def detect_regime_boundary(
    sw_series: list[float],
    cfg: Config,
    logger: logging.Logger,
    x_values: list[float] | None = None,
) -> int:
    """
    Fit piecewise linear model to Sw vs X and return the 0-based scan index
    of the last qualifying timestep X.

    x_values:
        Optional list of X-axis values aligned to sw_series (e.g. elapsed
        minutes from a reference saturation file). If provided, the piecewise
        fit is performed in this space so the breakpoint is found in physically
        meaningful units. If None, scan index (0, 1, 2, ...) is used.

        The returned value is always a scan index regardless of x_values.

    regime_cutoff="transition" (default):
        Three-segment fit. X = last timestep before second breakpoint.
        Includes displacement + transition phases, as in the paper.

    regime_cutoff="displacement":
        Three-segment fit. X = last timestep before first breakpoint.
        Displacement phase only (conservative).

    Fallback: fewer than cfg.min_scans_three_segment valid scans ->
        regime detection is skipped entirely. All timesteps qualify and
        X is set to the last scan. A warning is logged explaining why.
    """
    # Build (x_coord, sw, scan_index) triples — skip NaN in either series
    triples = []
    for i, sw in enumerate(sw_series):
        if np.isnan(sw):
            continue
        if x_values is not None:
            xv = x_values[i] if i < len(x_values) else float("nan")
            if np.isnan(xv):
                continue  # skip scans with no reference time
        else:
            xv = float(i)
        triples.append((xv, sw, i))

    n = len(triples)

    if n < cfg.min_scans_three_segment:
        x_label = "minutes" if x_values is not None else "scans"
        logger.warning(
            "only %d valid %s available — minimum for three-segment regime "
            "detection is %d. Regime detection skipped: all %d timesteps qualify. "
            "Set min_scans_three_segment lower if you want to attempt detection "
            "with fewer scans, but results will be unreliable.",
            n, x_label, cfg.min_scans_three_segment, len(sw_series),
        )
        return len(sw_series) - 1

    xs      = np.array([t[0] for t in triples], dtype=float)
    sw_vals = np.array([t[1] for t in triples], dtype=float)
    scan_ids = [t[2] for t in triples]

    bp1, bp2, slopes = _piecewise_linear_3seg(xs, sw_vals)

    x_unit = "min" if x_values is not None else "scan"
    logger.info(
        "three-segment fit: bp1=%.1f%s (slope %.4f->%.4f), "
        "bp2=%.1f%s (slope %.4f->%.4f)",
        bp1, x_unit, slopes[0], slopes[1],
        bp2, x_unit, slopes[1], slopes[2],
    )
    cutoff_x = bp1 if cfg.regime_cutoff == "displacement" else bp2
    logger.info(
        "regime_cutoff=%s: using breakpoint %.1f%s as X",
        cfg.regime_cutoff, cutoff_x, x_unit,
    )

    # Find the last scan whose x_coord is <= cutoff_x
    qualifying_scan_ids = [
        scan_i for (xv, sw, scan_i) in triples if xv <= cutoff_x
    ]
    if not qualifying_scan_ids:
        logger.warning(
            "no qualifying timesteps before cutoff %.1f%s — using all timesteps",
            cutoff_x, x_unit,
        )
        return len(sw_series) - 1

    X = max(qualifying_scan_ids)
    logger.info(
        "regime boundary: X = timestep index %d (%d qualifying timesteps total)",
        X, len(qualifying_scan_ids),
    )
    return X