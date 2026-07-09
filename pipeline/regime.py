from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Callable

import numpy as np

from .config import Config
from .io import label_histogram, read_avizo


def _prepass_one(args: "tuple[int, Path, int]") -> "tuple[int, float, dict]":
    """Decode one volume, return (idx, Sw, histogram). Runs in a worker process.
    Top-level (not a closure) so it is importable under Windows 'spawn'."""
    idx, path, gas_label = args
    vol, _, _ = read_avizo(path, parse_spacing=False, memmap_raw=False)
    hist = label_histogram(vol)
    del vol
    gc.collect()
    brine = hist.get(1, 0)
    gas = hist.get(gas_label, 0)
    total = brine + gas
    sw = (brine / total) if total > 0 else float("nan")
    return idx, sw, hist


def compute_sw_series(
    input_files: list[Path],
    cfg: Config,
    logger: logging.Logger,
    on_file_start: "Callable[[int, str], None] | None" = None,
    on_file_done: "Callable[[int, str, float], None] | None" = None,
) -> tuple[list[float], list[dict[int, int]]]:
    """Pre-pass: per file compute Sw = brine/(brine+gas) and the label histogram.

    Serial when cfg.prepass_workers == 1 (laptop-safe: one volume in RAM at a
    time). With prepass_workers > 1, decodes run in a process pool — each worker
    holds one full volume, so only raise this on a high-RAM machine. Results are
    placed by file index, so the output is identical to the serial path
    regardless of worker count or completion order.
    """
    n = len(input_files)
    sw_series: list[float] = [float("nan")] * n
    hist_series: list[dict[int, int]] = [{} for _ in range(n)]
    workers = max(1, int(getattr(cfg, "prepass_workers", 1)))

    def _record(idx: int, sw: float, hist: dict) -> None:
        sw_series[idx] = sw
        hist_series[idx] = hist
        path = input_files[idx]
        if np.isnan(sw):
            logger.warning("no brine or gas voxels found in %s — Sw set to NaN", path.name)
        logger.info("pre-pass %d/%d done | Sw=%.4f | %s",
                    idx + 1, n, sw if not np.isnan(sw) else -1, path.name)
        if on_file_done:
            on_file_done(idx, path.name, sw)

    if workers == 1:
        for idx, path in enumerate(input_files):
            logger.info("pre-pass %d/%d: %s", idx + 1, n, path.name)
            if on_file_start:
                on_file_start(idx, path.name)
            _, sw, hist = _prepass_one((idx, path, cfg.gas_label))
            _record(idx, sw, hist)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        logger.info("pre-pass: parallel decode with %d workers", workers)
        if on_file_start:
            for idx, path in enumerate(input_files):
                on_file_start(idx, path.name)
        tasks = [(idx, path, cfg.gas_label) for idx, path in enumerate(input_files)]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_prepass_one, t) for t in tasks]
            for fut in as_completed(futures):
                idx, sw, hist = fut.result()
                _record(idx, sw, hist)

    logger.info("Sw series: %s",
                ", ".join(f"{sw:.4f}" if not np.isnan(sw) else "NaN" for sw in sw_series))
    return sw_series, hist_series


def _sse(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2:
        return np.inf
    coeffs = np.polyfit(xs, ys, 1)
    return float(np.sum((np.polyval(coeffs, xs) - ys) ** 2))


def _piecewise_linear_3seg(x: np.ndarray, y: np.ndarray) -> tuple[float, float, list[float]]:
    """Fit three segments (two breakpoints) by exhaustive search. Returns (bp1, bp2, slopes)."""
    best_sse = np.inf
    best_i, best_j = 1, 2
    for i in range(1, len(x) - 2):
        sse_left = _sse(x[:i + 1], y[:i + 1])
        if sse_left >= best_sse:
            continue
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


def _select_regime_cutoff_terminal(triples, bp1, bp2, slopes, auto_cutoff,
                                   x_unit, regime_cutoff, logger):
    """Draw the Sw-vs-time(=PV) curve as an ASCII plot in the terminal, show a
    table of values with the three regime segments marked, and let the user type
    the cutoff timestep. No GUI / matplotlib needed, so it always works over RDP.

    Returns the chosen cutoff x-value; falls back to the automatic value on empty
    or invalid input.
    """
    xs = [t[0] for t in triples]
    sws = [t[1] for t in triples]
    scans = [t[2] for t in triples]

    def seg_of(x):
        if x <= bp1:
            return 0      # displacement
        if x <= bp2:
            return 1      # transition
        return 2          # dissolution
    seg_char = ["D", "T", "x"]          # marker per segment
    seg_name = ["displacement", "transition", "dissolution"]

    # ---- ASCII scatter: Sw axis fixed to the full physical range 0..1 ----
    # The origin (scan index -1, Sw=0 at time 0) is shown as 'O'. The Sw axis is
    # the full 0..1 so the whole displacement trajectory is visible.
    H = 21                               # 0.00, 0.05, ... 1.00
    SW_LO, SW_HI = 0.0, 1.0
    rng = SW_HI - SW_LO
    grid = [[" "] * len(triples) for _ in range(H)]
    for col, (xv, sw, sc) in enumerate(triples):
        row = int(round((1 - (sw - SW_LO) / rng) * (H - 1)))
        row = max(0, min(H - 1, row))
        grid[row][col] = "O" if sc < 0 else seg_char[seg_of(xv)]

    print("\n" + "=" * 70)
    print("  REGIME CUTOFF SELECTION   (Sw vs time/PV, steady injection)")
    print("=" * 70)
    print(f"  Sw axis: {SW_HI:.2f} (top) .. {SW_LO:.2f} (bottom)   "
          f"O=origin  D=displacement  T=transition  x=dissolution")
    print("  " + "-" * (len(triples) * 3 + 8))
    for r in range(H):
        sw_lab = SW_HI - (r / (H - 1)) * rng
        print(f"  {sw_lab:4.2f} |" + "".join(f" {c} " for c in grid[r]))
    print("       +" + "".join(" - " for _ in triples))
    print("  idx -> " + "".join(f"{('O' if sc < 0 else str(sc)):>2} " for sc in scans))
    print("  " + "-" * (len(triples) * 3 + 8))

    # ---- value table with segment + auto-cutoff marker ----
    print(f"  {'idx':>3} {'time/PV':>9} {'Sw':>8}  {'segment':>13}  marker")
    auto_scan = None
    for xv, sw, sc in triples:
        mark = ""
        if abs(xv - auto_cutoff) < 1e-6 and sc >= 0:
            mark = "<-- AUTO cutoff"
            auto_scan = sc
        idx_lbl = "O" if sc < 0 else str(sc)
        seg_lbl = "origin" if sc < 0 else seg_name[seg_of(xv)]
        print(f"  {idx_lbl:>3} {xv:>9.1f} {sw:>8.4f}  {seg_lbl:>13}  {mark}")
    print("  " + "-" * 66)
    print(f"  slopes: displacement={slopes[0]:.4f}  transition={slopes[1]:.4f}  "
          f"dissolution={slopes[2]:.4f}")
    print(f"  auto breakpoints: bp1={bp1:.0f}{x_unit} (D->T), bp2={bp2:.0f}{x_unit} (T->x)")
    print(f"  (origin O=(0,0) is a fixed boundary condition for the fit, not "
          f"a selectable timestep)")
    print(f"  regime_cutoff='{regime_cutoff}' -> AUTO cutoff = timestep {auto_scan} "
          f"(x={auto_cutoff:.0f}{x_unit})")
    print("=" * 70)

    # ---- prompt (only real scans, sc >= 0, are selectable) ----
    try:
        resp = input("  Enter cutoff timestep index [Enter = accept AUTO]: ").strip()
    except EOFError:
        resp = ""
    if resp == "":
        logger.info("user accepted AUTO regime cutoff (timestep %s)", auto_scan)
        return auto_cutoff
    try:
        chosen = int(resp)
    except ValueError:
        logger.warning("'%s' is not an integer; using AUTO cutoff.", resp)
        return auto_cutoff
    match = [xv for (xv, sw, sc) in triples if sc == chosen and sc >= 0]
    if not match:
        logger.warning("timestep %d not a selectable scan; using AUTO cutoff.", chosen)
        return auto_cutoff
    logger.info("user set regime cutoff to timestep %d (x=%.1f%s)",
                chosen, match[0], x_unit)
    return match[0]


def detect_regime_boundary(
    sw_series: list[float],
    cfg: Config,
    logger: logging.Logger,
    x_values: list[float] | None = None,
) -> int:
    """Piecewise-linear fit of Sw vs X; return 0-based scan index of last qualifying timestep X.

    regime_cutoff="transition" uses the second breakpoint; "displacement" the first.
    Falls back to all-qualify if fewer than cfg.min_scans_three_segment valid scans.
    """
    triples = []
    for i, sw in enumerate(sw_series):
        if np.isnan(sw):
            continue
        if x_values is not None:
            xv = x_values[i] if i < len(x_values) else float("nan")
            if np.isnan(xv):
                continue
        else:
            xv = float(i)
        triples.append((xv, sw, i))

    # Add the physical boundary condition at the origin: at time 0, before any
    # injection, Sw = 0. This is a real constraint on the displacement slope, so
    # it feeds the fit. It is NOT a real scan, so it carries scan index -1 and is
    # never selectable as a cutoff nor counted as a qualifying timestep.
    fit_triples = [(0.0, 0.0, -1)] + triples

    n = len(triples)
    if n < cfg.min_scans_three_segment:
        x_label = "minutes" if x_values is not None else "scans"
        logger.warning(
            "only %d valid %s available — minimum for three-segment regime detection is %d. "
            "Regime detection skipped: all %d timesteps qualify.",
            n, x_label, cfg.min_scans_three_segment, len(sw_series),
        )
        return len(sw_series) - 1

    xs = np.array([t[0] for t in fit_triples], dtype=float)
    sw_vals = np.array([t[1] for t in fit_triples], dtype=float)

    bp1, bp2, slopes = _piecewise_linear_3seg(xs, sw_vals)
    x_unit = "min" if x_values is not None else "scan"
    logger.info("three-segment fit: bp1=%.1f%s (slope %.4f->%.4f), bp2=%.1f%s (slope %.4f->%.4f)",
                bp1, x_unit, slopes[0], slopes[1], bp2, x_unit, slopes[1], slopes[2])

    cutoff_x = bp1 if cfg.regime_cutoff == "displacement" else bp2
    logger.info("regime_cutoff=%s: using breakpoint %.1f%s as X", cfg.regime_cutoff, cutoff_x, x_unit)

    # ---- user interjection: choose the regime cutoff in the terminal ----
    # Injection is at a steady rate, so time (minutes) is proportional to injected
    # pore volumes; the x-axis here is therefore equivalent to PV. An ASCII plot
    # and table are printed in the terminal (no GUI needed, works over RDP) and
    # the user types the cutoff timestep. Skipped when interactive_regime=False.
    if getattr(cfg, "interactive_regime", False):
        cutoff_x = _select_regime_cutoff_terminal(
            fit_triples, bp1, bp2, slopes, cutoff_x, x_unit, cfg.regime_cutoff, logger)

    qualifying_scan_ids = [scan_i for (xv, sw, scan_i) in triples if xv <= cutoff_x]
    if not qualifying_scan_ids:
        logger.warning("no qualifying timesteps before cutoff %.1f%s — using all timesteps", cutoff_x, x_unit)
        return len(sw_series) - 1

    X = max(qualifying_scan_ids)
    logger.info("regime boundary: X = timestep index %d (%d qualifying timesteps total)",
                X, len(qualifying_scan_ids))
    return X