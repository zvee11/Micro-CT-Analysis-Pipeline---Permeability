#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cluster_shrinkage_figures.py
============================

One-shot, NON-interactive figure generator for the thesis. Follows a single
tracked gas cluster across the qualifying timesteps and produces clean,
print-ready figures showing the cluster shrinking inside its frozen box.

Run from the repository root (the folder that CONTAINS the `pipeline/` package):

    cd THESIS
    python cluster_shrinkage_figures.py --run-id 20260618_155713_AMSDS99619 --track 4

Outputs (PNG, 300 dpi, white background, no interactivity):
    fig3_clusters_3d_shrinkage.png   3D render of the cluster at each timestep
    fig5_fixed_box_shrinkage.png     fixed-box schematic with real voxel counts
    fig_extra_volume_vs_sw.png       (optional, --extras)
    fig_extra_box_occupancy.png      (optional, --extras)

Two data sources, chosen automatically:
  * QUANTITATIVE figures (fig5 + extras) need only results.duckdb. They always run.
  * The 3D render (fig3) needs the per-scan gas geometry. It uses, in order:
        1) the gas-domain .raw files written by the pipeline
           (fixed_boxes.domain_gas), if present on disk;
        2) the original .am scans, re-deriving the gas mask inside the box;
        3) if neither is reachable, fig3 is skipped with a clear message and the
           quantitative figures are still produced.

This script is read-only w.r.t. the database and never re-runs the pipeline.
It reuses your own modules: pipeline.io.read_avizo, pipeline.config.Config,
pipeline.visualisation._track_colour, pipeline.simulation_domains constants.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import duckdb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ---------------------------------------------------------------------------
# Pipeline imports.  These match the uploaded tree exactly.
# ---------------------------------------------------------------------------
try:
    from pipeline.visualisation import _track_colour
except Exception:
    def _track_colour(track_id: int) -> str:        # fallback = pipeline's palette
        palette = ["#2196F3", "#F44336", "#4CAF50", "#FF9800",
                   "#9C27B0", "#00BCD4", "#E91E63", "#8BC34A",
                   "#FF5722", "#607D8B"]
        return palette[(track_id - 1) % len(palette)]

# Domain encoding (0 = open/fluid) — from pipeline.simulation_domains.
# A gas-domain .raw has value 0 exactly on the tracked cluster.
DOMAIN_FLUID_VALUE = 0
GAS_LABEL_DEFAULT = 2     # pipeline.config.Config.gas_label
BRINE_LABEL = 1


# ===========================================================================
# DB access
# ===========================================================================

def fetch_box(con, run_id: str, track_id: int) -> dict:
    """Frozen-box geometry (constant across scans) from the first fixed_boxes row."""
    row = con.execute(
        """
        SELECT z0, z1, y0, y1, x0, x1, extent_z, extent_y, extent_x, gas_voxels_at_X
        FROM fixed_boxes
        WHERE run_id = ? AND track_id = ?
        ORDER BY scan_index
        LIMIT 1
        """,
        [run_id, track_id],
    ).fetchone()
    if row is None:
        sys.exit(f"No fixed_boxes rows for run {run_id}, track {track_id}.")
    keys = ["z0", "z1", "y0", "y1", "x0", "x1",
            "extent_z", "extent_y", "extent_x", "gas_voxels_at_X"]
    return dict(zip(keys, row))


def fetch_track_series(con, run_id: str, track_id: int) -> list[dict]:
    """Per-scan record for the track, joined to scan metadata, ordered in time."""
    rows = con.execute(
        """
        SELECT  fb.scan_index,
                fb.gas_voxels,
                fb.sw_local,
                fb.domain_gas,
                fb.extent_z,
                s.file_name,
                s.Sw          AS sw_global,
                s.is_X,
                s.elapsed_minutes
        FROM fixed_boxes fb
        JOIN scans s
          ON s.run_id = fb.run_id AND s.scan_index = fb.scan_index
        WHERE fb.run_id = ? AND fb.track_id = ?
        ORDER BY fb.scan_index
        """,
        [run_id, track_id],
    ).fetchall()
    cols = ["scan_index", "gas_voxels", "sw_local", "domain_gas", "extent_z",
            "file_name", "sw_global", "is_X", "elapsed_minutes"]
    return [dict(zip(cols, r)) for r in rows]


# ===========================================================================
# Geometry loading for the 3D render (best-effort, three routes)
# ===========================================================================

def _resolve(path_str: str | None, data_root: Path | None) -> Path | None:
    """Resolve a stored (Windows-style) path against the local --data-root."""
    if not path_str:
        return None
    p = Path(str(path_str).replace("\\", "/"))
    if p.exists():
        return p
    if data_root is not None:
        for cand in (data_root / p.name,
                     (data_root / Path(*p.parts[-2:])) if len(p.parts) >= 2 else None,
                     (data_root / Path(*p.parts[-3:])) if len(p.parts) >= 3 else None):
            if cand is not None and cand.exists():
                return cand
    return None


def load_gas_mask_from_domain(raw_path: Path, box: dict) -> np.ndarray | None:
    """Gas-domain .raw is uint8 with 0 on the cluster; box extents give the shape."""
    nz = box["z1"] - box["z0"]
    ny = box["y1"] - box["y0"]
    nx = box["x1"] - box["x0"]
    expected = nz * ny * nx
    arr = np.fromfile(raw_path, dtype=np.uint8)
    if arr.size != expected:
        print(f"    domain {raw_path.name}: size {arr.size:,} != box {expected:,}, skipping")
        return None
    return arr.reshape((nz, ny, nx)) == DOMAIN_FLUID_VALUE


def load_gas_mask_from_am(am_path: Path, box: dict, gas_label: int) -> np.ndarray | None:
    """Re-derive the gas mask inside the frozen box straight from the .am scan."""
    try:
        from pipeline.io import read_avizo
    except Exception as e:
        print(f"    pipeline.io.read_avizo unavailable ({e})")
        return None
    vol, _spacing, _meta = read_avizo(am_path, parse_spacing=False)
    sub = vol[box["z0"]:box["z1"], box["y0"]:box["y1"], box["x0"]:box["x1"]]
    return np.asarray(sub == gas_label)


def mask_to_surface(mask: np.ndarray, downsample: int = 2):
    """Marching-cubes surface from a boolean mask; returns PyVista PolyData or None."""
    try:
        import pyvista as pv
        from skimage.measure import marching_cubes
    except Exception as e:
        print(f"    PyVista/skimage unavailable ({e}); 3D render skipped")
        return None
    if not mask.any():
        return None
    m = mask[::downsample, ::downsample, ::downsample] if downsample > 1 else mask
    if not m.any():
        m = mask
    try:
        verts, faces, _n, _v = marching_cubes(m.astype(np.uint8), level=0.5)
    except (ValueError, RuntimeError):
        return None
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64).ravel()
    return pv.PolyData(verts, faces_pv)


# ===========================================================================
# FIG 3 — 3D shrinkage, small multiples, identical camera per panel
# ===========================================================================

def make_fig3(series: list[dict], box: dict, track_id: int, colour: str,
              data_root: Path | None, gas_label: int, out: Path,
              downsample: int = 2) -> bool:
    try:
        import pyvista as pv
    except Exception as e:
        print(f"  fig3 skipped — PyVista not available ({e})")
        return False

    surfaces, kept = [], []
    for s in series:
        mask = None
        dom = _resolve(s["domain_gas"], data_root)
        if dom is not None:
            mask = load_gas_mask_from_domain(dom, box)
        if mask is None and data_root is not None:
            am = _resolve(s["file_name"], data_root)
            if am is not None:
                mask = load_gas_mask_from_am(am, box, gas_label)
        if mask is None:
            continue
        surf = mask_to_surface(mask, downsample)
        if surf is not None and surf.n_points:
            surfaces.append(surf)
            kept.append(s)

    if not surfaces:
        print("  fig3 skipped — no per-scan geometry found "
              "(no gas-domain .raw files and no .am scans reachable).")
        print("       Pass --data-root pointing at the run's output/ or data/ folder "
              "to enable the 3D render.")
        return False

    n = len(surfaces)
    pv.global_theme.background = "white"
    pv.global_theme.transparent_background = False
    pl = pv.Plotter(off_screen=True, shape=(1, n),
                    window_size=(330 * n, 460), border=False)
    cam = None
    for i, (surf, s) in enumerate(zip(surfaces, kept)):
        pl.subplot(0, i)
        pl.add_mesh(surf, color=colour, opacity=1.0, smooth_shading=True)
        tag = f"scan {s['scan_index']}" + ("  (X)" if s["is_X"] else "")
        pl.add_text(f"{tag}\nSw={s['sw_local']:.2f}\n{s['gas_voxels']:,} vox",
                    position="upper_left", font_size=9, color="black",
                    font="courier")
        if cam is None:
            pl.camera_position = "iso"
            pl.reset_camera()
            cam = pl.camera_position
        else:
            pl.camera_position = cam
    pl.link_views()
    pl.screenshot(str(out), scale=2)
    pl.close()
    print(f"  wrote {out}  ({n} panels)")
    return True


# ===========================================================================
# FIG 5 — fixed-box schematic with REAL voxel counts (no volume files needed)
# ===========================================================================

def make_fig5(series: list[dict], box: dict, track_id: int, out: Path,
              max_panels: int = 6):
    box_vox = box["extent_z"] * box["extent_y"] * box["extent_x"]
    x_pos = next((i for i, s in enumerate(series) if s["is_X"]), len(series) - 1)
    if len(series) > max_panels:
        picks = sorted(set(
            np.linspace(0, len(series) - 1, max_panels - 1).round().astype(int).tolist()
            + [x_pos]))
    else:
        picks = list(range(len(series)))
    panels = [series[i] for i in picks]

    fill = [s["gas_voxels"] / box_vox for s in panels]
    fmax = max(fill) if max(fill) > 0 else 1.0
    n = len(panels)

    fig = plt.figure(figsize=(2.05 * n, 3.2))
    gs = gridspec.GridSpec(1, n, wspace=0.18)

    for i, s in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        is_X = bool(s["is_X"])
        ax.add_patch(plt.Rectangle(
            (0.12, 0.16), 0.76, 0.72, fill=False,
            edgecolor="#185FA5" if is_X else "#888780",
            lw=2.2 if is_X else 1.3,
            linestyle="-" if is_X else (0, (5, 4))))
        r = 0.32 * np.sqrt(max(fill[i] / fmax, 0.02))
        ax.add_patch(plt.Circle((0.5, 0.52), r, color="#1D9E75", alpha=0.78))
        ax.text(0.5, 0.04, f"scan {s['scan_index']}\n$S_w$={s['sw_local']:.2f}",
                ha="center", va="center", fontsize=8)
        if is_X:
            ax.text(0.5, 0.955, "timestep X", ha="center", va="center",
                    fontsize=8, color="#185FA5")

    fig.suptitle(
        f"Track {track_id}: frozen box {box['extent_z']}×{box['extent_y']}×"
        f"{box['extent_x']} vox.  Cluster {panels[0]['gas_voxels']:,} → "
        f"{panels[-1]['gas_voxels']:,} gas voxels",
        fontsize=9, y=1.03)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}  ({n} panels)")


# ===========================================================================
# EXTRA quantitative diagnostics
# ===========================================================================

def make_extras(series: list[dict], box: dict, track_id: int, out_dir: Path):
    box_vox = box["extent_z"] * box["extent_y"] * box["extent_x"]
    sw = [s["sw_local"] for s in series]
    vox = [s["gas_voxels"] for s in series]
    idx = [s["scan_index"] for s in series]
    occ = [v / box_vox * 100 for v in vox]
    colour = "#1D9E75"

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot(sw, vox, "o-", color=colour, mfc="white")
    for x, y, is_X in zip(sw, vox, [s["is_X"] for s in series]):
        if is_X:
            ax.annotate("X", (x, y), fontsize=8, fontweight="bold",
                        xytext=(4, 4), textcoords="offset points", color="#185FA5")
    ax.set_xlabel("Local water saturation  $S_w^{local}$")
    ax.set_ylabel("Cluster gas voxels in box")
    ax.set_title(f"Track {track_id}: cluster volume vs saturation", fontsize=9)
    ax.grid(alpha=0.25)
    fig.savefig(out_dir / "fig_extra_volume_vs_sw.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    bars = ax.bar(range(len(occ)), occ, color="#185FA5", alpha=0.85)
    for i, s in enumerate(series):
        if s["is_X"]:
            bars[i].set_color("#1D9E75")
    ax.set_xticks(range(len(occ)))
    ax.set_xticklabels(idx, fontsize=7)
    ax.set_xlabel("Scan index")
    ax.set_ylabel("Box occupancy (%)")
    ax.set_title(f"Track {track_id}: frozen-box gas occupancy", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(out_dir / "fig_extra_box_occupancy.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote 2 extra diagnostics to {out_dir}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="results.duckdb", type=Path)
    ap.add_argument("--run-id", default="20260618_155713_AMSDS99619",
                    help="canonical run id (default: the 12-step run)")
    ap.add_argument("--track", type=int, default=4,
                    help="track_id to follow (default 4 — biggest shrinkage)")
    ap.add_argument("--gas-label", type=int, default=GAS_LABEL_DEFAULT)
    ap.add_argument("--data-root", type=Path, default=None,
                    help="folder containing the run's output/ or data/ files; "
                         "needed only for the 3D render (fig3)")
    ap.add_argument("--downsample", type=int, default=2,
                    help="voxel downsample for the 3D meshes (1 = full res)")
    ap.add_argument("--out", default=Path("figures"), type=Path)
    ap.add_argument("--extras", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.db.exists():
        sys.exit(f"Database not found: {args.db}")

    con = duckdb.connect(str(args.db), read_only=True)

    box = fetch_box(con, args.run_id, args.track)
    series = fetch_track_series(con, args.run_id, args.track)
    if not series:
        sys.exit(f"No data for run {args.run_id}, track {args.track}.")

    colour = _track_colour(args.track)
    print(f"Run {args.run_id} | track {args.track} | {len(series)} scans")
    print(f"Frozen box z[{box['z0']}:{box['z1']}] y[{box['y0']}:{box['y1']}] "
          f"x[{box['x0']}:{box['x1']}]  extent "
          f"{box['extent_z']}×{box['extent_y']}×{box['extent_x']}")
    print(f"Cluster gas voxels: {series[0]['gas_voxels']:,} (scan "
          f"{series[0]['scan_index']}) -> {series[-1]['gas_voxels']:,} "
          f"(scan {series[-1]['scan_index']}, X)")

    make_fig5(series, box, args.track, args.out / "fig5_fixed_box_shrinkage.png")
    if args.extras:
        make_extras(series, box, args.track, args.out)

    make_fig3(series, box, args.track, colour, args.data_root,
              args.gas_label, args.out / "fig3_clusters_3d_shrinkage.png",
              downsample=args.downsample)

    con.close()
    print("Done.")


if __name__ == "__main__":
    main()