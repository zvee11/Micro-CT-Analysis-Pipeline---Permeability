#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cluster_cca_shrinkage.py
========================

Run connected-component analysis (cc3d, 18N) on ONE tracked cluster at EVERY
qualifying timestep and render how it shrinks over time. No simulation, no
schematic — this is the real per-timestep geometry of a single matched cluster.

Run from the repository root (the folder containing the `pipeline/` package):

    cd THESIS
    python cluster_cca_shrinkage.py --track 4 --data-root output

What it does, per qualifying scan, for the chosen track:
  1. read the .am scan with pipeline.io.read_avizo
  2. take the frozen box for that track (from fixed_boxes) and crop to it
  3. extract the gas phase (vol == 2) inside the box
  4. run cc3d (18N) on that gas mask                         <-- the actual CCA
  5. keep the connected component that matches the tracked cluster, by maximum
     voxel overlap with the cluster footprint at timestep X (the clustermask
     written by the pipeline). This guarantees we follow the SAME cluster, not
     just the largest blob, even once it fragments near dissolution.
  6. render that component.

Outputs (static PNG, 300 dpi, white background):
    cca_shrinkage_3d_track{T}.png     small-multiples 3D render, one panel/scan
    cca_shrinkage_montage_track{T}.png mid-slice montage (2D, no PyVista needed)
    cca_shrinkage_counts_track{T}.png  matched-component voxel count vs scan

The 3D panel needs PyVista; the montage and counts plots do not, so you still
get figures on a machine without a GPU/VTK.

Matching note: when the matched component's voxel count differs from the
pipeline's box gas count (fixed_boxes.gas_voxels), the difference is gas in the
box that is NOT connected to the tracked cluster at that timestep (other
fragments). The script prints both so the divergence is visible — this is the
percolation-split / fragmentation signal, not an error.
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

try:
    import cc3d
except ImportError:
    sys.exit("cc3d is required. Install into the venv: pip install connected-components-3d")

try:
    from pipeline.visualisation import _track_colour
except Exception:
    def _track_colour(track_id: int) -> str:
        palette = ["#2196F3", "#F44336", "#4CAF50", "#FF9800",
                   "#9C27B0", "#00BCD4", "#E91E63", "#8BC34A",
                   "#FF5722", "#607D8B"]
        return palette[(track_id - 1) % len(palette)]

# pipeline.config: CC3D_CONNECTIVITY = {1: 6, 2: 18, 3: 26}; gas_label = 2
CC3D_CONN = {"6N": 6, "18N": 18, "26N": 26}
GAS_LABEL_DEFAULT = 2


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def fetch_box(con, run_id, track_id):
    row = con.execute(
        """SELECT z0,z1,y0,y1,x0,x1,connectivity FROM fixed_boxes
           WHERE run_id=? AND track_id=? ORDER BY scan_index LIMIT 1""",
        [run_id, track_id]).fetchone()
    if row is None:
        runs = [r[0] for r in con.execute("SELECT DISTINCT run_id FROM runs ORDER BY started_at").fetchall()]
        tracks = [r[0] for r in con.execute(
            "SELECT DISTINCT track_id FROM fixed_boxes WHERE run_id=? ORDER BY track_id", [run_id]).fetchall()]
        if run_id not in runs:
            sys.exit(f"Run {run_id!r} not found. Available: {', '.join(runs)}")
        sys.exit(f"Track {track_id} not in run. Available tracks: {', '.join(map(str, tracks))}")
    keys = ["z0", "z1", "y0", "y1", "x0", "x1", "connectivity"]
    return dict(zip(keys, row))


def fetch_series(con, run_id, track_id):
    rows = con.execute(
        """SELECT fb.scan_index, fb.gas_voxels, fb.sw_local, s.file_name, s.Sw, s.is_X,
                  fb.domain_gas, fb.z0, fb.z1
           FROM fixed_boxes fb
           JOIN scans s ON s.run_id=fb.run_id AND s.scan_index=fb.scan_index
           WHERE fb.run_id=? AND fb.track_id=? ORDER BY fb.scan_index""",
        [run_id, track_id]).fetchall()
    cols = ["scan_index", "box_gas_voxels", "sw_local", "file_name", "sw_global",
            "is_X", "domain_gas", "box_z0", "box_z1"]
    return [dict(zip(cols, r)) for r in rows]


def fetch_clustermask(con, run_id, track_id):
    """The pipeline's tracked-cluster mask at X, over the cluster's OWN bbox
    Z-extent (z0_bbox..z1_bbox), full Y/X. Returns (path, cm_z0, cm_z1) in
    full-volume Z coordinates, or None."""
    row = con.execute(
        """SELECT clustermask_raw, clustermask_z0, clustermask_z1, extent_y, extent_x
           FROM cluster_properties WHERE run_id=? AND track_id=? LIMIT 1""",
        [run_id, track_id]).fetchone()
    return row  # (path, cm_z0, cm_z1, ny, nx)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _resolve(path_str, data_root):
    if not path_str:
        return None
    p = Path(str(path_str).replace("\\", "/"))
    if p.exists():
        return p
    if data_root is not None:
        for cand in (data_root / p.name,
                     (data_root / Path(*p.parts[-2:])) if len(p.parts) >= 2 else None,
                     (data_root / Path(*p.parts[-3:])) if len(p.parts) >= 3 else None,
                     (data_root / Path(*p.parts[-4:])) if len(p.parts) >= 4 else None):
            if cand is not None and cand.exists():
                return cand
    return None


def _resolve_domain(stored_path, track_id, scan_index, output_root):
    """Find a gas-domain .raw by track+scan, ignoring the (sometimes stale) NZ
    token in the stored filename. Looks in the scan folder implied by the stored
    path, then falls back to a recursive search under output_root. Handles both
    the per-scan name (domain_gas_TT-S_..raw) and the X-timestep name
    (cluster_TT_domain_gas_*.raw)."""
    import glob
    stored = Path(str(stored_path).replace("\\", "/")) if stored_path else None
    pats = [f"domain_gas_{track_id:02d}-{scan_index}_*.raw",
            f"cluster_{track_id:02d}_domain_gas_*.raw"]

    search_dirs = []
    # 1) the directory the stored path points at (re-rooted onto output_root)
    if stored is not None and len(stored.parts) >= 2:
        scan_dir = Path(*stored.parts[-3:-1])  # e.g. <scanfolder>/18N
        if output_root is not None:
            search_dirs.append(output_root / scan_dir)
        search_dirs.append(scan_dir)
    # 2) recursive fallback under output_root
    if output_root is not None:
        search_dirs.append(output_root)

    for d in search_dirs:
        if d is None:
            continue
        for pat in pats:
            hits = sorted(glob.glob(str(d / "**" / pat), recursive=True)) \
                if d == output_root else sorted(glob.glob(str(d / pat)))
            if hits:
                return Path(hits[0])
    return None


def load_gas_mask(am_path, cz0, cz1, gas_label):
    """Gas mask over the CLUSTER's own Z-band (cz0..cz1 inclusive, full Y/X) —
    so the whole cluster is captured, not the flow-cropped slab."""
    from pipeline.io import read_avizo
    vol, _sp, _meta = read_avizo(am_path, parse_spacing=False)
    sub = vol[cz0:cz1 + 1, :, :]
    return np.ascontiguousarray(sub == gas_label)


def load_domain_gas(raw_path, ny=750, nx=750):
    """Read the pipeline's box-clipped gas domain .raw. The filename encodes the
    true dimensions as {nx}x{ny}x{nz}; the array is uint8 in C-order (nz, ny, nx)
    with value 0 on the cluster (open) and 1 elsewhere. Returns a boolean mask
    (True on the cluster), or None on a size mismatch."""
    import re
    m = re.search(r"(\d+)x(\d+)x(\d+)\.raw$", raw_path.name)
    if m:
        fx, fy, fz = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        fx, fy, fz = nx, ny, None
    arr = np.fromfile(raw_path, dtype=np.uint8)
    if fz is not None and arr.size == fz * fy * fx:
        nz, ny_, nx_ = fz, fy, fx
    else:
        # fall back: infer nz from total size and the known lateral dims
        if arr.size % (ny * nx) != 0:
            print(f"    domain {raw_path.name}: size {arr.size:,} not divisible "
                  f"by {ny}x{nx}; skipping")
            return None
        nz, ny_, nx_ = arr.size // (ny * nx), ny, nx
    return arr.reshape((nz, ny_, nx_)) == 0  # 0 = open = cluster


def load_X_footprint(cm_row, data_root):
    """Y/X footprint of the tracked cluster at X (Z collapsed), for matching.
    The clustermask is stored over the cluster's bbox Z-extent, full Y/X."""
    if cm_row is None:
        return None
    path, cm_z0, cm_z1, ny, nx = cm_row
    rp = _resolve(path, data_root)
    if rp is None:
        return None
    nz = cm_z1 - cm_z0 + 1
    arr = np.fromfile(rp, dtype=np.uint8)
    if arr.size != nz * ny * nx:
        print(f"    clustermask size {arr.size:,} != expected {nz*ny*nx:,}; "
              "footprint matching disabled")
        return None
    cm = arr.reshape((nz, ny, nx))
    return cm.any(axis=0)  # (ny, nx)


def matched_component(gas_mask, connectivity, footprint):
    """Run cc3d on the box gas mask; return (matched_bool_mask, n_voxels,
    n_components). Match = component with max overlap to the X footprint;
    fallback = largest component if no footprint available."""
    labels = cc3d.connected_components(gas_mask.astype(np.uint8),
                                       connectivity=CC3D_CONN[connectivity])
    n_comp = int(labels.max())
    if n_comp == 0:
        return np.zeros_like(gas_mask, dtype=bool), 0, 0

    if footprint is not None:
        # per-label overlap with the X footprint: count, for each label, how
        # many of its voxels fall in columns (y,x) that belong to the footprint
        flat_labels_in_fp = labels[:, footprint].ravel()
        counts = np.bincount(flat_labels_in_fp)
        best_id, best_overlap = 0, -1
        if counts.size > 1:
            counts[0] = 0
            best_id = int(counts.argmax())
            best_overlap = int(counts[best_id])
        if best_id == 0 or best_overlap <= 0:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            best_id = int(sizes.argmax())
    else:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        best_id = int(sizes.argmax())

    matched = labels == best_id
    return matched, int(matched.sum()), n_comp


def mask_to_surface(mask, downsample=2):
    try:
        import pyvista as pv
        from skimage.measure import marching_cubes
    except Exception as e:
        return None, f"PyVista/skimage unavailable ({e})"
    if not mask.any():
        return None, "empty mask"
    m = mask[::downsample, ::downsample, ::downsample] if downsample > 1 else mask
    if not m.any():
        m = mask
    try:
        verts, faces, _n, _v = marching_cubes(m.astype(np.uint8), level=0.5)
    except (ValueError, RuntimeError) as e:
        return None, str(e)
    # marching_cubes returns verts in array order (z, y, x). Reorder to (x, y, z)
    # so PyVista's world X/Y/Z match image X/Y and flow-Z, giving a natural view.
    verts = verts[:, [2, 1, 0]]
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64).ravel()
    return pv.PolyData(verts, faces_pv), None


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def render_3d(panels, colour, box, cband, inset_box, out, downsample):
    """Render the matched cluster at each timestep. When inset_box is True the
    analysed flow-crop box is drawn as a wireframe inside the full cluster.
    The mesh comes from marching_cubes on mask[::ds,::ds,::ds], so its vertex
    coordinates are in DOWNSAMPLED voxel units, axis order (z,y,x). The box
    bounds must be expressed in the same downsampled (z,y,x) frame: cluster-band
    z offset (box.z0 - cband_z0), then /downsample."""
    try:
        import pyvista as pv
    except Exception as e:
        print(f"  3D render skipped — PyVista unavailable ({e})")
        return False
    surfs, meta = [], []
    for p in panels:
        surf, err = mask_to_surface(p["mask"], downsample)
        if surf is not None and surf.n_points:
            surfs.append(surf); meta.append(p)
        else:
            print(f"    scan {p['scan_index']}: no surface ({err})")
    if not surfs:
        print("  3D render skipped — no surfaces produced.")
        return False

    ds = downsample
    box_bounds = None
    if inset_box and cband is not None:
        cz0 = cband[0]
        # verts are now (x, y, z); pv.Box wants (xmin,xmax,ymin,ymax,zmin,zmax)
        x_lo, x_hi = box["x0"] / ds, box["x1"] / ds
        y_lo, y_hi = box["y0"] / ds, box["y1"] / ds
        z_lo = (box["z0"] - cz0) / ds
        z_hi = (box["z1"] - cz0) / ds
        box_bounds = (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)

    n = len(surfs)
    pv.global_theme.background = "white"
    pv.global_theme.transparent_background = False
    pl = pv.Plotter(off_screen=True, shape=(1, n), window_size=(330 * n, 470), border=False)
    cam = None
    for i, (surf, p) in enumerate(zip(surfs, meta)):
        pl.subplot(0, i)
        pl.add_mesh(surf, color=colour, opacity=0.85 if inset_box else 1.0,
                    smooth_shading=True)
        if box_bounds is not None:
            wire = pv.Box(bounds=box_bounds)
            pl.add_mesh(wire, style="wireframe", color="#185FA5", line_width=2)
        tag = f"scan {p['scan_index']}" + ("  (X)" if p["is_X"] else "")
        pl.add_text(f"{tag}\nSw={p['sw_local']:.2f}\n{p['matched_voxels']:,} vox",
                    position="upper_left", font_size=9, color="black", font="courier")
        if cam is None:
            pl.camera_position = "iso"; pl.reset_camera(); cam = pl.camera_position
        else:
            pl.camera_position = cam
    pl.link_views()
    pl.screenshot(str(out), scale=2)
    pl.close()
    note = " (blue wireframe = analysed box)" if inset_box else ""
    print(f"  wrote {out}  ({n} panels){note}")
    return True


def render_montage(panels, box, cband, inset_box, out):
    """Side-view montage (no PyVista). Each panel is the cluster projected along
    Y, so the Z axis (flow direction) is vertical and the full Z-extent is
    visible. All panels share one fixed spatial frame so the shrinkage reads
    directly. When inset_box is True the analysed flow-crop box Z-band is shaded."""
    n = len(panels)
    b_lo = b_hi = None
    if inset_box and cband is not None:
        cz0 = cband[0]
        b_lo = box["z0"] - cz0
        b_hi = box["z1"] - cz0

    # one shared X window = union of all clusters' lateral extent, with a small
    # margin. Same window for every panel so sizes are comparable.
    nx_full = panels[0]["mask"].shape[2]
    xmin, xmax = nx_full, 0
    for p in panels:
        cols = np.where(p["mask"].any(axis=(0, 1)))[0]
        if cols.size:
            xmin = min(xmin, int(cols[0])); xmax = max(xmax, int(cols[-1]) + 1)
    if xmin >= xmax:
        xmin, xmax = 0, nx_full
    pad = max(8, (xmax - xmin) // 8)
    xmin = max(0, xmin - pad); xmax = min(nx_full, xmax + pad)
    nz_full = panels[0]["mask"].shape[0]
    width = xmax - xmin

    # figure sized so each panel's data aspect (nz tall : width wide) is honoured
    panel_w = 1.6
    panel_h = panel_w * (nz_full / width)
    fig = plt.figure(figsize=(panel_w * n, panel_h + 0.6))
    gs = gridspec.GridSpec(1, n, wspace=0.05)
    for i, p in enumerate(panels):
        ax = fig.add_subplot(gs[0, i]); ax.axis("off")
        proj = p["mask"].any(axis=1)[:, xmin:xmax]   # (nz, width) side view
        ax.imshow(proj, cmap="Greens", interpolation="nearest", vmin=0, vmax=1,
                  aspect="equal")
        ax.set_xlim(0, width); ax.set_ylim(nz_full, 0)   # identical limits all panels
        if b_lo is not None:
            ax.axhspan(b_lo, b_hi, color="#185FA5", alpha=0.13)
            ax.axhline(b_lo, color="#185FA5", lw=0.8)
            ax.axhline(b_hi, color="#185FA5", lw=0.8)
        ax.set_title(f"scan {p['scan_index']}" + ("\n(X)" if p["is_X"] else ""),
                     fontsize=8)
        ax.text(0.5, -0.03, f"$S_w$={p['sw_local']:.2f}", ha="center", va="top",
                transform=ax.transAxes, fontsize=7)
    title = ("Connected cluster, side view (Z = flow, vertical). "
             + ("Blue band = analysed flow-crop box." if b_lo is not None
                else "Box-clipped geometry (what is simulated)."))
    fig.suptitle(title, fontsize=9, y=1.0)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def render_counts(panels, track_id, out):
    sw = [p["sw_local"] for p in panels]
    matched = [p["matched_voxels"] for p in panels]
    boxall = [p["box_gas_voxels"] for p in panels]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.plot(sw, matched, "o-", color="#1D9E75", mfc="white", label="cc3d matched cluster")
    ax.plot(sw, boxall, "s--", color="#888780", mfc="white", ms=4, label="all gas in box")
    for x, y, isx in zip(sw, matched, [p["is_X"] for p in panels]):
        if isx:
            ax.annotate("X", (x, y), fontsize=8, fontweight="bold",
                        xytext=(4, 4), textcoords="offset points", color="#185FA5")
    ax.set_xlabel("Local water saturation  $S_w^{local}$")
    ax.set_ylabel("Gas voxels")
    ax.set_title(f"Track {track_id}: matched cluster vs all box gas", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="results.duckdb", type=Path)
    ap.add_argument("--run-id", default="20260618_155713_AMSDS99619")
    ap.add_argument("--track", type=int, default=4)
    ap.add_argument("--gas-label", type=int, default=GAS_LABEL_DEFAULT)
    ap.add_argument("--data-root", type=Path, default=Path("data"),
                    help="folder with the run's .am scans (default: data)")
    ap.add_argument("--output-root", type=Path, default=Path("output"),
                    help="pipeline output/ folder with clustermask + domain .raw "
                         "(default: output)")
    ap.add_argument("--source", choices=["domain", "am"], default="domain",
                    help="domain = read the pipeline's box-clipped gas .raw "
                         "(fast, exactly what is simulated); "
                         "am = re-read .am over the full cluster band and re-run "
                         "cc3d (slow, shows the full cluster with tails)")
    ap.add_argument("--max-panels", type=int, default=0,
                    help="cap on rendered timesteps (0 = all; always includes X)")
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--out", default=Path("figures"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.db.exists():
        sys.exit(f"Database not found: {args.db}")

    con = duckdb.connect(str(args.db), read_only=True)
    box = fetch_box(con, args.run_id, args.track)
    series = fetch_series(con, args.run_id, args.track)
    cm_row = fetch_clustermask(con, args.run_id, args.track)
    con.close()

    if cm_row is None:
        sys.exit("No clustermask row in cluster_properties for this track.")
    cband_full = (cm_row[1], cm_row[2])       # (cz0, cz1) full cluster Z-band
    conn = box["connectivity"]
    colour = _track_colour(args.track)

    # downselect scans (always keep X). max_panels <= 0 means use every timestep.
    x_pos = next((i for i, s in enumerate(series) if s["is_X"]), len(series) - 1)
    if args.max_panels and args.max_panels > 0 and len(series) > args.max_panels:
        picks = sorted(set(
            np.linspace(0, len(series) - 1, args.max_panels - 1).round().astype(int).tolist()
            + [x_pos]))
        chosen = [series[i] for i in picks]
    else:
        chosen = list(series)

    print(f"Run {args.run_id} | track {args.track} | connectivity {conn} | "
          f"source={args.source}")

    panels = []

    if args.source == "domain":
        # Read the pipeline's box-clipped gas domain .raw, then keep only the
        # largest CONNECTED component (the domain may contain small detached gas
        # fragments; we want the single tracked cluster).
        print(f"Reading box-clipped gas domains (the simulated geometry), "
              f"keeping the connected cluster:\n")
        cband = None                          # set per-scan from the mask shape
        for s in chosen:
            rp = _resolve_domain(s["domain_gas"], args.track, s["scan_index"],
                                 args.output_root)
            if rp is None:
                print(f"  scan {s['scan_index']:>2}: domain_gas not found "
                      f"(domain_gas_{args.track:02d}-{s['scan_index']}_*.raw) "
                      f"— skipping")
                continue
            raw_mask = load_domain_gas(rp)
            if raw_mask is None:
                continue
            # keep largest connected component
            labels = cc3d.connected_components(raw_mask.astype(np.uint8),
                                               connectivity=CC3D_CONN[conn])
            ncomp = int(labels.max())
            if ncomp == 0:
                mask = raw_mask; vox = 0
            else:
                sizes = np.bincount(labels.ravel()); sizes[0] = 0
                keep = int(sizes.argmax())
                mask = labels == keep
                vox = int(mask.sum())
            total = int(raw_mask.sum())
            panels.append({**s, "mask": mask, "matched_voxels": vox, "n_comp": ncomp})
            flag = "  <- X" if s["is_X"] else ""
            print(f"  scan {s['scan_index']:>2} | Sw={s['sw_local']:.3f} | "
                  f"connected cluster={vox:>9,} | all gas in domain={total:>9,} | "
                  f"comps={ncomp:>3}{flag}")
        inset_box = False

    else:  # args.source == "am"
        # FULL CLUSTER: re-read .am over the cluster Z-band, run cc3d, match to
        # the tracked cluster via the clustermask footprint.
        footprint = load_X_footprint(cm_row, args.output_root)
        if footprint is None:
            sys.exit("  clustermask .raw not found under --output-root "
                     f"({args.output_root}). It is required for correct matching "
                     "in --source am mode. Point --output-root at the pipeline "
                     "output/ folder, or use --source domain.")
        cband = cband_full
        print(f"Full cluster Z-band z[{cband[0]}:{cband[1]}] (extent "
              f"{cband[1]-cband[0]+1}); analysed box z[{box['z0']}:{box['z1']}] "
              f"shown as inset.")
        print(f"Re-reading .am and running cc3d on {len(chosen)} timesteps "
              f"(matched to tracked cluster):\n")
        for s in chosen:
            am = _resolve(s["file_name"], args.data_root)
            if am is None:
                print(f"  scan {s['scan_index']:>2}: .am not found "
                      f"({s['file_name']}) — skipping")
                continue
            gas = load_gas_mask(am, cband[0], cband[1], args.gas_label)
            mask, vox, ncomp = matched_component(gas, conn, footprint)
            panels.append({**s, "mask": mask, "matched_voxels": vox, "n_comp": ncomp})
            flag = "  <- X" if s["is_X"] else ""
            print(f"  scan {s['scan_index']:>2} | Sw={s['sw_local']:.3f} | "
                  f"cc3d comps={ncomp:>3} | matched={vox:>9,} | "
                  f"box gas(db)={s['box_gas_voxels']:>9,}{flag}")
        inset_box = True

    if not panels:
        sys.exit("\nNo timesteps processed. In --source domain mode check "
                 "--output-root; in --source am mode check --data-root.")

    print()
    render_montage(panels, box, cband, inset_box,
                   args.out / f"cca_shrinkage_montage_track{args.track}.png")
    render_counts(panels, args.track,
                  args.out / f"cca_shrinkage_counts_track{args.track}.png")
    render_3d(panels, colour, box, cband, inset_box,
              args.out / f"cca_shrinkage_3d_track{args.track}.png",
              args.downsample)
    print("Done.")


if __name__ == "__main__":
    main()