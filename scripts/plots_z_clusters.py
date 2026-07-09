"""
plot_cluster_from_mask.py — plot a cluster's gas-voxels-per-z-slice profile by
reading the clustermask .raw file the pipeline ALREADY saves. No CC re-run, no
.am volume, no pipeline change. Loads in under a second.

The normal pipeline writes, per cluster at timestep X:
    output/<scan>/<conn>/cluster_<NN>_clustermask_<conn>.raw   (uint8, 0/1)
and stores its path + z-bounds in results.duckdb (cluster_properties.clustermask_*).
This script looks those up, reads the .raw, sums each z-slice, redraws the crop,
and plots the taper — the shape the DB report alone can't show.

Run from your THESIS directory:

    python plot_cluster_from_mask.py --scan 3 --label 4
    python plot_cluster_from_mask.py --scan 3 --label 4 --threshold 0.10
    python plot_cluster_from_mask.py --scan 3 --label 4 --save track4.png
    python plot_cluster_from_mask.py --scan 3 --all          # every cluster at scan 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import duckdb


def latest_run(con):
    r = con.execute("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
    return r[0] if r else None


def crop_bounds(slice_counts, frac):
    """Reproduce find_flow_crop_z: first..last slice >= peak*frac (inclusive idx)."""
    peak = int(slice_counts.max())
    if peak == 0:
        return None
    qual = np.where(slice_counts >= peak * frac)[0]
    if qual.size == 0:
        return None
    return int(qual[0]), int(qual[-1])


def load_profile(con, run_id, scan, label, conn):
    row = con.execute("""
        SELECT clustermask_raw, clustermask_z0, clustermask_z1, crop_z0, crop_z1,
               connectivity, track_id
        FROM cluster_properties
        WHERE run_id=? AND scan_index=? AND label_id=?
        """ + ("AND connectivity=?" if conn else "") + """
        ORDER BY connectivity LIMIT 1
    """, [run_id, scan, label] + ([conn] if conn else [])).fetchone()
    if not row:
        return None
    raw_path, cm_z0, cm_z1, cz0, cz1, conn_name, track_id = row
    if not raw_path or not Path(raw_path).exists():
        raise SystemExit(f"clustermask file missing on disk: {raw_path}")
    if cm_z0 is None or cm_z1 is None:
        raise SystemExit("clustermask z-bounds are NULL in the DB for this cluster.")

    n_z = cm_z1 - cm_z0 + 1                       # inclusive bounds
    flat = np.fromfile(raw_path, dtype=np.uint8)  # 0/1 mask, shape (n_z, Y, X) flattened
    if flat.size % n_z != 0:
        raise SystemExit(f"raw size {flat.size} not divisible by n_z {n_z} — "
                         f"shape mismatch; check the DB z-bounds.")
    per_slice = flat.reshape(n_z, -1).sum(axis=1).astype(np.int64)  # gas voxels per z-slice
    return {
        "z0_bbox": cm_z0, "z1_bbox": cm_z1, "conn": conn_name, "track_id": track_id,
        "crop_z0_db": cz0, "crop_z1_db": cz1, "slice_counts": per_slice,
    }


def plot_one(d, label, scan, thr, ax):
    counts = d["slice_counts"]
    z0b = d["z0_bbox"]
    z = list(range(z0b, z0b + len(counts)))
    peak = int(counts.max())

    # crop at the requested threshold (recomputed from the mask itself)
    cb = crop_bounds(counts, thr)
    if cb:
        z0_crop, z1_crop = z0b + cb[0], z0b + cb[1]
    else:
        z0_crop, z1_crop = z0b, z0b + len(counts) - 1

    bbox_span = len(counts)
    crop_span = z1_crop - z0_crop + 1
    kept = 100.0 * crop_span / bbox_span if bbox_span else 0.0

    ax.plot(z, counts, color="#1f6feb", lw=1.5, label="gas voxels / slice")
    ax.axhline(peak * thr, ls="--", color="#d29922", lw=1.0, label=f"cut = {thr:.0%} of peak")
    ax.axvspan(z0_crop, z1_crop, color="#2ea043", alpha=0.15, label=f"kept z[{z0_crop}-{z1_crop}]")
    ax.axvline(z0_crop, color="#2ea043", lw=0.8)
    ax.axvline(z1_crop, color="#2ea043", lw=0.8)
    tid = d["track_id"]
    ax.set_title(f"scan {scan} · label {label}" + (f" · track {tid}" if tid is not None else "")
                 + f" · {d['conn']}\nbbox span {bbox_span} → kept {crop_span} ({kept:.0f}%)",
                 fontsize=9)
    ax.set_xlabel("z slice (absolute)")
    ax.set_ylabel("gas voxels")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.2)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="results.duckdb")
    ap.add_argument("--run", default=None)
    ap.add_argument("--scan", type=int, required=True)
    ap.add_argument("--label", type=int, default=None, help="single cluster label (1-based)")
    ap.add_argument("--all", action="store_true", help="plot every cluster at this scan")
    ap.add_argument("--connectivity", default=None, help="e.g. 18N (default: whatever is stored)")
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    if not args.all and args.label is None:
        raise SystemExit("give --label N, or --all")

    con = duckdb.connect(args.db, read_only=True)
    run_id = args.run or latest_run(con)
    if not run_id:
        raise SystemExit("no runs in DB")

    if args.all:
        labels = [r[0] for r in con.execute(
            "SELECT DISTINCT label_id FROM cluster_properties WHERE run_id=? AND scan_index=? ORDER BY label_id",
            [run_id, args.scan]).fetchall()]
    else:
        labels = [args.label]
    if not labels:
        raise SystemExit(f"no clusters for run {run_id} scan {args.scan}")

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import math

    n = len(labels)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.4 * rows), squeeze=False)
    for i, lbl in enumerate(labels):
        d = load_profile(con, run_id, args.scan, lbl, args.connectivity)
        ax = axes[i // cols][i % cols]
        if d is None:
            ax.set_title(f"label {lbl}: not found"); ax.axis("off"); continue
        kept = plot_one(d, lbl, args.scan, args.threshold, ax)
        print(f"label {lbl}: kept {kept:.0f}%")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"run {run_id}", fontsize=8, y=0.998)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=130, bbox_inches="tight")
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
