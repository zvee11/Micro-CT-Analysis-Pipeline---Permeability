"""
split_check.py  —  diagnostic, no pipeline changes.

Track 1's cluster shows 2 spanning components at timestep X (a split). This
tests whether that split is REAL or an artifact of the 10% Z-crop: it reloads
the ORIGINAL .am volume at X, extracts Track 1's frozen box, and compares the
spanning components with 0 vs ±PAD extra Z-slices outside the box.

If the 2 pieces MERGE into 1 when padding is added -> the connection runs
through cropped-off slices -> the split is a CROP ARTIFACT.
If they stay 2 -> the split is physically real.

    python split_check.py                 # track 1, X scan auto-detected, pad 20
    python split_check.py --track 1 --pad 20

Run from the THESIS folder (same place you run `python -m pipeline`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import duckdb

# Use the pipeline's own loaders so coordinates match exactly.
try:
    from pipeline.io import read_avizo, iter_input_files
    from pipeline.preprocessing import detect_and_fix_x_wraparound
    from pipeline.config import Config
    import cc3d
    import logging
except ImportError as e:
    print(f"ERROR importing pipeline modules / cc3d: {e}")
    print("Run this from the THESIS folder (where `python -m pipeline` works).")
    sys.exit(1)


def spanning_info(gas_mask: np.ndarray):
    """(n_components, spanning_count, [sizes]) for a 0/1 gas mask, Z = axis 0."""
    labels = cc3d.connected_components(gas_mask.astype(np.uint8), connectivity=26)
    n = int(labels.max())
    if n == 0:
        return 0, 0, []
    top = set(np.unique(labels[0])) - {0}
    bot = set(np.unique(labels[-1])) - {0}
    span = top & bot
    sizes = sorted((int((labels == c).sum()) for c in span), reverse=True)
    return n, len(span), sizes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="results.duckdb")
    ap.add_argument("--track", type=int, default=1)
    ap.add_argument("--connectivity", default="26N")
    ap.add_argument("--pad", type=int, default=20, help="extra Z-slices each side")
    ap.add_argument("--run", default=None)
    args = ap.parse_args()

    cfg = Config()
    logger = logging.getLogger("split_check")
    logging.basicConfig(level=logging.WARNING)

    con = duckdb.connect(args.db, read_only=True)
    run_id = args.run or con.execute(
        "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()[0]

    # The X scan row for this track: is_X in scans table; box coords in fixed_boxes.
    x_stem = con.execute(
        "SELECT file_name FROM scans WHERE run_id=? AND is_X=TRUE LIMIT 1", [run_id]).fetchone()
    if x_stem is None:
        print("Could not find timestep-X scan (is_X) in DB."); sys.exit(1)
    x_stem = x_stem[0]

    box = con.execute(
        """SELECT z0,z1,y0,y1,x0,x1 FROM fixed_boxes
           WHERE run_id=? AND connectivity=? AND track_id=?
           ORDER BY scan_index DESC LIMIT 1""",
        [run_id, args.connectivity, args.track]).fetchone()
    con.close()
    if box is None:
        print(f"No fixed_box for track {args.track}."); sys.exit(1)
    z0, z1, y0, y1, x0, x1 = box
    print(f"track {args.track} | X scan '{x_stem}' | box Z[{z0}-{z1}] Y[{y0}-{y1}] X[{x0}-{x1}]\n")

    # Locate the original .am for the X scan.
    am = None
    for f in iter_input_files(cfg):
        if f.stem == x_stem:
            am = f; break
    if am is None:
        print(f"Original .am for '{x_stem}' not found in {cfg.data_dir}."); sys.exit(1)

    print(f"loading {am.name} ...")
    vol, _, _ = read_avizo(am, parse_spacing=False, memmap_raw=False)
    vol = detect_and_fix_x_wraparound(vol, logger)
    Z = vol.shape[0]
    print(f"volume Z extent = {Z}\n")

    gl = cfg.gas_label
    # Box as saved (z1 inclusive in DB) -> +1 for slicing.
    z1i = z1 + 1

    for pad in (0, args.pad):
        zlo = max(0, z0 - pad)
        zhi = min(Z, z1i + pad)
        sub = vol[zlo:zhi, y0:y1 + 1, x0:x1 + 1]
        gas = (sub == gl)
        n, span, sizes = spanning_info(gas)
        label = "BOX ONLY (pad 0)" if pad == 0 else f"BOX +/- {pad} slices"
        print(f"{label}: Z[{zlo}-{zhi-1}] ({zhi-zlo} slices) | "
              f"components {n} | spanning {span} | sizes {sizes[:4]}")

    print("\nInterpretation:")
    print("  spanning drops 2 -> 1 when padded  => split is a CROP ARTIFACT "
          "(connection runs through cropped slices)")
    print("  spanning stays 2 when padded       => split is REAL")


if __name__ == "__main__":
    main()
