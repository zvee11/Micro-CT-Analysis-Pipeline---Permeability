"""
raw_cc_check.py  —  isolate the .am-loader / wrap-around question.

Reads a segmented .raw volume DIRECTLY (no .am parsing, no wrap-around
correction) and runs the pipeline's own slab-based connected-components
(topn_gas_cc) on it. Prints the top-N gas cluster sizes so you can compare
against (a) your pipeline's 18N/26N output and (b) the Avizo labelAna clusters.

If the top-N here MATCH your pipeline output  -> the loader is fine.
If they match Avizo but NOT your pipeline      -> the .am loader/wrap-around
                                                  is altering the data.

Run from the THESIS folder (so `pipeline` is importable):

    python raw_cc_check.py 9_5_sub_registered_filtered_thresholded_extracted.raw
    python raw_cc_check.py <file.raw> --conn 18N --nkeep 6
    python raw_cc_check.py <file.raw> --shape 3780 750 750

Defaults: shape 3780x750x750, uint8, gas label = 2.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# connectivity name -> the integer cc3d uses, matching the pipeline's convention
CONN = {"6N": 1, "18N": 2, "26N": 3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help="path to the segmented .raw volume")
    ap.add_argument("--shape", type=int, nargs=3, default=[3780, 750, 750],
                    metavar=("Z", "Y", "X"))
    ap.add_argument("--dtype", default="uint8")
    ap.add_argument("--gas", type=int, default=2, help="gas label value")
    ap.add_argument("--conn", default="18N", choices=list(CONN))
    ap.add_argument("--nkeep", type=int, default=6)
    ap.add_argument("--slab", type=int, default=128, help="slab depth (memory)")
    args = ap.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.exists():
        sys.exit(f"ERROR: {raw_path} not found")

    Z, Y, X = args.shape
    expected = Z * Y * X * np.dtype(args.dtype).itemsize
    actual = raw_path.stat().st_size
    if expected != actual:
        sys.exit(f"ERROR: shape {args.shape} x {args.dtype} = {expected:,} bytes "
                 f"but file is {actual:,} bytes. Fix --shape/--dtype.")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("raw_cc")

    # Import the pipeline's OWN slab cc3d so the algorithm is identical to a run.
    try:
        from pipeline.connected import topn_gas_cc
    except Exception as e:
        sys.exit(f"Could not import pipeline.connected.topn_gas_cc: {e}\n"
                 f"Run this from the THESIS folder.")

    print(f"reading RAW directly (no .am, no wrap-around): {raw_path.name}")
    print(f"shape {Z}x{Y}x{X}  dtype {args.dtype}  gas={args.gas}  "
          f"conn={args.conn}  n_keep={args.nkeep}\n")

    # memmap so we don't load 2 GB into RAM up front; topn_gas_cc reads it slab-wise
    vol = np.memmap(str(raw_path), dtype=args.dtype, mode="r", shape=(Z, Y, X))

    labels_out, report = topn_gas_cc(
        vol=vol,
        gas_label=args.gas,
        connectivity=CONN[args.conn],
        slab_depth=args.slab,
        n_keep=args.nkeep,
        logger=logger,
    )

    # report is [(final_label, size), ...]; sort by size desc
    report_sorted = sorted(report, key=lambda r: r[1], reverse=True)
    print(f"\n=== TOP-{args.nkeep} GAS CLUSTERS (cc3d on RAW, {args.conn}) ===")
    print(f"{'rank':>4} | {'label':>6} | {'voxels':>14}")
    print("-" * 34)
    for i, (lbl, size) in enumerate(report_sorted[:args.nkeep], 1):
        print(f"{i:>4} | {lbl:>6} | {size:>14,}")
    total_gas = int(sum(s for _, s in report))
    print(f"\n(top-{args.nkeep} captured; for reference compare these to your "
          f"pipeline {args.conn} gas_voxels_at_X and to Avizo Volume3d/voxel_vol)")


if __name__ == "__main__":
    main()
