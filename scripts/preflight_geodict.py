"""preflight_geodict.py - READ-ONLY check of whether the GeoDict batch will run on
the domains under output/. Touches NOTHING - deletes nothing, writes nothing.

It reproduces every gate the batch applies and reports, per domain, whether it
WOULD RUN or be blocked, and by which gate:

  gate 1  filename matches the domain_<...>.raw convention
  gate 2  the scan token parses as an integer (needed to key the DB row)
  gate 3  a result_summary.json already exists with a permeability (resume-skip)
          - and whether that summary MATCHES the domain (real) or MISMATCHES (stale)
  gate 4  the DB has the run_id, and Sw can be looked up (affects writeback only)

Usage (from the THESIS / pipeline folder):
    python preflight_geodict.py
    python preflight_geodict.py --output-root "C:\\...\\output" --connectivity 18N
    python preflight_geodict.py --db results.duckdb --run-id 20260626_xxxxx

Nothing is modified. To actually clear stale summaries, use why_skipped.py
--delete-stale, or the batch's --force - but only after this confirms the picture.
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path

_NAME_RE = re.compile(
    r"^domain_(?P<dtype>absolute|gas|water)_(?P<track>\d+)-(?P<scan>.+?)_"
    r"(?P<voxel>[0-9.]+)um_(?P<bits>\d+b[us])_(?P<nx>\d+)x(?P<ny>\d+)x(?P<nz>\d+)\.raw$",
    re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--connectivity", default=None)
    ap.add_argument("--db", default="results.duckdb")
    ap.add_argument("--run-id", default=None,
                    help="run these domains belong to (default: latest in DB)")
    args = ap.parse_args()

    root = Path(args.output_root)
    if not root.exists():
        print(f"output root not found: {root.resolve()} - pass --output-root")
        return

    # ---- gate 1: find + parse domains ----
    all_raw = sorted(root.rglob("domain_*.raw"))
    if args.connectivity:
        all_raw = [d for d in all_raw if d.parent.name.lower() == args.connectivity.lower()]
    if not all_raw:
        print(f"No domain_*.raw under {root.resolve()}"
              + (f" / {args.connectivity}" if args.connectivity else "")
              + "\nGate 1 FAILS for everything: the pipeline did not export domains here,")
        print("or they are under a different connectivity folder. Nothing for the batch to run.")
        return

    unmatched = [d for d in all_raw if not _NAME_RE.match(d.name)]
    matched = [d for d in all_raw if _NAME_RE.match(d.name)]

    # ---- optional DB connection for gate 4 ----
    con = None; run_id = args.run_id
    try:
        import duckdb
        con = duckdb.connect(args.db, read_only=True)
        if run_id is None:
            row = con.execute("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
            run_id = row[0] if row else None
    except Exception as e:
        print(f"(note: could not open DB for gate-4 checks: {e})\n")

    print(f"output root : {root.resolve()}")
    print(f"connectivity: {args.connectivity or '(all)'}")
    print(f"DB run_id   : {run_id}")
    print(f"domains     : {len(all_raw)} found, {len(matched)} parse, {len(unmatched)} unmatched\n")

    if unmatched:
        print("GATE 1 - filenames NOT matching convention (batch ignores these):")
        for d in unmatched[:20]:
            print(f"  {d.name}")
        print()

    will_run = 0; blocked_done = 0; blocked_stale = 0
    bad_scan = 0
    print(f"{'verdict':<10}{'gate':<14} domain")
    print("-" * 96)

    for d in matched:
        m = _NAME_RE.match(d.name)
        track, scan = m["track"], m["scan"]
        f_nx, f_ny, f_nz = int(m["nx"]), int(m["ny"]), int(m["nz"])
        f_vox = float(m["voxel"])

        # gate 2: scan token must be int-able
        try:
            int(scan)
        except ValueError:
            bad_scan += 1
            print(f"{'BLOCKED':<10}{'gate2 scan!=int':<14} {d.name}  (scan token '{scan}')")
            continue

        # gate 3: resume-skip via result_summary.json
        scan_dir = d.parent.parent
        summary = scan_dir / "geodict" / d.stem / "result_summary.json"
        if summary.exists():
            try:
                data = json.loads(summary.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            perm = data.get("permeability_z_m2")
            if perm is not None:
                # matches or stale?
                s_dims = (data.get("nx"), data.get("ny"), data.get("nz"))
                s_vox = data.get("voxel_m")
                mismatch = (s_dims != (f_nx, f_ny, f_nz) and None not in s_dims) or \
                           (s_vox is not None and abs(s_vox * 1e6 - f_vox) > 1e-3)
                if mismatch:
                    blocked_stale += 1
                    print(f"{'BLOCKED':<10}{'gate3 STALE':<14} {d.name}")
                    print(f"{'':10}{'':14}   file {f_nx}x{f_ny}x{f_nz}@{f_vox} vs "
                          f"summary {s_dims[0]}x{s_dims[1]}x{s_dims[2]}"
                          f"@{(s_vox*1e6 if s_vox else '?')}")
                else:
                    blocked_done += 1
                    print(f"{'BLOCKED':<10}{'gate3 done':<14} {d.name}  (perm={perm:.2e})")
                continue

        # passed all gates -> would run
        will_run += 1
        # gate 4 advisory: DB writeback
        note = ""
        if con is not None and run_id is not None:
            try:
                r = con.execute(
                    "SELECT COUNT(*) FROM fixed_boxes WHERE run_id=? AND track_id=? "
                    "AND connectivity=?", [run_id, int(track), d.parent.name]).fetchone()
                if r[0] == 0:
                    note = "  (warn: no fixed_boxes row -> Sw/writeback will be NULL)"
            except Exception:
                pass
        print(f"{'RUN':<10}{'-':<14} {d.name}{note}")

    print("-" * 96)
    print(f"WILL RUN : {will_run}")
    print(f"BLOCKED  : done={blocked_done}  stale={blocked_stale}  bad_scan={bad_scan}")
    print()
    if will_run == 0 and (blocked_done or blocked_stale):
        if blocked_stale and not blocked_done:
            print(">> Everything blocked by STALE summaries (leftover from a previous run on")
            print("   the same scan-folder names). The new data WOULD run once those are")
            print("   cleared. Use the batch's --force to run without deleting, or")
            print("   why_skipped.py --delete-stale to remove only the mismatched summaries.")
        elif blocked_done:
            print(">> Domains are blocked by summaries that MATCH them - they really were")
            print("   solved on this exact data. If you expected NEW data, the domains under")
            print("   output/ are themselves from the old run; re-run the pipeline on the new")
            print("   .am files to regenerate domains before simulating.")
    elif will_run > 0:
        print(f">> {will_run} domains will run as-is. Use --force only if you also want to")
        print("   re-run the BLOCKED ones.")
    if bad_scan:
        print(f">> {bad_scan} domains have a non-integer scan token and the batch will skip")
        print("   them (cannot key the DB row). Tell me the filename and we fix the parsing.")

    if con is not None:
        con.close()


if __name__ == "__main__":
    main()
