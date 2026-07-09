"""
run_geodict_batch.py  —  outer orchestrator (normal Python, runs on the workstation).

Walks the pipeline's output/ tree, finds every exported flow domain, and runs the
GeoPy macro `geodict_lir_job.py` in GeoDict ONCE PER DOMAIN (separate launch each,
so one failure does not stop the rest). For each domain it generates a per-domain
copy of the macro with raw_path swapped in (the macro derives nx/ny/nz/voxel and the
result folder from the filename itself).

Each domain's permeability and solver flags are read from the result_summary.json
the macro writes under output/<scan>/geodict/<domain_stem>/, and stored RAW in the
DuckDB simulation_results table (simulator='GeoDict'). No kr post-processing is done
here: k_z is stored, kr is left NULL, and the solver flags + metadata go into notes.
Sw is joined from fixed_boxes for convenience.

Domain filename convention:
    domain_<type>_<track>-<scan>_<voxel>um_<bits>_<NX>x<NY>x<NZ>.raw
e.g. domain_gas_02-8_4.99676um_8bu_750x750x273.raw

Usage:
    python run_geodict_batch.py ^
        --output-root  "C:\\...\\output" ^
        --geodict-exe  "C:\\Program Files\\Math2Market GmbH\\GeoDict 2026\\geodict2026.exe" ^
        --macro        "C:\\...\\scripts\\geodict_lir_job.py" ^
        --result-root  "C:\\...\\results_geodict" ^
        --db           "C:\\...\\results.duckdb" ^
        --run-id       <run_id> ^
        [--connectivity 18N] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# domain_<type>_<track>-<scan>_<voxel>_<bits>_<NX>x<NY>x<NZ>.raw
# scan may be a plain index (e.g. 7) or contain underscores; voxel is decimal
# (e.g. 4.99684um). Anchor on the trailing _<voxel>um_<bits>_<dims>.
_NAME_RE = re.compile(
    r"^domain_(?P<dtype>absolute|gas|water)_"
    r"(?P<track>\d+)-(?P<scan>.+?)_"
    r"(?P<voxel>[0-9.]+)um_"
    r"(?P<bits>\d+b[us])_"
    r"(?P<nx>\d+)x(?P<ny>\d+)x(?P<nz>\d+)\.raw$",
    re.IGNORECASE,
)


def parse_domain(path: Path):
    """Return a dict of parsed fields, or None if the name does not match."""
    m = _NAME_RE.match(path.name)
    if not m:
        return None
    g = m.groupdict()
    return {
        "path": path,
        "dtype": g["dtype"].lower(),
        "track": g["track"],
        "scan": g["scan"],
        "voxel_um": float(g["voxel"]),
        "bits": g["bits"],
        "nx": int(g["nx"]),
        "ny": int(g["ny"]),
        "nz": int(g["nz"]),
    }


def discover(output_root: Path, connectivity: str | None):
    """Group parsed domains by (track, scan[, connectivity])."""
    groups: dict[tuple, dict] = defaultdict(dict)
    for raw in output_root.rglob("domain_*.raw"):
        if connectivity and raw.parent.name.lower() != connectivity.lower():
            continue
        info = parse_domain(raw)
        if not info:
            print(f"  [skip] name did not match convention: {raw.name}")
            continue
        key = (info["track"], info["scan"], raw.parent.name)
        groups[key][info["dtype"]] = info
    return groups


def make_macro_for_domain(template_macro: Path, raw_path: Path, work_dir: Path) -> Path:
    """Copy the real macro and swap its hardcoded raw_path line to this domain.
    Returns the path to the generated per-domain macro.
    The macro derives nx/ny/nz/voxel and result_dir from raw_path itself, so only
    that one line needs replacing.
    """
    text = template_macro.read_text(encoding="utf-8")
    # The macro sets:  raw_path = '...'   (single line at the top of main()).
    raw_win = str(raw_path).replace("\\", "/")
    new_line = f"    raw_path = '{raw_win}'"
    new_text, n = re.subn(r"^    raw_path = .*$", new_line, text, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError("could not find the 'raw_path = ...' line in the macro template")
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / "geodict_job.py"
    out.write_text(new_text, encoding="utf-8")
    return out


def run_one(geodict_exe: Path, template_macro: Path, domain: dict,
            result_dir: Path, dry_run: bool) -> Path:
    """Generate a per-domain macro, invoke GeoDict on it, return the
    result_summary.json path the macro writes (inside output/<scan>/geodict/<stem>/).
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    job_macro = make_macro_for_domain(template_macro, domain["path"], result_dir)
    cmd = [str(geodict_exe), str(job_macro)]
    print("    " + " ".join(cmd))

    # The macro writes result_summary.json next to the domain, under
    # output/<scan_folder>/geodict/<domain_stem>/.
    scan_dir = domain["path"].parent.parent           # .../<scan>
    domain_stem = domain["path"].stem
    summary = scan_dir / "geodict" / domain_stem / "result_summary.json"

    if dry_run:
        return summary

    log = result_dir / "geodict_run.log"
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log.write_text(
        "CMD:\n" + " ".join(cmd) + "\n\nSTDOUT:\n" + proc.stdout +
        "\n\nSTDERR:\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"GeoDict failed for {domain['path'].name} (see {log})")
    return summary


def read_summary(summary_json: Path) -> dict:
    """Return the full result_summary.json dict (everything the macro stored),
    or an empty dict if missing/unreadable."""
    if not summary_json.exists():
        return {}
    try:
        return json.loads(summary_json.read_text(encoding="utf-8"))
    except Exception:
        return {}


def lookup_sw(con, run_id, scan_index, track_id, connectivity):
    """Sw for this (scan, track, connectivity) from fixed_boxes.sw_local."""
    try:
        r = con.execute(
            "SELECT sw_local FROM fixed_boxes WHERE run_id=? AND scan_index=? "
            "AND track_id=? AND connectivity=?",
            (run_id, scan_index, track_id, connectivity)).fetchone()
        return float(r[0]) if r and r[0] is not None else None
    except Exception:
        return None


def write_sim_result(con, run_id, scan_index, track_id, connectivity,
                     sim_type, summary, sw, domain_path, gdr_path):
    """Upsert one row into simulation_results (simulator='GeoDict'), storing
    everything the macro reported. No kr post-processing: kr is left NULL.
    The full summary json and the solver flags go into notes."""
    k_z = summary.get("permeability_z_m2")
    stopping = summary.get("stopping_criteria_flag")
    reached = summary.get("reached_error_bound_z")
    notes = json.dumps({
        "stopping_criteria_flag": stopping,
        "reached_error_bound_z": reached,
        "voxel_m": summary.get("voxel_m"),
        "nx": summary.get("nx"), "ny": summary.get("ny"), "nz": summary.get("nz"),
        "geodict_version": summary.get("geodict_version"),
        "result_txt": summary.get("result_txt"),
    })
    con.execute(
        "DELETE FROM simulation_results WHERE run_id=? AND scan_index=? AND "
        "track_id=? AND connectivity=? AND sim_type=? AND simulator='GeoDict'",
        (run_id, scan_index, track_id, connectivity, sim_type))
    con.execute(
        "INSERT INTO simulation_results (run_id, scan_index, track_id, "
        "connectivity, sim_type, simulator, k_z, k_eff, kr, Sw, domain_path, "
        "raw_output_path, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, scan_index, track_id, connectivity, sim_type, "GeoDict",
         k_z, k_z, None, sw, domain_path, gdr_path, notes))


# ─────────────────────────────────────────────────────────────────────────────
# Interactive selection (mirrors solve.py's ask() pattern; reads the DB to show
# what is available, then lets the user choose coarsely which sections to run).
# ─────────────────────────────────────────────────────────────────────────────

def _ask(prompt, choices=None, default=None):
    if choices:
        opts = "/".join(f"[{c}]" if c == default else c for c in choices)
        full = f"{prompt} ({opts}): "
    elif default is not None:
        full = f"{prompt} [{default}]: "
    else:
        full = f"{prompt}: "
    while True:
        try:
            raw = input(full).strip()
        except EOFError:
            raw = ""
        if not raw and default is not None:
            return default
        if choices and raw.lower() not in [c.lower() for c in choices]:
            print(f"  enter one of: {', '.join(choices)}")
            continue
        return raw


def _parse_int_list(s):
    """'1,2,5' -> {1,2,5}; '0-7' -> {0..7}; '' / 'all' -> None (= everything)."""
    s = (s or "").strip().lower()
    if s in ("", "all"):
        return None
    out = set()
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def profile_run(con, run_id, connectivity):
    """Per-track summary for the interactive table: scan count, dims, Sw range,
    done/undone counts, min z-extent (for the thinness flag). Reads fixed_boxes
    (per track/scan geometry + Sw) and scans (is_X / qualifying)."""
    conn_filter = "AND connectivity = ?" if connectivity else ""
    params = [run_id] + ([connectivity] if connectivity else [])
    rows = con.execute(f"""
        SELECT track_id, connectivity,
               COUNT(DISTINCT scan_index)      AS n_scans,
               MIN(extent_z)                   AS min_ext_z,
               MAX(extent_z)                   AS max_ext_z,
               MIN(sw_local)                   AS sw_min,
               MAX(sw_local)                   AS sw_max,
               MIN(extent_x)                   AS ext_x,
               MIN(extent_y)                   AS ext_y
        FROM fixed_boxes
        WHERE run_id = ? {conn_filter}
        GROUP BY track_id, connectivity
        ORDER BY track_id, connectivity
    """, params).fetchall()
    # which scan is X (per run)
    xrow = con.execute(
        "SELECT scan_index FROM scans WHERE run_id=? AND is_X=TRUE LIMIT 1", [run_id]
    ).fetchone()
    x_scan = xrow[0] if xrow else None
    return rows, x_scan


def interactive_select(con, run_id, connectivity, groups, thin_cutoff=200):
    """Show a per-track table and prompt for tracks / scans / domain-types /
    skip-done / thin-handling. Returns a dict of selection criteria the main
    loop applies. con must be a live (non-dry-run) DB connection."""
    rows, x_scan = profile_run(con, run_id, connectivity)
    if not rows:
        print("No fixed_boxes for this run — nothing to select. Falling back to flags.")
        return None

    print("\n" + "=" * 78)
    print(f"  SECTIONS AVAILABLE  ·  run {run_id}"
          + (f"  ·  {connectivity}" if connectivity else "")
          + (f"  ·  X = scan {x_scan}" if x_scan is not None else ""))
    print("=" * 78)
    print(f"  {'trk':>3} {'conn':>5} {'scans':>5} {'z-ext':>11} {'Sw range':>15} "
          f"{'dims(x,y)':>12}  flag")
    print("  " + "-" * 74)
    track_ids = []
    for (trk, conn, n_scans, min_ez, max_ez, sw_min, sw_max, ex, ey) in rows:
        track_ids.append(trk)
        ez = f"{min_ez}-{max_ez}" if min_ez != max_ez else f"{min_ez}"
        swr = (f"{sw_min:.3f}-{sw_max:.3f}"
               if sw_min is not None and sw_max is not None else "n/a")
        flag = "THIN" if (min_ez is not None and min_ez < thin_cutoff) else ""
        print(f"  {trk:>3} {conn:>5} {n_scans:>5} {ez:>11} {swr:>15} "
              f"{f'{ex}x{ey}':>12}  {flag}")
    print("  " + "-" * 74)
    print(f"  THIN = min z-extent < {thin_cutoff} voxels (e.g. boundary-touching / thin "
          f"sections)\n")

    # 1) tracks
    sel = _ask("Tracks to run (e.g. 1,2,5 or a-b range, Enter=all)", default="all")
    sel_tracks = _parse_int_list(sel)

    # 2) scans
    scan_mode = _ask("Scans", choices=["all", "qualifying", "range"], default="all")
    sel_scans = None
    qualifying_only = False
    if scan_mode == "qualifying":
        qualifying_only = True
    elif scan_mode == "range":
        sel_scans = _parse_int_list(
            _ask("  scan range/list (e.g. 0-7 or 1,3,5)", default="all"))

    # 3) domain types
    dtype_choice = _ask("Domain types", choices=["gas", "water", "absolute", "all"],
                        default="all")
    if dtype_choice == "all":
        sel_dtypes = {"absolute", "gas", "water"}
    else:
        sel_dtypes = {dtype_choice}

    # 4) skip already-done
    skip_done = _ask("Skip sections already solved?",
                     choices=["y", "n"], default="y").lower() == "y"

    # 5) thin handling
    thin_mode = _ask(f"Sections with z-extent < {thin_cutoff}",
                     choices=["include", "exclude"], default="include")
    exclude_thin = (thin_mode == "exclude")

    # thin track ids (min z-extent below cutoff)
    thin_tracks = {trk for (trk, conn, n_scans, min_ez, *_ ) in rows
                   if min_ez is not None and min_ez < thin_cutoff}

    print("\nSelection summary:")
    print(f"  tracks      : {'all' if sel_tracks is None else sorted(sel_tracks)}")
    print(f"  scans       : {scan_mode}"
          + (f" {sorted(sel_scans)}" if sel_scans else ""))
    print(f"  domain types: {sorted(sel_dtypes)}  (absolute runs once per track at X)")
    print(f"  skip done   : {skip_done}")
    print(f"  thin (<{thin_cutoff}): {'excluded' if exclude_thin else 'included'}"
          + (f" {sorted(thin_tracks)}" if thin_tracks else ""))
    go = _ask("Proceed?", choices=["y", "n"], default="y").lower()
    if go != "y":
        sys.exit("aborted by user.")

    return {
        "tracks": sel_tracks,                 # None = all
        "scans": sel_scans,                   # None unless explicit range/list
        "qualifying_only": qualifying_only,
        "dtypes": sel_dtypes,
        "skip_done": skip_done,
        "exclude_thin": exclude_thin,
        "thin_tracks": thin_tracks,
        "x_scan": x_scan,
    }


def qualifying_scan_set(con, run_id):
    """Set of scan indices flagged qualifying in the DB (for scans=qualifying)."""
    rows = con.execute(
        "SELECT scan_index FROM scans WHERE run_id=? AND qualifying=TRUE", [run_id]
    ).fetchall()
    return {r[0] for r in rows}


def _scan_prefix(file_name):
    """'8_2_sub_registered_filtered.am' -> '8_2' (leading numeric underscore groups)."""
    toks = []
    for p in str(file_name).split("_"):
        if p.isdigit():
            toks.append(p)
        else:
            break
    return "_".join(toks) if toks else str(file_name)


def list_runs(con):
    """Distinct runs with connectivity, scan/qualifying counts, track count, and
    the scan names (8_2-style prefixes). Returns a list of dicts, newest first."""
    rows = con.execute("""
        SELECT r.run_id, r.started_at, r.crop_mode, r.regime_cutoff,
               (SELECT COUNT(*) FROM scans s WHERE s.run_id=r.run_id) n_scans,
               (SELECT COUNT(*) FROM scans s WHERE s.run_id=r.run_id AND s.qualifying) n_qual,
               (SELECT string_agg(DISTINCT connectivity, ',')
                  FROM fixed_boxes f WHERE f.run_id=r.run_id) conns,
               (SELECT COUNT(DISTINCT track_id)
                  FROM fixed_boxes f WHERE f.run_id=r.run_id) n_tracks
        FROM runs r
        ORDER BY r.started_at DESC
    """).fetchall()
    out = []
    for (rid, started, crop, regime, n_scans, n_qual, conns, n_tracks) in rows:
        names = [
            _scan_prefix(x[0]) for x in con.execute(
                "SELECT file_name FROM scans WHERE run_id=? ORDER BY scan_index", [rid]
            ).fetchall()
        ]
        out.append({
            "run_id": rid, "started": started, "crop_mode": crop,
            "regime_cutoff": regime, "n_scans": n_scans, "n_qual": n_qual,
            "conns": conns or "?", "n_tracks": n_tracks, "scan_names": names,
        })
    return out


def pick_run(con):
    """Show the distinct runs in the DB and let the user choose one.
    Returns the chosen run_id, or exits."""
    runs = list_runs(con)
    if not runs:
        sys.exit("No runs found in the database.")

    print("\n" + "=" * 80)
    print("  RUNS IN DATABASE")
    print("=" * 80)
    for i, r in enumerate(runs, start=1):
        print(f"  [{i}] {r['run_id']}")
        print(f"      {str(r['started'])[:19]}  ·  conn={r['conns']}  ·  "
              f"{r['n_tracks']} tracks  ·  {r['n_scans']} scans "
              f"({r['n_qual']} qualifying)  ·  crop={r['crop_mode']}  ·  "
              f"regime={r['regime_cutoff']}")
        names = r["scan_names"]
        shown = ", ".join(names[:16]) + (" ..." if len(names) > 16 else "")
        print(f"      scans: {shown}")
    print("=" * 80)

    while True:
        raw = _ask(f"Choose a run [1-{len(runs)}]", default="1")
        try:
            idx = int(raw)
            if 1 <= idx <= len(runs):
                chosen = runs[idx - 1]["run_id"]
                print(f"  -> {chosen}\n")
                return chosen
        except ValueError:
            pass
        print(f"  enter a number 1-{len(runs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root",
                    default=r"\output",
                    help="pipeline output/ folder")
    ap.add_argument("--geodict-exe",
                    default=r"C:\Program Files\Math2Market GmbH\GeoDict 2026\geodict2026.exe")
    ap.add_argument("--macro",
                    default=r"\scripts\geodict_lir_job.py",
                    help="path to geodict_lir_job.py")
    ap.add_argument("--result-root",
                    default=r"\results_geodict")
    ap.add_argument("--db",
                    default=r"\results.duckdb",
                    help="results.duckdb path (writeback target)")
    ap.add_argument("--run-id", default=None,
                    help="run_id these domains belong to (default: latest run in the DB)")
    ap.add_argument("--connectivity", default=None, help="only this conn folder, e.g. 18N")
    ap.add_argument("--scans", default=None,
                    help="comma-separated scan indices to INCLUDE (e.g. '0' for dry-only, "
                         "'1,2,3' for fluid scans). Default: all scans.")
    ap.add_argument("--skip-scans", default=None,
                    help="comma-separated scan indices to EXCLUDE (e.g. '0' to skip the "
                         "dry scan). Ignored if --scans is given.")
    ap.add_argument("--dry-run", action="store_true", help="print commands, no run, no DB")
    ap.add_argument("--force", action="store_true",
                    help="re-run domains even if a valid result_summary.json already exists")
    ap.add_argument("--interactive", action="store_true",
                    help="read the DB, show available sections, and pick interactively "
                         "(overrides --scans/--skip-scans; absolute runs once per track at X)")
    ap.add_argument("--thin-cutoff", type=int, default=200,
                    help="z-extent (voxels) below which a section is flagged THIN "
                         "(default 200)")
    args = ap.parse_args()

    def _parse_scan_set(s):
        if not s:
            return None
        return {int(x) for x in s.replace(" ", "").split(",") if x != ""}
    include_scans = _parse_scan_set(args.scans)          # None = all
    exclude_scans = _parse_scan_set(args.skip_scans) or set()

    output_root = Path(args.output_root)
    geodict_exe = Path(args.geodict_exe)
    macro = Path(args.macro)
    result_root = Path(args.result_root)

    if not output_root.exists():
        sys.exit(f"output root not found: {output_root}")
    if not args.dry_run and not geodict_exe.exists():
        sys.exit(f"GeoDict exe not found: {geodict_exe}")

    groups = discover(output_root, args.connectivity)
    if not groups:
        sys.exit("No matching domain_*.raw files found. Check naming/convention.")
    print(f"found {len(groups)} (track, scan, connectivity) groups\n")

    con = None
    if not args.dry_run:
        import duckdb
        con = duckdb.connect(args.db)

    # Interactive selection (overrides --scans/--skip-scans). Needs a live DB.
    selection = None
    if args.interactive:
        if con is None:
            sys.exit("--interactive needs the database (cannot be combined with --dry-run).")

    # Resolve run_id: explicit --run-id wins; else in interactive mode show the run
    # menu; else fall back to the most recent run.
    run_id = args.run_id
    if run_id is None and con is not None:
        if args.interactive:
            run_id = pick_run(con)
        else:
            row = con.execute(
                "SELECT run_id FROM runs ORDER BY rowid DESC LIMIT 1").fetchone()
            if row:
                run_id = row[0]
                print(f"using latest run_id from DB: {run_id}\n")
    if run_id is None and not args.dry_run:
        sys.exit("no --run-id given and could not read a run from the DB.")

    if args.interactive:
        selection = interactive_select(con, run_id, args.connectivity, groups,
                                       thin_cutoff=args.thin_cutoff)
    qual_scans = qualifying_scan_set(con, run_id) if (selection and selection["qualifying_only"]) else None

    try:
        # Sort numerically by (track, scan) so scan 2 comes before scan 10
        # (plain sorted() would order the string keys lexicographically).
        def _sort_key(item):
            (track, scan, conn) = item[0]
            t = int(track) if track.isdigit() else track
            s = int(scan) if scan.isdigit() else scan
            return (t, s, conn)
        for (track, scan, conn), domains in sorted(groups.items(), key=_sort_key):
            present = [d for d in ("absolute", "gas", "water") if d in domains]
            if not present:
                continue
            try:
                scan_index = int(scan)
            except ValueError:
                print(f"[skip] scan '{scan}' is not an integer index; cannot key DB row")
                continue
            track_id = int(track)

            if selection is not None:
                # ── Interactive filtering (overrides flag-based scan filters) ──
                if selection["tracks"] is not None and track_id not in selection["tracks"]:
                    continue
                if selection["exclude_thin"] and track_id in selection["thin_tracks"]:
                    print(f"[skip] track {track} scan {scan} {conn}: THIN (excluded)")
                    continue
                if selection["qualifying_only"] and qual_scans is not None \
                        and scan_index not in qual_scans:
                    continue
                if selection["scans"] is not None and scan_index not in selection["scans"]:
                    continue
                present = [d for d in present if d in selection["dtypes"]]
                # Absolute runs ONCE per track, at the X timestep only. Drop it on
                # any non-X scan so it is not re-solved every timestep. compare_results
                # already broadcasts the single per-track absolute across all scans.
                if "absolute" in present and selection["x_scan"] is not None \
                        and scan_index != selection["x_scan"]:
                    present = [d for d in present if d != "absolute"]
                if not present:
                    continue
            else:
                # Scan filtering: --scans includes only the listed scans; otherwise
                # --skip-scans excludes the listed ones.
                if include_scans is not None:
                    if scan_index not in include_scans:
                        print(f"[skip] track {track} scan {scan} {conn}: not in --scans")
                        continue
                elif scan_index in exclude_scans:
                    print(f"[skip] track {track} scan {scan} {conn}: in --skip-scans")
                    continue

            print(f"=== track {track} | scan {scan} | {conn} ===")
            sw = lookup_sw(con, run_id, scan_index, track_id, conn) if con else None

            for dtype in present:
                dom = domains[dtype]
                rdir = result_root / conn / f"scan{scan}_track{track}" / dtype

                # Resume: skip a domain that already has a valid result (a
                # result_summary.json with a non-null permeability). Controlled by
                # --force (flag mode) or the interactive "skip done" choice.
                scan_dir = dom["path"].parent.parent
                existing_summary = (scan_dir / "geodict" / dom["path"].stem
                                    / "result_summary.json")
                skip_done = (selection["skip_done"] if selection is not None
                             else not args.force)
                if skip_done and existing_summary.exists():
                    prev = read_summary(existing_summary)
                    if prev.get("permeability_z_m2") is not None:
                        k_z = prev.get("permeability_z_m2")
                        print(f"    {dtype:>8}: SKIP (already done) K_z = {k_z} m^2")
                        if con is not None:
                            gdr_path = str(rdir / "StokesResult.gdr")
                            write_sim_result(con, run_id, scan_index, track_id, conn,
                                             dtype, prev, sw, str(dom["path"]), gdr_path)
                        continue

                summary_path = run_one(geodict_exe, macro, dom, rdir, args.dry_run)
                summary = read_summary(summary_path) if not args.dry_run else {}
                k_z = summary.get("permeability_z_m2")
                print(f"    {dtype:>8}: K_z = {k_z} m^2")

                if con is not None:
                    gdr_path = str(rdir / "StokesResult.gdr")
                    write_sim_result(con, run_id, scan_index, track_id, conn,
                                     dtype, summary, sw, str(dom["path"]), gdr_path)
            if con is not None:
                print(f"    -> wrote simulation_results (Sw={sw})\n")
            else:
                print()
    finally:
        if con is not None:
            con.close()
    print("done.")


if __name__ == "__main__":
    main()
