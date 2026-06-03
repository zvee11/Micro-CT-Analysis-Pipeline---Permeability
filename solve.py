"""
solve.py — MPLBM Permeability Solver Launcher
==============================================
Run from Anaconda Prompt or CMD, from your THESIS directory:

    python solve.py

The script will guide you through everything interactively.
No flags, no WSL commands, no venv activation needed.

Requirements
------------
- WSL installed with Ubuntu
- MPLBM-UT compiled in WSL (~/MPLBM-UT)
- mplbm_env venv in WSL (~/mplbm_env)
- results.duckdb in the current directory
- duckdb installed: pip install duckdb pyyaml
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path, PurePosixPath

import duckdb

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit if your setup differs
# ═══════════════════════════════════════════════════════════════════════════

WSL_MPLBM_DIR   = "~/MPLBM-UT"          # MPLBM repo in WSL
WSL_VENV        = "~/mplbm_env"          # Python venv in WSL
WSL_WORK_DIR    = "~/mplbm_workdir"      # Temporary simulation files in WSL
CONNECTIVITY    = "26N"
MPLBM_MAX_ITER  = 200_000
MPLBM_CONVERGE  = 0.001
MPLBM_PRESSURE  = 0.0001
SAMPLE_Z_SLICES = 50                     # default Z crop size (overridden interactively)

# Empirical MLUPS from your test run (4 cores, 277 MLUPS total)
MLUPS_PER_CORE  = 69.44

# ═══════════════════════════════════════════════════════════════════════════


# ── Helpers ─────────────────────────────────────────────────────────────────

def separator(title: str = "") -> None:
    w = 65
    if title:
        pad = (w - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*(w-pad-len(title)-2)}")
    else:
        print(f"\n{'─'*w}")


def ask(prompt: str, choices: list[str] | None = None,
        default: str | None = None) -> str:
    """Prompt user for input with optional choices and default."""
    if choices:
        opts = "/".join(
            f"[{c}]" if c == default else c for c in choices
        )
        full = f"{prompt} ({opts}): "
    elif default:
        full = f"{prompt} [{default}]: "
    else:
        full = f"{prompt}: "

    while True:
        raw = input(full).strip()
        if not raw and default is not None:
            return default
        if choices and raw.lower() not in [c.lower() for c in choices]:
            print(f"  Please enter one of: {', '.join(choices)}")
            continue
        return raw


def ask_int(prompt: str, default: int,
            lo: int = 1, hi: int = 999) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
            print(f"  Enter a number between {lo} and {hi}")
        except ValueError:
            print("  Please enter a whole number")


def win_to_wsl(win_path: str) -> str:
    """Convert Windows path to WSL /mnt/... path."""
    p = win_path.replace("\\", "/")
    # C:/Users/... -> /mnt/c/Users/...
    m = re.match(r"^([A-Za-z]):/(.+)$", p)
    if m:
        return f"/mnt/{m.group(1).lower()}/{m.group(2)}"
    return p


def wsl_run(cmd: str, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a command in WSL."""
    return subprocess.run(
        ["wsl", "bash", "-c", cmd],
        capture_output=capture,
        text=True,
    )


def check_wsl() -> bool:
    r = wsl_run("echo ok")
    return r.returncode == 0 and "ok" in r.stdout


def get_wsl_username() -> str:
    r = wsl_run("whoami")
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def get_available_ram_wsl_gb() -> float:
    r = wsl_run("awk '/MemAvailable/{print $2}' /proc/meminfo")
    try:
        return int(r.stdout.strip().split()[0]) / 1e6
    except Exception:
        r2 = wsl_run("awk '/MemTotal/{print $2}' /proc/meminfo")
        try:
            return int(r2.stdout.strip().split()[0]) / 1e6 * 0.85
        except Exception:
            return 8.0


def estimate_ram_gb(nz: int, ny: int, nx: int, n_procs: int,
                    fluid_voxels: int = 0) -> float:
    # Validated against real runs: Palabos only allocates LBM arrays
    # for fluid voxels (sparse). Geometry array covers full domain.
    total_vox  = nz * ny * nx
    fv         = fluid_voxels if fluid_voxels > 0 else int(total_vox * 0.08)
    geom_bytes = total_vox * 2
    lbm_bytes  = fv * 19 * 2 * 8
    halo_bytes = 0 if n_procs == 1 else n_procs * 2 * ny * nx * 19 * 8
    return (geom_bytes + lbm_bytes + halo_bytes) / 1e9


def estimate_minutes(fluid_vox: int, n_procs: int) -> float:
    mlups = MLUPS_PER_CORE * n_procs
    return (fluid_vox * MPLBM_MAX_ITER) / (mlups * 1e6) / 60


def lbm_to_darcy(k_lbm: float, voxel_m: float) -> float:
    return k_lbm * (voxel_m ** 2) / 9.869233e-13


def get_voxel_m(con: duckdb.DuckDBPyConnection) -> float:
    row = con.execute("""
        SELECT gas_voxels, gas_volume_mm3
        FROM fixed_boxes
        WHERE gas_voxels > 0 AND gas_volume_mm3 IS NOT NULL
        LIMIT 1
    """).fetchone()
    if row and row[0] and row[1]:
        return ((row[1] * 1e-9) / row[0]) ** (1 / 3)
    return 6.651e-6  # fallback from your data


def resolve_win_path(relative_path: str, thesis_dir: Path) -> Path:
    """Resolve a DB-stored relative Windows path to an absolute Path."""
    clean = relative_path.replace("\\", os.sep).replace("/", os.sep)
    return thesis_dir / clean


# ── DB queries ───────────────────────────────────────────────────────────────

def load_jobs(con: duckdb.DuckDBPyConnection,
              tracks: list[int],
              sim_types: list[str],
              thesis_dir: Path,
              sample_mode: bool,
              crop_z_start: int = 0,
              crop_z_slices: int = 50) -> list[dict]:
    """
    Build the flat list of simulation jobs from fixed_boxes.

    Rules:
    - Skip scan_index 0 (dry scan, no gas/brine fluid)
    - absolute: run once per track using scan_index 8 (timestep X)
    - gas/water: run for scan_indices 1–7 (qualifying timesteps)
    - sample_mode: use a 50-slice Z crop of each domain
    """
    track_clause = (
        f"AND fb.track_id IN ({','.join(str(t) for t in tracks)})"
        if tracks else ""
    )

    rows = con.execute(f"""
        SELECT
            fb.run_id, fb.scan_index, fb.track_id, fb.connectivity,
            fb.extent_z, fb.extent_y, fb.extent_x,
            fb.gas_voxels, fb.brine_voxels, fb.sw_local,
            fb.domain_absolute, fb.domain_gas, fb.domain_water
        FROM fixed_boxes fb
        WHERE fb.connectivity = '{CONNECTIVITY}'
          AND fb.scan_index > 0
          {track_clause}
        ORDER BY fb.track_id, fb.scan_index
    """).fetchall()

    col = ["run_id","scan_index","track_id","connectivity",
           "extent_z","extent_y","extent_x",
           "gas_voxels","brine_voxels","sw_local",
           "domain_absolute","domain_gas","domain_water"]
    all_rows = [dict(zip(col, r)) for r in rows]

    jobs: list[dict] = []
    seen_abs: set[tuple] = set()

    for row in all_rows:
        is_X = (row["scan_index"] == 8)

        for st in sim_types:
            if st == "absolute":
                # Only run absolute once per track, at scan_index 8 (timestep X)
                if not is_X:
                    continue
                abs_key = (row["run_id"], row["track_id"])
                if abs_key in seen_abs:
                    continue
                seen_abs.add(abs_key)
                domain_rel = row["domain_absolute"]
                fluid_vox  = (row["gas_voxels"] or 0) + (row["brine_voxels"] or 0)

            elif st == "gas":
                # Gas: qualifying timesteps (1–7), not scan 8
                if is_X:
                    continue
                domain_rel = row["domain_gas"]
                fluid_vox  = row["gas_voxels"] or 0

            else:  # water
                if is_X:
                    continue
                domain_rel = row["domain_water"]
                fluid_vox  = row["brine_voxels"] or 0

            if not domain_rel:
                continue

            win_path = resolve_win_path(domain_rel, thesis_dir)
            if not win_path.exists():
                print(f"  ⚠  Missing: {win_path.name} — will skip")
                continue

            jobs.append({
                **row,
                "sim_type":      st,
                "win_path":      win_path,
                "fluid_voxels":  fluid_vox,
                "sample_mode":   sample_mode,
                "crop_z_start":  crop_z_start if sample_mode else 0,
                "sample_z_slices": crop_z_slices if sample_mode else 50,
                "scan_for_db":   row["scan_index"] if st != "absolute" else 8,
            })

    return jobs


# ── Simulation table display ─────────────────────────────────────────────────

def print_job_table(jobs: list[dict], n_procs: int, avail_gb: float) -> bool:
    """Print the job plan. Returns False if any job won't fit in RAM."""
    separator("SIMULATION PLAN")
    print(f"\n  {'#':>3}  {'Track':>5}  {'Scan':>4}  {'Type':>8}  "
          f"{'RAM (GB)':>8}  {'Fit':>4}  {'Est. time':>10}")
    print(f"  {'─'*3}  {'─'*5}  {'─'*4}  {'─'*8}  "
          f"{'─'*8}  {'─'*4}  {'─'*10}")

    any_oom = False
    total_min = 0.0

    for i, j in enumerate(jobs):
        do_crop = j["sample_mode"]
        sim_nz  = (SAMPLE_Z_SLICES + 4) if do_crop else (j["extent_z"] + 4)
        ny  = j["extent_y"]
        nx  = j["extent_x"]
        fv  = j["fluid_voxels"]
        if do_crop:
            fv = int(fv * SAMPLE_Z_SLICES / max(j["extent_z"], 1))
        ram  = estimate_ram_gb(sim_nz, ny, nx, n_procs, fluid_voxels=fv)
        mins = estimate_minutes(fv, n_procs)
        total_min += mins
        fits = "✓" if ram < avail_gb * 0.88 else "✗OOM"
        if "OOM" in fits:
            any_oom = True
        mode = " [abs crop]" if do_crop else ""
        print(f"  {i+1:>3}  {j['track_id']:>5}  {j['scan_index']:>4}  "
              f"{j['sim_type']:>8}  {ram:>8.2f}  {fits:>4}  "
              f"{mins:>8.0f} min{mode}")

    print(f"\n  Total jobs: {len(jobs)}")
    print(f"  Estimated total: {total_min:.0f} min ({total_min/60:.1f} hours)")
    print(f"  Available RAM: {avail_gb:.1f} GB  |  MPI processes: {n_procs}")

    return any_oom


# ── WSL simulation runner ────────────────────────────────────────────────────

def build_wsl_sim_script(
    win_domain:   Path,
    nz: int, ny: int, nx: int,
    n_procs:      int,
    wsl_work:     str,
    wsl_mplbm:    str,
    wsl_venv:     str,
    sim_type:     str,
    track_id:     int,
    scan_index:   int,
    sample_mode:  bool,
    crop_z_start: int = 0,
    sample_z_slices: int = 50,
) -> str:
    """
    Build the bash script that runs inside WSL for one simulation.
    Returns the bash script as a string.
    """
    wsl_domain = win_to_wsl(str(win_domain))
    geom_name  = f"t{track_id:02d}_s{scan_index}_{sim_type}"
    work       = f"{wsl_work}/{geom_name}"
    # All domain types are cropped in sample mode.
    # We verified all 84-slice windows percolate for gas domains.
    # Optimal window Z=110-194 maximises gas voxel count.
    do_crop = sample_mode
    sim_nz  = sample_z_slices if do_crop else nz
    SAMPLE_Z_SLICES = sample_z_slices  # use parameter not global

    # Python snippet that creates the domain file and YAML inside WSL
    setup_py = f"""
import numpy as np, yaml, pathlib, shutil

domain_src  = pathlib.Path('{wsl_domain}')
work        = pathlib.Path('{work}')
(work / 'input').mkdir(parents=True, exist_ok=True)
(work / 'tmp').mkdir(parents=True, exist_ok=True)

raw  = np.fromfile(str(domain_src), dtype=np.uint8)
vol  = raw.reshape({nz}, {ny}, {nx})

{'# Crop Z slices: ' + str(crop_z_start) + ' to ' + str(crop_z_start + SAMPLE_Z_SLICES) if do_crop else '# Full domain'}
{'vol = vol[' + str(crop_z_start) + ':' + str(crop_z_start) + '+' + str(SAMPLE_Z_SLICES) + ', :, :]' if do_crop else ''}

nz_sim = vol.shape[0]
dest   = work / 'input' / '{geom_name}.raw'
vol.tofile(str(dest))

# Write YAML (swap XZ: our Z is flow direction, Palabos needs X)
cfg = {{
    'simulation type': '1-phase',
    'input output': {{
        'simulation directory': str(work),
        'input folder': 'input/',
        'output folder': 'tmp/',
    }},
    'geometry': {{
        'file name': '{geom_name}.raw',
        'data type': 'uint8',
        'geometry size': {{'Nx': {nx}, 'Ny': {ny}, 'Nz': nz_sim}},
    }},
    'domain': {{
        'geom name': '{geom_name}',
        'domain size': {{'nx': nz_sim, 'ny': {ny}, 'nz': {nx}}},
        'periodic boundary': {{'x': False, 'y': True, 'z': True}},
        'inlet and outlet layers': 2,
        'add mesh': False,
        'swap xz': True,
        'double geom resolution': False,
    }},
    'simulation': {{
        'num procs': {n_procs},
        'num geoms': 1,
        'pressure': {MPLBM_PRESSURE},
        'max iterations': {MPLBM_MAX_ITER},
        'convergence': {MPLBM_CONVERGE},
        'save vtks': False,
        'print geom': False,
        'print stl': False,
    }},
}}
with open(work / 'input.yml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

# Clean stale files
for pat in ['*.dat','*.xml','run_*.sh']:
    for p in (work/'input').glob(pat): p.unlink()
for p in (work/'tmp').glob('*'): p.unlink()

import gc, time
gc.collect()
time.sleep(5)
print('SETUP_OK')
"""

    script = (
        f"set -e\n"
        f"source {wsl_venv}/bin/activate\n"
        f"python3 - << 'PYEOF'\n"
        f"{setup_py}\n"
        f"PYEOF\n"
        f"cd {work}\n"
        f"sleep 15\n"  # allow Python GC to release edist memory before Palabos
        f"python3 {wsl_mplbm}/examples/single_phase_permeability/1_phase_sim.py 2>&1\n"
    )
    return script


def parse_permeability(stdout: str) -> float | None:
    matches = re.findall(
        r"Absolute Permeability\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        stdout,
    )
    if matches:
        val = float(matches[-1])
        return val if val > 0 else None
    return None


def parse_converged_iter(stdout: str) -> int | None:
    m = re.findall(r"End of simulation at iteration (\d+)", stdout)
    return int(m[-1]) if m else None


def run_job(job: dict, n_procs: int, wsl_work: str,
            wsl_mplbm: str, wsl_venv: str,
            logfile, voxel_m: float,
            con: duckdb.DuckDBPyConnection) -> float | None:
    """
    Run one simulation job via a persistent WSL bash session.
    Writing the script to a file and executing it avoids repeated
    WSL process launches which cause memory fragmentation.
    Returns K in Darcy or None on failure.
    """
    nz = job["extent_z"]
    ny = job["extent_y"]
    nx = job["extent_x"]

    script = build_wsl_sim_script(
        win_domain      = job["win_path"],
        nz=nz, ny=ny, nx=nx,
        n_procs         = n_procs,
        wsl_work        = wsl_work,
        wsl_mplbm       = wsl_mplbm,
        wsl_venv        = wsl_venv,
        sim_type        = job["sim_type"],
        track_id        = job["track_id"],
        scan_index      = job["scan_index"],
        sample_mode     = job["sample_mode"],
        crop_z_start    = job.get("crop_z_start", 0),
        sample_z_slices = job.get("sample_z_slices", SAMPLE_Z_SLICES),
    )

    logfile.write(f"\n{'='*60}\n")
    logfile.write(f"Track {job['track_id']:02d}  Scan {job['scan_index']}  "
                  f"{job['sim_type']}\n")
    logfile.flush()

    # Write the bash script to WSL filesystem and execute it there.
    # This uses a single persistent wsl call rather than a new wsl
    # process per job, avoiding repeated WSL session init overhead
    # and memory fragmentation from rapid process creation/teardown.
    geom_name  = f"t{job['track_id']:02d}_s{job['scan_index']}_{job['sim_type']}"
    script_path_wsl = f"{wsl_work}/{geom_name}.sh"

    # Write script file into WSL via wsl bash
    write_cmd = f"mkdir -p {wsl_work} && cat > {script_path_wsl} << 'ENDSCRIPT'\n{script}\nENDSCRIPT"
    write_result = subprocess.run(
        ["wsl", "bash", "-c", write_cmd],
        capture_output=True, text=True, timeout=30,
    )
    if write_result.returncode != 0:
        print(f"  ✗  Failed to write script to WSL: {write_result.stderr[:200]}")
        return None

    # Execute the script in WSL — single persistent session
    proc = subprocess.Popen(
        ["wsl", "bash", script_path_wsl],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    combined = []
    try:
        for line in proc.stdout:
            line_s = line.rstrip()
            combined.append(line)
            logfile.write(line)
            logfile.flush()
            if any(kw in line_s for kw in [
                "Iteration", "Permeability", "Absolute Permeability",
                "Relative Permeability", "convergence", "SETUP_OK",
                "Creating", "Running", "End of simulation",
            ]):
                print(f"    {line_s}")
        proc.wait(timeout=86400)
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"  ✗  Timed out after 24h")
        return None

    combined_str = "".join(combined)

    k_lbm = parse_permeability(combined_str)
    if k_lbm is None:
        print(f"  ✗  Could not parse permeability — check solver.log")
        return None

    k_darcy = lbm_to_darcy(k_lbm, voxel_m)
    conv    = parse_converged_iter(combined_str)
    sw      = job.get("sw_local") or float("nan")

    print(f"  ✓  K = {k_darcy:.4f} Darcy ({k_darcy*1000:.1f} mDarcy)"
          f"  converged @ iter {conv}")

    # Write to DB
    con.execute("""
        INSERT OR REPLACE INTO simulation_results
            (run_id, scan_index, track_id, connectivity,
             sim_type, simulator,
             k_x, k_y, k_z, k_eff, kr, Sw,
             domain_path, raw_output_path, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        job["run_id"], job["scan_for_db"], job["track_id"],
        job["connectivity"], job["sim_type"], "mplbm_MRT",
        k_darcy, k_darcy, k_darcy, k_darcy,
        float("nan"),  # kr filled in after K_abs known
        sw,
        str(job["win_path"]), "",
        f"K_LBM={k_lbm:.6f} voxel_m={voxel_m:.4e} "
        f"conv_iter={conv} sample={job['sample_mode']}",
    ))
    con.commit()
    return k_darcy


def fill_kr(con: duckdb.DuckDBPyConnection, run_id: str) -> None:
    """Back-fill kr = K_eff / K_abs for all rows where K_abs is now known."""
    abs_rows = con.execute("""
        SELECT track_id, k_eff FROM simulation_results
        WHERE run_id=? AND sim_type='absolute' AND simulator='mplbm_MRT'
    """, [run_id]).fetchall()

    for track_id, k_abs in abs_rows:
        if not k_abs or k_abs <= 0:
            continue
        con.execute("""
            UPDATE simulation_results
            SET kr = k_eff / ?
            WHERE run_id=? AND track_id=? AND sim_type != 'absolute'
            AND simulator='mplbm_MRT'
        """, [k_abs, run_id, track_id])
    con.commit()


# ── Main interactive loop ────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═"*65)
    print("  MPLBM Permeability Solver")
    print("═"*65)

    # ── Check WSL ───────────────────────────────────────────────────────────
    print("\n  Checking WSL...", end=" ", flush=True)
    if not check_wsl():
        print("✗\n\n  WSL is not available. Please install Ubuntu from the "
              "Microsoft Store and try again.")
        input("\nPress Enter to exit.")
        sys.exit(1)
    wsl_user = get_wsl_username()
    print(f"✓  (user: {wsl_user})")

    # ── Resolve WSL paths with actual username ───────────────────────────────
    wsl_mplbm  = WSL_MPLBM_DIR.replace("~", f"/home/{wsl_user}")
    wsl_venv   = WSL_VENV.replace("~", f"/home/{wsl_user}")
    wsl_work   = WSL_WORK_DIR.replace("~", f"/home/{wsl_user}")

    # Check MPLBM is compiled
    r = wsl_run(f"test -f {wsl_mplbm}/src/1-phase_LBM/permeability_MRT && echo ok")
    if "ok" not in r.stdout:
        print(f"\n  ✗  MPLBM executable not found at:\n"
              f"     {wsl_mplbm}/src/1-phase_LBM/permeability_MRT\n"
              f"     Please compile MPLBM first (see setup instructions).")
        input("\nPress Enter to exit.")
        sys.exit(1)
    print(f"  MPLBM:  {wsl_mplbm}  ✓")

    # ── Find DB ─────────────────────────────────────────────────────────────
    thesis_dir = Path.cwd()
    db_path    = thesis_dir / "results.duckdb"
    if not db_path.exists():
        print(f"\n  ✗  results.duckdb not found in:\n  {thesis_dir}\n"
              f"  Run this script from your THESIS directory.")
        input("\nPress Enter to exit.")
        sys.exit(1)

    con = duckdb.connect(str(db_path))
    run_id = con.execute("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
                         ).fetchone()[0]
    voxel_m = get_voxel_m(con)
    avail_gb = get_available_ram_wsl_gb()

    print(f"  DB:     {db_path.name}  ✓")
    print(f"  Run ID: {run_id}")
    print(f"  Voxel:  {voxel_m*1e6:.3f} µm")
    print(f"  RAM:    {avail_gb:.1f} GB available in WSL")

    # ── Show what's already done ─────────────────────────────────────────────
    done = con.execute("""
        SELECT sim_type, COUNT(*) FROM simulation_results
        WHERE run_id=? AND simulator='mplbm_MRT'
        GROUP BY sim_type
    """, [run_id]).fetchall()
    if done:
        separator("ALREADY COMPUTED")
        for st, n in done:
            print(f"  {st:<10} {n} results in DB")

    # ── Choose tracks ────────────────────────────────────────────────────────
    separator("TRACK SELECTION")
    track_info = con.execute("""
        SELECT t.track_id,
               COUNT(DISTINCT fb.scan_index) as n_scans,
               t.status,
               fb.extent_z, fb.extent_y, fb.extent_x
        FROM tracks t
        JOIN fixed_boxes fb ON fb.track_id=t.track_id AND fb.run_id=t.run_id
        WHERE t.run_id=? AND fb.connectivity=?
        GROUP BY t.track_id, t.status, fb.extent_z, fb.extent_y, fb.extent_x
        ORDER BY t.track_id
    """, [run_id, CONNECTIVITY]).fetchall()

    print(f"\n  {'Track':>5}  {'Status':>8}  {'Scans':>5}  {'Domain size':>15}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*5}  {'─'*15}")
    for row in track_info:
        tid, n_scans, status, ez, ey, ex = row
        print(f"  {tid:>5}  {status or '':>8}  {n_scans:>5}  "
              f"{ez}×{ey}×{ex}")

    print("\n  Enter track numbers separated by commas (e.g. 1,3,5)")
    print("  or press Enter to run all tracks")
    raw = input("  Tracks: ").strip()
    if raw:
        tracks = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
    else:
        tracks = []  # empty = all

    # ── Choose domain types ──────────────────────────────────────────────────
    separator("DOMAIN TYPE")
    print("\n  What to simulate:")
    print("  all      — gas + water (all timesteps) + absolute (timestep X)")
    print("  gas      — gas effective permeability only")
    print("  water    — water effective permeability only")
    print("  absolute — absolute permeability only (timestep X)")
    choice = ask("\n  Type", ["all","gas","water","absolute"], default="all")
    sim_types = (["gas","water","absolute"] if choice == "all"
                 else [choice])

    # ── Sample mode? ─────────────────────────────────────────────────────────
    separator("RUN MODE")
    print(f"\n  Full run:   uses complete domain (750×750 lateral extent)")
    print(f"  Sample run: crops to {SAMPLE_Z_SLICES} central Z slices — "
          f"fits in RAM, good for testing")
    mode = ask("\n  Mode", ["full","sample"], default="sample")
    sample_mode = (mode == "sample")

    # Defaults — overridden interactively if sample_mode
    crop_z_slices = SAMPLE_Z_SLICES
    crop_z_start  = 0

    if sample_mode:
        print(f"\n  Enter number of Z slices to use (default {SAMPLE_Z_SLICES}).")
        print(f"  Rule of thumb: available_RAM_GB / 0.085 = max slices")
        print(f"  e.g. 9GB / 0.085 = 105 max, use 84 for safety")
        crop_z_slices = ask_int("  Z slices", default=SAMPLE_Z_SLICES, lo=10, hi=500)
        print(f"\n  Enter Z start index (0 = inlet face).")
        print(f"  Use 0 for inlet, or a value that maximises gas connectivity.")
        print(f"  For Track 5: Z=110 gives most gas voxels (verified percolating)")
        crop_z_start = ask_int("  Z start index", default=0, lo=0, hi=500)
        print(f"\n  ℹ  Sample mode: cropping to Z={crop_z_start} to "
              f"Z={crop_z_start+crop_z_slices} ({crop_z_slices} slices).")
        print(f"     Results are valid locally but not equivalent to full domain.")


    # ── Build job list ───────────────────────────────────────────────────────
    jobs = load_jobs(con, tracks, sim_types, thesis_dir, sample_mode, crop_z_start, crop_z_slices)
    if not jobs:
        print("\n  No jobs found matching your selection.")
        input("\nPress Enter to exit.")
        con.close()
        return

    # ── Choose cores ─────────────────────────────────────────────────────────
    separator("CORES")
    n_procs = ask_int(
        "\n  Number of MPI processes (cores)", default=4, lo=1, hi=12
    )

    # ── RAM check and auto-suggest ───────────────────────────────────────────
    separator("RAM CHECK")
    any_oom = print_job_table(jobs, n_procs, avail_gb)

    if any_oom:
        print(f"\n  ✗  Some jobs may exceed available RAM ({avail_gb:.1f} GB) "
              f"with {n_procs} cores.")

        # Find minimum safe core count
        safe_cores = None
        for try_cores in range(n_procs - 1, 0, -1):
            still_oom = any(
                estimate_ram_gb(
                    (SAMPLE_Z_SLICES + 4) if j["sample_mode"] else (j["extent_z"] + 4),
                    j["extent_y"], j["extent_x"], try_cores
                ) >= avail_gb * 0.88
                for j in jobs
            )
            if not still_oom:
                safe_cores = try_cores
                break

        if safe_cores:
            print(f"\n  Suggestion: reduce to {safe_cores} core(s) to fit in RAM.")
            ans = ask(f"  Use {safe_cores} cores instead?", ["yes","no"],
                      default="yes")
            if ans == "yes":
                n_procs = safe_cores
                any_oom = print_job_table(jobs, n_procs, avail_gb)
            else:
                ans2 = ask("  Proceed anyway (may crash)?", ["yes","no"],
                           default="no")
                if ans2 == "no":
                    print("  Aborted.")
                    con.close()
                    return
        else:
            if not sample_mode:
                print("\n  Cannot fit even with 1 core in full mode.")
                print("  Try sample mode instead.")
                con.close()
                return
            else:
                ans = ask("  Proceed anyway with 1 core?", ["yes","no"],
                          default="no")
                if ans == "no":
                    con.close()
                    return
                n_procs = 1

    # ── Final confirmation ───────────────────────────────────────────────────
    separator("CONFIRM")
    print(f"\n  Ready to run {len(jobs)} simulations")
    print(f"  Cores: {n_procs}  |  Mode: {mode}  |  Types: {', '.join(sim_types)}")
    ans = ask("\n  Start?", ["yes","no"], default="yes")
    if ans != "yes":
        print("  Aborted.")
        con.close()
        return

    # ── Run ──────────────────────────────────────────────────────────────────
    separator("RUNNING")
    log_path = thesis_dir / "solver.log"
    print(f"\n  Log file: {log_path}")
    print(f"  Progress will print here as each job completes.\n")

    k_abs_store: dict[tuple, float] = {}
    success = 0
    failed  = 0

    with open(log_path, "a", encoding="utf-8") as logfile:
        for i, job in enumerate(jobs):
            label = (f"[{i+1}/{len(jobs)}] "
                     f"Track {job['track_id']:02d}  "
                     f"Scan {job['scan_index']:2d}  "
                     f"{job['sim_type']:8}")
            print(f"  {label}", end="  ", flush=True)

            k = run_job(job, n_procs, wsl_work, wsl_mplbm, wsl_venv, logfile, voxel_m, con)

            if k is None:
                failed += 1
            else:
                success += 1
                if job["sim_type"] == "absolute":
                    k_abs_store[(job["run_id"], job["track_id"])] = k

    # ── Back-fill kr ─────────────────────────────────────────────────────────
    if k_abs_store:
        fill_kr(con, run_id)
        print("\n  kr values computed and written to DB.")

    # ── Export CSV ───────────────────────────────────────────────────────────
    csv_path = thesis_dir / "simulation_results.csv"
    try:
        con.execute(
            f"COPY (SELECT * FROM simulation_results WHERE simulator='mplbm_MRT') "
            f"TO '{str(csv_path).replace(chr(92), '/')}' (HEADER, DELIMITER ',')"
        )
        print(f"  Results exported to simulation_results.csv")
    except Exception as e:
        print(f"  CSV export failed: {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    separator("DONE")
    print(f"\n  {success} succeeded  |  {failed} failed")
    if success:
        rows = con.execute("""
            SELECT sim_type, COUNT(*) as n,
                   ROUND(AVG(k_eff),4) as k_mean,
                   ROUND(AVG(kr),4)    as kr_mean
            FROM simulation_results
            WHERE run_id=? AND simulator='mplbm_MRT'
            GROUP BY sim_type ORDER BY sim_type
        """, [run_id]).fetchall()
        print(f"\n  {'Type':<10} {'N':>4}  {'K mean (D)':>12}  {'kr mean':>10}")
        print(f"  {'─'*10} {'─'*4}  {'─'*12}  {'─'*10}")
        for r in rows:
            kr_s = f"{r[3]:.4f}" if r[3] is not None and r[3]==r[3] else "—"
            print(f"  {r[0]:<10} {r[1]:>4}  {r[2]:>12.4f}  {kr_s:>10}")

    con.close()
    print()
    input("  Press Enter to exit.")


if __name__ == "__main__":
    main()