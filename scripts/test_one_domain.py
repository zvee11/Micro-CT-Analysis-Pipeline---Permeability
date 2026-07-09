"""
test_one_domain.py  —  run GeoDict on ONE domain and report the result.

Use this for the very first GeoDict test on the workstation, BEFORE the full
batch. It runs the macro on a single .raw, then prints whether GeoDict produced
a valid permeability — so you can confirm the import command and the result
extraction work before committing to a long batch.

It does NOT touch the database. It just runs one job and shows the outcome.

Usage:
    python test_one_domain.py ^
        --geodict-exe "C:\\Program Files\\Math2Market\\GeoDict2026\\GeoDict2026.exe" ^
        --macro       "C:\\...\\geodict_lir_job.py" ^
        --raw         "C:\\...\\output\\9_5\\26N\\domain_gas_03-7_4.99684um_8bu_750x750x523.raw"

Dimensions and voxel size are parsed from the filename
(..._{voxel}um_{bits}_{NX}x{NY}x{NZ}.raw); override with --nx/--ny/--nz/--voxel-m.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_NAME_RE = re.compile(
    r"(?P<voxel>[0-9.]+)um_\d+b[us]_(?P<nx>\d+)x(?P<ny>\d+)x(?P<nz>\d+)\.raw$",
    re.IGNORECASE,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geodict-exe", required=True)
    ap.add_argument("--macro", required=True)
    ap.add_argument("--raw", required=True, help="one domain .raw to test")
    ap.add_argument("--result-dir", default=None,
                    help="where GeoDict writes (default: ./test_one_result)")
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--nz", type=int, default=None)
    ap.add_argument("--voxel-m", type=float, default=None)
    args = ap.parse_args()

    raw = Path(args.raw)
    if not raw.exists():
        sys.exit(f"raw not found: {raw}")

    # dimensions: from filename unless overridden
    nx, ny, nz, voxel_m = args.nx, args.ny, args.nz, args.voxel_m
    m = _NAME_RE.search(raw.name)
    if m:
        nx = nx or int(m.group("nx"))
        ny = ny or int(m.group("ny"))
        nz = nz or int(m.group("nz"))
        voxel_m = voxel_m or float(m.group("voxel")) * 1e-6
    if not all([nx, ny, nz, voxel_m]):
        sys.exit("could not determine nx/ny/nz/voxel; pass --nx --ny --nz --voxel-m")

    # cross-check the file size matches the dimensions (uint8 = 1 byte/voxel)
    expected = nx * ny * nz
    actual = raw.stat().st_size
    if actual != expected:
        print(f"  WARNING: file is {actual:,} bytes but {nx}x{ny}x{nz} = "
              f"{expected:,} (uint8). Dimensions may be wrong.")

    result_dir = Path(args.result_dir) if args.result_dir else Path("test_one_result")
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.geodict_exe, args.macro,
        "-v", "raw_path",   str(raw),
        "-v", "nx",         str(nx),
        "-v", "ny",         str(ny),
        "-v", "nz",         str(nz),
        "-v", "voxel_m",    repr(voxel_m),
        "-v", "result_dir", str(result_dir),
    ]
    print("Running GeoDict on one domain:")
    print(f"  file   : {raw.name}")
    print(f"  dims   : {nx} x {ny} x {nz}   voxel = {voxel_m:.6e} m")
    print(f"  output : {result_dir}")
    print("  command:\n    " + " ".join(cmd) + "\n")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    (result_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (result_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")

    print(f"GeoDict exit code: {proc.returncode}")
    if proc.returncode != 0:
        print("\nGeoDict reported an error. Last lines of stderr:")
        print("\n".join(proc.stderr.splitlines()[-15:]))
        print(f"\nFull logs in {result_dir}\\stdout.txt / stderr.txt")
        sys.exit(1)

    summary = result_dir / "result_summary.json"
    if not summary.exists():
        print(f"\nGeoDict finished but wrote no result_summary.json in {result_dir}.")
        print("The macro's IMPORT or SOLVE step likely did not complete — check "
              "stdout.txt and confirm your recorded import command is in the macro.")
        sys.exit(1)

    data = json.loads(summary.read_text(encoding="utf-8"))
    perm = data.get("permeability_z_m2")
    print("\n" + "=" * 50)
    if perm is None:
        print("RESULT: solve ran but NO permeability was extracted.")
        print("The .gdr exists but the field PhysicalFlowPermeabilitiesZ was not")
        print("found by gd.getResultFile(). Send me the result_summary.json and")
        print("the .gdr folder so I can fix the extraction field name.")
        print("=" * 50)
        sys.exit(1)
    print(f"RESULT: SUCCESS")
    print(f"  permeability_z = {perm:.6e} m^2  ({perm * 1.01325e15:.3f} mDarcy)")
    print(f"  stopping flag  = {data.get('stopping_criteria_flag')}")
    print(f"  GeoDict        = {data.get('geodict_version')}")
    print("=" * 50)
    print("\nImport + solve + extraction all work. Safe to run the full batch.")


if __name__ == "__main__":
    main()
