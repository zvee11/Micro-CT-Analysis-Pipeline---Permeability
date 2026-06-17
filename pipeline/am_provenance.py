"""
am_provenance.py — extract the Avizo HistoryLog (prior-work processing chain)
from an .am file header, for documentation in the provenance database.

The .am header (ASCII, before the binary lattice) contains a HistoryLog with one
block per Avizo module that was applied, each carrying its parameters, label,
Avizo version and date. This parses that chain so it can be stored verbatim.

Pure-Python, no Avizo/ahds needed: only the ASCII header is read (bounded), so
this is cheap and safe to call per scan during a run.
"""
from __future__ import annotations

import re
from pathlib import Path

# How many bytes of the header to read. The HistoryLog sits near the top, well
# within this; we never read the (huge) binary lattice.
_HEADER_BYTES = 2_000_000


def _read_header_text(am_path: Path) -> str:
    with open(am_path, "rb") as fh:
        raw = fh.read(_HEADER_BYTES)
    # decode permissively; binary tail (if any sneaks in) becomes replacement chars
    return raw.decode("latin-1", errors="replace")


def parse_am_history(am_path: Path) -> list[dict]:
    """Return the processing chain as a list of step dicts, in file order.

    Each step: {order, module, label, avizo_version, date, parameters{...}}.
    Returns [] if no HistoryLog is present.
    """
    text = _read_header_text(Path(am_path))

    steps: list[dict] = []
    # Each module block starts at "Module:<Name> {" and we capture until the
    # matching Product/Version/Date that closes its UID record.
    # Blocks are delimited by "UID:<uuid> {"; split on those.
    uid_blocks = re.split(r"UID:[0-9a-fA-F-]{36}\s*\{", text)
    order = 0
    for block in uid_blocks:
        mod = re.search(r"Module:([A-Za-z0-9_\-]+)\s*\{", block)
        if not mod:
            continue
        order += 1
        module = mod.group(1)

        params: dict[str, str] = {}
        for pm in re.finditer(r'Parameter:([A-Za-z0-9_]+)\s+"([^"]*)"', block):
            params[pm.group(1)] = pm.group(2)

        label = None
        lm = re.search(r'Label\s+"([^"]*)"', block)
        if lm:
            label = lm.group(1)

        version = None
        vm = re.search(r'Version\s+"([^"]*)"', block)
        if vm:
            version = vm.group(1)

        date = None
        dm = re.search(r'Date\s+"([^"]*)"', block)
        if dm:
            date = dm.group(1)

        steps.append({
            "order": order,
            "module": module,
            "label": label,
            "avizo_version": version,
            "date": date,
            "parameters": params,
        })
    return steps


def parse_am_lattice(am_path: Path):
    """Return (nz, ny, nx) from 'define Lattice X Y Z' and the BoundingBox tuple
    if present, else (None, None). Avizo lattice order is X Y Z."""
    text = _read_header_text(Path(am_path))
    lattice = None
    lm = re.search(r"define\s+Lattice\s+(\d+)\s+(\d+)\s+(\d+)", text)
    if lm:
        x, y, z = int(lm.group(1)), int(lm.group(2)), int(lm.group(3))
        lattice = (z, y, x)  # return in (nz, ny, nx)
    bbox = None
    bm = re.search(r"BoundingBox\s+([-0-9.eE ]+)", text)
    if bm:
        nums = [float(v) for v in bm.group(1).split()][:6]
        if len(nums) == 6:
            bbox = tuple(nums)  # xmin xmax ymin ymax zmin zmax
    return lattice, bbox


def compute_voxel_size(lattice, bbox):
    """Voxel size (dz, dy, dx) in metres from lattice dims (nz,ny,nx) and
    BoundingBox (xmin xmax ymin ymax zmin zmax). Avizo uses node-centered
    coordinates, so the box spans voxel CENTRES: size = extent / (n - 1).

    This is the AUTHORITATIVE voxel size, extracted (not assumed) from the .am
    header. For these scans it comes to ~4.9968 µm, not the nominal 5 µm.
    Returns (dz, dy, dx) in metres, or None if inputs are missing.
    """
    if not lattice or not bbox:
        return None
    nz, ny, nx = lattice
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    if nx < 2 or ny < 2 or nz < 2:
        return None
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    dz = (zmax - zmin) / (nz - 1)
    return (dz, dy, dx)


if __name__ == "__main__":
    import sys, json
    p = Path(sys.argv[1])
    chain = parse_am_history(p)
    lat, bbox = parse_am_lattice(p)
    print(f"lattice (nz,ny,nx): {lat}")
    print(f"bounding box: {bbox}")
    print(f"{len(chain)} processing steps:\n")
    for s in chain:
        print(f"  [{s['order']:>2}] {s['module']:<26} "
              f"v{s['avizo_version']}  {s['date']}")
        if s['label']:
            print(f"       label: {s['label']}")
        for k, v in s['parameters'].items():
            print(f"       {k} = {v}")
        print()
