# Micro-CT Pore-Scale Analysis Pipeline

Automated data-management and analysis pipeline for time-resolved micro-CT
volumes of porous rock, for underground hydrogen storage research. The pipeline
ingests segmented Avizo/Amira volumes, detects saturation regimes, extracts and
tracks connected gas clusters, builds pore-scale flow-simulation domains, and
records every intermediate product in an embedded DuckDB database.

---

## Repository contents

```
pipeline/              The analysis pipeline (Python package)
solve.py               Interactive launcher for MPLBM-UT flow simulations (WSL/Linux)
viewer.py              Standalone 3D PyVista cluster viewer
check.py               Quick Avizo-file codec/shape check utility
validation/            Scripts comparing pipeline output against Avizo/manual results
visualization/         Standalone plotting/visualisation scripts
requirements.txt       Python dependencies
.gitignore             Excludes data, outputs, database, and virtual environments
```

Not in the repository (excluded by `.gitignore`, kept locally on each machine):
`data/` (input `.am` volumes and the reference saturation spreadsheet),
`output/` (generated domains and images), `results.duckdb` (generated database),
and the virtual environment.

---

## Requirements

- **Python 3.10 or newer** (the code uses `X | Y` type unions and
  `from __future__ import annotations`).
- The Python packages in `requirements.txt`.
- **For flow simulation only:** a Linux environment with MPLBM-UT and an MPI
  runtime installed. On Windows this is provided through WSL; on a Linux
  workstation it runs natively. The geometry pipeline itself does **not** need
  MPLBM-UT or MPI.

---

## Setting up the environment

The virtual environment is intentionally **not** committed to Git (it is large
and machine-specific). This project was built with **pip**, not conda — use pip
on every machine so the environment matches. Recreate it on each machine:

```bash
# from the repository root
python -m venv .venv

# activate it:
source .venv/bin/activate        # Linux / WSL / macOS
# .venv\Scripts\activate         # Windows PowerShell/CMD

pip install --upgrade pip
pip install -r requirements.lock.txt
```

Two requirements files are provided:

- **`requirements.lock.txt`** — exact pinned versions from a known-working
  environment, trimmed to only what this codebase imports (pipeline core +
  PyVista viewer + Dash dashboard). **Use this to install.**
- **`requirements.txt`** — human-readable list of direct dependencies and why
  each is needed.

The lock file keeps the PyVista 3D viewer and the Dash dashboard. It includes
`PyQt5` and `vtk`, which the PyVista background plotter needs — these require a
graphical display (or an X/VNC session) and OpenGL. If you run the workstation
**headlessly** (terminal only), the geometry pipeline and simulations still work,
but the interactive 3D viewer will not open; do that visualisation on a machine
with a display.

MPLBM-UT and MPI are **not** pip packages. Install them separately on the
machine that runs the flow simulations, the same way they are set up under WSL.

---

## Local data layout

The pipeline reads from `data/` and writes to `output/`, relative to the
directory you run it from (see `pipeline/config.py`). On each machine, create:

```
<project root>/
├── data/                     # place input .am volumes here
│   ├── *.am
│   └── 18_Sg_3d.xlsx         # reference saturation file (see config.py)
├── output/                   # created by the pipeline
└── results.duckdb            # created by the pipeline
```

Paths and all other parameters live in `pipeline/config.py` — edit there, not in
the code. Key settings include the gas label, number of clusters to keep,
connectivity, crop mode, regime-cutoff policy, and the reference-saturation file
path and column indices.

---

## Running the pipeline

**Geometry pipeline** (ingestion, regime detection, clustering, domain
construction, database population):

```bash
python -m pipeline
```

**Flow simulations** (run after the geometry pipeline has populated the database;
requires the Linux/MPLBM-UT environment). Run from the directory containing
`results.duckdb`:

```bash
python solve.py
```

`solve.py` auto-detects the WSL/Linux username and uses the MPLBM-UT, virtual
environment, and working-directory locations defined at the top of the file
(`WSL_MPLBM_DIR`, `WSL_VENV`, `WSL_WORK_DIR`) — adjust those constants if your
workstation differs from the default layout.

**3D viewer** (after a run):

```bash
python viewer.py                      # latest run, all scans
python viewer.py --connectivity 26N
```

---

## Reproducibility notes

- All run parameters are captured in the `runs` table of the database, so each
  set of results is traceable to the configuration that produced it.
- The flow simulation uses the multiple-relaxation-time (MRT) lattice-Boltzmann
  scheme, whose permeability is independent of the relaxation parameter.
- Results may differ in the last significant digits between machines or core
  counts, due to floating-point summation order in parallel execution. This does
  not affect physical conclusions.
