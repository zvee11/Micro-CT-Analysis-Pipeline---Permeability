# Micro-CT Analysis Pipeline — Permeability

Automated pipeline for computing relative permeability from time-resolved micro-CT
scans of Bentheimer sandstone, developed for underground hydrogen storage research
(MSc thesis, TU/e, in collaboration with Shell). The pipeline replaces a previously
manual workflow (segmentation cluster selection, regime detection, domain
construction, flow simulation) with a single reproducible, provenance-complete run:
every parameter, intermediate result, and simulation output is recorded in a
queryable database.

## What it does

Given a time series of segmented micro-CT volumes (Avizo `.am` label fields), the
pipeline:

1. **Prepass** — computes whole-volume water saturation (Sw) per scan and fits a
   three-segment piecewise-linear model to the Sw–time series to detect the
   displacement / transition / dissolution regime boundaries. The last qualifying
   scan (timestep X) is proposed automatically and confirmed interactively.
2. **Connected-component analysis** — identifies the largest gas clusters per scan
   using a memory-bounded, two-pass slab algorithm (cc3d + union-find seam
   stitching), so peak RAM scales with slab depth, not volume size.
3. **Cluster tracking** — matches clusters across consecutive scans by
   center-of-gravity proximity.
4. **Fixed-box extraction** — freezes a spatial box per tracked cluster at
   timestep X and applies the identical box to all earlier scans, so local
   saturations are measured over the same physical rock region at every time.
5. **Simulation domains** — exports three uint8 domains per (track, scan):
   absolute (pore space), gas, and water, with self-describing filenames.
6. **Flow simulation** — a batch orchestrator launches GeoDict (LIR Stokes
   solver) once per domain, extracts the Z-permeability, and writes it back to
   the database. Relative permeabilities follow as kr = K_phase / K_abs.
7. **Analysis** — `corey_curves.py` and the analysis notebooks compute per-section
   saturations, fit basic and modified Corey models, write the fitted parameters
   and per-point relative permeabilities back to the database, and generate the
   thesis figures.

## Repository layout

```
pipeline/            The pipeline package (run with: python -m pipeline)
  config.py            all run parameters (frozen dataclass)
  io.py                Avizo .am reader (header-driven, memmap)
  preprocessing.py     flow crop, wrap-around fix, COG
  connected.py         two-pass slab CCA + union-find
  saturation.py        experimental Sw reference loader
  tracking.py          cluster tracking across scans
  regime.py            Sw prepass + 3-segment regime fit + timestep X
  fixed_box.py         frozen-box definition and application
  simulation_domains.py  domain construction + filename convention
  outputs.py           domain writing
  am_provenance.py     Avizo processing-history extraction
  db.py                DuckDB schema and inserts
  ui.py                Rich terminal UI
  visualisation.py     optional live Dash + PyVista observers

geodict_flowsim.py   GeoDict batch orchestrator (one launch per domain, resumable)
corey_curves.py      Corey-model fitting: computes kr from the stored
                     permeabilities, fits basic and modified Corey curves against
                     the manual baseline, and writes both the fitted parameters
                     (corey_fits table) and the per-point kr values back to the
                     database

scripts/             Standalone tooling
  geodict_lir_job.py     GeoPy macro: one LIR Stokes run inside GeoDict
  preflight_geodict.py   read-only dry run of the batch gates
  test_one_domain.py     single-domain smoke test
  check_progress.py      poll batch progress from another terminal
  compare_results.py     kr computation + Corey fits vs prior-work baseline
  export_run_to_excel.py dump one run to xlsx (schema-introspecting)
  check_db.py            run inventory
  delete_empty_runs.py   prune runs without simulation results
  cluster_cca_shrinkage.py / cluster_shrinkage_figures.py  shrinkage figures
  plots_z_clusters.py    per-Z cluster profiles
  viewer.py              standalone 3D cluster viewer
  build_notebook.py      generates comparison_walkthrough.ipynb
  check.py, extract-all.py  small utilities

validation/          One-hypothesis-per-script diagnostics
  avizo-vs-cca.py        Avizo vs pipeline segmentation agreement
  disagreement_analysis.py  interactive drill-down into disagreement voxels
  raw_cc_check.py        loader-vs-CCA isolation test
  split_check.py         crop-artifact test for cluster splits
  connectivity_charts.py 18N vs 26N comparison

visualization/       Interactive comparison tools (Dash + PyVista)

notebooks/
  results_figures.ipynb        generates all results figures from the database
  comparison_walkthrough.ipynb step-by-step comparison narrative

figures/             Generated thesis figures (PNG) + figure index
data/                Small tracked references (saturation + baseline workbooks);
                     raw .am volumes live outside git (see below)
results.duckdb       The results database (runs, scans, tracks, fixed boxes,
                     cluster properties, simulation results, prior-work
                     provenance, Corey fits)
```

## Data conventions

Large binary data is **not** stored in git (see `.gitignore`):

- `data/*.am` — segmented Avizo label volumes (750×750×3780 voxels each)
- `output/` — exported simulation domains, named
  `domain_<type>_<track>-<scan>_<voxel>um_8bu_<NX>x<NY>x<NZ>.raw`
- `results_geodict/` — per-domain GeoDict macros and solver logs

These live on the workstation / network storage. The database records the paths,
parameters, and results, so a run is fully reconstructible from `results.duckdb`
plus the stored volumes.

Small reference tables **are** tracked, so the analysis and comparison stages run
without the raw volumes:

- `data/18_Sg_3d.xlsx` — experimental saturation reference (the `saturation_file`,
  `saturation_sg_col`, and `saturation_time_col` in `config.py` point here)
- `data/Methane_Parente.xlsx` — manual baseline workbook
- `data/kr_results_table.csv` — exported kr results

## Running

```bash
# 1. Environment
pip install -r requirements.txt

# 2. Configure the run
#    edit pipeline/config.py (data_dir, saturation reference, connectivity, n_keep)

# 3. Run the pipeline (prepass -> CCA -> tracking -> boxes -> domains -> DB)
python -m pipeline

# 4. Check what the simulation batch would do (read-only)
python scripts/preflight_geodict.py

# 5. Run the GeoDict batch (requires a licensed GeoDict installation)
python geodict_flowsim.py

# 6. Fit Corey curves and write kr back to the database
python corey_curves.py

# 7. Analysis and figures
jupyter notebook notebooks/results_figures.ipynb
```

GeoDict runs are restartable: domains with an existing database result are skipped.

## Dependencies

Python 3.10+. The direct dependencies the code imports are pinned in
`requirements.txt`; the full resolved environment (all transitive dependencies) is
captured in `requirements.lock.txt`. In brief: numpy, scipy, cc3d
(`connected-components-3d`), scikit-image, ahds (Avizo file reading), tifffile,
duckdb, openpyxl, pandas, rich, psutil, matplotlib, pillow, nbformat, and the
optional visualiser stack (plotly, dash, dash-bootstrap-components, pyvista,
pyvistaqt). GeoDict 2026 with the LIR flow solver is a separate licensed,
Windows-only installation required only for the flow-simulation stage.

## Database

`results.duckdb` — one file, keyed by `run_id` (timestamp + host). Tables:
`runs`, `scans`, `tracks`, `fixed_boxes`, `cluster_properties`,
`simulation_results`, `prior_work_provenance`, and `corey_fits` (the fitted
Corey parameters written by `corey_curves.py`). Open read-only for analysis:

```python
import duckdb
con = duckdb.connect("results.duckdb", read_only=True)
con.execute("SELECT run_id, started_at FROM runs").fetchall()
```
