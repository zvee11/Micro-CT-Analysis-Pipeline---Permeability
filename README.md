# Micro-CT Permeability Project — Sorted

Assembled from all 5 archives (all-zip-local, all-zip, full-zip, repo, semi-archive).
NO code logic was changed. The ONLY code edit is one default-path string in
geodict_flowsim.py (--macro now points to scripts/geodict_lir_job.py). All 113 .py
files were syntax-checked after sorting.

## Buckets
- pipeline/             CORE — git `main`. Live package, geodict_flowsim batch runner,
                        validation, notebooks, scripts (incl. geodict_lir_job macro),
                        visualization, current DB. Has .gitignore + output/ & data/ READMEs.
- useful_but_outdated/  Older but runnable versions (rollback/diff). git-ignored.
- _archive/             Dead/reference: pre-pipeline experiments, 2025 scan_tool scripts,
                        old CSV exports, misc refs. git-ignored.
- report/               Thesis material (built on disk for you; NOT in the delivery zip,
                        git-ignored): main.tex, 18 figures, references, presentations,
                        notes (incl. THE FILES.docx + midterm feedback), admin.

## The one code change (geodict_flowsim.py, line 437)
  BEFORE: default=r"...\Micro-CT-Analysis-Pipeline\geodict_lir_job.py"
  AFTER:  default=r"...\Micro-CT-Analysis-Pipeline\scripts\geodict_lir_job.py"
This is the single edit needed so the batch runner finds the single-domain macro now
that it lives in scripts/. geodict_lir_job.py itself was NOT changed.

## On NAS (excluded by .gitignore, not in tree)
  results_geodict/ (215 GeoDict job macros + logs), *.am volumes, Sg_3d_H2_19.xlsx,
  results-H2-19 / Vol_Frac3d raw CSVs, and the 5.4 GB compare-avizo-and-cca data.

## Figures
  18 of the 26 figures main.tex references are in report/figures/. The other 8
  (4 concept diagrams + 4 _disp sensitivity figures) are sourced from Overleaf and
  left untouched per your instruction.

See _MANIFEST_full.csv for every file with its source and the reason for its bucket.
