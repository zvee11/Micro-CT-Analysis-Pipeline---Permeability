# -*- coding: utf-8 -*-
# =====================================================================
#  GeoPy macro  —  run ONE LIR Stokes flow simulation on ONE .raw domain
#  and write the Z-permeability to a JSON sidecar.
#
#  Runs INSIDE GeoDict (headless), e.g.:
#    "C:\Program Files\Math2Market\GeoDict2026\GeoDict2026.exe" geodict_lir_job.py
#
#  Per-domain values (raw_path, nx, ny, nz, voxel_m, result_dir) are HARDCODED
#  at the top of main() for the single-domain test. Edit them to run another
#  file. (A later version can generate one macro per domain automatically.)
#
#  The SolveStokes settings below are copied VERBATIM from the paper's
#  production run (recorded by Shell, GeoDict 2026, PC07/PC19). The ONLY
#  things parameterised are the input file, its dimensions, voxel size,
#  and where to write the result.
#
#  PHASE MAPPING (matches our exported domains):
#     grayvalue 0 = FLUID (open / brine)   -> Material00 (Water/Brine)
#     grayvalue 1 = SOLID (blocked / rock) -> Material01 (Solid wall)
#  Both the import and the solve use this same order. The solid name does not
#  affect single-phase permeability (it is only a no-slip wall).
# =====================================================================

Header = {
    'Release': '2026',
}

Description = '''
Headless LIR Stokes runner for one binary domain (.raw, 0=fluid, 1=solid).
Solver settings replicate the thesis production run exactly.
'''

Variables = [
  ]

import json
import os


def _parse_domain_name(path):
    """Extract (nx, ny, nz, voxel_m) from a domain filename of the form
        domain_<type>_<track>-<scan>_<voxel>um_<bits>_<NX>x<NY>x<NZ>.raw
    e.g. domain_gas_02-8_4.99676um_8bu_750x750x273.raw
    """
    import re
    name = os.path.basename(path)
    m = re.search(r"_([0-9.]+)um_\d+b[us]_(\d+)x(\d+)x(\d+)\.raw$", name, re.IGNORECASE)
    if not m:
        raise RuntimeError("filename does not match the domain convention: " + name)
    voxel_m = float(m.group(1)) * 1e-6
    nx, ny, nz = int(m.group(2)), int(m.group(3)), int(m.group(4))
    return nx, ny, nz, voxel_m


def main():
    # ---- ONLY the input file path is per-domain (baked in per run). ----
    raw_path = 'C:/Users/99619/Desktop/SVETA/Micro-CT-Analysis-Pipeline/output/9_5_sub_registered_filtered_thresholded_extracted/18N/domain_gas_02-8_4.99676um_8bu_750x750x273.raw'

    # Dimensions + voxel size are PARSED from the filename (not hardcoded).
    nx, ny, nz, voxel_m = _parse_domain_name(raw_path)

    # Results go in output/<scan_folder>/geodict/<domain_stem>/
    #   raw lives at output/<scan_folder>/<conn>/<domain>.raw, so the scan folder
    #   is two levels up from the file.
    raw_dir    = os.path.dirname(raw_path)               # .../<scan>/<conn>
    scan_dir   = os.path.dirname(raw_dir)                # .../<scan>
    domain_stem = os.path.splitext(os.path.basename(raw_path))[0]
    result_dir = os.path.join(scan_dir, "geodict", domain_stem)

    os.makedirs(result_dir, exist_ok=True)

    # ---- RESUME: skip this domain if it already has a valid result. ----
    # An interrupted batch can be re-run; domains already solved are skipped.
    existing = os.path.join(result_dir, "result_summary.json")
    if os.path.isfile(existing):
        try:
            with open(existing, "r", encoding="utf-8") as ef:
                prev = json.load(ef)
            if prev.get("permeability_z_m2") is not None:
                print(f"  [macro] SKIP (already done): {os.path.basename(raw_path)} "
                      f"-> Kz = {prev.get('permeability_z_m2')} m^2")
                return
        except Exception:
            pass   # unreadable/partial -> fall through and re-solve

    # GeoDict has NO setProjectFolder; ResultFileName must be a BARE filename and
    # GeoDict writes it into its own project folder (read via getProjectFolder()).
    # We solve with the bare name, then move the outputs into our result_dir.
    gdr_filename = "StokesResult.gdr"

    # =================================================================
    #  STEP 1 — IMPORT the .raw as a structure.
    #
    #  This is the REAL recorded import command for GeoDict 2026 (recorded via
    #  Macro -> Session Macro, by Steffen Berg / Shell). Parameterised by the
    #  per-file variables (raw_path, nx, ny, nz, voxel_m) so it runs on any of
    #  our exported domains.
    #
    #  Material mapping matches our domains: gray 0 = Fluid (Water/Brine),
    #  gray 1 = Solid. ThresholdValue 0.5 splits the two on a binary mask.
    # =================================================================
    import_args = {
        'FileName'        : raw_path,
        'DimensionX'      : nx,
        'CutStartX'       : 1,
        'CutLengthX'      : nx,
        'DimensionY'      : ny,
        'CutStartY'       : 1,
        'CutLengthY'      : ny,
        'DimensionZ'      : nz,
        'CutStartZ'       : 1,
        'CutLengthZ'      : nz,
        'VoxelLength'     : voxel_m,
        'ResampleToVoxel' : True,
        'HeaderSize'      : 0,
        'ThresholdValue'  : 0.5,
        'IsRawFile'       : True,
        'IsBigEndian'     : False,
        'ImageDimensions' : 'Auto',
        'CutImage'        : False,
        'Material00' : {
            'Type'        : 'Fluid',
            'Name'        : 'Water (Brine)',
            'Information' : '',
        },
        'Material01' : {
            'Type'        : 'Solid',
            'Name'        : 'Manual',
            'Information' : '',
        },
    }
    gd.runCmd("ImportGeoVol:Import", import_args, Header['Release'])

    # =================================================================
    #  STEP 2 — SOLVE Stokes with LIR. Settings copied VERBATIM from the
    #  paper's production macro (PC07 / PC19). Do not change these.
    # =================================================================
    SolveStokes_args = {
        'ResultFileName': gdr_filename,
        'Experiment': {
            'DirectionEnabledX': False,
            'DirectionEnabledY': False,
            'DirectionEnabledZ': True,
            'CharLengthMode': 'PERMEABILITY',
            'GivenCharLength': 1,
            'PressureDifference': (0.02, 'Pa'),
            'MeanVelocity': (0.1, 'm/s'),
            'FlowRate': (60, 'l/min'),
            'FlowArea': (100, 'cm^2'),
            'ExperimentIO': 'PressureDrop',
            'AddInletOutlet': True,
            'InletLength': 10,
            'OutletLength': 10,
            'NormalBcType': 'Periodic',
            'SlipLength': 0,
            'PoreSolidBoundaryCondition': 'NoSlip',
            'TangentialBcYInX': 'Periodic',
            'TangentialBcZInX': 'Periodic',
            'TangentialBcXInY': 'Periodic',
            'TangentialBcZInY': 'Periodic',
            'TangentialBcXInZ': 'Periodic',
            'TangentialBcYInZ': 'Periodic',
            'UseSecondOrderSlip': True,
            'UseSharpCorner': True,
        },
        'ConstituentMaterials': {
            'Temperature': (293.15, 'K'),
            # Order matches the import: gray 0 -> Material00 (Fluid/Brine, the
            # flowing phase), gray 1 -> Material01 (Solid wall). The solid name
            # does not affect single-phase permeability (it is just a no-slip
            # wall), so 'Manual' is kept for consistency with the import.
            'Material00': {
                'Type': 'Fluid',
                'Name': 'Water (Brine)',
                'Information': '',
                'FluidProperties': {
                    'CurrentLaw': 1,
                    'NumberOfLaws': 3,
                    'MaterialLaw1': {
                        'Name': 'Sea Water (3.5%)',
                        'Density': (1025, 'kg/m^3'),
                        'Viscosity': (0.00109, 'kg/(ms)'),
                    },
                    'MaterialLaw2': {
                        'Name': 'Low Salinity Water',
                        'Density': (1001.8, 'kg/m^3'),
                        'Viscosity': (0.001011, 'kg/(ms)'),
                    },
                    'MaterialLaw3': {
                        'Name': 'Formation Water (20%)',
                        'Density': (1147.8, 'kg/m^3'),
                        'Viscosity': (0.001557, 'kg/(ms)'),
                    },
                },
            },
            'Material01': {
                'Type': 'Solid',
                'Name': 'Manual',
                'Information': '',
            },
            'ChosenFluid': {
                'Fluid': 'Water (Brine)',
                'CurrentLaw': 1,
            },
        },
        'SolverData': {
            'Solver': 'LIR',
            'LIR': {
                'Parallelization': {'Mode': 'LOCAL_MAX'},
                'DiscardTemporaryFiles': False,
                'AnalyzeGeometry': True,
                'RestartFileName': '',
                'Restart': False,
                'RestartSaveIntervalTime': 6,
                'Tolerance': 0.0001,
                'MaxNumberOfIterations': 100000,
                'MaximalSolverRunTime': (240, 'h'),
                'UseMaxIterations': False,
                'UseMaxTime': False,
                'UseTolerance': False,
                'UseErrorBound': True,
                'ErrorBound': 0.01,
                'UseLateral': False,
                'Refinement': 'ACCURACY',
                'AllowGridCoarsening': False,
                'RefinementAccuracy': 0.05,
                'AllowSubVoxelResolution': False,
                'NumberOfRefinements': 10,
                'Threshold': 0.1,
                'Optimization': 'Speed',
                'GridType': 'LIR-Tree',
                'UseMultigrid': True,
                'UseKrylov': 'Enabled',
                'Relaxation': 1,
                'WriteCompressedFields': True,
            },
            # GeoDict's SolveStokes pre-check requires ALL three solver blocks
            # to be present (LIR, SimpleFFT, EJ), even though Solver='LIR'. These
            # two are copied verbatim from the paper's recorded script.
            'SimpleFFT': {
                'UseResidual': False,
                'Residual': 0.0001,
                'Tolerance': 0.0001,
                'MaxNumberOfIterations': 100000,
                'MaximalSolverRunTime': (240, 'h'),
                'UseMaxIterations': False,
                'UseMaxTime': False,
                'UseTolerance': False,
                'UseErrorBound': True,
                'ErrorBound': 0.01,
                'UseLateral': False,
                'Parallelization': {'Mode': 'LOCAL_MAX'},
                'DiscardTemporaryFiles': False,
                'AnalyzeGeometry': True,
                'RestartFileName': '',
                'Restart': False,
                'RestartSaveIntervalTime': 6,
                'RelaxationVelocity': 1,
                'RelaxationPressure': 1,
                'TdmaMode': 'Automatic',
            },
            'EJ': {
                'UseResidual': False,
                'Residual': 0.001,
                'Tolerance': 0.001,
                'MaxNumberOfIterations': 100000,
                'MaximalSolverRunTime': (240, 'h'),
                'UseMaxIterations': False,
                'UseMaxTime': False,
                'UseTolerance': True,
                'Parallelization': {'Mode': 'LOCAL_MAX'},
                'DiscardTemporaryFiles': False,
                'AnalyzeGeometry': True,
                'RestartFileName': '',
                'Restart': False,
                'RestartSaveIntervalTime': 6,
            },
        },
        'Grid': {
            'UseBoxels': False,
            'BoxelLengthX': voxel_m,
            'BoxelLengthY': voxel_m,
            'BoxelLengthZ': voxel_m,
        },
    }
    gd.runCmd("FlowDict:SolveStokes", SolveStokes_args, Header['Release'])

    # =================================================================
    #  STEP 3 — EXTRACT the Z permeability from the .gdr result and write
    #  a small JSON the outer Python script reads. We pull the same field
    #  the paper reports: PhysicalFlowPermeabilitiesZ (m^2).
    # =================================================================
    # Where GeoDict wrote the result. The solver writes a plain-text result file
    # <project_folder>/<gdr_stem>/SolverResult_z.txt that contains the
    # permeability and stopping info. We parse that directly (the getResultFile()
    # API needs the .gdr loaded into the session, which it is not here).
    try:
        proj = gd.getProjectFolder()
    except Exception:
        proj = ""
    gdr_stem = os.path.splitext(gdr_filename)[0]            # "StokesResult"
    result_txt = os.path.join(proj, gdr_stem, "SolverResult_z.txt")

    perm_z = None
    stopping = None
    reached = None
    try:
        with open(result_txt, "r", encoding="utf-8", errors="replace") as rf:
            for line in rf:
                parts = line.split(None, 1)               # key, rest
                if len(parts) != 2:
                    continue
                key, val = parts[0].strip(), parts[1].strip()
                kl = key.lower()
                if kl == "physicalflowpermeabilitiesz":
                    perm_z = _to_float(val)
                elif kl == "stoppingcriteriaflag":
                    stopping = _to_float(val)
                elif kl == "reachederrorboundz":
                    reached = _to_float(val)
    except Exception as e:
        print("  [macro] could not read SolverResult_z.txt:", e)

    out = {
        "raw_path": raw_path,
        "gdr": os.path.join(proj, gdr_filename) if proj else gdr_filename,
        "project_folder": proj,
        "result_txt": result_txt,
        "nx": nx, "ny": ny, "nz": nz,
        "voxel_m": voxel_m,
        "permeability_z_m2": perm_z,
        "stopping_criteria_flag": stopping,
        "reached_error_bound_z": reached,
        "geodict_version": _safe(gd.getVersion),
    }
    with open(os.path.join(result_dir, "result_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  [macro] {os.path.basename(raw_path)} -> Kz = {perm_z} m^2")
    print(f"  [macro] read result from: {result_txt}")
    print(f"  [macro] result_summary.json written to {result_dir}")

    # ---- Collect GeoDict's output files into result_dir (move, not copy). ----
    # Everything EXCEPT the large FlowField_z.vap (~400 MB) is moved so the
    # project folder is clean for the next domain and nothing is overwritten.
    import shutil
    SKIP = {"flowfield_z.vap"}                       # too large to keep per-domain
    moved = []
    if proj:
        # 1) top-level result files next to the project folder
        for fname in (gdr_filename, gdr_stem + ".pdf"):
            src = os.path.join(proj, fname)
            if os.path.isfile(src):
                try:
                    shutil.move(src, os.path.join(result_dir, fname))
                    moved.append(fname)
                except Exception as e:
                    print(f"  [macro] could not move {fname}: {e}")
        # 2) the StokesResult/ subfolder contents (txt, log, pde, gdt, ...)
        sub = os.path.join(proj, gdr_stem)
        if os.path.isdir(sub):
            dest_sub = os.path.join(result_dir, gdr_stem)
            os.makedirs(dest_sub, exist_ok=True)
            for fname in os.listdir(sub):
                if fname.lower() in SKIP:
                    # delete the huge flow field rather than keep or move it
                    try:
                        os.remove(os.path.join(sub, fname))
                    except Exception:
                        pass
                    continue
                try:
                    shutil.move(os.path.join(sub, fname),
                                os.path.join(dest_sub, fname))
                    moved.append(gdr_stem + "/" + fname)
                except Exception as e:
                    print(f"  [macro] could not move {fname}: {e}")
            # remove the now-empty StokesResult/ in the project folder
            try:
                os.rmdir(sub)
            except Exception:
                pass
    print(f"  [macro] moved {len(moved)} GeoDict output files into {result_dir}")

    # Copy ALL GeoDict outputs into result_dir next to the json: the .gdr file
    # and the whole <gdr_stem> results folder (SolverResult_z.txt, fields, etc.).
    import shutil
    try:
        src_gdr = os.path.join(proj, gdr_filename)
        if os.path.isfile(src_gdr):
            shutil.copy2(src_gdr, os.path.join(result_dir, gdr_filename))
        src_resfolder = os.path.join(proj, gdr_stem)
        if os.path.isdir(src_resfolder):
            dst_resfolder = os.path.join(result_dir, gdr_stem)
            if os.path.isdir(dst_resfolder):
                shutil.rmtree(dst_resfolder)
            shutil.copytree(src_resfolder, dst_resfolder)
        print(f"  [macro] copied .gdr and {gdr_stem}/ into {result_dir}")
    except Exception as e:
        print("  [macro] could not copy GeoDict outputs:", e)


def _flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = d
    return out


def _to_float(v):
    try:
        if isinstance(v, (int, float)):
            return float(v)
        return float(str(v).split()[0])
    except Exception:
        return None


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None


# GeoDict executes this file directly (not as "__main__"), so call main()
# unconditionally at the top level.
main()