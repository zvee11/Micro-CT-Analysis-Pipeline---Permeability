"""
pipeline/saturation.py

Loads a user-supplied reference saturation file (CSV or Excel) and matches
each row to the pipeline's scan files by numeric prefix.

The file is expected to contain at minimum:
  - A scan name column (e.g. "8_2_sub_registered_filtered...")
  - A gas saturation Sg column (float, 0–1)
  - An elapsed time column (float, minutes from experiment start)

Column positions are specified in Config (0-based indices).
The file format is detected automatically from the extension.

Matching logic:
  Each row's name is split on underscore and the leading numeric tokens
  are extracted (e.g. "8_2_sub..." -> "8_2"). Each pipeline scan file
  is matched the same way. Rows with no matching scan are ignored.
  Scans with no matching row get NaN for elapsed_minutes and sw_ref.

Returns:
  A dict mapping scan_index -> SaturationRecord
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SaturationRecord:
    scan_index: int
    file_name: str
    elapsed_minutes: float      # from reference file
    sg_ref: float               # gas saturation from reference
    sw_ref: float               # 1 - sg_ref


def _extract_prefix(name: str) -> str:
    """
    Extract the leading numeric prefix from a scan name.

    Examples:
        '8_2_sub_registered...' -> '8_2'
        '9_1_sub_registered_filtered_thresholded_VolFraction3d' -> '9_1'
        '10_3_sub...' -> '10_3'
    """
    parts = str(name).split("_")
    tokens = []
    for p in parts:
        if re.match(r'^\d+$', p):
            tokens.append(p)
        else:
            break
    return "_".join(tokens) if tokens else ""


def _load_raw_rows(
    path: Path,
    name_col: int,
    sg_col: int,
    time_col: int,
    logger: logging.Logger,
) -> list[dict]:
    """
    Load rows from CSV or Excel. Returns list of dicts with keys:
    'name', 'sg', 'elapsed_minutes'.
    Skips rows where any required value is missing or non-numeric.
    """
    ext = path.suffix.lower()
    rows_raw = []

    if ext in (".xlsx", ".xls", ".xlsm"):
        try:
            import openpyxl
        except ImportError:
            raise ImportError(
                "openpyxl is required to read Excel files.\n"
                "Install with: pip install openpyxl"
            )
        wb = openpyxl.load_workbook(str(path), data_only=True)
        ws = wb.active
        for row in ws.iter_rows(values_only=True):
            rows_raw.append(list(row))

    elif ext == ".csv":
        import csv
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                rows_raw.append(row)
    else:
        raise ValueError(
            f"Unsupported saturation file format: {ext!r}. "
            "Expected .csv, .xlsx, or .xls"
        )

    results = []
    max_col = max(name_col, sg_col, time_col)

    for i, row in enumerate(rows_raw):
        if len(row) <= max_col:
            continue

        name_val = row[name_col]
        sg_val   = row[sg_col]
        time_val = row[time_col]

        # Skip empty or header rows
        if name_val is None or str(name_val).strip() == "":
            continue

        # Parse Sg
        try:
            sg = float(sg_val)
        except (TypeError, ValueError):
            continue
        if math.isnan(sg) or sg < 0 or sg > 1:
            continue

        # Parse elapsed minutes
        # Handle datetime.time objects from openpyxl
        import datetime
        if isinstance(time_val, datetime.time):
            # This is an Excel time — it represents hours:minutes of day,
            # NOT elapsed time. Skip — the formula-computed column (H) is
            # what we want. If this column was selected, it will be a float.
            continue
        try:
            elapsed = float(time_val)
        except (TypeError, ValueError):
            continue
        if math.isnan(elapsed) or elapsed < 0:
            continue

        prefix = _extract_prefix(str(name_val))
        if not prefix:
            continue

        results.append({
            "prefix":          prefix,
            "name":            str(name_val),
            "sg":              sg,
            "elapsed_minutes": elapsed,
        })

    logger.info(
        "saturation file: loaded %d valid rows from %s",
        len(results), path.name,
    )
    return results


def load_saturation_reference(
    saturation_file: str | Path,
    all_files: list[Path],
    name_col: int,
    sg_col: int,
    time_col: int,
    logger: logging.Logger,
) -> dict[int, SaturationRecord]:
    """
    Load reference saturation data and match to pipeline scan files.

    Parameters
    ----------
    saturation_file:
        Path to CSV or Excel file.
    all_files:
        List of scan file paths in pipeline order (index = scan_index).
    name_col, sg_col, time_col:
        0-based column indices for scan name, Sg, and elapsed minutes.

    Returns
    -------
    dict mapping scan_index -> SaturationRecord for matched scans only.
    Unmatched scans are absent from the dict.
    """
    path = Path(saturation_file)
    if not path.exists():
        logger.warning(
            "saturation file not found: %s — reference comparison disabled",
            path,
        )
        return {}

    raw_rows = _load_raw_rows(path, name_col, sg_col, time_col, logger)
    if not raw_rows:
        logger.warning("saturation file contained no valid rows — check column indices")
        return {}

    # Build prefix -> row lookup (last match wins if duplicates)
    prefix_to_row: dict[str, dict] = {}
    for row in raw_rows:
        prefix_to_row[row["prefix"]] = row

    # Match each scan file
    result: dict[int, SaturationRecord] = {}
    matched = 0
    unmatched = []

    for scan_index, fpath in enumerate(all_files):
        # Strip .am extension and try to extract prefix from file stem
        stem = fpath.stem  # e.g. '8_2_sub_registered_filtered_thresholded_extracted'
        prefix = _extract_prefix(stem)

        if prefix in prefix_to_row:
            row = prefix_to_row[prefix]
            sg = row["sg"]
            sw_ref = 1.0 - sg
            result[scan_index] = SaturationRecord(
                scan_index=scan_index,
                file_name=fpath.name,
                elapsed_minutes=row["elapsed_minutes"],
                sg_ref=sg,
                sw_ref=sw_ref,
            )
            matched += 1
        else:
            unmatched.append(f"{scan_index}:{fpath.name}")

    logger.info(
        "saturation matching: %d/%d scans matched",
        matched, len(all_files),
    )
    if unmatched:
        logger.warning(
            "unmatched scans (no reference row): %s",
            ", ".join(unmatched),
        )

    return result


def build_elapsed_minutes_array(
    all_files: list[Path],
    sat_records: dict[int, "SaturationRecord"],
) -> list[float]:
    """
    Build an elapsed_minutes array aligned to all_files.
    Scans with no reference record get NaN.
    Used as the X axis for regime detection and Dash plots.
    """
    result = []
    for i in range(len(all_files)):
        if i in sat_records:
            result.append(sat_records[i].elapsed_minutes)
        else:
            result.append(float("nan"))
    return result
