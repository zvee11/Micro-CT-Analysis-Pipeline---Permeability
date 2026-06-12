from __future__ import annotations

import csv
import datetime
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SaturationRecord:
    scan_index: int
    file_name: str
    elapsed_minutes: float
    sg_ref: float
    sw_ref: float


def _extract_prefix(name: str) -> str:
    """Leading numeric underscore-prefix of a scan name, e.g. '8_2_sub...' -> '8_2'."""
    tokens = []
    for p in str(name).split("_"):
        if re.match(r"^\d+$", p):
            tokens.append(p)
        else:
            break
    return "_".join(tokens) if tokens else ""


def _load_raw_rows(path: Path, name_col: int, sg_col: int, time_col: int,
                   logger: logging.Logger) -> list[dict]:
    ext = path.suffix.lower()
    rows_raw = []

    if ext in (".xlsx", ".xls", ".xlsm"):
        try:
            import openpyxl
        except ImportError:
            raise ImportError("openpyxl is required to read Excel files. Install with: pip install openpyxl")
        wb = openpyxl.load_workbook(str(path), data_only=True)
        for row in wb.active.iter_rows(values_only=True):
            rows_raw.append(list(row))
    elif ext == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                rows_raw.append(row)
    else:
        raise ValueError(f"Unsupported saturation file format: {ext!r}. Expected .csv, .xlsx, or .xls")

    results = []
    max_col = max(name_col, sg_col, time_col)

    for row in rows_raw:
        if len(row) <= max_col:
            continue
        name_val, sg_val, time_val = row[name_col], row[sg_col], row[time_col]

        if name_val is None or str(name_val).strip() == "":
            continue
        try:
            sg = float(sg_val)
        except (TypeError, ValueError):
            continue
        if math.isnan(sg) or sg < 0 or sg > 1:
            continue

        # Excel datetime.time means time-of-day, not elapsed minutes — skip it.
        if isinstance(time_val, datetime.time):
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

        results.append({"prefix": prefix, "name": str(name_val), "sg": sg, "elapsed_minutes": elapsed})

    logger.info("saturation file: loaded %d valid rows from %s", len(results), path.name)
    return results


def load_saturation_reference(
    saturation_file: str | Path,
    all_files: list[Path],
    name_col: int,
    sg_col: int,
    time_col: int,
    logger: logging.Logger,
) -> dict[int, SaturationRecord]:
    """Match reference saturation rows to scan files by numeric prefix; scan_index -> record."""
    path = Path(saturation_file)
    if not path.exists():
        logger.warning("saturation file not found: %s — reference comparison disabled", path)
        return {}

    raw_rows = _load_raw_rows(path, name_col, sg_col, time_col, logger)
    if not raw_rows:
        logger.warning("saturation file contained no valid rows — check column indices")
        return {}

    prefix_to_row = {row["prefix"]: row for row in raw_rows}

    result: dict[int, SaturationRecord] = {}
    matched = 0
    unmatched = []
    for scan_index, fpath in enumerate(all_files):
        prefix = _extract_prefix(fpath.stem)
        if prefix in prefix_to_row:
            row = prefix_to_row[prefix]
            sg = row["sg"]
            result[scan_index] = SaturationRecord(
                scan_index=scan_index, file_name=fpath.name,
                elapsed_minutes=row["elapsed_minutes"], sg_ref=sg, sw_ref=1.0 - sg,
            )
            matched += 1
        else:
            unmatched.append(f"{scan_index}:{fpath.name}")

    logger.info("saturation matching: %d/%d scans matched", matched, len(all_files))
    if unmatched:
        logger.warning("unmatched scans (no reference row): %s", ", ".join(unmatched))
    return result


def build_elapsed_minutes_array(
    all_files: list[Path],
    sat_records: dict[int, "SaturationRecord"],
) -> list[float]:
    """Elapsed-minutes array aligned to all_files; NaN where no reference record."""
    return [sat_records[i].elapsed_minutes if i in sat_records else float("nan")
            for i in range(len(all_files))]
