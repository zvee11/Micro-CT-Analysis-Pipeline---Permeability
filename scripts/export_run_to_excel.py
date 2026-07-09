"""
export_run_to_excel.py  —  dump everything stored for ONE pipeline run into a
single .xlsx, one sheet per database table, plus computed summary sheets.

The script INTROSPECTS the database: it discovers every table, and for any table
that has a `run_id` column it exports only the rows for the chosen run. Tables
without a `run_id` (reference/lookup tables) are skipped by default so the file
stays scoped to a single run. Two derived sheets are added on top of the raw
tables:

  - "summary_kr"        : per (track, scan) absolute/gas/brine permeability and
                          the relative permeabilities kr_gas = K_g / K_abs and
                          kr_brine = K_w / K_abs, with box-local Sw. This is the
                          curve table, rebuilt from simulation_results.
  - "summary_kabs"      : per-track absolute permeability (one value per track).
  - "run_overview"      : a single-row sheet of run-level facts and counts.

Default run = the most recent run that actually has simulation results (the one
that was analysed). Override with --run-id. Override the connectivity filter for
the computed sheets with --connectivity (raw table sheets are never filtered by
connectivity, so nothing is hidden).

Usage:
    python export_run_to_excel.py
    python export_run_to_excel.py --db results.duckdb --out run_export.xlsx
    python export_run_to_excel.py --run-id 20260618_155713 --connectivity 18N
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

try:
    import pandas as pd
except ImportError:
    sys.exit("ERROR: pip install pandas openpyxl")

# Darcy conversions (k in m^2 -> Darcy / milliDarcy)
DARCY_M2 = 9.869233e-13
M2_TO_DARCY = 1.0 / DARCY_M2
M2_TO_MDARCY = M2_TO_DARCY * 1000.0

# Excel sheet names are capped at 31 chars and cannot contain : \ / ? * [ ]
_BAD_SHEET = set(r':\/?*[]')


def safe_sheet_name(name: str, used: set[str]) -> str:
    """Return an Excel-legal, unique sheet name."""
    s = "".join("_" if c in _BAD_SHEET else c for c in name)[:31]
    if not s:
        s = "sheet"
    base, i = s, 1
    while s.lower() in {u.lower() for u in used}:
        suffix = f"_{i}"
        s = base[:31 - len(suffix)] + suffix
        i += 1
    used.add(s)
    return s


def list_tables(con) -> list[str]:
    """All base tables in the DuckDB 'main' schema."""
    rows = con.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
    ).fetchall()
    return [r[0] for r in rows]


def table_columns(con, table: str) -> list[str]:
    rows = con.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ?
        ORDER BY ordinal_position
        """,
        [table],
    ).fetchall()
    return [r[0] for r in rows]


def pick_default_run(con) -> str | None:
    """Most recent run that has at least one simulation result with a permeability.
    Falls back to most recent run overall, then any run."""
    # has simulation results with a non-null permeability?
    for order_col in ("started_at", "rowid"):
        try:
            row = con.execute(
                f"""
                SELECT r.run_id
                FROM runs r
                WHERE EXISTS (
                    SELECT 1 FROM simulation_results s
                    WHERE s.run_id = r.run_id AND s.k_z IS NOT NULL
                )
                ORDER BY r.{order_col} DESC
                LIMIT 1
                """
            ).fetchone()
            if row:
                return row[0]
        except Exception:
            continue
    # fallback: most recent run overall
    for order_col in ("started_at", "rowid"):
        try:
            row = con.execute(
                f"SELECT run_id FROM runs ORDER BY {order_col} DESC LIMIT 1"
            ).fetchone()
            if row:
                return row[0]
        except Exception:
            continue
    return None


def fetch_table_for_run(con, table: str, run_id: str):
    """Return a DataFrame for `table`. If the table has a run_id column, filter to
    this run; otherwise return None (skip reference tables to keep the file scoped)."""
    cols = table_columns(con, table)
    has_run = "run_id" in cols
    if not has_run:
        return None, has_run
    df = con.execute(
        f'SELECT * FROM "{table}" WHERE run_id = ?', [run_id]
    ).df()
    return df, has_run


def build_summary_kr(con, run_id: str, connectivity: str | None):
    """Rebuild the relative-permeability curve table from simulation_results.

    K_abs comes from sim_type='absolute' (per track, connectivity). kr is computed
    as the effective permeability divided by that track's K_abs. Sw is taken from
    fixed_boxes.sw_local. Returns a tidy per-(track, scan) DataFrame.
    """
    cols = table_columns(con, "simulation_results")
    if not {"k_z", "sim_type", "track_id", "scan_index"} <= set(cols):
        return pd.DataFrame()

    conn_clause = "AND connectivity = ?" if connectivity else ""
    params_abs = [run_id] + ([connectivity] if connectivity else [])

    # absolute permeability per track (one value per track; take the latest/any)
    abs_rows = con.execute(
        f"""
        SELECT track_id, AVG(k_z) AS k_abs_m2
        FROM simulation_results
        WHERE run_id = ? {conn_clause}
          AND lower(sim_type) = 'absolute' AND k_z IS NOT NULL
        GROUP BY track_id
        """,
        params_abs,
    ).fetchall()
    k_abs = {int(t): float(k) for t, k in abs_rows}

    # gas + brine effective permeability per (track, scan)
    # sim_type names vary ('gas'/'water' in the batch script); match both 'water'/'brine'.
    eff_rows = con.execute(
        f"""
        SELECT track_id, scan_index, lower(sim_type) AS st, AVG(k_z) AS k_m2
        FROM simulation_results
        WHERE run_id = ? {conn_clause}
          AND lower(sim_type) IN ('gas','water','brine') AND k_z IS NOT NULL
        GROUP BY track_id, scan_index, lower(sim_type)
        """,
        params_abs,
    ).fetchall()

    # Sw per (track, scan) from fixed_boxes if available
    sw_map = {}
    if "sw_local" in table_columns(con, "fixed_boxes"):
        sw_clause = "AND connectivity = ?" if connectivity else ""
        sw_params = [run_id] + ([connectivity] if connectivity else [])
        for t, s, sw in con.execute(
            f"""
            SELECT track_id, scan_index, sw_local
            FROM fixed_boxes
            WHERE run_id = ? {sw_clause} AND sw_local IS NOT NULL
            """,
            sw_params,
        ).fetchall():
            sw_map[(int(t), int(s))] = float(sw)

    # assemble
    rec: dict[tuple, dict] = {}
    for t, s, st, k in eff_rows:
        key = (int(t), int(s))
        d = rec.setdefault(key, {"track": int(t), "scan": int(s)})
        if st == "gas":
            d["K_gas_m2"] = float(k)
        else:  # water or brine
            d["K_brine_m2"] = float(k)

    out = []
    for (t, s), d in sorted(rec.items()):
        ka = k_abs.get(t)
        kg = d.get("K_gas_m2")
        kw = d.get("K_brine_m2")
        row = {
            "track": t,
            "scan": s,
            "Sw_local": sw_map.get((t, s)),
            "K_abs_mD": ka * M2_TO_MDARCY if ka else None,
            "K_gas_mD": kg * M2_TO_MDARCY if kg else None,
            "K_brine_mD": kw * M2_TO_MDARCY if kw else None,
            "kr_gas": (kg / ka) if (ka and kg is not None) else None,
            "kr_brine": (kw / ka) if (ka and kw is not None) else None,
        }
        out.append(row)
    return pd.DataFrame(out)


def build_summary_kabs(con, run_id: str, connectivity: str | None):
    """Per-track absolute permeability in m^2, mD, and Darcy."""
    cols = table_columns(con, "simulation_results")
    if "k_z" not in cols:
        return pd.DataFrame()
    conn_clause = "AND connectivity = ?" if connectivity else ""
    params = [run_id] + ([connectivity] if connectivity else [])
    rows = con.execute(
        f"""
        SELECT track_id, AVG(k_z) AS k_abs_m2
        FROM simulation_results
        WHERE run_id = ? {conn_clause}
          AND lower(sim_type) = 'absolute' AND k_z IS NOT NULL
        GROUP BY track_id
        ORDER BY track_id
        """,
        params,
    ).fetchall()
    out = []
    for t, k in rows:
        out.append({
            "track": int(t),
            "K_abs_m2": float(k),
            "K_abs_mD": float(k) * M2_TO_MDARCY,
            "K_abs_D": float(k) * M2_TO_DARCY,
        })
    return pd.DataFrame(out)


def build_run_overview(con, run_id: str, table_sheets: dict, connectivity: str | None):
    """Single-row sheet of run-level facts and counts."""
    facts = {"run_id": run_id, "connectivity_filter": connectivity or "(all)"}
    # pull run row fields if a runs table exists
    if "runs" in list_tables(con):
        try:
            rdf = con.execute(
                "SELECT * FROM runs WHERE run_id = ?", [run_id]
            ).df()
            if len(rdf):
                for c in rdf.columns:
                    facts[f"runs.{c}"] = rdf.iloc[0][c]
        except Exception:
            pass
    # counts from the per-run sheets we exported
    for name, df in table_sheets.items():
        facts[f"n_rows.{name}"] = len(df)
    # a few common derived counts (guarded)
    try:
        facts["n_scans"] = con.execute(
            "SELECT COUNT(*) FROM scans WHERE run_id = ?", [run_id]).fetchone()[0]
    except Exception:
        pass
    try:
        facts["n_sim_results"] = con.execute(
            "SELECT COUNT(*) FROM simulation_results WHERE run_id = ? AND k_z IS NOT NULL",
            [run_id]).fetchone()[0]
    except Exception:
        pass
    # one row -> transpose to (field, value) for readability
    return pd.DataFrame({"field": list(facts.keys()), "value": list(facts.values())})


def main():
    ap = argparse.ArgumentParser(description="Export one run's DB tables to Excel.")
    ap.add_argument("--db", default="results.duckdb", help="path to results.duckdb")
    ap.add_argument("--out", default=None, help="output .xlsx (default: run_<id>.xlsx)")
    ap.add_argument("--run-id", default=None,
                    help="run to export (default: latest run with simulation results)")
    ap.add_argument("--connectivity", default=None,
                    help="connectivity for the COMPUTED sheets only, e.g. 18N "
                         "(raw table sheets are never filtered by connectivity)")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        sys.exit(f"ERROR: database not found at {db_path.resolve()}")

    con = duckdb.connect(str(db_path), read_only=True)

    run_id = args.run_id or pick_default_run(con)
    if run_id is None:
        sys.exit("ERROR: no runs found in the database.")
    print(f"Exporting run: {run_id}")
    if args.connectivity:
        print(f"  computed sheets filtered to connectivity = {args.connectivity}")

    out_path = Path(args.out) if args.out else Path(f"run_{run_id}.xlsx")

    # 1) every table that has a run_id, filtered to this run
    table_sheets: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for tbl in list_tables(con):
        df, has_run = fetch_table_for_run(con, tbl, run_id)
        if df is None:
            skipped.append(tbl)
            continue
        table_sheets[tbl] = df

    # 2) computed summary sheets
    summary_kr = build_summary_kr(con, run_id, args.connectivity)
    summary_kabs = build_summary_kabs(con, run_id, args.connectivity)
    overview = build_run_overview(con, run_id, table_sheets, args.connectivity)

    con.close()

    # 3) write the workbook: overview first, then computed sheets, then raw tables
    used: set[str] = set()
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        overview.to_excel(xw, sheet_name=safe_sheet_name("run_overview", used), index=False)
        if len(summary_kr):
            summary_kr.to_excel(xw, sheet_name=safe_sheet_name("summary_kr", used), index=False)
        if len(summary_kabs):
            summary_kabs.to_excel(xw, sheet_name=safe_sheet_name("summary_kabs", used), index=False)
        for tbl, df in table_sheets.items():
            sheet = safe_sheet_name(tbl, used)
            # openpyxl cannot write an empty frame with no columns; guard it
            if df.shape[1] == 0:
                df = pd.DataFrame({"(no columns)": []})
            df.to_excel(xw, sheet_name=sheet, index=False)

    # 4) report
    print(f"\nWrote {out_path.resolve()}")
    print(f"  run_overview        : {len(overview)} fields")
    print(f"  summary_kr          : {len(summary_kr)} rows")
    print(f"  summary_kabs        : {len(summary_kabs)} rows")
    print(f"  raw table sheets    : {len(table_sheets)}")
    for t, df in table_sheets.items():
        print(f"     - {t:24s} {len(df):>6} rows  x {df.shape[1]} cols")
    if skipped:
        print(f"  skipped (no run_id) : {', '.join(skipped)}")


if __name__ == "__main__":
    main()
