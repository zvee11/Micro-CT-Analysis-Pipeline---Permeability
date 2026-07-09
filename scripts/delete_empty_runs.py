"""delete_empty_runs.py - delete runs that have NO simulation results.

Auto-discovers every table with a run_id column and deletes children before the
parent, committing after each table. (DuckDB enforces foreign keys in a way that
rejects a single wrapping transaction even when children are deleted first, so we
commit per table instead.) The DB is backed up first, so a mid-run failure is
recoverable by restoring the backup.

Run from the THESIS folder:  python delete_empty_runs.py
"""
import duckdb, shutil, sys
from datetime import datetime

DB = "results.duckdb"

# 1) back up (this is the safety net - no wrapping transaction to roll back)
backup = f"results_backup_{datetime.now():%Y%m%d_%H%M%S}.duckdb"
shutil.copy(DB, backup)
print(f"backup written: {backup}")
print(f"(if anything goes wrong, restore with:  copy {backup} {DB})\n")

con = duckdb.connect(DB)

# 2) discover every table with a run_id column
tabs = [r[0] for r in con.execute("""
    SELECT table_name FROM information_schema.columns
    WHERE column_name = 'run_id' GROUP BY table_name
""").fetchall()]
children = [t for t in tabs if t.lower() != "runs"]
print("tables referencing run_id:", ", ".join(sorted(tabs)))

# 3) runs with zero simulation_results
empty = [r[0] for r in con.execute("""
    SELECT r.run_id FROM runs r
    WHERE NOT EXISTS (SELECT 1 FROM simulation_results s WHERE s.run_id = r.run_id)
    ORDER BY r.started_at DESC
""").fetchall()]
if not empty:
    print("\nNo runs without simulations. Nothing to delete.")
    con.close(); sys.exit(0)

print("\nRuns to DELETE (no simulation_results):")
for rid in empty:
    print(f"  {rid}")
print("\nKEEPING (have simulations):")
for rid in con.execute(
        "SELECT DISTINCT run_id FROM simulation_results ORDER BY run_id DESC").fetchall():
    print(f"  {rid[0]}")

resp = input(f"\nDelete these {len(empty)} runs? Type 'yes' to confirm: ").strip().lower()
if resp != "yes":
    print("aborted - nothing deleted.")
    con.close(); sys.exit(0)

# 4) delete children first (commit per table), then the runs row
ph = ",".join("?" * len(empty))
try:
    for table in children:
        con.execute(f"DELETE FROM {table} WHERE run_id IN ({ph})", empty)
        con.commit()
        print(f"  cleared {table}")
    con.execute(f"DELETE FROM runs WHERE run_id IN ({ph})", empty)
    con.commit()
    print(f"\ndeleted {len(empty)} runs.")
except Exception as e:
    print(f"\nERROR during delete: {e}")
    print(f"DB may be partially modified - restore the backup if needed:")
    print(f"  copy {backup} {DB}")
    con.close(); sys.exit(1)

print("\nremaining runs:")
for r in con.execute("SELECT run_id FROM runs ORDER BY started_at DESC").fetchall():
    print(f"  {r[0]}")
con.close()