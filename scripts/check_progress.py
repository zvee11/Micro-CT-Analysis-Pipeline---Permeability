"""
check_progress.py  —  read-only progress check for the GeoDict batch.
Run any time in another terminal; it does not disturb the running batch.

    python check_progress.py
"""
import duckdb, os

DB = r"C:\Users\99619\Desktop\SVETA\Micro-CT-Analysis-Pipeline\results.duckdb"
OUT = r"C:\Users\99619\Desktop\SVETA\Micro-CT-Analysis-Pipeline\output"

# 1) DB rows written so far (GeoDict)
con = duckdb.connect(DB, read_only=True)
rows = con.execute(
    "SELECT sim_type, COUNT(*) , COUNT(k_z) "
    "FROM simulation_results WHERE simulator='GeoDict' GROUP BY sim_type"
).fetchall()
print("=== DB rows (simulator='GeoDict') ===")
total = 0
for sim_type, n, n_kz in rows:
    print(f"  {sim_type:>8}: {n} rows, {n_kz} with a permeability")
    total += n
print(f"  total: {total} rows\n")

# a few most-recent values as a sanity check
recent = con.execute(
    "SELECT scan_index, track_id, connectivity, sim_type, k_z "
    "FROM simulation_results WHERE simulator='GeoDict' AND k_z IS NOT NULL "
    "ORDER BY rowid DESC LIMIT 8"
).fetchall()
print("=== most recent results ===")
for s, t, c, st, k in recent:
    print(f"  scan {s} track {t} {c} {st:>8}: K_z = {k:.4e} m^2")
con.close()

# 2) result_summary.json files on disk
n_json = 0
n_with_perm = 0
for root, _, files in os.walk(OUT):
    if "result_summary.json" in files:
        n_json += 1
        try:
            import json
            d = json.load(open(os.path.join(root, "result_summary.json")))
            if d.get("permeability_z_m2") is not None:
                n_with_perm += 1
        except Exception:
            pass
print(f"\n=== files on disk ===")
print(f"  result_summary.json found: {n_json}  ({n_with_perm} with a permeability)")
