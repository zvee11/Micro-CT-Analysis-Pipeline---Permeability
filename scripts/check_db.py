"""check_db.py - full inventory of every run in results.duckdb.
Run from the THESIS folder:  python check_db.py
"""
import duckdb

con = duckdb.connect("results.duckdb", read_only=True)

def count(table, run_id, extra=""):
    try:
        q = f"SELECT COUNT(*) FROM {table} WHERE run_id=?" + (f" AND {extra}" if extra else "")
        return con.execute(q, [run_id]).fetchone()[0]
    except Exception:
        return "-"

runs = con.execute(
    "SELECT run_id, started_at, crop_mode, connectivity, regime_cutoff "
    "FROM runs ORDER BY started_at DESC"
).fetchall()

print(f"{'run_id':<30}{'date':<12}{'scans':>6}{'qual':>5}{'boxes':>6}"
      f"{'clust':>6}{'sims':>6}{'gd':>5}{'mplbm':>6}")
print("-" * 88)

for (rid, started, crop, conn, regime) in runs:
    scans   = count("scans", rid)
    qual    = count("scans", rid, "qualifying=TRUE")
    boxes   = count("fixed_boxes", rid)
    clust   = count("cluster_properties", rid)
    sims    = count("simulation_results", rid)
    gd      = count("simulation_results", rid, "simulator='GeoDict'")
    mplbm   = count("simulation_results", rid, "simulator='MPLBM'")
    date = str(started)[:10]
    print(f"{rid:<30}{date:<12}{scans:>6}{qual:>5}{boxes:>6}"
          f"{clust:>6}{sims:>6}{gd:>5}{mplbm:>6}")

print("-" * 88)
print("scans=timesteps  qual=qualifying  boxes=fixed_boxes  clust=cluster_properties")
print("sims=all simulation_results  gd=GeoDict rows  mplbm=MPLBM rows")
print("\nA 'full' run has nonzero scans/boxes/clusters; sims>0 means it was simulated.")

con.close()