"""
connectivity_charts.py

Dash dashboard comparing two pipeline runs of different connectivity (26N vs
18N) from results.duckdb. Clusters are matched ACROSS connectivities by physical
location at timestep X (track_ids are per-run, so they can't be compared
directly). A dropdown selects a matched pair; charts show the per-timestep
evolution of both connectivities overlaid.

No volumes are loaded — this is pure DB -> charts, runs in the browser.

    pip install dash plotly duckdb
    python connectivity_charts.py
    # open http://127.0.0.1:8050

    python connectivity_charts.py --a 26N --b 18N --tol 80
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import duckdb

try:
    import dash
    from dash import dcc, html, Input, Output
    import plotly.graph_objects as go
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\nInstall:  pip install dash plotly duckdb")


# ── DB helpers ───────────────────────────────────────────────────────────────

def latest_run_for(con, connectivity):
    row = con.execute(
        """SELECT fb.run_id FROM fixed_boxes fb JOIN runs r ON fb.run_id = r.run_id
           WHERE fb.connectivity = ? ORDER BY r.started_at DESC LIMIT 1""",
        [connectivity]).fetchone()
    return row[0] if row else None


def x_clusters(con, run_id, conn):
    rows = con.execute(
        """SELECT track_id, final_voxels, cog_z, cog_y, cog_x, status
           FROM tracks WHERE run_id = ? AND connectivity = ?
           ORDER BY final_voxels DESC""", [run_id, conn]).fetchall()
    return [{"track_id": t, "voxels": fv or 0, "cog": (cz, cy, cx), "status": st}
            for t, fv, cz, cy, cx, st in rows]


def match_clusters(a_list, b_list, tol):
    used = set(); matches = []
    for a in a_list:
        best = None; best_d = None; best_j = None
        for j, b in enumerate(b_list):
            if j in used or None in a["cog"] or None in b["cog"]:
                continue
            d = math.dist(a["cog"], b["cog"])
            if best_d is None or d < best_d:
                best, best_d, best_j = b, d, j
        if best is not None and best_d <= tol:
            used.add(best_j); matches.append((a, best, best_d))
        else:
            matches.append((a, None, None))
    b_only = [b for j, b in enumerate(b_list) if j not in used]
    return matches, b_only


def evolution(con, run_id, conn, track_id):
    rows = con.execute(
        """SELECT scan_index, gas_voxels, percolates, spanning_count, cluster_voxels
           FROM fixed_boxes WHERE run_id = ? AND connectivity = ? AND track_id = ?
           ORDER BY scan_index""", [run_id, conn, track_id]).fetchall()
    return {
        "scan": [r[0] for r in rows],
        "gas": [r[1] for r in rows],
        "perc": [r[2] for r in rows],
        "span": [r[3] for r in rows],
        "cluster": [r[4] for r in rows],
    }


# ── App ──────────────────────────────────────────────────────────────────────

def build_app(db_path, conn_a, conn_b, tol):
    con = duckdb.connect(db_path, read_only=True)
    run_a = latest_run_for(con, conn_a)
    run_b = latest_run_for(con, conn_b)
    if not run_a or not run_b:
        sys.exit(f"Could not find runs for {conn_a} ({run_a}) / {conn_b} ({run_b})")

    a_list = x_clusters(con, run_a, conn_a)
    b_list = x_clusters(con, run_b, conn_b)
    matches, b_only = match_clusters(a_list, b_list, tol)

    # Build selectable options: matched pairs + unmatched on either side.
    options = []
    pair_lookup = {}
    for i, (a, b, d) in enumerate(matches):
        if b is not None:
            label = (f"MATCH  {conn_a} t{a['track_id']:02d} <-> {conn_b} t{b['track_id']:02d}"
                     f"  (d={d:.0f})")
            pair_lookup[f"m{i}"] = ("pair", a["track_id"], b["track_id"])
        else:
            label = f"{conn_a}-only  t{a['track_id']:02d}  (no {conn_b} match)"
            pair_lookup[f"m{i}"] = ("a_only", a["track_id"], None)
        options.append({"label": label, "value": f"m{i}"})
    for j, b in enumerate(b_only):
        label = f"{conn_b}-only  t{b['track_id']:02d}  (no {conn_a} match)"
        pair_lookup[f"b{j}"] = ("b_only", None, b["track_id"])
        options.append({"label": label, "value": f"b{j}"})

    n_pairs = sum(1 for _, b, _ in matches if b is not None)

    COL_A, COL_B = "#4F8DFD", "#FF6B6B"   # 26N blue, 18N red

    app = dash.Dash(__name__)
    app.layout = html.Div(
        style={"font-family": "monospace", "background": "#111", "color": "#eee",
               "padding": "16px", "min-height": "100vh"},
        children=[
            html.H2(f"Connectivity comparison  —  {conn_a} (blue)  vs  {conn_b} (red)"),
            html.Div(f"{conn_a} run: {run_a}   |   {conn_b} run: {run_b}   |   "
                     f"{n_pairs} matched, {len(matches)-n_pairs} {conn_a}-only, "
                     f"{len(b_only)} {conn_b}-only   |   tol={tol}",
                     style={"color": "#999", "margin-bottom": "12px"}),
            dcc.Dropdown(id="pair", options=options,
                         value=options[0]["value"] if options else None,
                         clearable=False,
                         style={"color": "#000", "max-width": "640px"}),
            html.Div(id="summary", style={"margin": "10px 0", "color": "#bbb"}),
            dcc.Graph(id="gas_chart"),
            dcc.Graph(id="cluster_chart"),
            dcc.Graph(id="span_chart"),
        ],
    )

    def _trace(evo, name, colour, key):
        return go.Scatter(
            x=evo["scan"], y=evo[key], mode="lines+markers", name=name,
            line=dict(color=colour, width=2), marker=dict(size=7),
        )

    def _layout(title, ytitle):
        return dict(
            title=title, template="plotly_dark",
            xaxis_title="scan index (0 = earliest, last = timestep X)",
            yaxis_title=ytitle, height=330,
            margin=dict(l=60, r=20, t=40, b=40),
        )

    @app.callback(
        Output("gas_chart", "figure"),
        Output("cluster_chart", "figure"),
        Output("span_chart", "figure"),
        Output("summary", "children"),
        Input("pair", "value"),
    )
    def update(sel):
        kind, a_tid, b_tid = pair_lookup[sel]
        evo_a = evolution(con, run_a, conn_a, a_tid) if a_tid is not None else None
        evo_b = evolution(con, run_b, conn_b, b_tid) if b_tid is not None else None

        gas = go.Figure(); cluster = go.Figure(); span = go.Figure()
        if evo_a:
            gas.add_trace(_trace(evo_a, f"{conn_a} t{a_tid:02d}", COL_A, "gas"))
            cluster.add_trace(_trace(evo_a, f"{conn_a} cluster", COL_A, "cluster"))
            span.add_trace(_trace(evo_a, f"{conn_a} spanning#", COL_A, "span"))
        if evo_b:
            gas.add_trace(_trace(evo_b, f"{conn_b} t{b_tid:02d}", COL_B, "gas"))
            cluster.add_trace(_trace(evo_b, f"{conn_b} cluster", COL_B, "cluster"))
            span.add_trace(_trace(evo_b, f"{conn_b} spanning#", COL_B, "span"))
        gas.update_layout(**_layout("Total gas voxels in box", "gas voxels"))
        cluster.update_layout(**_layout("Percolating cluster voxels (spanning component)", "cluster voxels"))
        span.update_layout(**_layout("Spanning component count (>1 = cluster split)", "# spanning"))

        # Text summary
        parts = []
        if evo_a and evo_b and evo_a["gas"] and evo_b["gas"]:
            ga, gb = evo_a["gas"][-1] or 0, evo_b["gas"][-1] or 0
            diff = ga - gb
            rel = (100*diff/gb) if gb else float("nan")
            parts.append(f"At timestep X: {conn_a} gas={ga:,}  {conn_b} gas={gb:,}  "
                         f"diff={diff:+,} ({rel:+.1f}%)")
            # percolation agreement
            pa = evo_a["perc"]; pb = evo_b["perc"]
            n = min(len(pa), len(pb))
            disagree = sum(1 for i in range(n) if bool(pa[i]) != bool(pb[i]))
            parts.append(f"Percolation disagreement: {disagree}/{n} timesteps")
        elif kind == "a_only":
            parts.append(f"This cluster exists only in {conn_a} — no spatial match in {conn_b}.")
        elif kind == "b_only":
            parts.append(f"This cluster exists only in {conn_b} — no spatial match in {conn_a}.")
        return gas, cluster, span, "  |  ".join(parts)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="results.duckdb")
    ap.add_argument("--a", default="26N")
    ap.add_argument("--b", default="18N")
    ap.add_argument("--tol", type=float, default=80.0)
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()
    if not Path(args.db).exists():
        sys.exit(f"ERROR: {args.db} not found")
    app = build_app(args.db, args.a, args.b, args.tol)
    print(f"Open http://127.0.0.1:{args.port}")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
