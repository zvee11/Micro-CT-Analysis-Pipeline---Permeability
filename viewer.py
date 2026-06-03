"""
view_clusters.py

Standalone 3D viewer for pipeline output clusters.
Run from the project root (same folder as results.duckdb):

    python view_clusters.py                          # latest run, all scans
    python view_clusters.py --run 20260520_121143_S20200864
    python view_clusters.py --connectivity 26N
    python view_clusters.py --downsample 4           # finer mesh (slower)

The viewer shows all tracked clusters across all qualifying scans.
Use the slider to scrub between scans, checkboxes to toggle individual tracks.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np

try:
    import pyvista as pv
    from skimage.measure import marching_cubes
except ImportError:
    print("ERROR: pip install pyvista scikit-image")
    sys.exit(1)


# ── Colour palette — one per track ──────────────────────────────────────────

TRACK_COLOURS = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#E91E63", "#8BC34A",
    "#FF5722", "#607D8B",
]

def _colour(track_id: int) -> str:
    return TRACK_COLOURS[(track_id - 1) % len(TRACK_COLOURS)]


# ── Database query ───────────────────────────────────────────────────────────

def load_from_db(
    db_path: Path,
    run_id: str | None,
    connectivity: str,
) -> tuple[str, dict, list[int]]:
    """
    Returns:
        run_id          — the run that was loaded
        clusters        — {track_id: {"origin": (z0,y0,x0), "shape": (Z,Y,X),
                                      "scan_gas_paths": {scan_index: Path},
                                      "is_X": bool, "x_scan_index": int}}
        scan_indices    — sorted list of all qualifying scan indices
    """
    con = duckdb.connect(str(db_path), read_only=True)

    # Pick run_id
    if run_id is None:
        row = con.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            print("ERROR: no runs found in database")
            sys.exit(1)
        run_id = row[0]
    print(f"Loading run: {run_id}  connectivity: {connectivity}")

    # Fetch all fixed_box rows for this run + connectivity
    rows = con.execute(
        """
        SELECT
            fb.scan_index,
            fb.track_id,
            fb.z0, fb.y0, fb.x0,
            fb.extent_z, fb.extent_y, fb.extent_x,
            fb.domain_gas,
            s.is_X
        FROM fixed_boxes fb
        JOIN scans s
          ON s.run_id = fb.run_id AND s.scan_index = fb.scan_index
        WHERE fb.run_id = ?
          AND fb.connectivity = ?
          AND fb.domain_gas IS NOT NULL
        ORDER BY fb.track_id, fb.scan_index
        """,
        (run_id, connectivity),
    ).fetchall()

    con.close()

    if not rows:
        print(f"ERROR: no fixed_box rows found for run={run_id} connectivity={connectivity}")
        print("Check that the pipeline completed successfully and connectivity matches config.")
        sys.exit(1)

    clusters: dict[int, dict] = {}
    scan_indices: set[int] = set()
    x_scan_index: int = -1

    for scan_idx, track_id, z0, y0, x0, ez, ey, ex, gas_path_str, is_x in rows:
        gas_path = Path(gas_path_str)
        shape    = (int(ez), int(ey), int(ex))
        origin   = (int(z0), int(y0), int(x0))

        if track_id not in clusters:
            clusters[track_id] = {
                "origin":         origin,
                "shape":          shape,
                "scan_gas_paths": {},
                "x_scan_index":   -1,
            }

        clusters[track_id]["scan_gas_paths"][scan_idx] = gas_path
        scan_indices.add(scan_idx)

        if is_x:
            clusters[track_id]["x_scan_index"] = scan_idx
            x_scan_index = scan_idx

    print(f"  {len(clusters)} tracks  |  {len(scan_indices)} scans  |  timestep X = {x_scan_index}")
    return run_id, clusters, sorted(scan_indices), x_scan_index


# ── Mesh building ────────────────────────────────────────────────────────────

def build_mesh(
    gas_path: Path,
    shape: tuple[int, int, int],
    origin: tuple[int, int, int],
    downsample: int,
) -> "pv.PolyData | None":
    """Load .raw domain, run marching cubes, translate to full-volume position."""
    try:
        raw = np.fromfile(str(gas_path), dtype=np.uint8)
        if raw.size != int(np.prod(shape)):
            print(f"  WARNING: size mismatch for {gas_path.name} — skipping")
            return None

        vol    = raw.reshape(shape)
        d      = downsample
        vol_ds = vol[::d, ::d, ::d]

        # domain: 0=gas, 1=solid — invert for marching cubes
        gas_vol = (1 - vol_ds).astype(np.float32)
        if not gas_vol.any():
            return None

        padded = np.pad(gas_vol, 1, constant_values=0)
        verts, faces, _, _ = marching_cubes(
            padded, level=0.5,
            spacing=(float(d), float(d), float(d)),
            allow_degenerate=False,
        )

        # Undo padding offset, then translate to full-volume coordinates
        verts -= float(d)
        verts += np.array([float(origin[0]), float(origin[1]), float(origin[2])])

        n    = len(faces)
        mesh = pv.PolyData(
            verts,
            np.hstack([np.full((n, 1), 3), faces]).ravel(),
        )
        return mesh.smooth(n_iter=20, relaxation_factor=0.1)

    except Exception as e:
        print(f"  WARNING: mesh error for {gas_path.name}: {e}")
        return None


# ── Main viewer ──────────────────────────────────────────────────────────────

def launch_viewer(
    clusters: dict,
    scan_indices: list[int],
    x_scan_index: int,
    downsample: int,
) -> None:
    track_ids = sorted(clusters.keys())

    pv.set_plot_theme("dark")
    pl = pv.Plotter(
        window_size=(1400, 900),
        title="µCT Pipeline — Cluster Viewer",
    )

    track_visible: dict[int, bool] = {tid: True for tid in track_ids}

    def _actor(tid: int) -> str:
        return f"track_{tid:02d}"

    # ── Build initial meshes at timestep X ──────────────────────────────────
    print("\nBuilding meshes...")
    for tid in track_ids:
        info     = clusters[tid]
        x_idx    = info["x_scan_index"]
        gas_path = info["scan_gas_paths"].get(x_idx)
        shape    = info["shape"]
        origin   = info["origin"]

        if gas_path is None or not gas_path.exists():
            print(f"  Track {tid:02d}: domain file not found at {gas_path}")
            continue

        mesh = build_mesh(gas_path, shape, origin, downsample)
        if mesh is not None:
            pl.add_mesh(
                mesh, color=_colour(tid), opacity=0.8,
                smooth_shading=True, name=_actor(tid),
            )
            print(f"  Track {tid:02d}: {mesh.n_points:,} verts  "
                  f"origin=({origin[0]}, {origin[1]}, {origin[2]})")

    # ── Scan info text ───────────────────────────────────────────────────────
    _TEXT = "scan_info"

    def _info_text(scan_idx: int) -> str:
        pos    = scan_indices.index(scan_idx) + 1 if scan_idx in scan_indices else "?"
        marker = "  ← timestep X" if scan_idx == x_scan_index else ""
        return (
            f"Scan index : {scan_idx}{marker}\n"
            f"Scan       : {pos} / {len(scan_indices)}\n"
            f"Downsample : {downsample}x"
        )

    pl.add_text(
        _info_text(x_scan_index),
        position="upper_left", font_size=9,
        color="#e0e0e0", font="courier",
        name=_TEXT,
    )

    # ── Slider ───────────────────────────────────────────────────────────────
    def on_slider(value):
        idx      = max(0, min(int(round(value)), len(scan_indices) - 1))
        scan_idx = scan_indices[idx]

        for tid in track_ids:
            info     = clusters[tid]
            gas_path = info["scan_gas_paths"].get(scan_idx)
            shape    = info["shape"]
            origin   = info["origin"]

            pl.remove_actor(_actor(tid))

            if gas_path is None or not gas_path.exists():
                continue

            mesh = build_mesh(gas_path, shape, origin, downsample)
            if mesh is not None and track_visible[tid]:
                pl.add_mesh(
                    mesh, color=_colour(tid), opacity=0.8,
                    smooth_shading=True, name=_actor(tid),
                )

        pl.remove_actor(_TEXT)
        pl.add_text(
            _info_text(scan_idx),
            position="upper_left", font_size=9,
            color="#e0e0e0", font="courier",
            name=_TEXT,
        )
        pl.render()

    if len(scan_indices) > 1:
        n = len(scan_indices)
        init_val = scan_indices.index(x_scan_index) if x_scan_index in scan_indices else 0
        pl.add_slider_widget(
            callback=on_slider,
            rng=[0, n - 1],
            value=init_val,
            title=f"Scan  (0=earliest → {n-1}=timestep X)",
            pointa=(0.15, 0.06), pointb=(0.85, 0.06),
            style="modern",
        )

    # ── Checkboxes ───────────────────────────────────────────────────────────
    win_w, win_h = 1400, 900
    btn_size  = 28
    btn_gap   = 40
    btn_x     = win_w - btn_size - 10
    label_x   = btn_x - 80
    btn_y_top = win_h - 60

    for btn_idx, tid in enumerate(track_ids):
        col   = _colour(tid)
        btn_y = btn_y_top - btn_idx * btn_gap

        def _make_cb(t=tid):
            def cb(state):
                track_visible[t] = bool(state)
                for name, actor in pl.renderer.actors.items():
                    if name == _actor(t):
                        actor.SetVisibility(int(state))
                pl.render()
            return cb

        pl.add_checkbox_button_widget(
            callback=_make_cb(),
            value=True,
            position=(btn_x, btn_y),
            size=btn_size,
            border_size=2,
            color_on=col,
            color_off="#444444",
        )
        pl.add_text(
            f"Track {tid:02d}",
            position=(label_x, btn_y + 4),
            font_size=8, color=col, font="courier",
        )

    pl.camera_position = "iso"
    pl.add_axes()
    print("\nClose the window to exit.")
    pl.show()


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone 3D viewer for pipeline cluster output."
    )
    parser.add_argument(
        "--db", default="results.duckdb",
        help="Path to results.duckdb (default: results.duckdb in current folder)"
    )
    parser.add_argument(
        "--run", default=None,
        help="run_id to visualise (default: most recent run)"
    )
    parser.add_argument(
        "--connectivity", default="26N",
        help="Connectivity label to visualise: 6N, 18N, or 26N (default: 26N)"
    )
    parser.add_argument(
        "--downsample", type=int, default=8,
        help="Downsampling factor for mesh resolution (default: 8, lower = finer but slower)"
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}")
        print("Run from the project root folder, or pass --db path/to/results.duckdb")
        sys.exit(1)

    run_id, clusters, scan_indices, x_scan_index = load_from_db(
        db_path, args.run, args.connectivity
    )
    launch_viewer(clusters, scan_indices, x_scan_index, args.downsample)


if __name__ == "__main__":
    main()