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

    # Pull the full cluster-mask (saved at X) for each track, if present.
    con2 = duckdb.connect(str(db_path), read_only=True)
    try:
        cm_rows = con2.execute(
            """
            SELECT track_id, clustermask_raw, clustermask_z0, clustermask_z1,
                   crop_y0, crop_x0, extent_y, extent_x
            FROM cluster_properties
            WHERE run_id = ? AND connectivity = ?
              AND clustermask_raw IS NOT NULL
            """,
            (run_id, connectivity),
        ).fetchall()
        for tid, cm_path, cm_z0, cm_z1, cy0, cx0, cey, cex in cm_rows:
            if tid in clusters and cm_path:
                clusters[tid]["clustermask"] = {
                    "path": Path(cm_path),
                    "z0": int(cm_z0), "z1": int(cm_z1),
                    "y0": int(cy0), "x0": int(cx0),
                    "shape": (int(cm_z1) - int(cm_z0) + 1, int(cey), int(cex)),
                }
    except Exception as e:
        print(f"  (no cluster-mask data: {e})")
    con2.close()

    print(f"  {len(clusters)} tracks  |  {len(scan_indices)} scans  |  timestep X = {x_scan_index}")
    return run_id, clusters, sorted(scan_indices), x_scan_index


# ── Mesh building ────────────────────────────────────────────────────────────

def _mesh_from_bool(gas_bool, downsample, origin):
    """Marching-cubes mesh from a boolean gas array (True=gas), placed at origin."""
    d = downsample
    gas_ds = gas_bool[::d, ::d, ::d].astype(np.float32)
    if not gas_ds.any():
        return None
    padded = np.pad(gas_ds, 1, constant_values=0)
    verts, faces, _, _ = marching_cubes(
        padded, level=0.5, spacing=(float(d), float(d), float(d)),
        allow_degenerate=False,
    )
    verts -= float(d)
    verts += np.array([float(origin[0]), float(origin[1]), float(origin[2])])
    n = len(faces)
    mesh = pv.PolyData(verts, np.hstack([np.full((n, 1), 3), faces]).ravel())
    return mesh.smooth(n_iter=20, relaxation_factor=0.1)


def build_mesh(
    gas_path: Path,
    shape: tuple[int, int, int],
    origin: tuple[int, int, int],
    downsample: int,
) -> "tuple[pv.PolyData | None, bool]":
    """Load a gas domain (.raw, 0=gas/1=solid), isolate the PERCOLATING cluster
    (largest connected component spanning both Z-faces), and mesh it.

    Spanning is detected at FULL resolution (downsampling first would sever the
    thin thread); only the isolated cluster is downsampled for meshing.
    Returns (mesh, percolates). If nothing spans, percolates=False and the
    largest component is shown as a fallback.
    """
    try:
        import cc3d
    except ImportError:
        cc3d = None
    try:
        raw = np.fromfile(str(gas_path), dtype=np.uint8)
        if raw.size != int(np.prod(shape)):
            print(f"  WARNING: size mismatch for {gas_path.name} — skipping")
            return None, False
        vol = raw.reshape(shape)
        gas_full = (vol == 0)
        if not gas_full.any():
            return None, False

        percolates = True
        if cc3d is not None:
            labels = cc3d.connected_components(gas_full.astype(np.uint8), connectivity=26)
            if int(labels.max()) > 0:
                inlet = set(np.unique(labels[0])) - {0}
                outlet = set(np.unique(labels[-1])) - {0}
                spanning = inlet & outlet
                if spanning:
                    chosen = max(spanning, key=lambda c: int((labels == c).sum()))
                    percolates = True
                else:
                    counts = np.bincount(labels.ravel()); counts[0] = 0
                    chosen = int(counts.argmax()); percolates = False
                gas_full = (labels == chosen)

        return _mesh_from_bool(gas_full, downsample, origin), percolates
    except Exception as e:
        print(f"  WARNING: mesh error for {gas_path.name}: {e}")
        return None, False


def build_clustermask_mesh(cm: dict, downsample: int) -> "pv.PolyData | None":
    """Mesh the full tracked cluster mask saved at X (0/1 .raw, 1=cluster).
    Placed at its own (z0, y0, x0) origin so the box wireframe sits inside it."""
    try:
        shape = cm["shape"]
        raw = np.fromfile(str(cm["path"]), dtype=np.uint8)
        if raw.size != int(np.prod(shape)):
            print(f"  WARNING: clustermask size mismatch for {cm['path'].name}")
            return None
        mask = raw.reshape(shape) > 0
        origin = (cm["z0"], cm["y0"], cm["x0"])
        return _mesh_from_bool(mask, downsample, origin)
    except Exception as e:
        print(f"  WARNING: clustermask mesh error: {e}")
        return None


# ── Main viewer ──────────────────────────────────────────────────────────────

def launch_viewer(
    clusters: dict,
    scan_indices: list[int],
    x_scan_index: int,
    downsample: int,
) -> None:
    track_ids = sorted(clusters.keys())
    if not track_ids or not scan_indices:
        print("Nothing to display.")
        return

    pv.set_plot_theme("dark")
    pl = pv.Plotter(window_size=(1400, 900), title="µCT Pipeline — Cluster Viewer")

    state = {
        "track_i": 0,
        "scan_i": scan_indices.index(x_scan_index) if x_scan_index in scan_indices else 0,
    }
    ISO, BOX, TITLE, INFO = "iso", "box", "title_text", "info_text"

    def _box_wireframe(origin, shape):
        z0, y0, x0 = origin
        nz, ny, nx = shape
        return pv.Box(bounds=(float(z0), float(z0 + nz), float(y0), float(y0 + ny),
                              float(x0), float(x0 + nx)))

    def _draw(recenter: bool = True):
        tid = track_ids[state["track_i"]]
        scan_idx = scan_indices[state["scan_i"]]
        info = clusters[tid]
        shape = info["shape"]
        origin = info["origin"]
        is_x = (scan_idx == x_scan_index)

        pl.remove_actor(ISO)
        percolates = True
        whole = False

        # At timestep X, show the WHOLE tracked cluster (saved mask) if available;
        # otherwise show the in-box spanning percolating cluster.
        if is_x and "clustermask" in info:
            mesh = build_clustermask_mesh(info["clustermask"], downsample)
            whole = True
            if mesh is not None:
                pl.add_mesh(mesh, color=_colour(tid), opacity=0.85,
                            smooth_shading=True, name=ISO)
        else:
            gas_path = info["scan_gas_paths"].get(scan_idx)
            if gas_path is not None and gas_path.exists():
                mesh, percolates = build_mesh(gas_path, shape, origin, downsample)
                if mesh is not None:
                    pl.add_mesh(mesh, color=_colour(tid) if percolates else "#d32f2f",
                                opacity=0.85, smooth_shading=True, name=ISO)

        # Box wireframe always at the cropped extraction extent (so at X you see
        # the whole cluster with the box drawn inside it).
        box = _box_wireframe(origin, shape)
        pl.add_mesh(box, color=_colour(tid), style="wireframe", line_width=2,
                    opacity=0.6, name=BOX)

        pl.remove_actor(TITLE)
        if whole:
            status = "   [whole cluster @ X]"; tcol = _colour(tid)
        elif percolates:
            status = ""; tcol = _colour(tid)
        else:
            status = "   [NON-PERCOLATING]"; tcol = "#d32f2f"
        pl.add_text(f"Track {tid:02d}{status}", position="upper_left", font_size=12,
                    color=tcol, font="courier", name=TITLE)

        pl.remove_actor(INFO)
        marker = "  <- timestep X" if is_x else ""
        pl.add_text(
            f"scan {scan_idx}{marker}   ({state['scan_i']+1}/{len(scan_indices)})\n"
            f"downsample {downsample}x",
            position="upper_right", font_size=10, color="#e0e0e0",
            font="courier", name=INFO,
        )

        if recenter:
            pl.reset_camera(bounds=box.bounds)
        pl.render()

    def on_scan(value):
        i = max(0, min(int(round(value)), len(scan_indices) - 1))
        if i != state["scan_i"]:
            state["scan_i"] = i
            _draw(recenter=False)

    def on_track(value):
        i = max(0, min(int(round(value)), len(track_ids) - 1))
        if i != state["track_i"]:
            state["track_i"] = i
            _draw(recenter=True)

    print("\nBuilding viewer...")
    _draw(recenter=False)
    pl.camera_position = "iso"
    pl.reset_camera()
    pl.add_axes()

    if len(scan_indices) > 1:
        n = len(scan_indices)
        pl.add_slider_widget(
            callback=on_scan, rng=[0, n - 1], value=state["scan_i"],
            title=f"Scan  (0=earliest -> {n-1}=timestep X)",
            pointa=(0.20, 0.08), pointb=(0.80, 0.08), style="modern",
        )
    if len(track_ids) > 1:
        pl.add_slider_widget(
            callback=on_track, rng=[0, len(track_ids) - 1], value=0,
            title="Track  (slide to switch cluster)",
            pointa=(0.20, 0.16), pointb=(0.80, 0.16), style="modern",
        )

    print("Close the window to exit.")
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