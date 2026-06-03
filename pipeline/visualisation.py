"""
pipeline/visualisation.py

Two visualisers for the pipeline:

1.  DashVisualiser  — live analytics dashboard in the browser.
    Starts immediately after Stage 1 (Sw series known).
    Updates live as Step B processes each scan.
    Browser opens automatically.

2.  PyVistaVisualiser — 3D viewer for the 5 clusters at timestep X,
    with a timestep scrubber to scroll through all qualifying scans.
    Launched at the END of the pipeline run on the main thread
    (OpenGL requires main thread — this blocks until window is closed).

Dependencies (optional):
    pip install pyvista dash plotly dash-bootstrap-components

Usage in pipeline.py:
    from .visualisation import PyVistaVisualiser, DashVisualiser

    # After Stage 1:
    dash_vis = DashVisualiser(out_dir=cfg.out_dir, port=cfg.dash_port)
    dash_vis.init_sw_series(sw_series, X, all_files)
    dash_vis.launch()          # starts server + opens browser

    # During Step B:
    dash_vis.update_fixed_box(scan_index, track_id, gas_voxels, z_extent)

    # After pipeline completes:
    pv_vis.show_final()        # blocks main thread — closes when user shuts window
"""
from __future__ import annotations

import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

import numpy as np

# ── PyVista ──────────────────────────────────────────────────────────────────
try:
    import pyvista as pv
    from skimage.measure import marching_cubes
    _PYVISTA_OK = True
except ImportError:
    _PYVISTA_OK = False

# ── Dash ─────────────────────────────────────────────────────────────────────
try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    import plotly.graph_objects as go
    _DASH_OK = True
except ImportError:
    _DASH_OK = False


DOWNSAMPLE_DEFAULT = 8

TRACK_COLOURS = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#E91E63", "#8BC34A",
    "#FF5722", "#607D8B",
]

def _track_colour(track_id: int) -> str:
    return TRACK_COLOURS[(track_id - 1) % len(TRACK_COLOURS)]

def _apply_dark_layout(fig, xlab: str, ylab: str) -> None:
    fig.update_layout(
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0", family="monospace"),
        xaxis=dict(title=xlab, gridcolor="#2a2a4a", zeroline=False),
        yaxis=dict(title=ylab, gridcolor="#2a2a4a", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#444"),
        margin=dict(l=50, r=20, t=20, b=40),
        hovermode="x unified",
    )


# ═══════════════════════════════════════════════════════════════════════════
# DASH VISUALISER — live analytics, background thread
# ═══════════════════════════════════════════════════════════════════════════

class DashVisualiser:
    """
    Browser-based live analytics dashboard.

    Shows:
    - Sw curve with regime boundary and qualifying zone
    - Gas voxel count per track over timesteps
    - Z-extent per track over timesteps

    Launched in a background daemon thread immediately after Stage 1.
    Browser is opened automatically. Refreshes every 3 seconds.
    Navigate to http://127.0.0.1:{port} if browser doesn't open.
    """

    def __init__(self, out_dir: Path, port: int = 8050):
        if not _DASH_OK:
            raise ImportError(
                "Dash and plotly are required.\n"
                "Install with: pip install dash plotly"
            )
        self._port = port
        self._out_dir = out_dir
        self._lock = threading.Lock()
        self._app: Optional[dash.Dash] = None
        self._thread: Optional[threading.Thread] = None

        # Shared state
        self._sw_series: list[float] = []
        self._file_names: list[str] = []
        self._X: int = -1
        self._n_total: int = 0
        self._elapsed_minutes: list[float] = []   # aligned to sw_series; empty = use scan index
        self._sw_ref: list[float] = []            # reference Sw from saturation file; empty = not available
        self._track_data: dict[int, list[tuple[int, int]]] = {}
        self._track_zextent: dict[int, list[tuple[int, int]]] = {}

    # ── Data update API ─────────────────────────────────────────────────────

    def init_sw_series(
        self,
        sw_series: list[float],
        X: int,
        all_files: list[Path],
        elapsed_minutes: list[float] | None = None,
        sat_records: dict | None = None,
    ) -> None:
        with self._lock:
            self._sw_series = list(sw_series)
            self._X = X
            self._n_total = len(all_files)
            self._file_names = [f.name for f in all_files]
            self._elapsed_minutes = list(elapsed_minutes) if elapsed_minutes else []
            # Build sw_ref list aligned to all_files
            if sat_records:
                self._sw_ref = [
                    sat_records[i].sw_ref if i in sat_records else float("nan")
                    for i in range(len(all_files))
                ]
            else:
                self._sw_ref = []

    def update_fixed_box(
        self,
        scan_index: int,
        track_id: int,
        gas_voxels: int,
        z_extent: int,
    ) -> None:
        with self._lock:
            if track_id not in self._track_data:
                self._track_data[track_id] = []
                self._track_zextent[track_id] = []
            self._track_data[track_id].append((scan_index, gas_voxels))
            self._track_zextent[track_id].append((scan_index, z_extent))

    def update_cluster(
        self,
        scan_index: int,
        track_id: int,
        gas_voxels: int,
        z_extent: int,
    ) -> None:
        self.update_fixed_box(scan_index, track_id, gas_voxels, z_extent)

    # ── Launch ──────────────────────────────────────────────────────────────

    def launch(self) -> None:
        """Start Dash server in background thread and open browser."""
        self._app = self._build_app()
        self._thread = threading.Thread(
            target=self._run_server, daemon=True, name="dash-visualiser"
        )
        self._thread.start()
        # Open browser after server has had time to start
        def _open():
            time.sleep(2.5)
            webbrowser.open(f"http://127.0.0.1:{self._port}")
        threading.Thread(target=_open, daemon=True).start()

    def _run_server(self) -> None:
        import logging as _logging
        _logging.getLogger("werkzeug").setLevel(_logging.ERROR)
        self._app.run(
            debug=False,
            port=self._port,
            use_reloader=False,
            host="127.0.0.1",   # bind to loopback only — avoids firewall issues
        )

    # ── App layout ──────────────────────────────────────────────────────────

    def _build_app(self) -> dash.Dash:
        app = dash.Dash(
            __name__,
            title="µCT Pipeline — Live Analytics",
        )

        app.layout = html.Div(
            style={
                "backgroundColor": "#1a1a2e",
                "minHeight": "100vh",
                "fontFamily": "monospace",
                "color": "#e0e0e0",
                "padding": "20px",
            },
            children=[
                html.H2(
                    "µCT Pipeline — Live Analytics",
                    style={"color": "#2196F3", "marginBottom": "4px"},
                ),
                html.Div(
                    "Refreshes every 3 seconds",
                    style={"color": "#888", "fontSize": "12px", "marginBottom": "20px"},
                ),
                dcc.Graph(id="graph-sw",   style={"height": "320px", "marginBottom": "16px"}),
                dcc.Graph(id="graph-gas",  style={"height": "280px", "marginBottom": "16px"}),
                dcc.Graph(id="graph-zext", style={"height": "280px"}),
                dcc.Interval(id="interval", interval=3000, n_intervals=0),
            ],
        )

        @app.callback(
            Output("graph-sw",   "figure"),
            Output("graph-gas",  "figure"),
            Output("graph-zext", "figure"),
            Input("interval", "n_intervals"),
        )
        def refresh(_):
            with self._lock:
                sw          = list(self._sw_series)
                X           = self._X
                names       = list(self._file_names)
                elapsed     = list(self._elapsed_minutes)
                t_data      = {k: list(v) for k, v in self._track_data.items()}
                t_zext      = {k: list(v) for k, v in self._track_zextent.items()}

            # ── Sw curve ────────────────────────────────────────────────
            sw_fig = go.Figure()

            # X axis: elapsed minutes if available, else scan index
            use_time = bool(elapsed and any(v == v for v in elapsed))
            if use_time:
                xs = [v if v == v else None for v in elapsed]
                xlab = "Elapsed Time (min)"
                x_X = elapsed[X] if X >= 0 and X < len(elapsed) and elapsed[X] == elapsed[X] else None
            else:
                xs = list(range(len(sw)))
                xlab = "Scan Index"
                x_X = X if X >= 0 else None

            # Prepend (0, 0) — experiment always starts fully gas-saturated.
            # When using elapsed time, x=0 is the true experiment start.
            # When using scan index, use x=-1 so it doesn't collide with scan 0.
            pre_x   = 0 if use_time else -1
            sw_vals = [0.0] + [v if v == v else None for v in sw]
            xs      = [pre_x] + list(xs)
            names   = ["t=0 (gas saturated)"] + list(names)

            # Qualifying region shading
            if x_X is not None:
                sw_fig.add_vrect(
                    x0=0, x1=x_X,
                    fillcolor="rgba(33,150,243,0.08)", line_width=0,
                    annotation_text="qualifying",
                    annotation_position="top left",
                    annotation_font_color="#90CAF9",
                )
                sw_fig.add_vline(
                    x=x_X, line_dash="dash", line_color="#FFC107", line_width=2,
                    annotation_text=f" X (scan {X})" if use_time else f" X={X}",
                    annotation_font_color="#FFC107",
                )

            # Pipeline Sw — connectgaps keeps line continuous through missing data
            # First point (0,0) gets grey marker; qualifying scans green; excluded grey
            n_orig = len(sw_vals) - 1
            marker_colors = ["#AAAAAA"] + ["#4CAF50" if i <= X else "#607D8B" for i in range(n_orig)]
            sw_fig.add_trace(go.Scatter(
                x=xs, y=sw_vals,
                mode="lines+markers",
                connectgaps=True,
                line=dict(color="#2196F3", width=2),
                marker=dict(size=8, color=marker_colors),
                text=names,
                hovertemplate="%{text}<br>Sw = %{y:.4f}<extra></extra>",
                name="Sw (pipeline)",
            ))

            title_text = "Water Saturation vs " + ("Elapsed Time (min)" if use_time else "Scan Index")
            sw_fig.update_layout(
                title=dict(text=title_text, font=dict(color="#e0e0e0")),
            )
            _apply_dark_layout(sw_fig, xlab, "Sw")
            sw_fig.update_yaxes(range=[0, 1])
            sw_fig.update_xaxes(rangemode="tozero")

            # ── Gas voxels per track ─────────────────────────────────────
            gas_fig = go.Figure()
            for tid, data in sorted(t_data.items()):
                d = sorted(data)
                gas_fig.add_trace(go.Scatter(
                    x=[r[0] for r in d], y=[r[1] for r in d],
                    mode="lines+markers",
                    line=dict(color=_track_colour(tid), width=2),
                    marker=dict(size=6),
                    name=f"Track {tid:02d}",
                    hovertemplate=f"Track {tid:02d}<br>Scan %{{x}}<br>%{{y:,}} voxels<extra></extra>",
                ))
            gas_fig.update_layout(
                title=dict(text="Gas Voxels per Track (qualifying scans)",
                           font=dict(color="#e0e0e0")),
            )
            _apply_dark_layout(gas_fig, "Scan Index", "Gas Voxels")

            # ── Z extent per track ────────────────────────────────────────
            zext_fig = go.Figure()
            for tid, data in sorted(t_zext.items()):
                d = sorted(data)
                zext_fig.add_trace(go.Scatter(
                    x=[r[0] for r in d], y=[r[1] for r in d],
                    mode="lines+markers",
                    line=dict(color=_track_colour(tid), width=2),
                    marker=dict(size=6),
                    name=f"Track {tid:02d}",
                    hovertemplate=f"Track {tid:02d}<br>Scan %{{x}}<br>Z extent: %{{y}} vox<extra></extra>",
                ))
            zext_fig.update_layout(
                title=dict(text="Bounding Box Z-Extent per Track",
                           font=dict(color="#e0e0e0")),
            )
            _apply_dark_layout(zext_fig, "Scan Index", "Z Extent (voxels)")

            return sw_fig, gas_fig, zext_fig

        return app


# ═══════════════════════════════════════════════════════════════════════════
# PYVISTA VISUALISER — end-of-run 3D viewer, main thread
# ═══════════════════════════════════════════════════════════════════════════

class PyVistaVisualiser:
    """
    End-of-run 3D viewer. Call show_final() from the main thread after the
    pipeline completes — it blocks until the window is closed.

    Shows all 5 clusters at timestep X as coloured meshes.
    A timestep slider lets you scroll through qualifying scans and see
    how gas volume changes within each frozen box.

    Uses pv.Plotter (not BackgroundPlotter) — OpenGL requires main thread.
    """

    def __init__(self, downsample: int = DOWNSAMPLE_DEFAULT):
        if not _PYVISTA_OK:
            raise ImportError(
                "PyVista and scikit-image are required.\n"
                "Install with: pip install pyvista scikit-image"
            )
        self._downsample = downsample
        # Accumulated cluster data — filled during the run
        # {track_id: {"shape": (Z,Y,X), "gas_path_X": Path,
        #              "scan_gas_paths": {scan_idx: Path}}}
        self._clusters: dict[int, dict] = {}
        self._scan_indices: list[int] = []
        self._X_scan_index: int = -1
        self._spacing: tuple | None = None

    # ── Data registration (called during pipeline) ──────────────────────────

    def register_cluster_at_X(
        self,
        track_id: int,
        gas_domain_path: Path,
        shape: tuple[int, int, int],
        scan_index: int,
        spacing: tuple | None = None,
        origin: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        """Register a cluster's gas domain from timestep X.
        origin: (z0, y0, x0) of the bounding box in the full volume.
        """
        if track_id not in self._clusters:
            self._clusters[track_id] = {"scan_gas_paths": {}}
        self._clusters[track_id]["shape"] = shape
        self._clusters[track_id]["origin"] = origin
        self._clusters[track_id]["gas_path_X"] = Path(gas_domain_path)
        self._clusters[track_id]["scan_gas_paths"][scan_index] = Path(gas_domain_path)
        self._X_scan_index = scan_index
        if spacing is not None:
            self._spacing = spacing
        if scan_index not in self._scan_indices:
            self._scan_indices.append(scan_index)

    def register_cluster_at_scan(
        self,
        track_id: int,
        gas_domain_path: Path,
        shape: tuple[int, int, int],
        scan_index: int,
    ) -> None:
        """Register a cluster's gas domain from a Step B scan."""
        if track_id not in self._clusters:
            self._clusters[track_id] = {"scan_gas_paths": {}}
        self._clusters[track_id]["scan_gas_paths"][scan_index] = Path(gas_domain_path)
        if track_id in self._clusters and "shape" not in self._clusters[track_id]:
            self._clusters[track_id]["shape"] = shape
        if scan_index not in self._scan_indices:
            self._scan_indices.append(scan_index)

    # ── Main thread show — blocks until window closed ───────────────────────

    def show_final(self) -> None:
        """
        Open the PyVista 3D window on the main thread.
        Blocks until the user closes the window.
        Must be called from the main thread.

        Features:
        - All clusters rendered at their correct spatial position in the volume
        - Scan slider to scrub through qualifying timesteps
        - Per-track checkbox buttons to toggle individual cluster visibility
        """
        if not self._clusters:
            print("PyVista: no clusters registered — skipping viewer.")
            return

        self._scan_indices = sorted(self._scan_indices)
        track_ids = sorted(self._clusters.keys())

        pv.set_plot_theme("dark")
        pl = pv.Plotter(
            window_size=(1400, 900),
            title="µCT Pipeline — Connected Gas Clusters",
        )

        # ── Track visibility state ──────────────────────────────────────────
        # True = currently visible. All on by default.
        track_visible: dict[int, bool] = {tid: True for tid in track_ids}

        # Consistent actor name per track — used for both initial render and
        # slider updates so there is never more than one mesh per track.
        def _actor(tid: int) -> str:
            return f"track_{tid:02d}"

        # ── Build and add initial meshes (timestep X) ───────────────────────
        print("\nBuilding 3D meshes for PyVista viewer...")
        for tid in track_ids:
            info     = self._clusters[tid]
            gas_path = info.get("gas_path_X")
            shape    = info.get("shape")
            origin   = info.get("origin", (0, 0, 0))
            if gas_path is None or shape is None or not gas_path.exists():
                print(f"  Track {tid:02d}: domain file missing, skipping")
                continue
            mesh = self._load_mesh(gas_path, shape, origin)
            if mesh is not None:
                pl.add_mesh(
                    mesh, color=_track_colour(tid), opacity=0.8,
                    smooth_shading=True, name=_actor(tid),
                )
                print(f"  Track {tid:02d}: {mesh.n_points:,} verts  origin=({origin[0]},{origin[1]},{origin[2]})")

        # ── Scan slider ─────────────────────────────────────────────────────
        _TEXT_NAME = "scan_info_text"
        pl.add_text(
            self._make_scan_text(self._X_scan_index),
            position="upper_left", font_size=9,
            color="#e0e0e0", font="courier",
            name=_TEXT_NAME,
        )

        def on_slider_change(value):
            idx      = max(0, min(int(round(value)), len(self._scan_indices) - 1))
            scan_idx = self._scan_indices[idx]

            for tid in track_ids:
                info     = self._clusters[tid]
                shape    = info.get("shape")
                origin   = info.get("origin", (0, 0, 0))
                gas_path = info.get("scan_gas_paths", {}).get(scan_idx)
                if gas_path is None or shape is None or not gas_path.exists():
                    pl.remove_actor(_actor(tid))
                    continue
                mesh = self._load_mesh(gas_path, shape, origin)
                if mesh is not None:
                    # Remove then re-add with same name — guarantees only one
                    # mesh per track exists at any time (no stacking)
                    pl.remove_actor(_actor(tid))
                    if track_visible[tid]:
                        pl.add_mesh(
                            mesh, color=_track_colour(tid), opacity=0.8,
                            smooth_shading=True, name=_actor(tid),
                        )

            pl.remove_actor(_TEXT_NAME)
            pl.add_text(
                self._make_scan_text(scan_idx),
                position="upper_left", font_size=9,
                color="#e0e0e0", font="courier",
                name=_TEXT_NAME,
            )
            pl.render()

        # Slider — integer range 0..N-1 mapping to scan indices.
        # PyVista sliders are continuous floats; on_slider_change uses int(round())
        # to snap to the nearest integer. fmt="%0.0f" shows the integer label.
        # The scan info text (upper left) updates on every move so you always
        # know exactly which scan is displayed.
        if len(self._scan_indices) > 1:
            n_scans = len(self._scan_indices)
            pl.add_slider_widget(
                callback=on_slider_change,
                rng=[0, n_scans - 1],
                value=self._scan_indices.index(self._X_scan_index)
                      if self._X_scan_index in self._scan_indices else 0,
                title=f"Scan index  (0 = earliest,  {n_scans-1} = timestep X)",
                pointa=(0.15, 0.06), pointb=(0.85, 0.06),
                style="modern",
            )

        # ── Per-track checkbox buttons ──────────────────────────────────────
        # Placed in the upper-right corner using normalised viewport coords
        # so they are never cropped regardless of window size.
        # Each button is 30px, spaced 40px apart vertically.
        win_w, win_h = 1400, 900
        btn_size  = 28
        btn_gap   = 40        # vertical spacing between buttons
        btn_x     = win_w - btn_size - 10          # 10px from right edge
        label_x   = btn_x - 80                     # label to the left of button
        btn_y_top = win_h - 60                     # start below top edge

        for btn_idx, tid in enumerate(track_ids):
            col   = _track_colour(tid)
            btn_y = btn_y_top - btn_idx * btn_gap

            def _make_cb(t=tid):
                def callback(state):
                    track_visible[t] = bool(state)
                    for a_name, a_obj in pl.renderer.actors.items():
                        if a_name == _actor(t):
                            a_obj.SetVisibility(int(state))
                    pl.render()
                return callback

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
        print("Opening PyVista window — close the window to exit.")
        pl.show()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _load_mesh(
        self,
        gas_domain_path: Path,
        shape: tuple[int, int, int],
        origin: tuple[int, int, int] = (0, 0, 0),
    ) -> Optional["pv.PolyData"]:
        """
        Load a .raw gas domain file, downsample, run marching cubes, return mesh.

        gas domain: 0 = open/gas, 1 = solid/blocked.
        origin: (z0, y0, x0) of the bounding box in full-volume voxel coordinates.
                Vertices are translated by this offset after meshing so each
                cluster sits at its correct position in the 3780-slice volume.
        """
        try:
            raw = np.fromfile(str(gas_domain_path), dtype=np.uint8)
            if raw.size != np.prod(shape):
                return None
            vol = raw.reshape(shape)
            d   = self._downsample
            vol_ds = vol[::d, ::d, ::d]
            # Domain is 0=gas, 1=solid — invert so gas=1 for marching cubes
            gas_vol = (1 - vol_ds).astype(np.float32)
            if not gas_vol.any():
                return None
            padded = np.pad(gas_vol, 1, constant_values=0)
            verts, faces, _, _ = marching_cubes(
                padded, level=0.5,
                spacing=(float(d), float(d), float(d)),
                allow_degenerate=False,
            )
            # Remove the 1-voxel padding offset, then shift to full-volume position
            # marching_cubes returns verts in (Z, Y, X) order matching shape
            verts -= float(d)                       # undo padding
            verts += np.array([                     # translate to volume position
                float(origin[0]),                   # Z offset
                float(origin[1]),                   # Y offset
                float(origin[2]),                   # X offset
            ])
            n    = len(faces)
            mesh = pv.PolyData(
                verts,
                np.hstack([np.full((n, 1), 3), faces]).ravel(),
            )
            return mesh.smooth(n_iter=20, relaxation_factor=0.1)
        except Exception as e:
            print(f"  PyVista mesh error for {gas_domain_path.name}: {e}")
            return None

    def _make_scan_text(self, scan_idx: int) -> str:
        qual   = sorted(self._scan_indices)
        pos    = qual.index(scan_idx) + 1 if scan_idx in qual else "?"
        marker = " ← timestep X" if scan_idx == self._X_scan_index else ""
        return (
            f"Scan index : {scan_idx}{marker}\n"
            f"Scan       : {pos} / {len(qual)}\n"
            f"Tracks     : {len(self._clusters)}\n"
            f"Downsample : {self._downsample}x"
        )