"""
pipeline/visualisation.py

Two independent visualisers launched alongside the pipeline:

1.  PyVistaVisualiser  — 3D interactive viewer showing gas cluster meshes
    as they are extracted. Downsamples domains 8x before meshing.
    Launched in a background thread when the first cluster is saved.

2.  DashVisualiser     — Browser-based analytics dashboard showing:
    - Sw curve with regime boundary and qualifying zone
    - Gas voxel evolution per track over timesteps
    - Cluster bounding box dimensions over time
    Launched after Stage 1 (Sw series known) and updated live.

Both are optional — the pipeline runs correctly if neither is imported.
Launch is mode-aware:
    - Fixed mode:   PyVista launches after Step A (clusters defined at X)
                    Dash launches after Stage 1
    - Dynamic mode: PyVista updates after each qualifying timestep

Dependencies (optional, not in core requirements):
    pip install pyvista pyvistaqt dash plotly dash-bootstrap-components

Usage in pipeline.py:
    from .visualisation import PyVistaVisualiser, DashVisualiser

    pv_vis = PyVistaVisualiser(downsample=8)
    dash_vis = DashVisualiser(out_dir=cfg.out_dir)

    # After Stage 1:
    dash_vis.init_sw_series(sw_series, X, all_files)
    dash_vis.launch()

    # After Step A / after each CC run:
    pv_vis.add_cluster(domain_gas_path, track_id, spacing)
    pv_vis.launch_if_needed()

    # After each Step B file:
    dash_vis.update_fixed_box(file_stem, track_id, gas_voxels, sw)
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── PyVista ─────────────────────────────────────────────────────────────────
try:
    import pyvista as pv
    from pyvistaqt import BackgroundPlotter
    _PYVISTA_OK = True
except ImportError:
    _PYVISTA_OK = False

# ── Dash ─────────────────────────────────────────────────────────────────────
try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    import plotly.graph_objects as go
    import dash_bootstrap_components as dbc
    _DASH_OK = True
except ImportError:
    _DASH_OK = False


DOWNSAMPLE_DEFAULT = 8

# Colour palette for tracks — up to 10 distinct colours
TRACK_COLOURS = [
    "#2196F3",  # blue
    "#F44336",  # red
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#E91E63",  # pink
    "#8BC34A",  # light green
    "#FF5722",  # deep orange
    "#607D8B",  # blue grey
]


def _track_colour(track_id: int) -> str:
    return TRACK_COLOURS[(track_id - 1) % len(TRACK_COLOURS)]


# ── PyVista Visualiser ───────────────────────────────────────────────────────

class PyVistaVisualiser:
    """
    3D interactive viewer for gas cluster domains.

    Each call to add_cluster() loads a .raw binary domain file, downsamples
    it by `downsample` factor, runs marching cubes to produce a surface mesh,
    and adds it to the BackgroundPlotter window with a track-specific colour.

    The window stays open and updates live as new clusters are added.
    Downsampling at 8x reduces a 750x750x1000 volume to ~94x94x125 —
    small enough for smooth real-time rendering while preserving cluster shape.
    """

    def __init__(self, downsample: int = DOWNSAMPLE_DEFAULT):
        if not _PYVISTA_OK:
            raise ImportError(
                "PyVista and pyvistaqt are required for 3D visualisation.\n"
                "Install with: pip install pyvista pyvistaqt"
            )
        self._downsample = downsample
        self._plotter: Optional[BackgroundPlotter] = None
        self._launched = False
        self._meshes: dict[int, pv.PolyData] = {}
        self._lock = threading.Lock()

    def launch_if_needed(self) -> None:
        """Open the PyVista window if not already open."""
        if self._launched:
            return
        self._plotter = BackgroundPlotter(
            title="Gas Cluster Viewer — µCT Pipeline",
            window_size=(1200, 800),
        )
        self._plotter.set_background("black")
        self._plotter.add_axes()
        self._plotter.show_grid(color="gray")
        self._launched = True

    def add_cluster(
        self,
        domain_gas_path: Path | str,
        track_id: int,
        spacing: tuple[float, float, float] | None,
        label: str = "",
    ) -> None:
        """
        Load a .raw gas domain file, downsample, mesh, and add to viewer.

        Parameters
        ----------
        domain_gas_path:
            Path to .raw binary file (uint8, 0=gas/open, 1=solid/blocked).
            Shape must be inferrable from file size or passed via metadata.
        track_id:
            Used to assign a consistent colour across timesteps.
        spacing:
            Voxel spacing in metres (z, y, x). Used to scale the mesh.
            If None, unit spacing is used.
        label:
            Optional label shown in the legend (e.g. "Track 01 — scan 007").
        """
        if not self._launched:
            self.launch_if_needed()

        path = Path(domain_gas_path)
        if not path.exists():
            return

        raw = np.fromfile(path, dtype=np.uint8)

        # Infer shape — we need it to be passed or stored
        # For now accept shape as stored in a sidecar .shape file or derive
        # from the frozen box dimensions stored in the row dict
        # This method is called with the domain path; shape must be inferrable
        # We'll read the shape from a sidecar if present, else skip
        shape_file = path.with_suffix(".shape")
        if not shape_file.exists():
            return  # shape not available — skip rendering
        shape = tuple(int(x) for x in shape_file.read_text().strip().split(","))
        if len(shape) != 3 or np.prod(shape) != raw.size:
            return

        vol = raw.reshape(shape).astype(np.uint8)

        # Downsample — simple stride slicing, fast and sufficient for visualisation
        d = self._downsample
        vol_ds = vol[::d, ::d, ::d]

        # PyVista ImageData (voxel grid)
        grid = pv.ImageData(dimensions=tuple(s + 1 for s in vol_ds.shape))
        if spacing is not None:
            sz, sy, sx = spacing
            grid.spacing = (sx * d, sy * d, sz * d)

        # gas = 0, solid = 1 — we want to mesh the gas (0) surface
        # invert so gas = 1, then threshold and mesh
        grid.cell_data["gas"] = (1 - vol_ds).ravel(order="F").astype(np.float32)

        mesh = grid.threshold(0.5, scalars="gas").extract_surface()
        mesh = mesh.smooth(n_iter=50, relaxation_factor=0.1)
        mesh = mesh.decimate(0.5)  # reduce triangle count by 50%

        colour = _track_colour(track_id)
        actor_name = f"track_{track_id:02d}"

        with self._lock:
            # Remove previous mesh for this track (updates between timesteps)
            if track_id in self._meshes:
                self._plotter.remove_actor(actor_name)

            self._plotter.add_mesh(
                mesh,
                color=colour,
                opacity=0.75,
                smooth_shading=True,
                name=actor_name,
                label=label or f"Track {track_id:02d}",
            )
            self._meshes[track_id] = mesh
            self._plotter.add_legend(bcolor="black", face="circle", size=(0.15, 0.15))

    def write_shape_sidecar(self, domain_path: Path, shape: tuple[int, int, int]) -> None:
        """
        Write a .shape sidecar file alongside a .raw domain file.
        Called from save_outputs / apply_frozen_boxes after writing .raw.
        Enables PyVista to reconstruct the array without storing shape elsewhere.
        """
        sidecar = domain_path.with_suffix(".shape")
        sidecar.write_text(",".join(str(s) for s in shape))

    def close(self) -> None:
        if self._plotter is not None:
            self._plotter.close()


# ── Dash Analytics Dashboard ─────────────────────────────────────────────────

class DashVisualiser:
    """
    Browser-based analytics dashboard.

    Displays:
    - Sw curve over all scans with regime boundary marked
    - Gas voxel count per track over qualifying timesteps
    - Cluster bounding box Z-extent over time (fixed mode)
    - Qualifying/excluded scan indicator

    Launched in a background thread on port 8050.
    Data is updated via shared state — thread-safe via a lock.
    Navigate to http://localhost:8050 in any browser.
    """

    def __init__(self, out_dir: Path, port: int = 8050):
        if not _DASH_OK:
            raise ImportError(
                "Dash and plotly are required for the analytics dashboard.\n"
                "Install with: pip install dash plotly dash-bootstrap-components"
            )
        self._port = port
        self._out_dir = out_dir
        self._lock = threading.Lock()
        self._app: Optional[dash.Dash] = None
        self._thread: Optional[threading.Thread] = None

        # Shared state — updated by pipeline, read by Dash callbacks
        self._sw_series: list[float] = []
        self._file_names: list[str] = []
        self._X: int = -1
        self._n_total: int = 0
        # track_id -> list of (scan_index, gas_voxels)
        self._track_data: dict[int, list[tuple[int, int]]] = {}
        # track_id -> list of (scan_index, z_extent)
        self._track_zextent: dict[int, list[tuple[int, int]]] = {}

    # ── Data update API (called from pipeline) ──────────────────────────────

    def init_sw_series(
        self,
        sw_series: list[float],
        X: int,
        all_files: list[Path],
    ) -> None:
        with self._lock:
            self._sw_series = list(sw_series)
            self._X = X
            self._n_total = len(all_files)
            self._file_names = [f.name for f in all_files]

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
        """Same as update_fixed_box — works for dynamic mode too."""
        self.update_fixed_box(scan_index, track_id, gas_voxels, z_extent)

    # ── Launch ──────────────────────────────────────────────────────────────

    def launch(self) -> None:
        """Start the Dash server in a background daemon thread."""
        self._app = self._build_app()
        self._thread = threading.Thread(
            target=self._run_server, daemon=True, name="dash-visualiser"
        )
        self._thread.start()
        # Brief pause so server is up before first browser open
        time.sleep(1.0)
        print(f"\n  📊  Analytics dashboard: http://localhost:{self._port}\n")

    def _run_server(self) -> None:
        import logging as _logging
        _logging.getLogger("werkzeug").setLevel(_logging.ERROR)
        self._app.run(
            debug=False, port=self._port, use_reloader=False, host="0.0.0.0"
        )

    # ── App layout ──────────────────────────────────────────────────────────

    def _build_app(self) -> dash.Dash:
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            title="µCT Pipeline — Analytics",
        )

        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H3(
                    "µCT Relative Permeability Pipeline — Live Analytics",
                    className="text-center text-info py-3"
                ))
            ]),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Water Saturation Curve  (Sw vs Scan Index)",
                                       className="text-info"),
                        dbc.CardBody(dcc.Graph(id="graph-sw", style={"height": "340px"})),
                    ], color="dark", outline=True)
                ], width=12)
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Gas Voxel Count per Track  (over time)",
                                       className="text-info"),
                        dbc.CardBody(dcc.Graph(id="graph-gas", style={"height": "300px"})),
                    ], color="dark", outline=True)
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Bounding Box Z-Extent per Track  (over time)",
                                       className="text-info"),
                        dbc.CardBody(dcc.Graph(id="graph-zext", style={"height": "300px"})),
                    ], color="dark", outline=True)
                ], width=6),
            ], className="mb-3"),

            dcc.Interval(id="interval", interval=3000, n_intervals=0),  # refresh every 3s
        ], fluid=True, className="bg-dark text-light")

        # ── Callbacks ──────────────────────────────────────────────────────

        @app.callback(
            Output("graph-sw",   "figure"),
            Output("graph-gas",  "figure"),
            Output("graph-zext", "figure"),
            Input("interval", "n_intervals"),
        )
        def update_figures(_):
            with self._lock:
                sw       = list(self._sw_series)
                X        = self._X
                names    = list(self._file_names)
                n_total  = self._n_total
                t_data   = {k: list(v) for k, v in self._track_data.items()}
                t_zext   = {k: list(v) for k, v in self._track_zextent.items()}

            # ── Sw curve ───────────────────────────────────────────────
            sw_fig = go.Figure()
            xs = list(range(len(sw)))
            sw_clean = [v if v == v else None for v in sw]  # nan -> None

            # Qualifying region shading
            if X >= 0:
                sw_fig.add_vrect(
                    x0=-0.5, x1=X + 0.5,
                    fillcolor="rgba(33,150,243,0.08)",
                    line_width=0,
                    annotation_text="qualifying",
                    annotation_position="top left",
                    annotation_font_color="#90CAF9",
                )
                sw_fig.add_vline(
                    x=X, line_dash="dash", line_color="#FFC107", line_width=2,
                    annotation_text=f" X={X}", annotation_font_color="#FFC107",
                )

            sw_fig.add_trace(go.Scatter(
                x=xs, y=sw_clean,
                mode="lines+markers",
                line=dict(color="#2196F3", width=2),
                marker=dict(size=7, color=[
                    "#4CAF50" if i <= X else "#607D8B"
                    for i in range(len(sw))
                ]),
                text=names,
                hovertemplate="%{text}<br>Sw = %{y:.4f}<extra></extra>",
                name="Sw",
            ))
            _apply_dark_layout(sw_fig, "Scan Index", "S_w")

            # ── Gas voxels per track ────────────────────────────────────
            gas_fig = go.Figure()
            for tid, data in sorted(t_data.items()):
                data_sorted = sorted(data, key=lambda r: r[0])
                xs_t = [r[0] for r in data_sorted]
                ys_t = [r[1] for r in data_sorted]
                gas_fig.add_trace(go.Scatter(
                    x=xs_t, y=ys_t,
                    mode="lines+markers",
                    line=dict(color=_track_colour(tid), width=2),
                    marker=dict(size=6),
                    name=f"Track {tid:02d}",
                    hovertemplate=f"Track {tid:02d}<br>Scan %{{x}}<br>Gas voxels: %{{y:,}}<extra></extra>",
                ))
            _apply_dark_layout(gas_fig, "Scan Index", "Gas Voxels")

            # ── Z extent per track ──────────────────────────────────────
            zext_fig = go.Figure()
            for tid, data in sorted(t_zext.items()):
                data_sorted = sorted(data, key=lambda r: r[0])
                xs_t = [r[0] for r in data_sorted]
                ys_t = [r[1] for r in data_sorted]
                zext_fig.add_trace(go.Scatter(
                    x=xs_t, y=ys_t,
                    mode="lines+markers",
                    line=dict(color=_track_colour(tid), width=2),
                    marker=dict(size=6),
                    name=f"Track {tid:02d}",
                    hovertemplate=f"Track {tid:02d}<br>Scan %{{x}}<br>Z extent: %{{y}} vox<extra></extra>",
                ))
            _apply_dark_layout(zext_fig, "Scan Index", "Z Extent (voxels)")

            return sw_fig, gas_fig, zext_fig

        return app


def _apply_dark_layout(fig: "go.Figure", xlab: str, ylab: str) -> None:
    fig.update_layout(
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0", family="monospace"),
        xaxis=dict(title=xlab, gridcolor="#2a2a4a", zeroline=False),
        yaxis=dict(title=ylab, gridcolor="#2a2a4a", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#444"),
        margin=dict(l=50, r=20, t=20, b=40),
        hovermode="x unified",
    )
