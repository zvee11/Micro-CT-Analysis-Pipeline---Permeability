from __future__ import annotations

import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pyvista as pv
    from skimage.measure import marching_cubes
    _PYVISTA_OK = True
except ImportError:
    _PYVISTA_OK = False

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


class DashVisualiser:
    """Browser-based live analytics dashboard, served from a background daemon thread."""

    def __init__(self, out_dir: Path, port: int = 8050):
        if not _DASH_OK:
            raise ImportError("Dash and plotly are required. Install with: pip install dash plotly")
        self._port = port
        self._out_dir = out_dir
        self._lock = threading.Lock()
        self._app: Optional[dash.Dash] = None
        self._thread: Optional[threading.Thread] = None

        self._sw_series: list[float] = []
        self._file_names: list[str] = []
        self._X: int = -1
        self._n_total: int = 0
        self._elapsed_minutes: list[float] = []
        self._sw_ref: list[float] = []
        self._track_data: dict[int, list[tuple[int, int]]] = {}
        self._track_zextent: dict[int, list[tuple[int, int]]] = {}
        self._track_box_zextent: dict[int, int] = {}

    def init_sw_series(self, sw_series, X, all_files, elapsed_minutes=None, sat_records=None) -> None:
        with self._lock:
            self._sw_series = list(sw_series)
            self._X = X
            self._n_total = len(all_files)
            self._file_names = [f.name for f in all_files]
            self._elapsed_minutes = list(elapsed_minutes) if elapsed_minutes else []
            if sat_records:
                self._sw_ref = [sat_records[i].sw_ref if i in sat_records else float("nan")
                                for i in range(len(all_files))]
            else:
                self._sw_ref = []

    def update_fixed_box(self, scan_index, track_id, gas_voxels, z_extent, box_z_extent=None) -> None:
        with self._lock:
            if track_id not in self._track_data:
                self._track_data[track_id] = []
                self._track_zextent[track_id] = []
            self._track_data[track_id].append((scan_index, gas_voxels))
            self._track_zextent[track_id].append((scan_index, z_extent))
            if box_z_extent is not None:
                self._track_box_zextent[track_id] = box_z_extent

    def launch(self) -> None:
        self._app = self._build_app()
        self._thread = threading.Thread(target=self._run_server, daemon=True, name="dash-visualiser")
        self._thread.start()

        def _open():
            time.sleep(2.5)
            webbrowser.open(f"http://127.0.0.1:{self._port}")
        threading.Thread(target=_open, daemon=True).start()

    def _run_server(self) -> None:
        import logging as _logging
        _logging.getLogger("werkzeug").setLevel(_logging.ERROR)
        self._app.run(debug=False, port=self._port, use_reloader=False, host="127.0.0.1")

    def _build_app(self) -> dash.Dash:
        app = dash.Dash(__name__, title="µCT Pipeline — Live Analytics")
        app.layout = html.Div(
            style={"backgroundColor": "#1a1a2e", "minHeight": "100vh",
                   "fontFamily": "monospace", "color": "#e0e0e0", "padding": "20px"},
            children=[
                html.H2("µCT Pipeline — Live Analytics", style={"color": "#2196F3", "marginBottom": "4px"}),
                html.Div("Refreshes every 3 seconds",
                         style={"color": "#888", "fontSize": "12px", "marginBottom": "20px"}),
                dcc.Graph(id="graph-sw", style={"height": "320px", "marginBottom": "16px"}),
                dcc.Graph(id="graph-gas", style={"height": "280px", "marginBottom": "16px"}),
                dcc.Graph(id="graph-zext", style={"height": "280px"}),
                dcc.Interval(id="interval", interval=3000, n_intervals=0),
            ],
        )

        @app.callback(
            Output("graph-sw", "figure"),
            Output("graph-gas", "figure"),
            Output("graph-zext", "figure"),
            Input("interval", "n_intervals"),
        )
        def refresh(_):
            with self._lock:
                sw = list(self._sw_series)
                X = self._X
                names = list(self._file_names)
                elapsed = list(self._elapsed_minutes)
                t_data = {k: list(v) for k, v in self._track_data.items()}
                t_zext = {k: list(v) for k, v in self._track_zextent.items()}
                box_ext = dict(self._track_box_zextent)

            # ── Sw curve ──
            sw_fig = go.Figure()
            use_time = bool(elapsed and any(v == v for v in elapsed))
            if use_time:
                xs = [v if v == v else None for v in elapsed]
                xlab = "Elapsed Time (min)"
                x_X = elapsed[X] if 0 <= X < len(elapsed) and elapsed[X] == elapsed[X] else None
            else:
                xs = list(range(len(sw)))
                xlab = "Scan Index"
                x_X = X if X >= 0 else None

            # Prepend t=0 (experiment starts fully gas-saturated); x=0 in time, x=-1 in index.
            pre_x = 0 if use_time else -1
            sw_vals = [0.0] + [v if v == v else None for v in sw]
            xs = [pre_x] + list(xs)
            names = ["t=0 (gas saturated)"] + list(names)

            if x_X is not None:
                sw_fig.add_vrect(x0=0, x1=x_X, fillcolor="rgba(33,150,243,0.08)", line_width=0,
                                 annotation_text="qualifying", annotation_position="top left",
                                 annotation_font_color="#90CAF9")
                sw_fig.add_vline(x=x_X, line_dash="dash", line_color="#FFC107", line_width=2,
                                 annotation_text=f" X (scan {X})" if use_time else f" X={X}",
                                 annotation_font_color="#FFC107")

            n_orig = len(sw_vals) - 1
            marker_colors = ["#AAAAAA"] + ["#4CAF50" if i <= X else "#607D8B" for i in range(n_orig)]
            sw_fig.add_trace(go.Scatter(
                x=xs, y=sw_vals, mode="lines+markers", connectgaps=True,
                line=dict(color="#2196F3", width=2), marker=dict(size=8, color=marker_colors),
                text=names, hovertemplate="%{text}<br>Sw = %{y:.4f}<extra></extra>", name="Sw (pipeline)",
            ))
            sw_fig.update_layout(title=dict(
                text="Water Saturation vs " + ("Elapsed Time (min)" if use_time else "Scan Index"),
                font=dict(color="#e0e0e0")))
            _apply_dark_layout(sw_fig, xlab, "Sw")
            sw_fig.update_yaxes(range=[0, 1])
            sw_fig.update_xaxes(rangemode="tozero")

            # ── Gas voxels per track ──
            gas_fig = go.Figure()
            for tid, data in sorted(t_data.items()):
                d = sorted(data)
                gas_fig.add_trace(go.Scatter(
                    x=[r[0] for r in d], y=[r[1] for r in d], mode="lines+markers",
                    line=dict(color=_track_colour(tid), width=2), marker=dict(size=6),
                    name=f"Track {tid:02d}",
                    hovertemplate=f"Track {tid:02d}<br>Scan %{{x}}<br>%{{y:,}} voxels<extra></extra>",
                ))
            gas_fig.update_layout(title=dict(text="Gas Voxels per Track (qualifying scans)",
                                             font=dict(color="#e0e0e0")))
            _apply_dark_layout(gas_fig, "Scan Index", "Gas Voxels")

            # ── Z extent per track: cluster extent (line) + frozen box (dashed ref) ──
            zext_fig = go.Figure()
            for tid, data in sorted(t_zext.items()):
                d = sorted(data)
                xs = [r[0] for r in d]
                zext_fig.add_trace(go.Scatter(
                    x=xs, y=[r[1] for r in d], mode="lines+markers",
                    line=dict(color=_track_colour(tid), width=2), marker=dict(size=6),
                    name=f"Track {tid:02d} cluster",
                    hovertemplate=f"Track {tid:02d}<br>Scan %{{x}}<br>Cluster Z: %{{y}} vox<extra></extra>",
                ))
                if tid in box_ext and xs:
                    zext_fig.add_trace(go.Scatter(
                        x=[min(xs), max(xs)], y=[box_ext[tid], box_ext[tid]], mode="lines",
                        line=dict(color=_track_colour(tid), width=1, dash="dash"),
                        name=f"Track {tid:02d} box",
                        hovertemplate=f"Track {tid:02d} box Z: {box_ext[tid]} vox<extra></extra>",
                        showlegend=False,
                    ))
            zext_fig.update_layout(title=dict(text="Cluster Z-Extent per Track (dashed = frozen box)",
                                              font=dict(color="#e0e0e0")))
            _apply_dark_layout(zext_fig, "Scan Index", "Z Extent (voxels)")

            return sw_fig, gas_fig, zext_fig

        return app


class PyVistaVisualiser:
    """End-of-run 3D viewer (main thread, blocks until closed).

    Shows one isolated tracked cluster at a time with its extraction box;
    a Scan slider steps through qualifying timesteps, a Track slider switches cluster.
    Isolation is viewer-only — it does not alter the gas domain files or the simulation.
    """

    def __init__(self, downsample: int = DOWNSAMPLE_DEFAULT,
                 run_id: str | None = None, db_path=None, connectivity: str | None = None):
        if not _PYVISTA_OK:
            raise ImportError("PyVista and scikit-image are required. "
                              "Install with: pip install pyvista scikit-image")
        self._downsample = downsample
        self._run_id = run_id
        self._db_path = db_path
        self._connectivity = connectivity
        self._clusters: dict[int, dict] = {}
        self._scan_indices: list[int] = []
        self._X_scan_index: int = -1
        self._spacing: tuple | None = None
        # Per-track percolation results computed by the button:
        # {track_id: {scan_index: {"percolates": bool, "mesh": pv.PolyData|None}}}
        self._percolation: dict[int, dict] = {}
        # Names of the per-component spanning actors currently on screen, so the
        # next redraw can remove them (their count varies scan to scan).
        self._spanning_actor_names: list[str] = []

    def register_cluster_at_X(self, track_id, gas_domain_path, shape, scan_index,
                              spacing=None, origin=(0, 0, 0), clustermask=None) -> None:
        if track_id not in self._clusters:
            self._clusters[track_id] = {"scan_gas_paths": {}}
        self._clusters[track_id]["shape"] = shape
        self._clusters[track_id]["origin"] = origin
        self._clusters[track_id]["gas_path_X"] = Path(gas_domain_path)
        self._clusters[track_id]["scan_gas_paths"][scan_index] = Path(gas_domain_path)
        if clustermask is not None:
            self._clusters[track_id]["clustermask"] = clustermask
        self._X_scan_index = scan_index
        if spacing is not None:
            self._spacing = spacing
        if scan_index not in self._scan_indices:
            self._scan_indices.append(scan_index)

    def register_cluster_at_scan(self, track_id, gas_domain_path, shape, scan_index) -> None:
        if track_id not in self._clusters:
            self._clusters[track_id] = {"scan_gas_paths": {}}
        self._clusters[track_id]["scan_gas_paths"][scan_index] = Path(gas_domain_path)
        if "shape" not in self._clusters[track_id]:
            self._clusters[track_id]["shape"] = shape
        if scan_index not in self._scan_indices:
            self._scan_indices.append(scan_index)

    def show_final(self) -> None:
        if not self._clusters:
            print("PyVista: no clusters registered — skipping viewer.")
            return

        self._scan_indices = sorted(self._scan_indices)
        track_ids = sorted(self._clusters.keys())
        scans = self._scan_indices
        if not scans or not track_ids:
            print("PyVista: nothing to display.")
            return

        pv.set_plot_theme("dark")
        pl = pv.Plotter(window_size=(1100, 850), title="µCT Pipeline — Cluster Viewer")

        state = {"track_i": 0, "scan_i": 0}
        TITLE, INFO = "title_text", "info_text"

        def _draw(recenter=True):
            tid = track_ids[state["track_i"]]
            scan_idx = scans[state["scan_i"]]
            info = self._clusters.get(tid, {})
            shape = info.get("shape")
            origin = info.get("origin", (0, 0, 0))
            gas_path = info.get("scan_gas_paths", {}).get(scan_idx)
            is_x = (scan_idx == self._X_scan_index)

            pl.remove_actor("iso")
            pl.remove_actor("gas")
            pl.remove_actor("box")
            # Remove any spanning-component actors drawn on the previous redraw.
            for _name in list(self._spanning_actor_names):
                pl.remove_actor(_name)
            self._spanning_actor_names = []

            # Percolation result for this (track, scan), if the button computed it.
            perc = self._percolation.get(tid, {}).get(scan_idx)
            # gas colour: red when computed-and-not-percolating, else track colour.
            gas_colour = "#d32f2f" if (perc is not None and not perc["percolates"]) else _track_colour(tid)

            # All gas in the section at low opacity (every scan).
            if gas_path is not None and shape is not None and gas_path.exists():
                gas_mesh = self._load_gas_mesh(gas_path, shape, origin)
                if gas_mesh is not None:
                    pl.add_mesh(gas_mesh, color=gas_colour, opacity=0.15,
                                smooth_shading=True, name="gas")

            # At timestep X, show the whole tracked cluster at high opacity.
            whole = False
            cm = info.get("clustermask")
            if is_x and cm is not None:
                iso_mesh = self._clustermask_mesh(cm)
                whole = True
                if iso_mesh is not None:
                    pl.add_mesh(iso_mesh, color=_track_colour(tid), opacity=0.95,
                                smooth_shading=True, name="iso")
            # Non-X: if the button found spanning components, draw EACH of them.
            # The X route (matched to timestep X by inlet/outlet footprint) is
            # amber; the other percolating paths are a distinct colour. Trapped
            # (non-spanning) gas remains in the low-opacity context layer above.
            elif perc is not None and perc["percolates"]:
                spanning_meshes = perc.get("spanning_meshes") or []
                if spanning_meshes:
                    for i, (m, this_is_x) in enumerate(spanning_meshes):
                        if m is None:
                            continue
                        if this_is_x:
                            colour, opacity = "#FFD54F", 0.95   # amber: the X route
                        else:
                            colour, opacity = "#42A5F5", 0.85   # blue: other spanning paths
                        name = f"span_{i}"
                        pl.add_mesh(m, color=colour, opacity=opacity,
                                    smooth_shading=True, name=name)
                        self._spanning_actor_names.append(name)
                elif perc.get("mesh") is not None:
                    # Fallback (older result without per-component list).
                    colour = "#FFD54F" if perc.get("is_x_route") else _track_colour(tid)
                    pl.add_mesh(perc["mesh"], color=colour, opacity=0.95,
                                smooth_shading=True, name="iso")

            box = None
            if shape is not None:
                box = self._box_wireframe(origin, shape)
                pl.add_mesh(box, color=_track_colour(tid), style="wireframe",
                            line_width=2, opacity=0.6, name="box")

            pl.remove_actor(TITLE)
            status = "   [whole cluster @ X]" if whole else ""
            pl.add_text(f"Track {tid:02d}{status}", position="upper_left", font_size=12,
                        color=_track_colour(tid), font="courier", name=TITLE)
            pl.remove_actor(INFO)
            marker = "  <- timestep X" if is_x else ""
            pl.add_text(f"scan {scan_idx}{marker}   ({state['scan_i']+1}/{len(scans)})",
                        position="upper_right", font_size=10, color="#e0e0e0",
                        font="courier", name=INFO)

            if recenter and box is not None:
                pl.reset_camera(bounds=box.bounds)
            pl.render()

        def on_scan(value):
            i = max(0, min(int(round(value)), len(scans) - 1))
            if i != state["scan_i"]:
                state["scan_i"] = i
                _draw(recenter=False)

        def on_track(value):
            i = max(0, min(int(round(value)), len(track_ids) - 1))
            if i != state["track_i"]:
                state["track_i"] = i
                state["scan_i"] = 0
                _draw(recenter=True)

        def on_percolation(_state=None):
            tid = track_ids[state["track_i"]]
            print(f"\n[percolation] computing track {tid:02d} across all scans...")
            self._compute_percolation_for_track(tid)
            _draw(recenter=False)

        print("\nBuilding PyVista viewer...")
        _draw()
        pl.camera_position = "iso"
        pl.add_axes()

        if len(scans) > 1:
            pl.add_slider_widget(callback=on_scan, rng=[0, len(scans) - 1], value=0,
                                 title="Scan  (left = earliest, right = latest)",
                                 pointa=(0.20, 0.08), pointb=(0.80, 0.08), style="modern")
        if len(track_ids) > 1:
            pl.add_slider_widget(callback=on_track, rng=[0, len(track_ids) - 1], value=0,
                                 title="Track  (slide to switch cluster)",
                                 pointa=(0.20, 0.16), pointb=(0.80, 0.16), style="modern")

        # "Check percolation" button: computes the current track across all scans,
        # writes results to the DB, and highlights the spanning component per scan.
        pl.add_text("Check percolation", position=(40, 70), font_size=9,
                    color="#e0e0e0", font="courier", name="perc_label")
        pl.add_checkbox_button_widget(on_percolation, value=False,
                                      position=(10, 65), size=25,
                                      color_on="#4caf50", color_off="#888888")

        print("Opening PyVista window — close the window to exit.")
        pl.show()

    def _load_gas_mesh(self, gas_domain_path, shape, origin=(0, 0, 0)):
        """Load a gas domain (.raw, 0=gas/1=solid) and mesh ALL gas at the
        current downsample. No connectivity/percolation — just the full gas in
        the section, for low-opacity context. Returns a mesh or None."""
        try:
            raw = np.fromfile(str(gas_domain_path), dtype=np.uint8)
            if raw.size != int(np.prod(shape)):
                return None
            vol = raw.reshape(shape)
            gas = (vol == 0)
            if not gas.any():
                return None
            d = self._downsample
            return self._mesh_from_bool(gas[::d, ::d, ::d], d, origin)
        except Exception as e:
            print(f"  PyVista gas-mesh error for {gas_domain_path.name}: {e}")
            return None

    @staticmethod
    def _face_footprints(component: np.ndarray) -> "tuple[set, set]":
        """(inlet, outlet) footprints of a boolean component: the (y,x) pixels
        it occupies on the z=0 and z=max faces."""
        inlet = set(map(tuple, np.argwhere(component[0])))
        outlet = set(map(tuple, np.argwhere(component[-1])))
        return inlet, outlet

    @staticmethod
    def _overlap(a: set, b: set) -> float:
        """Jaccard overlap of two pixel sets; 0 if both empty."""
        if not a and not b:
            return 0.0
        union = len(a | b)
        return (len(a & b) / union) if union else 0.0

    def _x_route_footprints(self, tid):
        """Compute and cache the original X route's inlet/outlet footprints and
        CoG for track `tid`, from the X gas domain. Returns dict or None.

        The X route is the largest component of the X gas domain that spans both
        Z-faces — the same selection the pipeline used at X. Later scans are
        matched against this footprint, not re-derived, so 'same route' means
        'pierces the inlet/outlet at the same pixels as X did'."""
        info = self._clusters.get(tid, {})
        if "_x_route" in info:
            return info["_x_route"]
        shape = info.get("shape")
        x_path = info.get("gas_path_X")
        out = None
        if shape is not None and x_path is not None and Path(x_path).exists():
            import cc3d
            raw = np.fromfile(str(x_path), dtype=np.uint8)
            if raw.size == int(np.prod(shape)):
                gas = (raw.reshape(shape) == 0)
                if gas.any():
                    labels = cc3d.connected_components(gas.astype(np.uint8), connectivity=26)
                    inlet = set(np.unique(labels[0])) - {0}
                    outlet = set(np.unique(labels[-1])) - {0}
                    spanning = inlet & outlet
                    if spanning:
                        chosen = max(spanning, key=lambda c: int((labels == c).sum()))
                        comp = (labels == chosen)
                        fin, fout = self._face_footprints(comp)
                        cog = tuple(float(v) for v in np.argwhere(comp).mean(axis=0))
                        out = {"inlet": fin, "outlet": fout, "cog": cog}
                    del labels
        self._clusters.setdefault(tid, {})["_x_route"] = out
        return out

    def _compute_percolation_for_track(self, tid):
        """Button worker. For the current track, over all NON-X scans:
        load the scan's gas .raw, run cc3d at FULL resolution, take the largest
        component spanning both Z-faces, write percolates/spanning_count/
        cluster_voxels to the DB, and keep a small DOWNSAMPLED mesh of the
        spanning component for display. One full-res box in memory at a time
        (freed each scan). Full res is mandatory for cc3d; downsampling is only
        for the on-screen mesh.
        """
        import cc3d
        import gc as _gc

        info = self._clusters.get(tid, {})
        shape = info.get("shape")
        origin = info.get("origin", (0, 0, 0))
        gas_paths = info.get("scan_gas_paths", {})
        d = self._downsample
        self._percolation[tid] = {}

        con = self._open_db()
        try:
            for scan_idx in sorted(gas_paths):
                if scan_idx == self._X_scan_index:
                    continue  # X keeps its pipeline values; not recomputed
                gas_path = gas_paths[scan_idx]
                if shape is None or not Path(gas_path).exists():
                    print(f"  [percolation] track {tid:02d} scan {scan_idx}: gas missing, skipped")
                    continue

                raw = np.fromfile(str(gas_path), dtype=np.uint8)
                if raw.size != int(np.prod(shape)):
                    print(f"  [percolation] track {tid:02d} scan {scan_idx}: size mismatch, skipped")
                    continue
                gas_full = (raw.reshape(shape) == 0)  # FULL res, gas==0 in domain .raw

                percolates, spanning_count, cluster_voxels, mesh = False, 0, 0, None
                is_x_route = False
                spanning_meshes = []  # list of (mesh, is_x_route) for ALL spanning components
                if gas_full.any():
                    labels = cc3d.connected_components(gas_full.astype(np.uint8), connectivity=26)
                    inlet = set(np.unique(labels[0])) - {0}
                    outlet = set(np.unique(labels[-1])) - {0}
                    spanning = inlet & outlet
                    if spanning:
                        percolates = True
                        spanning_count = len(spanning)
                        x_route = self._x_route_footprints(tid)
                        # Score each spanning component by inlet+outlet footprint
                        # overlap with the original X route. STRICT: only the
                        # best-overlapping component, and only if overlap > 0, is
                        # marked as the X route. If the original route has pinched
                        # off, nothing here is the X route (is_x_route stays False)
                        # even though some other path still spans.
                        scored = []  # (component_label, score)
                        best_c, best_score = None, -1.0
                        for c in spanning:
                            comp_c = (labels == c)
                            if x_route is not None:
                                fin, fout = self._face_footprints(comp_c)
                                score = (self._overlap(fin, x_route["inlet"])
                                         + self._overlap(fout, x_route["outlet"]))
                            else:
                                score = float(int(comp_c.sum()))
                            scored.append((c, score))
                            if score > best_score:
                                best_c, best_score = c, score

                        # The X route is the best match, only if it overlaps at all.
                        x_label = best_c if (x_route is not None and best_score > 0.0) else None
                        is_x_route = x_label is not None

                        # Build a display mesh for EVERY spanning component, tagging
                        # the one that is the X route. The X route's voxel count is
                        # the one written to the DB (the kr-relevant cluster); if no
                        # component matches X, fall back to the largest spanning one.
                        chosen = best_c
                        component = (labels == chosen)
                        cluster_voxels = int(component.sum())
                        for c, score in scored:
                            comp_c = (labels == c)
                            this_is_x = (c == x_label)
                            m = self._mesh_from_bool(comp_c[::d, ::d, ::d], d, origin)
                            if m is not None:
                                spanning_meshes.append((m, this_is_x))
                            if this_is_x:
                                mesh = m  # keep the X-route mesh as the primary 'mesh'

                        if mesh is None and spanning_meshes:
                            mesh = spanning_meshes[0][0]  # no X match: show largest as primary

                        if x_route is not None:
                            cog = np.argwhere(component).mean(axis=0)
                            cog_dist = float(np.linalg.norm(cog - np.array(x_route["cog"])))
                            print(f"  [match] track {tid:02d} scan {scan_idx}: "
                                  f"spanning={spanning_count} x_route_overlap={best_score:.3f} "
                                  f"cog_dist={cog_dist:.1f}vox is_x_route={is_x_route}")
                    del labels
                del gas_full, raw

                if con is not None:
                    try:
                        con.execute(
                            "UPDATE fixed_boxes SET percolates=?, spanning_count=?, cluster_voxels=? "
                            "WHERE run_id=? AND scan_index=? AND track_id=? AND connectivity=?",
                            (percolates, spanning_count, cluster_voxels,
                             self._run_id, scan_idx, tid, self._connectivity),
                        )
                    except Exception as e:
                        print(f"  [percolation] DB write failed (scan {scan_idx}): {e}")

                self._percolation[tid][scan_idx] = {
                    "percolates": percolates, "mesh": mesh, "is_x_route": is_x_route,
                    "spanning_meshes": spanning_meshes,
                }
                print(f"  [percolation] track {tid:02d} scan {scan_idx}: "
                      f"percolates={percolates} spanning={spanning_count} voxels={cluster_voxels}")
                _gc.collect()
        finally:
            if con is not None:
                con.close()
        print(f"  [percolation] track {tid:02d} done.")

    def _open_db(self):
        """Short-lived DuckDB connection for the button to UPDATE results.
        Returns None (and prints) if unavailable, so the viewer never crashes."""
        if not (self._run_id and self._db_path and self._connectivity):
            print("  [percolation] no DB context (run_id/db_path/connectivity); "
                  "highlight only, not saved.")
            return None
        try:
            import duckdb
            return duckdb.connect(str(self._db_path))
        except Exception as e:
            print(f"  [percolation] could not open DB ({e}); highlight only, not saved.")
            return None

    def _clustermask_mesh(self, cm):
        """Mesh the whole tracked cluster mask saved at X (0/1 .raw, 1=cluster),
        placed at its own (z0,y0,x0) origin so the box wireframe sits inside it."""
        try:
            shape = cm["shape"]
            raw = np.fromfile(str(cm["path"]), dtype=np.uint8)
            if raw.size != int(np.prod(shape)):
                return None
            mask = raw.reshape(shape) > 0
            d = self._downsample
            mask_ds = mask[::d, ::d, ::d]
            return self._mesh_from_bool(mask_ds, d, (cm["z0"], cm["y0"], cm["x0"]))
        except Exception as e:
            print(f"  PyVista clustermask error: {e}")
            return None

    def _mesh_from_bool(self, gas_bool, d, origin):
        """Marching-cubes mesh from an already-downsampled boolean gas array.

        No smoothing: Laplacian smoothing pulls the surface inward and opens a
        false gap to the extraction box, so the raw voxel-boundary surface is kept.
        """
        try:
            if not gas_bool.any():
                return None
            padded = np.pad(gas_bool.astype(np.float32), 1, constant_values=0)
            verts, faces, _, _ = marching_cubes(
                padded, level=0.5, spacing=(float(d), float(d), float(d)), allow_degenerate=False)
            verts -= float(d)
            verts += np.array([float(origin[0]), float(origin[1]), float(origin[2])])
            n = len(faces)
            return pv.PolyData(verts, np.hstack([np.full((n, 1), 3), faces]).ravel())
        except Exception:
            return None

    def _box_wireframe(self, origin, shape):
        # Verts are placed in (Z, Y, X) order, so pv.Box bounds map x<-Z, y<-Y, z<-X.
        z0, y0, x0 = origin
        nz, ny, nx = shape
        return pv.Box(bounds=(float(z0), float(z0 + nz), float(y0), float(y0 + ny),
                              float(x0), float(x0 + nx)))