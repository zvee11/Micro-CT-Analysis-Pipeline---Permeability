"""
cluster_comparison_combined.py
-------------------------------
Combined Dash dashboard + PyVista 3D viewer for comparing an Avizo RAW
cluster export against a corrected CCA TIFF mask.

Running this single script:
  1. Loads and aligns both volumes (bounding-box auto-detected offsets)
  2. Opens a PyVista 3D window (main thread — required by OpenGL/VTK)
  3. Starts a Dash dashboard in a background thread (http://127.0.0.1:8050)

Bidirectional interactions
--------------------------
  Dash → PyVista:
    • Z-slider in Dash moves the cutting plane in the 3D window
    • Opacity slider controls the transparency of the "both" surface

  PyVista → Dash:
    • The current Z slice index shown in the 3D window is reflected in Dash
    • Agreement stats update live in both views

Shared state
------------
All cross-thread communication goes through `SHARED`, a dict protected
by `STATE_LOCK`. Dash callbacks write to it; PyVista's timer callback
reads it and updates actors.

Usage
-----
    pip install dash plotly pyvista trame trame-vtk trame-vuetify \\
                scikit-image pillow numpy
    python cluster_comparison_combined.py
    # Then open http://127.0.0.1:8050 in your browser.

Configuration
-------------
Edit RAW_PATH, TIF_PATH and DS at the top of this file.
"""

from __future__ import annotations

import sys
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import dash
    from dash import dcc, html, Input, Output, State
    import plotly.graph_objects as go
    import pyvista as pv
    from skimage.measure import marching_cubes
except ImportError as e:
    sys.exit(
        f"Missing dependency: {e}\n\n"
        "Install with:\n"
        "  pip install dash plotly pyvista scikit-image pillow numpy"
    )

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
RAW_PATH = Path("8_2_biggestcluster.view.raw")
TIF_PATH = Path("cluster_01_mask_26N.tiff")
DS       = 4       # Downsample factor for surface extraction
DASH_PORT = 8050
TIMER_MS  = 150    # PyVista timer poll interval in milliseconds

# ── SHARED STATE (Dash ↔ PyVista) ─────────────────────────────────────────────
STATE_LOCK = threading.Lock()
SHARED = {
    'z_idx':         0,       # current Z slice (canvas index), set by Dash slider
    'both_opacity':  0.25,    # opacity of agreement surface, set by Dash slider
    'pv_ready':      False,   # True once PyVista window is open
    'dash_ready':    False,   # True once Dash server is up
    'pv_z_idx':      0,       # Z index currently shown in PyVista (read by Dash)
}

# ── LOADERS ───────────────────────────────────────────────────────────────────

def load_tiff_stack(path: Path) -> np.ndarray:
    img = Image.open(path)
    frames = []
    for i in range(img.n_frames):
        img.seek(i)
        frames.append(np.array(img))
    return np.stack(frames, axis=0)


def infer_raw_shape(raw_path: Path, tif: np.ndarray) -> tuple[int, int, int]:
    ny, nx   = tif.shape[1], tif.shape[2]
    n_voxels = raw_path.stat().st_size // 2
    if n_voxels % (ny * nx) != 0:
        raise ValueError(f"Cannot infer RAW Z: {raw_path.stat().st_size} bytes / 2 "
                         f"not divisible by {ny}×{nx}.")
    return n_voxels // (ny * nx), ny, nx


def bbox3d(vol: np.ndarray) -> dict:
    idx = np.where(vol > 0)
    return {ax: (int(idx[i].min()), int(idx[i].max()))
            for i, ax in enumerate(['z', 'y', 'x'])}


# ── LOAD & ALIGN ──────────────────────────────────────────────────────────────

def load_and_align(raw_path: Path, tif_path: Path, ds: int) -> dict:
    """
    Load both volumes, compute bounding boxes, derive offsets, and return
    all data needed by both the Dash and PyVista views.
    """
    print("Loading TIFF stack…")
    tif_vol = load_tiff_stack(tif_path)

    print("Loading RAW (memmap)…")
    nz, ny, nx = infer_raw_shape(raw_path, tif_vol)
    raw_vol = np.memmap(str(raw_path), dtype=np.uint16, mode='r', shape=(nz, ny, nx))

    print("Computing bounding boxes…")
    rbb = bbox3d(raw_vol)
    tbb = bbox3d(tif_vol)

    Z_OFF      = tbb['z'][0] - rbb['z'][0]
    TIF_XSHIFT = rbb['x'][0] - tbb['x'][0]
    TIF_YSHIFT = rbb['y'][0] - tbb['y'][0]
    print(f"  Z_OFF={Z_OFF}  TIF_XSHIFT={TIF_XSHIFT}  TIF_YSHIFT={TIF_YSHIFT}")

    cz_min = tbb['z'][0];  cz_max = max(tbb['z'][1], rbb['z'][1] + Z_OFF)
    cy_min = rbb['y'][0];  cy_max = max(rbb['y'][1], tbb['y'][1] + TIF_YSHIFT)
    cx_min = rbb['x'][0];  cx_max = max(rbb['x'][1], tbb['x'][1] + TIF_XSHIFT)
    CZ = cz_max - cz_min + 1
    CY = cy_max - cy_min + 1
    CX = cx_max - cx_min + 1

    # Downsampled canvas dimensions
    dCZ = (CZ + ds - 1) // ds
    dCY = (CY + ds - 1) // ds
    dCX = (CX + ds - 1) // ds

    def or_ds(sl, dH, dW):
        """OR-downsample a 2D slice to (dH, dW), padding as needed."""
        padded = np.zeros((dH * ds, dW * ds), dtype=sl.dtype)
        h, w = min(sl.shape[0], dH * ds), min(sl.shape[1], dW * ds)
        padded[:h, :w] = sl[:h, :w]
        return (padded > 0).reshape(dH, ds, dW, ds).any(axis=(1, 3)).astype(np.uint8)

    # Full-res canvas for Dash slice viewer
    raw_canvas_full = np.zeros((CZ, CY, CX), dtype=np.uint8)
    tif_canvas_full = np.zeros((CZ, CY, CX), dtype=np.uint8)

    # Downsampled canvas for PyVista surfaces
    raw_canvas_ds = np.zeros((dCZ, dCY, dCX), dtype=np.uint8)
    tif_canvas_ds = np.zeros((dCZ, dCY, dCX), dtype=np.uint8)

    ty0 = cy_min - TIF_YSHIFT;  ty1 = cy_max - TIF_YSHIFT
    tx0 = cx_min - TIF_XSHIFT;  tx1 = cx_max - TIF_XSHIFT

    print("Building RAW canvas…")
    for iz in range(rbb['z'][0], rbb['z'][1] + 1):
        ciz = iz + Z_OFF - cz_min
        if ciz < 0 or ciz >= CZ:
            continue
        sl = raw_vol[iz, cy_min:cy_max + 1, cx_min:cx_max + 1]
        raw_canvas_full[ciz] = np.maximum(raw_canvas_full[ciz], (sl > 0).astype(np.uint8))
        raw_canvas_ds[ciz // ds] = np.maximum(raw_canvas_ds[ciz // ds], or_ds(sl, dCY, dCX))

    print("Building TIF canvas…")
    for iz in range(tbb['z'][0], tbb['z'][1] + 1):
        ciz = iz - cz_min
        if ciz < 0 or ciz >= CZ:
            continue
        sl = tif_vol[iz, ty0:ty1 + 1, tx0:tx1 + 1]
        h, w = min(sl.shape[0], CY), min(sl.shape[1], CX)
        tif_canvas_full[ciz, :h, :w] = np.maximum(
            tif_canvas_full[ciz, :h, :w], (sl[:h, :w] > 0).astype(np.uint8))
        tif_canvas_ds[ciz // ds] = np.maximum(tif_canvas_ds[ciz // ds], or_ds(sl, dCY, dCX))

    # Overlap volumes (full res for Dash, downsampled for PyVista)
    def overlap(r, t):
        return ((r > 0) & (t > 0)).astype(np.uint8), \
               ((r > 0) & (t == 0)).astype(np.uint8), \
               ((r == 0) & (t > 0)).astype(np.uint8)

    both_full, ro_full, to_full = overlap(raw_canvas_full, tif_canvas_full)
    both_ds,   ro_ds,   to_ds   = overlap(raw_canvas_ds,   tif_canvas_ds)

    # Overlap map (0=bg, 1=both, 2=raw_only, 3=tif_only) for Dash slices
    overlap_map = np.zeros_like(both_full)
    overlap_map[both_full > 0] = 1
    overlap_map[ro_full   > 0] = 2
    overlap_map[to_full   > 0] = 3

    total = int(both_full.sum() + ro_full.sum() + to_full.sum())
    agr   = 100 * both_full.sum() / total if total else 0

    print(f"\n  Agreement: {agr:.2f}%  "
          f"RAW only: {100*ro_full.sum()/total:.2f}%  "
          f"TIF only: {100*to_full.sum()/total:.2f}%")

    return dict(
        overlap_map=overlap_map,
        both_ds=both_ds, ro_ds=ro_ds, to_ds=to_ds,
        raw_profile=(raw_canvas_full > 0).sum(axis=(1, 2)),
        tif_profile=(tif_canvas_full > 0).sum(axis=(1, 2)),
        mip_z=overlap_map.max(axis=0),
        mip_x=overlap_map.max(axis=2),   # (CZ, CY) — collapse X → side view
        mip_y=overlap_map.max(axis=1),   # (CZ, CX) — collapse Y → front view
        CZ=CZ, CY=CY, CX=CX,
        dCZ=dCZ, dCY=dCY, dCX=dCX, ds=ds,
        cz_min=cz_min, cy_min=cy_min, cx_min=cx_min,
        Z_OFF=Z_OFF, TIF_XSHIFT=TIF_XSHIFT, TIF_YSHIFT=TIF_YSHIFT,
        rbb=rbb, tbb=tbb,
        agr=agr,
        both_n=int(both_full.sum()),
        ro_n=int(ro_full.sum()),
        to_n=int(to_full.sum()),
        raw_name=raw_path.name,
        tif_name=tif_path.name,
    )


# ── MARCHING CUBES ────────────────────────────────────────────────────────────

def vol_to_mesh(vol: np.ndarray, ds: int, smooth: int = 20) -> pv.PolyData | None:
    if not vol.any():
        return None
    padded = np.pad(vol, 1, constant_values=0)
    spacing = (float(ds), float(ds), float(ds))
    verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=spacing,
                                        allow_degenerate=False)
    verts -= np.array(spacing)
    n = len(faces)
    mesh = pv.PolyData(verts, np.hstack([np.full((n, 1), 3), faces]).ravel())
    return mesh.smooth(n_iter=smooth, relaxation_factor=0.1) if smooth > 0 else mesh


# ── DASH APP ──────────────────────────────────────────────────────────────────

COLORSCALE = [
    [0.00, 'rgb(7,9,15)'],   [0.33, 'rgb(7,9,15)'],
    [0.33, 'rgb(0,230,118)'], [0.66, 'rgb(0,230,118)'],
    [0.66, 'rgb(255,61,61)'], [0.99, 'rgb(255,61,61)'],
    [0.99, 'rgb(61,143,255)'],[1.00, 'rgb(61,143,255)'],
]
LAYOUT_BASE = dict(
    paper_bgcolor='#07090f', plot_bgcolor='#07090f',
    font=dict(family='JetBrains Mono, monospace', color='#d0dde8', size=11),
    margin=dict(l=55, r=15, t=36, b=36),
)

def build_dash_app(data: dict) -> dash.Dash:
    CZ = data['CZ']
    app = dash.Dash(__name__, title='Cluster Comparison')

    app.layout = html.Div(style={
        'backgroundColor': '#07090f', 'minHeight': '100vh',
        'fontFamily': 'JetBrains Mono, monospace', 'color': '#d0dde8', 'padding': '20px',
    }, children=[

        # Auto-refresh to pull PyVista state into Dash
        dcc.Interval(id='pv-poll', interval=500, n_intervals=0),

        # Header
        html.Div(style={'marginBottom': '16px'}, children=[
            html.H1('Cluster Comparison', style={
                'color': '#22aaff', 'fontSize': '18px', 'letterSpacing': '3px',
                'textTransform': 'uppercase', 'marginBottom': '4px',
            }),
            html.Div(f"{data['raw_name']}  vs  {data['tif_name']}",
                     style={'fontSize': '10px', 'color': '#4a6a8a', 'letterSpacing': '1px'}),
        ]),

        # Stats bar
        html.Div(style={
            'display': 'flex', 'gap': '28px', 'marginBottom': '18px',
            'padding': '12px 18px', 'background': 'rgba(10,14,26,0.9)',
            'borderRadius': '8px', 'border': '1px solid #1a2d4a',
        }, children=[
            _stat('Agreement',  f"{data['agr']:.2f}%",         '#00e676'),
            _stat('Both',       f"{data['both_n']:,}",         '#d0dde8'),
            _stat('RAW only',   f"{data['ro_n']:,}",           '#ff3d3d'),
            _stat('TIF only',   f"{data['to_n']:,}",           '#3d8fff'),
            _stat('Z_OFF',      str(data['Z_OFF']),            '#4a6a8a'),
            _stat('X_SHIFT',    str(data['TIF_XSHIFT']),       '#4a6a8a'),
            _stat('PyVista Z',  '—', '#4a6a8a', elem_id='pv-z-display'),
        ]),

        # Legend
        html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '16px',
                        'fontSize': '11px'}, children=[
            html.Span([html.Span('■ ', style={'color': '#00e676'}), 'Agreement']),
            html.Span([html.Span('■ ', style={'color': '#ff3d3d'}), 'RAW only']),
            html.Span([html.Span('■ ', style={'color': '#3d8fff'}), 'TIF only']),
        ]),

        # Controls row
        html.Div(style={
            'display': 'grid', 'gridTemplateColumns': '3fr 1fr',
            'gap': '16px', 'marginBottom': '16px',
            'padding': '12px 18px', 'background': 'rgba(10,14,26,0.9)',
            'borderRadius': '8px', 'border': '1px solid #1a2d4a',
        }, children=[
            html.Div([
                html.Div('Z SLICE  (→ moves cutting plane in PyVista 3D window)',
                         style={'fontSize': '10px', 'color': '#4a6a8a',
                                'letterSpacing': '1.5px', 'marginBottom': '8px'}),
                dcc.Slider(
                    id='z-slider', min=0, max=CZ - 1, step=1, value=CZ // 2,
                    marks={i: str(i) for i in range(0, CZ, max(1, CZ // 8))},
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    updatemode='drag',
                ),
            ]),
            html.Div([
                html.Div('SURFACE OPACITY',
                         style={'fontSize': '10px', 'color': '#4a6a8a',
                                'letterSpacing': '1.5px', 'marginBottom': '8px'}),
                dcc.Slider(
                    id='opacity-slider', min=0.0, max=1.0, step=0.05, value=0.25,
                    marks={0: '0', 0.5: '0.5', 1: '1'},
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    updatemode='drag',
                ),
            ]),
        ]),

        # Plots row 1: MIP + Profile
        html.Div(style={
            'display': 'grid', 'gridTemplateColumns': '1fr 1fr',
            'gap': '14px', 'marginBottom': '14px',
        }, children=[
            _card('Z MAX-INTENSITY PROJECTION', dcc.Graph(
                id='mip-graph',
                figure=_mip_fig(data['mip_z']),
                config={'displayModeBar': True},
            )),
            _card('MAX INTENSITY PROJECTION', html.Div([
                dcc.RadioItems(
                    id='mip-axis',
                    options=[
                        {'label': ' Z-MIP (top view)',   'value': 'z'},
                        {'label': ' X-MIP (side view)',  'value': 'x'},
                        {'label': ' Y-MIP (front view)', 'value': 'y'},
                    ],
                    value='z',
                    inline=True,
                    style={'fontSize': '11px', 'color': '#4a6a8a',
                           'marginBottom': '8px', 'gap': '16px'},
                    inputStyle={'marginRight': '5px'},
                ),
                dcc.Graph(id='mip-right-graph', config={'displayModeBar': True}),
            ])),
        ]),

        # Plots row 2: Orthogonal slices
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap': '14px'},
                 children=[
                     html.Div(id='slice-xy'),
                     html.Div(id='slice-xz'),
                     html.Div(id='slice-yz'),
                 ]),
    ])

    # ── Callbacks ─────────────────────────────────────────────────────────────

    @app.callback(
        Output('slice-xy', 'children'),
        Output('slice-xz', 'children'),
        Output('slice-yz', 'children'),
        Input('z-slider', 'value'),
    )
    def update_slices(z_idx):
        z_idx = int(z_idx)
        with STATE_LOCK:
            SHARED['z_idx'] = z_idx

        mid_y = data['CY'] // 2
        mid_x = data['CX'] // 2
        om    = data['overlap_map']
        cz    = z_idx + data['cz_min']

        return (
            _slice_panel(om[z_idx],      f'XY  (canvas z={cz})', 'X', 'Y'),
            _slice_panel(om[:, mid_y, :], f'XZ  (canvas y={mid_y + data["cy_min"]})', 'X', 'Z'),
            _slice_panel(om[:, :, mid_x], f'YZ  (canvas x={mid_x + data["cx_min"]})', 'Y', 'Z'),
        )

    @app.callback(
        Output('mip-right-graph', 'figure'),
        Input('mip-axis', 'value'),
    )
    def update_mip_right(axis):
        """Switch between Z, X and Y max-intensity projections."""
        if axis == 'z':
            mip  = data['mip_z']
            xlab, ylab = 'X (canvas)', 'Y (canvas)'
            title = 'Z-MIP — top view (collapse Z)'
            flip_y = True
        elif axis == 'x':
            mip  = data['mip_x']          # shape (CZ, CY)
            xlab, ylab = 'Y (canvas)', 'Z (canvas)'
            title = 'X-MIP — side view (collapse X)'
            flip_y = False
        else:  # y
            mip  = data['mip_y']          # shape (CZ, CX)
            xlab, ylab = 'X (canvas)', 'Z (canvas)'
            title = 'Y-MIP — front view (collapse Y)'
            flip_y = False

        fig = go.Figure(
            data=[go.Heatmap(z=mip, colorscale=COLORSCALE, zmin=0, zmax=3,
                             showscale=False,
                             hovertemplate=f'{xlab}=%{{x}}  {ylab}=%{{y}}<extra></extra>')],
            layout=go.Layout(
                **LAYOUT_BASE,
                title=dict(text=title, font=dict(size=11), x=0.01),
                xaxis=dict(title=xlab, color='#4a6a8a', gridcolor='#1a2d4a'),
                yaxis=dict(title=ylab, color='#4a6a8a', gridcolor='#1a2d4a',
                           autorange='reversed' if flip_y else True,
                           scaleanchor='x', scaleratio=1),
                height=420,
            )
        )
        return fig

    @app.callback(
        Output('pv-z-display', 'children'),
        Input('pv-poll', 'n_intervals'),
    )
    def poll_pyvista(_):
        with STATE_LOCK:
            pv_z = SHARED['pv_z_idx']
        return str(pv_z)

    @app.callback(
        Output('opacity-slider', 'value'),   # echo back (no-op) to trigger side-effect
        Input('opacity-slider', 'value'),
        prevent_initial_call=True,
    )
    def update_opacity(val):
        with STATE_LOCK:
            SHARED['both_opacity'] = float(val)
        return val

    return app


def _stat(label, value, color, elem_id=None):
    val_kwargs = {'style': {'color': color, 'fontSize': '16px', 'fontWeight': '600'}}
    if elem_id is not None:
        val_kwargs['id'] = elem_id
    return html.Div([
        html.Div(label, style={'color': '#4a6a8a', 'fontSize': '10px', 'letterSpacing': '2px'}),
        html.Div(value, **val_kwargs),
    ])


def _card(title, child):
    return html.Div(style={
        'background': 'rgba(10,14,26,0.9)', 'borderRadius': '8px',
        'border': '1px solid #1a2d4a', 'padding': '12px',
    }, children=[
        html.Div(title, style={'fontSize': '10px', 'color': '#4a6a8a',
                               'letterSpacing': '2px', 'marginBottom': '8px'}),
        child,
    ])


def _mip_fig(mip_z):
    return go.Figure(
        data=[go.Heatmap(z=mip_z, colorscale=COLORSCALE, zmin=0, zmax=3,
                         showscale=False)],
        layout=go.Layout(**LAYOUT_BASE,
                         xaxis=dict(title='X', color='#4a6a8a', gridcolor='#1a2d4a'),
                         yaxis=dict(title='Y', color='#4a6a8a', gridcolor='#1a2d4a',
                                    autorange='reversed',
                                    scaleanchor='x', scaleratio=1),
                         height=440),
    )


def _profile_fig(raw_p, tif_p):
    return go.Figure(
        data=[
            go.Scatter(y=raw_p.tolist(), name='RAW',
                       line=dict(color='#ff3d3d', width=1.5)),
            go.Scatter(y=tif_p.tolist(), name='TIF',
                       line=dict(color='#3d8fff', width=1.5)),
        ],
        layout=go.Layout(**LAYOUT_BASE,
                         xaxis=dict(title='Z (canvas)', color='#4a6a8a', gridcolor='#1a2d4a'),
                         yaxis=dict(title='Voxels',     color='#4a6a8a', gridcolor='#1a2d4a'),
                         legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1a2d4a',
                                     borderwidth=1),
                         height=340),
    )


def _slice_panel(sl, title, xlabel, ylabel):
    fig = go.Figure(
        data=[go.Heatmap(z=sl, colorscale=COLORSCALE, zmin=0, zmax=3, showscale=False)],
        layout=go.Layout(**LAYOUT_BASE,
                         title=dict(text=title, font=dict(size=10), x=0.01),
                         xaxis=dict(title=xlabel, color='#4a6a8a', gridcolor='#1a2d4a'),
                         yaxis=dict(title=ylabel, color='#4a6a8a', gridcolor='#1a2d4a',
                                    autorange='reversed',
                                    scaleanchor='x', scaleratio=1),
                         height=380),
    )
    return _card(title, dcc.Graph(figure=fig, config={'displayModeBar': False}))


# ── PYVISTA VIEWER ────────────────────────────────────────────────────────────

def run_pyvista(data: dict):
    """
    Build and show the PyVista 3D window.
    Must run on the main thread (OpenGL requirement).

    Interactions driven by SHARED state:
      SHARED['z_idx']        → position of the cutting plane (set by Dash Z-slider)
      SHARED['both_opacity'] → opacity of the agreement surface (set by Dash slider)
    """
    print("\nExtracting surfaces (marching cubes)…")
    ds = data['ds']
    mesh_both = vol_to_mesh(data['both_ds'], ds)
    mesh_ro   = vol_to_mesh(data['ro_ds'],   ds)
    mesh_to   = vol_to_mesh(data['to_ds'],   ds)

    for name, m in [('both', mesh_both), ('raw_only', mesh_ro), ('tif_only', mesh_to)]:
        if m:
            print(f"  {name}: {m.n_points:,} verts, {m.n_cells:,} faces")

    pv.set_plot_theme('dark')
    pl = pv.Plotter(window_size=(1100, 800),
                    title='Cluster Comparison — 3D  (linked to Dash)')

    # Add static meshes and keep actors for dynamic updates
    actor_both = pl.add_mesh(mesh_both, color='#00e676', opacity=0.25,
                             smooth_shading=True) if mesh_both else None
    if mesh_ro:
        pl.add_mesh(mesh_ro, color='#ff3d3d', opacity=1.0, smooth_shading=True)
    if mesh_to:
        pl.add_mesh(mesh_to, color='#3d8fff', opacity=1.0, smooth_shading=True)

    def timer_callback():
        """Called every TIMER_MS ms — syncs opacity from Dash slider."""
        with STATE_LOCK:
            both_opacity = float(SHARED['both_opacity'])
        if actor_both is not None:
            actor_both.GetProperty().SetOpacity(both_opacity)
            pl.render()

    pl.add_legend(
        labels=[('Agreement', '#00e676'), ('RAW only', '#ff3d3d'), ('TIF only', '#3d8fff')],
        bcolor='#07090f', face='circle', size=(0.22, 0.14),
    )
    pl.add_text(
        f"{data['raw_name']}  vs  {data['tif_name']}\n"
        f"Agreement: {data['agr']:.2f}%   DS={ds}\n"
        f"",
        position='upper_left', font_size=9, color='#d0dde8', font='courier',
    )
    pl.camera_position = 'iso'

    # Use VTK-level timer observer — more reliable than PyVista's add_timer_event wrapper.
    # CreateRepeatingTimer fires every TIMER_MS ms on the VTK interactor event loop.
    iren = pl.iren.interactor
    iren.AddObserver('TimerEvent', lambda obj, evt: timer_callback())
    iren.CreateRepeatingTimer(TIMER_MS)

    with STATE_LOCK:
        SHARED['pv_ready'] = True

    print("PyVista window open. Control the Z slice from the Dash dashboard.")
    pl.show()


# ── DASH THREAD ───────────────────────────────────────────────────────────────

def run_dash(app: dash.Dash):
    """Run Dash in a daemon background thread."""
    # Wait briefly so PyVista can claim the main thread first
    time.sleep(1.5)
    with STATE_LOCK:
        SHARED['dash_ready'] = True
    print(f"\nDash running at http://127.0.0.1:{DASH_PORT}")
    app.run(debug=False, port=DASH_PORT, use_reloader=False)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("Cluster Comparison — Dash + PyVista (combined)")
    print("=" * 64)

    if not RAW_PATH.exists():
        sys.exit(f"RAW file not found: {RAW_PATH}")
    if not TIF_PATH.exists():
        sys.exit(f"TIF file not found: {TIF_PATH}")

    # ── Load data (once, shared by both views) ────────────────────────────────
    data = load_and_align(RAW_PATH, TIF_PATH, DS)

    # Initialise shared Z to the middle of the canvas
    with STATE_LOCK:
        SHARED['z_idx'] = data['CZ'] // 2

    # ── Build Dash app ────────────────────────────────────────────────────────
    app = build_dash_app(data)

    # ── Start Dash in background thread ──────────────────────────────────────
    dash_thread = threading.Thread(target=run_dash, args=(app,), daemon=True)
    dash_thread.start()

    # Open browser after a short delay
    def _open_browser():
        time.sleep(3)
        webbrowser.open(f"http://127.0.0.1:{DASH_PORT}")
    threading.Thread(target=_open_browser, daemon=True).start()

    # ── Run PyVista on main thread (OpenGL requirement) ───────────────────────
    run_pyvista(data)

    print("\nDone.")


if __name__ == '__main__':
    main()
