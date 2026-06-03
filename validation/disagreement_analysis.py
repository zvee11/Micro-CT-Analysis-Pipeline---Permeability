"""
disagreement_analysis.py
------------------------
Dash dashboard + PyVista 3D viewer for analysing disagreement voxels
between Avizo RAW and CCA TIFF segmentations.

Layout
------
  LEFT sidebar  : Full cluster MIP overview (Z/X/Y switcher) with orange
                  bounding box highlight for the selected component.
  RIGHT column  : Component table → on click, shows:
                    • Stats card
                    • Centroid slice: the single Z-slice through the
                      component centre, zoomed to bbox, colour-coded
                    • Per-Z voxel bar chart: how RAW-only / TIF-only /
                      agreement evolves through the component depth

  PyVista 3D    : Shows a LOCAL cutout of the selected component.
                  Agreement (green, semi-transparent) + RAW-only (red) +
                  TIF-only (blue) — ALL in the same local coordinate system
                  so they are correctly attached to each other.

Coordinate fix
--------------
  Diff meshes (RAW-only, TIF-only) are always rendered at FULL RESOLUTION
  since their exact structure is the primary thing to inspect.
  Agreement surface is downsampled (DS=4 overview, DS=2 cutout) for performance.
  Coord consistency restored by scaling downsampled mesh vertices back to full-res
  canvas space before adding to the scene.

Usage
-----
    pip install dash plotly pyvista scikit-image pillow numpy scipy
    python disagreement_analysis.py
"""

from __future__ import annotations

import sys, threading, time, webbrowser
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label as scipy_label

try:
    import dash
    from dash import dcc, html, Input, Output, ctx
    import plotly.graph_objects as go
    import pyvista as pv
    from skimage.measure import marching_cubes
except ImportError as e:
    sys.exit(f"Missing: {e}\n\npip install dash plotly pyvista scikit-image pillow numpy scipy")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
RAW_PATH   = Path("8_2_biggestcluster.view.raw")
TIF_PATH   = Path("cluster_01_mask_26N.tiff")
MIN_VOXELS = 20
DASH_PORT  = 8050
TIMER_MS   = 200

# ── SHARED STATE ──────────────────────────────────────────────────────────────
LOCK   = threading.Lock()
SHARED = {'selected_component': None}

# ── STYLE ─────────────────────────────────────────────────────────────────────
BG     = '#07090f'
PANEL  = 'rgba(10,14,26,0.92)'
BORDER = '#1a2d4a'
TEXT   = '#d0dde8'
MUTED  = '#4a6a8a'
MONO   = 'JetBrains Mono, monospace'
GREEN  = '#00e676'
RED    = '#ff3d3d'
BLUE   = '#3d8fff'
ORANGE = '#ffaa00'

TYPE_COLOURS = {
    'interleaved_boundary': '#ffaa00',
    'mixed_boundary':       '#ff66ff',
    'raw_dominant':         '#ff3d3d',
    'tif_dominant':         '#3d8fff',
    'boundary_strip':       '#ffffff',
}
TYPE_DESC = {
    'interleaved_boundary': 'RAW and TIF alternate along the cluster boundary — typical of sub-voxel segmentation differences.',
    'mixed_boundary':       'Mixed RAW and TIF voxels at the surface — boundary placement differs between methods.',
    'raw_dominant':         'Mostly RAW-only voxels — Avizo included these; CCA did not.',
    'tif_dominant':         'Mostly TIF-only voxels — CCA included these; Avizo did not.',
    'boundary_strip':       'Long thin strip along one axis — likely a systematic edge effect.',
}

OVERLAP_CS = [
    [0.00,'rgb(7,9,15)'],    [0.33,'rgb(7,9,15)'],
    [0.33,'rgb(0,230,118)'], [0.66,'rgb(0,230,118)'],
    [0.66,'rgb(255,61,61)'], [0.99,'rgb(255,61,61)'],
    [0.99,'rgb(61,143,255)'],[1.00,'rgb(61,143,255)'],
]
LAYOUT_BASE = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(family=MONO, color=TEXT, size=10),
    margin=dict(l=45, r=8, t=28, b=36),
)

# ── LOADERS ───────────────────────────────────────────────────────────────────

def load_tiff(path):
    img = Image.open(path)
    frames = []
    for i in range(img.n_frames):
        img.seek(i); frames.append(np.array(img))
    return np.stack(frames, axis=0)


def build_canvases(raw_path, tif_path):
    print("Loading TIFF..."); tif_vol = load_tiff(tif_path)
    print("Loading RAW...")
    ny,nx = tif_vol.shape[1], tif_vol.shape[2]
    nz    = raw_path.stat().st_size // 2 // (ny*nx)
    raw_vol = np.memmap(str(raw_path), dtype=np.uint16, mode='r', shape=(nz,ny,nx))

    print("Bounding boxes...")
    ri = np.where(raw_vol>0)
    ti = np.where(tif_vol>0)
    rbb = {a:(int(ri[i].min()),int(ri[i].max())) for i,a in enumerate(['z','y','x'])}
    tbb = {a:(int(ti[i].min()),int(ti[i].max())) for i,a in enumerate(['z','y','x'])}

    Z_OFF      = tbb['z'][0]-rbb['z'][0]
    TIF_XSHIFT = rbb['x'][0]-tbb['x'][0]
    TIF_YSHIFT = rbb['y'][0]-tbb['y'][0]
    print(f"  Z_OFF={Z_OFF}  TIF_XSHIFT={TIF_XSHIFT}  TIF_YSHIFT={TIF_YSHIFT}")

    cz_min=tbb['z'][0]; cz_max=max(tbb['z'][1],rbb['z'][1]+Z_OFF)
    cy_min=rbb['y'][0]; cy_max=max(rbb['y'][1],tbb['y'][1]+TIF_YSHIFT)
    cx_min=rbb['x'][0]; cx_max=max(rbb['x'][1],tbb['x'][1]+TIF_XSHIFT)
    CZ,CY,CX = cz_max-cz_min+1, cy_max-cy_min+1, cx_max-cx_min+1
    print(f"  Canvas: {CZ}×{CY}×{CX}")

    raw_c = np.zeros((CZ,CY,CX),dtype=np.uint8)
    tif_c = np.zeros((CZ,CY,CX),dtype=np.uint8)

    print("Building RAW canvas...")
    for iz in range(rbb['z'][0],rbb['z'][1]+1):
        ciz=iz+Z_OFF-cz_min
        if 0<=ciz<CZ:
            sl=(raw_vol[iz,cy_min:cy_max+1,cx_min:cx_max+1]>0).astype(np.uint8)
            raw_c[ciz]=np.maximum(raw_c[ciz],sl)

    print("Building TIF canvas...")
    ty0=cy_min-TIF_YSHIFT; ty1=cy_max-TIF_YSHIFT
    tx0=cx_min-TIF_XSHIFT; tx1=cx_max-TIF_XSHIFT
    for iz in range(tbb['z'][0],tbb['z'][1]+1):
        ciz=iz-cz_min
        if 0<=ciz<CZ:
            sl=(tif_vol[iz,ty0:ty1+1,tx0:tx1+1]>0).astype(np.uint8)
            h,w=min(sl.shape[0],CY),min(sl.shape[1],CX)
            tif_c[ciz,:h,:w]=np.maximum(tif_c[ciz,:h,:w],sl[:h,:w])

    return raw_c,tif_c,dict(CZ=CZ,CY=CY,CX=CX,cz_min=cz_min,cy_min=cy_min,cx_min=cx_min,
                             Z_OFF=Z_OFF,TIF_XSHIFT=TIF_XSHIFT,TIF_YSHIFT=TIF_YSHIFT)


def find_components(raw_c,tif_c,min_vox=20):
    only_raw=((raw_c>0)&(tif_c==0))
    only_tif=((raw_c==0)&(tif_c>0))
    di=np.where(only_raw|only_tif)
    if len(di[0])==0: return [],[],only_raw,only_tif

    mg=3
    z0d=max(0,int(di[0].min())-mg); z1d=min(raw_c.shape[0],int(di[0].max())+mg+1)
    y0d=max(0,int(di[1].min())-mg); y1d=min(raw_c.shape[1],int(di[1].max())+mg+1)
    x0d=max(0,int(di[2].min())-mg); x1d=min(raw_c.shape[2],int(di[2].max())+mg+1)

    sub=((only_raw|only_tif)[z0d:z1d,y0d:y1d,x0d:x1d]).astype(np.uint8)
    lsub,n=scipy_label(sub,structure=np.ones((3,3,3),dtype=bool))
    print(f"Found {n} raw components")

    lf=np.zeros(raw_c.shape,dtype=np.uint16)
    lf[z0d:z1d,y0d:y1d,x0d:x1d]=lsub

    comps=[]
    for cid in range(1,n+1):
        m=lsub==cid; nv=int(m.sum())
        if nv<min_vox: continue
        zs,ys,xs=np.where(m)
        zg,yg,xg=zs+z0d,ys+y0d,xs+x0d
        z0c,z1c=int(zg.min()),int(zg.max())
        y0c,y1c=int(yg.min()),int(yg.max())
        x0c,x1c=int(xg.min()),int(xg.max())
        ez,ey,ex=z1c-z0c+1,y1c-y0c+1,x1c-x0c+1
        mf=lf==cid
        nr=int(only_raw[mf].sum()); nt=int(only_tif[mf].sum())
        rf=nr/max(nv,1)
        fill=nv/max(ez*ey*ex,1)
        elong=max(ez,ey,ex)/max(min(ez,ey,ex),1)
        if elong>8 and fill<0.25:       stype='boundary_strip'
        elif abs(rf-0.5)<0.2 and elong>3: stype='interleaved_boundary'
        elif rf>0.75:                   stype='raw_dominant'
        elif rf<0.25:                   stype='tif_dominant'
        else:                           stype='mixed_boundary'
        comps.append(dict(id=cid,n_vox=nv,n_raw=nr,n_tif=nt,
            z0=z0c,z1=z1c,y0=y0c,y1=y1c,x0=x0c,x1=x1c,
            ext_z=ez,ext_y=ey,ext_x=ex,
            elongation=round(elong,2),fill_ratio=round(fill,3),
            shape_type=stype,raw_frac=round(rf,3),
            centroid_z=round(float(zg.mean()),1),
            centroid_y=round(float(yg.mean()),1),
            centroid_x=round(float(xg.mean()),1)))
    comps.sort(key=lambda c:c['n_vox'],reverse=True)
    print(f"Major components (>={min_vox}): {len(comps)}")
    return comps,lf,only_raw,only_tif


# ── DASH ──────────────────────────────────────────────────────────────────────

def _card(title,child):
    return html.Div(style={
        'background':PANEL,'borderRadius':'8px','border':f'1px solid {BORDER}',
        'padding':'12px','marginBottom':'12px',
    },children=[
        html.Div(title,style={'fontSize':'9px','color':MUTED,'letterSpacing':'2px','marginBottom':'8px'}),
        child,
    ])


def _hmap(z,title='',xl='',yl='',flip_y=True,h=320):
    return go.Figure(
        data=[go.Heatmap(z=z,colorscale=OVERLAP_CS,zmin=0,zmax=3,showscale=False)],
        layout=go.Layout(**LAYOUT_BASE,
            title=dict(text=title,font=dict(size=9),x=0.01),
            xaxis=dict(title=xl,color=MUTED,gridcolor=BORDER),
            yaxis=dict(title=yl,color=MUTED,gridcolor=BORDER,
                       autorange='reversed' if flip_y else True,
                       scaleanchor='x',scaleratio=1),
            height=h))


def build_dash(raw_c,tif_c,comps,cinfo):
    CZ,CY,CX=cinfo['CZ'],cinfo['CY'],cinfo['CX']
    both  =((raw_c>0)&(tif_c>0)).astype(np.uint8)
    or_   =((raw_c>0)&(tif_c==0)).astype(np.uint8)
    ot_   =((raw_c==0)&(tif_c>0)).astype(np.uint8)
    ovmap =np.zeros((CZ,CY,CX),dtype=np.uint8)
    ovmap[both>0]=1; ovmap[or_>0]=2; ovmap[ot_>0]=3

    total=int(both.sum()+or_.sum()+ot_.sum())
    agr=100*both.sum()/total if total else 0

    app=dash.Dash(__name__,title='Disagreement Analysis')

    # Table rows
    rows=[]
    for c in comps:
        col=TYPE_COLOURS.get(c['shape_type'],'#888')
        rows.append(html.Tr(
            id={'type':'row','index':c['id']},
            style={'cursor':'pointer','borderBottom':f'1px solid {BORDER}'},
            children=[
                html.Td(str(c['id']),style={'padding':'5px 7px','color':MUTED}),
                html.Td(str(c['n_vox']),style={'padding':'5px 7px'}),
                html.Td(str(c['n_raw']),style={'padding':'5px 7px','color':RED}),
                html.Td(str(c['n_tif']),style={'padding':'5px 7px','color':BLUE}),
                html.Td(c['shape_type'].replace('_',' '),
                        style={'padding':'5px 7px','color':col,'fontSize':'8px'}),
                html.Td(f"Z {c['z0']}–{c['z1']}",
                        style={'padding':'5px 7px','fontSize':'8px','color':MUTED}),
            ]))

    app.layout=html.Div(style={
        'backgroundColor':BG,'minHeight':'100vh','fontFamily':MONO,'color':TEXT,'padding':'16px',
    },children=[
        dcc.Store(id='sel-id',data=None),

        # ── HEADER ────────────────────────────────────────────────────────────
        html.Div(style={'marginBottom':'12px'},children=[
            html.H1('Disagreement Analysis',style={'color':'#22aaff','fontSize':'16px',
                    'letterSpacing':'3px','textTransform':'uppercase','marginBottom':'3px'}),
            html.Div(f'{RAW_PATH.name}  vs  {TIF_PATH.name}',
                     style={'fontSize':'9px','color':MUTED,'letterSpacing':'1px'}),
        ]),

        # ── STATS BAR ─────────────────────────────────────────────────────────
        html.Div(style={'display':'flex','gap':'24px','marginBottom':'12px',
                        'padding':'8px 14px','background':PANEL,
                        'borderRadius':'8px','border':f'1px solid {BORDER}'},children=[
            html.Div([html.Div('AGREEMENT',style={'color':MUTED,'fontSize':'8px','letterSpacing':'2px'}),
                      html.Div(f'{agr:.4f}%',style={'color':GREEN,'fontSize':'15px','fontWeight':'600'})]),
            html.Div([html.Div('DISAGREEMENT VOXELS',style={'color':MUTED,'fontSize':'8px','letterSpacing':'2px'}),
                      html.Div(f"{or_.sum()+ot_.sum():,}",style={'fontSize':'15px','fontWeight':'600'})]),
            html.Div([html.Div('COMPONENTS SHOWN',style={'color':MUTED,'fontSize':'8px','letterSpacing':'2px'}),
                      html.Div(str(len(comps)),style={'fontSize':'15px','fontWeight':'600'})]),
            html.Div([html.Div('SELECTED →',style={'color':MUTED,'fontSize':'8px','letterSpacing':'2px'}),
                      html.Div('none',id='sel-label',style={'color':ORANGE,'fontSize':'15px','fontWeight':'600'})]),
        ]),

        # ── COLOUR LEGEND ─────────────────────────────────────────────────────
        html.Div(style={'display':'flex','gap':'14px','marginBottom':'12px',
                        'fontSize':'9px','flexWrap':'wrap'},children=[
            html.Span([html.Span('■ ',style={'color':GREEN}),'Both (agreement)']),
            html.Span([html.Span('■ ',style={'color':RED}),'RAW only (Avizo)']),
            html.Span([html.Span('■ ',style={'color':BLUE}),'TIF only (CCA)']),
            html.Span([html.Span('■ ',style={'color':'#07090f','border':f'1px solid {BORDER}'}),' Background/rock']),
        ]),

        # ── MAIN BODY: LEFT sidebar + RIGHT content ────────────────────────────
        html.Div(style={'display':'grid','gridTemplateColumns':'280px 1fr','gap':'12px'},
        children=[

            # ── LEFT: Full cluster MIP ─────────────────────────────────────────
            html.Div(children=[
                _card('FULL CLUSTER — MAX INTENSITY PROJECTION', html.Div([
                    html.Div('Switch projection axis:',
                             style={'fontSize':'9px','color':MUTED,'marginBottom':'5px'}),
                    dcc.RadioItems(id='mip-axis',value='z',inline=False,
                        options=[
                            {'label':' Z-MIP  (top view — collapses depth)',    'value':'z'},
                            {'label':' X-MIP  (side view — collapses left/right)','value':'x'},
                            {'label':' Y-MIP  (front view — collapses front/back)','value':'y'},
                        ],
                        style={'fontSize':'9px','color':MUTED,'lineHeight':'2'},
                        inputStyle={'marginRight':'5px'},
                    ),
                    html.Div(id='mip-axis-desc',
                             style={'fontSize':'8px','color':MUTED,'margin':'5px 0 8px',
                                    'lineHeight':'1.6','fontStyle':'italic'}),
                    dcc.Graph(id='mip-graph',config={'displayModeBar':True,
                        'modeBarButtonsToRemove':['select2d','lasso2d']}),
                ])),
            ]),

            # ── RIGHT: Table + detail ──────────────────────────────────────────
            html.Div(children=[

                # Component table
                _card('DISAGREEMENT COMPONENTS  (click a row to inspect)',
                    html.Div(style={'overflowY':'auto','maxHeight':'260px'},children=[
                        html.Table(style={'width':'100%','borderCollapse':'collapse'},children=[
                            html.Thead(html.Tr([
                                html.Th(h,style={'padding':'5px 7px','color':MUTED,'fontSize':'8px',
                                                 'letterSpacing':'1px','textAlign':'left',
                                                 'borderBottom':f'1px solid {BORDER}'})
                                for h in ['ID','Vox','RAW','TIF','Type','Z range']
                            ])),
                            html.Tbody(rows),
                        ])
                    ])
                ),

                # Detail panels (shown after selection)
                html.Div(id='detail-area'),
            ]),
        ]),
    ])

    # ── CALLBACKS ─────────────────────────────────────────────────────────────

    @app.callback(
        Output('sel-id','data'),
        Output('sel-label','children'),
        [Input({'type':'row','index':c['id']},'n_clicks') for c in comps],
        prevent_initial_call=True,
    )
    def select(*_):
        t=ctx.triggered_id
        if t is None: return None,'none'
        cid=t['index']
        with LOCK: SHARED['selected_component']=cid
        return cid,f'Component {cid}'

    @app.callback(
        Output('mip-graph','figure'),
        Output('mip-axis-desc','children'),
        Input('mip-axis','value'),
        Input('sel-id','data'),
    )
    def upd_mip(axis,sel_id):
        descs={
            'z':'Each pixel = was this (X,Y) position ever occupied across ALL depth slices?',
            'x':'Each pixel = was this (Y,Z) position occupied in ANY left-right column?',
            'y':'Each pixel = was this (X,Z) position occupied in ANY front-back row?',
        }
        if axis=='z':
            mip=ovmap.max(axis=0); xl,yl,fy='X (canvas)','Y (canvas)',True; t='Z-MIP — top view'
        elif axis=='x':
            mip=ovmap.max(axis=2); xl,yl,fy='Y (canvas)','Z (canvas)',False; t='X-MIP — side view'
        else:
            mip=ovmap.max(axis=1); xl,yl,fy='X (canvas)','Z (canvas)',False; t='Y-MIP — front view'

        fig=_hmap(mip,title=t,xl=xl,yl=yl,flip_y=fy,h=480)

        if sel_id is not None:
            c=next((x for x in comps if x['id']==sel_id),None)
            if c:
                if axis=='z':   x0r,x1r,y0r,y1r=c['x0'],c['x1'],c['y0'],c['y1']
                elif axis=='x': x0r,x1r,y0r,y1r=c['y0'],c['y1'],c['z0'],c['z1']
                else:           x0r,x1r,y0r,y1r=c['x0'],c['x1'],c['z0'],c['z1']
                fig.add_shape(type='rect',x0=x0r,x1=x1r,y0=y0r,y1=y1r,
                              line=dict(color=ORANGE,width=2))
                fig.add_annotation(x=(x0r+x1r)/2,y=y0r,text=f'C{sel_id}',
                                   showarrow=False,font=dict(color=ORANGE,size=9),yanchor='bottom')
        return fig, descs[axis]

    @app.callback(
        Output('detail-area','children'),
        Input('sel-id','data'),
    )
    def upd_detail(sel_id):
        if sel_id is None:
            return html.Div('← Click a component in the table to inspect it.',
                            style={'color':MUTED,'fontSize':'10px','padding':'20px'})

        c=next((x for x in comps if x['id']==sel_id),None)
        if c is None: return []

        mg=12
        CZ2,CY2,CX2=cinfo['CZ'],cinfo['CY'],cinfo['CX']
        z0,z1=max(0,c['z0']-mg),min(CZ2,c['z1']+mg+1)
        y0,y1=max(0,c['y0']-mg),min(CY2,c['y1']+mg+1)
        x0,x1=max(0,c['x0']-mg),min(CX2,c['x1']+mg+1)
        local=ovmap[z0:z1,y0:y1,x0:x1]

        col=TYPE_COLOURS.get(c['shape_type'],'#888')
        desc=TYPE_DESC.get(c['shape_type'],'')

        # ── Centroid slice (the single Z-slice through the component centre) ──
        cz_local=int(round(c['centroid_z']))-z0
        cz_local=max(0,min(cz_local,local.shape[0]-1))
        cent_slice=local[cz_local,:,:]

        cent_fig=_hmap(cent_slice,
            title=f'Z-slice through component centre (canvas z={int(round(c["centroid_z"]))})',
            xl='X (canvas)',yl='Y (canvas)',flip_y=True,h=280)
        # Mark the component bbox on the centroid slice
        cent_fig.add_shape(type='rect',
            x0=c['x0']-x0,x1=c['x1']-x0,y0=c['y0']-y0,y1=c['y1']-y0,
            line=dict(color=ORANGE,width=1.5,dash='dot'))

        # ── Per-Z voxel bar chart ──────────────────────────────────────────────
        # For each Z slice within the component bbox, count RAW-only / TIF-only / both
        z_indices=list(range(c['z0'],c['z1']+1))
        bar_both=[]; bar_ro=[]; bar_to=[]
        for zi in z_indices:
            sl=ovmap[zi,c['y0']:c['y1']+1,c['x0']:c['x1']+1]
            bar_both.append(int((sl==1).sum()))
            bar_ro.append(int((sl==2).sum()))
            bar_to.append(int((sl==3).sum()))

        bar_fig=go.Figure(data=[
            go.Bar(name='Agreement (both)',x=z_indices,y=bar_both,
                   marker_color=GREEN,opacity=0.7),
            go.Bar(name='RAW only (Avizo)',x=z_indices,y=bar_ro,marker_color=RED),
            go.Bar(name='TIF only (CCA)', x=z_indices,y=bar_to,marker_color=BLUE),
        ],layout=go.Layout(**LAYOUT_BASE,
            title=dict(text='Voxels per Z-slice within component bbox',font=dict(size=9),x=0.01),
            xaxis=dict(title='Z slice (canvas)',color=MUTED,gridcolor=BORDER),
            yaxis=dict(title='Voxel count',color=MUTED,gridcolor=BORDER),
            barmode='stack',
            legend=dict(bgcolor='rgba(0,0,0,0)',bordercolor=BORDER,borderwidth=1,
                        font=dict(size=8)),
            height=240,
        ))

        return html.Div(style={'display':'grid',
                               'gridTemplateColumns':'220px 1fr 1fr',
                               'gap':'10px'},children=[
            # Stats + explanation
            _card(f'COMPONENT {sel_id} — STATS', html.Div([
                html.Div(style={'marginBottom':'10px',
                                'padding':'8px','borderRadius':'6px',
                                'background':f'rgba({",".join(str(int(col.lstrip("#")[i:i+2],16)) for i in (0,2,4))},0.12)',
                                'border':f'1px solid {col}'},children=[
                    html.Div(c['shape_type'].replace('_',' ').upper(),
                             style={'color':col,'fontSize':'8px','letterSpacing':'1.5px','marginBottom':'4px'}),
                    html.Div(desc,style={'fontSize':'9px','color':TEXT,'lineHeight':'1.6'}),
                ]),
                html.Div(style={'fontSize':'9px','lineHeight':'2.2','color':TEXT},children=[
                    html.Div([html.Span('Total voxels: ',style={'color':MUTED}), str(c['n_vox'])]),
                    html.Div([html.Span('RAW only: ',style={'color':MUTED}),
                              html.Span(str(c['n_raw']),style={'color':RED})]),
                    html.Div([html.Span('TIF only: ',style={'color':MUTED}),
                              html.Span(str(c['n_tif']),style={'color':BLUE})]),
                    html.Div([html.Span('Elongation: ',style={'color':MUTED}), f"{c['elongation']:.1f}×"]),
                    html.Div([html.Span('Fill ratio: ',style={'color':MUTED}), f"{c['fill_ratio']:.3f}"]),
                    html.Div([html.Span('Z span: ',style={'color':MUTED}), f"{c['z0']}–{c['z1']} ({c['ext_z']} slices)"]),
                    html.Div([html.Span('Y span: ',style={'color':MUTED}), f"{c['y0']}–{c['y1']} ({c['ext_y']} px)"]),
                    html.Div([html.Span('X span: ',style={'color':MUTED}), f"{c['x0']}–{c['x1']} ({c['ext_x']} px)"]),
                ]),
            ])),

            # Centroid slice
            _card(f'SLICE THROUGH COMPONENT CENTRE',
                  html.Div([
                      html.Div('The cross-section at the component\'s central Z-slice. '
                               'Orange dashed box = component bbox. '
                               'Green = both methods agree. Red = Avizo only. Blue = CCA only.',
                               style={'fontSize':'8px','color':MUTED,'marginBottom':'6px','lineHeight':'1.6'}),
                      dcc.Graph(figure=cent_fig,config={'displayModeBar':False}),
                  ])),

            # Per-Z bar chart
            _card('VOXEL COUNT THROUGH COMPONENT DEPTH',
                  html.Div([
                      html.Div('How the disagreement evolves slice by slice through the component. '
                               'Large green bars = mostly agreement. '
                               'Red/blue bars = where the two methods diverge.',
                               style={'fontSize':'8px','color':MUTED,'marginBottom':'6px','lineHeight':'1.6'}),
                      dcc.Graph(figure=bar_fig,config={'displayModeBar':False}),
                  ])),
        ])

    return app


# ── PYVISTA ───────────────────────────────────────────────────────────────────

def _mesh(vol,smooth=15):
    if not vol.any(): return None
    pad=np.pad(vol,1,constant_values=0)
    try: v,f,_,_=marching_cubes(pad,level=0.5,allow_degenerate=False)
    except: return None
    v-=1.0  # undo padding offset → local coords start at 0
    n=len(f)
    m=pv.PolyData(v,np.hstack([np.full((n,1),3),f]).ravel())
    return m.smooth(n_iter=smooth) if smooth>0 and m.n_points>0 else m


def run_pyvista(raw_c,tif_c,comps,cinfo):
    both_v=((raw_c>0)&(tif_c>0)).astype(np.uint8)
    ro_v  =((raw_c>0)&(tif_c==0)).astype(np.uint8)
    to_v  =((raw_c==0)&(tif_c>0)).astype(np.uint8)

    pv.set_plot_theme('dark')
    pl=pv.Plotter(window_size=(1000,800),
                  title='3D Disagreement Viewer — select component in Dash')

    print("Building overview surfaces...")
    # Agreement (large) downsampled for performance — we don't need its fine detail.
    # Diff meshes (small, ~2000 voxels total) rendered at FULL RESOLUTION — this is
    # the most important part and must not lose any structural detail.
    # Coord consistency: all three are in the same canvas coordinate space because
    # we pass raw arrays directly (no local-box offset needed for overview).
    DS_AGREE = 4   # downsample agreement surface only
    mb=_mesh(both_v[::DS_AGREE,::DS_AGREE,::DS_AGREE],smooth=20)
    # Scale agreement mesh vertices back to full-res canvas coords
    if mb: mb.points *= DS_AGREE

    mr=_mesh(ro_v, smooth=5)   # full resolution
    mt=_mesh(to_v, smooth=5)   # full resolution

    if mb: pl.add_mesh(mb,color=GREEN, opacity=0.12,smooth_shading=True)
    if mr: pl.add_mesh(mr,color=RED,   opacity=1.0, smooth_shading=True)
    if mt: pl.add_mesh(mt,color=BLUE,  opacity=1.0, smooth_shading=True)

    pl.add_legend(labels=[('Agreement',GREEN),('RAW only',RED),('TIF only',BLUE)],
                  bcolor='#07090f',face='circle',size=(0.22,0.10))
    pl.add_text('Select a component in Dash to zoom in here.',
                position='upper_left',font_size=9,color=TEXT,font='courier')

    comp_actors=[]; last_id=[None]

    def timer_cb():
        with LOCK: sel=SHARED['selected_component']
        if sel==last_id[0]: return
        last_id[0]=sel

        for a in comp_actors:
            try: pl.remove_actor(a,render=False)
            except: pass
        comp_actors.clear()

        if sel is None: pl.render(); return
        c=next((x for x in comps if x['id']==sel),None)
        if c is None: pl.render(); return

        # ── LOCAL cutout — all meshes in identical LOCAL coordinate space ─────
        mg=15
        CZ2,CY2,CX2=cinfo['CZ'],cinfo['CY'],cinfo['CX']
        z0,z1=max(0,c['z0']-mg),min(CZ2,c['z1']+mg+1)
        y0,y1=max(0,c['y0']-mg),min(CY2,c['y1']+mg+1)
        x0,x1=max(0,c['x0']-mg),min(CX2,c['x1']+mg+1)

        loc_both=both_v[z0:z1,y0:y1,x0:x1]
        loc_ro  =ro_v  [z0:z1,y0:y1,x0:x1]
        loc_to  =to_v  [z0:z1,y0:y1,x0:x1]

        # Diff meshes at FULL RESOLUTION — these are small (~dozens to hundreds of voxels)
        # and their exact shape is the most important thing to preserve.
        # Agreement surface can be lightly smoothed; diffs should not be over-smoothed.
        DS_LOC = 2  # mild downsample for agreement only, keeps coords consistent
        loc_both_ds = loc_both[::DS_LOC, ::DS_LOC, ::DS_LOC]
        mb2 = _mesh(loc_both_ds, smooth=8)
        if mb2: mb2.points *= DS_LOC   # scale back to local full-res coords

        mr2 = _mesh(loc_ro, smooth=3)   # full resolution, minimal smoothing
        mt2 = _mesh(loc_to, smooth=3)   # full resolution, minimal smoothing

        for mesh,col,op in [(mb2,GREEN,0.3),(mr2,RED,1.0),(mt2,BLUE,1.0)]:
            if mesh is None: continue
            a=pl.add_mesh(mesh,color=col,opacity=op,smooth_shading=True,render=False)
            comp_actors.append(a)

        # Wireframe bbox in local coords
        lz,ly,lx=z1-z0,y1-y0,x1-x0
        # Component bbox relative to cutout origin
        cz0_l,cz1_l=c['z0']-z0,c['z1']-z0
        cy0_l,cy1_l=c['y0']-y0,c['y1']-y0
        cx0_l,cx1_l=c['x0']-x0,c['x1']-x0
        bb=pv.Box(bounds=(cx0_l,cx1_l,cy0_l,cy1_l,cz0_l,cz1_l))
        a=pl.add_mesh(bb,style='wireframe',color=ORANGE,line_width=2,render=False)
        comp_actors.append(a)

        col2=TYPE_COLOURS.get(c['shape_type'],'#888')
        pl.add_text(
            f"Component {sel}  |  {c['shape_type'].replace('_',' ')}\n"
            f"Voxels: {c['n_vox']}  (RAW={c['n_raw']}  TIF={c['n_tif']})\n"
            f"Elongation: {c['elongation']:.1f}×   Fill: {c['fill_ratio']:.3f}",
            position='upper_left',font_size=9,color=TEXT,font='courier',
        )
        pl.reset_camera(); pl.render()

    iren=pl.iren.interactor
    iren.AddObserver('TimerEvent',lambda o,e: timer_cb())
    iren.CreateRepeatingTimer(TIMER_MS)
    pl.camera_position='iso'
    pl.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("Disagreement Analysis — Dash + PyVista")
    print("="*60)
    for p in [RAW_PATH,TIF_PATH]:
        if not p.exists(): sys.exit(f"Not found: {p}")

    raw_c,tif_c,cinfo=build_canvases(RAW_PATH,TIF_PATH)
    comps,labels,only_raw,only_tif=find_components(raw_c,tif_c,MIN_VOXELS)
    if not comps: sys.exit("No disagreement components found.")

    app=build_dash(raw_c,tif_c,comps,cinfo)

    def _dash():
        time.sleep(1.5)
        app.run(debug=False,port=DASH_PORT,use_reloader=False)

    threading.Thread(target=_dash,daemon=True).start()
    threading.Thread(target=lambda:(time.sleep(3),
        webbrowser.open(f'http://127.0.0.1:{DASH_PORT}')),daemon=True).start()

    run_pyvista(raw_c,tif_c,comps,cinfo)
    print("Done.")


if __name__=='__main__':
    main()
