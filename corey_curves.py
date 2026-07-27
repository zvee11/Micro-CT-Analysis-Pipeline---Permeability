import json, statistics as st, datetime as dt
from pathlib import Path
from collections import defaultdict
import duckdb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

DB_PATH  = Path("results.duckdb")
FIG_DIR  = Path("figures"); FIG_DIR.mkdir(exist_ok=True)
TAB_DIR  = Path("tables");  TAB_DIR.mkdir(exist_ok=True)
M2_TO_D  = 1.0 / 9.869233e-13
M2_TO_MD = M2_TO_D * 1000.0

# ---- fixed Corey endpoints, set from the experiment and NOT fitted ----------
SWC_FIX, SNR_FIX = 0.0, 0.39
CW_FIX = 0.0                    # brine tail coefficient, fixed as in the baseline

# ---- fitting protocol ------------------------------------------------------
ANCHOR_KRG, ANCHOR_KRW = 1.0, 0.0   # her Sw = 0 constraint
PAST_RESIDUAL_INFLATE  = 50.0       # her down-weighting past the residual
BAND_DRAWS = 2000

# Her acceptable ranges, taken from her Tables 3.2, 3.3, 3.8 and 3.9. She sets
# them per experiment, so they are carried per dataset here rather than shared.
PARENTE_RANGES = {
  "h2":    dict(krw0=(0.19,1.0), krn0=(0.2,1.0), nw=(2.0,6.0), nn=(2.0,6.0), cn=(0.0,0.5)),
  "ch4_7": dict(krw0=(0.10,1.0), krn0=(0.2,1.0), nw=(2.0,6.0), nn=(1.3,6.0), cn=(0.0,0.5)),
  "ch4_4": dict(krw0=(0.10,1.0), krn0=(0.2,1.0), nw=(2.0,6.0), nn=(1.3,6.0), cn=(0.0,0.5)),
}

OUTLIER_TRACKS = {"ch4_7": {4}, "ch4_4": {4}, "h2": set()}

GAS_C, BRINE_C, ABS_C = "#c1121f", "#1d3557", "#2a9d8f"
PIPE_GAS, PIPE_BRINE  = "#e8843c", "#457b9d"
plt.rcParams.update({"figure.dpi":110,"savefig.dpi":200,"font.size":11,
    "axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True,
    "figure.facecolor":"white","axes.facecolor":"white"})

def q(sql,p=None):
    with duckdb.connect(str(DB_PATH),read_only=True) as c:
        return c.execute(sql,p or []).fetchall()

PARENTE_STD = {
  "h2":    dict(krw0=0.1901, krn0=0.9844, nw=5.97,  nn=2.009),
  "ch4_7": dict(krw0=0.1381, krn0=0.9467, nw=3.861, nn=2.154),
  "ch4_4": dict(krw0=0.1381, krn0=0.9467, nw=3.861, nn=2.154),
}

DATASETS = {
  "h2":    dict(title="Hydrogen",                 run="20260626_202151_AMSDS99619", conn="18N"),
  "ch4_7": dict(title="Methane (full sequence)",  run="20260618_155713_AMSDS99619", conn="18N"),
  "ch4_4": dict(title="Methane (four-timestep)",  run="20260622_150320_AMSDS99619", conn="18N"),
}

# ---- Corey forms (endpoints baked in as fixed constants) -------------------
def krn_mod(sw,k0,n,c):
    b=np.clip((1-sw-SNR_FIX)/(1-SWC_FIX-SNR_FIX),0,1); return k0*b**n+(c/(1+c))*b
def krw_mod(sw,k0,n,c):
    b=np.clip((sw-SWC_FIX)/(1-SWC_FIX-SNR_FIX),0,1); return k0*b**n+(c/(1+c))*b
def krn_basic(sw,k0,n):
    b=np.clip((1-sw-SNR_FIX)/(1-SWC_FIX-SNR_FIX),0,1); return k0*b**n
def krw_basic(sw,k0,n):
    b=np.clip((sw-SWC_FIX)/(1-SWC_FIX-SNR_FIX),0,1); return k0*b**n
def krn_mod_s(sw,k0,n,c): return krn_mod(sw,k0,n,c)
def krw_mod_s(sw,k0,n):   return krw_mod(sw,k0,n,CW_FIX)   # c_w fixed, as baseline
def krn_std_s(sw,k0,n):   return krn_basic(sw,k0,n)
def krw_std_s(sw,k0,n):   return krw_basic(sw,k0,n)


CFG = {
 "gas_mod":   dict(fn=krn_mod_s, names=["krn0","nn","cn"], form="modified", phase="gas"),
 "brine_mod": dict(fn=krw_mod_s, names=["krw0","nw"],      form="modified", phase="brine"),
 "gas_std":   dict(fn=krn_std_s, names=["krn0","nn"],      form="basic",    phase="gas"),
 "brine_std": dict(fn=krw_std_s, names=["krw0","nw"],      form="basic",    phase="brine"),
}

def ranges_for(key, names):
    """Her acceptable range for each fitted parameter, and a midpoint start."""
    R=PARENTE_RANGES[key]
    lo=[R[n][0] for n in names]; hi=[R[n][1] for n in names]
    p0=[(a+b)/2 for a,b in zip(lo,hi)]
    return p0, lo, hi
SW_CURVE = np.linspace(0.001, 1-SNR_FIX-0.001, 300)   # common x-axis, ends at Snr

# =============================================================================
#  Parente's error-bar rule, applied to measured data
# =============================================================================
def parente_error_bars(scans, values, sw):
    """One error bar per point, following the manual baseline.

    Her rule, verbatim: points sitting at the same saturation get a bar equal
    to the MAXIMUM DISTANCE between them; a lone point at a saturation takes
    the bar of its nearest neighbour; and points beyond the residual saturation
    are given deliberately large bars so that their weight in the fit is close
    to none.

    Here the sections of one timestep are the points sharing a saturation, so
    the bar is the range of the sections at that timestep: the same quantity she
    assigns by eye, measured instead of judged."""
    scans=np.asarray(scans); values=np.asarray(values,float); sw=np.asarray(sw,float)
    bar={}
    for s in np.unique(scans):
        v=values[scans==s]
        bar[s]=float(v.max()-v.min()) if v.size>1 else 0.0
    nz=[b for b in bar.values() if b>0]
    fallback=float(np.median(nz)) if nz else 1e-6      # lone point takes the typical bar
    out=np.array([bar[s] if bar[s]>0 else fallback for s in scans])
    past=sw > (1.0 - SNR_FIX)                          # gas below its residual
    out[past]*=PAST_RESIDUAL_INFLATE
    return out, int(past.sum())

def fit(fn,x,y,p0,lo,hi,sigma=None,names=None):
    """Bounded least squares, weighted when sigma is supplied. Reports the
    parameters, the covariance, both residual measures, and the names of any
    parameter resting on a bound."""
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    s=None if sigma is None else np.asarray(sigma,float)[m]
    if len(x)<3: return None
    try:
        popt,pcov=curve_fit(fn,x,y,p0=p0,bounds=(lo,hi),sigma=s,
                            absolute_sigma=(s is not None),maxfev=40000)
    except Exception as e:
        print("   fit failed:",e); return None
    resid=y-fn(x,*popt)
    rmse=float(np.sqrt(np.nanmean(resid**2)))
    chi2r=(float(np.sum((resid/s)**2)/max(len(x)-len(popt),1)) if s is not None else None)
    nm=list(names) if names else [f"p{i}" for i in range(len(popt))]
    tol=1e-6
    at_bound=[nm[i] for i in range(len(popt))
              if abs(popt[i]-lo[i])<=tol*max(1.0,abs(lo[i]))
              or abs(popt[i]-hi[i])<=tol*max(1.0,abs(hi[i]))]
    return dict(popt=popt,pcov=pcov,rmse=rmse,chi2_red=chi2r,
                at_bound=at_bound,n=len(x),names=nm)

def band_covariance(fn,sw,res,lo,hi,n_draw=BAND_DRAWS,seed=0):
    """95% band by drawing parameter sets from the fit covariance and taking the
    central 95% of the resulting curves, which is the baseline's procedure."""
    if res is None or res["pcov"] is None or not np.all(np.isfinite(res["pcov"])):
        return None,None
    rng=np.random.default_rng(seed)
    try: draws=rng.multivariate_normal(res["popt"],res["pcov"],size=n_draw)
    except np.linalg.LinAlgError: return None,None
    curves=np.array([fn(sw,*d) for d in np.clip(draws,lo,hi)])
    return np.nanpercentile(curves,2.5,axis=0), np.nanpercentile(curves,97.5,axis=0)


# =============================================================================
#  provenance
# =============================================================================
DDL = """
CREATE TABLE IF NOT EXISTS corey_fits (
    run_id VARCHAR NOT NULL, dataset VARCHAR NOT NULL,
    form VARCHAR NOT NULL, phase VARCHAR NOT NULL,
    swc_fixed DOUBLE, snr_fixed DOUBLE, cw_fixed DOUBLE,
    anchor_used BOOLEAN, weighted BOOLEAN, bar_rule VARCHAR,
    past_residual_inflate DOUBLE, n_past_residual INTEGER,
    n_points INTEGER, param_names VARCHAR, param_values VARCHAR,
    initial_guess VARCHAR, bounds_lo VARCHAR, bounds_hi VARCHAR,
    at_bound VARCHAR, rmse DOUBLE, chi2_reduced DOUBLE,
    band_method VARCHAR, fitted_at TIMESTAMP,
    PRIMARY KEY (run_id, dataset, form, phase)
);
"""
EXPECTED_COLS = ["run_id","dataset","form","phase","swc_fixed","snr_fixed","cw_fixed",
    "anchor_used","weighted","bar_rule","past_residual_inflate","n_past_residual",
    "n_points","param_names","param_values","initial_guess","bounds_lo","bounds_hi",
    "at_bound","rmse","chi2_reduced","band_method","fitted_at"]

def ensure_table(c):
    """Create corey_fits, replacing it if an earlier version with a different
    column set is present, so a schema change never fails silently."""
    have=[r[0] for r in c.execute(
        "SELECT column_name FROM duckdb_columns() WHERE table_name='corey_fits' "
        "ORDER BY column_index").fetchall()]
    if have and [h.lower() for h in have]!=[e.lower() for e in EXPECTED_COLS]:
        print("  corey_fits schema changed; replacing the table")
        c.execute("DROP TABLE corey_fits")
    c.execute(DDL)

def write_fits(records):
    with duckdb.connect(str(DB_PATH)) as c:
        ensure_table(c)
        for r in records:
            c.execute("DELETE FROM corey_fits WHERE run_id=? AND dataset=? AND form=? AND phase=?",
                      [r["run_id"],r["dataset"],r["form"],r["phase"]])
            c.execute("INSERT INTO corey_fits VALUES ("+",".join(["?"]*23)+")",
                      [r["run_id"],r["dataset"],r["form"],r["phase"],
                       SWC_FIX,SNR_FIX,CW_FIX,True,True,"parente_max_distance",
                       PAST_RESIDUAL_INFLATE,r["n_past"],r["n_points"],
                       json.dumps(r["param_names"]),json.dumps(r["param_values"]),
                       json.dumps(r["initial_guess"]),json.dumps(r["bounds_lo"]),
                       json.dumps(r["bounds_hi"]),json.dumps(r["at_bound"]),
                       r["rmse"],r["chi2_reduced"],r["band_method"],dt.datetime.now()])
    print(f"  -> {len(records)} rows written to corey_fits")

def write_kr_to_sim_results(run, conn, points):
    """Write the per-point relative permeabilities back onto their own rows in
    simulation_results. kr_gas lands on the sim_type='gas' row and kr_brine on
    the sim_type='water' row, keyed by (run_id, scan_index, track_id,
    connectivity, sim_type). Adds a `kr` column if the table does not have one."""
    with duckdb.connect(str(DB_PATH)) as c:
        cols = [r[0].lower() for r in c.execute(
            "SELECT column_name FROM duckdb_columns() "
            "WHERE table_name='simulation_results'").fetchall()]
        if "kr" not in cols:
            c.execute("ALTER TABLE simulation_results ADD COLUMN kr DOUBLE")
        n = 0
        for p in points:
            for sty, kr in (("gas", p["kr_gas"]), ("water", p["kr_brine"])):
                if kr is None:
                    continue
                c.execute(
                    "UPDATE simulation_results SET kr=? "
                    "WHERE run_id=? AND scan_index=? AND track_id=? "
                    "AND connectivity=? AND sim_type=?",
                    [kr, run, p["scan"], p["track"], conn, sty])
                n += 1
    print(f"  -> {n} kr values written to simulation_results")

# =============================================================================
#  data loading  (unchanged)
# =============================================================================
def scan_to_file_index(run):
    out={}
    for sc,fn in q("SELECT scan_index,file_name FROM scans WHERE run_id=?",[run]):
        grp=str(fn).split("_")
        try: out[sc]=int(grp[1])
        except Exception: out[sc]=None
    return out

def load_pipeline(run,conn,outliers=frozenset()):
    kabs=dict(q("SELECT track_id,k_z FROM simulation_results WHERE run_id=? AND connectivity=? AND sim_type='absolute'",[run,conn]))
    scan_sw=dict(q("SELECT scan_index,Sw FROM scans WHERE run_id=?",[run]))
    s2f=scan_to_file_index(run)
    fb=q("""SELECT track_id,scan_index,gas_voxels,brine_voxels,gas_voxels_at_X,cluster_voxels
            FROM fixed_boxes WHERE run_id=? AND connectivity=?""",[run,conn])
    by_tr=defaultdict(list)
    for tr,sc,g,b,gx,cv in fb: by_tr[tr].append((sc,g,b,gx,cv))
    sg_local={}
    for tr,recs in by_tr.items():
        box_total=max(g+b for _,g,b,_,_ in recs)          # fixed box total (gas+brine)
        for sc,g,b,gx,cv in recs:
            is_x=(cv is not None and g==cv==gx)            # defining timestep X
            gas_in_box=(box_total-b) if is_x else g        # corrected in-box gas at X
            denom=gas_in_box+b
            sg_local[(tr,sc)]=(gas_in_box/denom) if denom>0 else None
    rows=q("""SELECT track_id,scan_index,sim_type,k_z FROM simulation_results
              WHERE run_id=? AND connectivity=? AND sim_type IN ('gas','water')""",[run,conn])
    cell=defaultdict(dict)
    for tr,sc,sty,kz in rows: cell[(tr,sc)][sty]=kz
    pts=[]
    for (tr,sc),d in cell.items():
        if tr in outliers: continue
        ka=kabs.get(tr); sloc=sg_local.get((tr,sc)); swv=scan_sw.get(sc)
        if d.get("gas") is not None and d.get("water") is not None and ka and sloc is not None and swv is not None:
            pts.append(dict(track=tr,scan=sc,file_idx=s2f.get(sc),Sg_whole=1-swv,Sw_whole=swv,
                            Sg_local=sloc,Sw_local=1-sloc,
                            kr_gas=d["gas"]/ka,kr_brine=d["water"]/ka))   # kr = k_eff / k_abs
    kabs_mD={t:kabs[t]*M2_TO_MD for t in kabs if kabs[t] and t not in outliers}
    kabs_out={t:kabs[t]*M2_TO_MD for t in kabs if kabs[t] and t in outliers}
    return pts,kabs_mD,kabs_out


def pipe_per_timestep(points):
    ts=defaultdict(list)
    for p in points: ts[p["scan"]].append(p)
    out=[]
    for sc in sorted(ts):
        e=ts[sc]
        def ms(k):
            v=[p[k] for p in e if p.get(k) is not None]
            return (st.mean(v), st.pstdev(v) if len(v)>1 else 0.0) if v else (None,None)
        sgl,sgl_sd=ms("Sg_local"); gm,gs=ms("kr_gas"); wm,ws_=ms("kr_brine")
        out.append(dict(scan=sc,file_idx=e[0]["file_idx"],Sg_whole=e[0]["Sg_whole"],
                        Sg_local_mean=sgl,Sg_local_sd=sgl_sd,
                        kr_gas=gm,kr_gas_sd=gs,kr_brine=wm,kr_brine_sd=ws_,n=len(e)))
    return out


# =============================================================================
#  build one dataset
# =============================================================================
def build(key):
    cfg=DATASETS[key]; title=cfg["title"]; outliers=OUTLIER_TRACKS.get(key,set())
    pipe,kabs_mD,kabs_out=load_pipeline(cfg["run"],cfg["conn"],outliers)
    write_kr_to_sim_results(cfg["run"],cfg["conn"],pipe)
    pts_ts=pipe_per_timestep(pipe)

    swp  =[1-p["Sg_local"] for p in pipe]
    krgp =[p["kr_gas"]     for p in pipe]
    krwp =[p["kr_brine"]   for p in pipe]
    scanp=[p["scan"]       for p in pipe]

    sig_g,npast_g=parente_error_bars(scanp,krgp,swp)
    sig_w,npast_w=parente_error_bars(scanp,krwp,swp)

    def add_anchor(x,y,s,val):
        """Parente adds a point at Sw=0 where kr_gas=1 and kr_brine=0, which is
        known to be true because the sample begins fully gas-saturated. It fixes
        the gas endpoint permeability, which the measured points, confined to a
        narrow saturation band, cannot. It has no effect on the brine branch,
        where kr is identically zero at Sw=0 for any parameter set."""
        x=list(x)+[0.0]; y=list(y)+[val]
        if s is not None:
            tight=float(min(v for v in s if v>0))       # known-true point, tightest bar
            s=list(s)+[tight]
        return x,y,s

    RES,BAND={},{}
    for name,c in CFG.items():
        y  = krgp  if c["phase"]=="gas" else krwp
        sg = sig_g if c["phase"]=="gas" else sig_w
        p0,lo,hi = ranges_for(key, c["names"])
        xx,yy,ss = add_anchor(swp,y,sg, ANCHOR_KRG if c["phase"]=="gas" else ANCHOR_KRW)
        r=fit(c["fn"],xx,yy,p0,lo,hi,sigma=ss,names=c["names"])
        RES[name]=r
        if r is None: BAND[name]=(None,None,"none"); continue
        # The endpoint permeability of the gas branch is pinned at 1 by the
        # anchor, so its resting on the upper limit is physical rather than a
        # fitting artefact. A shape parameter resting on a limit is different:
        # the covariance then rests on a local approximation that does not hold
        # there, and the band it produces is not meaningful, so none is drawn.
        shape_railed=[n for n in r["at_bound"] if n in ("nn","nw","cn")]
        if shape_railed:
            BAND[name]=(None,None,"suppressed: "+",".join(shape_railed)+" at limit")
        else:
            blo,bhi=band_covariance(c["fn"],SW_CURVE,r,lo,hi)
            BAND[name]=(blo,bhi,"covariance" if blo is not None else "none")

    print(f"\n=== {title} ({key})  Parente protocol, her ranges ===")
    recs=[]
    for name,c in CFG.items():
        r=RES[name]
        if r is None: print(f"  {name:10s} FAILED"); continue
        ps="  ".join(f"{n}={v:.4g}" for n,v in zip(r["names"],r["popt"]))
        chi=f"  chi2r={r['chi2_red']:.2f}" if r["chi2_red"] is not None else ""
        flag=f"   ON BOUND: {','.join(r['at_bound'])}" if r["at_bound"] else ""
        print(f"  {name:10s} {ps}  rmse={r['rmse']:.4g}{chi}  band={BAND[name][2]}{flag}")
        p0,lo,hi = ranges_for(key, c["names"])
        recs.append(dict(run_id=cfg["run"],dataset=key,form=c["form"],phase=c["phase"],
                         n_points=r["n"],n_past=(npast_g if c["phase"]=="gas" else npast_w),
                         param_names=r["names"],param_values=[float(v) for v in r["popt"]],
                         initial_guess=p0,bounds_lo=lo,bounds_hi=hi,at_bound=r["at_bound"],
                         rmse=r["rmse"],chi2_reduced=r["chi2_red"],band_method=BAND[name][2]))
    if recs: write_fits(recs)

    sw_curve=SW_CURVE
    def draw_common(ax,logy):
        ax.scatter(swp,krgp,marker="o",facecolors="none",edgecolors=PIPE_GAS,s=10,linewidths=0.6,alpha=0.25,zorder=1)
        ax.scatter(swp,krwp,marker="^",facecolors="none",edgecolors=PIPE_BRINE,s=10,linewidths=0.6,alpha=0.25,zorder=1)
        ax.errorbar([1-r["Sg_local_mean"] for r in pts_ts],[r["kr_gas"] for r in pts_ts],
                    yerr=[r["kr_gas_sd"] for r in pts_ts],fmt="o",color=PIPE_GAS,ms=6,capsize=3,mfc="white",lw=1.2,zorder=6,label="Gas (timestep mean)")
        ax.errorbar([1-r["Sg_local_mean"] for r in pts_ts],[r["kr_brine"] for r in pts_ts],
                    yerr=[r["kr_brine_sd"] for r in pts_ts],fmt="^",color=PIPE_BRINE,ms=6,capsize=3,mfc="white",lw=1.2,zorder=6,label="Brine (timestep mean)")
        if True:
            ax.scatter([0],[ANCHOR_KRG],marker="*",s=150,color="#6a0572",zorder=8,
                       label=r"anchor $S_w{=}0$: $k_{r,g}{=}1$")
        ax.set_xlabel(r"Water Saturation $S_w$ (box-local)"); ax.set_ylabel(r"Relative Permeability $k_r$")
        ax.set_xlim(-0.02,1-SNR_FIX+0.02)
        if logy: ax.set_yscale("log"); ax.set_ylim(1e-3,1.0)
        else: ax.set_ylim(-0.01,0.33)

    def shade(ax,name,color):
        blo,bhi,_=BAND[name]
        if blo is not None: ax.fill_between(sw_curve,blo,bhi,color=color,alpha=0.15,lw=0,zorder=2)

    def draw_mod(ax,logy):
        draw_common(ax,logy)
        shade(ax,"gas_mod",PIPE_GAS); shade(ax,"brine_mod",PIPE_BRINE)
        if RES["gas_mod"]:   ax.plot(sw_curve,krn_mod_s(sw_curve,*RES["gas_mod"]["popt"]),"-",color=PIPE_GAS,lw=2.4,label="Gas (mod. Corey fit)")
        if RES["brine_mod"]: ax.plot(sw_curve,krw_mod_s(sw_curve,*RES["brine_mod"]["popt"]),"-",color=PIPE_BRINE,lw=2.4,label="Brine (mod. Corey fit)")

    def draw_std(ax,logy):
        draw_common(ax,logy)
        shade(ax,"gas_std",PIPE_GAS); shade(ax,"brine_std",PIPE_BRINE)
        if RES["gas_std"]:   ax.plot(sw_curve,krn_std_s(sw_curve,*RES["gas_std"]["popt"]),"-",color=PIPE_GAS,lw=2.4,label="Gas (basic Corey fit)")
        if RES["brine_std"]: ax.plot(sw_curve,krw_std_s(sw_curve,*RES["brine_std"]["popt"]),"-",color=PIPE_BRINE,lw=2.4,label="Brine (basic Corey fit)")

    fig,axes=plt.subplots(2,2,figsize=(15.5,11))
    draw_mod(axes[0,0],False); axes[0,0].set_title(f"(a) {title}: Modified Corey (linear)")
    draw_mod(axes[0,1],True);  axes[0,1].set_title(f"(b) {title}: Modified Corey (logarithmic)")
    draw_std(axes[1,0],False); axes[1,0].set_title(f"(c) {title}: Basic Corey (linear)")
    draw_std(axes[1,1],True);  axes[1,1].set_title(f"(d) {title}: Basic Corey (logarithmic)")
    axes[0,0].legend(fontsize=7,loc="upper center",ncol=2)
    axes[1,0].legend(fontsize=7,loc="upper center",ncol=2)   # lower row has its own labels
    fig.tight_layout(); fig.savefig(FIG_DIR/f"kr_pipeline_{key}.png",bbox_inches="tight"); plt.close(fig)
    return RES

# =============================================================================
#  LaTeX table, emitted from the database so it cannot drift from the curves
# =============================================================================
def emit_table():
    LAB={"h2":"Hydrogen","ch4_7":"CH$_4$, full","ch4_4":"CH$_4$, 4-step"}
    ORD=["h2","ch4_7","ch4_4"]
    SYM={"krw0":r"$k^0_{r,w}$","krn0":r"$k^0_{r,g}$","nw":r"$n_w$","nn":r"$n_g$","cn":r"$c_n$"}
    ROW={"basic":["krw0","krn0","nw","nn"],"modified":["krw0","krn0","nw","nn","cn"]}
    F={}
    for r in q("SELECT dataset,form,param_names,param_values,initial_guess,bounds_lo,bounds_hi,at_bound FROM corey_fits"):
        ds,form=r[0],r[1]
        for n,v,p0,lo,hi in zip(json.loads(r[2]),json.loads(r[3]),json.loads(r[4]),json.loads(r[5]),json.loads(r[6])):
            F[(ds,form,n)]=dict(val=v,p0=p0,lo=lo,hi=hi,bound=(n in json.loads(r[7])))
    if not F: print("  (no corey_fits rows; table not written)"); return
    L=[r"\begin{table}[htbp]",r"\centering",
       r"\caption{Corey fit configuration and fitted shape parameters, read from the "
       r"\texttt{corey\_fits} table of the run database. Endpoint saturations are fixed "
       r"from the experiment and are not fitted. Ranges are those of the baseline, set per experiment. Values marked $\\dagger$ rest on a range limit, as three of the baseline\'s own do.}",
       r"\label{tab:corey_config}",r"\small",r"\begin{tabular}{lccccc}",r"\toprule",
       r"\textbf{Parameter} & \textbf{Initial} & \textbf{Range} & "
       + " & ".join(r"\textbf{"+LAB[k]+"}" for k in ORD)+r" \\",r"\midrule"]
    for form in ("basic","modified"):
        L.append(r"\multicolumn{6}{l}{\textit{"+form.capitalize()+r" Corey}} \\")
        L.append(r"$S_{w,c}$ & --- & fixed & "+" & ".join([f"{SWC_FIX:.2f}"]*3)+r" \\")
        L.append(r"$S_{n,r}$ & --- & fixed & "+" & ".join([f"{SNR_FIX:.2f}"]*3)+r" \\")
        if form=="modified":
            L.append(r"$c_w$ & --- & fixed & "+" & ".join([f"{CW_FIX:.2f}"]*3)+r" \\")
        for p in ROW[form]:
            ref=next((F[(k,form,p)] for k in ORD if (k,form,p) in F),None)
            if ref is None: continue
            cells=[("---" if (k,form,p) not in F else
                    f"{F[(k,form,p)]['val']:.3f}"+(r"$^\dagger$" if F[(k,form,p)]['bound'] else ""))
                   for k in ORD]
            L.append(f"{SYM[p]} & {ref['p0']:g} & {ref['lo']:g}--{ref['hi']:g} & "+" & ".join(cells)+r" \\")
        L.append(r"\addlinespace")
    L+= [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (TAB_DIR/"table_corey_config.tex").write_text("\n".join(L),encoding="utf-8")
    print(f"  -> {TAB_DIR/'table_corey_config.tex'}")


if __name__=="__main__":
    for k in ["h2","ch4_7","ch4_4"]:
        build(k)
    print("\nLaTeX table:"); emit_table()
    print("\nBaseline comparison figures: run corey_compare.py")
    print("\nInspect the recorded configuration with:\n"
          "  SELECT dataset, form, phase, param_values, bounds_lo, bounds_hi,\n"
          "         at_bound, chi2_reduced, band_method FROM corey_fits ORDER BY 1,2,3;")