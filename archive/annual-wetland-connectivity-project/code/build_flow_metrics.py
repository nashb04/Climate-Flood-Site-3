"""Flow-RESPONSE metrics per gauge-year (the outcomes wetlands actually affect):
  peak_cfs        annual maximum daily discharge (flood peak)
  rb_flashiness   Richards-Baker index = Σ|ΔQ| / ΣQ  (timing/flashiness; wetlands ↓)
  q10_cfs         low-flow (10th pctile daily Q; baseflow proxy; wetlands ↑)
  q_mean_cfs      annual mean (volume; mostly precip — kept for contrast)
Wetlands attenuate peaks & flashiness and sustain baseflow, so these — NOT annual
mean volume — are the hypothesis-relevant outcomes.
"""
import time, io, warnings
from pathlib import Path
import numpy as np, pandas as pd, requests
warnings.filterwarnings("ignore")
ROOT=Path("/Users/jared/Wetland"); DATA=ROOT/"data"; OUT=ROOT/"outputs"/"sensors"
SENSOR_IDS=["04087000","04086600","04086500","04087120","04087030",
            "04087050","04087070","04087088","04087119"]
_PANEL_SF = Path("/Users/jared/Wetland/sensors/panel_sites.txt")
if _PANEL_SF.exists(): SENSOR_IDS = _PANEL_SF.read_text().split()
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def daily_Q(sid):
    f=DATA/f"dailyQ_{sid}.parquet"
    if f.exists(): return pd.read_parquet(f)
    url=(f"https://waterservices.usgs.gov/nwis/dv/?sites={sid}&parameterCd=00060"
         f"&statCd=00003&startDT=1985-01-01&endDT=2024-12-31&format=rdb")
    r=requests.get(url,timeout=120)
    if r.status_code!=200 or "#" not in r.text: return None
    lines=[l for l in r.text.splitlines() if not l.startswith("#")]
    if len(lines)<3: return None
    d=pd.read_csv(io.StringIO("\n".join(lines)),sep="\t",skiprows=[1],dtype=str)
    d.columns=d.columns.str.strip()
    vc=[c for c in d.columns if "00060" in c and not c.endswith("_cd")]
    if not vc: return None
    d["Q"]=pd.to_numeric(d[vc[0]],errors="coerce")
    d["date"]=pd.to_datetime(d["datetime"],errors="coerce")
    d=d.dropna(subset=["Q","date"]); d["year"]=d.date.dt.year
    d=d[["date","year","Q"]]; d.to_parquet(f); return d

rows=[]
for sid in SENSOR_IDS:
    d=daily_Q(sid)
    if d is None: log(f"{sid}: no data"); continue
    for y,g in d.groupby("year"):
        q=g.sort_values("date")["Q"].values
        if len(q)<300 or q.sum()<=0: continue
        rb=np.abs(np.diff(q)).sum()/q.sum()
        rows.append(dict(site_no=sid, year=int(y),
                         peak_cfs=round(float(q.max()),1),
                         rb_flashiness=round(float(rb),4),
                         q10_cfs=round(float(np.percentile(q,10)),3),
                         q_mean_cfs=round(float(q.mean()),2), ndays=len(q)))
    log(f"{sid}: {d.year.nunique()} yrs daily Q")
df=pd.DataFrame(rows).sort_values(["site_no","year"])
df.to_csv(OUT/"panel_flowmetrics.csv",index=False)
log(f"Wrote {OUT/'panel_flowmetrics.csv'} ({len(df)} rows)")
