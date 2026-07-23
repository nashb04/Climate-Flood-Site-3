"""Annual streamflow / stage (depth) per gauge, 1985–2024, from USGS waterservices.

Fetches daily-mean discharge (00060) and gage height / depth (00065). Most
Milwaukee-area gauges have long discharge records but short/no stage; we report
coverage so the regression outcome can be chosen (stage where available, else
discharge as the depth proxy).
"""
import time, warnings, io
from pathlib import Path
import numpy as np, pandas as pd, requests
warnings.filterwarnings("ignore")

ROOT=Path("/Users/jared/Wetland"); OUT=ROOT/"outputs"/"sensors"
SENSOR_IDS=["04087000","04086600","04086500","04087120","04087030",
            "04087050","04087070","04087088","04087119"]
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def fetch(sid, pc):
    url=(f"https://waterservices.usgs.gov/nwis/dv/?sites={sid}&parameterCd={pc}"
         f"&statCd=00003&startDT=1985-01-01&endDT=2024-12-31&format=rdb")
    r=requests.get(url,timeout=60)
    if r.status_code!=200 or "#" not in r.text: return None
    lines=[l for l in r.text.splitlines() if not l.startswith("#")]
    if len(lines)<3: return None
    df=pd.read_csv(io.StringIO("\n".join(lines)),sep="\t",skiprows=[1],dtype=str)
    df.columns=df.columns.str.strip()
    val=[c for c in df.columns if pc in c and not c.endswith("_cd")]
    if not val or "datetime" not in df.columns: return None
    df["v"]=pd.to_numeric(df[val[0]],errors="coerce")
    df["year"]=pd.to_datetime(df["datetime"],errors="coerce").dt.year
    return df.dropna(subset=["v","year"])

rows=[]
for sid in SENSOR_IDS:
    q=fetch(sid,"00060"); h=fetch(sid,"00065")
    yrs=set()
    if q is not None: yrs|=set(q.year.unique())
    if h is not None: yrs|=set(h.year.unique())
    for y in sorted(yrs):
        rec=dict(site_no=sid, year=int(y))
        if q is not None:
            qy=q[q.year==y]["v"]
            if len(qy)>=300:  # need near-complete year
                rec.update(q_mean_cfs=round(qy.mean(),2), q_med_cfs=round(qy.median(),2),
                           q_max_cfs=round(qy.max(),1), q_p90_cfs=round(qy.quantile(.9),2),
                           q_ndays=len(qy))
        if h is not None:
            hy=h[h.year==y]["v"]
            if len(hy)>=300:
                rec.update(stage_mean_ft=round(hy.mean(),3), stage_med_ft=round(hy.median(),3),
                           stage_max_ft=round(hy.max(),3), stage_ndays=len(hy))
        rows.append(rec)
    log(f"{sid}: Q yrs={0 if q is None else q.year.nunique()}, stage yrs={0 if h is None else h.year.nunique()}")

df=pd.DataFrame(rows).sort_values(["site_no","year"])
df.to_csv(OUT/"panel_flow_stage.csv", index=False)
log(f"Wrote {OUT/'panel_flow_stage.csv'}  ({len(df)} rows)")
# coverage summary
has_q=df.groupby("site_no")["q_mean_cfs"].apply(lambda s: s.notna().sum()) if "q_mean_cfs" in df else None
has_s=df.groupby("site_no")["stage_mean_ft"].apply(lambda s: s.notna().sum()) if "stage_mean_ft" in df else None
print("\nYears with complete DISCHARGE per gauge:\n", has_q.to_string() if has_q is not None else "none")
print("\nYears with complete STAGE per gauge:\n", has_s.to_string() if has_s is not None else "none")
