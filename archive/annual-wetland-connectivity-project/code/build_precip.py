"""Annual precipitation per sensor catchment, 1985–2023 (Daymet single-pixel REST
at the catchment centroid; pydaymet's THREDDS gridded endpoint is erroring).
Daily prcp summed to annual totals [mm] — the dominant control on discharge.
"""
import time, io, warnings
from pathlib import Path
import pandas as pd, geopandas as gpd, requests
warnings.filterwarnings("ignore")
ROOT=Path("/Users/jared/Wetland"); SENS=ROOT/"sensors"; OUT=ROOT/"outputs"/"sensors"
SENSOR_IDS=["04087000","04086600","04086500","04087120","04087030",
            "04087050","04087070","04087088","04087119"]
_PANEL_SF = Path("/Users/jared/Wetland/sensors/panel_sites.txt")
if _PANEL_SF.exists(): SENSOR_IDS = _PANEL_SF.read_text().split()
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

rows=[]
for sid in SENSOR_IDS:
    f=SENS/f"catchment_{sid}.gpkg"
    if not f.exists(): continue
    c=gpd.read_file(f).to_crs(4326).geometry.iloc[0].centroid
    url=(f"https://daymet.ornl.gov/single-pixel/api/data?lat={c.y:.4f}&lon={c.x:.4f}"
         f"&vars=prcp&start=1985-01-01&end=2023-12-31")
    try:
        r=requests.get(url,timeout=120); r.raise_for_status()
        lines=r.text.splitlines()
        hdr=[i for i,l in enumerate(lines) if l.startswith("year,")][0]
        df=pd.read_csv(io.StringIO("\n".join(lines[hdr:])))
        pcol=[c for c in df.columns if "prcp" in c][0]
        ann=df.groupby("year")[pcol].sum()
        for y,v in ann.items():
            rows.append(dict(site_no=sid, year=int(y), precip_mm=round(float(v),1)))
        log(f"{sid}: {len(ann)} yrs, {ann.min():.0f}-{ann.max():.0f} mm")
    except Exception as ex:
        log(f"{sid}: FAILED {ex}")

df=pd.DataFrame(rows).sort_values(["site_no","year"])
df.to_csv(OUT/"panel_precip_daymet.csv", index=False)
log(f"Wrote {OUT/'panel_precip_daymet.csv'}  ({len(df)} rows)")
