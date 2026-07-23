"""Dam dummy per gauge-year: Dam(i,t) = 1 if gauge i's catchment contains a dam
that existed in year t (NID, Year Completed <= t), else 0. Simplified per user
request ('this sensor has a dam that year or not').
"""
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
ROOT=Path("/Users/jared/Wetland"); SENS=ROOT/"sensors"; OUT=ROOT/"outputs"/"sensors"
PROJ="EPSG:32616"
sites=[s.strip() for s in (SENS/"panel_sites.txt").read_text().split()]
dams=gpd.read_file(SENS/"dams_aoi.gpkg").to_crs(PROJ)
yr_col="Year Completed"

rows=[]
for sid in sites:
    cf=SENS/f"catchment_{sid}.gpkg"
    if not cf.exists(): continue
    poly=gpd.read_file(cf).to_crs(PROJ).geometry.iloc[0]
    inside=dams[dams.within(poly)]
    n=len(inside)
    yrs=pd.to_numeric(inside[yr_col],errors="coerce").dropna()
    earliest=int(yrs.min()) if len(yrs) else None
    rows.append(dict(site_no=sid, n_dams=n, has_dam=int(n>0),
                     earliest_dam_year=earliest))
df=pd.DataFrame(rows)
df.to_csv(SENS/"dam_by_sensor.csv",index=False)
print(f"{len(df)} gauges; {df.has_dam.sum()} have >=1 dam in catchment")
print(df[df.has_dam==1][["site_no","n_dams","earliest_dam_year"]].to_string(index=False))
print("\n-> Dam(i,t) built in assembly = 1 if earliest_dam_year <= year else 0 "
      "(all dams pre-1985 here, so effectively time-invariant per gauge).")
