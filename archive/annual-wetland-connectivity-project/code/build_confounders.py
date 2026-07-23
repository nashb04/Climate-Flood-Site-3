"""Confounder screening for the wetland→depth regression — DAMS (National
Inventory of Dams).  For each sensor catchment, list dams and—critically—flag
dams *built / modified within the 1985–2024 study window*, which change
downstream stage independently of wetland change (time-varying confounders).

pygeohydro.NID is broken against the current NID schema, so we pull the public
NID CSV directly (key-free) and cache it.
"""
import warnings, requests, io
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Point
warnings.filterwarnings("ignore")

ROOT=Path("/Users/jared/Wetland"); DATA=ROOT/"data"; SENS=ROOT/"sensors"; OUT=ROOT/"outputs"/"sensors"
PROJ="EPSG:32616"
SENSOR_IDS=["04087000","04086600","04086500","04087120","04087030",
            "04087050","04087070","04087088","04087119"]
Y0,Y1=1985,2024

# ── NID inventory (cache the 67 MB national CSV once) ────────────────────────
nid_csv=DATA/"nid_full.csv"
if not nid_csv.exists():
    print("Downloading NID national inventory CSV (~67 MB, once)…")
    r=requests.get("https://nid.sec.usace.army.mil/api/nation/csv",timeout=300); r.raise_for_status()
    nid_csv.write_bytes(r.content); print("  cached.")
# header is on the 2nd line ("Data Last Updated" is line 1)
nid=pd.read_csv(nid_csv, skiprows=1, low_memory=False)
print(f"NID rows (nation): {len(nid):,}")

def col(*cands):
    for c in cands:
        if c in nid.columns: return c
    for c in nid.columns:
        if any(k.lower() in c.lower() for k in cands): return c
    return None
C_lat=col("Latitude"); C_lon=col("Longitude"); C_name=col("Dam Name")
C_yr=col("Year Completed"); C_mod=col("Year(s) Modified","Years Modified")
C_pur=col("Primary Purpose"); C_st=col("Max Storage (Acre-Ft)","NID Storage (Acre-Ft)","Max Storage")
C_h=col("Dam Height (Ft)","NID Height (Ft)"); C_riv=col("River or Stream Name","River")
C_state=col("State"); C_removed=col("Removed")
nid=nid.dropna(subset=[C_lat,C_lon])
dams=gpd.GeoDataFrame(nid, geometry=[Point(x,y) for x,y in zip(nid[C_lon],nid[C_lat])], crs=4326)

# ── Clip to AOI, save ────────────────────────────────────────────────────────
aoi=gpd.read_file(SENS/"aoi_watershed.gpkg").to_crs(4326)
dams_aoi=dams[dams.within(aoi.geometry.iloc[0])].copy()
print(f"Dams within AOI: {len(dams_aoi)}")
dams_aoi_p=dams_aoi.to_crs(PROJ)
keep=[c for c in [C_name,C_yr,C_mod,C_pur,C_st,C_h,C_riv,C_state,C_removed] if c]
dams_aoi[keep+["geometry"]].to_file(SENS/"dams_aoi.gpkg", driver="GPKG")

def yr(v):
    try: return int(float(v))
    except: return np.nan

# ── Per-sensor join to catchment polygons ───────────────────────────────────
rows=[]; per_dam=[]
sens=gpd.read_file(SENS/"sensors_dv_gauges.gpkg").set_index("site_no")
for sid in SENSOR_IDS:
    cf=SENS/f"catchment_{sid}.gpkg"
    if not cf.exists(): continue
    catch=gpd.read_file(cf).to_crs(PROJ)
    nm=sens.loc[sid,"station_nm"] if sid in sens.index else sid
    inside=gpd.sjoin(dams_aoi_p, catch, predicate="within", how="inner")
    n=len(inside)
    yrs=[yr(v) for v in inside[C_yr]] if n else []
    in_window=[y for y in yrs if not np.isnan(y) and Y0<=y<=Y1]
    big=inside.sort_values(C_st, ascending=False).head(3) if n else inside
    rows.append(dict(site_no=sid, station_nm=nm, n_dams=n,
                     dams_built_in_1985_2024=len(in_window),
                     built_window_years=",".join(map(str,sorted(in_window))) if in_window else "",
                     largest_storage_acreft=round(float(inside[C_st].max()),0) if n and inside[C_st].notna().any() else None))
    for _,d in inside.iterrows():
        per_dam.append(dict(site_no=sid, dam_name=d[C_name], year_completed=yr(d[C_yr]),
                            year_modified=d.get(C_mod), purpose=d.get(C_pur),
                            storage_acreft=d.get(C_st), height_ft=d.get(C_h), river=d.get(C_riv)))
summ=pd.DataFrame(rows)
per=pd.DataFrame(per_dam)
summ.to_csv(OUT/"confounders_dams_by_sensor.csv", index=False)
per.to_csv(OUT/"confounders_dams_detail.csv", index=False)
pd.set_option("display.width",220)
print("\n=== Dams per sensor catchment ===")
print(summ.to_string(index=False))
print(f"\nTotal dams across catchments (with nesting): {summ.n_dams.sum()}")
print(f"Dams completed within {Y0}-{Y1} (time-varying confounders) by sensor above.")

# ── Map: dams over AOI + catchments ──────────────────────────────────────────
fig,ax=plt.subplots(figsize=(11,13))
aoi.to_crs(PROJ).boundary.plot(ax=ax,color="black",lw=2,zorder=2)
cmap=plt.cm.tab10
for i,sid in enumerate(SENSOR_IDS):
    cf=SENS/f"catchment_{sid}.gpkg"
    if cf.exists(): gpd.read_file(cf).to_crs(PROJ).boundary.plot(ax=ax,color=cmap(i%10),lw=1.0,zorder=3,alpha=0.7)
yrs_all=np.array([yr(v) for v in dams_aoi_p[C_yr]])
in_win=(yrs_all>=Y0)&(yrs_all<=Y1)
dams_aoi_p[~in_win].plot(ax=ax,color="dimgray",markersize=18,marker="s",zorder=5,label="dam (pre-1985)")
dams_aoi_p[in_win].plot(ax=ax,color="red",markersize=55,marker="*",zorder=6,label=f"dam built {Y0}-{Y1} (confounder)")
ax.legend(loc="upper right",fontsize=9)
ax.set_title(f"Dams (NID) in AOI vs sensor catchments\n{len(dams_aoi)} dams; red = built within study window",fontsize=12)
ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)"); ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(OUT/"00_confounders_dams.png",dpi=150,bbox_inches="tight"); plt.close(fig)
print(f"\nSaved: confounders_dams_by_sensor.csv, confounders_dams_detail.csv, 00_confounders_dams.png")
