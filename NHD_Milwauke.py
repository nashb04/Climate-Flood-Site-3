import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

# Project folder: same folder as this Python script
BASE_DIR = Path(__file__).resolve().parent

# Input GeoPackage
gpkg_path = BASE_DIR / "NHD_H_04040003_HU8_GPKG.gpkg"

# Output folder
output_folder = BASE_DIR / "output"
output_folder.mkdir(exist_ok=True)

# Layers
flowline_layer = "NHDFlowline"
boundary_layer = "WBDHU8"

print("GeoPackage path:", gpkg_path)
print("File exists:", gpkg_path.exists())

print("Loading flowlines...")
flowlines = gpd.read_file(gpkg_path, layer=flowline_layer)

print("Loading watershed boundary...")
boundary = gpd.read_file(gpkg_path, layer=boundary_layer)

print("Flowline CRS:", flowlines.crs)
print("Boundary CRS:", boundary.crs)

# Reproject to meter-based CRS for Milwaukee / Wisconsin area
flowlines = flowlines.to_crs(epsg=26916)
boundary = boundary.to_crs(epsg=26916)

# Clip flowlines to watershed boundary
flowlines_clipped = gpd.clip(flowlines, boundary)

# Separate main rivers and smaller streams
if "StreamOrde" in flowlines_clipped.columns:
    main_rivers = flowlines_clipped[flowlines_clipped["StreamOrde"] >= 4]
    small_streams = flowlines_clipped[flowlines_clipped["StreamOrde"] < 4]
else:
    main_rivers = flowlines_clipped.iloc[0:0]
    small_streams = flowlines_clipped

# Save outputs
flowline_output = output_folder / "Milwaukee_watershed_flowlines_clipped.gpkg"
boundary_output = output_folder / "Milwaukee_watershed_boundary.gpkg"
map_output = output_folder / "Milwaukee_watershed_river_map.png"

flowlines_clipped.to_file(flowline_output, layer="flowlines_clipped", driver="GPKG")
boundary.to_file(boundary_output, layer="watershed_boundary", driver="GPKG")

# Plot map
fig, ax = plt.subplots(figsize=(8, 11))

boundary.boundary.plot(
    ax=ax,
    color="black",
    linewidth=1.2
)

small_streams.plot(
    ax=ax,
    color="#9fd8f2",
    linewidth=0.7,
    alpha=0.9
)

if len(main_rivers) > 0:
    main_rivers.plot(
        ax=ax,
        color="#168aad",
        linewidth=1.4,
        alpha=0.95
    )

ax.set_title(
    "Milwaukee Watershed River Network",
    fontsize=15,
    pad=12
)

ax.set_axis_off()
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(map_output, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved map to:")
print(map_output)

print("\nSaved clipped flowlines to:")
print(flowline_output)

print("\nSaved watershed boundary to:")
print(boundary_output)