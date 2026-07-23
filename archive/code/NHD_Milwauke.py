import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path


def get_project_paths():
    """
    Define project paths.
    The input GeoPackage should be in the same folder as this Python script.
    """
    base_dir = Path(__file__).resolve().parent

    gpkg_path = base_dir / "NHD_H_04040003_HU8_GPKG.gpkg"

    output_folder = base_dir / "NHD_outputs"
    output_folder.mkdir(exist_ok=True)

    return gpkg_path, output_folder


def load_nhd_layers(gpkg_path, flowline_layer="NHDFlowline", boundary_layer="WBDHU8"):
    """
    Load NHD flowlines and watershed boundary from the GeoPackage.
    """
    print("GeoPackage path:", gpkg_path)
    print("File exists:", gpkg_path.exists())

    if not gpkg_path.exists():
        raise FileNotFoundError(f"GeoPackage file not found: {gpkg_path}")

    print("Loading flowlines...")
    flowlines = gpd.read_file(gpkg_path, layer=flowline_layer)

    print("Loading watershed boundary...")
    boundary = gpd.read_file(gpkg_path, layer=boundary_layer)

    print("Flowline CRS:", flowlines.crs)
    print("Boundary CRS:", boundary.crs)

    return flowlines, boundary


def reproject_layers(flowlines, boundary, epsg=26916):
    """
    Reproject layers to a meter-based CRS.

    EPSG:26916 is NAD83 / UTM zone 16N, which is suitable for the
    Milwaukee / Wisconsin area.
    """
    flowlines_projected = flowlines.to_crs(epsg=epsg)
    boundary_projected = boundary.to_crs(epsg=epsg)

    return flowlines_projected, boundary_projected


def clip_flowlines_to_boundary(flowlines, boundary):
    """
    Clip the NHD flowlines to the watershed boundary.
    """
    print("Clipping flowlines to watershed boundary...")
    flowlines_clipped = gpd.clip(flowlines, boundary)

    return flowlines_clipped


def split_streams_by_order(flowlines_clipped, stream_order_threshold=4):
    """
    Separate larger rivers and smaller streams using StreamOrde.

    If StreamOrde is not available, all flowlines are treated as smaller streams.
    """
    if "StreamOrde" in flowlines_clipped.columns:
        main_rivers = flowlines_clipped[
            flowlines_clipped["StreamOrde"] >= stream_order_threshold
        ]

        small_streams = flowlines_clipped[
            flowlines_clipped["StreamOrde"] < stream_order_threshold
        ]
    else:
        main_rivers = flowlines_clipped.iloc[0:0]
        small_streams = flowlines_clipped

    return main_rivers, small_streams


def save_spatial_outputs(flowlines_clipped, boundary, output_folder):
    """
    Save clipped flowlines and watershed boundary to the output folder.
    """
    flowline_output = output_folder / "Milwaukee_watershed_flowlines_clipped.gpkg"
    boundary_output = output_folder / "Milwaukee_watershed_boundary.gpkg"

    print("Saving clipped flowlines...")
    flowlines_clipped.to_file(
        flowline_output,
        layer="flowlines_clipped",
        driver="GPKG"
    )

    print("Saving watershed boundary...")
    boundary.to_file(
        boundary_output,
        layer="watershed_boundary",
        driver="GPKG"
    )

    return flowline_output, boundary_output


def process_nhd_data(gpkg_path, output_folder):
    """
    Process NHD data.

    This function handles loading, reprojecting, clipping, separating streams,
    and saving spatial outputs. It does not create the map.
    """
    flowlines, boundary = load_nhd_layers(gpkg_path)

    flowlines, boundary = reproject_layers(flowlines, boundary)

    flowlines_clipped = clip_flowlines_to_boundary(flowlines, boundary)

    main_rivers, small_streams = split_streams_by_order(flowlines_clipped)

    flowline_output, boundary_output = save_spatial_outputs(
        flowlines_clipped,
        boundary,
        output_folder
    )

    return {
        "boundary": boundary,
        "flowlines_clipped": flowlines_clipped,
        "main_rivers": main_rivers,
        "small_streams": small_streams,
        "flowline_output": flowline_output,
        "boundary_output": boundary_output
    }


def plot_river_network(boundary, small_streams, main_rivers, output_folder):
    """
    Plot the Milwaukee watershed river network.

    This function is separate from the processing function so it can be reused
    by other scripts.
    """
    map_output = output_folder / "Milwaukee_watershed_river_map.png"

    print("Creating map...")

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

    return map_output


def main():
    """
    Run the full NHD workflow.
    """
    gpkg_path, output_folder = get_project_paths()

    processed_data = process_nhd_data(gpkg_path, output_folder)

    map_output = plot_river_network(
        boundary=processed_data["boundary"],
        small_streams=processed_data["small_streams"],
        main_rivers=processed_data["main_rivers"],
        output_folder=output_folder
    )

    print("\nSaved map to:")
    print(map_output)

    print("\nSaved clipped flowlines to:")
    print(processed_data["flowline_output"])

    print("\nSaved watershed boundary to:")
    print(processed_data["boundary_output"])


if __name__ == "__main__":
    main()