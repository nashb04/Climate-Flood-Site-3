#!/usr/bin/env python3
"""Build clipped 3DEP terrain data for one AOI."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "terrain_sop_matplotlib"))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
from shapely.validation import make_valid

from sda_soil_downloader import load_aoi_gdf, resolve_shapefile_input


TNM_PRODUCTS_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"
DEFAULT_DEM_DATASET = "National Elevation Dataset (NED) 1/3 arc-second Current"
DEFAULT_MAP_CRS = "EPSG:3071"
DEFAULT_BUFFER_METERS = 500.0
DEFAULT_NODATA = -9999.0
ACRES_TO_SQUARE_METERS = 4046.8564224


@dataclass(frozen=True)
class DemProduct:
    title: str
    url: str
    size_bytes: int | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clipped USGS 3DEP terrain data.")
    parser.add_argument("--input", type=Path, required=True, help="AOI zipped shapefile.")
    parser.add_argument("--input-crs", help="CRS to assign if the AOI has no .prj/CRS.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder for all products.")
    parser.add_argument("--dem-dataset", default=DEFAULT_DEM_DATASET, help="TNMAccess DEM dataset name.")
    parser.add_argument("--map-crs", default=DEFAULT_MAP_CRS, help="Projected CRS for DEM derivatives.")
    parser.add_argument("--buffer-meters", type=float, default=DEFAULT_BUFFER_METERS)
    parser.add_argument("--target-resolution", type=float, help="Output DEM cell size in map CRS units.")
    parser.add_argument("--redownload", action="store_true", help="Re-download DEM tiles even if present.")
    parser.add_argument("--skip-visuals", action="store_true", help="Skip QA maps/charts.")
    parser.add_argument("--max-products", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-sleep", type=float, default=0.2)
    return parser.parse_args(argv)


def check_input_args(args: argparse.Namespace) -> None:
    if args.input.suffix.lower() != ".zip":
        raise SystemExit("Expected --input to be a zipped shapefile (.zip).")
    if args.buffer_meters < 0:
        raise SystemExit("--buffer-meters must be 0 or positive.")
    if args.target_resolution is not None and args.target_resolution <= 0:
        raise SystemExit("--target-resolution must be positive.")


def step_1_load_aoi(args: argparse.Namespace, output_dir: Path):
    print("\nStep 1/7 - Load AOI boundary")
    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    shapefile = resolve_shapefile_input(args, work_dir)
    aoi = load_aoi_gdf(gpd, shapefile, args.input_crs)
    aoi = dissolve_aoi(aoi)

    aoi_map = aoi.to_crs(args.map_crs)
    area_acres = float(aoi_map.geometry.area.sum() / ACRES_TO_SQUARE_METERS)
    print(f"  AOI features after dissolve: {len(aoi):,}")
    print(f"  AOI CRS: {aoi.crs}")
    print(f"  AOI area: {area_acres:,.0f} acres")
    return aoi, area_acres


def dissolve_aoi(aoi):
    fixed = aoi.copy()
    fixed["geometry"] = fixed.geometry.apply(fix_geometry)
    fixed = fixed[fixed.geometry.notnull() & ~fixed.geometry.is_empty].copy()
    if fixed.empty:
        raise SystemExit("AOI has no usable polygon geometry after repair.")

    if hasattr(fixed.geometry, "union_all"):
        union_geom = fixed.geometry.union_all()
    else:
        union_geom = fixed.unary_union
    union_geom = fix_geometry(union_geom)
    return gpd.GeoDataFrame({"name": ["aoi"]}, geometry=[union_geom], crs=fixed.crs)


def fix_geometry(geom):
    if geom is None or geom.is_empty or geom.is_valid:
        return geom
    try:
        return make_valid(geom)
    except Exception:
        return geom.buffer(0)


def buffered_aoi(aoi, map_crs: str, buffer_meters: float):
    projected = aoi.to_crs(map_crs)
    geom = projected.geometry.iloc[0]
    if buffer_meters:
        geom = geom.buffer(buffer_meters)
    return gpd.GeoDataFrame({"name": ["aoi_buffer"]}, geometry=[geom], crs=map_crs)


def step_2_find_dem_tiles(args: argparse.Namespace, aoi, output_dir: Path):
    print("\nStep 2/7 - Find USGS 3DEP DEM tiles")
    search_aoi = buffered_aoi(aoi, args.map_crs, args.buffer_meters)
    bbox = search_aoi.to_crs("EPSG:4326").total_bounds
    bbox_text = ",".join(f"{value:.8f}" for value in bbox)

    params = {
        "bbox": bbox_text,
        "datasets": args.dem_dataset,
        "prodFormats": "GeoTIFF",
        "max": str(args.max_products),
        "outputFormat": "JSON",
    }
    data = get_json_with_retries(TNM_PRODUCTS_URL, params, args.retries, args.timeout)
    products = parse_dem_products(data)
    if not products:
        raise SystemExit(f"No GeoTIFF DEM products found for dataset: {args.dem_dataset}")

    manifest_path = output_dir / "tile_manifest.csv"
    write_product_manifest(manifest_path, products)
    print(f"  Dataset: {args.dem_dataset}")
    print(f"  Search bbox: {bbox_text}")
    print(f"  DEM tiles: {len(products):,}")
    print(f"  Manifest: {manifest_path}")
    return products, search_aoi, bbox_text


def get_json_with_retries(url: str, params: dict[str, str], retries: int, timeout: float) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if "errorMessage" in data:
                raise RuntimeError(data["errorMessage"])
            return data
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(1.5 * attempt)
    raise SystemExit(f"TNMAccess request failed: {last_error}")


def parse_dem_products(data: dict) -> list[DemProduct]:
    products: list[DemProduct] = []
    seen_urls: set[str] = set()
    for item in data.get("items", []):
        url = item.get("downloadURL") or ""
        if not url.lower().endswith((".tif", ".tiff")):
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        products.append(
            DemProduct(
                title=item.get("title") or Path(url).name,
                url=url,
                size_bytes=as_int(item.get("sizeInBytes")),
            )
        )
    products.sort(key=lambda product: product.title)
    return products


def as_int(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def write_product_manifest(path: Path, products: list[DemProduct]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "size_mb", "url"])
        writer.writeheader()
        for product in products:
            size_mb = "" if product.size_bytes is None else round(product.size_bytes / 1024 / 1024, 2)
            writer.writerow({"title": product.title, "size_mb": size_mb, "url": product.url})


def step_3_download_tiles(args: argparse.Namespace, products: list[DemProduct], output_dir: Path):
    print("\nStep 3/7 - Download raw DEM tiles")
    raw_dir = output_dir / "raw_tiles"
    raw_dir.mkdir(parents=True, exist_ok=True)

    tile_paths: list[Path] = []
    for idx, product in enumerate(products, start=1):
        path = raw_dir / Path(product.url).name
        print(f"  [{idx}/{len(products)}] {path.name}")
        download_product(product, path, args.redownload, args.retries, args.timeout)
        tile_paths.append(path)
        time.sleep(args.request_sleep)
    return tile_paths


def download_product(product: DemProduct, path: Path, redownload: bool, retries: int, timeout: float) -> None:
    if path.exists() and not redownload:
        if product.size_bytes is None or path.stat().st_size == product.size_bytes:
            print("    Already present.")
            return

    tmp_path = path.with_suffix(path.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(product.url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp_path.replace(path)
            if product.size_bytes is not None and path.stat().st_size != product.size_bytes:
                raise RuntimeError("Downloaded file size does not match TNM manifest.")
            print(f"    Downloaded {path.stat().st_size / 1024 / 1024:.1f} MB.")
            return
        except Exception as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == retries:
                break
            time.sleep(2.0 * attempt)
    raise SystemExit(f"Download failed for {product.url}: {last_error}")


def step_4_mosaic_and_project(args: argparse.Namespace, tile_paths: list[Path], search_aoi, output_dir: Path):
    print("\nStep 4/7 - Mosaic and project DEM")
    mosaic_dir = output_dir / "mosaic"
    work_dir = output_dir / "_work"
    mosaic_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    mosaic_path = mosaic_dir / "terrain_dem_mosaic_source_crs.tif"
    projected_path = work_dir / "terrain_dem_projected_buffered.tif"

    make_source_mosaic(tile_paths, search_aoi, mosaic_path)
    resolution = infer_target_resolution(args.dem_dataset, args.target_resolution)
    project_raster(mosaic_path, projected_path, args.map_crs, resolution)

    print(f"  Mosaic: {mosaic_path}")
    print(f"  Projected buffered DEM: {projected_path}")
    return mosaic_path, projected_path


def make_source_mosaic(tile_paths: list[Path], search_aoi, out_path: Path) -> None:
    with rasterio.open(tile_paths[0]) as first:
        source_crs = first.crs
        nodata = first.nodata if first.nodata is not None else DEFAULT_NODATA
        source_profile = first.profile.copy()

    if source_crs is None:
        raise SystemExit("DEM tile has no CRS.")

    bounds = tuple(float(v) for v in search_aoi.to_crs(source_crs).total_bounds)
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(path)) for path in tile_paths]
        mosaic, transform = merge(
            sources,
            bounds=bounds,
            indexes=1,
            nodata=nodata,
            dtype="float32",
            target_aligned_pixels=True,
        )

    profile = source_profile
    profile.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        count=1,
        dtype="float32",
        crs=source_crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        predictor=3,
        tiled=True,
        bigtiff="if_safer",
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic.astype("float32"))


def infer_target_resolution(dataset: str, override: float | None) -> float | None:
    if override is not None:
        return override
    text = dataset.lower()
    if "1/9" in text:
        return 3.0
    if "1/3" in text:
        return 10.0
    if "1 arc-second" in text:
        return 30.0
    if "1 meter" in text or "1m" in text:
        return 1.0
    return None


def project_raster(src_path: Path, out_path: Path, dst_crs: str, resolution: float | None) -> None:
    with rasterio.open(src_path) as src:
        kwargs = {}
        if resolution is not None:
            kwargs["resolution"] = resolution
        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            **kwargs,
        )
        nodata = src.nodata if src.nodata is not None else DEFAULT_NODATA
        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            nodata=nodata,
            dtype="float32",
            compress="deflate",
            predictor=3,
            tiled=True,
            bigtiff="if_safer",
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=nodata,
                dst_transform=transform,
                dst_crs=dst_crs,
                dst_nodata=nodata,
                resampling=Resampling.bilinear,
            )


def step_5_clip_dem(aoi, projected_dem: Path, output_dir: Path):
    print("\nStep 5/7 - Clip DEM to AOI")
    clipped_dir = output_dir / "clipped"
    clipped_dir.mkdir(parents=True, exist_ok=True)
    clipped_dem = clipped_dir / "terrain_dem_clipped.tif"
    clip_raster_to_aoi(projected_dem, aoi, clipped_dem)
    print(f"  Clipped DEM: {clipped_dem}")
    return clipped_dem


def clip_raster_to_aoi(src_path: Path, aoi, out_path: Path) -> None:
    with rasterio.open(src_path) as src:
        shapes = [geom.__geo_interface__ for geom in aoi.to_crs(src.crs).geometry]
        clipped, transform = mask(src, shapes, crop=True, nodata=src.nodata, filled=True)
        profile = src.profile.copy()
        profile.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=transform,
            compress="deflate",
            predictor=3,
            tiled=True,
            bigtiff="if_safer",
        )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(clipped)


def step_6_build_derivatives(aoi, projected_dem: Path, output_dir: Path):
    print("\nStep 6/7 - Build terrain derivatives")
    work_dir = output_dir / "_work"
    derivatives_dir = output_dir / "derivatives"
    derivatives_dir.mkdir(parents=True, exist_ok=True)

    buffered_paths = build_derivatives(projected_dem, work_dir)
    clipped_paths: dict[str, Path] = {}
    for name, path in buffered_paths.items():
        clipped_path = derivatives_dir / f"terrain_{name}.tif"
        clip_raster_to_aoi(path, aoi, clipped_path)
        clipped_paths[name] = clipped_path
        print(f"  {name}: {clipped_path}")
    return clipped_paths


def build_derivatives(projected_dem: Path, out_dir: Path) -> dict[str, Path]:
    with rasterio.open(projected_dem) as src:
        dem = src.read(1).astype("float32")
        profile = src.profile.copy()
        nodata = src.nodata if src.nodata is not None else DEFAULT_NODATA
        dx = abs(src.transform.a)
        dy = abs(src.transform.e)

    valid = np.isfinite(dem) & (dem != nodata)
    dem = np.where(valid, dem, np.nan)
    dz_dy, dz_dx = np.gradient(dem, dy, dx)

    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    slope_degrees = np.degrees(slope_rad)
    aspect_degrees = (90.0 - np.degrees(np.arctan2(dz_dy, -dz_dx))) % 360.0
    relief = dem - np.nanmin(dem)
    hillshade = make_hillshade(slope_rad, aspect_degrees)

    arrays = {
        "slope_degrees": slope_degrees,
        "aspect_degrees": aspect_degrees,
        "hillshade": hillshade,
        "relative_relief_m": relief,
    }
    paths: dict[str, Path] = {}
    for name, array in arrays.items():
        path = out_dir / f"terrain_{name}_buffered.tif"
        write_float_raster(path, profile, array)
        paths[name] = path
    return paths


def make_hillshade(slope_rad: np.ndarray, aspect_degrees: np.ndarray, azimuth=315.0, altitude=45.0) -> np.ndarray:
    aspect_rad = np.deg2rad(aspect_degrees)
    azimuth_rad = np.deg2rad(360.0 - azimuth + 90.0)
    altitude_rad = np.deg2rad(altitude)
    shaded = 255.0 * (
        np.sin(altitude_rad) * np.cos(slope_rad)
        + np.cos(altitude_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
    )
    return np.clip(shaded, 0, 255)


def write_float_raster(path: Path, profile: dict, array: np.ndarray) -> None:
    out = np.where(np.isfinite(array), array, DEFAULT_NODATA).astype("float32")
    out_profile = profile.copy()
    out_profile.update(dtype="float32", nodata=DEFAULT_NODATA, compress="deflate", predictor=3, bigtiff="if_safer")
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(out, 1)


def step_7_visualize(args: argparse.Namespace, aoi, raster_paths: dict[str, Path], output_dir: Path):
    print("\nStep 7/7 - Create visualization and QA report")
    if args.skip_visuals:
        print("  Skipped by --skip-visuals.")
        return None

    report_dir = output_dir / "visualization_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    build_visual_report(aoi, raster_paths, report_dir, output_dir)
    print(f"  Report folder: {report_dir}")
    return report_dir


def build_visual_report(aoi, raster_paths: dict[str, Path], report_dir: Path, output_dir: Path) -> None:
    map_specs = {
        "elevation_m": ("Elevation (m)", "terrain", "terrain_map_elevation.png"),
        "hillshade": ("Hillshade", "gray", "terrain_map_hillshade.png"),
        "slope_degrees": ("Slope (degrees)", "magma", "terrain_map_slope_degrees.png"),
        "aspect_degrees": ("Aspect (degrees)", "hsv", "terrain_map_aspect_degrees.png"),
    }
    for key, (title, cmap, filename) in map_specs.items():
        make_raster_map(raster_paths[key], aoi, title, cmap, report_dir / filename)

    make_overview_panel(aoi, raster_paths, report_dir / "terrain_overview_panel.png")
    make_histogram(raster_paths["elevation_m"], "Elevation (m)", report_dir / "terrain_chart_elevation_histogram.png")
    make_histogram(raster_paths["slope_degrees"], "Slope (degrees)", report_dir / "terrain_chart_slope_histogram.png")
    write_file_summary(raster_paths, report_dir / "terrain_file_summary.csv", output_dir)
    write_stats_summary(raster_paths, report_dir / "terrain_stats_summary.csv")


def make_raster_map(path: Path, aoi, title: str, cmap: str, out_path: Path) -> None:
    arr, extent, crs = read_raster_for_plot(path)
    fig, ax = plt.subplots(figsize=(7.5, 9))
    image = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap)
    aoi.to_crs(crs).boundary.plot(ax=ax, color="black", linewidth=0.7)
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, shrink=0.72)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_overview_panel(aoi, raster_paths: dict[str, Path], out_path: Path) -> None:
    specs = [
        ("elevation_m", "Elevation (m)", "terrain"),
        ("hillshade", "Hillshade", "gray"),
        ("slope_degrees", "Slope (degrees)", "magma"),
        ("aspect_degrees", "Aspect (degrees)", "hsv"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, (key, title, cmap) in zip(axes.ravel(), specs):
        arr, extent, crs = read_raster_for_plot(raster_paths[key])
        ax.imshow(arr, extent=extent, origin="upper", cmap=cmap)
        aoi.to_crs(crs).boundary.plot(ax=ax, color="black", linewidth=0.55)
        ax.set_title(title)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_histogram(path: Path, label: str, out_path: Path) -> None:
    values = read_raster_values(path, max_values=2_000_000)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(values, bins=60, color="#477f6f", edgecolor="white", linewidth=0.4)
    ax.set_xlabel(label)
    ax.set_ylabel("Sampled pixels")
    ax.set_title(f"{label} distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def read_raster_for_plot(path: Path, max_pixels: int = 1_800_000):
    with rasterio.open(path) as src:
        factor = max(1, math.ceil(math.sqrt((src.width * src.height) / max_pixels)))
        out_height = max(1, src.height // factor)
        out_width = max(1, src.width // factor)
        arr = src.read(1, masked=True, out_shape=(out_height, out_width), resampling=Resampling.bilinear)
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        return arr, extent, src.crs


def read_raster_values(path: Path, max_values: int = 3_000_000) -> np.ndarray:
    with rasterio.open(path) as src:
        factor = max(1, math.ceil(math.sqrt((src.width * src.height) / max_values)))
        out_height = max(1, src.height // factor)
        out_width = max(1, src.width // factor)
        arr = src.read(1, masked=True, out_shape=(out_height, out_width), resampling=Resampling.bilinear)
    values = np.asarray(arr.compressed(), dtype="float64")
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise SystemExit(f"No valid raster values found in {path}")
    return values


def write_file_summary(raster_paths: dict[str, Path], out_path: Path, output_dir: Path) -> None:
    rows = []
    for name, path in raster_paths.items():
        with rasterio.open(path) as src:
            rows.append(
                {
                    "name": name,
                    "path": str(path.relative_to(output_dir)),
                    "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
                    "width": src.width,
                    "height": src.height,
                    "crs": str(src.crs),
                    "pixel_width": abs(src.transform.a),
                    "pixel_height": abs(src.transform.e),
                    "nodata": src.nodata,
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_stats_summary(raster_paths: dict[str, Path], out_path: Path) -> None:
    rows = []
    for name, path in raster_paths.items():
        values = read_raster_values(path)
        rows.append(
            {
                "name": name,
                "sampled_pixels": int(values.size),
                "min": float(np.min(values)),
                "p05": float(np.percentile(values, 5)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values)),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_run_summary(
    output_dir: Path,
    args: argparse.Namespace,
    area_acres: float,
    products: list[DemProduct],
    bbox_text: str,
    raster_paths: dict[str, Path],
    report_dir: Path | None,
) -> None:
    summary = {
        "input": str(args.input.expanduser().resolve()),
        "dem_dataset": args.dem_dataset,
        "map_crs": args.map_crs,
        "buffer_meters": args.buffer_meters,
        "aoi_area_acres": area_acres,
        "tnm_search_bbox": bbox_text,
        "tile_count": len(products),
        "outputs": {name: str(path) for name, path in raster_paths.items()},
        "visualization_report": None if report_dir is None else str(report_dir),
    }
    with (output_dir / "run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    check_input_args(args)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    aoi, area_acres = step_1_load_aoi(args, output_dir)
    products, search_aoi, bbox_text = step_2_find_dem_tiles(args, aoi, output_dir)
    tile_paths = step_3_download_tiles(args, products, output_dir)
    _mosaic_path, projected_dem = step_4_mosaic_and_project(args, tile_paths, search_aoi, output_dir)
    clipped_dem = step_5_clip_dem(aoi, projected_dem, output_dir)
    derivatives = step_6_build_derivatives(aoi, projected_dem, output_dir)

    raster_paths = {"elevation_m": clipped_dem, **derivatives}
    report_dir = step_7_visualize(args, aoi, raster_paths, output_dir)
    write_run_summary(output_dir, args, area_acres, products, bbox_text, raster_paths, report_dir)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
