#!/usr/bin/env python3
"""Build clipped SSURGO soil data for one AOI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from sda_soil_downloader import (
    download_tabular_csvs,
    load_aoi_gdf,
    require_geo_dependencies,
    resolve_shapefile_input,
    unique_mukeys,
)
from soil_visualization_report import build_report
from ssurgo_package_downloader import (
    download_ssurgo_zip,
    extract_ssurgo_zip,
    find_intersecting_survey_areas,
    get_survey_areas_from_sda,
    merge_frames,
    parse_areasymbols,
    read_and_clip_soilmu_a,
    write_summary,
    write_survey_area_manifest,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clipped SSURGO soil data.")

    parser.add_argument("--input", type=Path, required=True, help="AOI zipped shapefile.")
    parser.add_argument("--input-crs", help="CRS to assign if the AOI has no .prj/CRS.")

    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder for all products.")
    parser.add_argument(
        "--areasymbols",
        help="Optional known SSURGO SSA symbols, e.g. WI027,WI039. If omitted, they are found automatically.",
    )
    parser.add_argument("--equal-area-crs", default="EPSG:3071", help="CRS for Wisconsin area/clip work.")
    parser.add_argument("--output-crs", default="EPSG:4326", help="CRS for final GIS outputs.")
    parser.add_argument("--redownload", action="store_true", help="Re-download SSURGO zips even if present.")
    parser.add_argument("--skip-tabular", action="store_true", help="Skip mapunit/muaggatt/component CSV downloads.")
    parser.add_argument("--skip-visuals", action="store_true", help="Skip QA visualization report.")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-sleep", type=float, default=0.2)
    return parser.parse_args(argv)


def check_input_args(args: argparse.Namespace) -> None:
    if args.input.suffix.lower() != ".zip":
        raise SystemExit("Expected --input to be a zipped shapefile (.zip).")


def step_1_load_aoi(args: argparse.Namespace, gpd, output_dir: Path):
    print("\nStep 1/7 - Load AOI boundary")
    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    shapefile = resolve_shapefile_input(args, work_dir)
    aoi = load_aoi_gdf(gpd, shapefile, args.input_crs)
    print(f"  AOI features: {len(aoi):,}")
    print(f"  AOI CRS: {aoi.crs}")
    return aoi


def step_2_find_survey_areas(args: argparse.Namespace, gpd, shapely_module, requests_module, aoi, output_dir: Path):
    print("\nStep 2/7 - Find intersecting SSURGO Soil Survey Areas")
    known_areas = parse_areasymbols(args.areasymbols)

    if known_areas:
        areas = get_survey_areas_from_sda(requests_module, known_areas, args.retries, args.timeout)
        print("  Used provided areasymbols.")
    else:
        areas = find_intersecting_survey_areas(
            gpd,
            shapely_module,
            requests_module,
            aoi,
            output_dir,
            args.retries,
            args.timeout,
        )
        print("  Found areasymbols by intersecting AOI with NRCS Soil Data Availability polygons.")

    write_survey_area_manifest(output_dir / "survey_area_manifest.csv", areas)
    print("  SSAs:", ", ".join(area.areasymbol for area in areas))
    return areas


def step_3_and_4_download_and_clip(
    args: argparse.Namespace,
    gpd,
    shapely_module,
    requests_module,
    multi_polygon_factory,
    aoi,
    areas,
    output_dir: Path,
):
    print("\nStep 3/7 - Download complete SSURGO packages")
    print("Step 4/7 - Clip each soilmu_a layer to AOI")

    raw_zip_dir = output_dir / "raw_ssurgo_zips"
    extracted_dir = output_dir / "ssurgo_packages"
    clipped_dir = output_dir / "clipped_by_area"
    clipped_dir.mkdir(parents=True, exist_ok=True)

    clipped_frames = []
    for idx, area in enumerate(areas, start=1):
        print(f"  [{idx}/{len(areas)}] {area.areasymbol}")
        zip_path = download_ssurgo_zip(
            requests_module=requests_module,
            area=area,
            raw_dir=raw_zip_dir,
            redownload=args.redownload,
            retries=args.retries,
            timeout=args.timeout,
        )
        area_dir = extract_ssurgo_zip(zip_path, extracted_dir)
        clipped = read_and_clip_soilmu_a(
            gpd,
            shapely_module,
            multi_polygon_factory,
            area_dir,
            aoi,
            args.equal_area_crs,
            args.output_crs,
        )
        if clipped.empty:
            print("    No polygons after clip.")
            continue
        out_path = clipped_dir / f"{area.areasymbol}_soilmu_a_clipped.gpkg"
        if out_path.exists():
            out_path.unlink()
        clipped.to_file(out_path, layer="soilmu_a", driver="GPKG")
        clipped_frames.append(clipped)
        print(f"    Clipped polygons: {len(clipped):,}")

    if not clipped_frames:
        raise SystemExit("No clipped soil polygons were produced.")
    return clipped_frames


def step_5_merge(args: argparse.Namespace, gpd, clipped_frames, output_dir: Path):
    print("\nStep 5/7 - Merge clipped polygons")
    merged, merged_path = merge_frames(
        gpd,
        clipped_frames,
        output_dir,
        args.output_crs,
        write_final_shapefile=True,
    )
    print(f"  Merged polygons: {len(merged):,}")
    print(f"  Unique mukey: {len(unique_mukeys(merged)):,}")
    print(f"  Output: {merged_path}")
    return merged, merged_path


def step_6_download_tables(args: argparse.Namespace, requests_module, merged, output_dir: Path):
    print("\nStep 6/7 - Download key tabular attributes by mukey")
    if args.skip_tabular:
        print("  Skipped by --skip-tabular.")
        return None

    download_tabular_csvs(
        requests_module,
        unique_mukeys(merged),
        output_dir,
        include_horizons=False,
        retries=args.retries,
        timeout=args.timeout,
        request_sleep=args.request_sleep,
    )
    return output_dir / "tabular_csv" / "muaggatt.csv"


def step_7_visualize(args: argparse.Namespace, merged_path: Path, muaggatt_csv: Path | None, output_dir: Path):
    print("\nStep 7/7 - Create visualization and QA report")
    if args.skip_visuals:
        print("  Skipped by --skip-visuals.")
        return
    if muaggatt_csv is None or not muaggatt_csv.exists():
        print("  Skipped because muaggatt.csv is not available.")
        return

    report_args = argparse.Namespace(
        soil_gpkg=merged_path,
        soil_layer="soilmu_a",
        muaggatt_csv=muaggatt_csv,
        output_dir=output_dir / "visualization_report",
        map_crs=args.equal_area_crs,
        top_n=15,
    )
    build_report(report_args)
    print(f"  Report folder: {report_args.output_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    check_input_args(args)

    gpd, shapely_module, _pyproj, requests_module, factories = require_geo_dependencies()
    _box_factory, multi_polygon_factory, _transform_factory = factories

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    aoi = step_1_load_aoi(args, gpd, output_dir)
    areas = step_2_find_survey_areas(args, gpd, shapely_module, requests_module, aoi, output_dir)
    clipped_frames = step_3_and_4_download_and_clip(
        args,
        gpd,
        shapely_module,
        requests_module,
        multi_polygon_factory,
        aoi,
        areas,
        output_dir,
    )
    merged, merged_path = step_5_merge(args, gpd, clipped_frames, output_dir)
    muaggatt_csv = step_6_download_tables(args, requests_module, merged, output_dir)
    step_7_visualize(args, merged_path, muaggatt_csv, output_dir)
    write_summary(output_dir, areas, merged_path, merged)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
