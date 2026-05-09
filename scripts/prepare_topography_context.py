from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import rioxarray
import rasterio
from rasterio.merge import merge
from rasterio.io import MemoryFile
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_RASTER_EXTENSIONS = {".tif", ".tiff", ".bil", ".asc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align a real DEM to the PRISM grid and write static topography covariates."
    )
    parser.add_argument(
        "--dem-path",
        required=True,
        help="Path to a real DEM raster or a directory of DEM GeoTIFFs. Synthetic terrain is not accepted.",
    )
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--era5-path", default=None, help="ERA5 NetCDF path used to define the Georgia domain.")
    parser.add_argument("--prism-path", default=None, help="PRISM raster directory or file. Defaults to dataset version paths.")
    parser.add_argument("--output", default="data_processed/static/georgia_prism_topography.nc")
    parser.add_argument("--metadata-out", default=None)
    parser.add_argument(
        "--features",
        nargs="+",
        choices=["elevation", "slope", "aspect", "terrain_gradient_magnitude"],
        default=["elevation", "slope", "aspect", "terrain_gradient_magnitude"],
    )
    parser.add_argument("--source-name", default="USGS 3DEP", help="Human-readable DEM source label.")
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_prism_path(args: argparse.Namespace) -> Path:
    if args.prism_path is not None:
        return repo_path(args.prism_path)

    from datasets.dataset_paths import paths_for_dataset_version

    _, prism_path = paths_for_dataset_version(args.dataset_version, project_root=PROJECT_ROOT)
    return repo_path(prism_path)


def resolve_era5_path(args: argparse.Namespace) -> Path:
    if args.era5_path is not None:
        return repo_path(args.era5_path)

    from datasets.dataset_paths import paths_for_dataset_version

    era5_path, _ = paths_for_dataset_version(args.dataset_version, project_root=PROJECT_ROOT)
    return repo_path(era5_path)


def first_raster(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"PRISM path not found: {path}")
    candidates = [
        item
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.suffix.lower() in SUPPORTED_RASTER_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(f"No PRISM raster found under {path}")
    return candidates[0]


def raster_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"DEM path not found: {path}")
    files = [
        item
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.suffix.lower() in SUPPORTED_RASTER_EXTENSIONS
    ]
    if not files:
        raise FileNotFoundError(f"No DEM rasters found under {path}")
    return files


def read_single_band(path: Path) -> xr.DataArray:
    raster = rioxarray.open_rasterio(path, masked=True)
    if "band" in raster.dims:
        raster = raster.isel(band=0, drop=True)
    raster = raster.squeeze(drop=True)
    if raster.rio.crs is None:
        raise ValueError(f"Raster has no CRS metadata: {path}")
    return raster


def era5_bounds(path: Path) -> Tuple[float, float, float, float]:
    ds = xr.open_dataset(path)
    try:
        min_lon = float(ds["longitude"].min())
        max_lon = float(ds["longitude"].max())
        min_lat = float(ds["latitude"].min())
        max_lat = float(ds["latitude"].max())
    finally:
        ds.close()
    return min_lon, max_lon, min_lat, max_lat


def prism_template(prism_raster: Path, era5_path: Path) -> xr.DataArray:
    prism_ref = read_single_band(prism_raster)
    if prism_ref.rio.crs.to_epsg() != 4326:
        prism_ref = prism_ref.rio.reproject("EPSG:4326")
    min_lon, max_lon, min_lat, max_lat = era5_bounds(era5_path)
    clipped = prism_ref.rio.clip_box(minx=min_lon, maxx=max_lon, miny=min_lat, maxy=max_lat)
    if clipped.sizes.get("x", 0) == 0 or clipped.sizes.get("y", 0) == 0:
        raise ValueError("PRISM raster has no overlap with ERA5 bounds after clipping")
    return clipped


def read_dem(path: Path) -> xr.DataArray:
    files = raster_files(path)
    if len(files) == 1:
        return read_single_band(files[0])

    sources = [rasterio.open(item) for item in files]
    try:
        crs_values = {str(src.crs) for src in sources}
        if len(crs_values) != 1:
            raise ValueError(f"DEM tiles do not share one CRS: {sorted(crs_values)}")
        mosaic, transform = merge(sources)
        profile = sources[0].profile.copy()
        profile.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "count": 1,
                "dtype": str(mosaic.dtype),
            }
        )
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset:
                dataset.write(mosaic[0], 1)
            with memfile.open() as dataset:
                raster = rioxarray.open_rasterio(dataset, masked=True).load()
    finally:
        for src in sources:
            src.close()

    if "band" in raster.dims:
        raster = raster.isel(band=0, drop=True)
    raster = raster.squeeze(drop=True)
    if raster.rio.crs is None:
        raise ValueError(f"DEM mosaic has no CRS metadata: {path}")
    return raster


def finite_fill(values: np.ndarray) -> Tuple[np.ndarray, int, float]:
    arr = np.asarray(values, dtype=np.float32)
    invalid = (~np.isfinite(arr)) | (np.abs(arr) > 1.0e20)
    n_invalid = int(invalid.sum())
    if n_invalid == arr.size:
        raise ValueError("DEM alignment produced no finite elevation values")
    fill_value = float(np.nanmedian(np.where(~invalid, arr, np.nan)))
    if n_invalid:
        arr = arr.copy()
        arr[invalid] = fill_value
    return arr, n_invalid, fill_value


def terrain_derivatives(elevation: np.ndarray, aligned_dem: xr.DataArray) -> Dict[str, np.ndarray]:
    x_res_deg, y_res_deg = aligned_dem.rio.resolution()
    y_values = np.asarray(aligned_dem["y"].values, dtype=np.float64)
    mean_lat = float(np.nanmean(y_values)) if y_values.size else 0.0
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = max(1.0, 111_320.0 * np.cos(np.deg2rad(mean_lat)))
    x_res = abs(float(x_res_deg)) * meters_per_degree_lon
    y_res = abs(float(y_res_deg)) * meters_per_degree_lat
    if x_res == 0.0:
        x_res = 1.0
    if y_res == 0.0:
        y_res = 1.0
    dz_dy, dz_dx = np.gradient(elevation.astype(np.float64), y_res, x_res)
    gradient = np.sqrt(dz_dx**2 + dz_dy**2)
    slope = np.degrees(np.arctan(gradient))
    aspect = (np.degrees(np.arctan2(-dz_dx, dz_dy)) + 360.0) % 360.0
    return {
        "slope": slope.astype(np.float32),
        "aspect": aspect.astype(np.float32),
        "terrain_gradient_magnitude": gradient.astype(np.float32),
    }


def build_dataset(
    *,
    aligned_dem: xr.DataArray,
    features: Iterable[str],
    source_name: str,
) -> Tuple[xr.Dataset, Dict[str, object]]:
    elevation, n_filled, fill_value = finite_fill(aligned_dem.values)
    derived = terrain_derivatives(elevation, aligned_dem)
    feature_arrays: Dict[str, np.ndarray] = {"elevation": elevation, **derived}

    selected = list(features)
    data_vars = {
        name: (("y", "x"), feature_arrays[name])
        for name in selected
    }
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "y": aligned_dem["y"].values,
            "x": aligned_dem["x"].values,
        },
        attrs={
            "source_name": source_name,
            "crs": str(aligned_dem.rio.crs),
            "note": "Static DEM-derived covariates aligned to the PRISM grid. Not normalized.",
        },
    )
    ds = ds.rio.write_crs(aligned_dem.rio.crs)

    metadata: Dict[str, object] = {
        "source_name": source_name,
        "shape": [int(elevation.shape[0]), int(elevation.shape[1])],
        "features": selected,
        "crs": str(aligned_dem.rio.crs),
        "resolution": [float(v) for v in aligned_dem.rio.resolution()],
        "filled_nonfinite_pixels": n_filled,
        "fill_value": fill_value,
        "feature_stats": {
            name: {
                "mean": float(np.mean(feature_arrays[name])),
                "std": float(np.std(feature_arrays[name])),
                "min": float(np.min(feature_arrays[name])),
                "max": float(np.max(feature_arrays[name])),
            }
            for name in selected
        },
        "normalization_note": "Feature stats are full-domain raw stats. Training code computes input normalization from train indices.",
    }
    return ds, metadata


def main() -> None:
    args = parse_args()
    dem_path = repo_path(args.dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM path not found: {dem_path}")

    prism_raster = first_raster(resolve_prism_path(args))
    era5_path = resolve_era5_path(args)
    dem = read_dem(dem_path)
    prism_ref = prism_template(prism_raster, era5_path)
    aligned_dem = dem.rio.reproject_match(prism_ref)

    ds, metadata = build_dataset(
        aligned_dem=aligned_dem,
        features=args.features,
        source_name=str(args.source_name),
    )
    metadata.update(
        {
            "dem_path": str(dem_path),
            "prism_reference": str(prism_raster),
            "era5_reference": str(era5_path),
            "dataset_version": str(args.dataset_version),
            "alignment_method": "DEM reproject_match to PRISM raster clipped to ERA5 bounds",
        }
    )

    output_path = repo_path(args.output)
    metadata_path = repo_path(args.metadata_out) if args.metadata_out else output_path.with_suffix(".metadata.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"wrote topography features: {output_path}")
    print(f"wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
