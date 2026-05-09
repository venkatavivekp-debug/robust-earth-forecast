from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import rioxarray
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_RASTER_EXTENSIONS = {".tif", ".tiff", ".bil", ".asc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align a real DEM to the PRISM grid and write static topography covariates."
    )
    parser.add_argument("--dem-path", required=True, help="Path to a real DEM raster. Synthetic terrain is not accepted.")
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--prism-path", default=None, help="PRISM raster directory or file. Defaults to dataset version paths.")
    parser.add_argument("--output", default="data_raw/static/topography/georgia_prism_topography.nc")
    parser.add_argument("--metadata-out", default=None)
    parser.add_argument(
        "--features",
        nargs="+",
        choices=["elevation", "slope", "aspect", "terrain_gradient"],
        default=["elevation", "slope", "aspect", "terrain_gradient"],
    )
    parser.add_argument("--source-name", default="unverified_dem", help="Human-readable DEM source label.")
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


def read_single_band(path: Path) -> xr.DataArray:
    raster = rioxarray.open_rasterio(path, masked=True)
    if "band" in raster.dims:
        raster = raster.isel(band=0, drop=True)
    raster = raster.squeeze(drop=True)
    if raster.rio.crs is None:
        raise ValueError(f"Raster has no CRS metadata: {path}")
    return raster


def finite_fill(values: np.ndarray) -> Tuple[np.ndarray, int, float]:
    arr = np.asarray(values, dtype=np.float32)
    invalid = ~np.isfinite(arr)
    n_invalid = int(invalid.sum())
    if n_invalid == arr.size:
        raise ValueError("DEM alignment produced no finite elevation values")
    fill_value = float(np.nanmedian(np.where(np.isfinite(arr), arr, np.nan)))
    if n_invalid:
        arr = arr.copy()
        arr[invalid] = fill_value
    return arr, n_invalid, fill_value


def terrain_derivatives(elevation: np.ndarray, resolution: Tuple[float, float]) -> Dict[str, np.ndarray]:
    x_res = abs(float(resolution[0])) if resolution[0] else 1.0
    y_res = abs(float(resolution[1])) if resolution[1] else 1.0
    dz_dy, dz_dx = np.gradient(elevation.astype(np.float64), y_res, x_res)
    gradient = np.sqrt(dz_dx**2 + dz_dy**2)
    slope = np.degrees(np.arctan(gradient))
    aspect = (np.degrees(np.arctan2(-dz_dx, dz_dy)) + 360.0) % 360.0
    return {
        "slope": slope.astype(np.float32),
        "aspect": aspect.astype(np.float32),
        "terrain_gradient": gradient.astype(np.float32),
    }


def build_dataset(
    *,
    aligned_dem: xr.DataArray,
    features: Iterable[str],
    source_name: str,
) -> Tuple[xr.Dataset, Dict[str, object]]:
    elevation, n_filled, fill_value = finite_fill(aligned_dem.values)
    derived = terrain_derivatives(elevation, aligned_dem.rio.resolution())
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
    }
    return ds, metadata


def main() -> None:
    args = parse_args()
    dem_path = repo_path(args.dem_path)
    if not dem_path.is_file():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")

    prism_raster = first_raster(resolve_prism_path(args))
    dem = read_single_band(dem_path)
    prism_ref = read_single_band(prism_raster)
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
            "dataset_version": str(args.dataset_version),
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
