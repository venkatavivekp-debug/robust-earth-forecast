from __future__ import annotations

import argparse
import calendar
from pathlib import Path
import socket
import tempfile
import zipfile

import numpy as np
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ERA5 surface variables for Georgia")
    parser.add_argument("--year", type=int, default=2023, help="Year (default: 2023)")
    parser.add_argument("--month", type=int, default=1, help="Month 1-12 (default: 1)")
    parser.add_argument(
        "--out",
        type=str,
        default="data_raw/era5_georgia_temp.nc",
        help="Output NetCDF path (default: data_raw/era5_georgia_temp.nc)",
    )
    parser.add_argument(
        "--area",
        type=float,
        nargs=4,
        default=[35.0, -85.0, 30.0, -80.0],
        metavar=("N", "W", "S", "E"),
        help="Bounding box [north west south east] for Georgia",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload even if output file already exists",
    )
    return parser.parse_args()


def _infer_level_dim(ds: xr.Dataset) -> str:
    for name in ("pressure_level", "level"):
        if name in ds.dims:
            return name
    raise ValueError(f"Could not infer pressure level dimension from {list(ds.dims)}")


def _daily_relative_humidity(t_c: xr.DataArray, td_c: xr.DataArray) -> xr.DataArray:
    # Magnus formula
    a = 17.625
    b = 243.04
    es = np.exp((a * t_c) / (b + t_c))
    e = np.exp((a * td_c) / (b + td_c))
    rh = 100.0 * (e / es)
    return rh.clip(min=0.0, max=100.0).astype(np.float32)


def _open_cds_dataset(path: Path, extract_dir: Path) -> xr.Dataset:
    if not zipfile.is_zipfile(path):
        return xr.open_dataset(path, engine="netcdf4")

    with zipfile.ZipFile(path, "r") as archive:
        archive.extractall(extract_dir)

    nc_files = sorted(extract_dir.rglob("*.nc"))
    if not nc_files:
        raise RuntimeError(f"Downloaded ZIP archive {path} does not contain a NetCDF file")

    datasets = [xr.open_dataset(nc_file, engine="netcdf4") for nc_file in nc_files]
    return xr.merge(datasets, compat="override")


def main() -> None:
    args = parse_args()

    if not 1 <= args.month <= 12:
        raise ValueError("--month must be in 1..12")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(f"ERA5 file already exists, skipping download: {output_path}")
        print("Use --overwrite to force a new download.")
        return

    cdsapirc_path = Path.home() / ".cdsapirc"
    if not cdsapirc_path.exists():
        raise RuntimeError(
            "ERA5 download requires CDS API credentials. "
            "Create ~/.cdsapirc first: https://cds.climate.copernicus.eu/how-to-api"
        )

    # Avoid long CDS retry loops when network/DNS is unavailable.
    try:
        socket.gethostbyname("cds.climate.copernicus.eu")
    except OSError as exc:
        raise RuntimeError(
            "Cannot resolve cds.climate.copernicus.eu. "
            "Check internet/DNS connectivity before downloading ERA5."
        ) from exc

    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'cdsapi'. Install requirements first: pip install -r requirements.txt"
        ) from exc

    days_in_month = calendar.monthrange(args.year, args.month)[1]

    print("Starting ERA5 download...")
    print(f"  year={args.year} month={args.month:02d}")
    print(f"  area={args.area}")
    print(f"  output={output_path}")

    days = [f"{day:02d}" for day in range(1, days_in_month + 1)]
    hours = [f"{hour:02d}:00" for hour in range(24)]
    with tempfile.TemporaryDirectory(prefix="era5_dl_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        single_path = tmp_dir_path / "era5_single.nc"
        pressure_path = tmp_dir_path / "era5_pressure.nc"

        try:
            client = cdsapi.Client()
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "2m_temperature",
                        "2m_dewpoint_temperature",
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                        "surface_pressure",
                        "total_precipitation",
                    ],
                    "year": str(args.year),
                    "month": f"{args.month:02d}",
                    "day": days,
                    "time": hours,
                    "area": args.area,
                    "format": "netcdf",
                },
                str(single_path),
            )
            client.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": "reanalysis",
                    "variable": ["temperature", "geopotential", "relative_humidity"],
                    "pressure_level": ["500", "850"],
                    "year": str(args.year),
                    "month": f"{args.month:02d}",
                    "day": days,
                    "time": hours,
                    "area": args.area,
                    "format": "netcdf",
                },
                str(pressure_path),
            )
        except Exception as exc:
            raise RuntimeError(f"ERA5 download failed: {exc}") from exc

        if not single_path.exists() or single_path.stat().st_size == 0:
            raise RuntimeError("ERA5 single-level download output is missing or empty")
        if not pressure_path.exists() or pressure_path.stat().st_size == 0:
            raise RuntimeError("ERA5 pressure-level download output is missing or empty")

        single_ds = _open_cds_dataset(single_path, tmp_dir_path / "single_extract")
        pressure_ds = _open_cds_dataset(pressure_path, tmp_dir_path / "pressure_extract")
        level_dim = _infer_level_dim(pressure_ds)

        z_var = "z" if "z" in pressure_ds.data_vars else "geopotential"
        t_var = "t" if "t" in pressure_ds.data_vars else "temperature"
        r_var = "r" if "r" in pressure_ds.data_vars else "relative_humidity"
        t2m_var = "t2m" if "t2m" in single_ds.data_vars else "2m_temperature"
        d2m_var = "d2m" if "d2m" in single_ds.data_vars else "2m_dewpoint_temperature"

        t2m_c = single_ds[t2m_var] - 273.15 if float(single_ds[t2m_var].mean(skipna=True)) > 150.0 else single_ds[t2m_var]
        d2m_c = single_ds[d2m_var] - 273.15 if float(single_ds[d2m_var].mean(skipna=True)) > 150.0 else single_ds[d2m_var]
        rh2m = _daily_relative_humidity(t2m_c, d2m_c).rename("rh2m")

        output_ds = single_ds.copy()
        output_ds["rh2m"] = rh2m
        output_ds["temperature_850"] = pressure_ds[t_var].sel({level_dim: 850}).squeeze(drop=True)
        output_ds["temperature_500"] = pressure_ds[t_var].sel({level_dim: 500}).squeeze(drop=True)
        output_ds["geopotential_height_850"] = (pressure_ds[z_var].sel({level_dim: 850}).squeeze(drop=True) / 9.80665)
        output_ds["geopotential_height_500"] = (pressure_ds[z_var].sel({level_dim: 500}).squeeze(drop=True) / 9.80665)
        output_ds["relative_humidity_850"] = pressure_ds[r_var].sel({level_dim: 850}).squeeze(drop=True)
        output_ds["relative_humidity_500"] = pressure_ds[r_var].sel({level_dim: 500}).squeeze(drop=True)

        output_ds.to_netcdf(output_path)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("ERA5 merged file is missing or empty after processing")

    print(f"Saved ERA5 file to: {output_path}")


if __name__ == "__main__":
    main()
