from __future__ import annotations

import argparse
import calendar
from datetime import date, datetime, timedelta
from pathlib import Path
import socket
import tempfile
import zipfile
from typing import Iterator, List, Tuple

import numpy as np
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ERA5 surface variables for Georgia")
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Inclusive range start YYYYMMDD (requires --end-date). Uses same bbox/variables as monthly mode.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Inclusive range end YYYYMMDD (requires --start-date). Full calendar months are downloaded then time-sliced.",
    )
    parser.add_argument("--year", type=int, default=2023, help="Single-month mode: year (default: 2023)")
    parser.add_argument("--month", type=int, default=1, help="Single-month mode: month 1-12 (default: 1)")
    parser.add_argument(
        "--out",
        type=str,
        default="data_raw/era5_georgia_multi.nc",
        help="Output NetCDF path",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing ERA5 NetCDF at --out without downloading",
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


def _normalize_time_coord(ds: xr.Dataset) -> xr.Dataset:
    if "time" in ds.dims or "time" in ds.coords:
        return ds
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        return ds.rename({"valid_time": "time"})
    return ds


def _validate_era5_dataset(ds: xr.Dataset, *, required_vars: list[str]) -> None:
    ds = _normalize_time_coord(ds)
    missing = [v for v in required_vars if v not in ds.data_vars]
    if missing:
        raise RuntimeError(f"ERA5 dataset missing required variables: {missing}. Found: {list(ds.data_vars)}")

    for v in required_vars:
        da = ds[v]
        for dim in ("time", "latitude", "longitude"):
            if dim not in da.dims:
                raise RuntimeError(f"Variable '{v}' missing required dim '{dim}'. dims={list(da.dims)}")
        if da.dims != ("time", "latitude", "longitude"):
            da = da.transpose("time", "latitude", "longitude")

        values = da.values
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite values found in ERA5 variable '{v}'")

    time_vals = ds["time"].values if "time" in ds.coords else None
    if time_vals is None or len(time_vals) < 2:
        raise RuntimeError("ERA5 dataset has no usable time coordinate")
    if not np.all(np.diff(time_vals.astype("datetime64[ns]")) > np.timedelta64(0, "ns")):
        raise RuntimeError("ERA5 time coordinate is not strictly increasing")

    shapes = {v: tuple(int(ds[v].sizes[d]) for d in ("time", "latitude", "longitude")) for v in required_vars}
    first_shape = next(iter(shapes.values()))
    mismatched = {v: s for v, s in shapes.items() if s != first_shape}
    if mismatched:
        raise RuntimeError(f"ERA5 variable shapes do not match: {mismatched}")


def _print_basic_stats(ds: xr.Dataset, vars_to_print: list[str]) -> None:
    ds = _normalize_time_coord(ds)
    print(f"ERA5 variables: {list(ds.data_vars)}")
    for v in vars_to_print:
        if v not in ds.data_vars:
            continue
        da = ds[v].transpose("time", "latitude", "longitude")
        arr = da.values.astype(np.float64)
        print(
            f"{v}: shape={arr.shape} min={np.nanmin(arr):.4f} max={np.nanmax(arr):.4f} "
            f"mean={np.nanmean(arr):.4f} std={np.nanstd(arr):.4f}"
        )


def _iter_year_month(d0: date, d1: date) -> Iterator[Tuple[int, int]]:
    y, m = d0.year, d0.month
    end_key = (d1.year, d1.month)
    while (y, m) <= end_key:
        yield y, m
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1


def _build_output_ds_from_raw(single_ds: xr.Dataset, pressure_ds: xr.Dataset) -> xr.Dataset:
    single_ds = _normalize_time_coord(single_ds)
    pressure_ds = _normalize_time_coord(pressure_ds)
    level_dim = _infer_level_dim(pressure_ds)

    z_var = "z" if "z" in pressure_ds.data_vars else "geopotential"
    t_var = "t" if "t" in pressure_ds.data_vars else "temperature"
    r_var = "r" if "r" in pressure_ds.data_vars else "relative_humidity"
    t2m_var = "t2m" if "t2m" in single_ds.data_vars else "2m_temperature"
    d2m_var = "d2m" if "d2m" in single_ds.data_vars else "2m_dewpoint_temperature"

    rename_single = {
        t2m_var: "t2m",
        d2m_var: "d2m",
        "u10": "u10",
        "10m_u_component_of_wind": "u10",
        "v10": "v10",
        "10m_v_component_of_wind": "v10",
        "sp": "sp",
        "surface_pressure": "sp",
        "tp": "tp",
        "total_precipitation": "tp",
    }
    applied = {k: v for k, v in rename_single.items() if k in single_ds.data_vars and k != v}
    if applied:
        single_ds = single_ds.rename(applied)

    t2m_c = single_ds[t2m_var] - 273.15 if float(single_ds[t2m_var].mean(skipna=True)) > 150.0 else single_ds[t2m_var]
    d2m_c = single_ds[d2m_var] - 273.15 if float(single_ds[d2m_var].mean(skipna=True)) > 150.0 else single_ds[d2m_var]
    rh2m = _daily_relative_humidity(t2m_c, d2m_c).rename("rh2m")

    output_ds = _normalize_time_coord(single_ds.copy())
    output_ds["rh2m"] = rh2m
    output_ds["temperature_850"] = pressure_ds[t_var].sel({level_dim: 850}).squeeze(drop=True)
    output_ds["temperature_500"] = pressure_ds[t_var].sel({level_dim: 500}).squeeze(drop=True)
    output_ds["geopotential_height_850"] = (pressure_ds[z_var].sel({level_dim: 850}).squeeze(drop=True) / 9.80665)
    output_ds["geopotential_height_500"] = (pressure_ds[z_var].sel({level_dim: 500}).squeeze(drop=True) / 9.80665)
    output_ds["relative_humidity_850"] = pressure_ds[r_var].sel({level_dim: 850}).squeeze(drop=True)
    output_ds["relative_humidity_500"] = pressure_ds[r_var].sel({level_dim: 500}).squeeze(drop=True)
    return output_ds


def _download_one_calendar_month(
    client: object,
    *,
    year: int,
    month: int,
    area: List[float],
    tmp_dir_path: Path,
) -> xr.Dataset:
    days_in_month = calendar.monthrange(year, month)[1]
    days = [f"{day:02d}" for day in range(1, days_in_month + 1)]
    hours = [f"{hour:02d}:00" for hour in range(24)]
    single_path = tmp_dir_path / f"era5_single_{year}_{month:02d}.nc"
    pressure_path = tmp_dir_path / f"era5_pressure_{year}_{month:02d}.nc"

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
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": hours,
            "area": area,
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
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": hours,
            "area": area,
            "format": "netcdf",
        },
        str(pressure_path),
    )

    if not single_path.exists() or single_path.stat().st_size == 0:
        raise RuntimeError(f"ERA5 single-level download missing or empty for {year}-{month:02d}")
    if not pressure_path.exists() or pressure_path.stat().st_size == 0:
        raise RuntimeError(f"ERA5 pressure-level download missing or empty for {year}-{month:02d}")

    single_ds = _open_cds_dataset(single_path, tmp_dir_path / f"single_extract_{year}_{month:02d}")
    pressure_ds = _open_cds_dataset(pressure_path, tmp_dir_path / f"pressure_extract_{year}_{month:02d}")
    return _build_output_ds_from_raw(single_ds, pressure_ds)


def _slice_time_inclusive(ds: xr.Dataset, d0: date, d1: date) -> xr.Dataset:
    ds = _normalize_time_coord(ds)
    t0 = np.datetime64(f"{d0.isoformat()}T00:00:00")
    t1 = np.datetime64(f"{d1.isoformat()}T23:59:59")
    return ds.sel(time=slice(t0, t1))


def main() -> None:
    args = parse_args()
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        if not output_path.exists():
            raise FileNotFoundError(f"ERA5 file not found for validation: {output_path}")
        ds = xr.open_dataset(output_path)
        required = ["t2m", "u10", "v10", "sp"]
        _validate_era5_dataset(ds, required_vars=required)
        _print_basic_stats(ds, vars_to_print=required + ["tp", "rh2m"])
        print("ERA5 validation OK")
        return

    use_range = args.start_date is not None and args.end_date is not None
    if (args.start_date is not None) ^ (args.end_date is not None):
        raise ValueError("Provide both --start-date and --end-date, or neither (use --year/--month).")
    if not use_range and not (1 <= args.month <= 12):
        raise ValueError("--month must be in 1..12 in single-month mode")

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

    client = cdsapi.Client()
    area = list(args.area)

    if use_range:
        d0 = datetime.strptime(args.start_date, "%Y%m%d").date()
        d1 = datetime.strptime(args.end_date, "%Y%m%d").date()
        if d1 < d0:
            raise ValueError("--end-date must be on or after --start-date")
        print("Starting ERA5 multi-month download...")
        print(f"  range={args.start_date}..{args.end_date} (inclusive)")
        print(f"  area={area}")
        print(f"  output={output_path}")

        month_parts: List[xr.Dataset] = []
        with tempfile.TemporaryDirectory(prefix="era5_dl_range_") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            for y, m in _iter_year_month(d0, d1):
                print(f"  downloading {y}-{m:02d} ...")
                try:
                    ds_m = _download_one_calendar_month(client, year=y, month=m, area=area, tmp_dir_path=tmp_dir_path)
                except Exception as exc:
                    raise RuntimeError(f"ERA5 download failed for {y}-{m:02d}: {exc}") from exc
                month_parts.append(ds_m.load())

        merged = xr.concat(month_parts, dim="time", data_vars="minimal", coords="minimal", compat="override")
        merged = _normalize_time_coord(merged)
        merged = merged.sortby("time")
        times = merged["time"].values
        _, unique_idx = np.unique(times, return_index=True)
        merged = merged.isel(time=np.sort(unique_idx))
        merged = _slice_time_inclusive(merged, d0, d1)
        merged.to_netcdf(output_path)
    else:
        print("Starting ERA5 download (single calendar month)...")
        print(f"  year={args.year} month={args.month:02d}")
        print(f"  area={area}")
        print(f"  output={output_path}")

        with tempfile.TemporaryDirectory(prefix="era5_dl_") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            try:
                ds_out = _download_one_calendar_month(
                    client, year=args.year, month=args.month, area=area, tmp_dir_path=tmp_dir_path
                )
            except Exception as exc:
                raise RuntimeError(f"ERA5 download failed: {exc}") from exc
            ds_out = ds_out.load()
            ds_out.to_netcdf(output_path)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("ERA5 merged file is missing or empty after processing")

    ds = _normalize_time_coord(xr.open_dataset(output_path))
    required = ["t2m", "u10", "v10", "sp"]
    _validate_era5_dataset(ds, required_vars=required)
    _print_basic_stats(ds, vars_to_print=required + ["tp", "rh2m"])
    print(f"Saved ERA5 file to: {output_path}")


if __name__ == "__main__":
    main()
