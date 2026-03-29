from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rioxarray
import torch
import xarray as xr
from torch.utils.data import Dataset

SUPPORTED_PRISM_EXTENSIONS = {".tif", ".tiff", ".bil", ".asc"}
TIME_DIM_CANDIDATES = ("time", "valid_time", "date")
LAT_DIM_CANDIDATES = ("latitude", "lat", "y")
LON_DIM_CANDIDATES = ("longitude", "lon", "x")
DATE_PATTERN = re.compile(r"(19|20)\d{6}")


@dataclass(frozen=True)
class SampleMetadata:
    index: int
    date: pd.Timestamp
    prism_path: str
    era5_shape: Tuple[int, int]
    prism_shape: Tuple[int, int]


class ERA5_PRISM_Dataset(Dataset):
    """Paired ERA5->PRISM temperature dataset for baseline spatial downscaling.

    Alignment strategy:
    1) ERA5 NetCDF temperature is aggregated to daily means.
    2) PRISM rasters are read via rioxarray, reprojected to EPSG:4326, and
       clipped to the ERA5 bounding box.
    3) Samples are paired by date parsed from each PRISM file name (YYYYMMDD).
    """

    def __init__(
        self,
        era5_path: str,
        prism_path: str,
        era5_variable: Optional[str] = None,
        auto_scale_prism: bool = True,
    ) -> None:
        self.era5_path = Path(era5_path)
        self.prism_path = Path(prism_path)
        self.auto_scale_prism = auto_scale_prism

        if not self.era5_path.exists():
            raise FileNotFoundError(f"ERA5 file not found: {self.era5_path}")

        # Convert hourly ERA5 into daily fields used for date-wise pairing.
        self.era5_daily = self._load_era5_daily(self.era5_path, era5_variable)
        self.era5_dates = pd.to_datetime(self.era5_daily["time"].values).normalize()
        self.era5_date_set = set(self.era5_dates)
        self.era5_bounds = self._get_era5_bounds(self.era5_daily)

        prism_files = self._resolve_prism_files(self.prism_path)
        self._samples: List[Tuple[np.ndarray, np.ndarray, pd.Timestamp, Path]] = []
        parsed_prism_dates: List[pd.Timestamp] = []

        # First valid PRISM raster defines the target grid for all following dates.
        template_raster = None
        for prism_file in prism_files:
            sample_date = self._parse_date_from_filename(prism_file)
            if sample_date is None:
                raise ValueError(
                    f"Could not parse YYYYMMDD date from PRISM file name: {prism_file.name}"
                )
            parsed_prism_dates.append(sample_date)

            if sample_date not in self.era5_date_set:
                continue

            prism_da = self._open_prism_raster(prism_file)
            prism_da = self._clip_prism_to_era5(prism_da, self.era5_bounds)

            if template_raster is None:
                template_raster = prism_da
            else:
                prism_da = prism_da.rio.reproject_match(template_raster)

            prism_array = prism_da.values.astype(np.float32)
            prism_array = self._prepare_prism_values(prism_array)

            era5_array = (
                self.era5_daily.sel(time=np.datetime64(sample_date)).values.astype(np.float32)
            )
            era5_array = self._fill_missing(era5_array)

            self._samples.append((era5_array, prism_array, sample_date, prism_file))

        if not self._samples:
            era5_start = self.era5_dates.min().strftime("%Y-%m-%d")
            era5_end = self.era5_dates.max().strftime("%Y-%m-%d")
            prism_preview = ", ".join(sorted({d.strftime("%Y-%m-%d") for d in parsed_prism_dates})[:5])
            raise RuntimeError(
                "No aligned ERA5/PRISM pairs were created. "
                f"ERA5 date range: {era5_start} to {era5_end}. "
                f"First PRISM dates found: {prism_preview or 'none'}. "
                "Check PRISM filenames/dates and temporal overlap."
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        era5_array, prism_array, _, _ = self._samples[idx]
        x = torch.from_numpy(era5_array).unsqueeze(0)
        y = torch.from_numpy(prism_array).unsqueeze(0)
        return x, y

    def metadata(self, idx: int) -> SampleMetadata:
        era5_array, prism_array, sample_date, prism_file = self._samples[idx]
        return SampleMetadata(
            index=idx,
            date=sample_date,
            prism_path=str(prism_file),
            era5_shape=tuple(int(v) for v in era5_array.shape),
            prism_shape=tuple(int(v) for v in prism_array.shape),
        )

    @staticmethod
    def _find_first_dim(dims: Sequence[str], candidates: Sequence[str], label: str) -> str:
        for name in candidates:
            if name in dims:
                return name
        raise ValueError(f"Could not infer {label} dimension from {list(dims)}")

    def _load_era5_daily(self, era5_path: Path, era5_variable: Optional[str]) -> xr.DataArray:
        try:
            era5_ds = xr.open_dataset(era5_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to open ERA5 NetCDF: {era5_path}") from exc

        if era5_variable is None:
            era5_variable = "t2m" if "t2m" in era5_ds.data_vars else list(era5_ds.data_vars)[0]

        if era5_variable not in era5_ds.data_vars:
            raise ValueError(
                f"ERA5 variable '{era5_variable}' not found. Available: {list(era5_ds.data_vars)}"
            )

        era5_da = era5_ds[era5_variable]

        time_dim = self._find_first_dim(era5_da.dims, TIME_DIM_CANDIDATES, "time")
        lat_dim = self._find_first_dim(era5_da.dims, LAT_DIM_CANDIDATES, "latitude")
        lon_dim = self._find_first_dim(era5_da.dims, LON_DIM_CANDIDATES, "longitude")

        era5_da = era5_da.transpose(time_dim, lat_dim, lon_dim)

        if float(era5_da.mean()) > 150.0:
            era5_da = era5_da - 273.15

        era5_daily = era5_da.resample({time_dim: "1D"}).mean(keep_attrs=True)
        era5_daily = era5_daily.rename(
            {time_dim: "time", lat_dim: "latitude", lon_dim: "longitude"}
        )
        return era5_daily

    @staticmethod
    def _get_era5_bounds(era5_daily: xr.DataArray) -> Tuple[float, float, float, float]:
        min_lon = float(era5_daily["longitude"].min())
        max_lon = float(era5_daily["longitude"].max())
        min_lat = float(era5_daily["latitude"].min())
        max_lat = float(era5_daily["latitude"].max())
        return min_lon, max_lon, min_lat, max_lat

    @staticmethod
    def _resolve_prism_files(prism_path: Path) -> List[Path]:
        if prism_path.is_dir():
            files = sorted(
                p for p in prism_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_PRISM_EXTENSIONS
            )
            if files:
                return files

            zip_files = sorted(prism_path.rglob("*.zip"))
            if zip_files:
                raise ValueError(
                    "Found PRISM zip archives but no extracted raster files. "
                    "Extract the archives first (e.g., unzip PRISM_*.zip -d data_raw/prism)."
                )

            raise FileNotFoundError(
                f"No supported PRISM rasters found under {prism_path}. "
                f"Supported extensions: {sorted(SUPPORTED_PRISM_EXTENSIONS)}"
            )

        if prism_path.is_file():
            suffix = prism_path.suffix.lower()
            if suffix == ".zip":
                raise ValueError(
                    f"PRISM path points to zip archive: {prism_path}. "
                    "Extract it first and pass either the extracted raster file or containing directory."
                )
            if suffix not in SUPPORTED_PRISM_EXTENSIONS:
                raise ValueError(
                    f"Unsupported PRISM file type: {prism_path.suffix}. "
                    f"Supported extensions: {sorted(SUPPORTED_PRISM_EXTENSIONS)}"
                )
            return [prism_path]

        raise FileNotFoundError(f"PRISM path does not exist: {prism_path}")

    @staticmethod
    def _open_prism_raster(prism_file: Path) -> xr.DataArray:
        try:
            prism_da = rioxarray.open_rasterio(prism_file, masked=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read PRISM raster '{prism_file}'. Ensure the file is a valid scientific raster."
            ) from exc

        if "band" in prism_da.dims:
            if prism_da.sizes.get("band", 0) < 1:
                raise ValueError(f"PRISM raster has no readable bands: {prism_file}")
            prism_da = prism_da.isel(band=0)

        if prism_da.rio.crs is None:
            raise ValueError(f"PRISM raster is missing CRS information: {prism_file}")

        if prism_da.rio.crs.to_epsg() != 4326:
            prism_da = prism_da.rio.reproject("EPSG:4326")

        return prism_da.astype(np.float32)

    @staticmethod
    def _clip_prism_to_era5(
        prism_da: xr.DataArray, era5_bounds: Tuple[float, float, float, float]
    ) -> xr.DataArray:
        min_lon, max_lon, min_lat, max_lat = era5_bounds
        try:
            clipped = prism_da.rio.clip_box(
                minx=min_lon,
                maxx=max_lon,
                miny=min_lat,
                maxy=max_lat,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to clip PRISM raster to ERA5 bounds. "
                "Verify that both datasets overlap in Georgia and use valid georeferencing."
            ) from exc

        if clipped.sizes.get("x", 0) == 0 or clipped.sizes.get("y", 0) == 0:
            raise ValueError(
                "PRISM raster has no spatial overlap with ERA5 extent after clipping."
            )

        return clipped

    @staticmethod
    def _parse_date_from_filename(file_path: Path) -> Optional[pd.Timestamp]:
        match = DATE_PATTERN.search(file_path.name)
        if not match:
            return None
        return pd.Timestamp(pd.to_datetime(match.group(0), format="%Y%m%d")).normalize()

    def _prepare_prism_values(self, prism_array: np.ndarray) -> np.ndarray:
        prism_array = self._fill_missing(prism_array)

        # Some PRISM sources store temperatures in hundredths of degrees C.
        if self.auto_scale_prism:
            p95 = float(np.percentile(np.abs(prism_array), 95))
            if p95 > 500.0:
                prism_array = prism_array / 100.0

        return prism_array.astype(np.float32)

    @staticmethod
    def _fill_missing(array: np.ndarray) -> np.ndarray:
        if not np.isnan(array).any():
            return array

        finite = array[np.isfinite(array)]
        if finite.size == 0:
            raise ValueError("Encountered raster with all values as NaN")

        return np.nan_to_num(array, nan=float(finite.mean())).astype(np.float32)
