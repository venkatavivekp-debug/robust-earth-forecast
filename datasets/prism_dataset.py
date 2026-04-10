from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

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
class SampleRecord:
    index: int
    date: pd.Timestamp
    prism_path: str
    era5_start_idx: int
    era5_end_idx: int


@dataclass(frozen=True)
class SampleMetadata:
    index: int
    date: pd.Timestamp
    prism_path: str
    era5_shape: Tuple[int, ...]
    prism_shape: Tuple[int, ...]


class ERA5_PRISM_Dataset(Dataset):
    """Temporal ERA5->PRISM dataset for regional downscaling.

    Input  X: [T, 1, H_era5, W_era5]
    Target Y: [1, H_prism, W_prism]
    """

    def __init__(
        self,
        era5_path: str,
        prism_path: str,
        history_length: int = 3,
        era5_variable: Optional[str] = None,
        auto_scale_prism: bool = True,
        verbose: bool = True,
    ) -> None:
        self.era5_path = Path(era5_path)
        self.prism_path = Path(prism_path)
        self.history_length = int(history_length)
        self.auto_scale_prism = auto_scale_prism

        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")
        if not self.era5_path.exists():
            raise FileNotFoundError(f"ERA5 file not found: {self.era5_path}")

        self.era5_daily = self._load_era5_daily(self.era5_path, era5_variable)
        self.era5_dates = pd.to_datetime(self.era5_daily["time"].values).normalize()
        self.era5_bounds = self._get_era5_bounds(self.era5_daily)
        self._era5_index_by_date: Dict[pd.Timestamp, int] = {
            date: idx for idx, date in enumerate(self.era5_dates)
        }

        prism_files = self._resolve_prism_files(self.prism_path)
        self._records: List[SampleRecord] = []
        self._prism_arrays: Dict[pd.Timestamp, np.ndarray] = {}

        stats = {
            "prism_files_scanned": 0,
            "candidate_dates": 0,
            "usable_samples": 0,
            "skipped_bad_filename": 0,
            "skipped_duplicate_date": 0,
            "skipped_missing_era5_date": 0,
            "skipped_insufficient_history": 0,
            "skipped_nonconsecutive_history": 0,
            "skipped_raster_read_error": 0,
            "skipped_clip_or_projection_error": 0,
        }

        parsed_dates_preview: List[pd.Timestamp] = []
        template_raster: Optional[xr.DataArray] = None

        for prism_file in prism_files:
            stats["prism_files_scanned"] += 1

            sample_date = self._parse_date_from_filename(prism_file)
            if sample_date is None:
                stats["skipped_bad_filename"] += 1
                continue

            stats["candidate_dates"] += 1
            parsed_dates_preview.append(sample_date)

            if sample_date in self._prism_arrays:
                stats["skipped_duplicate_date"] += 1
                continue

            era5_end_idx = self._era5_index_by_date.get(sample_date)
            if era5_end_idx is None:
                stats["skipped_missing_era5_date"] += 1
                continue

            era5_start_idx = era5_end_idx - self.history_length + 1
            if era5_start_idx < 0:
                stats["skipped_insufficient_history"] += 1
                continue

            window_dates = self.era5_dates[era5_start_idx : era5_end_idx + 1]
            expected_dates = pd.date_range(end=sample_date, periods=self.history_length, freq="D")
            if not np.array_equal(window_dates.values, expected_dates.values):
                stats["skipped_nonconsecutive_history"] += 1
                continue

            try:
                prism_da = self._open_prism_raster(prism_file)
            except Exception:
                stats["skipped_raster_read_error"] += 1
                continue

            try:
                prism_da = self._clip_prism_to_era5(prism_da, self.era5_bounds)
                if template_raster is None:
                    template_raster = prism_da
                else:
                    prism_da = prism_da.rio.reproject_match(template_raster)
            except Exception:
                stats["skipped_clip_or_projection_error"] += 1
                continue

            prism_array = self._prepare_prism_values(prism_da.values.astype(np.float32))
            self._prism_arrays[sample_date] = prism_array
            self._records.append(
                SampleRecord(
                    index=len(self._records),
                    date=sample_date,
                    prism_path=str(prism_file),
                    era5_start_idx=era5_start_idx,
                    era5_end_idx=era5_end_idx,
                )
            )
            stats["usable_samples"] += 1

        self._records.sort(key=lambda r: r.date)
        self._records = [
            SampleRecord(
                index=i,
                date=r.date,
                prism_path=r.prism_path,
                era5_start_idx=r.era5_start_idx,
                era5_end_idx=r.era5_end_idx,
            )
            for i, r in enumerate(self._records)
        ]

        if not self._records:
            era5_start = self.era5_dates.min().strftime("%Y-%m-%d")
            era5_end = self.era5_dates.max().strftime("%Y-%m-%d")
            prism_preview = ", ".join(sorted({d.strftime("%Y-%m-%d") for d in parsed_dates_preview})[:5])
            raise RuntimeError(
                "No aligned ERA5/PRISM pairs were created. "
                f"ERA5 date range: {era5_start} to {era5_end}. "
                f"History length: {self.history_length}. "
                f"First PRISM dates found: {prism_preview or 'none'}."
            )

        self.summary_stats = stats
        if verbose:
            print("Dataset summary:")
            print(f"  history_length={self.history_length}")
            print(f"  total_candidate_dates={stats['candidate_dates']}")
            print(f"  usable_samples={stats['usable_samples']}")
            print(f"  skipped_bad_filename={stats['skipped_bad_filename']}")
            print(f"  skipped_duplicate_date={stats['skipped_duplicate_date']}")
            print(f"  skipped_missing_era5_date={stats['skipped_missing_era5_date']}")
            print(f"  skipped_insufficient_history={stats['skipped_insufficient_history']}")
            print(f"  skipped_nonconsecutive_history={stats['skipped_nonconsecutive_history']}")
            print(f"  skipped_raster_read_error={stats['skipped_raster_read_error']}")
            print(f"  skipped_clip_or_projection_error={stats['skipped_clip_or_projection_error']}")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self._records[idx]

        era5_window = (
            self.era5_daily
            .isel(time=slice(rec.era5_start_idx, rec.era5_end_idx + 1))
            .values.astype(np.float32)
        )
        era5_window = self._fill_missing(era5_window)
        prism_array = self._prism_arrays[rec.date]

        x = torch.from_numpy(era5_window).unsqueeze(1)  # [T, 1, H, W]
        y = torch.from_numpy(prism_array).unsqueeze(0)  # [1, H_high, W_high]
        return x, y

    def metadata(self, idx: int) -> SampleMetadata:
        rec = self._records[idx]
        era5_shape = (
            self.history_length,
            1,
            int(self.era5_daily.shape[-2]),
            int(self.era5_daily.shape[-1]),
        )
        prism_shape = tuple(int(v) for v in self._prism_arrays[rec.date].shape)
        return SampleMetadata(
            index=idx,
            date=rec.date,
            prism_path=rec.prism_path,
            era5_shape=era5_shape,
            prism_shape=prism_shape,
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

        if not era5_ds.data_vars:
            raise ValueError("ERA5 dataset has no data variables")

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

        if era5_daily.sizes.get("time", 0) < 1:
            raise ValueError("ERA5 dataset has no usable daily timesteps")

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
                p
                for p in prism_path.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_PRISM_EXTENSIONS
            )
            if files:
                return files

            zip_files = sorted(prism_path.rglob("*.zip"))
            if zip_files:
                raise ValueError(
                    "Found PRISM zip archives but no extracted raster files. "
                    "Extract archives first (or rerun data_pipeline/download_prism.py)."
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
                    "Extract it first and pass the extracted directory or raster path."
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
                f"Failed to read PRISM raster '{prism_file}'. Ensure it is a valid geospatial raster."
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
        clipped = prism_da.rio.clip_box(minx=min_lon, maxx=max_lon, miny=min_lat, maxy=max_lat)

        if clipped.sizes.get("x", 0) == 0 or clipped.sizes.get("y", 0) == 0:
            raise ValueError("PRISM raster has no spatial overlap with ERA5 extent after clipping")

        return clipped

    @staticmethod
    def _parse_date_from_filename(file_path: Path) -> Optional[pd.Timestamp]:
        match = DATE_PATTERN.search(file_path.name)
        if not match:
            return None
        return pd.Timestamp(pd.to_datetime(match.group(0), format="%Y%m%d")).normalize()

    def _prepare_prism_values(self, prism_array: np.ndarray) -> np.ndarray:
        prism_array = self._fill_missing(prism_array)

        if self.auto_scale_prism:
            p95 = float(np.percentile(np.abs(prism_array), 95))
            if p95 > 500.0:
                prism_array = prism_array / 100.0

        return prism_array.astype(np.float32)

    @staticmethod
    def _fill_missing(array: np.ndarray) -> np.ndarray:
        if not np.isnan(array).any():
            return array.astype(np.float32)

        finite = array[np.isfinite(array)]
        if finite.size == 0:
            raise ValueError("Encountered array with all values as NaN")

        return np.nan_to_num(array, nan=float(finite.mean())).astype(np.float32)
