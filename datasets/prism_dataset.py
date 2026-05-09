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
CORE4_CHANNELS = [
    ("t2m", ("t2m", "2m_temperature")),
    ("u10", ("u10", "10m_u_component_of_wind")),
    ("v10", ("v10", "10m_v_component_of_wind")),
    ("sp", ("sp", "surface_pressure")),
]
EXTENDED_CHANNELS = CORE4_CHANNELS + [
    ("tp", ("tp", "total_precipitation")),
    ("rh2m", ("rh2m", "relative_humidity_2m")),
    ("rh_850", ("rh_850", "relative_humidity_850")),
    ("rh_500", ("rh_500", "relative_humidity_500")),
    ("t_850", ("t_850", "temperature_850")),
    ("t_500", ("t_500", "temperature_500")),
    ("gh_850", ("gh_850", "geopotential_height_850")),
    ("gh_500", ("gh_500", "geopotential_height_500")),
]
CHANNEL_SETS = {
    "t2m": [("t2m", ("t2m", "2m_temperature"))],
    "core4": CORE4_CHANNELS,
    "extended": EXTENDED_CHANNELS,
    "core4_elev": CORE4_CHANNELS,
    "core4_topo": CORE4_CHANNELS,
}
STATIC_FEATURE_SETS = {
    "core4_elev": ["elevation"],
    "core4_topo": ["elevation", "slope", "aspect", "terrain_gradient_magnitude"],
}


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

    Input  X: [T, 4, H_era5, W_era5]
    Target Y: [1, H_prism, W_prism]
    """

    def __init__(
        self,
        era5_path: str,
        prism_path: str,
        history_length: int = 3,
        input_set: str = "extended",
        era5_variable: Optional[str] = None,
        static_covariate_path: Optional[str] = None,
        auto_scale_prism: bool = True,
        verbose: bool = True,
    ) -> None:
        self.era5_path = Path(era5_path)
        self.prism_path = Path(prism_path)
        self.history_length = int(history_length)
        self.input_set = str(input_set)
        self.static_covariate_path = Path(static_covariate_path) if static_covariate_path else None
        self.auto_scale_prism = auto_scale_prism

        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")
        if self.input_set not in CHANNEL_SETS:
            raise ValueError(f"input_set must be one of {list(CHANNEL_SETS.keys())}, got '{self.input_set}'")
        if self.input_set in STATIC_FEATURE_SETS:
            if self.static_covariate_path is None:
                raise ValueError(f"input_set='{self.input_set}' requires static_covariate_path")
            if not self.static_covariate_path.exists():
                raise FileNotFoundError(f"Static covariate file not found: {self.static_covariate_path}")
        if not self.era5_path.exists():
            raise FileNotFoundError(f"ERA5 file not found: {self.era5_path}")

        self.era5_daily = self._load_era5_daily(self.era5_path, era5_variable)
        self.era5_channel_count = int(self.era5_daily.sizes["channel"])
        self.channel_names = [str(v) for v in self.era5_daily["channel"].values.tolist()]
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
            print(f"  input_set={self.input_set} channels={self.channel_names}")
            if self.static_covariate_path is not None:
                print(f"  static_covariate_path={self.static_covariate_path}")
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
        if idx < 0 or idx >= len(self._records):
            raise IndexError(f"Sample index out of range: {idx}")

        rec = self._records[idx]

        era5_window = (
            self.era5_daily
            .isel(time=slice(rec.era5_start_idx, rec.era5_end_idx + 1))
            .transpose("time", "channel", "latitude", "longitude")
            .values.astype(np.float32)
        )
        era5_window = self._fill_missing(era5_window)
        prism_array = self._prism_arrays[rec.date]

        x = torch.from_numpy(era5_window)  # [T, 4, H, W]
        y = torch.from_numpy(prism_array).unsqueeze(0)  # [1, H_high, W_high]

        if x.dim() != 4:
            raise RuntimeError(f"Expected ERA5 tensor shape [T, C, H, W], got {tuple(x.shape)}")
        if y.dim() != 3:
            raise RuntimeError(f"Expected PRISM tensor shape [C, H, W], got {tuple(y.shape)}")
        if x.shape[0] != self.history_length:
            raise RuntimeError(
                f"History length mismatch: expected {self.history_length}, got {x.shape[0]}"
            )
        if not torch.isfinite(x).all():
            raise RuntimeError(f"Non-finite values found in ERA5 tensor for index {idx} ({rec.date.date()})")
        if not torch.isfinite(y).all():
            raise RuntimeError(f"Non-finite values found in PRISM tensor for index {idx} ({rec.date.date()})")

        return x, y

    def metadata(self, idx: int) -> SampleMetadata:
        rec = self._records[idx]
        era5_shape = (
            self.history_length,
            self.era5_channel_count,
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
        if era5_variable is not None:
            supported = {name for _, candidates in EXTENDED_CHANNELS for name in candidates}
            if era5_variable not in supported:
                raise ValueError(
                    f"era5_variable='{era5_variable}' is not supported for multivariable loading. "
                    f"Expected one of {sorted(supported)}."
                )

        channel_arrays: List[xr.DataArray] = []
        missing_channels: List[str] = []
        channel_names: List[str] = []
        selected_channels = CHANNEL_SETS[self.input_set]

        base_cache: Dict[str, xr.DataArray] = {}

        def load_base(candidates: Sequence[str]) -> Optional[xr.DataArray]:
            for name in candidates:
                if name in base_cache:
                    return base_cache[name]
                if name in era5_ds.data_vars:
                    da = era5_ds[name]
                    time_dim = self._find_first_dim(da.dims, TIME_DIM_CANDIDATES, "time")
                    lat_dim = self._find_first_dim(da.dims, LAT_DIM_CANDIDATES, "latitude")
                    lon_dim = self._find_first_dim(da.dims, LON_DIM_CANDIDATES, "longitude")
                    da = da.transpose(time_dim, lat_dim, lon_dim).rename(
                        {time_dim: "time", lat_dim: "latitude", lon_dim: "longitude"}
                    )
                    base_cache[name] = da
                    return da
            return None

        def convert_channel(canonical_name: str, da_hourly: xr.DataArray) -> xr.DataArray:
            da_daily = da_hourly.resample(time="1D").sum(keep_attrs=True) if canonical_name == "tp" else da_hourly.resample(time="1D").mean(keep_attrs=True)

            if canonical_name.startswith("t") and float(da_daily.mean(skipna=True)) > 150.0:
                da_daily = da_daily - 273.15
            if canonical_name == "sp" and float(da_daily.mean(skipna=True)) > 2000.0:
                da_daily = da_daily / 100.0
            if canonical_name.startswith("gh") and float(da_daily.mean(skipna=True)) > 20000.0:
                da_daily = da_daily / 9.80665
            if canonical_name == "tp":
                da_daily = da_daily * 1000.0
            if canonical_name.startswith("rh"):
                da_daily = da_daily.clip(min=0.0, max=100.0)

            return da_daily.astype(np.float32)

        for canonical_name, candidates in selected_channels:
            da_hourly = load_base(candidates)
            if da_hourly is None:
                missing_channels.append(canonical_name)
                continue

            channel_arrays.append(convert_channel(canonical_name, da_hourly))
            channel_names.append(canonical_name)

        if missing_channels:
            raise ValueError(
                "ERA5 dataset is missing required variables: "
                f"{missing_channels}. Available: {list(era5_ds.data_vars)}"
            )

        if not channel_arrays:
            raise ValueError("No ERA5 variables were loaded")

        aligned = xr.align(*channel_arrays, join="inner")
        stacked = xr.concat(aligned, dim="channel").assign_coords(channel=channel_names)
        stacked = stacked.transpose("time", "channel", "latitude", "longitude")

        if stacked.sizes.get("time", 0) < 1:
            raise ValueError("ERA5 dataset has no usable daily timesteps after alignment")

        static_features = STATIC_FEATURE_SETS.get(self.input_set, [])
        if static_features:
            stacked = self._append_static_channels(stacked, static_features)

        return stacked

    def _append_static_channels(self, stacked: xr.DataArray, feature_names: Sequence[str]) -> xr.DataArray:
        if self.static_covariate_path is None:
            raise ValueError(f"input_set='{self.input_set}' requires static_covariate_path")
        try:
            static_ds = xr.open_dataset(self.static_covariate_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to open static covariate NetCDF: {self.static_covariate_path}") from exc

        static_arrays: List[xr.DataArray] = []
        for feature in feature_names:
            if feature not in static_ds.data_vars:
                raise ValueError(
                    f"Static covariate file is missing '{feature}'. "
                    f"Available: {list(static_ds.data_vars)}"
                )
            da = static_ds[feature]
            rename_map = {}
            if "y" in da.dims:
                rename_map["y"] = "latitude"
            if "x" in da.dims:
                rename_map["x"] = "longitude"
            if rename_map:
                da = da.rename(rename_map)
            if "latitude" not in da.dims or "longitude" not in da.dims:
                raise ValueError(f"Static feature '{feature}' must have y/x or latitude/longitude dimensions")
            if not np.all(np.diff(da["latitude"].values) > 0):
                da = da.sortby("latitude")
            if not np.all(np.diff(da["longitude"].values) > 0):
                da = da.sortby("longitude")

            aligned = da.interp(
                latitude=stacked["latitude"],
                longitude=stacked["longitude"],
                method="linear",
            )
            arr = aligned.values.astype(np.float32)
            if not np.isfinite(arr).all():
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    raise ValueError(f"Static feature '{feature}' has no finite values after ERA5-grid alignment")
                arr = np.nan_to_num(arr, nan=float(finite.mean())).astype(np.float32)
            aligned = xr.DataArray(
                arr,
                dims=("latitude", "longitude"),
                coords={
                    "latitude": stacked["latitude"].values,
                    "longitude": stacked["longitude"].values,
                },
                name=feature,
            )
            expanded = aligned.expand_dims(time=stacked["time"]).transpose("time", "latitude", "longitude")
            static_arrays.append(expanded)

        static_stack = xr.concat(static_arrays, dim="channel").assign_coords(channel=list(feature_names))
        static_stack = static_stack.transpose("time", "channel", "latitude", "longitude")
        return xr.concat([stacked, static_stack], dim="channel").transpose("time", "channel", "latitude", "longitude")

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

        if not np.isfinite(prism_array).all():
            raise ValueError("PRISM raster contains non-finite values after preprocessing")

        return prism_array.astype(np.float32)

    @staticmethod
    def _fill_missing(array: np.ndarray) -> np.ndarray:
        if not np.isnan(array).any():
            return array.astype(np.float32)

        finite = array[np.isfinite(array)]
        if finite.size == 0:
            raise ValueError("Encountered array with all values as NaN")

        return np.nan_to_num(array, nan=float(finite.mean())).astype(np.float32)
