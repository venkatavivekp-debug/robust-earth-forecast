import xarray as xr
import numpy as np
from scipy.ndimage import zoom

ds = xr.open_dataset("data_raw/era5_georgia.nc")

t2m = ds["t2m"].values

low_res = t2m
high_res = zoom(t2m, (1, 4, 4))

print("Low resolution shape:", low_res.shape)
print("High resolution shape:", high_res.shape)
