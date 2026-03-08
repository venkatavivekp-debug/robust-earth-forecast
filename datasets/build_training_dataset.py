import xarray as xr
import numpy as np
from scipy.ndimage import zoom

ds = xr.open_dataset("data_raw/era5_georgia_temp.nc")
t2m = ds["t2m"].values

low_res = t2m
high_res = zoom(t2m, (1,4,4))

print("Low resolution:", low_res.shape)
print("High resolution:", high_res.shape)
