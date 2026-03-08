import xarray as xr

ds = xr.open_dataset("data_raw/prism/prism_tmean_2023.nc")

print(ds)
