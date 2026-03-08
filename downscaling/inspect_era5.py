import xarray as xr

ds = xr.open_dataset("data_raw/era5_georgia_temp.nc")

print(ds)
print("\nVariables:")
print(ds.data_vars)

print("\nDimensions:")
print(ds.dims)
