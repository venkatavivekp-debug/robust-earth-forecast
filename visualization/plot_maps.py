import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs

ds = xr.open_dataset("data_raw/era5_georgia_temp.nc")

temp = ds["t2m"].isel(valid_time=0)

plt.figure(figsize=(8,6))

ax = plt.axes(projection=ccrs.PlateCarree())

plt.imshow(temp)

plt.colorbar()

plt.title("ERA5 Temperature Map")

plt.show()
