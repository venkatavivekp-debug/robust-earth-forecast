import matplotlib.pyplot as plt
import xarray as xr

ds = xr.open_dataset("data_raw/era5_georgia_temp.nc")

temp = ds["t2m"].isel(valid_time=0)

plt.imshow(temp)
plt.colorbar()
plt.title("ERA5 Temperature")
plt.show()
