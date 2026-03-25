import torch
import xarray as xr

class ERA5_PRISM_Dataset:
    def __init__(self, era5_path: str, prism_path: str):
        self.era5_path = era5_path
        self.prism_path = prism_path

    def load_data(self):
        # Load ERA5 data
        era5_data = xr.open_dataset(self.era5_path)
        prism_data = xr.open_dataset(self.prism_path)

        # Here you would typically preprocess and align dates, but for brevity,
        # we're going to focus on getting the temperature data directly.

        era5_temp = era5_data['temperature'].isel(time=0)  # Select the first time point
        prism_temp = prism_data['temperature'].isel(time=0)  # Select the first time point

        return self._process_data(era5_temp, prism_temp)

    def _process_data(self, era5_temp, prism_temp):
        # Assumes both datasets have the same shape; adjust as needed
        combined_temp = (era5_temp + prism_temp) / 2  # Simple average for example

        # Convert to tensor and reshape
        temperature_tensor = torch.tensor(combined_temp.values).unsqueeze(0)
        return temperature_tensor.view(1, 21, 21)  # Reshape to [1, 21, 21]