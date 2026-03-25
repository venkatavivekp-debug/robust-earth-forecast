import xarray as xr
import torch
from torch.utils.data import Dataset

class ERA5_PRISM_Dataset(Dataset):
    def __init__(self, era5_path, prism_path):
        self.era5 = xr.open_dataset(era5_path)
        self.prism = xr.open_dataset(prism_path)

        self.era5_var = list(self.era5.data_vars)[0]
        self.prism_var = list(self.prism.data_vars)[0]

        self.length = self.era5.dims['time']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.era5[self.era5_var].isel(time=idx).values
        y = self.prism[self.prism_var].isel(time=idx).values

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

        return x, y
