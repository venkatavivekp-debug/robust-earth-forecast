import torch
import xarray as xr
import numpy as np

class ClimateDataset(torch.utils.data.Dataset):

    def __init__(self, file):

        ds = xr.open_dataset(file)

        temp = ds["t2m"].values
        uwind = ds["u10"].values
        vwind = ds["v10"].values
        pressure = ds["sp"].values

        stacked = np.stack([temp, uwind, vwind, pressure], axis=1)

        self.data = torch.tensor(stacked).float()

    def __len__(self):

        return self.data.shape[0]

    def __getitem__(self, idx):

        x = self.data[idx]

        return x
