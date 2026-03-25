import torch
import xarray as xr
import numpy as np

class ClimateDataset(torch.utils.data.Dataset):

    def __init__(self, file):
        ds = xr.open_dataset(file)

        # Use only temperature
        temp = ds["t2m"].values  # (time, lat, lon)

        # Normalize (important for stable training)
        temp = (temp - temp.mean()) / temp.std()

        # Add channel dimension → (time, 1, lat, lon)
        temp = np.expand_dims(temp, axis=1)

        self.data = torch.tensor(temp).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]
