import torch
import xarray as xr

class ClimateDataset(torch.utils.data.Dataset):

    def __init__(self, file):

        ds = xr.open_dataset(file)

        temp = ds["t2m"].values

        self.data = torch.tensor(temp).float()

    def __len__(self):

        return self.data.shape[0]

    def __getitem__(self, idx):

        x = self.data[idx]

        return x.unsqueeze(0)
