import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
import numpy as np
from scipy.ndimage import zoom
from downscale_model import DownscaleCNN

ds = xr.open_dataset("data_raw/era5_georgia_temp.nc")

t2m = ds["t2m"].values

low_res = t2m
high_res = zoom(t2m, (1,4,4))

X = torch.tensor(low_res).unsqueeze(1).float()
Y = torch.tensor(high_res).unsqueeze(1).float()

model = DownscaleCNN()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(5):

    optimizer.zero_grad()

    preds = model(X)

    preds = torch.nn.functional.interpolate(preds, size=(84,84))

    loss = loss_fn(preds, Y)

    loss.backward()

    optimizer.step()

    print("Epoch", epoch, "Loss:", loss.item())
