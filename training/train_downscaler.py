import torch
import torch.nn as nn
import torch.optim as optim
from datasets.climate_dataset import ClimateDataset
from models.cnn_downscaler import CNNDownscaler
from torch.utils.data import DataLoader
from scipy.ndimage import zoom

dataset = ClimateDataset("data_raw/era5_georgia_temp.nc")

loader = DataLoader(dataset, batch_size=8)

model = CNNDownscaler()

optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.MSELoss()

for epoch in range(10):

    for batch in loader:

        x = batch

        high_res = zoom(x.numpy(), (1,1,4,4))

        y = torch.tensor(high_res).float()

        optimizer.zero_grad()

        preds = model(x)

        preds = torch.nn.functional.interpolate(preds, size=(84,84))

        loss = loss_fn(preds, y[:,0:1])

        loss.backward()

        optimizer.step()

    print("Epoch:", epoch, "Loss:", loss.item())
