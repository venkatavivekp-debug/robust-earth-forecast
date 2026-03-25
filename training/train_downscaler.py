import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.prism_dataset import ERA5_PRISM_Dataset
from models.cnn_downscaler import CNNDownscaler


dataset = ERA5_PRISM_Dataset(
    "data_raw/era5_georgia_temp.nc"
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)


model = CNNDownscaler()


optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()


for epoch in range(10):
    total_loss = 0

    for x, y in loader:
        optimizer.zero_grad()

        preds = model(x)

        # match size
        preds = torch.nn.functional.interpolate(preds, size=y.shape[-2:])

        loss = loss_fn(preds, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} Loss: {total_loss/len(loader):.6f}")


torch.save(model.state_dict(), "model.pth")
print("Model saved!")
