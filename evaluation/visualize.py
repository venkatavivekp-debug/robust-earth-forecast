import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import matplotlib.pyplot as plt
from datasets.prism_dataset import ERA5_PRISM_Dataset
from models.cnn_downscaler import CNNDownscaler
import torch.nn.functional as F


dataset = ERA5_PRISM_Dataset(
    "data_raw/era5_georgia_temp.nc",
    "data_raw/prism/prism_tmean_2023.nc"
)

x, y = dataset[0]

x = x.unsqueeze(0)


model = CNNDownscaler()
model.load_state_dict(torch.load("model.pth"))
model.eval()

with torch.no_grad():
    pred = model(x)
    pred = F.interpolate(pred, size=y.shape[-2:])

low_res = x.squeeze().numpy()
prediction = pred.squeeze().numpy()
ground_truth = y.squeeze().numpy()


plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("ERA5 (Input)")
plt.imshow(low_res, cmap="coolwarm")

plt.subplot(1,3,2)
plt.title("Prediction")
plt.imshow(prediction, cmap="coolwarm")

plt.subplot(1,3,3)
plt.title("Target")
plt.imshow(ground_truth, cmap="coolwarm")

plt.tight_layout()
plt.savefig("result.png")
plt.show()
