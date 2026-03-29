import torch
import numpy as np
from models.cnn_downscaler import CNNDownscaler
from datasets.build_training_dataset import get_dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNNDownscaler().to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

loader = get_dataloader(train=False)

mse_list = []
mae_list = []

with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        pred = model(x)

        mse = torch.mean((pred - y) ** 2).item()
        mae = torch.mean(torch.abs(pred - y)).item()

        mse_list.append(mse)
        mae_list.append(mae)

print("RMSE:", np.sqrt(np.mean(mse_list)))
print("MAE:", np.mean(mae_list))
