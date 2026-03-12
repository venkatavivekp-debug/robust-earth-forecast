import torch
from evaluation.metrics import rmse
from models.cnn_downscaler import CNNDownscaler

model = CNNDownscaler()

pred = torch.randn(1,1,84,84)

target = torch.randn(1,1,84,84)

print("RMSE:", rmse(pred,target))
