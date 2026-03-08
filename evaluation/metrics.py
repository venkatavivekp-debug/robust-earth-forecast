import torch

def rmse(pred, target):

    return torch.sqrt(torch.mean((pred - target) ** 2))

def mae(pred, target):

    return torch.mean(torch.abs(pred - target))
