import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from torch.utils.data import Dataset, DataLoader

from src.models.cnn3d_forecaster import CNN3DForecaster

class NPZGridDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)  # (N, T, H, W, C)
        self.Y = torch.from_numpy(Y)  # (N, H, W, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def make_loaders(npz_path: str, batch_size: int = 8, num_workers: int = 0):
    d = np.load(npz_path, allow_pickle=True)
    train_ds = NPZGridDataset(d["X_train"], d["Y_train"])
    val_ds = NPZGridDataset(d["X_val"], d["Y_val"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    meta = {k: d[k] for k in d.files if k not in ["X_train","Y_train","X_val","Y_val"]}
    return train_loader, val_loader, meta

class LitForecaster(L.LightningModule):
    def __init__(self, t_in: int, in_channels: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = CNN3DForecaster(in_channels=in_channels, t_in=t_in)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = F.mse_loss(yhat, y)
        self.log("train_mse", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = F.mse_loss(yhat, y)
        self.log("val_mse", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
