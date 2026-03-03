import os
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from src.training.spatiotemporal_module import make_loaders, LitForecaster

NPZ_PATH = "data_processed/era5_seus_2024_01_Tin6_Tout1_spatiotemporal.npz"

def main():
    os.makedirs("checkpoints", exist_ok=True)

    train_loader, val_loader, meta = make_loaders(NPZ_PATH, batch_size=8)

    t_in = int(meta["t_in"])
    in_channels = train_loader.dataset.X.shape[-1]  # should be 3

    model = LitForecaster(t_in=t_in, in_channels=in_channels, lr=1e-3)

    ckpt = ModelCheckpoint(
        dirpath="checkpoints",
        filename="cnn3d-spatiotemporal-{epoch:02d}-{val_mse:.6f}",
        monitor="val_mse",
        mode="min",
        save_top_k=1,
    )

    trainer = L.Trainer(
        max_epochs=10,
        callbacks=[ckpt],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)
    print("Best checkpoint:", ckpt.best_model_path)

if __name__ == "__main__":
    main()
