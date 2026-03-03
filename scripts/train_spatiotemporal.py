import os
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from src.training.spatiotemporal_module import make_loaders, LitForecaster

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str, required=True)
    p.add_argument("--model", type=str, default="cnn3d", choices=["cnn3d", "convlstm"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    os.makedirs("checkpoints", exist_ok=True)

    train_loader, val_loader, meta = make_loaders(args.npz, batch_size=args.batch_size)

    t_in = int(meta["t_in"])
    in_channels = train_loader.dataset.X.shape[-1]

    model = LitForecaster(model_name=args.model, t_in=t_in, in_channels=in_channels, lr=1e-3)

    tag = os.path.basename(args.npz).replace(".npz","")
    ckpt = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{args.model}-{tag}" + "-{epoch:02d}-{val_mse:.6f}",
        monitor="val_mse",
        mode="min",
        save_top_k=1,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        callbacks=[ckpt],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)
    print("Best checkpoint:", ckpt.best_model_path)

if __name__ == "__main__":
    main()
