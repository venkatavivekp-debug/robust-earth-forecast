import os
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from src.training.spatiotemporal_module import make_loaders, LitForecaster

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)

    train_loader, val_loader, meta = make_loaders(args.npz, batch_size=args.batch_size)

    t_in = int(meta["t_in"])
    in_channels = int(meta["in_channels"])
    out_channels = int(meta["out_channels"])

    model = LitForecaster(
        model_name="convlstm",
        t_in=t_in,
        in_channels=in_channels,
        out_channels=out_channels,
        lr=1e-3
    )

    tag = os.path.basename(args.npz).replace(".npz", "")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"convlstm-{tag}" + "-{epoch:02d}-{val_mse:.6f}",
        monitor="val_mse",
        mode="min",
        save_top_k=1,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)

    print("Best checkpoint:", checkpoint_callback.best_model_path)

if __name__ == "__main__":
    main()
