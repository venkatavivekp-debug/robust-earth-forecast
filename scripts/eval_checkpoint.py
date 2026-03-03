import argparse
import numpy as np
import torch

from src.training.spatiotemporal_module import make_loaders, LitForecaster

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", required=True, choices=["cnn3d", "convlstm"])
    args = p.parse_args()

    _, val_loader, meta = make_loaders(args.npz, batch_size=8)
    t_in = int(meta["t_in"])
    t2m_std = float(meta["std"][0])   # °C std for t2m (channel 0)

    in_channels = val_loader.dataset.X.shape[-1]
    model = LitForecaster.load_from_checkpoint(args.ckpt, model_name=args.model, t_in=t_in, in_channels=in_channels)
    model.eval()

    mse_sum = 0.0
    n_sum = 0

    # persistence baseline: predict y(t_out) ~= last input t2m frame
    base_mse_sum = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            yhat = model(x)
            mse = torch.mean((yhat - y) ** 2).item()
            mse_sum += mse * x.shape[0]
            n_sum += x.shape[0]

            # persistence: last time step, channel 0 (t2m)
            last_t2m = x[:, -1, :, :, 0].unsqueeze(-1)  # (B,H,W,1)
            base_mse = torch.mean((last_t2m - y) ** 2).item()
            base_mse_sum += base_mse * x.shape[0]

    val_mse_norm = mse_sum / n_sum
    base_mse_norm = base_mse_sum / n_sum

    rmse_c = (val_mse_norm ** 0.5) * t2m_std
    base_rmse_c = (base_mse_norm ** 0.5) * t2m_std

    skill = 1.0 - (rmse_c / base_rmse_c)

    print("Normalized val_mse:", val_mse_norm)
    print("RMSE (°C):", rmse_c)
    print("Persistence RMSE (°C):", base_rmse_c)
    print("Skill vs persistence:", skill)

if __name__ == "__main__":
    main()
