import argparse
import numpy as np
import torch
from src.training.spatiotemporal_module import make_loaders, LitForecaster

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    _, val_loader, meta = make_loaders(args.npz, batch_size=8)

    t_in = int(meta["t_in"])
    in_channels = int(meta["in_channels"])
    out_channels = int(meta["out_channels"])
    levels = meta["levels"]

    std_lv = meta["std"]          # shape (L, V)
    t_std = std_lv[0]  # temperature std per level

    model = LitForecaster.load_from_checkpoint(
        args.ckpt,
        model_name="convlstm",
        t_in=t_in,
        in_channels=in_channels,
        out_channels=out_channels
    )

    model.eval()

    mse_sum = np.zeros((out_channels,))
    base_mse_sum = np.zeros((out_channels,))
    n_batches = 0

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            yhat = model(x) 
            err = (yhat - y) ** 2
            mse = err.mean(dim=(0,1,2)).cpu().numpy()
            mse_sum += mse

            # persistence baseline
            B, T, H, W, C = x.shape
            V = 5
            temp_idxs = [lvl * V + 0 for lvl in range(out_channels)]
            last_temp = x[:, -1, :, :, temp_idxs]

            base_err = (last_temp - y) ** 2
            base_mse = base_err.mean(dim=(0,1,2)).cpu().numpy()
            base_mse_sum += base_mse

            n_batches += 1

    mse_norm = mse_sum / n_batches
    base_mse_norm = base_mse_sum / n_batches
    
    rmse_c = np.sqrt(mse_norm) * t_std
    base_rmse_c = np.sqrt(base_mse_norm) * t_std
    skill = 1.0 - (rmse_c / base_rmse_c)

    print("Levels (hPa):", levels)
    print("RMSE per level (°C):", rmse_c)
    print("Persistence RMSE per level (°C):", base_rmse_c)
    print("Skill per level:", skill)
    print("Mean RMSE (°C):", rmse_c.mean())
    print("Mean skill:", skill.mean())

if __name__ == "__main__":
    main()
