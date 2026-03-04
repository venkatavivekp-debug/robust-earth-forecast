import os
import argparse
import numpy as np
import xarray as xr

RAW_PATH = "data_raw/era5_pl_seus_2024_01.nc"
OUT_DIR = "data_processed"

VARS = ["t", "u", "v", "r", "z"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t_in", type=int, default=6)
    p.add_argument("--t_out", type=int, default=12)
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_file = os.path.join(
        OUT_DIR,
        f"era5_pl_seus_2024_01_Tin{args.t_in}_Tout{args.t_out}.npz"
    )

    ds = xr.open_dataset(RAW_PATH)

    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    if "pressure_level" in ds.coords:
        ds = ds.rename({"pressure_level": "level"})
    if "pressure_level" in ds.dims:
        ds = ds.rename({"pressure_level": "level"})

    print("Dataset dims:", ds.dims)
    print("Variables:", list(ds.data_vars))

    t = ds["t"].values.astype(np.float32) - 273.15
    u = ds["u"].values.astype(np.float32)
    v = ds["v"].values.astype(np.float32)
    r = ds["r"].values.astype(np.float32) / 100.0
    z = ds["z"].values.astype(np.float32) / 9.80665

    X5 = np.stack([t, u, v, r, z], axis=-1)

    mean = X5.mean(axis=(0,1,2,3), keepdims=True)
    std = X5.std(axis=(0,1,2,3), keepdims=True) + 1e-6
    Xn5 = (X5 - mean) / std

    T, L, H, W, Vv = Xn5.shape

    Xn = Xn5.transpose(0,2,3,1,4).reshape(T, H, W, L*Vv)

    t_mean = mean[..., 0]
    t_std  = std[..., 0]
    Yn = ((t[..., None] - t_mean) / t_std).squeeze(-1)

    X_list, Y_list = [], []
    for ti in range(args.t_in, T - args.t_out + 1):
        x = Xn[ti - args.t_in:ti]
        y = Yn[ti + args.t_out - 1].transpose(1,2,0)
        X_list.append(x)
        Y_list.append(y)

    X = np.stack(X_list).astype(np.float32)
    Y = np.stack(Y_list).astype(np.float32)

    n = X.shape[0]
    split = int(n * 0.8)
    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val     = X[split:], Y[split:]

    levels = ds["level"].values.astype(np.int32)

    np.savez_compressed(
        out_file,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        mean=mean.squeeze().astype(np.float32),
        std=std.squeeze().astype(np.float32),
        levels=levels,
        t_in=np.array(args.t_in),
        t_out=np.array(args.t_out),
        H=np.array(H), W=np.array(W),
        in_channels=np.array(X.shape[-1]),
        out_channels=np.array(L),
    )

    print("Saved:", out_file)
    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)
    print("Levels:", levels)

if __name__ == "__main__":
    main()
