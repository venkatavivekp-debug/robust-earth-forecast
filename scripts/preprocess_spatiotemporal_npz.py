import os
import argparse
import numpy as np
import xarray as xr

RAW_PATH = "data_raw/era5_seus_2024_01.nc"
OUT_DIR = "data_processed"

VARS_X = ["t2m", "u10", "v10"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t_in", type=int, default=6)
    p.add_argument("--t_out", type=int, default=1)  # horizon in hours
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_file = os.path.join(
        OUT_DIR,
        f"era5_seus_2024_01_Tin{args.t_in}_Tout{args.t_out}_spatiotemporal.npz"
    )

    ds = xr.open_dataset(RAW_PATH)
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    ds["t2m"] = ds["t2m"] - 273.15

    t2m = ds["t2m"].values.astype(np.float32)
    u10 = ds["u10"].values.astype(np.float32)
    v10 = ds["v10"].values.astype(np.float32)

    X_all = np.stack([t2m, u10, v10], axis=-1)  # (T,H,W,3)

    mean = X_all.mean(axis=(0,1,2), keepdims=True)
    std = X_all.std(axis=(0,1,2), keepdims=True) + 1e-6
    Xn = (X_all - mean) / std

    Yn_t2m = Xn[..., 0]  # normalized t2m

    T = Xn.shape[0]
    H, W = Xn.shape[1], Xn.shape[2]

    X_list, Y_list = [], []
    for t in range(args.t_in, T - args.t_out + 1):
        x = Xn[t - args.t_in:t]               # (t_in,H,W,3)
        y = Yn_t2m[t + args.t_out - 1]        # (H,W)
        X_list.append(x)
        Y_list.append(y[..., None])           # (H,W,1)

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)

    n = X.shape[0]
    split = int(n * 0.8)
    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val = X[split:], Y[split:]

    np.savez_compressed(
        out_file,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        mean=mean.squeeze().astype(np.float32),
        std=std.squeeze().astype(np.float32),
        vars=np.array(VARS_X),
        t_in=np.array(args.t_in),
        t_out=np.array(args.t_out),
        H=np.array(H), W=np.array(W),
    )

    print("Saved:", out_file)
    print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("X_val  :", X_val.shape,   "Y_val  :", Y_val.shape)

if __name__ == "__main__":
    main()
