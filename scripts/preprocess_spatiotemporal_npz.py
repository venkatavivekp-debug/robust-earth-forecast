import os
import numpy as np
import xarray as xr

RAW_PATH = "data_raw/era5_seus_2024_01.nc"
OUT_DIR = "data_processed"
OUT_FILE = os.path.join(OUT_DIR, "era5_seus_2024_01_Tin6_Tout1_spatiotemporal.npz")

T_IN = 6   # past hours
T_OUT = 1  # next hour
VARS_X = ["t2m", "u10", "v10"]
VAR_Y = "t2m"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = xr.open_dataset(RAW_PATH)

    # Your file uses valid_time
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    # Kelvin -> Celsius for t2m
    ds["t2m"] = ds["t2m"] - 273.15

    # Extract arrays: (T, H, W)
    t2m = ds["t2m"].values.astype(np.float32)
    u10 = ds["u10"].values.astype(np.float32)
    v10 = ds["v10"].values.astype(np.float32)

    # Build X: stack channels => (T, H, W, C)
    X_all = np.stack([t2m, u10, v10], axis=-1)  # (T, H, W, 3)

    # Normalize per-channel using whole month (baseline)
    mean = X_all.mean(axis=(0,1,2), keepdims=True)
    std = X_all.std(axis=(0,1,2), keepdims=True) + 1e-6
    Xn = (X_all - mean) / std

    # Target is next-hour t2m only (normalized using same t2m stats from Xn channel 0)
    # Since Xn[...,0] is normalized t2m, Y should be normalized similarly
    Yn_t2m = Xn[..., 0]  # (T, H, W)

    T = Xn.shape[0]
    H, W = Xn.shape[1], Xn.shape[2]

    X_list, Y_list = [], []
    for t in range(T_IN, T - T_OUT + 1):
        x = Xn[t - T_IN:t]               # (T_IN, H, W, 3)
        y = Yn_t2m[t + T_OUT - 1]        # (H, W)
        X_list.append(x)
        Y_list.append(y[..., None])      # (H, W, 1)

    X = np.stack(X_list, axis=0)  # (N, T_IN, H, W, 3)
    Y = np.stack(Y_list, axis=0)  # (N, H, W, 1)

    # Time-ordered split (important for forecasting)
    n = X.shape[0]
    split = int(n * 0.8)
    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val = X[split:], Y[split:]

    np.savez_compressed(
        OUT_FILE,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        mean=mean.squeeze().astype(np.float32),
        std=std.squeeze().astype(np.float32),
        vars=np.array(VARS_X),
        t_in=np.array(T_IN),
        t_out=np.array(T_OUT),
        H=np.array(H), W=np.array(W),
    )

    print("Saved:", OUT_FILE)
    print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("X_val  :", X_val.shape,   "Y_val  :", Y_val.shape)

if __name__ == "__main__":
    main()
