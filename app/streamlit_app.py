import glob
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import torch

from src.training.spatiotemporal_module import make_loaders, LitForecaster

NPZ_PATH = "data_processed/era5_seus_2024_01_Tin6_Tout1_spatiotemporal.npz"

def find_best_ckpt():
    cands = glob.glob("checkpoints/cnn3d-spatiotemporal-*.ckpt")
    if not cands:
        return None
    # best ckpt is saved as top_k=1; newest usually fine
    return sorted(cands)[-1]

@st.cache_resource
def load_model_data():
    train_loader, val_loader, meta = make_loaders(NPZ_PATH, batch_size=8)
    t_in = int(meta["t_in"])
    in_channels = train_loader.dataset.X.shape[-1]

    ckpt = find_best_ckpt()
    if ckpt is None:
        model = LitForecaster(t_in=t_in, in_channels=in_channels)
        st.warning("No checkpoint found. Train first: python scripts/train_spatiotemporal_cnn3d.py")
    else:
        model = LitForecaster.load_from_checkpoint(ckpt)
    model.eval()
    return model, val_loader, meta, ckpt

def plot_map(arr2d, title):
    fig = plt.figure()
    plt.imshow(arr2d)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.title("ERA5 Spatiotemporal Forecast Dashboard (3D CNN)")
    model, val_loader, meta, ckpt = load_model_data()

    st.write("Dataset:", NPZ_PATH)
    st.write("Checkpoint:", ckpt if ckpt else "None")

    idx = st.slider("Validation sample index", 0, len(val_loader.dataset)-1, 0)

    x, y = val_loader.dataset[idx]   # x: (T,H,W,C) y:(H,W,1)
    x_in = x.unsqueeze(0)

    with torch.no_grad():
        yhat = model(x_in).squeeze(0).cpu().numpy()  # (H,W,1)

    y_np = y.cpu().numpy()
    err = np.abs(yhat[...,0] - y_np[...,0])

    plot_map(y_np[...,0], "Ground Truth t2m (normalized)")
    plot_map(yhat[...,0], "Prediction t2m (normalized)")
    plot_map(err, "Absolute Error (normalized)")

    mse = float(np.mean((yhat[...,0] - y_np[...,0])**2))
    st.write({"mse": mse})

if __name__ == "__main__":
    main()
