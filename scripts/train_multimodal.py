import torch
import torch.optim as optim

from src.models.multimodal_forecaster import MultimodalForecaster


def main():

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = MultimodalForecaster().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Dummy example inputs
    era5 = torch.randn(4, 6, 37, 61, 5).to(device)
    satellite = torch.randn(4, 3, 64, 64).to(device)

    target = torch.randn(4, 5).to(device)

    model.train()

    pred = model(era5, satellite)

    loss = ((pred - target) ** 2).mean()

    loss.backward()
    optimizer.step()

    print("Training step complete")
    print("Loss:", loss.item())


if __name__ == "__main__":
    main()
