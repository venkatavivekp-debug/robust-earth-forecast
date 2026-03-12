import matplotlib.pyplot as plt
import torch

era5 = torch.randn(21,21)

downscaled = torch.randn(84,84)

plt.subplot(1,2,1)
plt.title("ERA5")
plt.imshow(era5)

plt.subplot(1,2,2)
plt.title("Downscaled")
plt.imshow(downscaled)

plt.show()
