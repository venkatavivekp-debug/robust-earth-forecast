import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.remote_sensing.cnn_landcover import LandCoverCNN

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

dataset = datasets.EuroSAT(
    root="data",
    download=True,
    transform=transform
)

loader = DataLoader(dataset,batch_size=32,shuffle=True)

model = LandCoverCNN()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):

    for x,y in loader:

        yhat = model(x)
        loss = loss_fn(yhat,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch",epoch,"loss",loss.item())
