import torch
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from ..src.model import Puzzler
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

import warnings

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3

COMPILE = False
SAVE = False


print(
    f"""
{BATCH_SIZE=}, 
{EPOCHS=}
"""
)

if not Path("puzzler/weights").exists():
    Path("puzzler/weights").mkdir(parents=True)

model = Puzzler()
model = model.to(DEVICE)
model = model.to(DTYPE)


# Load the MNIST dataset
train = CIFAR100(root="./puzzler/data", train=True, download=True, transform=ToTensor())
test = CIFAR100(root="./puzzler/data", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(
    train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=4
)
test_loader = DataLoader(
    test, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4
)

# Train the model
optimizer = Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss().to(DEVICE)

test_accuracy = 0
for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, leave=False)

    for x, y in pbar:
        optimizer.zero_grad()

        x, y = x.to(DEVICE), y.to(DEVICE)
        x, y = x.to(DTYPE), y.to(torch.long)

        y_hat = model(x)

        loss = criterion(y_hat, y)
        loss: torch.Tensor
        loss.backward()

        optimizer.step()

        model.clamp_params()

        pbar.set_description(
            f"Epoch {epoch}. Train: {loss.item():.4f}, Test: {test_accuracy:.2%}"
        )

    model.eval()
    if SAVE:
        torch.save(model.state_dict(), f"puzzler/weights/{epoch:03d}.ckpt")

    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, leave=False):

            x: torch.Tensor
            y: torch.Tensor

            x, y = x.to(DEVICE), y.to(DEVICE)
            x, y = x.to(DTYPE), y.to(torch.long)

            y_hat = model(x)

            _, predicted = torch.max(y_hat, dim=1)
            total += y.shape[0]
            correct += (predicted == y).sum().item()

    test_accuracy = correct / total
    print(f"Epoch {epoch + 1}: {test_accuracy:.2%} accuracy on test set")
