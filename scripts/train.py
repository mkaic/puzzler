import torch
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from ..src.model import Puzzler
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

KERNEL_SIZE = 8
NUM_FILTERS = 64
HIDDEN_STATE_SIZE = 256
NUM_CLASSES = 100
MID_LAYER_SIZE = 128
N_MAIN_LAYERS = 4
ITERATIONS = 8


print(
    f"{KERNEL_SIZE=}, {HIDDEN_STATE_SIZE=}, {NUM_CLASSES=}, {MID_LAYER_SIZE=}, {ITERATIONS=}"
)

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
EPOCHS = 40
BATCH_SIZE = 256
LR = 1e-3
COMPILE = True
SAVE = False


print(
    f"""
{BATCH_SIZE=}, 
{EPOCHS=}
"""
)

if not Path("puzzler/weights").exists():
    Path("puzzler/weights").mkdir(parents=True)

model = Puzzler(
    kernel_size=KERNEL_SIZE,
    num_filters=NUM_FILTERS,
    hidden_state_size=HIDDEN_STATE_SIZE,
    num_classes=NUM_CLASSES,
    mid_layer_size=MID_LAYER_SIZE,
    loss_function=nn.CrossEntropyLoss(),
    dtype=DTYPE,
)
model = model.to(DEVICE)
model = model.to(DTYPE)

num_params = sum(p.numel() for p in model.parameters())
print(f"{num_params:,} trainable parameters")


# Load the MNIST dataset
train = CIFAR100(root="./puzzler/data", train=True, download=True, transform=ToTensor())
test = CIFAR100(root="./puzzler/data", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(
    train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=8
)
test_loader = DataLoader(
    test, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=8
)

# Train the model
optimizer = Adam(model.parameters(), lr=LR)

test_accuracy = 0
for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, leave=False)

    for images, labels in pbar:
        optimizer.zero_grad()

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        images, labels = images.to(DTYPE), labels.to(torch.long)

        loss, _ = model.multistep(image=images, labels=labels, iterations=ITERATIONS)

        loss.backward()

        optimizer.step()

        pbar.set_description(
            f"Epoch {epoch}. Train: {loss.item():.4f}, Test: {test_accuracy:.2%}"
        )

    model.eval()
    if SAVE:
        torch.save(model.state_dict(), f"puzzler/weights/{epoch:03d}.ckpt")

    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, leave=False):

            images: torch.Tensor
            labels: torch.Tensor

            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images, labels = images.to(DTYPE), labels.to(torch.long)

            _, predictions = model.multistep(
                image=images, labels=labels, iterations=ITERATIONS
            )
            _, predicted = torch.max(predictions, dim=1)

            total += labels.shape[0]
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f"Epoch {epoch + 1}: {test_accuracy:.2%} accuracy on test set")
