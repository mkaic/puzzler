import torch
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F


DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
ITERATIONS = 1_000_000
BATCH_SIZE = 512
LR = 1e-3
COMPILE = True
SAVE = False

weights_path = Path("puzzler/weights/interpolator/")
weights_path.mkdir(parents=True, exist_ok=True)


if not Path("puzzler/weights").exists():
    Path("puzzler/weights").mkdir(parents=True)

model = BilinearInterpolator()
model.train()
model = model.to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())
print(f"{num_params:,} trainable parameters")


# Train the model
optimizer = Adam(model.parameters(), lr=LR)

test_accuracy = 0
pbar = tqdm(range(ITERATIONS))
for i in pbar:

    optimizer.zero_grad()

    input_images = torch.rand((BATCH_SIZE, 3, 32, 32), device=DEVICE, dtype=DTYPE)
    input_sample_coords = torch.rand((BATCH_SIZE, 1, 1, 2), device=DEVICE, dtype=DTYPE)
    input_labels = F.grid_sample(
        input_images, input_sample_coords, align_corners=True
    ).detach()

    output = model(input_images, input_sample_coords)
    loss = F.mse_loss(output, input_labels)

    loss.backward()

    optimizer.step()

    pbar.set_description(f"{loss.item():.6f}")
    if SAVE and i + 1 % 10_000 == 0:
        torch.save(model.state_dict(), weights_path / f"{i+1:07d}.ckpt")
