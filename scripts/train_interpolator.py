import torch
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from ..src.neural_interp import BilinearInterpolator


DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
ITERATIONS = 1_000_000
BATCH_SIZE = 512
LR = 1e-5
COMPILE = False
SAVE = True

weights_path = Path("puzzler/weights/interpolator/")
weights_path.mkdir(parents=True, exist_ok=True)

model = BilinearInterpolator(n_layers=16, c=128, l=128)
model.train()
model = model.to(DEVICE)

# model = torch.compile(model)

num_params = sum(p.numel() for p in model.parameters())
print(f"{num_params:,} trainable parameters")


# Train the model
optimizer = Adam(model.parameters(), lr=LR)

test_accuracy = 0
pbar = tqdm(range(ITERATIONS))
for i in pbar:

    optimizer.zero_grad()

    input_images = torch.rand((BATCH_SIZE, 3, 8, 8), device=DEVICE, dtype=DTYPE)
    input_images = F.interpolate(input_images, size=(32, 32), mode="bilinear")


    input_sample_coords = torch.rand((BATCH_SIZE, 1, 1, 2), device=DEVICE, dtype=DTYPE)
    input_labels = (
        F.grid_sample(input_images, input_sample_coords, align_corners=False)
        .squeeze(2)
        .detach()
    )

    input_sample_coords = input_sample_coords.squeeze(2)

    output = model(input_images, input_sample_coords)
    loss = F.mse_loss(output, input_labels)

    loss.backward()

    optimizer.step()

    pbar.set_description(f"{loss.item():.6f}")
    if SAVE and ((i + 1) % 100) == 0:
        torch.save(model.state_dict(), weights_path / f"{i+1:07d}.ckpt")
