import torch
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from ..src.neural_interp import BilinearInterpolator
import wandb

wandb.init(project="bilinear_interpolator", entity="mkaichristensen")

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
ITERATIONS = 100_000
BATCH_SIZE = 32
LR = 1e-3
SAVE = True

weights_path = Path("puzzler/weights/interpolator/")
weights_path.mkdir(parents=True, exist_ok=True)

model = BilinearInterpolator(n_layers=8, c=16, l=16)
model.train()
model = model.to(DEVICE)

model = torch.compile(model, dynamic=True)

num_params = sum(p.numel() for p in model.parameters())
print(f"{num_params:,} trainable parameters")


# Train the model
optimizer = Adam(model.parameters(), lr=LR)

pbar = tqdm(range(ITERATIONS))
for i in pbar:

    if i % 32 == 0:
        optimizer.zero_grad()

    H_, W_ = torch.randint(8, 256, (2,), device=DEVICE)
    H, W = torch.randint(32, 256, (2,), device=DEVICE)
    

    input_images = torch.rand((BATCH_SIZE, 3, H_, W_), device=DEVICE, dtype=DTYPE)
    input_images = F.interpolate(input_images, size=(H, W), mode="bilinear")

    input_sample_coords = torch.rand((BATCH_SIZE, 1, 1, 2), device=DEVICE, dtype=DTYPE)
    for_grid_sample = (input_sample_coords - 0.5) * 2 # scale to [-1, 1]
    for_grid_sample = for_grid_sample[..., [1, 0]] # swap x and y
    input_labels = (
        F.grid_sample(input_images, input_sample_coords, align_corners=False)
        .squeeze(2)
        .detach()
    )

    input_sample_coords = input_sample_coords.squeeze(2)

    output = model(input_images, input_sample_coords)
    loss = F.mse_loss(output, input_labels)

    loss.backward()

    if i % 32 == 0:
        optimizer.step()

    pbar.set_description(f"{loss.item():.6f}")

    if i % 200 == 0:
        wandb.log(data={"loss": loss.item()}, step=i)

    if SAVE and ((i + 1) % 10_000) == 0:
        torch.save(model.state_dict(), weights_path / f"{i+1:07d}.ckpt")
