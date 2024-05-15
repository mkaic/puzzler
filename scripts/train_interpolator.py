import torch
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from torch import Tensor


DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
ITERATIONS = 1_000_000
BATCH_SIZE = 512
LR = 1e-3
COMPILE = True
SAVE = False


class BilinearInterpolator(nn.Module):
    """A convolutional neural network which learns bilinear sampling."""

    def __init__(self):
        super().__init__()
        layers = [
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32x32
            nn.ReLU(),
            *[nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU()] * 8,
            nn.AvgPool2d(2),  # 32x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            *[nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU()] * 8,
            nn.AdaptiveAvgPool2d((1,1)),  # 64x1
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, image: Tensor, coords: Tensor) -> Tensor:

        # append a 3-pixel-wide block of coords_values to the right of the image
        B, C, H, W = image.shape
        device, dtype = image.device, image.dtype
        x = torch.ones(B, 3, H // 2, 3, device=device, dtype=dtype)
        x = x * coords[..., 0].view(B, 1, 1, 1)

        y = torch.ones(B, 3, H - (H // 2), 3, device=device, dtype=dtype)
        y = y * coords[..., 1].view(B, 1, 1, 1)

        coords = torch.cat([x, y], dim=2)  # B, 1, H, 3
        image = torch.cat([image, coords], dim=3)

        return self.layers(x)


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
