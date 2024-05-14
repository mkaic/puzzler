import torch
from torchvision.transforms.functional import to_tensor
from .src.fourier_interp import fourier_interp_2d
import matplotlib.pyplot as plt
from PIL import Image

BATCH_SIZE = 3
H_i, W_i = 64, 64

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

image = Image.open("puzzler/branos_64.jpg")
image = to_tensor(image)
C, H_i, W_i = image.shape
image = image.reshape(1, C, H_i, W_i)
image = image.expand(BATCH_SIZE, C, H_i, W_i)

axes[0].imshow(image[1].permute(1, 2, 0))
axes[0].set_title("Input")
axes[0].axis("off")

for i, resolution in enumerate((32, 64, 128), start=1):

    H_a, W_a = (resolution, resolution)

    output_points = torch.meshgrid(
        torch.arange(H_a) / (H_a - 1),
        torch.arange(W_a) / (W_a - 1),
        indexing="ij",
    )
    output_points = torch.stack(output_points, dim=-1)
    output_points = output_points
    output_points = output_points.reshape(1, H_a, W_a, 2)
    output_points = output_points.expand(BATCH_SIZE, H_a, W_a, 2)
    output_points = output_points.reshape(BATCH_SIZE, H_a * W_a, 2)

    # freqency_mask = torch.meshgrid(
    #     torch.arange(H_i) / (H_i - 1),
    #     torch.arange(W_i) / (W_i - 1),
    #     indexing="ij",
    # )
    # frequency_mask = torch.stack(freqency_mask, dim=-1)
    # frequency_mask = torch.square(frequency_mask)
    # frequency_mask = torch.sum(frequency_mask, dim=-1)
    # frequency_mask = torch.sqrt(frequency_mask)
    # frequency_mask = frequency_mask < 1

    fourier_output = fourier_interp_2d(image=image, sample_points=output_points)

    fourier_output = fourier_output.view(BATCH_SIZE, C, H_a, W_a)[1]

    axes[i].imshow(fourier_output.permute(1, 2, 0))
    axes[i].set_title(f"Output {H_a}x{W_a}")
    axes[i].axis("off")

plt.tight_layout()
plt.savefig(f"puzzler/fourier_interp_sanity_check.png")
