import torch.nn as nn
from torch import Tensor, device
import torch
import math
import torch.nn.functional as F

EPSILON = 1e-8


# @torch.compile()
def gaussian_interp_2d(
    image: Tensor, coords: Tensor, device: device = None, pixel_std: float = 2
) -> Tensor:

    if device == None:
        device = image.device

    B, C, H, W = image.shape
    B, N, D = coords.shape

    # variance should be pixel-level. This formula means variance is one pixel, assuming a square image.
    std = pixel_std / image.shape[-2]

    gaussian = torch.meshgrid(
        torch.arange(-3, H + 3, dtype=torch.float32, device=device) / (H),
        torch.arange(-3, W + 3, dtype=torch.float32, device=device) / (W),
        indexing="ij",
    )

    image = F.pad(image, (3, 3, 3, 3), mode="reflect")

    _, _, H_pad, W_pad = image.shape

    gaussian = (
        torch.stack(gaussian, dim=0)
        .reshape(1, 1, D, H_pad, W_pad)
        .expand(B, N, D, H_pad, W_pad)
    )

    coords = coords.reshape(B, N, D, 1, 1).expand(B, N, D, H_pad, W_pad)

    gaussian = gaussian - coords
    gaussian = gaussian / std

    gaussian = torch.norm(gaussian, dim=2, keepdim=True)  # B x N x 1 x H_pad x W_pad

    gaussian = gaussian * (1 / (std * math.sqrt(2 * torch.pi)))

    gaussian = torch.exp(-(gaussian**2) / 2)

    image = image.reshape(B, 1, C, H_pad, W_pad).expand(B, N, C, H_pad, W_pad)

    image = image * gaussian

    pixel_values = torch.sum(image, dim=(-2, -1)) / (
        torch.sum(gaussian, dim=(-2, -1)) + EPSILON
    )

    return pixel_values
