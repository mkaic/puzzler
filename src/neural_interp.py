import torch.nn as nn
from torch import Tensor
import torch


class ConvAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=0,
        stride=1,
        residual=False,
    ):
        super().__init__()

        self.residual = residual

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):

        x_ = self.conv(x)
        if self.residual:
            x = x + x_
        else:
            x = x_

        return self.act(x)


class BilinearInterpolator(nn.Module):
    """A convolutional neural network which learns bilinear sampling."""

    def __init__(self, n_reps, c, l):
        super().__init__()
        layers = [
            ConvAct(3 + 2 + 2, c, kernel_size=1, padding=0),
            ConvAct(c, c, kernel_size=1, padding=0),
            ConvAct(c, c, kernel_size=1, padding=0),
            ConvAct(c, c, kernel_size=1, padding=0),
            ConvAct(c, c, kernel_size=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c, 3),
            nn.Sigmoid(),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, image: Tensor, coords: Tensor) -> Tensor:

        # append a 3-pixel-wide block of coords_values to the right of the image
        B, C, H, W = image.shape
        B, N, D = coords.shape

        position_encoding = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=image.device) / (H - 1),
            torch.arange(W, dtype=torch.float32, device=image.device) / (W - 1),
            indexing="ij",
        )
        position_encoding = (
            torch.stack(position_encoding, dim=0).view(1, 2, H, W).expand(B, 2, H, W)
        )
        image = torch.cat([image, position_encoding], dim=1)

        image = image.reshape(B, 1, 3 + 2, H, W).expand(B, N, 3 + 2, H, W)
        image = image.reshape(B * N, 3 + 2, H, W)

        coords = coords.reshape(B * N, D, 1, 1).expand(B * N, D, H, W)

        image = torch.cat([image, coords], dim=1)  # B*N x 3+2+2 x H x W

        predictions: Tensor = self.layers(image)
        predictions = predictions.reshape(B, N, 3)
        predictions = predictions.permute(0, 2, 1)

        return predictions
