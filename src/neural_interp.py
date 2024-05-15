import torch.nn as nn
from torch import Tensor
import torch


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class BilinearInterpolator(nn.Module):
    """A convolutional neural network which learns bilinear sampling."""

    def __init__(self, n_layers, c, l):
        super().__init__()
        c = c
        l = l
        n_layers = 16
        conv_layers = [
            ConvRelu(3, c, kernel_size=3, padding=1),  # c x 32x32

            *[ConvRelu(c, c, kernel_size=3, padding=1)] * n_layers,

            *[ConvRelu(c, c, kernel_size=2, stride=2, padding=0)] * 5, # c x 1 x 1

            ConvRelu(c, l, kernel_size=1, padding=0),  # l x 1 x 1
            nn.Flatten(),
        ]
        linear_layers = [
            nn.Linear(l + 2, l),
            nn.ReLU(),
            *[nn.Linear(l, l), nn.ReLU()]*3,
            nn.Linear(l, 3),
            nn.Sigmoid(),
        ]

        self.conv_layers = nn.Sequential(*conv_layers)
        self.linear_layers = nn.Sequential(*linear_layers)

    def forward(self, image: Tensor, coords: Tensor) -> Tensor:

        # append a 3-pixel-wide block of coords_values to the right of the image
        B, C, H, W = image.shape
        B, N, D = coords.shape

        image = image.reshape(B, 1, C, H, W).expand(B, N, C, H, W)
        image = image.reshape(B*N, C, H, W)

        coords = coords.reshape(B*N, D)

        image_features = self.conv_layers(image).reshape(B*N, -1)

        features = torch.cat([image_features, coords], dim=-1)

        predictions:Tensor = self.linear_layers(features)
        predictions = predictions.reshape(B, N, 3)
        predictions = predictions.permute(0, 2, 1)

        return predictions
