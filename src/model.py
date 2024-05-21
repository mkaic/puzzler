import torch.nn as nn
import torch
from torch import Tensor
from typing import Tuple
from .gaussian_interp import gaussian_interp_2d
EPSILON = 1e-6

import math
import torch

# from https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
def sinusoidal_encode(position, dims):
    """position is in [0, 1] and has shape (batch_size, 1)"""
    pe = torch.zeros(1, dims)
    ndims = dims // 2

    position = position.float()
    div_term = torch.exp(torch.arange(0, ndims, 2).float() * -(math.log(10000.0) / ndims))
    pe[0, 0::2] = torch.sin(position * div_term)
    pe[0, 1::2] = torch.cos(position * div_term)

    return pe

class Act(nn.Module):
    def __init__(self):
        super().__init__()

        self.a = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = torch.abs(x)
        return x

class Puzzler(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        hidden_state_size: int,
        num_classes: int,
        input_channels: int = 3,
        mid_layer_size: int = 256,
        num_main_layers: int = 4,
        loss_function: nn.Module = nn.CrossEntropyLoss(),
        dtype: torch.dtype = torch.float32,
        pe_dim: int = 1
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.num_pixel_values = kernel_size * kernel_size
        self.hidden_state_size = hidden_state_size
        self.loss_function = loss_function
        self.dtype = dtype
        self.pe_dim = pe_dim

        input_vector_size = (
            self.pe_dim * 2
            + self.pe_dim
            + (self.num_pixel_values * self.input_channels)
            + self.hidden_state_size
        )
        output_vector_size = 2 + 1 + self.hidden_state_size

        layers = [nn.Linear(input_vector_size, mid_layer_size), Act()]
        for i in range(num_main_layers):
            layers.append(nn.Linear(mid_layer_size, mid_layer_size))
            layers.append(Act())
        layers.append(nn.Linear(mid_layer_size, output_vector_size))
        self.main_mlp = nn.Sequential(*layers)

        self.classifier_mlp = nn.Sequential(
            nn.Linear(hidden_state_size, mid_layer_size),
            Act(),
            nn.Linear(mid_layer_size, mid_layer_size),
            Act(),
            nn.Linear(mid_layer_size, num_classes),
            nn.Softmax(dim=-1),
        )

        base_meshgrid = torch.meshgrid(
            *[
                torch.arange(kernel_size, dtype=self.dtype) / kernel_size
                for _ in range(2)
            ],
            indexing="ij"
        )  # grid_sample expects xy indexing
        base_meshgrid = torch.stack(
            base_meshgrid, dim=-1
        )  # (kernel_size, kernel_size, 2)
        self.register_buffer("meshgrid", base_meshgrid)

    def forward(
        self,
        image: Tensor,
        locations: Tensor = None,
        scales: Tensor = None,
        hidden_state: Tensor = None,
    ) -> dict:

        device = image.device
        B, C, H, W = image.shape
        if locations is None:
            locations = torch.rand(B, 2, dtype=self.dtype, device=device)

        if scales is None:
            scales = torch.rand(B, 1, dtype=self.dtype, device=device) * 0.5
        if hidden_state is None:
            hidden_state = (
                torch.randn(B, self.hidden_state_size, dtype=self.dtype, device=device)
                * 0.01
            )

        coordinates = torch.clone(self.meshgrid).to(device)
        coordinates = coordinates - locations.view(B, 1, 1, 2)
        coordinates = coordinates * scales.view(B, 1, 1, 1)
        coordinates = coordinates.reshape(B, self.num_pixel_values, 2)

        pixel_values = gaussian_interp_2d(image=image, coords=coordinates, pixel_std=2)
        pixel_values = pixel_values.reshape(
            B, self.num_pixel_values * self.input_channels
        )

        input_vectors = torch.cat(
            [locations, scales, pixel_values, torch.tanh(hidden_state)], dim=-1
        )

        output_vectors = self.main_mlp(input_vectors)

        locations = torch.sigmoid(output_vectors[:, :2])

        scales = torch.sigmoid(output_vectors[:, 2:3]) * 0.5

        hidden_state = hidden_state + output_vectors[:, 3:]  # residual connection

        predictions = self.classifier_mlp(torch.tanh(hidden_state))

        return {
            "locations": locations,
            "scales": scales,
            "hidden_state": hidden_state,
            "predictions": predictions,
        }

    def multistep(
        self, image, iterations, labels=None, print_locations=False, random_locations=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        multistep_losses = []
        states = []
        state = {}
        for _ in range(iterations):
            if random_locations:
                state["locations"] = None
            state: dict = self.forward(image=image, **state)
            states.append(state)

            predictions = state.pop("predictions")

            if labels is not None:
                multistep_losses.append(self.loss_function(predictions, labels))

        if labels is not None:

            classification_loss: torch.Tensor = torch.stack(
                multistep_losses, dim=-1
            ).mean(dim=-1)

            scales = torch.stack([state["scales"] for state in states], dim=-1)

            locations = torch.stack([state["locations"] for state in states], dim=-1)

            if print_locations:
                print(locations[0, :, [0, 1, -2, -1]])
                print(scales[0, :, [0, 1, -2, -1]])
                print(self.main_mlp[5].a.item())

            loss = classification_loss

            return loss, predictions
        else:
            return predictions
