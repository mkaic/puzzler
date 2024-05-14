import torch.nn as nn
import torch
from torch import Tensor
from typing import Tuple
from .fourier_interp import fourier_interp_2d

class ResidualLinearLayer(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(size, size)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x) + x

class Puzzler(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        num_filters: int,
        hidden_state_size: int,
        num_classes: int,
        input_channels: int = 3,
        mid_layer_size: int = 256,
        num_main_layers: int = 4,
        loss_function: nn.Module = nn.CrossEntropyLoss(),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.num_pixel_values = kernel_size * kernel_size
        self.num_filters = num_filters
        self.hidden_state_size = hidden_state_size
        self.loss_function = loss_function
        self.dtype = dtype

        input_vector_size = 2 + 1 + (self.num_pixel_values * self.input_channels) + self.hidden_state_size
        output_vector_size = 2 + 1 + self.hidden_state_size

        # self.conv = nn.Conv2d(
        #     in_channels=3,
        #     out_channels=num_filters,
        #     kernel_size=kernel_size,
        # )

        layers = [nn.Linear(input_vector_size, mid_layer_size)]
        for i in range(num_main_layers):
            layers.append(ResidualLinearLayer(mid_layer_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(mid_layer_size, output_vector_size))
        self.main_mlp = nn.Sequential(*layers)

        self.classifier_mlp = nn.Sequential(
            nn.Linear(hidden_state_size, mid_layer_size),
            nn.ReLU(),
            ResidualLinearLayer(mid_layer_size),
            nn.ReLU(),
            nn.Linear(mid_layer_size, num_classes),
            nn.Softmax(dim=-1),
        )

        base_meshgrid = torch.meshgrid(
            *[torch.arange(kernel_size, dtype=self.dtype) / kernel_size for _ in range(2)],
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
            locations = (
                torch.tensor([0.0, 0.0], dtype=self.dtype, device=device)
                .unsqueeze(0)
                .expand(B, 2)
            )
        if scales is None:
            scales = (
                torch.tensor([0.5], dtype=self.dtype, device=device)
                .unsqueeze(0)
                .expand(B, 1)
            )
        if hidden_state is None:
            hidden_state = torch.zeros(
                B, self.hidden_state_size, dtype=self.dtype, device=device
            )

        coordinates = torch.clone(self.meshgrid).to(device)
        coordinates = coordinates - locations.view(B, 1, 1, 2)
        coordinates = coordinates * scales.view(B, 1, 1, 1)
        coordinates = coordinates.reshape(B, self.num_pixel_values, 2)

        pixel_values = fourier_interp_2d(image=image, sample_points=coordinates)
        pixel_values = pixel_values.reshape(B, self.num_pixel_values * self.input_channels)
        
        # filtered = self.conv(pixel_values).reshape(batch_size, -1)

        input_vectors = torch.cat([locations, scales, pixel_values, hidden_state], dim=-1)

        output_vectors = torch.tanh(self.main_mlp(input_vectors))

        locations = locations + output_vectors[:, :2] 
        scales = scales + output_vectors[:, 2:3]

        hidden_state = hidden_state + output_vectors[:, 3:]  # residual connection

        predictions = self.classifier_mlp(hidden_state)

        # print(locations[0])

        return {
            "locations": locations,
            "scales": scales,
            "hidden_state": hidden_state,
            "predictions": predictions,
        }

    def multistep(
        self, image, iterations, labels=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        multistep_losses = []
        state = {}
        for _ in range(iterations):
            state: dict = self.forward(image=image, **state)
            predictions = state.pop("predictions")

            if labels is not None:
                multistep_losses.append(self.loss_function(predictions, labels))

        if labels is not None:
            loss: torch.Tensor = torch.stack(multistep_losses, dim=-1).mean(dim=-1)
            return loss, predictions
        else:
            return predictions
