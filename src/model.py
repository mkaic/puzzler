import torch.nn as nn
import torch
from typing import Tuple
from torch.nn.functional import grid_sample


class Puzzler(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        hidden_state_size: int,
        num_classes: int,
        mid_layer_size: int = 256,
        loss_function: nn.Module = nn.CrossEntropyLoss(),
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.hidden_state_size = hidden_state_size

        input_vector_size = 2 + 1 + (self.kernel_size**2 * 3) + self.hidden_state_size
        output_vector_size = 2 + 1 + self.hidden_state_size

        self.main_mlp = nn.Sequential(
            nn.Linear(input_vector_size, mid_layer_size),
            nn.ReLU(),
            nn.Linear(mid_layer_size, mid_layer_size),
            nn.ReLU(),
            nn.Linear(mid_layer_size, mid_layer_size),
            nn.ReLU(),
            nn.Linear(mid_layer_size, output_vector_size),
        )

        self.classifier_mlp = nn.Sequential(
            nn.Linear(hidden_state_size, mid_layer_size),
            nn.ReLU(),
            nn.Linear(mid_layer_size, mid_layer_size),
            nn.ReLU(),
            nn.Linear(mid_layer_size, num_classes),
            nn.Softmax(dim=-1),
        )

        self.loss_function = loss_function

    def get_kernel_coordinates(
        self, centers: Tuple[float, float], scales: float, device: str = None
    ):

        # center is (i,j) indexed. center and scale are both relative to the image shape and clamped to [-1,1]

        linspaces = [
            torch.linspace(i - scales, i + scales, self.kernel_size, device=device)
            for i in centers
        ]
        meshgrid = torch.meshgrid(
            *linspaces, indexing="xy"
        )  # grid_sample expects xy indexing
        kernel_coordinates = torch.stack(meshgrid, dim=-1)
        return kernel_coordinates

    def forward(self, image, locations=None, scales=None, hidden_state=None) -> dict:

        device = image.device
        batch_size = image.shape[0]
        if locations is None:
            locations = (
                torch.tensor([0.0, 0.0], device=device)
                .unsqueeze(0)
                .expand(batch_size, 2)
            )
        if scales is None:
            scales = (
                torch.tensor([1.0], device=device).unsqueeze(0).expand(batch_size, 1)
            )
        if hidden_state is None:
            hidden_state = torch.zeros(
                batch_size, self.hidden_state_size, device=device
            )

        coordinates = self.get_kernel_coordinates(
            centers=locations, scales=scales, device=device
        )
        coordinates = coordinates.unsqueeze(0).expand(batch_size, *coordinates.shape)
        pixel_values = grid_sample(
            image, coordinates, align_corners=False, mode="bilinear"
        )
        pixel_values = pixel_values.flatten(start_dim=1, end_dim=-1)

        input_vectors = torch.cat(
            [locations, scales, pixel_values, hidden_state], dim=-1
        )

        output_vectors = self.main_mlp(input_vectors)

        locations = output_vectors[:, :2]
        scales = output_vectors[:, 2:3]
        hidden_state = output_vectors[:, 3:]

        predictions = self.classifier_mlp(hidden_state)

        return {
            "locations": locations,
            "scales": scales,
            "hidden_state": hidden_state,
            "predictions": predictions,
        }
    
    def multistep(self, image, iterations, labels=None) -> Tuple[torch.Tensor, torch.Tensor]:
        
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
