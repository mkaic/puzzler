import torch.nn as nn
import torch
from typing import Tuple



class Puzzler(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        pass

    def forward(self, image, iterations: int = 1):
        
        center = (image.shape[-2] // 2, image.shape[-1] // 2)
        coordinates = get_kernel_coordinates(kernel_size=self.kernel_size)