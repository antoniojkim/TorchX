# -*- coding: utf-8 -*-
import torch

from ..utils import pixel_norm


class PixelwiseNorm(torch.nn.Module):
    """Torch module encapsulating the pixel norm operator"""

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor):
        return pixel_norm(x, self.epsilon)
