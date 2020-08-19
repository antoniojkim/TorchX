# -*- coding: utf-8 -*-
import torch

from ..utils import lerp


class Lerp(torch.nn.Module):
    """
    Linear Interpolation in a module
    """

    def __init__(self, a, b, t):
        super().__init__()
        self.a = a
        self.b = b
        self.t = t

    def forward(self, x: torch.Tensor):
        return lerp(self.a(x), self.b(x), self.t)
