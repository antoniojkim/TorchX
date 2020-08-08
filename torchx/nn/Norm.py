# -*- coding: utf-8 -*-
import torch


class PixelwiseNorm(torch.nn.Module):
    """
    Implemented as described here: https://arxiv.org/pdf/1710.10196.pdf
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, a: torch.Tensor):
        b = (a.pow(2).mean(axis=1, keepdim=True) + self.epsilon).sqrt()
        return a / b
