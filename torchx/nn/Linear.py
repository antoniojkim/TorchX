# -*- coding: utf-8 -*-
import torch
import numpy as np


class EqualizedLinear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True,
    ):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.normal_(self.weight)

        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))

        fan_in = np.sqrt(in_features)
        self.scale = np.sqrt(2) / fan_in

    def forward(self, x):
        return torch.nn.functional.linear(
            input=x, weight=self.weight * self.scale, bias=self.bias,
        )
