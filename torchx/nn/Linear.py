# -*- coding: utf-8 -*-
import torch
from numpy import sqrt


class Linear(torch.nn.Module):
    """
    Applies a linear transformation to the incoming data

    A simpler, modified version of the normal torch.nn.Conv2d which supports an
    equalized learning rate by scaling the weights dynamically in each forward pass.
    Implemented as described in https://arxiv.org/pdf/1710.10196.pdf
    Reference: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L23-L29

    The weight parameter is initialized using the standard normal if use_wscale is True.
    The bias parameter is initialized to zero.

    Parameters
    ----------
    in_features: size of each input sample
    out_features: size of each output sample
    bias: If set to True, the layer will add a learnable additive bias.
    """  # noqa: E501

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gain: float = sqrt(2),
        use_wscale: bool = False,
        fan_in: float = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        if fan_in is None:
            fan_in = in_features

        self.wscale = gain / sqrt(fan_in)

        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        if use_wscale:
            torch.nn.init.normal_(self.weight)
        else:
            torch.nn.init.normal_(self.weight, 0, self.wscale)
            self.wscale = 1

        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return torch.nn.functional.linear(
            input=x, weight=self.weight * self.wscale, bias=self.bias,
        )

    def extra_repr(self):
        return ", ".join(
            f"in_features={self.in_features}",
            f"out_features={self.out_features}",
            f"bias={self.bias is not None}",
        )
