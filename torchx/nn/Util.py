# -*- coding: utf-8 -*-
import torch

from ..utils import minibatch_stddev_layer


class View(torch.nn.Module):
    """
    Set the view of a Tensor in a module
    """

    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor):
        return x.view(*self.shape)

    def __repr__(self):
        return f"View({', '.join(self.shape)})"


class MinibatchStddev(torch.nn.Module):
    """
    Increase the variation using minibatch standard deviation in a module
    """

    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor):
        return minibatch_stddev_layer(x, self.group_size)


class PrintShape(torch.nn.Module):
    """
    Print shape of tensor and then forward it to next module
    For debugging purposes
    """

    def __init__(self, format="{}"):
        super().__init__()
        self.format = format

    def forward(self, x: torch.Tensor):
        print(self.format.format(x.shape))
        return x


class Cond(torch.nn.Module):
    """
    Similar to tf.cond
    """

    def __init__(self, cond, a, b):
        super().__init__()
        self.cond = cond
        self.a = a
        self.b = b

    def forward(self, x: torch.Tensor):
        if self.cond():
            return self.a(x)
        else:
            return self.b(x)
