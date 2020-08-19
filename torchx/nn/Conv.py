# -*- coding: utf-8 -*-
import torch
import numpy as np


def Conv2dBatch(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    bias: bool = True,
    leaky: float = None,
    **kwargs
):
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True)
        if leaky is None
        else torch.nn.LeakyReLU(leaky, inplace=True),
    )


def ConvTranspose2dBatch(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 4,
    stride: int = 2,
    padding: int = 0,
    bias: bool = False,
    leaky: float = None,
    **kwargs
):
    return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True)
        if leaky is None
        else torch.nn.LeakyReLU(leaky, inplace=True),
    )


def Conv2dGroup(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    bias: bool = True,
    num_groups=1,
    **kwargs
):
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
        torch.nn.GroupNorm(num_groups, out_channels),
        torch.nn.ReLU(inplace=True),
    )


def DSConv(in_channels: int, out_channels: int, stride: int = 1, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False
        ),
        torch.nn.BatchNorm2d(in_channels),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(in_channels, out_channels, 1, bias=False),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True),
    )


def DWConv(in_channels: int, out_channels: int, stride: int = 1, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False
        ),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True),
    )


class EqualizedConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()

        self.stride = stride
        self.padding = padding

        self.weight = torch.nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        torch.nn.init.normal_(self.weight)

        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))

        fan_in = in_channels * kernel_size ** 2
        self.scale = np.sqrt(2) / fan_in

    def forward(self, x):
        return torch.nn.functional.conv2d(
            input=x,
            weight=self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )


class EqualizedConvTranspose2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()

        self.stride = stride
        self.padding = padding

        self.weight = torch.nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        )
        torch.nn.init.normal_(self.weight)

        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))

        fan_in = np.sqrt(in_channels)
        self.scale = np.sqrt(2) / fan_in

    def forward(self, x):
        return torch.nn.functional.conv_transpose2d(
            input=x,
            weight=self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
