# -*- coding: utf-8 -*-

import torch


def Conv2dBatch(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    bias: bool = True,
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
        torch.nn.ReLU(inplace=True),
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
