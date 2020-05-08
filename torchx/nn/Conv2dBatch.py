
import torch

def Conv2dBatch(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    padding=0,
    bias=True,
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
