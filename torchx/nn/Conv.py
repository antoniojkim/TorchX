# -*- coding: utf-8 -*-
import torch
from numpy import prod, sqrt


class Conv2d(torch.nn.Module):
    r"""Applies a 2D convolution over an input signal composed of several input planes.

    A simpler, modified version of the standard `torch.nn.Conv2d`, which supports an
    equalized learning rate by scaling the weights dynamically in each forward pass.
    Implemented as described in https://arxiv.org/pdf/1710.10196.pdf
    Reference: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L23-L29

    The weight parameter is initialized using the standard normal if use_wscale is True.
    The bias parameter is initialized to zero.

    Parameters:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel. Default: 3
        stride (int or tuple): Stride of the convolution. Default: 1
        padding (int or tuple): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple): Spacing between kernel elements. Default: 1
        bias (bool): If True, adds a learnable bias to the output. Default: True
        gain (float): The gain for the scaled weight. Default: sqrt(2)
        use_wscale (bool): If True, scales the weights in each forward pass. Default: False
        fan_in (float): Size of the weight parameter to scale by. Default: None

    Note:

        If :attr:`fan_in` is not provided, it is computed as :math:`\text{fan_in} = \text{in_channels} \times \text{kernel_size} ^ 2`

    Note:

        The :attr:`wscale` is computed as :math:`\text{wscale} = \frac{\text{gain}}{\sqrt{\text{fan_in}}}`

    Note:

        See `torch.nn.Conv2d <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d>`_
        for more details on the 2d convolution operator.
    """  # noqa: E501

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        gain: float = sqrt(2),
        use_wscale: bool = False,
        fan_in: float = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)
        self.padding = torch.nn.modules.utils._pair(padding)
        self.dilation = torch.nn.modules.utils._pair(dilation)

        if fan_in is None:
            fan_in = in_channels * prod(kernel_size)

        self._wscale = gain / sqrt(fan_in)
        self.use_wscale = use_wscale

        self.weight = torch.nn.Parameter(
            torch.Tensor(out_channels, in_channels, *kernel_size)
        )

        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))

    def reset_parameters(self):
        if self.use_wscale:
            torch.nn.init.normal_(self.weight)
            self.wscale = self._wscale
        else:
            torch.nn.init.normal_(self.weight, 0, self._wscale)
            self.wscale = 1

        if self.bias is not None:
            self.bias.fill_(0)

    def forward(self, x):
        return torch.nn.functional.conv2d(
            input=x,
            weight=(self.weight * self.wscale) if self.use_wscale else self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    def extra_repr(self):
        return ", ".join(
            str(self.in_channels),
            str(self.out_channels),
            f"kernel_size={self.kernel_size}",
            f"stride={self.stride}",
            f"padding={self.padding}",
            "bias=False" if self.bias is None else "",
            "use_wscale=True" if self.use_wscale else "",
        )


class ConvTranspose2d(torch.nn.Module):
    r"""Applies a 2D convolution transpose over an input signal composed of several input planes.

    A simpler, modified version of the standard `torch.nn.ConvTranspose2d`, which supports an
    equalized learning rate by scaling the weights dynamically in each forward pass.
    Implemented as described in https://arxiv.org/pdf/1710.10196.pdf
    Reference: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L23-L29

    The weight parameter is initialized using the standard normal if use_wscale is True.
    The bias parameter is initialized to zero.

    Parameters:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        bias: If True, adds a learnable bias to the output
        gain: The gain for the scaled weight
        use_wscale: If True, scales the weights in each forward pass
        fan_in: Size of the weight parameter to scale by

    Note:

        If :attr:`fan_in` is not provided, it is computed as :math:`\text{fan_in} = \text{in_channels} \times \text{kernel_size} ^ 2`

    Note:

        The :attr:`wscale` is computed as :math:`\text{wscale} = \frac{\text{gain}}{\sqrt{\text{fan_in}}}`

    Note:

        See `torch.nn.ConvTranspose2d <https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d>`_
        for more details on the 2d convolution operator.
    """  # noqa: E501

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        gain: float = sqrt(2),
        use_wscale: bool = False,
        fan_in: float = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if fan_in is None:
            fan_in = in_channels * kernel_size ** 2

        self.wscale = gain / sqrt(fan_in)

        self.weight = torch.nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )

        if use_wscale:
            torch.nn.init.normal_(self.weight)
        else:
            torch.nn.init.normal_(self.weight, 0, self.wscale)
            self.wscale = 1

        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        return torch.nn.functional.conv_transpose2d(
            input=x,
            weight=self.weight * self.wscale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

    def extra_repr(self):
        return ", ".join(
            str(self.in_channels),
            str(self.out_channels),
            f"kernel_size={self.kernel_size}",
            f"stride={self.stride}",
            f"padding={self.padding}",
            "bias=False" if self.bias is None else "",
            "use_wscale=True" if self.wscale != 1 else "",
        )


def Conv2dBatch(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    bias: bool = True,
    leaky: float = None,
    **kwargs,
):
    """A 2D convolution followed by a batch normalization and ReLU activation."""
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
    **kwargs,
):
    """A 2D convolution transpose followed by a batch normalization
    and ReLU activation.
    """
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
    **kwargs,
):
    """A 2D convolution followed by a group norm and ReLU activation."""
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
    """Depth-wise separable convolution followed by a 2D convolution
    each followed by a batch normalization and ReLU activation.
    """
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
    """Depth-wise separable convolution followed by a batch normalization
    and ReLU activation.
    """
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False
        ),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True),
    )
