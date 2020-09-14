# -*- coding: utf-8 -*-
import torch


def pixel_norm(x: torch.Tensor, epsilon: float = 1e-8):
    """Applies a pixel-wise normalization.

    Note:
        Implemented as described in `this paper <https://arxiv.org/pdf/1710.10196.pdf>`_.
        `Reference <https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120-L122>`_.
    """  # noqa: E501
    return x * (x.pow(2).mean(axis=1, keepdim=True) + epsilon).rsqrt()
