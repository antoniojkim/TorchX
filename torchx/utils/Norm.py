# -*- coding: utf-8 -*-


def pixel_norm(x, epsilon=1e-8):
    """
    Implemented as described here: https://arxiv.org/pdf/1710.10196.pdf
    Reference: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120-L122
    """  # noqa: E501
    return x * (x.pow(2).mean(axis=1, keepdim=True) + epsilon).rsqrt()
