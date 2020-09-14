# -*- coding: utf-8 -*-
import torch


def minibatch_stddev_layer(x, group_size=4):
    """Appends a feature map containing the standard deviation of the minibatch.

    Note:
        Implemented as described in `this paper <https://arxiv.org/pdf/1710.10196.pdf>`_.
        `Reference <https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L127-L139>`_.
    """  # noqa: E501
    # Minibatch must be divisible by (or smaller than) group_size.
    group_size = min(group_size, x.shape[0])
    s = x.shape  # [NCHW]  Input shape.
    # [GMCHW] Split minibatch into M groups of size G.
    y = x.reshape(group_size, -1, s[1], s[2], s[3])
    y = y.float()  # [GMCHW] Cast to FP32.
    y -= y.mean(axis=0, keepdims=True)  # [GMCHW] Subtract mean over group.
    y = y.pow(2).mean(axis=0)  # [MCHW]  Calc variance over group.
    y = (y + 1e-8).sqrt()  # [MCHW]  Calc stddev over group.
    # [M111]  Take average over fmaps and pixels.
    y = y.mean(axis=[1, 2, 3], keepdims=True)
    y = y.to(x.dtype)  # [M111]  Cast back to original data type.
    y = y.repeat(group_size, 1, s[2], s[3])  # [N1HW]  Replicate over group and pixels.
    return torch.cat([x, y], axis=1)  # [NCHW]  Append as new fmap.
