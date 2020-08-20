# TorchX

An "eXtension" to the PyTorch Deep Learning Framework. It contains "eXtra" features and functionality that I wish were in the official PyTorch framework.

## Install

TorchX is set up as a pip installable Python package:

```
pip install -U git+https://github.com/antoniojkim/TorchX.git
```

## Features

### nn

#### [Module](https://github.com/antoniojkim/TorchX/blob/master/torchx/nn/Module.py#L5)

A class that derives from `torch.nn.Module`. Features a number of convenient methods for saving and loading the model as well as a way to initializing weights.

#### Conv

A number of convenient Conv2D modules.
* [`Conv2d`](https://github.com/antoniojkim/TorchX/blob/master/torchx/nn/Conv.py#L6-L88): A simpler, modified version of the normal `torch.nn.Conv2d` which supports an
    equalized learning rate by scaling the weights dynamically in each forward pass.
* [`ConvTranspose2d`](https://github.com/antoniojkim/TorchX/blob/master/torchx/nn/Conv.py#L91-L173): A simpler, modified version of the normal `torch.nn.ConvTranspose2d` which supports an
    equalized learning rate by scaling the weights dynamically in each forward pass.
* [`Conv2dBatch`](https://github.com/antoniojkim/TorchX/blob/master/torchx/nn/Conv.py#L176-L199): a 2d convolution followed by a [batch normalization](https://arxiv.org/pdf/1502.03167.pdf) and a ReLU activation
* [`ConvTranspose2dBatch`](https://github.com/antoniojkim/TorchX/blob/master/torchx/nn/Conv.py#L202-L225): a 2d convolution transpose followed by a [batch normalization](https://arxiv.org/pdf/1502.03167.pdf) and a ReLU activation
* [`Conv2dGroup`](https://github.com/antoniojkim/TorchX/blob/master/torchx/nn/Conv.py#L228-L249): a 2d convolution followed by a [group normalization](https://arxiv.org/pdf/1803.08494.pdf) and a ReLU activation
* [`DSConv`](https://github.com/antoniojkim/TorchX/blob/master/torchx/nn/Conv.py#L252-L262)
* [`DWConv`](https://github.com/antoniojkim/TorchX/blob/master/torchx/nn/Conv.py#L265-L272)

#### Linear

* [`Linear`](https://github.com/antoniojkim/TorchX/blob/master/torchx/nn/Linear.py#L6-L66): a linear (dense) layer which supports an equalized learning rate by scaling the weights dynamically in each forward pass.

#### Norm

A number of additional norm modules.
* [`PixelwiseNorm`](https://github.com/antoniojkim/TorchX/blob/master/torchx/nn/Norm.py#L7): a module encapsulating the pixel norm operator (see utils below)

#### Util

A number of additional utility modules.
* [`View`](https://github.com/antoniojkim/TorchX/blob/master/torchx/nn/Util.py#L7): Set the view of a tensor within a module.
* [`MinibatchStddev`](https://github.com/antoniojkim/TorchX/blob/master/torchx/nn/Util.py#L23): a module encapsulating the Minibatch Standar Deviation function (see utils below)
* [`Lerp`](https://github.com/antoniojkim/TorchX/blob/master/torchx/nn/Interpolate.py#L7): a module encapsulating the lerp function (see utils below)

### optim

#### lr_scheduler

A number of convenient Learning Rate Schedulers
* [`Polynomial Decay`](https://github.com/antoniojkim/TorchX/blob/master/torchx/optim/lr_scheduler/PolynomialLR.py#L6)

### params

Contains a convenient class that reads parameters from a yaml file so that all parameters don't need to be passed in through the command line.
* [`Parameters`](https://github.com/antoniojkim/TorchX/blob/master/torchx/params/Parameters.py#L15)

### utils

Contains a number of useful utility functions
* [`encode_array`](https://github.com/antoniojkim/TorchX/blob/master/torchx/utils/OneHot.py#L7): Efficient way to encode an image into classes
* [`decode_array`](https://github.com/antoniojkim/TorchX/blob/master/torchx/utils/OneHot.py#L31): Efficient way to decode an array of labels to an image
* [`hex_to_rgb`](https://github.com/antoniojkim/TorchX/blob/master/torchx/utils/Colour.py#L7): Converts hex string into an rgb array

Contains a number of additional norm functions
* [`pixel_norm`](https://github.com/antoniojkim/TorchX/blob/master/torchx/utils/Norm.py#L4): a pixelwise feature vector normalization as described in the [Pro-GAN paper](https://arxiv.org/pdf/1710.10196.pdf)

Contains useful batch operations
* [`minibatch_stddev_layer`](https://github.com/antoniojkim/TorchX/blob/master/torchx/utils/Batch.py#L5): increase variance using minibatch standard deviation as described in the [Pro-GAN paper](https://arxiv.org/pdf/1710.10196.pdf)

Contains useful interpolation operations
* [`lerp`](https://github.com/antoniojkim/TorchX/blob/master/torchx/utils/Interpolate.py#L4): linear interpolation
