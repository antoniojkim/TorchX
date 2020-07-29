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
* [`Conv2dBatch`](https://github.com/antoniojkim/TorchX/blob/master/torchx/nn/Conv.py#L5): a 2d convolution followed by a [batch normalization](https://arxiv.org/pdf/1502.03167.pdf) and a ReLU activation
* [`ConvTranspose2dBatch`](https://github.com/antoniojkim/TorchX/blob/master/torchx/nn/Conv.py#L28): a 2d convolution transpose followed by a [batch normalization](https://arxiv.org/pdf/1502.03167.pdf) and a ReLU activation
* [`Conv2dGroup`](https://github.com/antoniojkim/TorchX/blob/master/torchx/nn/Conv.py#L51): a 2d convolution followed by a [group normalization](https://arxiv.org/pdf/1803.08494.pdf) and a ReLU activation
* [`DSConv`](https://github.com/antoniojkim/TorchX/blob/master/torchx/nn/Conv.py#L75)
* [`DWConv`](https://github.com/antoniojkim/TorchX/blob/master/torchx/nn/Conv.py#L88)

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
