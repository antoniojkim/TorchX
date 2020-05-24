# -*- coding: utf-8 -*-
from typing import Tuple

import numpy


def encode_array(array: numpy.ndarray, encode_list: Tuple[int, numpy.ndarray]):
    """Encodes the provided array using the provided encoding list.

    Note: this function is vectorized and is thus very fast.

    Args:
        array: The array to encode
        encode_list: A tuple of encoding value, index pairs. Each index must be
            an integer and each encoding value must be a numpy array of the same size
            as the last axis in the array that is to be encoded.

    Raises:
        AssertionError if the size of the last axis of the array does not match
            the sizes of the encoding values
    """
    assert all(array.shape[-1] == value.shape[0] for value, index in encode_list)

    encoded = numpy.zeros(array.shape[:-1])
    for value, index in encode_list:
        encoded[numpy.where(numpy.all(array == value, axis=-1))] = index

    return encoded


def decode_array(array: numpy.ndarray, decode_list: Tuple[int, numpy.ndarray]):
    """Decodes the provided array using the provided decoding list.

    Note: this function is vectorized and is thus very fast.

    Args:
        array: The array to decode
        encode_list: A tuple of decoding value, index pairs. Each index must be
            an integer and each encoding value must be a numpy array of the same size
            as the last axis in the array that is to be decoded.

    Raises:
        AssertionError the decoding values are not all the same size
    """
    axis_size = numpy.unique([value.shape[0] for value, index in decode_list])
    assert len(axis_size) == 1
    axis_size = axis_size[0]

    colourized = numpy.zeros(list(array.shape) + [axis_size], numpy.uint32)
    for value, index in decode_list:
        colourized[array == index] = value

    return colourized
