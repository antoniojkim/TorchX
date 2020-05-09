# -*- coding: utf-8 -*-
from typing import Dict

import numpy


def encode_array(array: numpy.ndarray, encode_dict: Dict[int, numpy.ndarray]):
    """Encodes the provided array using the provided encoding dictionary.

    Note: this function is vectorized and is thus very fast.

    Args:
        array: The array to encode
        encode_dict: An dictionary of index, encoding value pairs. Each index must be
            an integer and each encoding value must be a numpy array of the same size
            as the last axis in the array that is to be encoded.

    Raises:
        AssertionError if the size of the last axis of the array does not match
            the sizes of the encoding values
    """
    assert all(array.shape[-1] == value.shape[0] for value in encode_dict.values())

    encoded = numpy.zeros(array.shape[:-1])
    for index, value in encode_dict.items():
        encoded[numpy.where(numpy.all(array == value, axis=-1))] = index

    return encoded


def decode_array(array: numpy.ndarray, decode_dict: Dict[int, numpy.ndarray]):
    """Decodes the provided array using the provided decoding dictionary.

    Note: this function is vectorized and is thus very fast.

    Args:
        array: The array to decode
        encode_dict: An dictionary of index, decoding value pairs. Each index must be
            an integer and each encoding value must be a numpy array of the same size
            as the last axis in the array that is to be decoded.

    Raises:
        AssertionError the decoding values are not all the same size
    """
    axis_size = numpy.unique([value.shape[0] for value in decode_dict.values()])
    assert len(axis_size) == 1
    axis_size = axis_size[0]

    colourized = numpy.zeros(list(array.shape) + [axis_size], numpy.uint32)
    for index, value in decode_dict.items():
        colourized[array == index] = value

    return colourized
