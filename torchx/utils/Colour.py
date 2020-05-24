# -*- coding: utf-8 -*-
import re

import numpy


def hex_to_rgb(hex_colour: str):
    """Converts a colour from hex to rgb

    Params:
        hex_colour: a hex string matching the regex '[0-9a-fA-F]{6}'

    Returns:
        a numpy array containing the rgb values
    """
    return numpy.array([int(h, 16) for h in re.findall(r"[0-9a-fA-F]{2}", hex_colour)])
