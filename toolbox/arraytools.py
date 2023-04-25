"""
Work with 1D arrays.
"""


import math

import numpy as np


def lowpass(data, n = 101):
    """
    Average symetrically over n data points.

    Parameters
    ----------
    data: array-like
        List or array to filter.
    n: int, optional
        Number of data points to average over.

    Returns
    -------
    np.ndarray
    """
    
    return(np.array([np.mean(data[i : i + n]) for i in range(data.size - n + 1)]))


def highpass(data, n = 101):
    """Filter out low-frequency drift.
    Offsets each datapoint by the average of the surrounding n data points. 
    N must be odd.
    """
    
    if(n % 2 != 1):
        raise Exception("n must be odd!")
    return(np.array(data[int((n - 1) / 2) : -int((n - 1) / 2)] - lowpass(data, n)))


def interp(array, pos):
    """
    Linearly interpolate between neighboring indexes.
    
    Parameters
    ----------
    array: 1D list-like
    pos: float

    Returns
    -------
    float
        interpolated value
    """
    if(math.floor(pos)==math.ceil(pos)):
        return array[int(pos)]

    i = math.floor(pos)
    w = pos - i
    d = array[i+1] - array[i]
    return array[i] + w * d
