"""Work with 1D arrays."""


import math

import numpy as np

import pandas as pd

import numba as nb

LISTLIKE_TYPES = (tuple, list, np.ndarray, pd.core.series.Series)


def _new_pd_index(series, n):
    return series.index[(n - 1) // 2: -(n // 2)]


def lowpass(data, n=101, new_index=None):
    """
    Average symetrically over n data points.

    Accepts numpy arrays, lists and pandas Series.

    Parameters
    ----------
    data: array-like
        np.array, list, tuple or pd.Series to filter.
    n: int, optional
        Number of data points to average over.
    new_index: list-like, optional
        If a pandas Series is provided as data,
        use this new_index.

    Returns
    -------
    np.ndarray or pd.Series
    """
    # input verification
    if n == 1:
        return data

    # pandas Series
    if isinstance(data, pd.core.series.Series):
        new_index = _new_pd_index(data, n) if new_index is None else new_index
        return pd.Series(
            lowpass_core(np.array(data), n),
            index=new_index,
        )

    # np.array, list or tuple
    if isinstance(data, LISTLIKE_TYPES):
        return lowpass_core(np.array(data), n)

    # fail if no supported data type
    raise TypeError("Data type not supported:\n{}".format(type(data)))


@nb.jit(nopython=True, parallel=True)
def lowpass_core(data, n):
    """
    Average symetrically over n data points.

    Parameters
    ----------
    data: np.array
        Array to filter.
    n: int, optional
        Number of data points to average over.

    Returns
    -------
    np.ndarray
    """
    size = data.size - n + 1

    array = np.empty(size, dtype=data.dtype)
    for i in nb.prange(size):
        array[i] = np.mean(data[i: i + n])

    return array


def highpass(
    data,
    n=101,
    new_index=None,
):
    """
    Filter out low-frequency drift.

    Offsets each datapoint by the average of the surrounding n data points.
    N must be odd.

    Parameters
    ----------
    data: array-like
        np.array, list, tuple or pd.Series to filter.
    n: int, optional
        Number of data points to average over.
    new_index: list-like, optional
        If a pandas Series is provided as data,
        use this new_index.

    Returns
    -------
    np.ndarray or pd.Series
    """
    # input verification
    if n == 1:
        return data
    if n % 2 == 0:
        raise ValueError("n must be odd!")

    # pandas Series
    if isinstance(data, pd.core.series.Series):
        new_index = _new_pd_index(data, n) if new_index is None else new_index
        return data[((n - 1) // 2): -((n - 1) // 2)] - lowpass(
            np.array(data), n, new_index=new_index
        )

    # np.array, list or tuple
    if isinstance(data, LISTLIKE_TYPES):
        return np.array(data[((n - 1) // 2): -((n - 1) // 2)]) - lowpass_core(
            np.array(data), n
        )

    # fail if no supported data type
    raise TypeError("Data type not supported:\n{}".format(type(data)))


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
    if math.floor(pos) == math.ceil(pos):
        return array[int(pos)]

    i = math.floor(pos)
    w = pos - i
    d = array[i + 1] - array[i]
    return array[i] + w * d
