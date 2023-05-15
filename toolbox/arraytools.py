"""Work with 1D arrays."""


import math

import numpy as np

import scipy.stats as sp_stats

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


class Regression:
    def __init__(
        self,
        x,
        y,
        linspace=100,
    ):
        """
        Model regression and its parameters.

        Instance will provide:
            x, y: array-like
                The original data
            p: np.ndarray
                Polynom parameters
            cov: float
            y_model: np.ndarray
                The regression modeled y values for the input x
            n: int
                Number of observations
            m: int
                Number of parameters
            dof: int
                Degree of freedoms
                n - m
            t: float
                t statistics



        """
        self.x = x
        self.y = y
        # parameters and covariance from of the fit of 1-D polynom.
        self.p, self.cov = np.polyfit(
            x,
            y,
            1,
            cov=True,
        )
        self.y_model = (
            # model using the fit parameters; NOTE: parameters here are
            np.polyval(
                self.p,
                x,
            )
        )

        self.n = y.size  # number of observations
        self.m = self.p.size  # number of parameters
        self.dof = self.n - self.m  # degrees of freedom
        self.t = sp_stats.t.ppf(  # t-statistic; used for CI and PI bands
            0.975,
            self.n - self.m,
        )

        # Estimates of Error in Data/Model
        self.resid = (
            y - self.y_model
        )  # residuals; diff. actual data from predicted values
        self.chi2 = np.sum(  # chi-squared; estimates error in data
            (self.resid / self.y_model) ** 2
        )
        self.chi2_red = (
            self.chi2 / self.dof
        )  # reduced chi-squared; measures goodness of fit
        self.s_err = np.sqrt(  # standard deviation of the error
            np.sum(self.resid**2) / self.dof
        )

        self.x2 = np.linspace(np.min(self.x), np.max(self.x), 100)
        self.y2 = np.polyval(self.p, self.x2)

        # confidence interval
        self.ci = (
            self.t
            * self.s_err
            * np.sqrt(
                1 / self.n
                + (self.x2 - np.mean(self.x)) ** 2
                / np.sum((self.x - np.mean(self.x)) ** 2)
            )
        )

        # prediction interval
        self.pi = (
            self.t
            * self.s_err
            * np.sqrt(
                1
                + 1 / self.n
                + (self.x2 - np.mean(self.x)) ** 2
                / np.sum((self.x - np.mean(self.x)) ** 2)
            )
        )
