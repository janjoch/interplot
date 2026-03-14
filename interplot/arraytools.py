"""
A collection of useful functions to work with numpy arrays.
"""

import math
from warnings import warn

import numpy as np

import scipy.stats as sp_stats

try:
    from pandas.core.series import Series as pd_Series
    from pandas import Series

except ImportError:

    class pd_Series:
        pass

    class Series:
        pass


__warn_numba_import = False

try:
    from numba import jit, prange

    __warn_numba_import = False

except ImportError:
    __warn_numba_import = True

    def jit(**_):
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    prange = range

from . import plot

LISTLIKE_TYPES = (tuple, list, np.ndarray, pd_Series)


def _new_pd_index(series, n):
    return series.index[(n - 1) // 2 : -(n // 2)]


def lowpass(data, n=101, new_index=None):
    """
    Moving average symetrically over n data points.

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

    global __warn_numba_import
    if __warn_numba_import:
        warn(
            "Import numba to speed up the lowpass filtering: `pip install numba`"
        )
        __warn_numba_import = False

    # pandas Series
    if isinstance(data, pd_Series):
        new_index = _new_pd_index(data, n) if new_index is None else new_index
        return Series(
            lowpass_core(np.array(data), n),
            index=new_index,
        )

    # np.array, list or tuple
    if isinstance(data, LISTLIKE_TYPES):
        return lowpass_core(np.array(data), n)

    # fail if no supported data type
    raise TypeError("Data type not supported:\n{}".format(type(data)))


@jit(nopython=True, parallel=True)
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
    for i in prange(size):
        array[i] = np.mean(data[i : i + n])

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
    if isinstance(data, pd_Series):
        new_index = _new_pd_index(data, n) if new_index is None else new_index
        return data[((n - 1) // 2) : -((n - 1) // 2)] - lowpass(
            np.array(data), n, new_index=new_index
        )

    # np.array, list or tuple
    if isinstance(data, LISTLIKE_TYPES):
        return np.array(data[((n - 1) // 2) : -((n - 1) // 2)]) - lowpass_core(
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


def _stepsize(N, max_length):
    return int(math.ceil(N / max_length))


def downsample(max_length, *x, mode="step", axis=0):
    """
    Reduce size of array to `max_length` or lower.

    If `mode="step"`, each `stepsize`-th element is returned.

    If `mode="average"`, the array is averaged in bins of size `stepsize`.

    If multiple x are provided, they are are assumed to have the same shape.
    `stepsize` is determined from the first element.

    Parameters
    ----------
    max_length: int
    *x: np.ndarray
        The array(s) to be downsampled. If multiple arrays are provided,
        the result will be a list of the downsampled arrays.
    mode: str, default: step
        Mode for downsampling to reduce file size.

        Options:

        - step:
            Based on `max_length` and the array size, the smallest
            `stepsize` is determined, such that the new length is
            smaller or equal to `max_length`. Each `stepsize`-th element
            is displayed.
        - average:
            Bins of length `stepsize` are averaged. The remainders
            are discarded.
    axis: int, default: 0
        The axis along which to perform the downsampling.
    """
    if mode == "step":
        return downsample_step(max_length, *x, axis=axis)

    elif mode == "average":
        return downsample_average(max_length, *x, axis=axis)

    else:
        raise NotImplementedError(
            f"{mode=} is not implemented for downsampling."
        )


def downsample_step(max_length, *x, axis=0):
    """
    Reduce size of array to `max_length` or lower by returning each
    `stepsize`-th element.

    If multiple x are provided, they are are assumed to have the same shape.
    `stepsize` is determined from the first element.

    Parameters
    ----------
    max_length: int
    *x: np.ndarray
        The array(s) to be downsampled. If multiple arrays are provided,
        the result will be a list of the downsampled arrays.
    axis: int, default: 0
        The axis along which to perform the downsampling.
    """
    N = x[0].shape[axis]

    if N < max_length:
        if len(x) == 1:
            return x[0]
        return list(x)

    s = [slice(None)] * x[0].ndim
    step = _stepsize(N, max_length)
    s[axis] = slice(None, None, step)
    s = tuple(s)

    if len(x) == 1:
        return x[0][s]

    return [y[s] for y in x]


def _downsample_average_item(x, transp, step, length):
    """
    Performs the binning and averaging for one `np.ndarray`.
    """
    x = x.transpose(transp)[: length * step, ...]
    shape = x.shape
    x = x.reshape((length, step, *shape[1:])).mean(axis=1)
    return x.transpose(transp)


def downsample_average(max_length, *x, axis=0):
    """
    Reduce size of array to `max_length` or lower by returning each
    `stepsize`-th element.

    If multiple x are provided, they are are assumed to have the same shape.
    `stepsize` is determined from the first element.

    Parameters
    ----------
    max_length: int
    *x: np.ndarray
        The array(s) to be downsampled. If multiple arrays are provided,
        the result will be a list of the downsampled arrays.
    axis: int, default: 0
        The axis along which to perform the downsampling.
    """
    N = x[0].shape[axis]

    if N < max_length:
        if len(x) == 1:
            return x[0]
        return list(x)

    step = _stepsize(N, max_length)
    length = N // step

    transp = np.arange(x[0].ndim)
    transp[axis] = 0
    transp[0] = axis

    if len(x) == 1:
        return _downsample_average_item(x[0], transp, step, length)

    return [_downsample_average_item(y, transp, step, length) for y in x]


class LinearRegression(plot.NotebookInteraction):
    """
    Model regression and its parameters.

    Parameters
    ----------
    x, y: array-like
        Data points.
    p: float, default: 0.05
        p-value.
    linspace: int, default: 101
        Number of data points for linear regression model
        and conficence and prediction intervals.

    Attributes
    ----------
    The instance will provide the following data attributes:
    x, y: array-like
        The original data.
    p: float
        The original p-value.
    poly: np.ndarray of 2x float
        Polynomial coefficients.
        [a, b] -> a * x + b.
    cov: float
        Covariance matrix of the polynomial coefficient estimates.
        See for poly, cov:
        https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
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
    ...
    """

    def __init__(
        self,
        x,
        y,
        p=0.05,
        linspace=101,
    ):
        self.x = np.array(x)
        self.y = np.array(y)
        self.p = p
        self.is_linreg = True

        # parameters and covariance from of the fit of 1-D polynom.
        self.poly, self.cov = np.polyfit(
            x,
            y,
            1,
            cov=True,
        )
        self.y_model = (
            # model using the fit parameters; NOTE: parameters here are
            np.polyval(
                self.poly,
                x,
            )
        )

        self.n = y.size  # number of observations
        self.m = self.poly.size  # number of parameters
        self.dof = self.n - self.m  # degrees of freedom
        self.t = sp_stats.t.ppf(  # t-statistic; used for CI and PI bands
            1 - p / 2,
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

        self.x2 = np.linspace(np.min(self.x), np.max(self.x), linspace)
        self.y2 = np.polyval(self.poly, self.x2)

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

    @plot.magic_plot
    def plot(
        self,
        fig=None,  # inserted by plot.magic_plot decorator
        plot_ci=True,
        plot_pi=True,
        label=None,
        label_data="data",
        label_reg="regression",
        label_ci="confidence interval",
        label_pi="prediction interval",
        line_style_reg="solid",
        line_style_pi="dotted",
        color=None,
        color_data=None,
        color_reg=None,
        color_ci=None,
        color_pi=None,
        kwargs_data=None,
        kwargs_reg=None,
        kwargs_ci=None,
        kwargs_pi=None,
        **kwargs,
    ):
        """
        Plot the correlation analysis.

        Parameters
        ----------
        plot_ci, plot_pi: bool, optional
            Plot the confidence and prediction intervals.
            Default: True
        label: str or interplot.Labelgroup, optional
        label_data, label_reg, label_ci, label_pi: str or callable, optional
            Trace labels.
        color_data, color_reg, color_ci, color_pi: str, optional
            Trace color.
            Can be hex, rgb(a) or any named color that is understood
            by matplotlib.
            Default: None
            In the default case, Plot will cycle through COLOR_CYCLE.
        kwargs_data, kwargs_reg, kwargs_ci, kwargs_pi: dict, optional
            Keyword arguments to pass to corresponding figure element.
        **kwargs: optional
            Keyword arguments to pass to each figure element.
        """
        # input validation
        if kwargs_data is None:
            kwargs_data = dict()
        if kwargs_reg is None:
            kwargs_reg = dict()
        if kwargs_ci is None:
            kwargs_ci = dict()
        if kwargs_pi is None:
            kwargs_pi = dict()
        if color is None:
            color = fig.get_cycle_color()

        if not isinstance(label, plot.LabelGroup):
            row = kwargs.get("row", 0)
            col = kwargs.get("col", 0)
            group_id = "regression_{}_{}_{}".format(
                row,
                col,
                fig.element_count[row, col],
            )
            label = plot.LabelGroup(
                group_id=group_id,
                group_title="Regression" if label is None else label,
            )

        # data points
        fig.add_scatter(
            self.x,
            self.y,
            label=(
                label_data
                if callable(label_data)
                else label.element(label_data)
            ),
            color=color if color_data is None else color_data,
            **kwargs_data,
            **kwargs,
        )

        # regression line
        fig.add_line(
            self.x2,
            self.y2,
            line_style=line_style_reg,
            label=(
                label_reg if callable(label_reg) else label.element(label_reg)
            ),
            color=color if color_reg is None else color_reg,
            **kwargs_reg,
            **kwargs,
        )

        if plot_ci:
            fig.add_fill(
                self.x2,
                self.y2 - self.ci,
                self.y2 + self.ci,
                label=(
                    label_ci
                    if callable(label_ci)
                    else plot.LabelGroup(
                        group_id=label.group_id,
                        default_label=label_ci,
                    )
                ),
                color=color if color_ci is None else color_ci,
                **kwargs_ci,
                **kwargs,
            )

        if plot_pi:
            fig.add_line(
                self.x2,
                self.y2 + self.pi,
                label=(
                    label_pi if callable(label_pi) else label.element(label_pi)
                ),
                line_style=line_style_pi,
                color=color if color_pi is None else color_pi,
                **kwargs_pi,
                **kwargs,
            )
            fig.add_line(
                self.x2,
                self.y2 - self.pi,
                label=(
                    label_pi
                    if callable(label_pi)
                    else label.element(label_pi, show=False)
                ),
                line_style=line_style_pi,
                show_legend=False,
                color=color if color_pi is None else color_pi,
                **kwargs_pi,
                **kwargs,
            )
