"""Work with 1D arrays."""


import math

import numpy as np

import scipy.stats as sp_stats

import pandas as pd

import numba as nb

from . import plot


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


class LinearRegression(plot.NotebookInteraction):
    def __init__(
        self,
        x,
        y,
        p=0.05,
        linspace=101,
    ):
        """
        Model regression and its parameters.

        Parameters
        ----------
        x, y: array-like
            Data points.
        p: float, optional
            p-value.
            Default: 0.05
        linspace: int, optional
            Number of data points for linear regression model
            and conficence and prediction intervals.
            Default: 101

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

        Code derived from pylang's StackOverflow post:
        https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
        """
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
        label_data="data",
        label_reg="regression",
        label_ci="confidence interval",
        label_pi="prediction interval",
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
        label_data, label_reg, label_ci, label_pi: str
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

        Returns
        -------
        plot.Plot instance
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

        # data as dots
        if fig.interactive:
            kwargs_data.update(dict(mode="markers"))
        else:
            kwargs_data.update(dict(linestyle="", marker="o"))

        fig.add_line(
            self.x,
            self.y,
            label=label_data,
            color=color if color_data is None else color_data,
            **kwargs_data,
            **kwargs,
        )

        fig.add_line(
            self.x2,
            self.y2,
            label=label_reg,
            color=color if color_reg is None else color_reg,
            **kwargs_reg,
            **kwargs,
        )

        if plot_ci:
            fig.add_fill(
                self.x2,
                self.y2 - self.ci,
                self.y2 + self.ci,
                label=label_ci,
                color=color if color_ci is None else color_ci,
                **kwargs_ci,
                **kwargs,
            )

        if plot_pi:
            if fig.interactive:
                legendgroup = "pi_" + str(len(fig.fig.data))
            else:
                legendgroup = None
            fig.add_line(
                self.x2,
                self.y2 + self.pi,
                label=label_pi,
                color=color if color_pi is None else color_pi,
                kwargs_pty=dict(
                    legendgroup=legendgroup,
                ),
                **kwargs_pi,
                **kwargs,
            )
            fig.add_line(
                self.x2,
                self.y2 - self.pi,
                label=label_pi,
                show_legend=False,
                color=color if color_pi is None else color_pi,
                kwargs_pty=dict(
                    legendgroup=legendgroup,
                ),
                **kwargs_pi,
                **kwargs,
            )
