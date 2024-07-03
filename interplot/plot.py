"""
Create `matplotlib/plotly` hybrid plots with a few lines of code.

It combines the best of the `matplotlib` and the `plotly` worlds through
a unified, flat API.
All the necessary boilerplate code is contained in this module.

Currently supported:

- line plots (scatter)
- line fills
- histograms
- heatmaps
- boxplot
- linear regression
- text annotations
- 2D subplots
- color cycling
"""


import re
from warnings import warn
from pathlib import Path
from functools import wraps
from datetime import datetime
from io import BytesIO

import numpy as np

from pandas.core.series import Series as pd_Series
from pandas.core.frame import DataFrame as pd_DataFrame

from xarray.core.dataarray import DataArray as xr_DataArray

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import plotly.offline

from . import conf
from .iter import ITERABLE_TYPES, zip_smart, filter_nozip
from interplot import arraytools


def init_notebook_mode(connected=False):
    """
    Initialize plotly.js in the browser if not already done,
    and deactivate matplotlib auto-display.

    Parameters
    ----------
    connected: bool, optional
        If True, the plotly.js library will be loaded from an online CDN.
        If False, the plotly.js library will be loaded locally.
        Default: False
    """
    plotly.offline.init_notebook_mode(connected=connected)

    # turn off matplotlib auto-display
    plt.plot()
    plt.close()
    plt.ioff()


# if imported in notebook, init plotly notebook mode
try:
    __IPYTHON__  # type: ignore
    from IPython.core.display import display_html, display_png
    CALLED_FROM_NOTEBOOK = True
except NameError:
    CALLED_FROM_NOTEBOOK = False
if CALLED_FROM_NOTEBOOK:
    init_notebook_mode()


def _rewrite_docstring(doc_core, doc_decorator=None, kwargs_remove=()):
    """
    Appends arguments to a docstring.

    Returns original docstring if conf._REWRITE_DOCSTRING is set to False.

    Attempts:
    1. Search for [decorator.*?].
    2. Search for numpy-style "Parameters" block.
    3. Append to the end.

    Parameters
    ----------
    doc_core: str
        Original docstring.
    doc_decorator: str,
        docstring to insert.
    kwargs_remove: tuple of strs, optional
        remove parameters in the docstring.

    Returns
    -------
    str:
        Rewritten docstring
    """
    # check rewrite flag
    if not conf._REWRITE_DOCSTRING:
        return doc_core

    # input check
    doc_core = "" if doc_core is None else doc_core
    doc_decorator = (
        conf._DOCSTRING_DECORATOR
        if doc_decorator is None
        else doc_decorator
    )

    # find indentation level of doc_core
    match = re.match("^\n?(?P<indent_core>[ \t]*)", doc_core)
    indent_core = match.group("indent_core") if match else ""

    # find indentation level of doc_decorator
    match = re.match("^\n?(?P<indent_core>[ \t]*)", doc_decorator)
    indent_decorator = match.group("indent_core") if match else ""

    # remove kwargs from doc_decorator
    for kwarg_key in kwargs_remove:
        # remove docstring entry if it is the only argument
        doc_decorator = re.sub(
            (
                r"\n{0}"  # indent
                r"{1}[ ]*:"  # kwarg_key followed by colon
                r".*(?:\n{0}[ \t]+.*)*"  # the following further indented lines
            ).format(
                indent_decorator,
                kwarg_key,
            ),
            r"",
            doc_decorator,
        )

        # remove docstring key if it is found in a list
        doc_decorator = re.sub(
            (
                (  # preceding kwarg_key
                    r"(?P<front>"  # named group
                    r"\n{0}"  # indentation
                    r"(?:[a-zA-Z_]+)??"  # first arg
                    r"(?:[ ]*,[ ]*[a-zA-Z_]+)??"  # following args
                    r")"  # end named group
                ) +
                (  # kwarg_key
                    r"(?P<leading_coma>[ ]*,[ ]*)?"  # leading coma
                    r"{1}"  # kwarg_key
                    r"(?(leading_coma)|(?:[ ]*,[ ]*)?)"  # following coma if no leading coma  # noqa: E501
                ) +
                r"(?P<back>(?:[ ]*,[ ]*[a-zA-Z_]+)*[ ]*?)"  # following arguments  # noqa: E501
            ).format(indent_decorator, kwarg_key),
            r"\g<front>\g<back>",
            doc_decorator,
        )

    # search "[decorator parameters]"
    match = re.search(r"\n[ \t]*\[decorator.*?]", doc_core)
    if match:
        return re.sub(
            r"\n[ \t]*\[decorator.*?]",
            _adjust_indent(indent_decorator, indent_core, doc_decorator),
            doc_core,
        )

    # test for numpy-style doc_core
    docstring_query = (
        r"(?P<desc>(?:.*\n)*?)"  # desc
        r"(?P<params>(?P<indent_core>[ \t]*)Parameters[ \t]*"  # params header
        r"(?:\n(?!(?:[ \t]*\n)|(?:[ \t]*$)).*)*)"  # non-whitespace lines
        r"(?P<rest>(?:.*\n)*.*$)"
    )
    docstring_query = (
        r"(?P<desc>(?:.*\n)*?)"  # desc
        r"(?P<params>(?P<indent_core>[ \t]*)Parameters[ \t]*"  # params header
        r"(?:.*\n)*?)"  # params content
        r"(?P<rest>[ \t]*\n[ \t]*[A-Za-z0-9 \t]*[ \t]*\n[ \t]*---(?:.*\n)*.*)$"
        # next chapter
    )
    match = re.match(docstring_query, doc_core)
    if match:
        doc_parts = match.groupdict()
        return (
            doc_parts["desc"]
            + doc_parts["params"]
            + _adjust_indent(
                indent_decorator,
                doc_parts["indent_core"],
                doc_decorator,
            )
            + doc_parts["rest"]
        )

    # non-numpy _DOCSTRING_DECORATOR, just append in the end
    return doc_core + _adjust_indent(
        indent_decorator,
        indent_core,
        doc_decorator,
    )


def _plt_cmap_extremes(cmap, under=None, over=None, bad=None):
    """
    Get cmap with under and over range colors.

    Parameters
    ----------
    cmap: str or plt cmap
        Colormap to use.
        https://matplotlib.org/stable/gallery/color/colormap_reference.html
    under, over, bad: color, optional
        Color to use for under / over range values and bad values.

    Returns
    -------
    cmap: plt cmap
        Provide cmap to plt: plt.imshow(cmap=cmap)
    """
    cmap = plt.get_cmap(cmap).copy()
    if under:
        cmap.set_under(under)
    if over:
        cmap.set_over(over)
    if bad:
        cmap.set_bad(bad)
    return cmap


def _plotly_colormap_extremes(cs, under=None, over=None):
    """
    Append under and over range colors to plotly figure.

    Parameters
    ----------
    fig: plotly.Figure
        Plotly Figure instance which contains a colormap.
    under, over: color, optional
        Color to use for under / over range values.

    Returns
    -------
    fig: plotly.Figure
    """
    cs = [[b for b in a] for a in cs]  # convert tuple to list
    if under:
        cs[0][0] = 0.0000001
        cs = [[0, under]] + cs
    if over:
        cs[-1][0] = 0.9999999
        cs = cs + [[1.0, over]]
    return cs


def _adjust_indent(indent_decorator, indent_core, docstring):
    """Adjust indentation of docstsrings."""
    return re.sub(
        r"\n{}".format(indent_decorator),
        r"\n{}".format(indent_core),
        docstring,
    )


def _serialize_2d(serialize_pty=True, serialize_mpl=True):
    """Decorator to catch 2D arrays and other data types to unpack."""

    def decorator(core):

        @wraps(core)
        def wrapper(self, x, y=None, label=None, **kwargs):
            """
            Wrapper function for a method.

            If a pandas object is provided, the index will be used as x
            if no x is provided.
            Pandas column naming:
                * If no label is set, the column name will be used by default.
                * Manually set label string has priority.
                * Label strings may contain a {} to insert the column name.
                * Instead of setting a string, a callable may be provided to
                reformat the column name. It must accept the column name
                and return a string. E.g.:

                > interplot.line(df, label=lambda n: n.strip())

                > def capitalize(prefix="", suffix=""):
                >     return lambda name: prefix + name.upper() + suffix
                > interplot.line(df, label=capitalize("Cat. A: "))
            xarray DataArrays will be convered to pandas and then handled
            accordingly.
            """
            # reallocate x/y
            if y is None:

                # xarray DataArray
                if isinstance(x, xr_DataArray):
                    x = x.to_pandas()

                # pd.Series
                if isinstance(x, pd_Series):
                    x, y = x.index, x
                    if label is None:
                        label = y.name
                    elif isinstance(label, str) and "{}" in label:
                        label = label.format(y.name)
                    elif callable(label):
                        label = label(y.name)

                # pd.DataFrame: split columns to pd.Series and iterate
                elif isinstance(x, pd_DataFrame):
                    if (
                        self.interactive and serialize_pty
                        or not self.interactive and serialize_mpl
                    ):
                        for (
                            i, ((_, series), label_)
                        ) in enumerate(zip_smart(x.items(), label)):
                            _serialize_2d(
                                serialize_pty=serialize_pty,
                                serialize_mpl=serialize_mpl,
                            )(core)(
                                self,
                                series,
                                label=label_,
                                _serial_i=i,
                                _serial_n=len(x.columns),
                                **kwargs,
                            )
                        return

                else:
                    if hasattr(x, 'copy') and callable(getattr(x, 'copy')):
                        y = x.copy()
                    else:
                        y = x
                    x = np.arange(len(y))

            # 2D np.array
            if isinstance(y, np.ndarray) and len(y.shape) == 2:
                if (
                    self.interactive and serialize_pty
                    or not self.interactive and serialize_mpl
                ):
                    for (
                        i, (y_, label_)
                    ) in enumerate(zip_smart(y.T, label)):
                        _serialize_2d(
                            serialize_pty=serialize_pty,
                            serialize_mpl=serialize_mpl,
                        )(core)(
                            self,
                            x,
                            y_,
                            label=label_,
                            _serial_i=i,
                            _serial_n=y.shape[1],
                            **kwargs,
                        )
                    return

            return core(self, x, y, label=label, **kwargs)

        return wrapper

    return decorator


def _serialize_save(core):
    """Decorator to serialise saving multiple figures."""

    @wraps(core)
    def wrapper(self, path, export_format=None, **kwargs):
        """
        Wrapper function for a method.

        """
        if (
            isinstance(path, ITERABLE_TYPES)
            or isinstance(export_format, ITERABLE_TYPES)
        ):
            for path_, export_format_ in zip_smart(path, export_format):
                self.save(path_, export_format_, **kwargs)

            return

        return core(self, path, export_format=export_format, **kwargs)

    return wrapper


class NotebookInteraction:
    """
    Parent class for automatic display in Jupyter Notebook.

    Calls the child's `show()._repr_html_()` for automatic display
    in Jupyter Notebooks.
    """
    JS_RENDER_WARNING = '''
        <div class="alert alert-block alert-warning"
            id="notebook-js-warning">
            <p>
                Unable to render javascript-based plotly plot.<br>
                Call interplot.init_notebook_mode() or re-run this cell.<br>
                If viewing on GitHub, render the notebook in
                <a href="https://nbviewer.org/" target="_blank">
                    NBViewer</a> instead.
            </p>
        </div>
        <script type="text/javascript">
            function hide_warning() {
                var element = document.getElementById(
                    "notebook-js-warning"
                );
                element.parentNode.removeChild(element);
            }
            hide_warning();
        </script>
    ''' if CALLED_FROM_NOTEBOOK else ""

    def __call__(self, *args, **kwargs):
        """Calls the `self.show()` or `self.plot()` method."""
        # look for show() method
        try:
            return self.show(*args, **kwargs)

        # fall back to plot() method
        except AttributeError:
            return self.plot(*args, **kwargs)

    def _repr_html_(self):
        """Call forward to `self.show()._repr_html_()`."""
        init_notebook_mode()

        # look for show() method
        try:
            return self.JS_RENDER_WARNING + self.show()._repr_html_()

        # fall back to plot() method
        except AttributeError:
            try:
                return self.JS_RENDER_WARNING + self.plot()._repr_html_()

            # not implemented
            except AttributeError:
                raise NotImplementedError

    def _repr_mimebundle_(self, *args, **kwargs):
        """Call forward to `self.show()_repr_mimebundle_()`"""
        # look for show() method
        try:
            return self.show()._repr_mimebundle_(*args, **kwargs)

        # fall back to plot() method
        except AttributeError:
            try:
                return self.plot()._repr_mimebundle_(*args, **kwargs)

            # not implemented
            except AttributeError:
                raise NotImplementedError


class Plot(NotebookInteraction):
    """
    Create `matplotlib/plotly` hybrid plots with a few lines of code.

    It combines the best of the `matplotlib` and the `plotly` worlds through
    a unified, flat API.
    All the necessary boilerplate code is contained in this module.

    Currently supported:

    - line plots (scatter)
    - line fills
    - histograms
    - heatmaps
    - boxplot
    - linear regression
    - text annotations
    - 2D subplots
    - color cycling

    Parameters
    ----------

    Examples
    --------
    >>> fig = interplot.Plot(
    ...     interactive=True,
    ...     title="Everything Under Control",
    ...     fig_size=(800, 500),
    ...     rows=1,
    ...     cols=2,
    ...     shared_yaxes=True,
    ...     save_fig=True,
    ...     save_format=("html", "png"),
    ...     # ...
    ... )
    ... fig.add_hist(np.random.normal(1, 0.5, 1000), row=0, col=0)
    ... fig.add_boxplot(
    ...     [
    ...         np.random.normal(20, 5, 1000),
    ...         np.random.normal(40, 8, 1000),
    ...         np.random.normal(60, 5, 1000),
    ...     ],
    ...     row=0,
    ...     col=1,
    ... )
    ... # ...
    ... fig.post_process()
    ... fig.show()
    saved figure at Everything-Under-Control.html
    saved figure at Everything-Under-Control.png

    .. raw:: html
        :file: ../source/plot_examples/Everything-Under-Control.html
    """
    __doc__ = _rewrite_docstring(__doc__)

    def __init__(
        self,
        interactive=None,
        rows=1,
        cols=1,
        title=None,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        shared_xaxes=False,
        shared_yaxes=False,
        column_widths=None,
        row_heights=None,
        fig_size=None,
        dpi=None,
        legend_loc=None,
        legend_title=None,
        save_fig=None,
        save_format=None,
        save_config=None,
        pty_update_layout=None,
        pty_custom_func=None,
        mpl_custom_func=None,
    ):
        # input verification
        if shared_xaxes == "cols":
            shared_xaxes = "columns"
        if shared_yaxes == "cols":
            shared_yaxes = "columns"

        self.interactive = (
            conf.INTERACTIVE
            if interactive is None
            else interactive
        )
        self.rows = rows
        self.cols = cols
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.legend_loc = legend_loc
        self.legend_title = legend_title
        self.dpi = conf.DPI if dpi is None else dpi
        self.save_fig = save_fig
        self.save_format = save_format
        self.save_config = save_config
        self.pty_update_layout = pty_update_layout
        self.pty_custom_func = pty_custom_func
        self.mpl_custom_func = mpl_custom_func
        self.element_count = np.zeros((rows, cols), dtype=int)
        self.i_color = 0

        # init plotly
        if self.interactive:
            self.title = self._encode_html(self.title)
            self.fig_size = (
                conf.PTY_FIG_SIZE
                if fig_size is None
                else fig_size
            )

            # init fig
            figure = go.Figure(
                layout=go.Layout(legend={'traceorder': 'normal'}),
            )
            self.fig = sp.make_subplots(
                rows=rows,
                cols=cols,
                shared_xaxes=shared_xaxes,
                shared_yaxes=shared_yaxes,
                row_heights=row_heights,
                column_widths=column_widths,
                figure=figure,
            )

            # unpacking
            width, height = self.fig_size
            if isinstance(legend_title, ITERABLE_TYPES):
                warn(
                    "Plotly only has one legend, however multiple legend_"
                    "titles were provided. Only the first one will be used!"
                )
                legend_title = legend_title[0]
                if isinstance(legend_title, ITERABLE_TYPES):
                    legend_title = legend_title[0]

            # update layout
            self.fig.update_layout(
                title=self.title,
                legend_title=legend_title,
                height=height,
                width=width,
                barmode="group",
            )

            # axis limits
            for i_row, xlim_row, ylim_row in zip_smart(
                range(1, self.rows + 1),
                filter_nozip(self.xlim),
                filter_nozip(self.ylim),
            ):
                for i_col, xlim_tile, ylim_tile in zip_smart(
                    range(1, self.cols + 1),
                    filter_nozip(xlim_row),
                    filter_nozip(ylim_row),
                ):
                    if (
                        xlim_tile is not None
                        and isinstance(xlim_tile[0], datetime)
                    ):
                        xlim_tile = (
                            xlim_tile[0].timestamp()*1000,
                            xlim_tile[1].timestamp()*1000,
                        )
                    self.fig.update_xaxes(
                        range=xlim_tile,
                        row=i_row,
                        col=i_col,
                    )
                    self.fig.update_yaxes(
                        range=ylim_tile,
                        row=i_row,
                        col=i_col,
                    )

            # axis labels
            for text, i_col in zip_smart(xlabel, range(1, cols+1)):
                self.fig.update_xaxes(title_text=text, row=rows, col=i_col)
            for text, i_row in zip_smart(ylabel, range(1, rows+1)):
                self.fig.update_yaxes(title_text=text, row=i_row, col=1)

        # init matplotlib
        else:
            gridspec_kw = dict(
                width_ratios=column_widths,
                height_ratios=row_heights,
            )

            # convert px to inches
            self.fig_size = (
                conf.MPL_FIG_SIZE
                if fig_size is None
                else fig_size
            )
            px = 1 / self.dpi
            figsize = (self.fig_size[0] * px, self.fig_size[1] * px)

            # init fig
            self.fig, self.ax = plt.subplots(
                rows,
                cols,
                figsize=figsize,
                dpi=self.dpi,
                squeeze=False,
                gridspec_kw=gridspec_kw,
            )

            # title
            if self.cols == 1:
                self.ax[0, 0].set_title(self.title)
            else:
                self.fig.suptitle(self.title)

            # shared axes
            for i_row in range(self.rows):
                for i_col in range(self.cols):

                    # skip 0/0
                    if i_col == 0 and i_row == 0:
                        continue

                    # set shared x axes
                    if (
                        shared_xaxes == "all"
                        or type(shared_xaxes) is bool and shared_xaxes is True
                    ):
                        self.ax[i_row, i_col].sharex(self.ax[0, 0])
                    elif shared_xaxes == "columns":
                        self.ax[i_row, i_col].sharex(self.ax[0, i_col])
                    elif shared_xaxes == "rows":
                        self.ax[i_row, i_col].sharex(self.ax[i_row, 0])

                    # set shared y axes
                    if (
                        shared_yaxes == "all"
                        or type(shared_yaxes) is bool and shared_yaxes is True
                    ):
                        self.ax[i_row, i_col].sharey(self.ax[0, 0])
                    elif shared_yaxes == "columns":
                        self.ax[i_row, i_col].sharey(self.ax[0, i_col])
                    elif shared_yaxes == "rows":
                        self.ax[i_row, i_col].sharey(self.ax[i_row, 0])

            # axis labels
            if isinstance(self.xlabel, ITERABLE_TYPES) or (
                self.cols == 1 and self.rows == 1
            ):
                for text, i_col in zip_smart(self.xlabel, range(self.cols)):
                    self.ax[self.rows - 1, i_col].set_xlabel(text)
            else:
                self.fig.supxlabel(self.xlabel)
            if isinstance(self.ylabel, ITERABLE_TYPES) or (
                self.cols == 1 and self.rows == 1
            ):
                for text, i_row in zip_smart(self.ylabel, range(self.rows)):
                    self.ax[i_row, 0].set_ylabel(text)
            else:
                self.fig.supylabel(self.ylabel)

    @staticmethod
    def init(fig, *args, **kwargs):
        """
        Initialize a Plot instance, if not already initialized.

        Parameters
        ----------
        fig: Plot or any
            If fig is a Plot instance, return it.
            Otherwise, create a new Plot instance.
        *args, **kwargs: any
            Passed to Plot.__init__.
        """
        if isinstance(fig, Plot):
            return fig
        return Plot(*args, **kwargs)

    @staticmethod
    def _get_plotly_legend_args(label, default_label=None, show_legend=None):
        """
        Return keyword arguments for label configuration.

        Parameters
        ----------
        label: str
            Name to display.
        default_label: str, optional
            If label is None, fall back to default_label.
            Default: None
            By default, plotly will enumerate the unnamed traces itself.
        show_legend: bool, optional
            Show label in legend.
            Default: None
            By default, the label will be displayed if it is not None
            (in case of label=None, the automatic label will only be displayed
            on hover)
        """
        legend_kwargs = dict(
            name=default_label if label is None else str(label)
        )
        if show_legend:
            legend_kwargs["showlegend"] = True
        elif isinstance(show_legend, bool) and not show_legend:
            legend_kwargs["showlegend"] = False
        else:
            legend_kwargs["showlegend"] = False if label is None else True

        return legend_kwargs

    @staticmethod
    def _get_plotly_anchor(axis, cols, row, col):
        """
        Get axis id based on row and col.

        Parameters
        ----------
        axis: str
            x or y.
        cols: int
            Number of columns.
            Usually self.cols
        row, col: int
            Row and col index in plotly manner:
            STARTING WITH 1.

        Returns
        -------
        axis id: str
        """
        id = (row - 1) * cols + col
        if id == 1:
            return axis
        return axis + str((row - 1) * cols + col)

    @staticmethod
    def _encode_html(text):
        if text is None:
            return None
        return re.sub(r"\n", "<br>", text)

    def get_cycle_color(self, increment=1, i=None):
        """
        Retrieves the next color in the color cycle.

        Parameters
        ----------
        increment: int, optional
            If the same color should be returned the next time, pass 0.
            To jump the next color, pass 2.
            Default: 1
        i: int, optional
            Get a fixed index of the color cycle instead of the next one.
            This will not modify the regular color cycle iteration.

        Returns
        -------
        color: str
            HEX color, with leading hashtag
        """
        if i is None:
            if self.i_color >= len(conf.COLOR_CYCLE):
                self.i_color = 0
            color = conf.COLOR_CYCLE[self.i_color]
            self.i_color += increment
            return color
        else:
            return conf.COLOR_CYCLE[i]

    def digest_color(self, color=None, alpha=None, increment=1):
        """
        Parse color with matplotlib.colors to a rgba array.

        Parameters
        ----------
        color: any color format matplotlib accepts, optional
            E.g. "blue", "#0000ff", "C3"
            If None is provided, the next one from COLOR_CYCLE will be picked.
        alpha: float, optional
            Set alpha / opacity.
            Overrides alpha contained in color input.
            Default: None (use the value contained in color or default to 1)
        increment: int, optional
            If a color from the cycler is picked, increase the cycler by
            this increment.
        """
        # if color undefined, cycle COLOR_CYCLE
        if color is None:
            color = self.get_cycle_color(increment)

        # get index from COLOR_CYCLE
        elif color[0] == "C" or color[0] == "c":
            color = conf.COLOR_CYCLE[int(color[1:]) % len(conf.COLOR_CYCLE)]

        rgba = list(mcolors.to_rgba(color))
        if alpha is not None:
            rgba[3] = alpha

        # PLOTLY
        if self.interactive:
            return "rgba({},{},{},{:.4f})".format(
                *[int(d * 255) for d in rgba[:3]],
                rgba[3],
            )

        # MATPLOTLIB
        return tuple(rgba)

    @staticmethod
    def digest_marker(
        marker,
        mode,
        interactive,
        recursive=False,
        **pty_marker_kwargs,
    ):
        """
        Digests the marker parameter based on the given mode.

        Parameters
        ----------
        marker: int or str
            The marker to be digested. If an integer is provided, it is
            converted to the corresponding string marker using `plotly`
            numbering.
            If not provided, the default marker "circle" is used.
        mode: str
            The mode to determine if markers should be used.
            If no markers should be drawn, None is returned.

        Returns
        -------
        str or None
            The digested marker string if markers are requested,
            otherwise None.
        """
        if isinstance(marker, ITERABLE_TYPES):
            if interactive:
                return dict(
                    symbol=[
                        Plot.digest_marker(
                            marker=m,
                            mode=mode,
                            interactive=interactive,
                            recursive=True,
                        )
                        for m
                        in marker
                    ],
                    **pty_marker_kwargs,
                )
            return Plot.digest_marker(
                marker=marker[0],
                mode=mode,
                interactive=interactive,
                recursive=True,
            )

        if "markers" not in mode:
            return None

        if isinstance(marker, (int, np.integer)):
            marker = conf.PTY_MARKERS_LIST[marker]

        if marker is None:
            marker = conf.PTY_MARKERS_LIST[0]

        if interactive:
            if marker not in conf.PTY_MARKERS_LIST:
                marker = conf.PTY_MARKERS.get(marker, marker)
            if recursive:
                return marker
            return dict(
                symbol=marker,
                **pty_marker_kwargs,
            )

        return conf.MPL_MARKERS.get(marker, marker)

    @_serialize_2d()
    def add_line(
        self,
        x,
        y=None,
        x_error=None,
        y_error=None,
        mode=None,
        line_style="solid",
        marker=None,
        marker_size=None,
        marker_line_width=1,
        marker_line_color=None,
        label=None,
        show_legend=None,
        color=None,
        opacity=None,
        linewidth=None,
        row=0,
        col=0,
        _serial_i=0,
        _serial_n=1,
        pty_marker_kwargs=None,
        kwargs_pty=None,
        kwargs_mpl=None,
        **kwargs,
    ):
        """
        Draw a line or scatter plot.

        Parameters
        ----------
        x: array-like
        y: array-like, optional
            If only `x` is defined, it will be assumed as `y`.
            If a pandas `Series` is provided, the index will
            be taken as `x`.
            Else if a pandas `DataFrame` is provided, the method call
            is looped for each column.
            Else `x` will be an increment, starting from `0`.
            If a 2D numpy `array` is provided, the method call
            is looped for each column.
        x_error, y_error: number or shape(N,) or shape(2, N), optional
            The errorbar sizes (`matplotlib` style):
                - scalar: Symmetric +/- values for all data points.
                - shape(N,): Symmetric +/-values for each data point.
                - shape(2, N): Separate - and + values for each bar. First row
                    contains the lower errors, the second row contains the
                    upper errors.
                - None: No errorbar.
        mode: str, optional
            Options: `lines` / `lines+markers` / `markers`

            The default depends on the method called:
                - `add_line`: `lines`
                - `add_scatter`: `markers`
                - `add_linescatter`: `lines+markers`
        line_style: str, optional
            Line style.
            Options: `solid`, `dashed`, `dotted`, `dashdot`

            Aliases: `-`, `--`, `dash`, `:`, `dot`, `-.`
        marker: int or str, optional
            Marker style.
            If an integer is provided, it will be converted to the
            corresponding string marker using `plotly` numbering.
            If not provided, the default marker `circle` is used.
        marker_size: int, optional
        marker_line_width: int, optional
        marker_line_color: str, optional
            Can be hex, rgb(a) or any named color that is understood
            by matplotlib.

            Default: same color as `color`.
        label: str, optional
            Trace label for legend.
        show_legend: bool, optional
            Whether to show the label in the legend.

            By default, it will be shown if a label is defined.
        color: str, optional
            Trace color.

            Can be hex, rgb(a) or any named color that is understood
            by matplotlib.

            The color cycle can be accessed with "C0", "C1", ...

            Default: color is retrieved from `Plot.digest_color`,
            which cycles through `COLOR_CYCLE`.
        opacity: float, optional
            Opacity (=alpha) of the fill.

            By default, fallback to alpha value provided with color argument,
            or 1.
        row, col: int, optional
            If the plot contains a grid, provide the coordinates.

            Attention: Indexing starts with 0!
        pty_marker_kwargs: dict, optional
            PLOTLY ONLY.

            Additional marker arguments.
        kwargs_pty, kwargs_mpl, **kwargs: optional
            Pass specific keyword arguments to the line core method.

        Examples
        --------

        Using the interplot.Plot method:

        >>> fig = interplot.Plot(title="line, linescatter and scatter")
        ... fig.add_line([0,4,6,7], [1,2,4,8])
        ... fig.add_linescatter([0,4,6,7], [8,4,2,1])
        ... fig.add_scatter([0,4,6,7], [2,4,4,2], )
        ... fig.post_process()
        ... fig.show()

        [plotly figure, "line, linescatter and scatter"]

        Using interplot.line:

        >>> interplot.line([0,4,6,7], [1,2,4,8])

        .. raw:: html
            :file: ../source/plot_examples/basic_plot_pty.html

        >>> interplot.line(
        ...     x=[0,4,6,7],
        ...     y=[1,2,4,8],
        ...     interactive=False,
        ...     color="red",
        ...     title="matplotlib static figure",
        ...     xlabel="abscissa",
        ...     ylabel="ordinate",
        ... )

        .. image:: plot_examples/basic_plot_mpl.png
            :alt: [matplotlib plot "Normally distributed Noise]
        """
        self.element_count[row, col] += 1
        mode = "lines" if mode is None else mode
        color = self.digest_color(color, opacity)

        # PLOTLY
        if self.interactive:
            if kwargs_pty is None:
                kwargs_pty = dict()
            if pty_marker_kwargs is None:
                pty_marker_kwargs = dict()
            pty_marker_kwargs.update(
                dict(
                    size=marker_size,
                    line_width=marker_line_width,
                    line_color=(
                        color
                        if marker_line_color is None
                        else self.digest_color(marker_line_color, 1)
                    ),
                )
            )
            if x_error is not None:
                if not isinstance(x_error, ITERABLE_TYPES):
                    x_error = np.array((x_error, ) * len(x))
                if isinstance(x_error[0], ITERABLE_TYPES):
                    x_error = dict(
                        type="data",
                        array=x_error[1],
                        arrayminus=x_error[0],
                    )
                else:
                    x_error = dict(
                        type="data",
                        array=x_error,
                    )
            if y_error is not None:
                if not isinstance(y_error, ITERABLE_TYPES):
                    y_error = np.array((y_error, ) * len(y))
                if isinstance(y_error[0], ITERABLE_TYPES):
                    y_error = dict(
                        type="data",
                        array=y_error[1],
                        arrayminus=y_error[0],
                    )
                else:
                    y_error = dict(
                        type="data",
                        array=y_error,
                    )
            row += 1
            col += 1

            self.fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    error_x=x_error,
                    error_y=y_error,
                    mode=mode,
                    marker=self.digest_marker(
                        marker,
                        mode,
                        interactive=self.interactive,
                        **pty_marker_kwargs,
                    ),
                    **self._get_plotly_legend_args(
                        label,
                        show_legend=show_legend,
                    ),
                    marker_color=color,
                    line=dict(
                        width=linewidth,
                        dash=conf.PTY_LINE_STYLES.get(line_style, line_style),
                    ),
                    **kwargs_pty,
                    **kwargs,
                ),
                row=row,
                col=col,
            )

        # MATPLOTLIB
        else:
            if kwargs_mpl is None:
                kwargs_mpl = dict()
            self.ax[row, col].errorbar(
                x,
                y,
                xerr=x_error,
                yerr=y_error,
                label=None if show_legend is False else label,
                color=color,
                lw=linewidth,
                linestyle=(
                    conf.MPL_LINE_STYLES.get(line_style, line_style)
                    if "lines" in mode
                    else "None"
                ),
                marker=self.digest_marker(
                    marker,
                    mode,
                    interactive=self.interactive,
                ),
                markersize=marker_size,
                markeredgewidth=marker_line_width,
                markeredgecolor=(
                    color
                    if marker_line_color is None
                    else self.digest_color(marker_line_color)
                ),
                **kwargs_mpl,
                **kwargs,
            )

    @wraps(add_line)
    def add_scatter(
        self,
        *args,
        mode="markers",
        **kwargs,
    ):
        self.add_line(
            *args,
            mode=mode,
            **kwargs,
        )

    @wraps(add_line)
    def add_linescatter(
        self,
        *args,
        mode="markers+lines",
        **kwargs,
    ):
        self.add_line(
            *args,
            mode=mode,
            **kwargs,
        )

    @_serialize_2d()
    def add_bar(
        self,
        x,
        y=None,
        horizontal=False,
        width=0.8,
        label=None,
        show_legend=None,
        color=None,
        opacity=None,
        line_width=1,
        line_color=None,
        row=0,
        col=0,
        _serial_i=0,
        _serial_n=1,
        kwargs_pty=None,
        kwargs_mpl=None,
        **kwargs,
    ):
        """
        Draw a bar plot.

        Parameters
        ----------
        x: array-like
        y: array-like, optional
            If only either `x` or `y` is defined, it will be assumed
            as the size of the bar, regardless whether it's horizontal
            or vertical.
            If both `x` and `y` are defined, `x` will be taken as the
            position of the bar, and `y` as the size, regardless of
            the orientation.
            If a pandas `Series` is provided, the index will
            be taken as the position.
            Else if a pandas `DataFrame` is provided, the method call
            is looped for each column.
            If a 2D numpy `array` is provided, the method call
            is looped for each column, with the index as the position.
        horizontal: bool, optional
            If True, the bars are drawn horizontally. Default is False.
        width: float, optional
            Relative width of the bar. Must be in the range (0, 1).
        label: str, optional
            Trace label for legend.
        show_legend: bool, optional
            Whether to show the label in the legend.

            By default, it will be shown if a label is defined.
        color: str, optional
            Trace color.

            Can be hex, rgb(a) or any named color that is understood
            by matplotlib.

            The color cycle can be accessed with "C0", "C1", ...

            Default: color is retrieved from `Plot.digest_color`,
            which cycles through `COLOR_CYCLE`.
        opacity: float, optional
            Opacity (=alpha) of the fill.

            By default, fallback to alpha value provided with color argument,
            or 1.
        line_width: float, optional
            The width of the bar outline. Default is 1.
        line_color: str, optional
            The color of the bar outline. This can be a named color or a tuple
            specifying the RGB values.

            By default, the same color as the fill is used.
        row, col: int, optional
            If the plot contains a grid, provide the coordinates.

            Attention: Indexing starts with 0!
        kwargs_pty, kwargs_mpl, **kwargs: optional
            Pass specific keyword arguments to the line core method.
        """
        self.element_count[row, col] += 1
        # PLOTLY
        if self.interactive:
            if kwargs_pty is None:
                kwargs_pty = dict()

            if horizontal:
                x, y = y, x

            row += 1
            col += 1
            self.fig.add_trace(
                go.Bar(
                    x=x,
                    y=y,
                    orientation="h" if horizontal else "v",
                    **self._get_plotly_legend_args(
                        label,
                        show_legend=show_legend,
                    ),
                    marker_color=self.digest_color(color, opacity),
                    marker=dict(
                        line=dict(
                            width=0 if line_color is None else line_width,
                            color=(
                                color
                                if line_color is None
                                else self.digest_color(line_color, 1)
                            ),
                        ),
                    ),
                    **kwargs_pty,
                    **kwargs,
                ),
                row=row,
                col=col,
            )

        else:
            if kwargs_mpl is None:
                kwargs_mpl = dict()
            offset = ((2*_serial_i + 1) / 2 / _serial_n - 0.5) * width
            (
                self.ax[row, col].barh
                if horizontal
                else self.ax[row, col].bar
            )(
                np.arange(len(x)) + offset,
                y,
                (width / _serial_n),
                color=self.digest_color(color, opacity),
                edgecolor=(
                    self.digest_color(line_color, 1)
                    if line_color is not None
                    else None
                ),
                linewidth=line_width,
                label=None if show_legend is False else label,
                **kwargs_mpl,
                **kwargs,
            )
            (
                self.ax[row, col].set_yticks
                if horizontal
                else self.ax[row, col].set_xticks
            )(
                np.arange(len(x)),
                x,
            )

    def add_hist(
        self,
        x=None,
        y=None,
        bins=None,
        density=False,
        label=None,
        show_legend=None,
        color=None,
        opacity=None,
        row=0,
        col=0,
        kwargs_pty=None,
        kwargs_mpl=None,
        **kwargs,
    ):
        """
        Draw a histogram.

        Parameters
        ----------
        x: array-like
            Histogram data.
        bins: int, optional
            Number of bins.
            If undefined, plotly/matplotlib will detect automatically.
            Default: None
        label: str, optional
            Trace label for legend.
        color: str, optional
            Trace color.

            Can be hex, rgb(a) or any named color that is understood
            by matplotlib.

            The color cycle can be accessed with "C0", "C1", ...

            Default: color is retrieved from `Plot.digest_color`,
            which cycles through `COLOR_CYCLE`.
        opacity: float, optional
            Opacity (=alpha) of the fill.

            By default, fallback to alpha value provided with color argument,
            or 1.
        row, col: int, default: 0
            If the plot contains a grid, provide the coordinates.

            Attention: Indexing starts with 0!
        kwargs_pty, kwargs_mpl, **kwargs: optional
            Pass specific keyword arguments to the hist core method.
        """
        # input verification
        if x is None and y is None:
            raise ValueError("Either x or y must be defined.")
        if x is not None and y is not None:
            raise ValueError("x and y cannot be defined both.")

        bins_attribute = dict(nbinsx=bins) if y is None else dict(nbinsy=bins)
        self.element_count[row, col] += 1

        # PLOTLY
        if self.interactive:
            if kwargs_pty is None:
                kwargs_pty = dict()
            if density:
                kwargs_pty.update(dict(histnorm='probability'))
            row += 1
            col += 1
            self.fig.add_trace(
                go.Histogram(
                    x=x,
                    y=y,
                    **self._get_plotly_legend_args(
                        label,
                        show_legend=show_legend,
                    ),
                    **bins_attribute,
                    marker_color=self.digest_color(color, opacity),
                    **kwargs_pty,
                    **kwargs,
                ),
                row=row,
                col=col,
            )

        # MATPLOTLIB
        else:
            if kwargs_mpl is None:
                kwargs_mpl = dict()
            if x is None:
                x = y
                orientation = "horizontal"
            else:
                orientation = "vertical"
            self.ax[row, col].hist(
                x,
                label=None if show_legend is False else label,
                bins=bins,
                density=density,
                color=self.digest_color(color, opacity),
                orientation=orientation,
                **kwargs_mpl,
                **kwargs,
            )

    def add_boxplot(
        self,
        x,
        horizontal=False,
        label=None,
        show_legend=None,
        color=None,
        color_median="black",
        opacity=None,
        notch=True,
        row=0,
        col=0,
        kwargs_pty=None,
        kwargs_mpl=None,
        **kwargs,
    ):
        """
        Draw a boxplot.

        Parameters
        ----------
        x: array or sequence of vectors
            Data to build boxplot from.
        horizontal: bool, default: False
            Show boxplot horizontally.
        label: tuple of strs, optional
            Trace labels for legend.
        color: tuple of strs, optional
            Fill colors.

            Can be hex, rgb(a) or any named color that is understood
            by matplotlib.

            The color cycle can be accessed with "C0", "C1", ...

            Default: color is retrieved from `Plot.digest_color`,
            which cycles through `COLOR_CYCLE`.
        color_median: color, default: "black"
            MPL only.
            Color of the median line.
        opacity: float, optional
            Opacity (=alpha) of the fill.

            By default, fallback to alpha value provided with color argument,
            or 1.
        row, col: int, optional
            If the plot contains a grid, provide the coordinates.

            Attention: Indexing starts with 0!
        kwargs_pty, kwargs_mpl, **kwargs: optional
            Pass specific keyword arguments to the boxplot core method.
        """
        # determine number of boxplots
        if isinstance(x[0], (int, float)):
            n = 1
        else:
            n = len(x)
        # input validation
        if not isinstance(label, ITERABLE_TYPES):
            label = (label, ) * n
        if not isinstance(color, ITERABLE_TYPES):
            color = (color, ) * n

        # PLOTLY
        if self.interactive:
            if kwargs_pty is None:
                kwargs_pty = dict()

            # if x contains multiple datasets, iterate add_boxplot
            if not n == 1:
                for x_i, label_, show_legend_, color_, opacity_ in zip_smart(
                    x, label, show_legend, color, opacity,
                ):
                    self.add_boxplot(
                        x_i,
                        horizontal=horizontal,
                        label=label_,
                        show_legend=show_legend_,
                        row=row,
                        col=col,
                        color=color_,
                        opacity=opacity_,
                        kwargs_pty=kwargs_pty,
                        **kwargs,
                    )

            # draw a single plotly boxplot
            else:
                row += 1
                col += 1
                kw_data = "x" if horizontal else "y"
                pty_kwargs = {
                    kw_data: x,
                }
                self.fig.add_trace(
                    go.Box(
                        **pty_kwargs,
                        **self._get_plotly_legend_args(
                            label[0],
                            show_legend=show_legend,
                        ),
                        marker_color=self.digest_color(color[0], opacity),
                        **kwargs_pty,
                        **kwargs,
                    ),
                    row=row,
                    col=col,
                )

        # MATPLOTLIB
        else:
            if kwargs_mpl is None:
                kwargs_mpl = dict()
            bplots = self.ax[row, col].boxplot(
                x,
                vert=not horizontal,
                labels=None if show_legend is False else label,
                patch_artist=True,
                notch=notch,
                medianprops=dict(color=color_median),
                **kwargs_mpl,
                **kwargs,
            )
            for bplot, color_ in zip_smart(bplots["boxes"], color):
                bplot.set_facecolor(self.digest_color(color_, opacity))

    def add_heatmap(
        self,
        data,
        lim=(None, None),
        aspect=1,
        invert_x=False,
        invert_y=False,
        cmap="rainbow",
        cmap_under=None,
        cmap_over=None,
        cmap_bad=None,
        row=0,
        col=0,
        kwargs_pty=None,
        kwargs_mpl=None,
        **kwargs,
    ):
        """
        Draw a heatmap.

        Parameters
        ----------
        data: 2D array-like
            2D data to show heatmap.
        lim: list/tuple of 2x float, optional
            Lower and upper limits of the color map.
        aspect: float, default: 1
            Aspect ratio of the axes.
        invert_x, invert_y: bool, optional
            Invert the axes directions.
            Default: False
        cmap: str, default: "rainbow"
            Color map to use.
            https://matplotlib.org/stable/gallery/color/colormap_reference.html
            Note: Not all cmaps are available for both libraries,
            and may differ slightly.
        cmap_under, cmap_over, cmap_bad: str, optional
            Colors to display if under/over range or a pixel is invalid,
            e.g. in case of `np.nan`.
            `cmap_bad` is not available for interactive plotly plots.
        row, col: int, optional
            If the plot contains a grid, provide the coordinates.

            Attention: Indexing starts with 0!
        kwargs_pty, kwargs_mpl, **kwargs: optional
            Pass specific keyword arguments to the heatmap core method.
        """
        # input verification
        if lim is None:
            lim = [None, None]
        else:
            lim = list(lim)
        if len(lim) != 2:
            raise ValueError("lim must be a tuple or dict with two items.")
        self.element_count[row, col] += 1

        # PLOTLY
        if self.interactive:
            # input verification
            if kwargs_pty is None:
                kwargs_pty = dict()
            if cmap_bad is not None:
                warn("cmap_bad is not supported for plotly.")
            row += 1
            col += 1

            # add colorscale limits
            if cmap_under is not None or cmap_over is not None:
                # crappy workaround to make plotly translate named cmap to list
                cmap = _plotly_colormap_extremes(
                    px.imshow(
                        img=[[0, 0], [0, 0]],
                        color_continuous_scale=cmap,
                    ).layout.coloraxis.colorscale,
                    cmap_under,
                    cmap_over,
                )
                if lim != [None, None]:
                    delta = lim[1] - lim[0]
                    lim[0] = lim[0] - 0.000001 * delta
                    lim[1] = lim[1] + 0.000001 * delta

            self.fig.add_trace(
                go.Heatmap(
                    z=data,
                    zmin=lim[0],
                    zmax=lim[1],
                    colorscale=cmap,
                    **kwargs_pty,
                    **kwargs,
                ),
                row=row,
                col=col,
            )
            self.fig.update_xaxes(
                autorange=("reversed" if invert_x else None),
                row=row,
                col=col,
            )
            self.fig.update_yaxes(
                scaleanchor=self._get_plotly_anchor("x", self.cols, row, col),
                scaleratio=aspect,
                autorange=("reversed" if invert_x else None),
                row=row,
                col=col,
            )

        # MATPLOTLIB
        else:
            if kwargs_mpl is None:
                kwargs_mpl = dict()
            cmap = _plt_cmap_extremes(
                cmap,
                under=cmap_under,
                over=cmap_over,
                bad=cmap_bad,
            )
            imshow = self.ax[row, col].imshow(
                data,
                cmap=cmap,
                aspect=aspect,
                vmin=lim[0],
                vmax=lim[1],
                **kwargs_mpl,
                **kwargs,
            )
            self.fig.colorbar(imshow)
            if invert_x:
                self.ax[row, col].axes.invert_xaxis()
            if not invert_y:
                self.ax[row, col].axes.invert_yaxis()

    def add_regression(
        self,
        x,
        y=None,
        p=0.05,
        linspace=101,
        **kwargs,
    ):
        """
        Draw a linear regression plot.

        Parameters
        ----------
        x: array-like or `interplot.arraytools.LinearRegression` instance
            X axis data, or pre-existing LinearRegression instance.
        y: array-like, optional
            Y axis data.
            If a LinearRegression instance is provided for x,
            y can be omitted and will be ignored.
        p: float, default: 0.05
            p-value.
        linspace: int, default: 101
            Number of data points for linear regression model
            and conficence and prediction intervals.
        kwargs:
            Keyword arguments for `interplot.arraytools.LinearRegression.plot`.
        """
        if (
            isinstance(x, arraytools.LinearRegression)
            or hasattr(x, "is_linreg")
        ):
            x.plot(fig=self, **kwargs)
        else:
            arraytools.LinearRegression(
                x,
                y,
                p=p,
                linspace=linspace,
            ).plot(fig=self, **kwargs)

    def add_fill(
        self,
        x,
        y1,
        y2=None,
        label=None,
        show_legend=False,
        mode="lines",
        color=None,
        opacity=0.5,
        line_width=0.,
        line_opacity=1.,
        line_color=None,
        row=0,
        col=0,
        kwargs_pty=None,
        kwargs_mpl=None,
        **kwargs,
    ):
        """
        Draw a fill between two y lines.

        Parameters
        ----------
        x: array-like
        y1, y2: array-like, optional
            If only `x` and `y1` is defined,
            it will be assumed as `y1` and `y2`,
            and `x` will be the index, starting from 0.
        label: str, optional
            Trace label for legend.
        color, line_color: str, optional
            Fill and line color.

            Can be hex, rgb(a) or any named color that is understood
            by matplotlib.

            The color cycle can be accessed with "C0", "C1", ...

            If line_color is undefined, the the fill color will be used.

            Default: color is retrieved from `Plot.digest_color`,
            which cycles through `COLOR_CYCLE`.
        opacity, line_opacity: float, default: 0.5
            Opacity (=alpha) of the fill.

            Set to None to use a value provided with the color argument.
        line_width: float, default: 0.
            Boundary line width.
        row, col: int, default: 0
            If the plot contains a grid, provide the coordinates.

            Attention: Indexing starts with 0!
        kwargs_pty, kwargs_mpl, **kwargs: optional
            Pass specific keyword arguments to the fill core method.
        """
        # input verification
        if y2 is None:
            y1, y2 = x, y1
            x = np.arange(len(y1))
        self.element_count[row, col] += 1

        fill_color = self.digest_color(
            color,
            opacity,
            increment=0 if line_color is None else 1,
        )
        line_color = self.digest_color(
            color if line_color is None else line_color,
            line_opacity,
        )

        # PLOTLY
        if self.interactive:
            if kwargs_pty is None:
                kwargs_pty = dict()
            legendgroup = "fill_{}".format(self.element_count[row, col])
            row += 1
            col += 1
            self.fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y1,
                    mode=mode,
                    **self._get_plotly_legend_args(
                        label, "fill border 1", show_legend=show_legend),
                    line=dict(width=line_width),
                    marker_color=line_color,
                    legendgroup=legendgroup,
                    **kwargs_pty,
                    **kwargs,
                ),
                row=row,
                col=col,
            )
            self.fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y2,
                    mode=mode,
                    **self._get_plotly_legend_args(label, "fill border 1"),
                    fill="tonexty",
                    fillcolor=fill_color,
                    line=dict(width=line_width),
                    marker_color=line_color,
                    legendgroup=legendgroup,
                    **kwargs_pty,
                    **kwargs,
                ),
                row=row,
                col=col,
            )

        # MATPLOTLIB
        else:
            if kwargs_mpl is None:
                kwargs_mpl = dict()
            self.ax[row, col].fill_between(
                x,
                y1,
                y2,
                label=None if show_legend is False else label,
                linewidth=line_width,
                edgecolor=self.digest_color(
                    line_color, line_opacity, increment=0),
                facecolor=self.digest_color(color, opacity),
                **kwargs_mpl,
                **kwargs,
            )

    def add_text(
        self,
        x,
        y,
        text,
        horizontal_alignment="center",
        vertical_alignment="center",
        text_alignment=None,
        data_coords=None,
        x_data_coords=True,
        y_data_coords=True,
        color="black",
        opacity=None,
        row=0,
        col=0,
        kwargs_pty=None,
        kwargs_mpl=None,
        **kwargs,
    ):
        """
        Draw text.

        Parameters
        ----------
        x, y: float
            Coordinates of the text.
        text: str
            Text to add.
        horizontal_alignment, vertical_alignment: str, default: "center"
            Where the coordinates of the text box anchor.

            Options for `horizontal_alignment`:
                - "left"
                - "center"
                - "right"

            Options for `vertical_alignment`:
                - "top"
                - "center"
                - "bottom"
        text_alignment: str, optional
            Set how text is aligned inside its box.

            If left undefined, horizontal_alignment will be used.
        data_coords: bool, default: True
            Whether the `x`, `y` coordinates are provided in data coordinates
            or in relation to the axes.

            If set to `False`, `x`, `y` should be in the range (0, 1).
            If `data_coords` is set, it will override
            `x_data_coords` and `y_data_coords`.
        x_data_coords, y_data_coords: bool, default: True
            PTY only.
            Specify the anchor for each axis separate.
        color: str, default: "black"
            Trace color.

            Can be hex, rgb(a) or any named color that is understood
            by matplotlib.

            The color cycle can be accessed with "C0", "C1", ...

            Default: color is retrieved from `Plot.digest_color`,
            which cycles through `COLOR_CYCLE`.
        opacity: float, optional
            Opacity (=alpha) of the fill.

            By default, fallback to alpha value provided with color argument,
            or 1.
        row, col: int, optional
            If the plot contains a grid, provide the coordinates.

            Attention: Indexing starts with 0!
        kwargs_pty, kwargs_mpl, **kwargs: optional
            Pass specific keyword arguments to the line core method.
        """
        # input verification
        if data_coords is not None:
            x_data_coords = data_coords
            y_data_coords = data_coords
        text_alignment = (
            horizontal_alignment
            if text_alignment is None
            else text_alignment
        )

        # PLOTLY
        if self.interactive:
            if kwargs_pty is None:
                kwargs_pty = dict()
            if vertical_alignment == "center":
                vertical_alignment = "middle"
            row += 1
            col += 1
            x_domain = "" if x_data_coords else " domain"
            y_domain = "" if y_data_coords else " domain"
            self.fig.add_annotation(
                x=x,
                y=y,
                text=self._encode_html(text),
                align=text_alignment,
                xanchor=horizontal_alignment,
                yanchor=vertical_alignment,
                xref=self._get_plotly_anchor(
                    "x", self.cols, row, col
                ) + x_domain,
                yref=self._get_plotly_anchor(
                    "y", self.cols, row, col
                ) + y_domain,
                font=dict(color=self.digest_color(color, opacity)),
                row=row,
                col=col,
                showarrow=False,
                **kwargs_pty,
            )

        # MATPLOTLIB
        else:
            # input validation
            if kwargs_mpl is None:
                kwargs_mpl = dict()
            if not x_data_coords == y_data_coords:
                warn(
                    "x_data_coords and y_data_coords must correspond "
                    "for static matplotlib plot. x_data_coords was used."
                )
            transform = (
                dict()
                if x_data_coords
                else dict(transform=self.ax[row, col].transAxes)
            )
            self.ax[row, col].text(
                x,
                y,
                s=text,
                color=self.digest_color(color, opacity),
                horizontalalignment=horizontal_alignment,
                verticalalignment=vertical_alignment,
                multialignment=text_alignment,
                **transform,
                **kwargs_mpl,
                **kwargs,
            )

    def post_process(
        self,
        pty_update_layout=None,
        pty_custom_func=None,
        mpl_custom_func=None,
    ):
        """
        Finish the plot.

        Parameters
        ----------
        Note: If not provided, the parameters given on init will be used.
        pty_update_layout: dict, optional
            PLOTLY ONLY.
            Pass keyword arguments to plotly's
            fig.update_layout(**pty_update_layout)
            Thus, take full control over
            Default: None
        pty_custom_func: function, optional
            PLOTLY ONLY.
            Pass a function reference to further style the plotly graphs.
            Function must accept fig and return fig.

            Default: None

            Example:

            >>> def pty_custom_func(fig):
            ...     fig.do_stuff()
            ...     return fig
        mpl_custom_func: function, optional
            MATPLOTLIB ONLY.
            Pass a function reference to further style the matplotlib graphs.
            Function must accept fig, ax and return fig, ax.

            Default: None

            Example:

            >>> def mpl_custom_func(fig, ax):
            ...     fig.do_stuff()
            ...     ax.do_more()
            ...     return fig, ax
        """
        # input verification
        pty_update_layout = (
            self.pty_update_layout
            if pty_update_layout is None
            else pty_update_layout
        )
        pty_custom_func = (
            self.pty_custom_func
            if pty_custom_func is None
            else pty_custom_func
        )
        mpl_custom_func = (
            self.mpl_custom_func
            if mpl_custom_func is None
            else mpl_custom_func
        )

        # PLOTLY
        if self.interactive:
            if pty_update_layout is not None:
                self.fig.update_layout(**pty_update_layout)
            if pty_custom_func is not None:
                self.fig = pty_custom_func(self.fig)

        # MATPLOTLIB
        else:

            # axis limits
            for ax_row, xlim_row, ylim_row in zip_smart(
                self.ax,
                filter_nozip(self.xlim),
                filter_nozip(self.ylim),
            ):
                for ax_tile, xlim_tile, ylim_tile in zip_smart(
                    ax_row,
                    filter_nozip(xlim_row),
                    filter_nozip(ylim_row),
                ):
                    ax_tile.set_xlim(xlim_tile)
                    ax_tile.set_ylim(ylim_tile)

            # legend for each subplot
            for i_row, loc_row, title_row in zip_smart(
                range(self.rows),
                self.legend_loc,
                self.legend_title,
            ):
                for i_col, loc_tile, title_tile in zip_smart(
                    range(self.cols),
                    loc_row,
                    title_row,
                ):
                    # don't show legend
                    if (
                        type(loc_tile) is bool and loc_tile is False
                        or loc_tile is None
                            and self.element_count[i_row, i_col] < 2
                    ):
                        pass

                    # show legend if n>=2 or set to True
                    else:
                        if type(loc_tile) is bool and loc_tile is True:
                            loc_tile = "best"
                        self.ax[i_row, i_col].legend(
                            title=title_tile,
                            loc=loc_tile,
                        )

            self.fig.tight_layout(pad=1.5)

            if mpl_custom_func is not None:
                self.fig, self.ax = mpl_custom_func(self.fig, self.ax)

        if self.save_fig is not None:
            self.save(self.save_fig, export_format=self.save_format)

    @_serialize_save
    def save(self, path, export_format=None, print_confirm=True, **kwargs):
        """
        Save the plot.

        Parameters
        ----------
        path: str, pathlib.Path, bool
            May point to a directory or a filename.
            If only a directory is provided (or `True` for local directory),
            the filename will automatically be generated from the plot title.

            An iterable of multiple paths may be provided. In this case
            the save command will be repeated for each element.
        export_format: str, optional
            If the format is not indicated in the file name, specify a format.

            An iterable of multiple formats may be provided. In this case
            the save command will be repeated for each element.
        print_confirm: bool, optional
            Print a confirmation message where the file has been saved.
            Default: True

        Returns
        -------
        pathlib.Path
            Path to the exported file.
        """
        # input verification
        if isinstance(path, bool):
            if path:
                path = Path()
            else:
                return
        else:
            path = Path(path)

        # auto-generate filename
        if path.is_dir():
            filename = self.title
            if filename is None or str(filename) == "":
                filename = "interplot_figure"
            else:
                filename = str(filename)

            for key, value in conf.EXPORT_REPLACE.items():
                filename = re.sub(key, value, filename)
            filename += "." + (
                conf.EXPORT_FORMAT if export_format is None else export_format
            )
            path = path / filename

        # PLOTLY
        if self.interactive:

            # HTML
            if str(path)[-5:] == ".html":
                self.fig.write_html(
                    path,
                    config=(
                        conf.PTY_CONFIG
                        if self.save_config is None
                        else self.save_config
                    ),
                    **kwargs,
                )

            # image
            else:
                scale = self.dpi / 100.
                self.fig.write_image(
                    path,
                    scale=scale,
                    **kwargs,
                )

        # MATPLOTLIB
        else:
            self.fig.savefig(
                path,
                facecolor="white",
                bbox_inches="tight",
                **kwargs,
            )

        if print_confirm:
            print("saved figure at {}".format(str(path)))

        return path

    def show(self):
        """Show the plot."""
        if CALLED_FROM_NOTEBOOK:
            if self.interactive:
                init_notebook_mode()
                display_html(self.JS_RENDER_WARNING, raw=True)
                return self.fig.show(
                    config=(
                        conf.PTY_CONFIG
                        if self.save_config is None
                        else self.save_config
                    )
                )
            display_png(self._repr_png_(), raw=True)
            return

        if self.interactive:
            return self.fig.show(
                config=(
                    conf.PTY_CONFIG
                    if self.save_config is None
                    else self.save_config
                )
            )
        return self.fig.show()

    def _repr_mimebundle_(self, *args, **kwargs):
        if self.interactive:
            return self.fig._repr_mimebundle_(*args, **kwargs)

        else:
            raise NotImplementedError

    def _repr_html_(self):
        if self.interactive:
            init_notebook_mode()
            return self.JS_RENDER_WARNING + self.fig._repr_html_()
        raise NotImplementedError

    def _repr_png_(self):
        if self.interactive:
            raise NotImplementedError
        bio = BytesIO()
        self.fig.savefig(bio, format="png")
        bio.seek(0)
        return bio.read()


def magic_plot(core, doc_decorator=None):
    """
    Plot generator wrapper.

    Your function feeds the data, the wrapper gives control over the plot
    to the user.

    Parameters
    ----------
    doc_decorator: str, optional
        Append the docstring with the decorated parameters.

        By default, the global variable `_DOCSTRING_DECORATOR` will be used.

    Examples
    --------
    >>> @interplot.magic_plot
    ... def plot_lines(samples=100, n=10, label="sigma={0}, mu={1}", fig=None):
    ...     \"\"\"
    ...     Plot Gaussian noise.
    ...
    ...     The function must accept the `fig` parameter from the decorator.
    ...     \"\"\"
    ...     for i in range(1, n+1):
    ...         fig.add_line(
    ...             np.random.normal(i*10,i,samples),
    ...             label=label.format(i, i*10),
    ...         )

    >>> plot_lines(samples=200, title="Normally distributed Noise")

    .. raw:: html
        :file: ../source/plot_examples/gauss_plot_pty.html

    >>> plot_lines(
    ...     samples=200, interactive=False, title="Normally distributed Noise")

    .. image:: plot_examples/gauss_plot_mpl.png
        :alt: [matplotlib plot "Normally distributed Noise]
    """
    doc_decorator = (
        conf._DOCSTRING_DECORATOR
        if doc_decorator is None
        else doc_decorator
    )

    def wrapper(
        *args,
        interactive=None,
        rows=1,
        cols=1,
        fig=None,
        skip_post_process=False,
        title=None,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        shared_xaxes=False,
        shared_yaxes=False,
        column_widths=None,
        row_heights=None,
        fig_size=None,
        dpi=None,
        legend_loc=None,
        legend_title=None,
        save_fig=None,
        save_format=None,
        save_config=None,
        pty_update_layout=None,
        pty_custom_func=None,
        mpl_custom_func=None,
        **kwargs,
    ):
        # init Plot
        fig = Plot.init(
            fig,
            interactive=interactive,
            rows=rows,
            cols=cols,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes,
            column_widths=column_widths,
            row_heights=row_heights,
            fig_size=fig_size,
            dpi=dpi,
            legend_loc=legend_loc,
            legend_title=legend_title,
            save_fig=save_fig,
            save_format=save_format,
            save_config=save_config,
            pty_update_layout=pty_update_layout,
            pty_custom_func=pty_custom_func,
            mpl_custom_func=mpl_custom_func,
        )

        # execute core method
        core(*args, fig=fig, **kwargs)

        # post-processing
        if not skip_post_process:
            fig.post_process()

        # return
        return fig

    # rewrite _DOCSTRING_DECORATOR
    wrapper.__doc__ = _rewrite_docstring(
        core.__doc__,
        doc_decorator,
    ) + "\n"

    return wrapper


def magic_plot_preset(doc_decorator=None, **kwargs_preset):
    """
    Plot generator wrapper, preconfigured.

    Your function feeds the data, the wrapper gives control over the plot
    to the user.

    Parameters
    ----------
    doc_decorator: str, optional
        Append the docstring with the decorated parameters.

        By default, the global variable `conf._DOCSTRING_DECORATOR`
        will be used.
    **kwargs_preset: dict
        Define presets for any keyword arguments accepted by `Plot`.

        Setting `strict_preset=True` prevents overriding the preset.

    Examples
    --------
    >>> @interplot.magic_plot_preset(
    ...     title="Data view",
    ...     interactive=False,
    ...     strict_preset=False,
    ... )
    ... def line(
    ...     data,
    ...     fig,
    ...     **kwargs,
    ... ):
    ...     fig.add_line(data)
    ...
    ... line([0,4,6,7], xlabel="X axis")

    [matplotlib figure, "Data view"]
    """
    strict_preset = kwargs_preset.get("strict_preset", False)
    if "strict_preset" in kwargs_preset:
        del kwargs_preset["strict_preset"]

    def decorator(core):

        def inner(*args_inner, **kwargs_inner):
            # input clash check
            # decorator is set to strict presets
            if strict_preset:
                for kwarg in kwargs_inner:
                    if kwarg in kwargs_preset:
                        raise ValueError(
                            "Keyword argument '" + kwarg + "' cannot be set.\n"
                            "Overriding keyword arguments was deactivated with"
                            " strict_preset=True in the decorator function."
                        )

            # default behaviour: presets can be overridden
            else:
                for kwarg in kwargs_inner:
                    if kwarg in kwargs_preset:
                        del kwargs_preset[kwarg]

            return magic_plot(core, doc_decorator=doc_decorator)(
                *args_inner,
                **kwargs_inner,
                **kwargs_preset,
            )

        # rewrite DOCSTRING_DECORATOR
        inner.__doc__ = _rewrite_docstring(
            core.__doc__,
            doc_decorator,
            kwargs_remove=kwargs_preset if strict_preset else (),
        )
        return inner

    return decorator


@magic_plot
@wraps(Plot.add_line)
def line(
    *args,
    fig,
    **kwargs,
):
    fig.add_line(*args, **kwargs)


@magic_plot
@wraps(Plot.add_scatter)
def scatter(
    *args,
    fig,
    **kwargs,
):
    fig.add_scatter(*args, **kwargs)


@magic_plot
@wraps(Plot.add_linescatter)
def linescatter(
    *args,
    fig,
    **kwargs,
):
    fig.add_linescatter(*args, **kwargs)


@magic_plot
@wraps(Plot.add_bar)
def bar(
    *args,
    fig,
    **kwargs,
):
    fig.add_bar(*args, **kwargs)


@magic_plot
@wraps(Plot.add_fill)
def fill(
    *args,
    fig,
    **kwargs,
):
    fig.add_fill(*args, **kwargs)


@magic_plot
@wraps(Plot.add_text)
def text(
    *args,
    fig,
    **kwargs,
):
    fig.add_text(*args, **kwargs)


@magic_plot
@wraps(Plot.add_hist)
def hist(
    *args,
    fig,
    **kwargs,
):
    fig.add_hist(*args, **kwargs)


@magic_plot
@wraps(Plot.add_boxplot)
def boxplot(
    *args,
    fig,
    **kwargs,
):
    fig.add_boxplot(*args, **kwargs)


@magic_plot
@wraps(Plot.add_heatmap)
def heatmap(
    *args,
    fig,
    **kwargs,
):
    fig.add_heatmap(*args, **kwargs)


@magic_plot
@wraps(Plot.add_regression)
def regression(
    *args,
    fig,
    **kwargs,
):
    fig.add_regression(*args, **kwargs)


class ShowDataArray(NotebookInteraction):
    """
    Automatically display a `xarray.DataArray` in a Jupyter notebook.

    If the DataArray has more than two dimensions, provide default
    sel or isel selectors to reduce to two dimensions.

    Parameters
    ----------
    data: xarray.DataArray
    default_sel: dict, optional
        Select a subset of a the DataArray by label.
        Can be a slice or the type of the dimension.
    default_isel: dict, optional
        Select a subset of a the DataArray by integer count.
        Can be a integer slice or an integer.
    """

    def __init__(
        self,
        data,
        default_sel=None,
        default_isel=None,
    ):
        self.data = data
        self.default_sel = default_sel
        self.default_isel = default_isel

    @magic_plot
    def _plot_core(
        self,
        data,
        sel=None,
        isel=None,
        fig=None,
        **kwargs,
    ):
        sel = {} if sel is None else sel
        isel = {} if isel is None else isel
        fig.add_line(data.sel(**sel).isel(**isel), **kwargs)

    def plot(
        self,
        *args,
        sel=None,
        isel=None,
        **kwargs,
    ):
        """
        Show the DataArray.

        Parameters
        ----------
        sel: dict, optional
            Select a subset of a the DataArray by label.
            Can be a slice or the type of the dimension.
            If None, default_sel will be used.
        isel: dict, optional
            Select a subset of a the DataArray by integer count.
            Can be a integer slice or an integer.
            If None, default_isel will be used.

        Returns
        -------
        Plot.fig
        """
        sel = self.default_sel if sel is None else sel
        isel = self.default_isel if isel is None else isel
        return self._plot_core(self.data, *args, sel=sel, isel=isel, **kwargs)


class ShowDataset(ShowDataArray):
    """
    Automatically display a `xarray.Dataset` in a Jupyter notebook.

    Provide a default variable to display from the Dataset for
    automatic display.
    If the Dataset has more than two dimensions, provide default
    sel or isel selectors to reduce to two dimensions.

    Parameters
    ----------
    data: xarray.DataArray
    default_var: str, optional
        Select the variable of the Dataset to display by label.
    default_sel: dict, optional
        Select a subset of a the Dataset by label.
        Can be a slice or the type of the dimension.
    default_isel: dict, optional
        Select a subset of a the Dataset by integer count.
        Can be a integer slice or an integer.
    """

    def __init__(
        self,
        data,
        default_var=None,
        default_sel=None,
        default_isel=None,
    ):
        self.data = data
        self.default_var = default_var
        self.default_sel = default_sel
        self.default_isel = default_isel

    def plot(
        self,
        *args,
        var=None,
        sel=None,
        isel=None,
        **kwargs,
    ):
        """
        Show a variable of the Dataset.

        Parameters
        ----------
        var: str, optional
            Select the variable of the Dataset to display by label.
            If None, default_var will be used.
        sel: dict, optional
            Select a subset of a the DataArray by label.
            Can be a slice or the type of the dimension.
            If None, default_sel will be used.
        isel: dict, optional
            Select a subset of a the DataArray by integer count.
            Can be a integer slice or an integer.
            If None, default_isel will be used.

        Returns
        -------
        Plot.fig
        """

        var = self.default_var if var is None else var
        sel = self.default_sel if sel is None else sel
        isel = self.default_isel if isel is None else isel
        return super()._plot_core(
            self.data[var], *args, sel=sel, isel=isel, **kwargs
        )
