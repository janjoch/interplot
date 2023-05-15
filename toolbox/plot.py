"""
Create matplotlib/plotly hybrid plots with a few lines of code.

It combines the best of the matplotlib and the plotly worlds.
All the necessary boilerplate code is contained in this module.

Currently supported:
* line plots (scatter)
* histograms
* heatmaps

Example:
```
>>> toolbox.plot.line([0,4,6,7], [1,2,4,8])
[plotly figure]

>>> toolbox.plot.line(
>>>     [0,4,6,7],
>>>     [1,2,4,8],
>>>     interactive=False,
>>>     title="matploblib static figure",
>>>     xlabel="X",
>>>     ylabel="Y",
>>> )
[matplotlib figure "matploblib static figure"]
```
"""


import re
from warnings import warn

import numpy as np

import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import plotly.offline

from .iter import ITERABLE_TYPES, zip_smart, filter_nozip


def init_notebook_mode(connected=False):
    """
    Initialize plotly.js in the browser if not already done.

    Parameters
    ----------
    connected: bool, optional
        If True, the plotly.js library will be loaded from an online CDN.
        If False, the plotly.js library will be loaded locally.
        Default: False
    """
    plotly.offline.init_notebook_mode(connected=connected)


# if imported in notebook, init plotly notebook mode
try:
    __IPYTHON__  # type: ignore
    CALLED_FROM_NOTEBOOK = True
except NameError:
    CALLED_FROM_NOTEBOOK = False
if CALLED_FROM_NOTEBOOK:
    init_notebook_mode()


REWRITE_DOCSTRING = True

DOCSTRING_DECORATOR = """
    interactive: bool, optional
        Display an interactive plotly line plot
        instead of the default matplotlib figure.
        Default: True
    rows, cols: int, optional
        Create a grid with x rows and y columns.
        Default: 1
    title: str, optional
        Plot title.
        Default: None
    xlabel, ylabel: str or str tuple, optional
        Axis labels.
        Either one title for the entire axis or one for each row/column.
        Default: None
    xlim, ylim: tuple of 2 numbers or nested, optional
        In case of multiple rows/cols:
        Provide either:
        - a tuple
        - a tuple for each row
        - a tuple for each row containing tuple for each column.
        Axis range limits.
    shared_xaxes, shared_yaxes: str, optional
        Define how multiple subplots share there axes.
        Options:
            "all"
            "rows"
            "columns" or "cols"
            None or False
        Default: None
    column_widths, row_heights: tuple/list, optional
        Ratios of the width/height dimensions in each column/row.
        Will be normalised to a sum of 1.
        Default: None
    fig_size: tuple of 2x float, optional
        Figure size in pixels.
        Default: None
    dpi: int, optional
        Plot resolution.
        Default 100.
    legend_loc: str, optional
        MATPLOTLIB ONLY.
        Default:
            In case of 1 line: None
            In case of >1 line: "best" (auto-detect)
    legend_title: str, optional
        Default: None
    save_fig: str or pathlib.Path, optional
        Provide a path to export the plot.
        Possible formats: png, jpg, svg, html, ...
        The figure will only be saved on calling the instance's .post_process()
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
        Example:
        >>> def pty_custom_func(fig):
        >>>     fig.do_stuff()
        >>>     return fig
        Default: None
    mpl_custom_func: function, optional
        MATPLOTLIB ONLY.
        Pass a function reference to further style the matplotlib graphs.
        Function must accept fig, ax and return fig, ax.
        Note: ax always has row and col coordinates, even if the plot is
        just 1x1.
        Example:
        >>> def mpl_custom_func(fig, ax):
        >>>     fig.do_stuff()
        >>>     ax[0, 0].do_more()
        >>>     return fig, ax
        Default: None"""


def _rewrite_docstring(doc_core, doc_decorator=None, kwargs_remove=()):
    """
    Appends arguments to a docstring.

    Returns original docstring if REWRITE_DOCSTRING is set to False.

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

    Returns
    -------
    str:
        Rewritten docstring
    """
    # check rewrite flag
    if not REWRITE_DOCSTRING:
        return doc_core

    # input check
    doc_core = "" if doc_core is None else doc_core
    doc_decorator = (
        DOCSTRING_DECORATOR
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

    # non-numpy DOCSTRING_DECORATOR, just append in the end
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


class NotebookInteraction:
    """
    Calls the child's show()._repr_html_()
    for automatic display in Jupyter Notebooks
    """

    def __call__(self, *args, **kwargs):
        # look for show() method
        try:
            return self.show(*args, **kwargs)

        # fall back to plot() method
        except AttributeError:
            return self.plot(*args, **kwargs)

    def _repr_html_(self):
        init_notebook_mode()
        # look for show() method
        try:
            return self.show()._repr_html_()

        # fall back to plot() method
        except AttributeError:
            return self.plot()._repr_html_()


class Plot:
    """
    Create matplotlib/plotly hybrid plots with a flat API in few lines of code.

    It combines the best of the matplotlib and the plotly worlds.
    All the necessary boilerplate code is contained in this class.

    See also the magic_plot decorator function, which wraps the Plot class
    around a user-written function.

    Currently supported:
    * line plots (scatter)
    * histograms
    * heatmaps
    * subplot grid

    Parameters
    ----------
    """
    __doc__ = _rewrite_docstring(__doc__)

    def __init__(
        self,
        interactive=True,
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
        dpi=100,
        legend_loc=None,
        legend_title=None,
        save_fig=None,
        pty_update_layout=None,
        pty_custom_func=None,
        mpl_custom_func=None,
    ):
        # input verification
        if shared_xaxes == "cols":
            shared_xaxes = "columns"
        if shared_yaxes == "cols":
            shared_yaxes = "columns"

        self.interactive = interactive
        self.rows = rows
        self.cols = cols
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.legend_loc = legend_loc
        self.legend_title = legend_title
        self.dpi = dpi
        self.save_fig = save_fig
        self.pty_update_layout = pty_update_layout
        self.pty_custom_func = pty_custom_func
        self.mpl_custom_func = mpl_custom_func
        self.element_count = np.zeros((rows, cols), dtype=int)

        # init plotly
        if self.interactive:

            # init fig
            self.fig = sp.make_subplots(
                rows=rows,
                cols=cols,
                shared_xaxes=shared_xaxes,
                shared_yaxes=shared_yaxes,
                row_heights=row_heights,
                column_widths=column_widths,
            )

            # unpacking
            width, height = fig_size if fig_size is not None else (None, None)
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
                title=title,
                legend_title=legend_title,
                height=height,
                width=width,
                barmode="overlay",
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
            if fig_size is not None:
                px = 1 / dpi
                fig_size = (fig_size[0] * px, fig_size[1] * px)
            self.fig, self.ax = plt.subplots(
                rows,
                cols,
                figsize=fig_size,
                dpi=dpi,
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
                    if shared_xaxes == "all":
                        self.ax[i_row, i_col].sharex(self.ax[0, 0])
                    elif (
                        shared_xaxes == "columns"
                        or type(shared_xaxes) is bool and shared_xaxes is True
                    ):
                        self.ax[i_row, i_col].sharex(self.ax[0, i_col])
                    elif shared_xaxes == "rows":
                        self.ax[i_row, i_col].sharex(self.ax[i_row, 0])

                    # set shared y axes
                    if shared_yaxes == "all":
                        self.ax[i_row, i_col].sharey(self.ax[0, 0])
                    elif shared_yaxes == "columns":
                        self.ax[i_row, i_col].sharey(self.ax[0, i_col])
                    elif (
                        shared_yaxes == "rows"
                        or type(shared_yaxes) is bool and shared_yaxes is True
                    ):
                        self.ax[i_row, i_col].sharey(self.ax[i_row, 0])

            # axis labels
            if isinstance(self.xlabel, ITERABLE_TYPES):
                for text, i_col in zip_smart(self.xlabel, range(self.cols)):
                    self.ax[self.rows - 1, i_col].set_xlabel(text)
            else:
                self.fig.supxlabel(self.xlabel)
            if isinstance(self.ylabel, ITERABLE_TYPES):
                for text, i_row in zip_smart(self.ylabel, range(self.rows)):
                    self.ax[i_row, 0].set_ylabel(text)
            else:
                self.fig.supylabel(self.ylabel)

    def add_line(
        self,
        x,
        y=None,
        label=None,
        color=None,
        row=0,
        col=0,
        **kwargs,
    ):
        """
        Add a line to the plot.

        Parameters
        ----------
        x: array-like
        y: array-like, optional
            If only x is defined, it will be assumed as x,
            and x will be the index, starting from 0.
        label: str, optional
            Trace label for legend.
        color: str, optional
            Trace color.
        row, col: int, optional
            If the plot contains a grid, provide the coordinates.
            Attention: Indexing starts with 0!
        **kwargs: optional
            Pass specific keyword arguments to the line core method.
        """
        # input verification
        if y is None:
            y = x
            x = np.arange(len(y))
        self.element_count[row, col] += 1

        # PLOTLY
        if self.interactive:
            row += 1
            col += 1
            self.fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=label,
                    marker_color=color,
                    **kwargs,
                ),
                row=row,
                col=col,
            )

        # MATPLOTLIB
        else:
            self.ax[row, col].plot(x, y, label=label, color=color, **kwargs)

    def add_hist(
        self,
        x=None,
        y=None,
        bins=None,
        label=None,
        color=None,
        row=0,
        col=0,
        **kwargs,
    ):
        """
        Add a histogram to the plot.

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
        row, col: int, optional
            If the plot contains a grid, provide the coordinates.
            Attention: Indexing starts with 0!
        **kwargs: optional
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
            row += 1
            col += 1
            self.fig.add_trace(
                go.Histogram(
                    x=x,
                    y=y,
                    name=label,
                    **bins_attribute,
                    marker_color=color,
                    **kwargs,
                ),
                row=row,
                col=col,
            )

        # MATPLOTLIB
        else:
            if x is None:
                x = y
                orientation = "horizontal"
            else:
                orientation = "vertical"
            self.ax[row, col].hist(
                x,
                label=label,
                bins=bins,
                color=color,
                orientation=orientation,
                **kwargs,
            )

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
        **kwargs,
    ):
        """
        Add a heatmap to the plot.

        Parameters
        ----------
        data: 2D array-like
            2D data to show heatmap.
        lim: list/tuple of 2x float, optional
            Lower and upper limits of the color map.
        cmap: str, optional
            Color map to use.
            https://matplotlib.org/stable/gallery/color/colormap_reference.html
            Note: Not all cmaps are available for both libraries,
            and may differ slightly.
            Default: "rainbow"
        cmap_under, cmap_over, cmap_bad: str, optional
            Colors to display if under/over range or a pixel is invalid,
            e.g. in case of np.nan.
            cmap_bad is not available for interactive plotly plots.
        row, col: int, optional
            If the plot contains a grid, provide the coordinates.
            Attention: Indexing starts with 0!
        **kwargs: optional
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
                    **kwargs,
                ),
                row=row,
                col=col,
            )
            if row == 1 and col == 1:
                scaleanchor = "x"
            else:
                scaleanchor = "x{}".format((row-1)*self.cols + col)
            self.fig.update_xaxes(
                autorange=("reversed" if invert_x else None),
                row=row,
                col=col,
            )
            self.fig.update_yaxes(
                scaleanchor=scaleanchor,
                scaleratio=aspect,
                autorange=("reversed" if invert_x else None),
                row=row,
                col=col,
            )

        # MATPLOTLIB
        else:
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
                **kwargs,
            )
            self.fig.colorbar(imshow)
            if invert_x:
                self.ax[row, col].axes.invert_xaxis()
            if not invert_y:
                self.ax[row, col].axes.invert_yaxis()

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
            Example:
            >>> def pty_custom_func(fig):
            >>>     fig.do_stuff()
            >>>     return fig
            Default: None
        mpl_custom_func: function, optional
            MATPLOTLIB ONLY.
            Pass a function reference to further style the matplotlib graphs.
            Function must accept fig, ax and return fig, ax.
            Example:
            >>> def mpl_custom_func(fig, ax):
            >>>     fig.do_stuff()
            >>>     ax.do_more()
            >>>     return fig, ax
            Default: None
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
            self.save(self.save_fig)

    def save(self, path, **kwargs):

        # PLOTLY
        if self.interactive:

            # HTML
            if str(path)[-5:] == ".html":
                self.fig.write_html(path, **kwargs)

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
                face_color="white",
                bbox_inches="tight",
                **kwargs,
            )

        print("saved figure at {}".format(str(path)))

    def show(self):
        return self.fig.show()

    def _repr_html_(self):
        if self.interactive:
            init_notebook_mode()
            return self.fig._repr_html_()
        return self.fig.show()


def magic_plot(core, doc_decorator=None):
    """
    Boilerplate code to advance Python plots.
    """
    doc_decorator = (
        DOCSTRING_DECORATOR
        if doc_decorator is None
        else doc_decorator
    )

    def wrapper(
        *args,
        interactive=True,
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
        dpi=100,
        legend_loc=None,
        legend_title=None,
        save_fig=None,
        pty_update_layout=None,
        pty_custom_func=None,
        mpl_custom_func=None,
        **kwargs,
    ):
        # init Plot
        if fig is None:
            fig = Plot(
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

    # rewrite DOCSTRING_DECORATOR
    wrapper.__doc__ = _rewrite_docstring(
        core.__doc__,
        doc_decorator,
    )

    return wrapper


def magic_plot_preset(doc_decorator=None, **kwargs_preset):
    """Pre-configure the magic_plot decorator"""
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
def line(
    *args,
    fig,
    **kwargs,
):
    """
    Draw a simple line plot.

    Parameters
    ----------
    x: array-like
    y: array-like, optional
        If only x is defined, it will be assumed as x,
        and x will be the index, starting from 0.
    label: str, optional
        Trace label for legend.
    color: str, optional
        Trace color.
    **kwargs: optional
        Pass specific keyword arguments to the line core method.
    """
    fig.add_line(*args, **kwargs)


@magic_plot
def hist(
    *args,
    fig,
    **kwargs,
):
    """
    Draw a simple histogram.

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
    **kwargs: optional
        Pass specific keyword arguments to the hist core method.
    """
    fig.add_hist(*args, **kwargs)


@magic_plot
def heatmap(
    *args,
    fig,
    **kwargs,
):
    """
    Draw a simple heatmap.

    Parameters
    ----------
    data: 2D array-like
        2D data to show heatmap.
    lim: list/tuple of 2x float, optional
        Lower and upper limits of the color map.
    cmap: str, optional
        Color map to use.
        https://matplotlib.org/stable/gallery/color/colormap_reference.html
        Note: Not all cmaps are available for both libraries,
        and may differ slightly.
        Default: "rainbow"
    cmap_under, cmap_over, cmap_bad: str, optional
        Colors to display if under/over range or a pixel is invalid,
        e.g. in case of np.nan.
        cmap_bad is not available for interactive plotly plots.
    **kwargs: optional
        Pass specific keyword arguments to the hist core method.
    """
    fig.add_heatmap(*args, **kwargs)
