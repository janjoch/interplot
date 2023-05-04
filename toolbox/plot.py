"""
Boilerplate code to advance Python plots.

It combines the best of the matplotlib and the plotly worlds.

Example:
```
>>> @tb.plot.lineplot_advanced
>>> def plot(*xy, add_trace=None, **kw_xy):
>>>     add_trace(*xy, **kw_xy)

>>> plot([1,2,4,8])
[plotly figure]

>>> plot(
>>>     [0,4,6,7], [1,2,4,8],
>>>     interactive=False,
>>>     title="matploblib static figure",
>>> )
[matplotlib figure]
```
"""


import re
import functools

import numpy as np

import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px


REWRITE_DOCSTRING = True

DOC_INTERACTIVE = """
    interactive: bool, optional
        Display an interactive plotly line plot
        instead of the default matplotlib figure.
        Default: False"""
DOC_LINEPLOT = """
    title: str, optional
        Plot title.
        Default: None
    xlabel, ylabel: str, optional
        Axis labels.
        Default: None
    fig_size: tuple of 2x float, optional
        Default: None
        PLOTLY: dimensions in px.
        MATPLOTLIB: dimensions in inch.
    xlim, ylim: tuple of 2 numbers, optional
        Axis range limits.
    legend_loc: str, optional
        MATPLOTLIB ONLY.
        Default:
            In case of 1 line: None
            In case of >1 line: "best" (auto-detect)
    legend_title: str, optional
        Default: None
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
        Default: None"""


class NotebookInteraction:
    """
    Calls the child's show()._repr_html_()
    for automatic display in Jupyter Notebooks
    """

    def _repr_html_(self):
        # look for show() method
        try:
            return self.show()._repr_html_()

        # fall back to plot() method
        except AttributeError:
            return self.plot()._repr_html_()


class Plot:
    def __init__(
        self,
        interactive,
        title=None,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        fig_size=None,
        legend_loc=None,
        legend_title=None,
    ):
        self.interactive = interactive
        self.count = 0
        self.legend_loc = legend_loc
        self.legend_title = legend_title

        # init plotly
        if self.interactive:
            self.fig = px.line()
            height, width = fig_size if fig_size is not None else (None, None)
            self.fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                legend_title=legend_title,
                height=height,
                width=width,
            )
            self.fig.update_xaxes(range=xlim)
            self.fig.update_yaxes(range=ylim)

        # init matplotlib
        else:
            self.fig, self.ax = plt.subplots(figsize=fig_size)

    def post_process(
        self,
        title=None,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        pty_update_layout=None,
        pty_custom_func=None,
        mpl_custom_func=None,
        save_fig=None,
    ):
        # PLOTLY
        if self.interactive:
            if pty_update_layout is not None:
                self.fig.update_layout(**pty_update_layout)
            if pty_custom_func is not None:
                self.fig = pty_custom_func(self.fig)

        # MATPLOTLIB
        else:
            self.ax.set_title(title)
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            if self.legend_loc or self.count > 1:
                self.legend_loc = self.legend_loc or "best"
                self.ax.legend(loc=self.legend_loc, title=self.legend_title)
            plt.tight_layout(pad=1.5)
            if mpl_custom_func is not None:
                self.fig, self.ax = mpl_custom_func(self.fig, self.ax)

        if save_fig is not None:
            self.save(save_fig)

    def save(self, path, **kwargs):

        # PLOTLY
        if self.interactive:

            # HTML
            if str(path)[-5:] == ".html":
                self.fig.write_html(path, **kwargs)

            # image
            else:
                self.fig.write_image(path, **kwargs)

        # MATPLOTLIB
        else:
            self.fig.savefig(
                path,
                face_color="white",
                bbox_inches="tight",
                **kwargs,
            )

        print("saved figure at {}".format(str(path)))

    def _repr_html_(self):
        if self.interactive:
            return self.fig._repr_html_()
        return self.fig.show()


class LinePlot(Plot):
    def add_trace(self, x, y=None, label=None, color=None):
        """
        Add a new histogram to the plot.

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
        """
        # input verification
        if y is None:
            y = x
            x = np.arange(len(y))
        self.count += 1

        # PLOTLY
        if self.interactive:
            self.fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=label,
                    marker_color=color,
                ),
            )

        # MATPLOTLIB
        else:
            self.ax.plot(x, y, label=label, color=color)


class HistPlot(Plot):
    def add_trace(self, x, bins=None, label=None, color=None):
        """
        Add a new histogram to the plot.

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
        """
        self.count += 1
        # PLOTLY
        if self.interactive:
            self.fig.add_trace(
                go.Histogram(
                    x=x,
                    name=label,
                    nbinsx=bins,
                    marker_color=color,
                ),
            )

        # MATPLOTLIB
        else:
            self.ax.hist(x, label=label, bins=bins, color=color)

    def post_process(self, *args, **kwargs):
        # overlay histograms by default
        if self.interactive:
            self.fig.update_layout(barmode="overlay")
        super().post_process(*args, **kwargs)


def _rewrite_docstring(doc, doc_insert):
    """
    Appends arguments to a docstring.

    Returns original docstring if REWRITE_DOCSTRING is set to False.

    Attempts:
    1. Search for [decorator.*?].
    2. Search for numpy-style "Parameters" block.
    3. Append to the end.

    Parameters
    ----------
    doc: str
        Original docstring.
    doc_insert: str,
        Docstring to insert.

    Returns
    -------
    str:
        Rewritten docstring
    """
    # check rewrite flag
    if not REWRITE_DOCSTRING:
        return doc

    # input check
    doc = "" if doc is None else doc

    # find indentation level of doc
    match = re.match("^\n?(?P<indent>[ \t]*)", doc)
    indent = match.group("indent") if match else ""

    # find indentation level of doc_insert
    match = re.match("^\n?(?P<indent>[ \t]*)", doc_insert)
    insert_indent = match.group("indent") if match else ""

    # search "[decorator parameters]"
    match = re.search(r"\n[ \t]*\[decorator.*?]", doc)
    if match:
        return re.sub(
            r"\n[ \t]*\[decorator.*?]",
            re.sub(
                r"\n{}".format(insert_indent),
                r"\n{}".format(indent),
                doc_insert,
            ),
            doc,
        )

    # test for numpy-style docstring
    docstring_query = (
        r"(?P<desc>(?:.*\n)*?)"  # desc
        r"(?P<params>(?P<indent>[ \t]*)Parameters[ \t]*"  # params header
        r"(?:\n(?!(?:[ \t]*\n)|(?:[ \t]*$)).*)*)"  # non-whitespace lines
        r"(?P<rest>(?:.*\n)*.*$)"
    )
    match = re.match(docstring_query, doc)
    if match:
        doc_parts = match.groupdict()
        return (
            doc_parts["desc"]
            + doc_parts["params"]
            + re.sub(
                r"\n{}".format(insert_indent),
                r"\n{}".format(doc_parts["indent"]),
                doc_insert,
            )
            + doc_parts["rest"]
        )

    # non-numpy docstring, just append in the end
    return doc + re.sub(
        r"\n{}".format(insert_indent), r"\n{}".format(indent), doc_insert
    )


def generic_plot_advanced(core, PlotClass):
    """
    Boilerplate code to advance Python plots.
    """
    # @functools.wraps(core)
    def wrapper(
        *args,
        interactive=True,
        title=None,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        fig_size=None,
        legend_loc=None,
        legend_title=None,
        pty_update_layout=None,
        pty_custom_func=None,
        mpl_custom_func=None,
        save_fig=None,
        **kwargs
    ):
        # preparation
        plot = PlotClass(
            interactive,
            title,
            xlabel,
            ylabel,
            xlim,
            ylim,
            fig_size,
            legend_loc,
            legend_title,
        )

        # execute core method
        core(*args, add_trace=plot.add_trace, **kwargs)

        # post-processing
        plot.post_process(
            title,
            xlabel,
            ylabel,
            xlim,
            ylim,
            pty_update_layout,
            pty_custom_func,
            mpl_custom_func,
            save_fig,
        )

        # return
        return plot

    # rewrite docstring
    wrapper.__doc__ = _rewrite_docstring(
        core.__doc__,
        DOC_INTERACTIVE + DOC_LINEPLOT,
    )

    return wrapper


def lineplot_advanced(core, *args_dec, **kwargs_dec):
    """
    Boilerplate code to advance Python line plots.
    """

    @functools.wraps(generic_plot_advanced)
    def wrapper(*args, **kwargs):
        return generic_plot_advanced(
            core,
            PlotClass=LinePlot,
            *args_dec,
            **kwargs_dec,
        )(*args, **kwargs)

    return wrapper


def histplot_advanced(core, *args_dec, **kwargs_dec):
    """
    Boilerplate code to advance Python line plots.
    """

    @functools.wraps(core)
    def wrapper(*args, **kwargs):
        return generic_plot_advanced(
            core,
            PlotClass=HistPlot,
            *args_dec,
            **kwargs_dec,
        )(*args, **kwargs)

    wrapper.__doc__ = _rewrite_docstring(core.__doc__, DOC_LINEPLOT)

    return wrapper


def lineplot_static(core, *args_dec, **kwargs_dec):
    """Enforce a static matplotlib plot upon lineplot_advanced"""

    def wrapper(*args, **kwargs):
        return lineplot_advanced(core, *args_dec, **kwargs_dec)(
            *args, interactive=False, **kwargs
        )

    wrapper.__doc__ = _rewrite_docstring(core.__doc__, DOC_LINEPLOT)

    return wrapper


def lineplot_dynamic(core, *args_dec, **kwargs_dec):
    """Enforce a dynamic plotly plot upon lineplot_advanced"""

    def wrapper(*args, **kwargs):
        return lineplot_advanced(core, *args_dec, **kwargs_dec)(
            *args, interactive=True, **kwargs
        )

    wrapper.__doc__ = _rewrite_docstring(core.__doc__, DOC_LINEPLOT)

    return wrapper
