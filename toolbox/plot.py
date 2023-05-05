"""
Boilerplate code to advance Python plots.

It combines the best of the matplotlib and the plotly worlds.

Example:
```
>>> @toolbox.plot.lineplot_advanced
>>> def plot(*xy, add_trace=None, **kwargs):
>>>     add_trace(*xy, **kwargs)

>>> plot([0,4,6,7], [1,2,4,8])
[plotly figure]

>>> plot([0,4,6,7], [1,2,4,8],
>>>     interactive=False,
>>>     title="matploblib static figure",
>>>     xlabel="X",
>>>     ylabel="Y",
>>> )
[matplotlib figure]
```
"""


import re

import numpy as np

import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px


REWRITE_DOCSTRING = True

DOCSTRING_DECORATOR = """
    interactive: bool, optional
        Display an interactive plotly line plot
        instead of the default matplotlib figure.
        Default: False
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
                barmode="overlay",
            )
            self.fig.update_xaxes(range=xlim)
            self.fig.update_yaxes(range=ylim)

        # init matplotlib
        else:
            self.fig, self.ax = plt.subplots(figsize=fig_size)

    def add_line(self, x, y=None, label=None, color=None):
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

    def add_hist(self, x=None, y=None, bins=None, label=None, color=None):
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
        # input verification
        if x is None and y is None:
            raise ValueError("Either x or y must be defined.")
        if x is not None and y is not None:
            raise ValueError("x and y cannot be defined both.")

        bins_attribute = dict(nbinsx=bins) if y is None else dict(nbinsy=bins)
        self.count += 1

        # PLOTLY
        if self.interactive:
            self.fig.add_trace(
                go.Histogram(
                    x=x,
                    y=y,
                    name=label,
                    **bins_attribute,
                    marker_color=color,
                ),
            )

        # MATPLOTLIB
        else:
            self.ax.hist(x, label=label, bins=bins, color=color)

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


def _adjust_indent(indent_decorator, indent_core, docstring):
    """Adjust indentation of docstsrings."""
    return re.sub(
        r"\n{}".format(indent_decorator),
        r"\n{}".format(indent_core),
        docstring,
    )


def _rewrite_docstring(doc_core, doc_decorator=None, kwargs_remove=()):
    """
    Appends arguments to a DOCSTRING_DECORATOR.

    Returns original DOCSTRING_DECORATOR if REWRITE_DOCSTRING is set to False.

    Attempts:
    1. Search for [decorator.*?].
    2. Search for numpy-style "Parameters" block.
    3. Append to the end.

    Parameters
    ----------
    doc_core: str
        Original DOCSTRING_DECORATOR.
    doc_decorator: str,
        DOCSTRING_DECORATOR to insert.

    Returns
    -------
    str:
        Rewritten DOCSTRING_DECORATOR
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
                    # r"\n{0}(?:[a-zA-Z_]+(?:[ ]*,[ ]*)?)*?
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
        **kwargs,
    ):
        # preparation
        fig = Plot(
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
        core(*args, fig=fig, **kwargs)

        # post-processing
        fig.post_process(
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
