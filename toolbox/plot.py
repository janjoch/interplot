"""
Boilerplate code to advance Python line plots.

It combines the best of the matplotlib and the plotly worlds.
"""


import re

import numpy as np

import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px


REWRITE_DOCSTRING = True


class LinePlot:

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
        if(self.interactive):
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
    
    def add_trace(self, x, y=None, label=None):
        # input verification
        if(y is None):
            y = x
            x = np.arange(len(y))
        self.count += 1
        
        # PLOTLY
        if(self.interactive):
            self.fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=label,
                ),
            )

        # MATPLOTLIB
        else:
            self.ax.plot(x, y, label=label)
    
    def post_process(
        self,
        title=None,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
    ):
        if(not self.interactive):
            self.ax.set_title(title)
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            if(self.legend_loc or self.count > 1):
                self.legend_loc = self.legend_loc or "best"
                self.ax.legend(loc=self.legend_loc, title=self.legend_title)

    def save(self, path, **kwargs):

        # PLOTLY
        if(self.interactive):

            # HTML
            if(str(path)[-5:] == ".html"):
                self.fig.write_html(path, **kwargs)

            # image
            else:
                self.fig.write_image(path, **kwargs)

        # MATPLOTLIB
        else:
            self.fig.savefig(path, **kwargs)


    def _repr_html_(self):
        if(self.interactive):
            return self.fig._repr_html_()
        return self.fig.show()


def _rewrite_docstring(doc, doc_insert):
    # check rewrite flag
    if(not REWRITE_DOCSTRING):
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
    match = re.search(r"\[decorator parameters]", doc)
    if(match):
        return re.sub(
            r"\n[ \t]*\[decorator parameters]",
            re.sub(r"\n{}".format(insert_indent), r"\n{}".format(indent), doc_insert),
            doc,
        )
    
    # test for numpy-style docstring
    docstring_query = (
        r"(?P<desc>(?:.*\n)*?)" # desc
        r"(?P<params>(?P<indent>[ \t]*)Parameters[ \t]*" # params header
        r"(?:\n(?!(?:[ \t]*\n)|(?:[ \t]*$)).*)*)" # anything but a whitespace line
        r"(?P<rest>(?:.*\n)*.*$)"
    )
    match = re.match(docstring_query, doc)
    if(match):
        doc_parts = match.groupdict()
        return (
            doc_parts["desc"]
            + doc_parts["params"]
            + re.sub(r"\n{}".format(insert_indent), r"\n{}".format(doc_parts["indent"]), doc_insert)
            + doc_parts["rest"]
        )

    # non-numpy docstring, just append in the end
    return doc + re.sub(r"\n{}".format(insert_indent), r"\n{}".format(indent), doc_insert)


def advanced_lineplot(core):
    """
    Boilerplate code to advance Python line plots.
    """
    doc_insert = """
        interactive: bool, optional
            Display an interactive plotly line plot
            instead of the default matplotlib figure.
            Default: False
        title: str, optional
            Plot title.
        xlabel, ylabel: str, optional
            Axis labels.
        xlim, ylim: tuple of 2 numbers, optional
            Axis range limits.
        legend_title: str, optional
            Legend title.
        ...: ...
            ..."""

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
        **kwargs
    ):
        # preparation
        plot = LinePlot(
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

        )
        
        # return
        return plot

    # rewrite docstring
    wrapper.__doc__ = _rewrite_docstring(core.__doc__, doc_insert)

    return wrapper
