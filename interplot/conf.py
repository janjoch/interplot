"""
Modify the default behavior of the `interplot` package.

Example:

.. code-block:: python

   >>> # default behavior
   ... interplot.line(x, y)

.. raw:: html
     :file: ../source/plot_examples/default_sin.html


.. code-block:: python

   >>> # modify default behavior
   ... interplot.conf.INTERACTIVE = False
   ... interplot.conf.COLOR_CYCLE[0] = "#000000"
   ... interplot.conf.DPI = 150
   ... interplot.conf.MPL_FIG_SIZE = (400, 400)
   ...
   ... # user-modified default behavior
   ... interplot.line(x, y)


.. image:: plot_examples/default_sin.png
    :alt: [matplotlib plot of a sinus curve]
"""

INTERACTIVE = True
"""
Generate a `plotly` figure by default.
"""

COLOR_CYCLE = [  # optimised for color vision deficiencies
    "#006BA4",
    "#FF800E",
    "#ABABAB",
    "#595959",
    "#5F9ED1",
    "#C85200",
    "#898989",
    "#A2C8EC",
    "#FFBC79",
    "#CFCFCF",
]
"""
Colors to be cycled through by default.

This default cycle is optimised for color vision deficiencies.
"""

DPI = 100
"""
Figure resolution in dots per inch.
"""


PTY_FIG_SIZE = (None, 500)  # px, None for responsive
"""
Default figure size for the `plotly` backend, in px.
Use `None` for adaptive size.
"""

MPL_FIG_SIZE = (700, 450)  # px
"""
Default figure size for the `matplotlib` backend, in px.
"""


EXPORT_FORMAT = "png"
"""
Default export format.
"""

EXPORT_REPLACE = {
    "\n": "_",  # replace line breaks \\n with underscores
    r"<\s*br\s*/?\s*>": "_",  # replace line breaks <br> with underscores
    r"[ ]?[/|\\][ ]?": "_",  # replace slashes with underscores
    " ": "-",  # replace spaces with dashes
    "<.*?>": "",  # remove html tags
    "[@!?,:;+*%&()=#|'\"<>]": "",  # remove special characters
    r"\\": "_",  # replace backslashes with underscores
    r"\s": "_",  # replace any special whitespace with underscores
    # "^.*$": lambda s: s.group().lower()  # lowercase everything
}
"""
Replace characters in the figure title to use as default filename.

Use a dictionary with regex patterns as keys
and the (regex) replacement as values.

Default
-------
    - line breaks with underscores
    - slashes with underscores
    - spaces with dashes
    - special characters are removed
    - backslashes with underscores
    - any special whitespace with underscores

Optional
--------
    - lowercase everything
        ```ip.conf.EXPORT_REPLACE["^.*$"] = lambda s: s.group().lower()```
"""


def GLOBAL_CUSTOM_FUNC(fig):
    """
    Apply a custom function on figures on finishing the plot.

    Will be called in `interplot.Plot.post_process()`.

    When overriding this function, make sure it accepts `fig`.
    There is no need to return `fig`.

    Parameters
    ----------
    fig: interplot.Plot instance

    Examples
    --------
    >>> def global_custom_func(fig):
    ...     # do your customisations
    ...     fig.add_text(0, 0, "watermark", color="gray")
    ...
    ...     # also include default function
    ...     conf.GLOBAL_CUSTOM_FUNC(fig)
    ...
    ... fig = interplot.Plot(
    ...     global_custom_func=global_custom_func
    ... )
    ... fig.add_line(x, y)
    ... fig.post_process()  # global_custom_func will be executed here
    ... fig.show()
    """
    pass


def MPL_CUSTOM_FUNC(fig, ax):
    """
    Apply a custom function on matplotlib figures on finishing the plot.

    Will be called in `interplot.Plot.post_process()`.

    When overriding this function, make sure it accepts `fig` and `ax`
    and returns `fig` and `ax`.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The modified figure object.
    ax : matplotlib.axes.Axes
        The modified axes object.

    Examples
    --------
    >>> def mpl_custom_func(fig, ax):
    ...     # do your customisations
    ...     fig.do_stuff()
    ...     ax[0, 0].do_more()
    ...
    ...     # also include default function
    ...     fig, ax = conf.MPL_CUSTOM_FUNC(fig, ax)
    ...
    ...     return fig, ax
    ...
    ... fig = interplot.Plot(
    ...     interactive=False,
    ...     mpl_custom_func=mpl_custom_func
    ... )
    ... fig.add_line(x, y)
    ... fig.post_process()  # mpl_custom_func will be executed here
    ... fig.show()
    """
    return fig, ax


def PTY_CUSTOM_FUNC(fig):
    """
    Apply a custom function on plotly figures on finishing the plot.

    Will be called in `interplot.Plot.post_process()`.

    When overriding this function, make sure it accepts `fig`
    and returns `fig`.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure object.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The modified figure object.

    Examples
    --------
    >>> def pty_custom_func(fig):
    ...     # do your customisations
    ...     fig.do_stuff()
    ...
    ...     # also include default function
    ...     fig = conf.PTY_CUSTOM_FUNC(fig)
    ...
    ...     return fig
    ...
    ... fig = interplot.Plot(
    ...     interactive=True,
    ...     pty_custom_func=pty_custom_func
    ... )
    ... fig.add_line(x, y)
    ... fig.post_process()  # pty_custom_func will be executed here
    ... fig.show()
    """
    return fig


PTY_LEGEND_TOGGLEGROUP = True
"""
PLOTLY ONLY.
If True, elements with the same legend group will be toggled together
when clicking on a legend item.
"""


PTY_UPDATE_LAYOUT = dict()
"""
PLOTLY ONLY.

Pass keyword arguments to plotly's
`plotly.graph_objects.Figure.update_layout(**pty_update_layout)`

Default: None
"""


PTY_CONFIG = dict(
    displayModeBar=True,
    displaylogo=False,
)
"""
Modify the layout of the plotly figure.

See https://plotly.com/python/reference/layout/ for reference.
"""


PTY_LINE_STYLES = {
    "-": "solid",
    "--": "dash",
    "-.": "dashdot",
    ":": "dot",
    "solid": "solid",
    "dashed": "dash",
    "dashdot": "dashdot",
    "dotted": "dot",
}
"""
Mapping for line styles for `plotly`.
"""

MPL_LINE_STYLES = {value: key for key, value in PTY_LINE_STYLES.items()}
"""
Mapping for line styles for `matplotlib`.
"""

PTY_MARKERS = {
    ".": "circle",
    "s": "square",
    "D": "diamond",
    "P": "cross",
    "X": "x",
    "^": "triangle-up",
    "v": "triangle-down",
    "<": "triangle-left",
    ">": "triangle-right",
    "triangle-ne": "triangle-ne",
    "triangle-se": "triangle-se",
    "triangle-sw": "triangle-sw",
    "triangle-nw": "triangle-nw",
    "p": "pentagon",
    "h": "hexagon",
    "H": "hexagon2",
    "8": "octagon",
    "*": "star",
    "hexagram": "hexagram",
    "star-triangle-up": "star-triangle-up",
    "star-triangle-down": "star-triangle-down",
    "star-square": "star-square",
    "star-diamond": "star-diamond",
    "d": "diamond-tall",
    "diamond-wide": "diamond-wide",
    "hourglass": "hourglass",
    "bowtie": "bowtie",
    "circle-cross": "circle-cross",
    "circle-x": "circle-x",
    "square-cross": "square-cross",
    "square-x": "square-x",
    "diamond-cross": "diamond-cross",
    "diamond-x": "diamond-x",
    "+": "cross-thin",
    "x": "x-thin",
    "asterisk": "asterisk",
    "hash": "hash",
    "2": "y-up",
    "1": "y-down",
    "3": "y-left",
    "4": "y-right",
    "_": "line-ew",
    "|": "line-ns",
    "line-ne": "line-ne",
    "line-nw": "line-nw",
    6: "arrow-up",
    7: "arrow-down",
    4: "arrow-left",
    5: "arrow-right",
    "arrow-bar-up": "arrow-bar-up",
    "arrow-bar-down": "arrow-bar-down",
    "arrow-bar-left": "arrow-bar-left",
    "arrow-bar-right": "arrow-bar-right",
    "arrow": "arrow",
    "arrow-wide": "arrow-wide",
}
"""
Mapping for marker styles for `plotly`.
"""

PTY_MARKERS_LIST = list(PTY_MARKERS.values())
"""
Possible line styles for `plotly`.
"""

MPL_MARKERS = {value: key for key, value in PTY_MARKERS.items()}
"""
Mapping for marker styles for `matplotlib`.
"""
MPL_MARKERS.update(
    {  # next best matches
        "triangle-nw": "^",
        "triangle-ne": ">",
        "triangle-se": "v",
        "triangle-sw": "<",
        "hexagram": "*",
        "star-triangle-up": "^",
        "star-triangle-down": "v",
        "star-square": "s",
        "star-diamond": "D",
        "diamond-wide": "D",
        "hourglass": "d",
        "bowtie": "D",
        "circle-cross": "+",
        "circle-x": "x",
        "cross-thin": "+",
        "square-cross": "s",
        "square-x": "s",
        "diamond-cross": "D",
        "diamond-x": "D",
        "x-thin": "x",
        "hash": "*",
        "asterisk": "*",
        "line-ne": "|",
        "line-nw": "_",
        "arrow-bar-up": 6,
        "arrow-bar-down": 7,
        "arrow-bar-left": 4,
        "arrow-bar-right": 5,
        "arrow": 6,
        "arrow-wide": 6,
    }
)
MPL_MARKERS_LIST = list(MPL_MARKERS.values())
"""
Possible line styles for `matplotlib`.
"""

_REWRITE_DOCSTRING = True

_DOCSTRING_DECORATOR = """
    interactive: bool, default: True
        Display an interactive plotly line plot
        instead of the default matplotlib figure.
    rows, cols: int, default: 1
        Create a grid with `n` rows and `m` columns.
    title: str, default: None
        Plot title.
    xlabel, ylabel: str or str tuple, default: None
        Axis labels.

        Either one title for the entire axis or one for each row/column.
    xlim, ylim: tuple of 2 numbers or nested, default: None
        Axis range limits.

        In case of multiple rows/cols provide either:
            - a tuple
            - a tuple for each row
            - a tuple for each row containing a tuple for each column.
    xlog, ylog: bool or bool tuple, default: False
        Logarithmic scale for the axis.

        Either one boolean for the entire axis or one for each row/column.
    shared_xaxes, shared_yaxes: str, default: None
        Define how multiple subplots share there axes.

        Options:
            - "all" or True
            - "rows"
            - "columns" or "cols"
            - None or False
    column_widths, row_heights: tuple/list, default: None
        Ratios of the width/height dimensions in each column/row.
        Will be normalised to a sum of 1.
    fig_size: tuple of 2x float, optional
        Width, height in pixels.

        Default behavior defined by `conf.MPL_FIG_SIZE`
        and `conf.PTY_FIG_SIZE`.
        Plotly allows None for responsive size adapting to viewport.
    dpi: int, default: 100
        Plot resolution.
    legend_loc: str, optional
        MATPLOTLIB ONLY.

        Default:
            - In case of 1 line: None
            - In case of >1 line: "best" (auto-detect)
    legend_title: str, default: None
        MPL: Each subplot has its own legend, so a 2d list in the shape of
        the subplots may be provided.

        PTY: Just provide a `str`.
    legend_togglegroup: bool, default: False
        PLOTLY ONLY.

        Whether legend items with the same group will be toggled together
        when clicking on a legend item.
    color_cycle: list, optional
        Cycle through colors in the provided list.

        If left `None`, the default color cycle
        `conf.COLOR_CYCLE` will be used.
    save_fig: str or pathlib.Path, default: None
        Provide a path to export the plot.

        Possible formats: png, jpg, svg, html, ...

        The figure will only be saved on calling the instance's
        `.post_process()`.

        If a directory (or `True` for local directory) is provided,
        the filename will be automatically generated based on the title.

        An iterable of multiple paths / filenames may be provided. In this case
        the save command will be repeated for each element.
    save_format: str, default: None
        Provide a format for the exported plot, if not declared in `save_fig`.

        An iterable of multiple formats may be provided. In this case
        the save command will be repeated for each element.
    save_config: dict, default: None
        Plotly only.

        Pass config options to
        `plotly.graph_objects.Figure.write_html(config=save_config)`.
    global_custom_func: function, default: None
        Pass a function reference to further style the figures.

        >>> def global_custom_func(fig):
        ...     # do your customisations
        ...     fig.add_text(0, 0, "watermark", color="gray")
        ...
        ...     # also include default function
        ...     conf.GLOBAL_CUSTOM_FUNC(fig)
        ...
        ... fig = interplot.Plot(
        ...     global_custom_func=global_custom_func
        ... )
        ... fig.add_line(x, y)
        ... fig.post_process()  # global_custom_func will be executed here
        ... fig.show()
    mpl_custom_func: function, default: None
        MATPLOTLIB ONLY.

        Pass a function reference to further style the matplotlib graphs.
        Function must accept `fig, ax` and return `fig, ax`.

        Note: `ax` always has `row` and `col` coordinates, even if the
        plot is just 1x1.

        >>> def mpl_custom_func(fig, ax):
        ...     # do your customisations
        ...     fig.do_stuff()
        ...     ax[0, 0].do_more()
        ...
        ...     # also include default function
        ...     fig, ax = conf.MPL_CUSTOM_FUNC(fig, ax)
        ...
        ...     return fig, ax
        ...
        ... fig = interplot.Plot(
        ...     interactive=False,
        ...     mpl_custom_func=mpl_custom_func
        ... )
        ... fig.add_line(x, y)
        ... fig.post_process()  # mpl_custom_func will be executed here
        ... fig.show()
    pty_custom_func: function, default: None
        PLOTLY ONLY.

        Pass a function reference to further style the plotly graphs.
        Function must accept `fig` and return `fig`.

        >>> def pty_custom_func(fig):
        ...     # do your customisations
        ...     fig.do_stuff()
        ...
        ...     # also include default function
        ...     fig = conf.PTY_CUSTOM_FUNC(fig)
        ...
        ...     return fig
        ...
        ... fig = interplot.Plot(
        ...     interactive=True,
        ...     pty_custom_func=pty_custom_func
        ... )
        ... fig.add_line(x, y)
        ... fig.post_process()  # pty_custom_func will be executed here
        ... fig.show()
    pty_update_layout: dict, default: None
        PLOTLY ONLY.

        Pass keyword arguments to plotly's
        `plotly.graph_objects.Figure.update_layout(**pty_update_layout)`

        Default: None

    Returns
    -------
    `interplot.Plot` instance
"""
