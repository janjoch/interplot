# interplot
Create `matplotlib/plotly` hybrid plots with a few lines of code.

It combines the best of the `matplotlib` and the `plotly` worlds through
a unified, flat API.
All the necessary boilerplate code is contained in this module.

Currently supported building blocks:

- scatter plots
    - `line`
    - `scatter`
    - `linescatter`
- histogram `hist`
- boxplot `boxplot`
- heatmap `heatmap`
- linear regression `regression`
- line fill `fill`
- annotations `text`

Supported
- 2D subplots
- automatic color cycling
- 3 different API modes
    - One line of code
        ```python
        >>> interplot.line([0,4,6,7], [1,2,4,8])
        [plotly line figure]

        >>> interplot.hist(np.random.normal(40, 8, 1000), interactive=False)
        [plotly hist figure]

        >>> interplot.boxplot(
        >>>     [
        >>>         np.random.normal(20, 5, 1000),
        >>>         np.random.normal(40, 8, 1000),
        >>>         np.random.normal(60, 5, 1000),
        >>>     ],
        >>> )
        [matplotlib boxplots]
        ```

    - Decorator to auto-initialize plots to use in your methods
        ```python
        >>> @interplot.magic_plot
        >>> def plot_my_data(fig=None):
        >>>     # import and process your data...
        >>>     data = np.random.normal(2, 3, 1000)
        >>>     # draw with the fig instance obtained from the decorator function
        >>>     fig.add_line(data, label="my data")
        >>>     fig.add_fill((0, 999), (-1, -1), (5, 5), label="sigma")

        >>> plot_my_data(title="My Recording")
        [plotly figure "My Recording"]

        >>> @interplot.magic_plot_preset(interactive=False, title="Preset Title")
        >>> def plot_my_data_preconfigured(fig=None):
        >>>     # import and process your data...
        >>>     data = np.random.normal(2, 3, 1000)
        >>>     # draw with the fig instance obtained from the decorator function
        >>>     fig.add_line(data, label="my data")
        >>>     fig.add_fill((0, 999), (-1, -1), (5, 5), label="sigma")

        >>> plot_my_data_preconfigured()
        [matplotlib figure "Preset Title"]
        ```

    - The ```interplot.Plot``` class for full control
        ```python
        >>> fig = interplot.Plot(
        >>>     interactive=True,
        >>>     title="Everything Under Control",
        >>>     fig_size=(800, 500),
        >>>     rows=1,
        >>>     cols=2,
        >>>     shared_yaxes=True,
        >>>     # ...
        >>> )
        >>> fig.add_hist(np.random.normal(1, 0.5, 1000), row=0, col=0)
        >>> fig.add_boxplot(
        >>>     [
        >>>         np.random.normal(20, 5, 1000),
        >>>         np.random.normal(40, 8, 1000),
        >>>         np.random.normal(60, 5, 1000),
        >>>     ],
        >>>     row=0,
        >>>     col=1,
        >>> )
        >>> # ...
        >>> fig.post_process()
        >>> fig.show()
        [plotly figure "Everything Under Control"]

        >>> fig.save("export/path/file.html")
        saved figure at export/path/file.html
        ```


## Resources

- **Documentation:** https://interplot.janjo.ch
- **Source Code:** https://github.com/janjoch/interplot
- **PyPI:** https://pypi.org/project/interplot/


## Licence
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


## Demo

View on `NBViewer`:
[![NBViewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/janjoch/interplot/tree/main/)


Try on `Binder`:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/janjoch/interplot/HEAD)


## Install
```pip install interplot```


### dev installation
1. ```git clone https://github.com/janjoch/interplot```
2. ```cd interplot```
2. ```pip install -e .```
