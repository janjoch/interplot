# toolbox
Janosch's small Python code snippets making life a bit easier.

## Licence
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Docs
See [toolbox.janjo.ch](https://toolbox.janjo.ch) for the documentation.

## Demo
View on NBViewer: [![NBViewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/janjoch/toolbox/tree/main/demo/)

Try on Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/janjoch/toolbox/HEAD)

## Install
```pip install git+https://github.com/janjoch/toolbox#egg=toolbox```

### dev installation
1. Navigate to toolbox dir
2. ```pip install -e .```

## Modules

Note: This is just a **sneak peek**. Refer to `/demo` and ultimately `/toolbox` to see everything.

### arraytools
```python
# lowpass
>>> toolbox.arraytools.lowpass([1,2,3,4,5], n=3)
array([2, 3, 4])

# highpass
>>> toolbox.arraytools.highpass([1,2,3,4,5], n=3)
array([0, 0, 0])

# interpolate between two array elements
# aka "index with a floating point number"
>>> toolbox.arraytools.interp([0, 10, 20], 1.4)
14
```

### datetimeparser
Parse str timestamps to datetime.datetime.

```python
>>> import toolbox.datetimeparser as dtp

>>> dtp.ymd("2023-05-08T14:30:02Z")
datetime.datetime(2023, 5, 8, 14, 30, 2)

>>> dtp.dmy("31.12.2023")
datetime.datetime(2023, 12, 31, 0, 0)

>>> dtp.dmy("31.12.23 14:30:02.123", microsecond_shift=3)
datetime.datetime(2023, 12, 31, 14, 30, 2, 123000)

>>> dtp.dmy("The moonlanding happened on 20.07.1969 20:17:40")
datetime.datetime(1969, 7, 20, 20, 17, 40)

>>> dtp.time("It is now 14:30:12")
datetime.time(14, 30, 12)

>>> dtp.iso_tight("20230508T143002Z")
datetime.datetime(2023, 5, 8, 14, 30, 2)
```

### iter
Tools to iterate python objects.

```python
>>> from toolbox.iter import zip_smart, repeat

>>> for a, b, c, d, e in zip_smart(
...     ("A", "B", "C", "D"),
...     True,
...     [1, 2, 3, 4],
...     "always the same",
...     repeat((1, 2)),
... ):
...     print(a, b, c, d, e)
A True 1 always the same (1, 2)
B True 2 always the same (1, 2)
C True 3 always the same (1, 2)
D True 4 always the same (1, 2)
```

### plot
Create matplotlib/plotly hybrid plots with a few lines of code.

It combines the best of the matplotlib and the plotly worlds.
All the necessary boilerplate code is contained in this module.

Currently supported:
* line plots (scatter)
* line fills
* histograms
* heatmaps
* boxplot
* linear regression

* text annotations
* 2D subplots
* color cycling

```python
>>> toolbox.plot.line([0,4,6,7], [1,2,4,8])
[plotly figure]

>>> toolbox.plot.line([0,4,6,7], [1,2,4,8], interactive=False)
[matplotlib figure]

>>> toolbox.plot.line(
...     [0,4,6,7],
...     [1,2,4,8],
...     interactive=False,
...     xlim=(0, 10),
...     title="Matploblib Static Figure",
...     xlabel="X",
...     ylabel="Y",
...     save_fig="export/path/export.png",
...     dpi=300,
... )
[matplotlib figure, custom formatted]

>>> # toolbox.plot.NotebookInteraction searches the .show() or .plot() methods for Notebook representation
>>> class ReadTrace(toolbox.plot.NotebookInteraction):
... 
...     def __init__(self, file):
...         self.data = pd.read_csv(file)
...     
...     @toolbox.plot.magic_plot_preset(title="Automatic callback of show() in a Jupyter notebook")
...     def show(self, col="signal", fig=None):
...         fig.add_line(self.data[col])
...
... ReadTrace("path/to/data.csv")
[plotly figure is shown automatically]
```
