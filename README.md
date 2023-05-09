# toolbox
Janosch's small Python code snippets making life a bit easier.

## Licence
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Install
```pip install git+https://github.com/janjoch/toolbox#egg=toolbox```

## Modules

Note: This is just a sneak peek. Refer to /demo and ultimately /toolbox to see everything.

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

>>> dtp.dmy("31.12.2023")
datetime.datetime(2023, 12, 31, 0, 0)

>>> dtp.dmy("1.2.23 18:40:59.123456")
datetime.datetime(2023, 2, 1, 18, 40, 59, 123456)

>>> dtp.dmy("1.2.23 18:40:59.123", microsecond_shift=3)
datetime.datetime(2023, 2, 1, 18, 40, 59, 123000)

>>> dtp.ymd("Recording started on 2023-12-31 11:30:59.123456 in Zurich")
datetime.datetime(2023, 12, 31, 11, 30, 59, 123456)
```

### iter
Tools to iterate python objects.

```python
>>> from toolbox.iter import zip_smart

>>> for a, b, c in zip_smart(
>>>     ("A", "B", "C", "D"),
>>>     [1, 2, 3, 4],
>>>     True,
>>> ):
>>>     print(a, b, c)
A 1 True
B 2 True
C 3 True
D 4 True
```

### plot
Boilerplate code to advance Python plots.

It combines the best of the matplotlib and the plotly worlds.

```python
>>> @toolbox.plot.magic_plot
>>> def plot(*xy, fig, **kwargs):
>>>     fig.add_line(*xy, **kwargs)

>>> plot([0,4,6,7], [1,2,4,8])
[plotly figure]

>>> plot([0,4,6,7], [1,2,4,8],
>>>     interactive=False,
>>>     xlim=(0, 10),
>>>     title="Matploblib Static Figure",
>>>     xlabel="X",
>>>     ylabel="Y",
>>>     save_fig="export/path/export.png",
>>>     dpi=300,
>>> )
[matplotlib figure]

>>> class ReadTrace(toolbox.plot.NotebookInteraction):
>>> 
>>>     def __init__(self, file):
>>>         self.data = pd.read_csv(file)
>>>     
>>>     @toolbox.plot.magic_plot_preset(title="Automatic callback of show() in a Jupyter notebook")
>>>     def show(self, col="signal", fig=None):
>>>         fig.add_line(self.data[col])

>>> ReadTrace("path/to/data.csv")
[plotly figure is shown automatically]
```
