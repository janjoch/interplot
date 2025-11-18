"""
This is the compressed version of interplot.

Unpacking:

>>> from interplot_minimal import unpack
... unpack()

Then either:
```bash
cd interplot_module
pip install -e .
```

or
1. move to the directory interplot_module
2. create a new notebook
3. >>> pip install -e .
4. if no imports possible: >>> pip install --no-dependencies -e .

or
1. move the inner `interplot` folder to the directory where you want to use it.


to use it:
>>> import interplot as ip
"""

import re
import io
from pathlib import Path


files_to_minimize = [
    "setup.py",
    "requirements.txt",
    "README.md",
    "minimize.py",
    "interplot",
]


def minimize(files: None):
    if files is None:
        files = files_to_minimize

    with io.open(Path(__file__), "r", encoding="utf-8") as f:
        minimizer = f.read()

    with io.open(Path("interplot_minimal.py"), "w+", encoding="utf-8") as f:

        f.write(minimizer)

        for file in files:
            minimize_path_object(file, f)

        f.write("\n#############\n" "#### END ####\n" "#############\n")


def minimize_path_object(path, f):
    if isinstance(path, str):
        path = Path(path)
    if path.is_dir():
        for path_ in path.iterdir():
            minimize_path_object(path_, f)
    else:
        if re.search(r"\.(py|txt|md)$", path.name):
            minimize_file(path, f)


def minimize_file(file, f):
    print("WRITING", file)

    path_str = str(file)

    f.write(
        "\n"
        + ("#" * (len(path_str) + 10))
        + "\n"
        + "#### "
        + path_str
        + " ####\n"
        + ("#" * (len(path_str) + 10))
    )

    with io.open(file, "r", encoding="utf-8") as f_:
        content = f_.read()
        content = re.sub(
            r"\n",
            r"\n#   ",
            content,
        )
        f.write("\n#   ")
        f.write(content)


def unpack(dir="interplot_module"):
    """Unpack the module to the current directory."""
    this_file = Path(__file__)
    with io.open(this_file, "r", encoding="utf-8") as file:
        content = file.read()
    dir = this_file.parent / dir
    dir.mkdir(parents=True, exist_ok=True)
    (dir / "interplot").mkdir(parents=True, exist_ok=True)

    for match in re.finditer(
        (
            r"^#### ([a-zA-Z0-9._/-]*?) ####\n####+?\n"  # file flag
            r"(.*?)"  # file content
            r"####+?\n"  # next file flag
        ),
        content,
        re.DOTALL | re.MULTILINE,
    ):
        path = dir / match.group(1).strip()
        filecontent = re.sub(
            r"\n#[ ]{0,3}",
            r"\n",
            match.group(2),
        )[4:]
        print("unpacking:", path)
        # print(filecontent[:100])
        with io.open(path, "w+", encoding="utf-8") as file:
            file.write(filecontent)

    print("Successfully unpacked interplot to current directory.\n")
    print("Now call:")
    print("cd interplot_module")
    print("pip install -e .")
    print("or")
    print("pip install --no-dependencies -e .")
    print("")
    print("If pip installs are not possible,")
    print("just move the inner `interplot` folder")
    print("to the directory where you want to use it.")


if __name__ == "__main__":
    minimize(None)

    print("Successfully minimized interplot to interplot_minimal.py.")

    # unpack()

##################
#### setup.py ####
##################
#   from setuptools import setup
#   
#   with open("requirements.txt", "r", encoding="utf-8") as f:
#       requirements = f.read().splitlines()
#   
#   # read the contents of your README file
#   from pathlib import Path
#   
#   this_directory = Path(__file__).parent
#   long_description = (this_directory / "README.md").read_text()
#   
#   setup(
#       name="interplot",
#       version="1.1.0",
#       description=(
#           "Create matplotlib and plotly charts with the same few lines of code."
#       ),
#       long_description=long_description,
#       long_description_content_type="text/markdown",
#       url="https://github.com/janjoch/interplot",
#       author="Janosch Jörg",
#       author_email="janjo@duck.com",
#       license="GPL v3",
#       packages=["interplot"],
#       install_requires=requirements,
#       classifiers=[
#           "Development Status :: 5 - Production/Stable",
#           "Intended Audience :: Science/Research",
#           "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
#           "Operating System :: MacOS :: MacOS X",
#           "Operating System :: Microsoft :: Windows",
#           "Programming Language :: Python :: 3",
#           "Programming Language :: Python :: 3.6",
#           "Programming Language :: Python :: 3.7",
#           "Programming Language :: Python :: 3.8",
#           "Framework :: Jupyter :: JupyterLab :: 4",
#           "Framework :: Matplotlib",
#           "Topic :: Scientific/Engineering :: Visualization",
#       ],
#   )
#   
##########################
#### requirements.txt ####
##########################
#   numba
#   numpy
#   pandas
#   matplotlib
#   plotly
#   kaleido
#   scipy
#   xarray
#   
###################
#### README.md ####
###################
#   # interplot
#   
#   [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/janjoch/interplot/HEAD) [![NBViewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/janjoch/interplot/tree/main/demo/)
#   
#   Create `matplotlib` and `plotly` charts with the same few lines of code.
#   
#   It combines the best of the `matplotlib` and the `plotly` worlds through a unified, flat API.
#   
#   Switch between `matplotlib` and `plotly` with the single keyword `interactive`. All the necessary boilerplate code to translate between the packages is contained in this module.
#   
#   Currently supported building blocks:
#   
#   - scatter plots
#       - `line`
#       - `scatter`
#       - `linescatter`
#   - bar charts `bar`
#   - histogram `hist`
#   - boxplot `boxplot`
#   - heatmap `heatmap`
#   - linear regression `regression`
#   - line fill `fill`
#   - annotations `text`
#   
#   Supported
#   - 2D subplots
#   - automatic color cycling
#   - 3 different API modes
#       - One line of code
#           ```python
#           >>> interplot.line([0,4,6,7], [1,2,4,8])
#           [plotly line figure]
#   
#           >>> interplot.hist(np.random.normal(40, 8, 1000), interactive=False)
#           [matplotlib hist figure]
#   
#           >>> interplot.boxplot(
#           >>>     [
#           >>>         np.random.normal(20, 5, 1000),
#           >>>         np.random.normal(40, 8, 1000),
#           >>>         np.random.normal(60, 5, 1000),
#           >>>     ],
#           >>> )
#           [plotly boxplots]
#           ```
#   
#       - Decorator to auto-initialize plots to use in your methods
#           ```python
#           >>> @interplot.magic_plot
#           >>> def plot_my_data(fig=None):
#           >>>     # import and process your data...
#           >>>     data = np.random.normal(2, 3, 1000)
#           >>>     # draw with the fig instance obtained from the decorator function
#           >>>     fig.add_line(data, label="my data")
#           >>>     fig.add_fill((0, 999), (-1, -1), (5, 5), label="sigma")
#   
#           >>> plot_my_data(title="My Recording")
#           [plotly figure "My Recording"]
#   
#           >>> @interplot.magic_plot_preset(interactive=False, title="Preset Title")
#           >>> def plot_my_data_preconfigured(fig=None):
#           >>>     # import and process your data...
#           >>>     data = np.random.normal(2, 3, 1000)
#           >>>     # draw with the fig instance obtained from the decorator function
#           >>>     fig.add_line(data, label="my data")
#           >>>     fig.add_fill((0, 999), (-1, -1), (5, 5), label="sigma")
#   
#           >>> plot_my_data_preconfigured()
#           [matplotlib figure "Preset Title"]
#           ```
#   
#       - The `interplot.Plot` class for full control
#           ```python
#           >>> fig = interplot.Plot(
#           >>>     interactive=True,
#           >>>     title="Everything Under Control",
#           >>>     fig_size=(800, 500),
#           >>>     rows=1,
#           >>>     cols=2,
#           >>>     shared_yaxes=True,
#           >>>     # ...
#           >>> )
#           >>> fig.add_hist(np.random.normal(1, 0.5, 1000), row=0, col=0)
#           >>> fig.add_boxplot(
#           >>>     [
#           >>>         np.random.normal(20, 5, 1000),
#           >>>         np.random.normal(40, 8, 1000),
#           >>>         np.random.normal(60, 5, 1000),
#           >>>     ],
#           >>>     row=0,
#           >>>     col=1,
#           >>> )
#           >>> # ...
#           >>> fig.post_process()
#           >>> fig.show()
#           [plotly figure "Everything Under Control"]
#   
#           >>> fig.save("export/path/file.html")
#           saved figure at export/path/file.html
#           ```
#   
#   
#   ## Resources
#   
#   - **Documentation:** https://interplot.janjo.ch
#   - **Demo Notebooks:** https://nbviewer.org/github/janjoch/interplot/tree/main/demo/
#   - **Source Code:** https://github.com/janjoch/interplot
#   - **PyPI:** https://pypi.org/project/interplot/
#   
#   
#   ## Licence
#   [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
#   
#   
#   ## Demo
#   
#   View on `NBViewer`:
#   [![NBViewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/janjoch/interplot/tree/main/demo/)
#   
#   
#   Try on `Binder`:
#   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/janjoch/interplot/HEAD)
#   
#   
#   ## Install
#   ```pip install interplot```
#   
#   ### install development branch
#   ```pip install git+https://github.com/janjoch/interplot.git@development```
#   
#   ### active development installation
#   1. ```git clone https://github.com/janjoch/interplot```
#   2. ```cd interplot```
#   2. ```pip install -e .```
#   
#   
#   ## Contribute
#   
#   Ideas, bug reports/fixes, feature requests and code submissions are very welcome! Please write to [janjo@duck.com](mailto:janjo@duck.com) or directly into a pull request.
#   
#####################
#### minimize.py ####
#####################
#   """
#   This is the compressed version of interplot.
#   
#   Unpacking:
#   
#   >>> from interplot_minimal import unpack
#   ... unpack()
#   
#   Then either:
#   ```bash
#   cd interplot_module
#   pip install -e .
#   ```
#   
#   or
#   1. move to the directory interplot_module
#   2. create a new notebook
#   3. >>> pip install -e .
#   4. if no imports possible: >>> pip install --no-dependencies -e .
#   
#   or
#   1. move the inner `interplot` folder to the directory where you want to use it.
#   
#   
#   to use it:
#   >>> import interplot as ip
#   """
#   
#   import re
#   import io
#   from pathlib import Path
#   
#   
#   files_to_minimize = [
#       "setup.py",
#       "requirements.txt",
#       "README.md",
#       "minimize.py",
#       "interplot",
#   ]
#   
#   
#   def minimize(files: None):
#       if files is None:
#           files = files_to_minimize
#   
#       with io.open(Path(__file__), "r", encoding="utf-8") as f:
#           minimizer = f.read()
#   
#       with io.open(Path("interplot_minimal.py"), "w+", encoding="utf-8") as f:
#   
#           f.write(minimizer)
#   
#           for file in files:
#               minimize_path_object(file, f)
#   
#           f.write("\n#############\n" "#### END ####\n" "#############\n")
#   
#   
#   def minimize_path_object(path, f):
#       if isinstance(path, str):
#           path = Path(path)
#       if path.is_dir():
#           for path_ in path.iterdir():
#               minimize_path_object(path_, f)
#       else:
#           if re.search(r"\.(py|txt|md)$", path.name):
#               minimize_file(path, f)
#   
#   
#   def minimize_file(file, f):
#       print("WRITING", file)
#   
#       path_str = str(file)
#   
#       f.write(
#           "\n"
#           + ("#" * (len(path_str) + 10))
#           + "\n"
#           + "#### "
#           + path_str
#           + " ####\n"
#           + ("#" * (len(path_str) + 10))
#       )
#   
#       with io.open(file, "r", encoding="utf-8") as f_:
#           content = f_.read()
#           content = re.sub(
#               r"\n",
#               r"\n#   ",
#               content,
#           )
#           f.write("\n#   ")
#           f.write(content)
#   
#   
#   def unpack(dir="interplot_module"):
#       """Unpack the module to the current directory."""
#       this_file = Path(__file__)
#       with io.open(this_file, "r", encoding="utf-8") as file:
#           content = file.read()
#       dir = this_file.parent / dir
#       dir.mkdir(parents=True, exist_ok=True)
#       (dir / "interplot").mkdir(parents=True, exist_ok=True)
#   
#       for match in re.finditer(
#           (
#               r"^#### ([a-zA-Z0-9._/-]*?) ####\n####+?\n"  # file flag
#               r"(.*?)"  # file content
#               r"####+?\n"  # next file flag
#           ),
#           content,
#           re.DOTALL | re.MULTILINE,
#       ):
#           path = dir / match.group(1).strip()
#           filecontent = re.sub(
#               r"\n#[ ]{0,3}",
#               r"\n",
#               match.group(2),
#           )[4:]
#           print("unpacking:", path)
#           # print(filecontent[:100])
#           with io.open(path, "w+", encoding="utf-8") as file:
#               file.write(filecontent)
#   
#       print("Successfully unpacked interplot to current directory.\n")
#       print("Now call:")
#       print("cd interplot_module")
#       print("pip install -e .")
#       print("or")
#       print("pip install --no-dependencies -e .")
#       print("")
#       print("If pip installs are not possible,")
#       print("just move the inner `interplot` folder")
#       print("to the directory where you want to use it.")
#   
#   
#   if __name__ == "__main__":
#       minimize(None)
#   
#       print("Successfully minimized interplot to interplot_minimal.py.")
#   
#       # unpack()
#   
#################################
#### interplot/arraytools.py ####
#################################
#   """Work with 1D arrays."""
#   
#   import math
#   
#   import numpy as np
#   
#   import scipy.stats as sp_stats
#   
#   import pandas as pd
#   
#   import numba as nb
#   
#   from . import plot
#   
#   
#   LISTLIKE_TYPES = (tuple, list, np.ndarray, pd.core.series.Series)
#   
#   
#   def _new_pd_index(series, n):
#       return series.index[(n - 1) // 2 : -(n // 2)]
#   
#   
#   def lowpass(data, n=101, new_index=None):
#       """
#       Average symetrically over n data points.
#   
#       Accepts numpy arrays, lists and pandas Series.
#   
#       Parameters
#       ----------
#       data: array-like
#           np.array, list, tuple or pd.Series to filter.
#       n: int, optional
#           Number of data points to average over.
#       new_index: list-like, optional
#           If a pandas Series is provided as data,
#           use this new_index.
#   
#       Returns
#       -------
#       np.ndarray or pd.Series
#       """
#       # input verification
#       if n == 1:
#           return data
#   
#       # pandas Series
#       if isinstance(data, pd.core.series.Series):
#           new_index = _new_pd_index(data, n) if new_index is None else new_index
#           return pd.Series(
#               lowpass_core(np.array(data), n),
#               index=new_index,
#           )
#   
#       # np.array, list or tuple
#       if isinstance(data, LISTLIKE_TYPES):
#           return lowpass_core(np.array(data), n)
#   
#       # fail if no supported data type
#       raise TypeError("Data type not supported:\n{}".format(type(data)))
#   
#   
#   @nb.jit(nopython=True, parallel=True)
#   def lowpass_core(data, n):
#       """
#       Average symetrically over n data points.
#   
#       Parameters
#       ----------
#       data: np.array
#           Array to filter.
#       n: int, optional
#           Number of data points to average over.
#   
#       Returns
#       -------
#       np.ndarray
#       """
#       size = data.size - n + 1
#   
#       array = np.empty(size, dtype=data.dtype)
#       for i in nb.prange(size):
#           array[i] = np.mean(data[i : i + n])
#   
#       return array
#   
#   
#   def highpass(
#       data,
#       n=101,
#       new_index=None,
#   ):
#       """
#       Filter out low-frequency drift.
#   
#       Offsets each datapoint by the average of the surrounding n data points.
#       N must be odd.
#   
#       Parameters
#       ----------
#       data: array-like
#           np.array, list, tuple or pd.Series to filter.
#       n: int, optional
#           Number of data points to average over.
#       new_index: list-like, optional
#           If a pandas Series is provided as data,
#           use this new_index.
#   
#       Returns
#       -------
#       np.ndarray or pd.Series
#       """
#       # input verification
#       if n == 1:
#           return data
#       if n % 2 == 0:
#           raise ValueError("n must be odd!")
#   
#       # pandas Series
#       if isinstance(data, pd.core.series.Series):
#           new_index = _new_pd_index(data, n) if new_index is None else new_index
#           return data[((n - 1) // 2) : -((n - 1) // 2)] - lowpass(
#               np.array(data), n, new_index=new_index
#           )
#   
#       # np.array, list or tuple
#       if isinstance(data, LISTLIKE_TYPES):
#           return np.array(data[((n - 1) // 2) : -((n - 1) // 2)]) - lowpass_core(
#               np.array(data), n
#           )
#   
#       # fail if no supported data type
#       raise TypeError("Data type not supported:\n{}".format(type(data)))
#   
#   
#   def interp(array, pos):
#       """
#       Linearly interpolate between neighboring indexes.
#   
#       Parameters
#       ----------
#       array: 1D list-like
#       pos: float
#   
#       Returns
#       -------
#       float
#           interpolated value
#       """
#       if math.floor(pos) == math.ceil(pos):
#           return array[int(pos)]
#   
#       i = math.floor(pos)
#       w = pos - i
#       d = array[i + 1] - array[i]
#       return array[i] + w * d
#   
#   
#   class LinearRegression(plot.NotebookInteraction):
#       def __init__(
#           self,
#           x,
#           y,
#           p=0.05,
#           linspace=101,
#       ):
#           """
#           Model regression and its parameters.
#   
#           Parameters
#           ----------
#           x, y: array-like
#               Data points.
#           p: float, optional
#               p-value.
#               Default: 0.05
#           linspace: int, optional
#               Number of data points for linear regression model
#               and conficence and prediction intervals.
#               Default: 101
#   
#           The instance will provide the following data attributes:
#               x, y: array-like
#                   The original data.
#               p: float
#                   The original p-value.
#               poly: np.ndarray of 2x float
#                   Polynomial coefficients.
#                   [a, b] -> a * x + b.
#               cov: float
#                   Covariance matrix of the polynomial coefficient estimates.
#                   See for poly, cov:
#                   https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
#               y_model: np.ndarray
#                   The regression modeled y values for the input x
#               n: int
#                   Number of observations
#               m: int
#                   Number of parameters
#               dof: int
#                   Degree of freedoms
#                   n - m
#               t: float
#                   t statistics
#               ...
#   
#           Code derived from pylang's StackOverflow post:
#           https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
#           """
#           self.x = np.array(x)
#           self.y = np.array(y)
#           self.p = p
#           self.is_linreg = True
#   
#           # parameters and covariance from of the fit of 1-D polynom.
#           self.poly, self.cov = np.polyfit(
#               x,
#               y,
#               1,
#               cov=True,
#           )
#           self.y_model = (
#               # model using the fit parameters; NOTE: parameters here are
#               np.polyval(
#                   self.poly,
#                   x,
#               )
#           )
#   
#           self.n = y.size  # number of observations
#           self.m = self.poly.size  # number of parameters
#           self.dof = self.n - self.m  # degrees of freedom
#           self.t = sp_stats.t.ppf(  # t-statistic; used for CI and PI bands
#               1 - p / 2,
#               self.n - self.m,
#           )
#   
#           # Estimates of Error in Data/Model
#           self.resid = (
#               y - self.y_model
#           )  # residuals; diff. actual data from predicted values
#           self.chi2 = np.sum(  # chi-squared; estimates error in data
#               (self.resid / self.y_model) ** 2
#           )
#           self.chi2_red = (
#               self.chi2 / self.dof
#           )  # reduced chi-squared; measures goodness of fit
#           self.s_err = np.sqrt(  # standard deviation of the error
#               np.sum(self.resid**2) / self.dof
#           )
#   
#           self.x2 = np.linspace(np.min(self.x), np.max(self.x), linspace)
#           self.y2 = np.polyval(self.poly, self.x2)
#   
#           # confidence interval
#           self.ci = (
#               self.t
#               * self.s_err
#               * np.sqrt(
#                   1 / self.n
#                   + (self.x2 - np.mean(self.x)) ** 2
#                   / np.sum((self.x - np.mean(self.x)) ** 2)
#               )
#           )
#   
#           # prediction interval
#           self.pi = (
#               self.t
#               * self.s_err
#               * np.sqrt(
#                   1
#                   + 1 / self.n
#                   + (self.x2 - np.mean(self.x)) ** 2
#                   / np.sum((self.x - np.mean(self.x)) ** 2)
#               )
#           )
#   
#       @plot.magic_plot
#       def plot(
#           self,
#           fig=None,  # inserted by plot.magic_plot decorator
#           plot_ci=True,
#           plot_pi=True,
#           label=None,
#           label_data="data",
#           label_reg="regression",
#           label_ci="confidence interval",
#           label_pi="prediction interval",
#           line_style_reg="solid",
#           line_style_pi="dotted",
#           color=None,
#           color_data=None,
#           color_reg=None,
#           color_ci=None,
#           color_pi=None,
#           kwargs_data=None,
#           kwargs_reg=None,
#           kwargs_ci=None,
#           kwargs_pi=None,
#           **kwargs,
#       ):
#           """
#           Plot the correlation analysis.
#   
#           Parameters
#           ----------
#           plot_ci, plot_pi: bool, optional
#               Plot the confidence and prediction intervals.
#               Default: True
#           label: str or interplot.Labelgroup, optional
#           label_data, label_reg, label_ci, label_pi: str or callable, optional
#               Trace labels.
#           color_data, color_reg, color_ci, color_pi: str, optional
#               Trace color.
#               Can be hex, rgb(a) or any named color that is understood
#               by matplotlib.
#               Default: None
#               In the default case, Plot will cycle through COLOR_CYCLE.
#           kwargs_data, kwargs_reg, kwargs_ci, kwargs_pi: dict, optional
#               Keyword arguments to pass to corresponding figure element.
#           **kwargs: optional
#               Keyword arguments to pass to each figure element.
#   
#           Returns
#           -------
#           plot.Plot instance
#           """
#           # input validation
#           if kwargs_data is None:
#               kwargs_data = dict()
#           if kwargs_reg is None:
#               kwargs_reg = dict()
#           if kwargs_ci is None:
#               kwargs_ci = dict()
#           if kwargs_pi is None:
#               kwargs_pi = dict()
#           if color is None:
#               color = fig.get_cycle_color()
#   
#           if not isinstance(label, plot.LabelGroup):
#               row = kwargs.get("row", 0)
#               col = kwargs.get("col", 0)
#               group_id = "regression_{}_{}_{}".format(
#                   row,
#                   col,
#                   fig.element_count[row, col],
#               )
#               label = plot.LabelGroup(
#                   group_id=group_id,
#                   group_title="Regression" if label is None else label,
#               )
#   
#           # data points
#           fig.add_scatter(
#               self.x,
#               self.y,
#               label=(label_data if callable(label_data) else label.element(label_data)),
#               color=color if color_data is None else color_data,
#               **kwargs_data,
#               **kwargs,
#           )
#   
#           # regression line
#           fig.add_line(
#               self.x2,
#               self.y2,
#               line_style=line_style_reg,
#               label=(label_reg if callable(label_reg) else label.element(label_reg)),
#               color=color if color_reg is None else color_reg,
#               **kwargs_reg,
#               **kwargs,
#           )
#   
#           if plot_ci:
#               fig.add_fill(
#                   self.x2,
#                   self.y2 - self.ci,
#                   self.y2 + self.ci,
#                   label=(
#                       label_ci
#                       if callable(label_ci)
#                       else plot.LabelGroup(
#                           group_id=label.group_id,
#                           default_label=label_ci,
#                       )
#                   ),
#                   color=color if color_ci is None else color_ci,
#                   **kwargs_ci,
#                   **kwargs,
#               )
#   
#           if plot_pi:
#               fig.add_line(
#                   self.x2,
#                   self.y2 + self.pi,
#                   label=(label_pi if callable(label_pi) else label.element(label_pi)),
#                   line_style=line_style_pi,
#                   color=color if color_pi is None else color_pi,
#                   **kwargs_pi,
#                   **kwargs,
#               )
#               fig.add_line(
#                   self.x2,
#                   self.y2 - self.pi,
#                   label=(
#                       label_pi
#                       if callable(label_pi)
#                       else label.element(label_pi, show=False)
#                   ),
#                   line_style=line_style_pi,
#                   show_legend=False,
#                   color=color if color_pi is None else color_pi,
#                   **kwargs_pi,
#                   **kwargs,
#               )
#   
###########################
#### interplot/iter.py ####
###########################
#   """Tools to iterate Python objects."""
#   
#   from warnings import warn
#   from datetime import datetime
#   from types import GeneratorType
#   
#   from numpy import ndarray as np_ndarray
#   from pandas.core.series import Series as pd_Series
#   
#   
#   ITERABLE_TYPES = (
#       tuple,
#       list,
#       dict,
#       np_ndarray,
#       pd_Series,
#       range,
#       GeneratorType,
#   )
#   NON_ITERABLE_TYPES = (str,)
#   CUSTOM_DIGESTION = ((dict, (lambda dct: [elem for _, elem in dct.items()])),)
#   
#   MUTE_STRICT_ZIP_WARNING = False
#   
#   
#   def repeat(arg, unpack_nozip=True):
#       """
#       A generator that always returns `arg`.
#   
#       Parameters
#       ----------
#       arg
#           Any arbitraty object.
#       unpack_nozip: bool, default: True
#           Deprecation: Instead of `NoZip`, use `repeat`.
#   
#           Unpack objects protected by `NoZip`.
#   
#       Returns
#       -------
#       generator function
#           which always returns arg.
#   
#       Examples
#       --------
#       >>> for a, b, c, d, e in interplot.zip_smart(
#       ...     ("A", "B", "C", "D"),
#       ...     True,
#       ...     [1, 2, 3, 4, 5],  # notice the extra element won't be unpacked
#       ...     "always the same",
#       ...     interplot.repeat((1, 2)),
#       ... ):
#       ...     print(a, b, c, d, e)
#       A True 1 always the same (1, 2)
#       B True 2 always the same (1, 2)
#       C True 3 always the same (1, 2)
#       D True 4 always the same (1, 2)
#       """
#       if unpack_nozip and isinstance(arg, NoZip):
#           arg = arg()
#       while True:
#           yield arg
#   
#   
#   def zip_smart(*iterables, unpack_nozip=True, strict=False):
#       """
#       Iterate over several iterables in parallel,
#       producing tuples with an item from each one.
#   
#       Like Python's builtin `zip` function,
#       but if an argument is not iterable, it will be repeated each iteration.
#   
#       Exception: strings will be repeated by default.
#       Override `interplot.conf.NON_ITERABLE_TYPES` to change this behavior.
#   
#       To be iterated, the item needs to have an `__iter__` attribute.
#       Otherwise, it will be repeated.
#   
#       Pay attention with the `strict` parameter:
#           - only working with Python >=3.10
#               - will be ignored on Python <3.10
#               - raises a warning if ignored,
#                 unless `interplot.conf.MUTE_STRICT_ZIP_WARNING` is set to `False`
#           - always raises an error if an item is repeated using \
#           `interplot.repeat()`, since the generator is endless.
#   
#       Parameters
#       ----------
#       *iterables: misc
#           Elements to iterate or repeat.
#       unpack_nozip: bool, default: True
#           Unpack a `NoZip`-wrapped iterable.
#       strict: bool, default: True
#           Fail if iterables are not the same length.
#   
#           Warning: Not supported in Python <3.10.
#   
#       Returns
#       -------
#       zip object
#           Use it as you would use `zip`
#   
#       Examples
#       --------
#       >>> for a, b, c, d, e in interplot.zip_smart(
#       ...     ("A", "B", "C", "D"),
#       ...     True,
#       ...     [1, 2, 3, 4, 5],  # notice the extra element won't be unpacked
#       ...     "always the same",
#       ...     interplot.repeat((1, 2)),
#       ... ):
#       ...     print(a, b, c, d, e)
#       A True 1 always the same (1, 2)
#       B True 2 always the same (1, 2)
#       C True 3 always the same (1, 2)
#       D True 4 always the same (1, 2)
#       """
#       iterables = list(iterables)
#   
#       for i, arg in enumerate(iterables):
#           if not hasattr(arg, "__iter__") or isinstance(arg, NON_ITERABLE_TYPES):
#               iterables[i] = repeat(arg, unpack_nozip=unpack_nozip)
#   
#       try:
#           return zip(*iterables, strict=strict)
#   
#       # strict mode not implemented in Python<3.10
#       except TypeError:
#           if strict:
#               if not MUTE_STRICT_ZIP_WARNING:
#                   warn(
#                       "zip's strict mode not supported in Python<3.10.\n\n"
#                       "Falling back to non-strict mode."
#                   )
#           return zip(*iterables)
#   
#   
#   def sum_nested(
#       inp,
#       iterable_types=None,
#       depth=-1,
#       custom_digestion=None,
#   ):
#       """
#       Add up all values in iterable objects.
#   
#       Nested structures are added up recursively.
#       In dictionaries, only the values are used.
#   
#       Parameters
#       ----------
#       inp: iterable
#           Object to iterate over.
#   
#           If it is not a iterable type, the object itself is returned.
#       iterable_types: tuple of types, optional
#           If iterable is one of these types, hand to zip() directly without
#           repeating.
#   
#           Default: (tuple, list, np.ndarray, pandas.Series)
#       depth: int, optional
#           Maximum depth to recurse.
#           Set to -1 to recurse infinitely.
#   
#           Default -1.
#       custom_digestion: tuple of tuples, optional
#           Each element of the tuple must be a tuple of the following structure:
#   
#           ( \
#               type or tuple of types, \
#               lambda function to digest the elements,
#           )
#   
#           The result of the lambda function will then be treated
#           like the new type.
#   
#           By default, `interplot.CUSTOM_DIGESTION` will be used:
#           Dicts will be digested to a list of their values.
#   
#       Returns
#       -------
#       sum
#       """
#       # input verification
#       depth = -1 if depth is None else depth
#       iterable_types = iterable_types or ITERABLE_TYPES
#       custom_digestion = custom_digestion or CUSTOM_DIGESTION
#   
#       # custom digestion
#       for type_, lambda_ in custom_digestion:
#           if isinstance(inp, type_):
#               inp = lambda_(inp)
#   
#       # if is not iterable, return as-is
#       if not isinstance(inp, ITERABLE_TYPES):
#           return inp
#   
#       # check recursion level
#       if depth is None or depth == 0:
#           raise TypeError(
#               (
#                   "Iterable type detected, but recursion has reached "
#                   "its maximum depth.\n\n"
#                   "Element:\n{}\n\n"
#                   "Type:\n{}"
#               ).format(str(inp), str(type(inp)))
#           )
#   
#       # iterate
#       val = 0
#       for elem in inp:
#   
#           # sum_nested only returns non-iterable types
#           val += sum_nested(
#               elem,
#               iterable_types=iterable_types,
#               depth=depth - 1,
#               custom_digestion=custom_digestion,
#           )
#   
#       return val
#   
#   
#   def filter_nozip(iterable, no_iter_types=None, depth=0, length=(2,)):
#       """
#       Prevent certain patterns from being unpacked in `interplot.zip_smart`.
#   
#       Wraps fixed patterns into `interplot.repeat`, so that they are not unpacked
#       in `interplot.zip_smart`.
#   
#       By default, iterables of length 2 containing only `float`, `int`, and
#       `datetime` objects will not be unpacked.
#   
#       Example
#       -------
#       >>> A = [1, 2]
#       ... B = ["a", "b"]
#       ...
#       ... for a, b in interplot.zip_smart(
#       ...     interplot.filter_nozip(A),
#       ...     interplot.filter_nozip(B),
#       ... ):
#       ...     print(a, b)
#   
#       [1, 2] a
#       [1, 2] b
#   
#       Parameters
#       ----------
#       iterable: Any
#           The object to potentially iterate.
#       no_iter_types, tuple, optional
#           If only these types are found in the iterable, it will not be unpacked,
#           given it has the correct length.
#   
#           Default: (`float`, `int`, `datetime`)
#       depth: int, default: 0
#           Maximum depth to recurse.
#   
#           Depth 0 will only check the first level,
#           depth 1 will check two levels, ...
#   
#           Set to -1 to recurse infinitely.
#       length: tuple, default: (2, )
#           If the iterable has one of the lengths in this tuple,
#           it will not be unpacked.
#   
#       Returns
#       -------
#       either `iterable` or `repeat(iterable)`
#       """
#       # input validation
#       no_iter_types = (float, int, datetime) if no_iter_types is None else no_iter_types
#       if not isinstance(length, ITERABLE_TYPES):
#           length = (length,)
#   
#       # non-iterable
#       if not isinstance(iterable, ITERABLE_TYPES):
#           return iterable
#   
#       # catch forbidden iterable
#       if isinstance(iterable, ITERABLE_TYPES) and len(iterable) in length:
#           all_allowed = True
#           for elem in iterable:
#               if not isinstance(elem, no_iter_types):
#                   all_allowed = False
#                   break
#           if all_allowed:
#               return repeat(iterable)
#   
#       # otherwise recursively
#       if depth != 0:
#           return [
#               filter_nozip(
#                   i,
#                   no_iter_types=no_iter_types,
#                   depth=depth - 1,
#                   length=length,
#               )
#               for i in iterable
#           ]
#   
#       # no hit
#       return iterable
#   
#   
#   class NoZip:
#       """
#       DEPRECATED: use `interplot.repeat` instead.
#   
#       Avoid iteration in `zip` and `interplot.zip_smart`
#       """
#   
#       def __init__(self, iterable):
#           """
#           DEPRECATED: use `interplot.repeat` instead.
#   
#           Avoid iteration of an iterable data type in the `zip` function.
#   
#           Class allows iteration and subscription.
#   
#           Call the instance to release the original variable.
#   
#           Parameters
#           ----------
#           iterable
#               Iterable variable which should be "hidden".
#           """
#           self.iterable = iterable
#   
#       def __iter__(self):
#           return iter(self.iterable)
#   
#       def __getitem__(self, item):
#           return self.iterable[item]
#   
#       def __call__(self):
#           return self.release()
#   
#       def __repr__(self):
#           return "NoZip({})".format(self.iterable.__repr__())
#   
#       def release(self):
#           """Return the original iterable variable."""
#           return self.iterable
#   
#   
#   def _repeat(arg, iterable_types, maxlen, unpack_nozip):
#       """
#       DEPRECATE: USE `repeat` INSTEAD.
#   
#       If `arg` is not an instance of `iterable_types`, repeat maxlen times.
#   
#       Unpacks `NoZip` instances by default.
#       """
#       if isinstance(arg, iterable_types):
#           return arg
#       if unpack_nozip and isinstance(arg, NoZip):
#           arg = arg()
#       return (arg,) * maxlen
#   
###########################
#### interplot/plot.py ####
###########################
#   """
#   Create `matplotlib/plotly` hybrid plots with a few lines of code.
#   
#   It combines the best of the `matplotlib` and the `plotly` worlds through
#   a unified, flat API.
#   All the necessary boilerplate code is contained in this module.
#   
#   Currently supported:
#   
#   - line plots (scatter)
#   - line fills
#   - histograms
#   - heatmaps
#   - boxplot
#   - linear regression
#   - text and image annotations
#   - 2D subplots
#   - color cycling
#   """
#   
#   import re
#   from warnings import warn
#   from pathlib import Path
#   from functools import wraps
#   from datetime import datetime
#   from io import BytesIO
#   from PIL import Image
#   import uuid
#   
#   import numpy as np
#   
#   from pandas.core.series import Series as pd_Series
#   from pandas.core.frame import DataFrame as pd_DataFrame
#   
#   from xarray.core.dataarray import DataArray as xr_DataArray
#   
#   import matplotlib.pyplot as plt
#   import matplotlib.colors as mcolors
#   
#   import plotly.graph_objects as go
#   import plotly.express as px
#   import plotly.subplots as sp
#   import plotly.offline
#   
#   from . import conf
#   from .iter import ITERABLE_TYPES, zip_smart, filter_nozip
#   from interplot import arraytools
#   
#   
#   def init_notebook_mode(connected=False):
#       """
#       Initialize plotly.js in the browser if not already done,
#       and deactivate matplotlib auto-display.
#   
#       Parameters
#       ----------
#       connected: bool, optional
#           If True, the plotly.js library will be loaded from an online CDN.
#           If False, the plotly.js library will be loaded locally.
#           Default: False
#       """
#       plotly.offline.init_notebook_mode(connected=connected)
#   
#       # turn off matplotlib auto-display
#       plt.plot()
#       plt.close()
#       plt.ioff()
#   
#   
#   def close():
#       """
#       Close all open matplotlib figures.
#       """
#       plt.close("all")
#   
#   
#   # if imported in notebook, init plotly notebook mode
#   try:
#       __IPYTHON__  # type: ignore
#       from IPython.core.display import display_html, display_png
#   
#       CALLED_FROM_NOTEBOOK = True
#   except NameError:
#       CALLED_FROM_NOTEBOOK = False
#   if CALLED_FROM_NOTEBOOK:
#       init_notebook_mode()
#   
#   
#   def pick_non_none(*args, fail=False):
#       """
#       Return the first non-None argument.
#   
#       Parameters
#       ----------
#       *args: any
#           Any number of arguments.
#       fail: bool, default: False
#           Throw a ValueError if all args are None.
#   
#           If set to False, None will be returned.
#   
#       Returns
#       -------
#       any
#           The first non-None argument.
#       """
#       for arg in args:
#           if arg is not None:
#               return arg
#       if fail:
#           raise ValueError("All arguments have value None.")
#       return None
#   
#   
#   def _rewrite_docstring(doc_core, doc_decorator=None, kwargs_remove=()):
#       """
#       Appends arguments to a docstring.
#   
#       Returns original docstring if conf._REWRITE_DOCSTRING is set to False.
#   
#       Attempts:
#       1. Search for [decorator.*?].
#       2. Search for numpy-style "Parameters" block.
#       3. Append to the end.
#   
#       Parameters
#       ----------
#       doc_core: str
#           Original docstring.
#       doc_decorator: str,
#           docstring to insert.
#       kwargs_remove: tuple of strs, optional
#           remove parameters in the docstring.
#   
#       Returns
#       -------
#       str:
#           Rewritten docstring
#       """
#       # check rewrite flag
#       if not conf._REWRITE_DOCSTRING:
#           return doc_core
#   
#       # input check
#       doc_core = "" if doc_core is None else doc_core
#       doc_decorator = (
#           conf._DOCSTRING_DECORATOR if doc_decorator is None else doc_decorator
#       )
#   
#       # find indentation level of doc_core
#       match = re.match("^\n?(?P<indent_core>[ \t]*)", doc_core)
#       indent_core = match.group("indent_core") if match else ""
#   
#       # find indentation level of doc_decorator
#       match = re.match("^\n?(?P<indent_core>[ \t]*)", doc_decorator)
#       indent_decorator = match.group("indent_core") if match else ""
#   
#       # remove kwargs from doc_decorator
#       for kwarg_key in kwargs_remove:
#           # remove docstring entry if it is the only argument
#           doc_decorator = re.sub(
#               (
#                   r"\n{0}"  # indent
#                   r"{1}[ ]*:"  # kwarg_key followed by colon
#                   r".*(?:\n{0}[ \t]+.*)*"  # the following further indented lines
#               ).format(
#                   indent_decorator,
#                   kwarg_key,
#               ),
#               r"",
#               doc_decorator,
#           )
#   
#           # remove docstring key if it is found in a list
#           doc_decorator = re.sub(
#               (
#                   (  # preceding kwarg_key
#                       r"(?P<front>"  # named group
#                       r"\n{0}"  # indentation
#                       r"(?:[a-zA-Z_]+)??"  # first arg
#                       r"(?:[ ]*,[ ]*[a-zA-Z_]+)??"  # following args
#                       r")"  # end named group
#                   )
#                   + (  # kwarg_key
#                       r"(?P<leading_coma>[ ]*,[ ]*)?"  # leading coma
#                       r"{1}"  # kwarg_key
#                       r"(?(leading_coma)|(?:[ ]*,[ ]*)?)"  # following coma if no leading coma  # noqa: E501
#                   )
#                   + r"(?P<back>(?:[ ]*,[ ]*[a-zA-Z_]+)*[ ]*?)"  # following arguments  # noqa: E501
#               ).format(indent_decorator, kwarg_key),
#               r"\g<front>\g<back>",
#               doc_decorator,
#           )
#   
#       # search "[decorator parameters]"
#       match = re.search(r"\n[ \t]*\[decorator.*?]", doc_core)
#       if match:
#           return re.sub(
#               r"\n[ \t]*\[decorator.*?]",
#               _adjust_indent(indent_decorator, indent_core, doc_decorator),
#               doc_core,
#           )
#   
#       # test for numpy-style doc_core
#       docstring_query = (
#           r"(?P<desc>(?:.*\n)*?)"  # desc
#           r"(?P<params>(?P<indent_core>[ \t]*)Parameters[ \t]*"  # params header
#           r"(?:\n(?!(?:[ \t]*\n)|(?:[ \t]*$)).*)*)"  # non-whitespace lines
#           r"(?P<rest>(?:.*\n)*.*$)"
#       )
#       docstring_query = (
#           r"(?P<desc>(?:.*\n)*?)"  # desc
#           r"(?P<params>(?P<indent_core>[ \t]*)Parameters[ \t]*"  # params header
#           r"(?:.*\n)*?)"  # params content
#           r"(?P<rest>[ \t]*\n[ \t]*[A-Za-z0-9 \t]*[ \t]*\n[ \t]*---(?:.*\n)*.*)$"
#           # next chapter
#       )
#       match = re.match(docstring_query, doc_core)
#       if match:
#           doc_parts = match.groupdict()
#           return (
#               doc_parts["desc"]
#               + doc_parts["params"]
#               + _adjust_indent(
#                   indent_decorator,
#                   doc_parts["indent_core"],
#                   doc_decorator,
#               )
#               + doc_parts["rest"]
#           )
#   
#       # non-numpy _DOCSTRING_DECORATOR, just append in the end
#       return doc_core + _adjust_indent(
#           indent_decorator,
#           indent_core,
#           doc_decorator,
#       )
#   
#   
#   def _plt_cmap_extremes(cmap, under=None, over=None, bad=None):
#       """
#       Get cmap with under and over range colors.
#   
#       Parameters
#       ----------
#       cmap: str or plt cmap
#           Colormap to use.
#           https://matplotlib.org/stable/gallery/color/colormap_reference.html
#       under, over, bad: color, optional
#           Color to use for under / over range values and bad values.
#   
#       Returns
#       -------
#       cmap: plt cmap
#           Provide cmap to plt: plt.imshow(cmap=cmap)
#       """
#       cmap = plt.get_cmap(cmap).copy()
#       if under:
#           cmap.set_under(under)
#       if over:
#           cmap.set_over(over)
#       if bad:
#           cmap.set_bad(bad)
#       return cmap
#   
#   
#   def _plotly_colormap_extremes(cs, under=None, over=None):
#       """
#       Append under and over range colors to plotly figure.
#   
#       Parameters
#       ----------
#       fig: plotly.Figure
#           Plotly Figure instance which contains a colormap.
#       under, over: color, optional
#           Color to use for under / over range values.
#   
#       Returns
#       -------
#       fig: plotly.Figure
#       """
#       cs = [[b for b in a] for a in cs]  # convert tuple to list
#       if under:
#           cs[0][0] = 0.0000001
#           cs = [[0, under]] + cs
#       if over:
#           cs[-1][0] = 0.9999999
#           cs = cs + [[1.0, over]]
#       return cs
#   
#   
#   def _adjust_indent(indent_decorator, indent_core, docstring):
#       """Adjust indentation of docstsrings."""
#       return re.sub(
#           r"\n{}".format(indent_decorator),
#           r"\n{}".format(indent_core),
#           docstring,
#       )
#   
#   
#   def _serialize_2d(serialize_pty=True, serialize_mpl=True):
#       """Decorator to catch 2D arrays and other data types to unpack."""
#   
#       def decorator(core):
#   
#           @wraps(core)
#           def wrapper(self, x, y=None, label=None, **kwargs):
#               """
#               Wrapper function for a method.
#   
#               If a pandas object is provided, the index will be used as x
#               if no x is provided.
#               Pandas column naming:
#                   * If no label is set, the column name will be used by default.
#                   * Manually set label string has priority.
#                   * Label strings may contain a {} to insert the column name.
#                   * Instead of setting a string, a callable may be provided to
#                   reformat the column name. It must accept the column name
#                   and return a string. E.g.:
#   
#                   > interplot.line(df, label=lambda n: n.strip())
#   
#                   > def capitalize(prefix="", suffix=""):
#                   >     return lambda name: prefix + name.upper() + suffix
#                   > interplot.line(df, label=capitalize("Cat. A: "))
#               xarray DataArrays will be convered to pandas and then handled
#               accordingly.
#               """
#               # reallocate x/y
#               if y is None:
#   
#                   # xarray DataArray
#                   if isinstance(x, xr_DataArray):
#                       x = x.to_pandas()
#   
#                   # pd.Series
#                   if isinstance(x, pd_Series):
#                       x, y = x.index, x
#                       if label is None:
#                           label = y.name
#                       elif isinstance(label, str) and "{}" in label:
#                           label = label.format(y.name)
#   
#                   # pd.DataFrame: split columns to pd.Series and iterate
#                   elif isinstance(x, pd_DataFrame):
#                       if (
#                           self.interactive
#                           and serialize_pty
#                           or not self.interactive
#                           and serialize_mpl
#                       ):
#                           for i, ((_, series), label_) in enumerate(
#                               zip_smart(x.items(), label)
#                           ):
#                               _serialize_2d(
#                                   serialize_pty=serialize_pty,
#                                   serialize_mpl=serialize_mpl,
#                               )(core)(
#                                   self,
#                                   series,
#                                   label=label_,
#                                   _serial_i=i,
#                                   _serial_n=len(x.columns),
#                                   **kwargs,
#                               )
#                           return
#   
#                   else:
#                       if hasattr(x, "copy") and callable(getattr(x, "copy")):
#                           y = x.copy()
#                       else:
#                           y = x
#                       x = np.arange(len(y))
#   
#               # 2D np.array
#               if isinstance(y, np.ndarray) and len(y.shape) == 2:
#                   if (
#                       self.interactive
#                       and serialize_pty
#                       or not self.interactive
#                       and serialize_mpl
#                   ):
#                       for i, (y_, label_) in enumerate(zip_smart(y.T, label)):
#                           _serialize_2d(
#                               serialize_pty=serialize_pty,
#                               serialize_mpl=serialize_mpl,
#                           )(core)(
#                               self,
#                               x,
#                               y_,
#                               label=label_,
#                               _serial_i=i,
#                               _serial_n=y.shape[1],
#                               **kwargs,
#                           )
#                       return
#   
#               return core(self, x, y, label=label, **kwargs)
#   
#           return wrapper
#   
#       return decorator
#   
#   
#   def _serialize_save(core):
#       """Decorator to serialise saving multiple figures."""
#   
#       @wraps(core)
#       def wrapper(self, path, export_format=None, **kwargs):
#           """
#           Wrapper function for a method.
#   
#           """
#           if isinstance(path, ITERABLE_TYPES) or isinstance(
#               export_format, ITERABLE_TYPES
#           ):
#               for path_, export_format_ in zip_smart(path, export_format):
#                   self.save(path_, export_format_, **kwargs)
#   
#               return
#   
#           return core(self, path, export_format=export_format, **kwargs)
#   
#       return wrapper
#   
#   
#   class NotebookInteraction:
#       """
#       Parent class for automatic display in Jupyter Notebook.
#   
#       Calls the child's `show()._repr_html_()` for automatic display
#       in Jupyter Notebooks.
#       """
#   
#       JS_RENDER_WARNING = (
#           """
#           <div class="alert alert-block alert-warning"
#               id="notebook-js-warning">
#               <p>
#                   Unable to render javascript-based plotly plot.<br>
#                   Call interplot.init_notebook_mode() or re-run this cell.<br>
#                   If viewing on GitHub, render the notebook in
#                   <a href="https://nbviewer.org/" target="_blank">
#                       NBViewer</a> instead.
#               </p>
#           </div>
#           <script type="text/javascript">
#               function hide_warning() {
#                   var element = document.getElementById(
#                       "notebook-js-warning"
#                   );
#                   element.parentNode.removeChild(element);
#               }
#               hide_warning();
#           </script>
#       """
#           if CALLED_FROM_NOTEBOOK
#           else ""
#       )
#   
#       def __call__(self, *args, **kwargs):
#           """Calls the `self.show()` or `self.plot()` method."""
#           # look for show() method
#           try:
#               return self.show(*args, **kwargs)
#   
#           # fall back to plot() method
#           except AttributeError:
#               return self.plot(*args, **kwargs)
#   
#       def _repr_html_(self):
#           """Call forward to `self.show()._repr_html_()`."""
#           init_notebook_mode()
#   
#           # look for show() method
#           try:
#               return self.JS_RENDER_WARNING + self.show()._repr_html_()
#   
#           # fall back to plot() method
#           except AttributeError:
#               try:
#                   return self.JS_RENDER_WARNING + self.plot()._repr_html_()
#   
#               # not implemented
#               except AttributeError:
#                   raise NotImplementedError
#   
#       def _repr_mimebundle_(self, *args, **kwargs):
#           """Call forward to `self.show()_repr_mimebundle_()`"""
#           # look for show() method
#           try:
#               return self.show()._repr_mimebundle_(*args, **kwargs)
#   
#           # fall back to plot() method
#           except AttributeError:
#               try:
#                   return self.plot()._repr_mimebundle_(*args, **kwargs)
#   
#               # not implemented
#               except AttributeError:
#                   raise NotImplementedError
#   
#   
#   class LabelGroup:
#       """
#       Grouping Labels in interactive plots.
#   
#       Grouping is not supported in matplotlib legends.
#       In matplotlib, only the label and show parameters are used.
#   
#       Toggling behavior can be set via `ip.Plot(..., legend_togglegroup=<bool>)`
#       or globally with `conf.PTY_LEGEND_TOGGLEGROUP`.
#   
#       Parameters
#       ----------
#       group_title: str, optional
#           Group title for the legend group. Will be shown above the group if
#           specified.
#       group_id: str, optional
#           Must be unique for each group. If none is provided,
#           a UUID will be generated.
#       default_show: bool or "first", default: True
#           Whether to show the label in the legend.
#   
#           If set to "first", the element dispatcher will check, whether
#           the figure instance already has an trace of this group_id.
#           If not, the label will be shown in the legend, otherwise it won't.
#       default_label: str, optional
#           Default label for elements in this group.
#       default_legend_only: bool, default: False
#           Whether to show the trace only in the legend by default.
#       """
#   
#       def __init__(
#           self,
#           group_title=None,
#           group_id=None,
#           default_show=True,
#           default_label=None,
#           default_legend_only=False,
#       ):
#           self.group_title = group_title
#           self.group_id = pick_non_none(group_id, uuid.uuid1().hex)
#           self.default_label = default_label
#           self.default_show = default_show
#           self.default_legend_only = default_legend_only
#   
#       def __call__(self, *args, **kwargs):
#           """
#           Return an element with the default parameters.
#           """
#           return self.element(*args, **kwargs)
#   
#       def element(self, label=None, show=None, legend_only=None):
#           """
#           Define a label for an element in this group.
#   
#           Parameters
#           ----------
#           label: str, optional
#               Label for the element.
#   
#               If not specified, the default label will be used.
#           show: bool, optional
#               Whether to show the label in the legend.
#   
#               If not specified, the default show value will be used.
#           legend_only: bool, optional
#               Whether to show the trace only in the legend.
#   
#               If not specified, the default legend_only value will be used.
#           """
#   
#           def inner(
#               inst,
#               default_label=None,
#               group_title=self.group_title,
#               group_id=self.group_id,
#               label=pick_non_none(label, self.default_label),
#               show=pick_non_none(show, self.default_show),
#               legend_only=pick_non_none(legend_only, self.default_legend_only),
#           ):
#               if label is None:
#                   label = default_label
#   
#               if show == "first":
#                   show = group_id not in inst.legend_ids
#               inst.legend_ids.add(group_id)
#   
#               # PLOTLY
#               if inst.interactive:
#                   legend_kwargs = dict(
#                       legendgroup=group_id,
#                       name=label,
#                       showlegend=show,
#                   )
#   
#                   if legend_only:
#                       legend_kwargs["visible"] = "legendonly"
#                   if group_title is not None:
#                       legend_kwargs["legendgrouptitle_text"] = group_title
#   
#                   return legend_kwargs
#   
#               # MATPLOTLIB
#               if show:
#                   return dict(label=label)
#               return dict()
#   
#           return inner
#   
#   
#   class Plot(NotebookInteraction):
#       """
#       Create `matplotlib/plotly` hybrid plots with a few lines of code.
#   
#       It combines the best of the `matplotlib` and the `plotly` worlds through
#       a unified, flat API.
#       All the necessary boilerplate code is contained in this module.
#   
#       Currently supported:
#   
#       - line plots (scatter)
#       - line fills
#       - histograms
#       - heatmaps
#       - boxplot
#       - linear regression
#       - text annotations
#       - 2D subplots
#       - color cycling
#   
#       Parameters
#       ----------
#   
#       Examples
#       --------
#       >>> fig = interplot.Plot(
#       ...     interactive=True,
#       ...     title="Everything Under Control",
#       ...     fig_size=(800, 500),
#       ...     rows=1,
#       ...     cols=2,
#       ...     shared_yaxes=True,
#       ...     save_fig=True,
#       ...     save_format=("html", "png"),
#       ...     # ...
#       ... )
#       ... fig.add_hist(np.random.normal(1, 0.5, 1000), row=0, col=0)
#       ... fig.add_boxplot(
#       ...     [
#       ...         np.random.normal(20, 5, 1000),
#       ...         np.random.normal(40, 8, 1000),
#       ...         np.random.normal(60, 5, 1000),
#       ...     ],
#       ...     row=0,
#       ...     col=1,
#       ... )
#       ... # ...
#       ... fig.post_process()
#       ... fig.show()
#       saved figure at Everything-Under-Control.html
#       saved figure at Everything-Under-Control.png
#   
#       .. raw:: html
#           :file: ../source/plot_examples/Everything-Under-Control.html
#       """
#   
#       __doc__ = _rewrite_docstring(__doc__)
#   
#       def __init__(
#           self,
#           interactive=None,
#           rows=1,
#           cols=1,
#           title=None,
#           xlabel=None,
#           ylabel=None,
#           xlim=None,
#           ylim=None,
#           xlog=False,
#           ylog=False,
#           shared_xaxes=False,
#           shared_yaxes=False,
#           column_widths=None,
#           row_heights=None,
#           fig_size=None,
#           dpi=None,
#           legend_loc=None,
#           legend_title=None,
#           legend_togglegroup=None,
#           color_cycle=None,
#           save_fig=None,
#           save_format=None,
#           save_config=None,
#           global_custom_func=None,
#           mpl_custom_func=None,
#           pty_custom_func=None,
#           pty_update_layout=None,
#       ):
#           # input verification
#           if shared_xaxes == "cols":
#               shared_xaxes = "columns"
#           if shared_yaxes == "cols":
#               shared_yaxes = "columns"
#           if shared_xaxes is True:
#               shared_xaxes = "all"
#           if shared_yaxes is True:
#               shared_yaxes = "all"
#   
#           self.interactive = pick_non_none(
#               interactive,
#               conf.INTERACTIVE,
#           )
#           self.rows = rows
#           self.cols = cols
#           self.title = title
#           self.xlabel = xlabel
#           self.ylabel = ylabel
#           self.xlim = xlim
#           self.ylim = ylim
#           self.xlog = xlog
#           self.ylog = ylog
#           self.fig_size = fig_size
#           self.dpi = pick_non_none(
#               dpi,
#               conf.DPI,
#           )
#           self.legend_loc = legend_loc
#           self.legend_title = legend_title
#           self.legend_ids = set()
#           self.color_cycle = pick_non_none(
#               color_cycle,
#               conf.COLOR_CYCLE,
#           )
#           self.save_fig = save_fig
#           self.save_format = save_format
#           self.save_config = save_config
#           self.global_custom_func = global_custom_func
#           self.mpl_custom_func = mpl_custom_func
#           self.pty_custom_func = pty_custom_func
#           self.pty_update_layout = pty_update_layout
#           self.element_count = np.zeros((rows, cols), dtype=int)
#           self.i_color = 0
#   
#           # init plotly
#           if self.interactive:
#   
#               # init fig
#               figure = go.Figure()
#               self.fig = sp.make_subplots(
#                   rows=rows,
#                   cols=cols,
#                   shared_xaxes=shared_xaxes,
#                   shared_yaxes=shared_yaxes,
#                   row_heights=row_heights,
#                   column_widths=column_widths,
#                   figure=figure,
#               )
#   
#           # init matplotlib
#           else:
#               gridspec_kw = dict(
#                   width_ratios=column_widths,
#                   height_ratios=row_heights,
#               )
#   
#               # convert px to inches
#               self.fig_size = pick_non_none(
#                   fig_size,
#                   conf.MPL_FIG_SIZE,
#               )
#               px = 1 / self.dpi
#               figsize = (self.fig_size[0] * px, self.fig_size[1] * px)
#   
#               # init fig
#               self.fig, self.ax = plt.subplots(
#                   rows,
#                   cols,
#                   figsize=figsize,
#                   dpi=self.dpi,
#                   squeeze=False,
#                   gridspec_kw=gridspec_kw,
#               )
#   
#               # shared axes
#               for i_row in range(self.rows):
#                   for i_col in range(self.cols):
#   
#                       # skip 0/0
#                       if i_col == 0 and i_row == 0:
#                           continue
#   
#                       # set shared x axes
#                       if (
#                           shared_xaxes == "all"
#                           or type(shared_xaxes) is bool
#                           and shared_xaxes is True
#                       ):
#                           self.ax[i_row, i_col].sharex(self.ax[0, 0])
#                       elif shared_xaxes == "columns":
#                           self.ax[i_row, i_col].sharex(self.ax[0, i_col])
#                       elif shared_xaxes == "rows":
#                           self.ax[i_row, i_col].sharex(self.ax[i_row, 0])
#   
#                       # set shared y axes
#                       if (
#                           shared_yaxes == "all"
#                           or type(shared_yaxes) is bool
#                           and shared_yaxes is True
#                       ):
#                           self.ax[i_row, i_col].sharey(self.ax[0, 0])
#                       elif shared_yaxes == "columns":
#                           self.ax[i_row, i_col].sharey(self.ax[0, i_col])
#                       elif shared_yaxes == "rows":
#                           self.ax[i_row, i_col].sharey(self.ax[i_row, 0])
#   
#           self.update(
#               title=title,
#               xlabel=xlabel,
#               ylabel=ylabel,
#               xlim=xlim,
#               ylim=ylim,
#               xlog=xlog,
#               ylog=ylog,
#               fig_size=fig_size,
#               dpi=dpi,
#               legend_loc=legend_loc,
#               legend_title=legend_title,
#               legend_togglegroup=legend_togglegroup,
#               color_cycle=color_cycle,
#               save_fig=save_fig,
#               save_format=save_format,
#               save_config=save_config,
#               global_custom_func=global_custom_func,
#               mpl_custom_func=mpl_custom_func,
#               pty_custom_func=pty_custom_func,
#               pty_update_layout=pty_update_layout,
#           )
#   
#       @staticmethod
#       def init(fig=None, *args, **kwargs):
#           """
#           Initialize a Plot instance, if not already initialized.
#   
#           Parameters
#           ----------
#           fig: Plot or any
#               If fig is a Plot instance, return it.
#               Otherwise, create a new Plot instance.
#           *args, **kwargs: any
#               Passed to Plot.__init__.
#           """
#           if isinstance(fig, Plot):
#               return fig
#           return Plot(*args, **kwargs)
#   
#       def _digest_label(self, label, default_label=None, show_legend=None):
#           if isinstance(label, LabelGroup):
#               return label()(self, default_label=default_label)
#   
#           if callable(label):
#               return label(self, default_label=default_label)
#   
#           # PLOTLY
#           if self.interactive:
#               return self._get_plotly_legend_args(
#                   label,
#                   default_label=default_label,
#                   show_legend=show_legend,
#               )
#   
#           # MATPLOTLIB
#           return dict(
#               label=(
#                   None
#                   if show_legend is False
#                   else (default_label if label is None else label)
#               )
#           )
#   
#       def update(
#           self,
#           title=None,
#           xlabel=None,
#           ylabel=None,
#           xlim=None,
#           ylim=None,
#           xlog=None,
#           ylog=None,
#           fig_size=None,
#           dpi=None,
#           legend_loc=None,
#           legend_title=None,
#           legend_togglegroup=None,
#           color_cycle=None,
#           save_fig=None,
#           save_format=None,
#           save_config=None,
#           global_custom_func=None,
#           mpl_custom_func=None,
#           pty_custom_func=None,
#           pty_update_layout=None,
#       ):
#           """
#           Update plot parameters set during initialisation.
#   
#           Parameters
#           ----------
#   
#           Examples
#           --------
#           >>> fig = interplot.Plot(fig_size=(600, 400))
#           ... fig.add_line((1,2,4,3))
#           ... fig.save("export_landscape.png")
#           ... fig.save("export_fullsize.html")
#           ... fig.update(fig_size=(400, 600))
#           ... fig.save("export_portrait.png")
#   
#           >>> @interplot.magic_plot
#           ... def plot_points(data, fig=None):
#           ...     fig.add_line(data)
#           ...     fig.update(title="SUM: {}".format(sum(data)))
#           ... plot_points([1,2,4,3])
#           """
#           self.title = pick_non_none(title, self.title)
#           self.xlabel = pick_non_none(xlabel, self.xlabel)
#           self.ylabel = pick_non_none(ylabel, self.ylabel)
#           self.xlim = pick_non_none(xlim, self.xlim)
#           self.ylim = pick_non_none(ylim, self.ylim)
#           self.xlog = pick_non_none(xlog, self.xlog)
#           self.ylog = pick_non_none(ylog, self.ylog)
#           self.dpi = pick_non_none(dpi, self.dpi)
#           self.legend_loc = pick_non_none(legend_loc, self.legend_loc)
#           self.legend_title = pick_non_none(legend_title, self.legend_title)
#           self.color_cycle = pick_non_none(
#               color_cycle,
#               self.color_cycle,
#           )
#           self.save_fig = pick_non_none(save_fig, self.save_fig)
#           self.save_format = pick_non_none(save_format, self.save_format)
#           self.save_config = pick_non_none(save_config, self.save_config)
#           self.global_custom_func = pick_non_none(
#               global_custom_func,
#               self.global_custom_func,
#           )
#           self.mpl_custom_func = pick_non_none(
#               mpl_custom_func,
#               self.mpl_custom_func,
#           )
#           self.pty_custom_func = pick_non_none(
#               pty_custom_func,
#               self.pty_custom_func,
#           )
#           self.pty_update_layout = pick_non_none(
#               pty_update_layout,
#               self.pty_update_layout,
#           )
#   
#           # PLOTLY
#           if self.interactive:
#               self.title = self._encode_html(self.title)
#               self.fig_size = pick_non_none(
#                   fig_size,
#                   self.fig_size,
#                   conf.PTY_FIG_SIZE,
#               )
#   
#               # unpacking
#               width, height = self.fig_size
#               if isinstance(legend_title, ITERABLE_TYPES):
#                   warn(
#                       "Plotly only has one legend, however multiple legend_"
#                       "titles were provided. Only the first one will be used!"
#                   )
#                   legend_title = legend_title[0]
#                   if isinstance(legend_title, ITERABLE_TYPES):
#                       legend_title = legend_title[0]
#   
#               # update layout
#               self.fig.update_layout(
#                   title=self.title,
#                   legend_title=legend_title,
#                   height=height,
#                   width=width,
#                   barmode="group",
#               )
#               if not pick_non_none(
#                   legend_togglegroup,
#                   conf.PTY_LEGEND_TOGGLEGROUP,
#               ):
#                   self.fig.update_layout(
#                       legend_groupclick="toggleitem",
#                   )
#   
#               # axis limits and log scale
#               for (
#                   i_row,
#                   xlim_row,
#                   ylim_row,
#                   xlog_row,
#                   ylog_row,
#               ) in zip_smart(
#                   range(1, self.rows + 1),
#                   filter_nozip(self.xlim),
#                   filter_nozip(self.ylim),
#                   xlog,
#                   ylog,
#               ):
#                   for (
#                       i_col,
#                       xlim_tile,
#                       ylim_tile,
#                       xlog_tile,
#                       ylog_tile,
#                   ) in zip_smart(
#                       range(1, self.cols + 1),
#                       filter_nozip(xlim_row),
#                       filter_nozip(ylim_row),
#                       xlog_row,
#                       ylog_row,
#                   ):
#                       if xlim_tile is not None and isinstance(xlim_tile[0], datetime):
#                           xlim_tile = (
#                               xlim_tile[0].timestamp() * 1000,
#                               xlim_tile[1].timestamp() * 1000,
#                           )
#                       self.fig.update_xaxes(
#                           range=xlim_tile,
#                           row=i_row,
#                           col=i_col,
#                           type="log" if xlog_tile else None,
#                       )
#                       self.fig.update_yaxes(
#                           range=ylim_tile,
#                           row=i_row,
#                           col=i_col,
#                           type="log" if ylog_tile else None,
#                       )
#   
#               # axis labels
#               for text, i_col in zip_smart(xlabel, range(1, self.cols + 1)):
#                   self.fig.update_xaxes(title_text=text, row=self.rows, col=i_col)
#               for text, i_row in zip_smart(ylabel, range(1, self.rows + 1)):
#                   self.fig.update_yaxes(title_text=text, row=i_row, col=1)
#   
#           # MATPLOTLIB
#           else:
#               # convert px to inches
#               self.fig_size = pick_non_none(
#                   fig_size,
#                   self.fig_size,
#                   conf.MPL_FIG_SIZE,
#               )
#   
#               px = 1 / self.dpi
#               figsize = (self.fig_size[0] * px, self.fig_size[1] * px)
#               self.fig.set_figwidth(figsize[0])
#               self.fig.set_figheight(figsize[1])
#   
#               # title
#               if self.cols == 1:
#                   self.ax[0, 0].set_title(self.title)
#               else:
#                   self.fig.suptitle(self.title)
#   
#               # axis labels
#               if isinstance(self.xlabel, ITERABLE_TYPES) or (
#                   self.cols == 1 and self.rows == 1
#               ):
#                   for text, i_col in zip_smart(self.xlabel, range(self.cols)):
#                       self.ax[self.rows - 1, i_col].set_xlabel(text)
#               else:
#                   self.fig.supxlabel(self.xlabel)
#               if isinstance(self.ylabel, ITERABLE_TYPES) or (
#                   self.cols == 1 and self.rows == 1
#               ):
#                   for text, i_row in zip_smart(self.ylabel, range(self.rows)):
#                       self.ax[i_row, 0].set_ylabel(text)
#               else:
#                   self.fig.supylabel(self.ylabel)
#   
#               # log scale
#               for row, xlog_row in zip_smart(range(self.rows), xlog):
#                   for col, xlog_tile in zip_smart(range(self.cols), xlog_row):
#                       if xlog_tile:
#                           self.ax[row, col].set_xscale("log")
#               for row, ylog_row in zip_smart(range(self.rows), ylog):
#                   for col, ylog_tile in zip_smart(range(self.cols), ylog_row):
#                       if ylog_tile:
#                           self.ax[row, col].set_yscale("log")
#   
#       update.__doc__ = _rewrite_docstring(
#           update.__doc__,
#           kwargs_remove=(
#               "rows, cols",
#               "shared_xaxes, shared_yaxes",
#               "column_widths, row_heights",
#           ),
#       )
#   
#       @staticmethod
#       def _get_plotly_legend_args(label, default_label=None, show_legend=None):
#           """
#           Return keyword arguments for label configuration.
#   
#           Parameters
#           ----------
#           label: str
#               Name to display.
#           default_label: str, optional
#               If label is None, fall back to default_label.
#               Default: None
#               By default, plotly will enumerate the unnamed traces itself.
#           show_legend: bool, optional
#               Show label in legend.
#               Default: None
#               By default, the label will be displayed if it is not None
#               (in case of label=None, the automatic label will only be displayed
#               on hover)
#           """
#           if isinstance(label, LabelGroup):
#               return label.get_element()
#   
#           if isinstance(label, dict):
#               return label
#   
#           legend_kwargs = dict(
#               name=pick_non_none(label, default_label),
#           )
#           if show_legend:
#               legend_kwargs["showlegend"] = True
#           elif isinstance(show_legend, bool) and not show_legend:
#               legend_kwargs["showlegend"] = False
#           else:
#               legend_kwargs["showlegend"] = False if label is None else True
#   
#           return legend_kwargs
#   
#       @staticmethod
#       def _get_plotly_anchor(axis, cols, row, col):
#           """
#           Get axis id based on row and col.
#   
#           Parameters
#           ----------
#           axis: str
#               x or y.
#           cols: int
#               Number of columns.
#               Usually self.cols
#           row, col: int
#               Row and col index in plotly manner:
#               STARTING WITH 1.
#   
#           Returns
#           -------
#           axis id: str
#           """
#           id = (row - 1) * cols + col
#           if id == 1:
#               return axis
#           return axis + str((row - 1) * cols + col)
#   
#       @staticmethod
#       def _encode_html(text):
#           if text is None:
#               return None
#           return re.sub(r"\n", "<br>", text)
#   
#       def _mpl_coords_data_to_axes(self, x, y, row, col):
#           """
#           Convert data coordinates to axes coordinates.
#   
#           Parameters
#           ----------
#           x, y: float
#               Data coordinates.
#           row, col: int
#               Row and col index in matplotlib manner:
#               STARTING WITH 0.
#   
#           Returns
#           -------
#           x_ax, y_ax: float
#               Axes coordinates.
#           """
#           return (
#               self.ax[row, col].transData + self.ax[row, col].transAxes.inverted()
#           ).transform((x, y))
#   
#       def get_cycle_color(self, increment=1, i=None):
#           """
#           Retrieves the next color in the color cycle.
#   
#           Parameters
#           ----------
#           increment: int, optional
#               If the same color should be returned the next time, pass 0.
#               To jump the next color, pass 2.
#               Default: 1
#           i: int, optional
#               Get a fixed index of the color cycle instead of the next one.
#               This will not modify the regular color cycle iteration.
#   
#           Returns
#           -------
#           color: str
#               HEX color, with leading hashtag
#           """
#           if i is None:
#               if self.i_color >= len(self.color_cycle):
#                   self.i_color = 0
#               color = self.color_cycle[self.i_color]
#               self.i_color += increment
#               return color
#           else:
#               return self.color_cycle[i]
#   
#       def digest_color(self, color=None, alpha=None, increment=1):
#           """
#           Parse color with matplotlib.colors to a rgba array.
#   
#           Parameters
#           ----------
#           color: any color format matplotlib accepts, optional
#               E.g. "blue", "#0000ff", "C3"
#               If None is provided, the next one from COLOR_CYCLE will be picked.
#           alpha: float, optional
#               Set alpha / opacity.
#               Overrides alpha contained in color input.
#               Default: None (use the value contained in color or default to 1)
#           increment: int, optional
#               If a color from the cycler is picked, increase the cycler by
#               this increment.
#           """
#           # if color undefined, cycle COLOR_CYCLE
#           if color is None:
#               color = self.get_cycle_color(increment)
#   
#           # get index from COLOR_CYCLE
#           elif isinstance(color, int) or isinstance(color, np.integer):
#               color = self.color_cycle[color % len(self.color_cycle)]
#           elif color[0] == "C" or color[0] == "c":
#               color = self.color_cycle[int(color[1:]) % len(self.color_cycle)]
#   
#           rgba = list(mcolors.to_rgba(color))
#           if alpha is not None:
#               rgba[3] = alpha
#   
#           # PLOTLY
#           if self.interactive:
#               return "rgba({},{},{},{:.4f})".format(
#                   *[int(d * 255) for d in rgba[:3]],
#                   rgba[3],
#               )
#   
#           # MATPLOTLIB
#           return tuple(rgba)
#   
#       @staticmethod
#       def digest_marker(
#           marker,
#           mode,
#           interactive,
#           recursive=False,
#           **pty_marker_kwargs,
#       ):
#           """
#           Digests the marker parameter based on the given mode.
#   
#           Parameters
#           ----------
#           marker: int or str
#               The marker to be digested. If an integer is provided, it is
#               converted to the corresponding string marker using `plotly`
#               numbering.
#               If not provided, the default marker "circle" is used.
#           mode: str
#               The mode to determine if markers should be used.
#               If no markers should be drawn, None is returned.
#   
#           Returns
#           -------
#           str or None
#               The digested marker string if markers are requested,
#               otherwise None.
#           """
#           if isinstance(marker, ITERABLE_TYPES):
#               if interactive:
#                   return dict(
#                       symbol=[
#                           Plot.digest_marker(
#                               marker=m,
#                               mode=mode,
#                               interactive=interactive,
#                               recursive=True,
#                           )
#                           for m in marker
#                       ],
#                       **pty_marker_kwargs,
#                   )
#               return Plot.digest_marker(
#                   marker=marker[0],
#                   mode=mode,
#                   interactive=interactive,
#                   recursive=True,
#               )
#   
#           if "markers" not in mode:
#               return None
#   
#           if isinstance(marker, (int, np.integer)):
#               marker = conf.PTY_MARKERS_LIST[marker]
#   
#           if marker is None:
#               marker = conf.PTY_MARKERS_LIST[0]
#   
#           if interactive:
#               if marker not in conf.PTY_MARKERS_LIST:
#                   marker = conf.PTY_MARKERS.get(marker, marker)
#               if recursive:
#                   return marker
#               return dict(
#                   symbol=marker,
#                   **pty_marker_kwargs,
#               )
#   
#           return conf.MPL_MARKERS.get(marker, marker)
#   
#       @_serialize_2d()
#       def add_line(
#           self,
#           x,
#           y=None,
#           x_error=None,
#           y_error=None,
#           mode=None,
#           line_style="solid",
#           marker=None,
#           marker_size=None,
#           marker_line_width=1,
#           marker_line_color=None,
#           label=None,
#           show_legend=None,
#           color=None,
#           opacity=None,
#           linewidth=None,
#           row=0,
#           col=0,
#           _serial_i=0,
#           _serial_n=1,
#           pty_marker_kwargs=None,
#           kwargs_pty=None,
#           kwargs_mpl=None,
#           **kwargs,
#       ):
#           """
#           Draw a line or scatter plot.
#   
#           Parameters
#           ----------
#           x: array-like
#           y: array-like, optional
#               If only `x` is defined, it will be assumed as `y`.
#               If a pandas `Series` is provided, the index will
#               be taken as `x`.
#               Else if a pandas `DataFrame` is provided, the method call
#               is looped for each column.
#               Else `x` will be an increment, starting from `0`.
#               If a 2D numpy `array` is provided, the method call
#               is looped for each column.
#           x_error, y_error: number or shape(N,) or shape(2, N), optional
#               The errorbar sizes (`matplotlib` style):
#                   - scalar: Symmetric +/- values for all data points.
#                   - shape(N,): Symmetric +/-values for each data point.
#                   - shape(2, N): Separate - and + values for each bar. First row
#                       contains the lower errors, the second row contains the
#                       upper errors.
#                   - None: No errorbar.
#           mode: str, optional
#               Options: `lines` / `lines+markers` / `markers`
#   
#               The default depends on the method called:
#                   - `add_line`: `lines`
#                   - `add_scatter`: `markers`
#                   - `add_linescatter`: `lines+markers`
#           line_style: str, optional
#               Line style.
#               Options: `solid`, `dashed`, `dotted`, `dashdot`
#   
#               Aliases: `-`, `--`, `dash`, `:`, `dot`, `-.`
#           marker: int or str, optional
#               Marker style.
#               If an integer is provided, it will be converted to the
#               corresponding string marker using `plotly` numbering.
#               If not provided, the default marker `circle` is used.
#           marker_size: int, optional
#           marker_line_width: int, optional
#           marker_line_color: str, optional
#               Can be hex, rgb(a) or any named color that is understood
#               by matplotlib.
#   
#               Default: same color as `color`.
#           label: str, optional
#               Trace label for legend.
#           show_legend: bool, optional
#               Whether to show the label in the legend.
#   
#               By default, it will be shown if a label is defined.
#           color: str, optional
#               Trace color.
#   
#               Can be hex, rgb(a) or any named color that is understood
#               by matplotlib.
#   
#               The color cycle can be accessed with "C0", "C1", ...
#   
#               Default: color is retrieved from `Plot.digest_color`,
#               which cycles through `COLOR_CYCLE`.
#           opacity: float, optional
#               Opacity (=alpha) of the fill.
#   
#               By default, fallback to alpha value provided with color argument,
#               or 1.
#           row, col: int, optional
#               If the plot contains a grid, provide the coordinates.
#   
#               Attention: Indexing starts with 0!
#           pty_marker_kwargs: dict, optional
#               PLOTLY ONLY.
#   
#               Additional marker arguments.
#           kwargs_pty, kwargs_mpl, **kwargs: optional
#               Pass specific keyword arguments to the line core method.
#   
#           Examples
#           --------
#   
#           Using the interplot.Plot method:
#   
#           >>> fig = interplot.Plot(title="line, linescatter and scatter")
#           ... fig.add_line([0,4,6,7], [1,2,4,8])
#           ... fig.add_linescatter([0,4,6,7], [8,4,2,1])
#           ... fig.add_scatter([0,4,6,7], [2,4,4,2], )
#           ... fig.post_process()
#           ... fig.show()
#   
#           [plotly figure, "line, linescatter and scatter"]
#   
#           Using interplot.line:
#   
#           >>> interplot.line([0,4,6,7], [1,2,4,8])
#   
#           .. raw:: html
#               :file: ../source/plot_examples/basic_plot_pty.html
#   
#           >>> interplot.line(
#           ...     x=[0,4,6,7],
#           ...     y=[1,2,4,8],
#           ...     interactive=False,
#           ...     color="red",
#           ...     title="matplotlib static figure",
#           ...     xlabel="abscissa",
#           ...     ylabel="ordinate",
#           ... )
#   
#           .. image:: plot_examples/basic_plot_mpl.png
#               :alt: [matplotlib plot "Normally distributed Noise]
#           """
#           self.element_count[row, col] += 1
#           mode = "lines" if mode is None else mode
#           color = self.digest_color(color, opacity)
#   
#           # PLOTLY
#           if self.interactive:
#               if kwargs_pty is None:
#                   kwargs_pty = dict()
#               if pty_marker_kwargs is None:
#                   pty_marker_kwargs = dict()
#               pty_marker_kwargs.update(
#                   dict(
#                       size=marker_size,
#                       line_width=marker_line_width,
#                       line_color=(
#                           color
#                           if marker_line_color is None
#                           else self.digest_color(marker_line_color, 1)
#                       ),
#                   )
#               )
#               if x_error is not None:
#                   if not isinstance(x_error, ITERABLE_TYPES):
#                       x_error = np.array((x_error,) * len(x))
#                   if isinstance(x_error[0], ITERABLE_TYPES):
#                       x_error = dict(
#                           type="data",
#                           array=x_error[1],
#                           arrayminus=x_error[0],
#                       )
#                   else:
#                       x_error = dict(
#                           type="data",
#                           array=x_error,
#                       )
#               if y_error is not None:
#                   if not isinstance(y_error, ITERABLE_TYPES):
#                       y_error = np.array((y_error,) * len(y))
#                   if isinstance(y_error[0], ITERABLE_TYPES):
#                       y_error = dict(
#                           type="data",
#                           array=y_error[1],
#                           arrayminus=y_error[0],
#                       )
#                   else:
#                       y_error = dict(
#                           type="data",
#                           array=y_error,
#                       )
#               row += 1
#               col += 1
#   
#               self.fig.add_trace(
#                   go.Scatter(
#                       x=x,
#                       y=y,
#                       error_x=x_error,
#                       error_y=y_error,
#                       mode=mode,
#                       marker=self.digest_marker(
#                           marker,
#                           mode,
#                           interactive=self.interactive,
#                           **pty_marker_kwargs,
#                       ),
#                       **self._digest_label(
#                           label,
#                           show_legend=show_legend,
#                       ),
#                       marker_color=color,
#                       line=dict(
#                           width=linewidth,
#                           dash=conf.PTY_LINE_STYLES.get(line_style, line_style),
#                       ),
#                       **kwargs_pty,
#                       **kwargs,
#                   ),
#                   row=row,
#                   col=col,
#               )
#   
#           # MATPLOTLIB
#           else:
#               if kwargs_mpl is None:
#                   kwargs_mpl = dict()
#               self.ax[row, col].errorbar(
#                   x,
#                   y,
#                   xerr=x_error,
#                   yerr=y_error,
#                   **self._digest_label(label, show_legend=show_legend),
#                   color=color,
#                   lw=linewidth,
#                   linestyle=(
#                       conf.MPL_LINE_STYLES.get(line_style, line_style)
#                       if "lines" in mode
#                       else "None"
#                   ),
#                   marker=self.digest_marker(
#                       marker,
#                       mode,
#                       interactive=self.interactive,
#                   ),
#                   markersize=marker_size,
#                   markeredgewidth=marker_line_width,
#                   markeredgecolor=(
#                       color
#                       if marker_line_color is None
#                       else self.digest_color(marker_line_color)
#                   ),
#                   **kwargs_mpl,
#                   **kwargs,
#               )
#   
#       @wraps(add_line)
#       def add_scatter(
#           self,
#           *args,
#           mode="markers",
#           **kwargs,
#       ):
#           self.add_line(
#               *args,
#               mode=mode,
#               **kwargs,
#           )
#   
#       @wraps(add_line)
#       def add_linescatter(
#           self,
#           *args,
#           mode="markers+lines",
#           **kwargs,
#       ):
#           self.add_line(
#               *args,
#               mode=mode,
#               **kwargs,
#           )
#   
#       @_serialize_2d()
#       def add_bar(
#           self,
#           x,
#           y=None,
#           horizontal=False,
#           width=0.8,
#           label=None,
#           show_legend=None,
#           color=None,
#           opacity=None,
#           line_width=1,
#           line_color=None,
#           row=0,
#           col=0,
#           _serial_i=0,
#           _serial_n=1,
#           kwargs_pty=None,
#           kwargs_mpl=None,
#           **kwargs,
#       ):
#           """
#           Draw a bar plot.
#   
#           Parameters
#           ----------
#           x: array-like
#           y: array-like, optional
#               If only either `x` or `y` is defined, it will be assumed
#               as the size of the bar, regardless whether it's horizontal
#               or vertical.
#               If both `x` and `y` are defined, `x` will be taken as the
#               position of the bar, and `y` as the size, regardless of
#               the orientation.
#               If a pandas `Series` is provided, the index will
#               be taken as the position.
#               Else if a pandas `DataFrame` is provided, the method call
#               is looped for each column.
#               If a 2D numpy `array` is provided, the method call
#               is looped for each column, with the index as the position.
#           horizontal: bool, optional
#               If True, the bars are drawn horizontally. Default is False.
#           width: float, optional
#               Relative width of the bar. Must be in the range (0, 1).
#           label: str, optional
#               Trace label for legend.
#           show_legend: bool, optional
#               Whether to show the label in the legend.
#   
#               By default, it will be shown if a label is defined.
#           color: str, optional
#               Trace color.
#   
#               Can be hex, rgb(a) or any named color that is understood
#               by matplotlib.
#   
#               The color cycle can be accessed with "C0", "C1", ...
#   
#               Default: color is retrieved from `Plot.digest_color`,
#               which cycles through `COLOR_CYCLE`.
#           opacity: float, optional
#               Opacity (=alpha) of the fill.
#   
#               By default, fallback to alpha value provided with color argument,
#               or 1.
#           line_width: float, optional
#               The width of the bar outline. Default is 1.
#           line_color: str, optional
#               The color of the bar outline. This can be a named color or a tuple
#               specifying the RGB values.
#   
#               By default, the same color as the fill is used.
#           row, col: int, optional
#               If the plot contains a grid, provide the coordinates.
#   
#               Attention: Indexing starts with 0!
#           kwargs_pty, kwargs_mpl, **kwargs: optional
#               Pass specific keyword arguments to the line core method.
#           """
#           self.element_count[row, col] += 1
#           # PLOTLY
#           if self.interactive:
#               if kwargs_pty is None:
#                   kwargs_pty = dict()
#   
#               if horizontal:
#                   x, y = y, x
#   
#               row += 1
#               col += 1
#               self.fig.add_trace(
#                   go.Bar(
#                       x=x,
#                       y=y,
#                       orientation="h" if horizontal else "v",
#                       **self._digest_label(
#                           label,
#                           show_legend=show_legend,
#                       ),
#                       marker_color=self.digest_color(color, opacity),
#                       marker=dict(
#                           line=dict(
#                               width=0 if line_color is None else line_width,
#                               color=(
#                                   color
#                                   if line_color is None
#                                   else self.digest_color(line_color, 1)
#                               ),
#                           ),
#                       ),
#                       **kwargs_pty,
#                       **kwargs,
#                   ),
#                   row=row,
#                   col=col,
#               )
#   
#           else:
#               if kwargs_mpl is None:
#                   kwargs_mpl = dict()
#               offset = ((2 * _serial_i + 1) / 2 / _serial_n - 0.5) * width
#               (self.ax[row, col].barh if horizontal else self.ax[row, col].bar)(
#                   np.arange(len(x)) + offset,
#                   y,
#                   (width / _serial_n),
#                   color=self.digest_color(color, opacity),
#                   edgecolor=(
#                       self.digest_color(line_color, 1) if line_color is not None else None
#                   ),
#                   linewidth=line_width,
#                   **self._digest_label(label, show_legend=show_legend),
#                   **kwargs_mpl,
#                   **kwargs,
#               )
#               (
#                   self.ax[row, col].set_yticks
#                   if horizontal
#                   else self.ax[row, col].set_xticks
#               )(
#                   np.arange(len(x)),
#                   x,
#               )
#   
#       def add_hist(
#           self,
#           x=None,
#           y=None,
#           bins=None,
#           density=False,
#           label=None,
#           show_legend=None,
#           color=None,
#           opacity=None,
#           row=0,
#           col=0,
#           kwargs_pty=None,
#           kwargs_mpl=None,
#           **kwargs,
#       ):
#           """
#           Draw a histogram.
#   
#           Parameters
#           ----------
#           x: array-like
#               Histogram data.
#           bins: int, optional
#               Number of bins.
#               If undefined, plotly/matplotlib will detect automatically.
#               Default: None
#           label: str, optional
#               Trace label for legend.
#           color: str, optional
#               Trace color.
#   
#               Can be hex, rgb(a) or any named color that is understood
#               by matplotlib.
#   
#               The color cycle can be accessed with "C0", "C1", ...
#   
#               Default: color is retrieved from `Plot.digest_color`,
#               which cycles through `COLOR_CYCLE`.
#           opacity: float, optional
#               Opacity (=alpha) of the fill.
#   
#               By default, fallback to alpha value provided with color argument,
#               or 1.
#           row, col: int, default: 0
#               If the plot contains a grid, provide the coordinates.
#   
#               Attention: Indexing starts with 0!
#           kwargs_pty, kwargs_mpl, **kwargs: optional
#               Pass specific keyword arguments to the hist core method.
#           """
#           # input verification
#           if x is None and y is None:
#               raise ValueError("Either x or y must be defined.")
#           if x is not None and y is not None:
#               raise ValueError("x and y cannot be defined both.")
#   
#           bins_attribute = dict(nbinsx=bins) if y is None else dict(nbinsy=bins)
#           self.element_count[row, col] += 1
#   
#           # PLOTLY
#           if self.interactive:
#               if kwargs_pty is None:
#                   kwargs_pty = dict()
#               if density:
#                   kwargs_pty.update(dict(histnorm="probability"))
#               row += 1
#               col += 1
#               self.fig.add_trace(
#                   go.Histogram(
#                       x=x,
#                       y=y,
#                       **self._digest_label(
#                           label,
#                           show_legend=show_legend,
#                       ),
#                       **bins_attribute,
#                       marker_color=self.digest_color(color, opacity),
#                       **kwargs_pty,
#                       **kwargs,
#                   ),
#                   row=row,
#                   col=col,
#               )
#   
#           # MATPLOTLIB
#           else:
#               if kwargs_mpl is None:
#                   kwargs_mpl = dict()
#               if x is None:
#                   x = y
#                   orientation = "horizontal"
#               else:
#                   orientation = "vertical"
#               self.ax[row, col].hist(
#                   x,
#                   **self._digest_label(label, show_legend=show_legend),
#                   bins=bins,
#                   density=density,
#                   color=self.digest_color(color, opacity),
#                   orientation=orientation,
#                   **kwargs_mpl,
#                   **kwargs,
#               )
#   
#       def add_boxplot(
#           self,
#           x,
#           horizontal=False,
#           label=None,
#           show_legend=None,
#           color=None,
#           color_median="black",
#           opacity=None,
#           notch=True,
#           row=0,
#           col=0,
#           kwargs_pty=None,
#           kwargs_mpl=None,
#           **kwargs,
#       ):
#           """
#           Draw a boxplot.
#   
#           Parameters
#           ----------
#           x: array or sequence of vectors
#               Data to build boxplot from.
#           horizontal: bool, default: False
#               Show boxplot horizontally.
#           label: tuple of strs, optional
#               Trace labels for legend.
#           color: tuple of strs, optional
#               Fill colors.
#   
#               Can be hex, rgb(a) or any named color that is understood
#               by matplotlib.
#   
#               The color cycle can be accessed with "C0", "C1", ...
#   
#               Default: color is retrieved from `Plot.digest_color`,
#               which cycles through `COLOR_CYCLE`.
#           color_median: color, default: "black"
#               MPL only.
#               Color of the median line.
#           opacity: float, optional
#               Opacity (=alpha) of the fill.
#   
#               By default, fallback to alpha value provided with color argument,
#               or 1.
#           row, col: int, optional
#               If the plot contains a grid, provide the coordinates.
#   
#               Attention: Indexing starts with 0!
#           kwargs_pty, kwargs_mpl, **kwargs: optional
#               Pass specific keyword arguments to the boxplot core method.
#           """
#           # determine number of boxplots
#           if isinstance(x[0], (int, float)):
#               n = 1
#           else:
#               n = len(x)
#           # input validation
#           if not isinstance(label, ITERABLE_TYPES):
#               label = (label,) * n
#           if not isinstance(color, ITERABLE_TYPES):
#               color = (color,) * n
#   
#           # PLOTLY
#           if self.interactive:
#               if kwargs_pty is None:
#                   kwargs_pty = dict()
#   
#               # if x contains multiple datasets, iterate add_boxplot
#               if not n == 1:
#                   for x_i, label_, show_legend_, color_, opacity_ in zip_smart(
#                       x,
#                       label,
#                       show_legend,
#                       color,
#                       opacity,
#                   ):
#                       self.add_boxplot(
#                           x_i,
#                           horizontal=horizontal,
#                           label=label_,
#                           show_legend=show_legend_,
#                           row=row,
#                           col=col,
#                           color=color_,
#                           opacity=opacity_,
#                           kwargs_pty=kwargs_pty,
#                           **kwargs,
#                       )
#   
#               # draw a single plotly boxplot
#               else:
#                   row += 1
#                   col += 1
#                   kw_data = "x" if horizontal else "y"
#                   pty_kwargs = {
#                       kw_data: x,
#                   }
#                   self.fig.add_trace(
#                       go.Box(
#                           **pty_kwargs,
#                           **self._digest_label(
#                               label[0],
#                               show_legend=show_legend,
#                           ),
#                           marker_color=self.digest_color(color[0], opacity),
#                           **kwargs_pty,
#                           **kwargs,
#                       ),
#                       row=row,
#                       col=col,
#                   )
#   
#           # MATPLOTLIB
#           else:
#               if kwargs_mpl is None:
#                   kwargs_mpl = dict()
#               bplots = self.ax[row, col].boxplot(
#                   x,
#                   vert=not horizontal,
#                   labels=None if show_legend is False else label,
#                   patch_artist=True,
#                   notch=notch,
#                   medianprops=dict(color=color_median),
#                   **kwargs_mpl,
#                   **kwargs,
#               )
#               for bplot, color_ in zip_smart(bplots["boxes"], color):
#                   bplot.set_facecolor(self.digest_color(color_, opacity))
#   
#       def add_heatmap(
#           self,
#           data,
#           lim=(None, None),
#           aspect=1,
#           invert_x=False,
#           invert_y=False,
#           cmap="rainbow",
#           cmap_under=None,
#           cmap_over=None,
#           cmap_bad=None,
#           row=0,
#           col=0,
#           kwargs_pty=None,
#           kwargs_mpl=None,
#           **kwargs,
#       ):
#           """
#           Draw a heatmap.
#   
#           Parameters
#           ----------
#           data: 2D array-like
#               2D data to show heatmap.
#           lim: list/tuple of 2x float, optional
#               Lower and upper limits of the color map.
#           aspect: float, default: 1
#               Aspect ratio of the axes.
#           invert_x, invert_y: bool, optional
#               Invert the axes directions.
#               Default: False
#           cmap: str, default: "rainbow"
#               Color map to use.
#               https://matplotlib.org/stable/gallery/color/colormap_reference.html
#               Note: Not all cmaps are available for both libraries,
#               and may differ slightly.
#           cmap_under, cmap_over, cmap_bad: str, optional
#               Colors to display if under/over range or a pixel is invalid,
#               e.g. in case of `np.nan`.
#               `cmap_bad` is not available for interactive plotly plots.
#           row, col: int, optional
#               If the plot contains a grid, provide the coordinates.
#   
#               Attention: Indexing starts with 0!
#           kwargs_pty, kwargs_mpl, **kwargs: optional
#               Pass specific keyword arguments to the heatmap core method.
#           """
#           # input verification
#           if lim is None:
#               lim = [None, None]
#           else:
#               lim = list(lim)
#           if len(lim) != 2:
#               raise ValueError("lim must be a tuple or dict with two items.")
#           self.element_count[row, col] += 1
#   
#           # PLOTLY
#           if self.interactive:
#               # input verification
#               if kwargs_pty is None:
#                   kwargs_pty = dict()
#               if cmap_bad is not None:
#                   warn("cmap_bad is not supported for plotly.")
#               row += 1
#               col += 1
#   
#               # add colorscale limits
#               if cmap_under is not None or cmap_over is not None:
#                   # crappy workaround to make plotly translate named cmap to list
#                   cmap = _plotly_colormap_extremes(
#                       px.imshow(
#                           img=[[0, 0], [0, 0]],
#                           color_continuous_scale=cmap,
#                       ).layout.coloraxis.colorscale,
#                       cmap_under,
#                       cmap_over,
#                   )
#                   if lim != [None, None]:
#                       delta = lim[1] - lim[0]
#                       lim[0] = lim[0] - 0.000001 * delta
#                       lim[1] = lim[1] + 0.000001 * delta
#   
#               self.fig.add_trace(
#                   go.Heatmap(
#                       z=data,
#                       zmin=lim[0],
#                       zmax=lim[1],
#                       colorscale=cmap,
#                       **kwargs_pty,
#                       **kwargs,
#                   ),
#                   row=row,
#                   col=col,
#               )
#               self.fig.update_xaxes(
#                   autorange=("reversed" if invert_x else None),
#                   row=row,
#                   col=col,
#               )
#               self.fig.update_yaxes(
#                   scaleanchor=self._get_plotly_anchor("x", self.cols, row, col),
#                   scaleratio=aspect,
#                   autorange=("reversed" if invert_x else None),
#                   row=row,
#                   col=col,
#               )
#   
#           # MATPLOTLIB
#           else:
#               if kwargs_mpl is None:
#                   kwargs_mpl = dict()
#               cmap = _plt_cmap_extremes(
#                   cmap,
#                   under=cmap_under,
#                   over=cmap_over,
#                   bad=cmap_bad,
#               )
#               imshow = self.ax[row, col].imshow(
#                   data,
#                   cmap=cmap,
#                   aspect=aspect,
#                   vmin=lim[0],
#                   vmax=lim[1],
#                   **kwargs_mpl,
#                   **kwargs,
#               )
#               self.fig.colorbar(imshow)
#               if invert_x:
#                   self.ax[row, col].axes.invert_xaxis()
#               if not invert_y:
#                   self.ax[row, col].axes.invert_yaxis()
#   
#       def add_regression(
#           self,
#           x,
#           y=None,
#           p=0.05,
#           linspace=101,
#           **kwargs,
#       ):
#           """
#           Draw a linear regression plot.
#   
#           Parameters
#           ----------
#           x: array-like or `interplot.arraytools.LinearRegression` instance
#               X axis data, or pre-existing LinearRegression instance.
#           y: array-like, optional
#               Y axis data.
#               If a LinearRegression instance is provided for x,
#               y can be omitted and will be ignored.
#           p: float, default: 0.05
#               p-value.
#           linspace: int, default: 101
#               Number of data points for linear regression model
#               and conficence and prediction intervals.
#           kwargs:
#               Keyword arguments for `interplot.arraytools.LinearRegression.plot`.
#           """
#           if isinstance(x, arraytools.LinearRegression) or hasattr(x, "is_linreg"):
#               x.plot(fig=self, **kwargs)
#           else:
#               arraytools.LinearRegression(
#                   x,
#                   y,
#                   p=p,
#                   linspace=linspace,
#               ).plot(fig=self, **kwargs)
#   
#       def add_fill(
#           self,
#           x,
#           y1,
#           y2=None,
#           label=None,
#           mode="lines",
#           color=None,
#           opacity=0.5,
#           line_width=0.0,
#           line_opacity=1.0,
#           line_color=None,
#           row=0,
#           col=0,
#           kwargs_pty=None,
#           kwargs_mpl=None,
#           **kwargs,
#       ):
#           """
#           Draw a fill between two y lines.
#   
#           Parameters
#           ----------
#           x: array-like
#           y1, y2: array-like, optional
#               If only `x` and `y1` is defined,
#               it will be assumed as `y1` and `y2`,
#               and `x` will be the index, starting from 0.
#           label: str or interplot.LabelGroup, optional
#               Trace label for legend.
#           color, line_color: str, optional
#               Fill and line color.
#   
#               Can be hex, rgb(a) or any named color that is understood
#               by matplotlib.
#   
#               The color cycle can be accessed with "C0", "C1", ...
#   
#               If line_color is undefined, the the fill color will be used.
#   
#               Default: color is retrieved from `Plot.digest_color`,
#               which cycles through `COLOR_CYCLE`.
#           opacity, line_opacity: float, default: 0.5
#               Opacity (=alpha) of the fill.
#   
#               Set to None to use a value provided with the color argument.
#           line_width: float, default: 0.
#               Boundary line width.
#           row, col: int, default: 0
#               If the plot contains a grid, provide the coordinates.
#   
#               Attention: Indexing starts with 0!
#           kwargs_pty, kwargs_mpl, **kwargs: optional
#               Pass specific keyword arguments to the fill core method.
#           """
#           # input verification
#           if y2 is None:
#               y1, y2 = x, y1
#               x = np.arange(len(y1))
#           self.element_count[row, col] += 1
#   
#           fill_color = self.digest_color(
#               color,
#               opacity,
#               increment=0 if line_color is None else 1,
#           )
#           line_color = self.digest_color(
#               color if line_color is None else line_color,
#               line_opacity,
#           )
#   
#           if not isinstance(label, LabelGroup):
#               label = LabelGroup(
#                   "fill_{}_{}_{}".format(row, col, self.element_count[row, col]),
#                   default_label="fill" if label is None else label,
#               )
#   
#           # PLOTLY
#           if self.interactive:
#               if kwargs_pty is None:
#                   kwargs_pty = dict()
#               row += 1
#               col += 1
#               self.fig.add_trace(
#                   go.Scatter(
#                       x=x,
#                       y=y1,
#                       mode=mode,
#                       **self._digest_label(
#                           label.element(
#                               show=False,
#                           ),
#                       ),
#                       line=dict(width=line_width),
#                       marker_color=line_color,
#                       **kwargs_pty,
#                       **kwargs,
#                   ),
#                   row=row,
#                   col=col,
#               )
#               self.fig.add_trace(
#                   go.Scatter(
#                       x=x,
#                       y=y2,
#                       mode=mode,
#                       **self._digest_label(
#                           label.element(),
#                       ),
#                       fill="tonexty",
#                       fillcolor=fill_color,
#                       line=dict(width=line_width),
#                       marker_color=line_color,
#                       **kwargs_pty,
#                       **kwargs,
#                   ),
#                   row=row,
#                   col=col,
#               )
#   
#           # MATPLOTLIB
#           else:
#               if kwargs_mpl is None:
#                   kwargs_mpl = dict()
#               self.ax[row, col].fill_between(
#                   x,
#                   y1,
#                   y2,
#                   **self._digest_label(
#                       label.element(),
#                   ),
#                   linewidth=line_width,
#                   edgecolor=self.digest_color(line_color, line_opacity, increment=0),
#                   facecolor=self.digest_color(color, opacity),
#                   **kwargs_mpl,
#                   **kwargs,
#               )
#   
#       def add_text(
#           self,
#           x,
#           y,
#           text,
#           horizontal_alignment="center",
#           vertical_alignment="center",
#           text_alignment=None,
#           data_coords=None,
#           x_data_coords=True,
#           y_data_coords=True,
#           color="black",
#           opacity=None,
#           row=0,
#           col=0,
#           kwargs_pty=None,
#           kwargs_mpl=None,
#           **kwargs,
#       ):
#           """
#           Draw text.
#   
#           Parameters
#           ----------
#           x, y: float
#               Coordinates of the text.
#           text: str
#               Text to add.
#           horizontal_alignment, vertical_alignment: str, default: "center"
#               Where the coordinates of the text box anchor.
#   
#               Options for `horizontal_alignment`:
#                   - "left"
#                   - "center"
#                   - "right"
#   
#               Options for `vertical_alignment`:
#                   - "top"
#                   - "center"
#                   - "bottom"
#           text_alignment: str, optional
#               Set how text is aligned inside its box.
#   
#               If left undefined, horizontal_alignment will be used.
#           data_coords: bool, default: True
#               Whether the `x`, `y` coordinates are provided in data coordinates
#               or in relation to the axes.
#   
#               If set to `False`, `x`, `y` should be in the range (0, 1).
#               If `data_coords` is set, it will override
#               `x_data_coords` and `y_data_coords`.
#           x_data_coords, y_data_coords: bool, default: True
#               PTY only.
#               Specify the anchor for each axis separate.
#           color: str, default: "black"
#               Trace color.
#   
#               Can be hex, rgb(a) or any named color that is understood
#               by matplotlib.
#   
#               The color cycle can be accessed with "C0", "C1", ...
#   
#               Default: color is retrieved from `Plot.digest_color`,
#               which cycles through `COLOR_CYCLE`.
#           opacity: float, optional
#               Opacity (=alpha) of the fill.
#   
#               By default, fallback to alpha value provided with color argument,
#               or 1.
#           row, col: int, optional
#               If the plot contains a grid, provide the coordinates.
#   
#               Attention: Indexing starts with 0!
#           kwargs_pty, kwargs_mpl, **kwargs: optional
#               Pass specific keyword arguments to the line core method.
#           """
#           # input verification
#           if data_coords is not None:
#               x_data_coords = data_coords
#               y_data_coords = data_coords
#           text_alignment = (
#               horizontal_alignment if text_alignment is None else text_alignment
#           )
#   
#           # PLOTLY
#           if self.interactive:
#               if kwargs_pty is None:
#                   kwargs_pty = dict()
#               if vertical_alignment == "center":
#                   vertical_alignment = "middle"
#               row += 1
#               col += 1
#               x_domain = "" if x_data_coords else " domain"
#               y_domain = "" if y_data_coords else " domain"
#               self.fig.add_annotation(
#                   x=x,
#                   y=y,
#                   text=self._encode_html(text),
#                   align=text_alignment,
#                   xanchor=horizontal_alignment,
#                   yanchor=vertical_alignment,
#                   xref=self._get_plotly_anchor("x", self.cols, row, col) + x_domain,
#                   yref=self._get_plotly_anchor("y", self.cols, row, col) + y_domain,
#                   font=dict(color=self.digest_color(color, opacity)),
#                   row=row,
#                   col=col,
#                   showarrow=False,
#                   **kwargs_pty,
#               )
#   
#           # MATPLOTLIB
#           else:
#               # input validation
#               if kwargs_mpl is None:
#                   kwargs_mpl = dict()
#               if not x_data_coords == y_data_coords:
#                   warn(
#                       "x_data_coords and y_data_coords must correspond "
#                       "for static matplotlib plot. x_data_coords was used."
#                   )
#               transform = (
#                   dict() if x_data_coords else dict(transform=self.ax[row, col].transAxes)
#               )
#               self.ax[row, col].text(
#                   x,
#                   y,
#                   s=text,
#                   color=self.digest_color(color, opacity),
#                   horizontalalignment=horizontal_alignment,
#                   verticalalignment=vertical_alignment,
#                   multialignment=text_alignment,
#                   **transform,
#                   **kwargs_mpl,
#                   **kwargs,
#               )
#   
#       def add_image(
#           self,
#           x,
#           y,
#           image,
#           horizontal_alignment="center",
#           vertical_alignment="center",
#           data_coords=True,
#           x_size=1,
#           y_size=1,
#           sizing="contain",
#           opacity=None,
#           row=0,
#           col=0,
#           kwargs_pty=None,
#           kwargs_mpl=None,
#           **kwargs,
#       ):
#           """
#           Draw an image.
#   
#           Parameters
#           ----------
#           x, y: float
#               Coordinates of the text.
#           image: Image object or str
#               The image as a PIL Image object.
#   
#               For plotly, URLs are also accepted.
#           horizontal_alignment, vertical_alignment: str, default: "center"
#               Where the coordinates of the text box anchor.
#   
#               Options for `horizontal_alignment`:
#                   - "left"
#                   - "center"
#                   - "right"
#   
#               Options for `vertical_alignment`:
#                   - "top"
#                   - "center"
#                   - "bottom"
#           data_coords: bool, default: True
#               Whether the `x`, `y` coordinates are provided in data coordinates
#               or in relation to the axes.
#   
#               If set to `False`, `x`, `y` should be in the range [0, 1].
#           sizing: str, default: "contain"
#               How the image should be sized.
#   
#               Options:
#                   - "contain": fit the image inside the box. The entire image
#                   will be visible, and the aspect ratio will be preserved.
#                   - "stretch": stretch the image to fit the box. The image
#                   may be distorted.
#                   - "fill": fill the box with the image. The image may be
#                   cropped, but will keep its aspect ratio. Only available
#                   for plotly.
#           opacity: float, optional
#               Opacity (=alpha) of the fill.
#           row, col: int, optional
#               If the plot contains a grid, provide the coordinates.
#   
#               Attention: Indexing starts with 0!
#           kwargs_pty, kwargs_mpl, **kwargs: optional
#               Pass specific keyword arguments to the line core method.
#           """
#           # input validation
#           if x_size is None and y_size is None:
#               x_size = 1
#               y_size = 1
#   
#           # PLOTLY
#           if self.interactive:
#               if kwargs_pty is None:
#                   kwargs_pty = dict()
#               if vertical_alignment == "center":
#                   vertical_alignment = "middle"
#               row += 1
#               col += 1
#               x_domain = "" if data_coords else " domain"
#               y_domain = "" if data_coords else " domain"
#   
#               self.fig.add_layout_image(
#                   dict(
#                       source=image,
#                       x=x,
#                       y=y,
#                       xref=self._get_plotly_anchor("x", self.cols, row, col) + x_domain,
#                       yref=self._get_plotly_anchor("y", self.cols, row, col) + y_domain,
#                       xanchor=horizontal_alignment,
#                       yanchor=vertical_alignment,
#                       sizex=x_size,
#                       sizey=y_size,
#                       sizing=sizing,
#                       opacity=opacity,
#                       **kwargs_pty,
#                       **kwargs,
#                   )
#               )
#   
#           # MATPLOTLIB
#           else:
#               # input validation
#               if kwargs_mpl is None:
#                   kwargs_mpl = dict()
#               if not isinstance(image, Image.Image):
#                   warn("Image must be a PIL Image object for static " "matplotlib plot.")
#   
#               if data_coords or data_coords is None:
#                   x1 = x + x_size
#                   y1 = y + y_size
#                   if horizontal_alignment == "center":
#                       x -= x_size / 2
#                       x1 -= x_size / 2
#                   elif horizontal_alignment == "right":
#                       x -= x_size
#                       x1 -= x_size
#                   if vertical_alignment == "center":
#                       y -= y_size / 2
#                       y1 -= y_size / 2
#                   elif vertical_alignment == "top":
#                       y -= y_size
#                       y1 -= y_size
#   
#                   x0, y0 = self._mpl_coords_data_to_axes(x, y, row, col)
#                   x1, y1 = self._mpl_coords_data_to_axes(x1, y1, row, col)
#   
#               else:
#                   x0 = x
#                   x1 = x + x_size
#                   y0 = y
#                   y1 = y + y_size
#                   if horizontal_alignment == "center":
#                       x0 -= x_size / 2
#                       x1 -= x_size / 2
#                   elif horizontal_alignment == "right":
#                       x0 -= x_size
#                       x1 -= x_size
#                   if vertical_alignment == "center":
#                       y0 -= y_size / 2
#                       y1 -= y_size / 2
#                   elif vertical_alignment == "top":
#                       y0 -= y_size
#                       y1 -= y_size
#   
#               aspect = "auto" if sizing == "stretch" else 1.0
#               if sizing == "fill":
#                   warn(
#                       "sizing='fill' is not supported for static "
#                       "matplotlib plot. 'contain' behavior is used instead."
#                   )
#               if sizing == "contain" and (
#                   horizontal_alignment != "center" or vertical_alignment != "center"
#               ):
#                   warn(
#                       "When using `sizing='contain'` with `horizontal_alignment`"
#                       " or `vertical_alignment` not set to 'center', the image "
#                       "may not be positioned as expected. "
#                       "The image box is created according to the alignment, "
#                       "but the image itself is always centered inside the box."
#                   )
#   
#               inset = self.ax[row, col].inset_axes(
#                   [x0, y0, x1 - x0, y1 - y0],
#               )
#               inset.set_axis_off()
#               inset.imshow(
#                   image,
#                   aspect=aspect,
#                   alpha=opacity,
#               )
#   
#       def post_process(
#           self,
#           global_custom_func=None,
#           mpl_custom_func=None,
#           pty_custom_func=None,
#           pty_update_layout=None,
#       ):
#           """
#           Finish the plot.
#   
#           Parameters
#           ----------
#           Note: If not provided, the parameters given on init or
#           the `interplot.conf` default values will be used.
#           mpl_custom_func: function, default: None
#               MATPLOTLIB ONLY.
#   
#               Pass a function reference to further style the matplotlib graphs.
#               Function must accept `fig, ax` and return `fig, ax`.
#   
#               If providing your own function reference, `conf.MPL_CUSTOM_FUNC`
#               will not be executed.
#   
#               Note: `ax` always has `row` and `col` coordinates, even if the
#               plot is just 1x1.
#   
#               >>> def mpl_custom_func(fig, ax):
#               ...     # do your customisations
#               ...     fig.do_stuff()
#               ...     ax[0, 0].do_more()
#               ...
#               ...     # also include default function
#               ...     fig, ax = conf.MPL_CUSTOM_FUNC(fig, ax)
#               ...
#               ...     return fig, ax
#               ...
#               ... fig = interplot.Plot(
#               ...     interactive=False,
#               ...     mpl_custom_func=mpl_custom_func
#               ... )
#               ... fig.add_line(x, y)
#               ... fig.post_process()  # mpl_custom_func will be executed here
#               ... fig.show()
#           pty_custom_func: function, default: None
#               PLOTLY ONLY.
#   
#               Pass a function reference to further style the plotly graphs.
#               Function must accept `fig` and return `fig`.
#   
#               If providing your own function reference, `conf.PTY_CUSTOM_FUNC`
#               will not be executed.
#   
#               >>> def pty_custom_func(fig):
#               ...     # do your customisations
#               ...     fig.do_stuff()
#               ...
#               ...     # also include default function
#               ...     fig = conf.PTY_CUSTOM_FUNC(fig)
#               ...
#               ...     return fig
#               ...
#               ... fig = interplot.Plot(
#               ...     interactive=True,
#               ...     pty_custom_func=pty_custom_func
#               ... )
#               ... fig.add_line(x, y)
#               ... fig.post_process()  # pty_custom_func will be executed here
#               ... fig.show()
#           pty_update_layout: dict, default: None
#               PLOTLY ONLY.
#   
#               Pass keyword arguments to plotly's
#               `plotly.graph_objects.Figure.update_layout(**pty_update_layout)`
#           """
#           # input verification
#           global_custom_func = pick_non_none(
#               global_custom_func,
#               self.global_custom_func,
#               conf.GLOBAL_CUSTOM_FUNC,
#               lambda _: None,
#           )
#           mpl_custom_func = pick_non_none(
#               mpl_custom_func,
#               self.mpl_custom_func,
#               conf.MPL_CUSTOM_FUNC,
#               lambda fig, ax: (fig, ax),
#           )
#           pty_custom_func = pick_non_none(
#               pty_custom_func,
#               self.pty_custom_func,
#               conf.PTY_CUSTOM_FUNC,
#               lambda fig: fig,
#           )
#           pty_update_layout = pick_non_none(
#               pty_update_layout,
#               self.pty_update_layout,
#               conf.PTY_UPDATE_LAYOUT,
#               dict(),
#           )
#   
#           # global custom function
#           global_custom_func(self)
#   
#           # PLOTLY
#           if self.interactive:
#               self.fig.update_layout(**pty_update_layout)
#               self.fig = pty_custom_func(self.fig)
#   
#           # MATPLOTLIB
#           else:
#   
#               # axis limits
#               for ax_row, xlim_row, ylim_row in zip_smart(
#                   self.ax,
#                   filter_nozip(self.xlim),
#                   filter_nozip(self.ylim),
#               ):
#                   for ax_tile, xlim_tile, ylim_tile in zip_smart(
#                       ax_row,
#                       filter_nozip(xlim_row),
#                       filter_nozip(ylim_row),
#                   ):
#                       ax_tile.set_xlim(xlim_tile)
#                       ax_tile.set_ylim(ylim_tile)
#   
#               # legend for each subplot
#               for i_row, loc_row, title_row in zip_smart(
#                   range(self.rows),
#                   self.legend_loc,
#                   self.legend_title,
#               ):
#                   for i_col, loc_tile, title_tile in zip_smart(
#                       range(self.cols),
#                       loc_row,
#                       title_row,
#                   ):
#                       # don't show legend
#                       if (
#                           type(loc_tile) is bool
#                           and loc_tile is False
#                           or loc_tile is None
#                           and self.element_count[i_row, i_col] < 2
#                       ):
#                           pass
#   
#                       # show legend if n>=2 or set to True
#                       else:
#                           if type(loc_tile) is bool and loc_tile is True:
#                               loc_tile = "best"
#                           self.ax[i_row, i_col].legend(
#                               title=title_tile,
#                               loc=loc_tile,
#                           )
#   
#               self.fig.tight_layout(pad=1.5)
#   
#               if mpl_custom_func is not None:
#                   self.fig, self.ax = mpl_custom_func(self.fig, self.ax)
#   
#           if self.save_fig is not None:
#               self.save(self.save_fig, export_format=self.save_format)
#   
#       @_serialize_save
#       def save(
#           self,
#           path,
#           export_format=None,
#           html_no_fig_size=True,
#           print_confirm=True,
#           **kwargs,
#       ):
#           """
#           Save the plot.
#   
#           Parameters
#           ----------
#           path: str, pathlib.Path, bool
#               May point to a directory or a filename.
#               If only a directory is provided (or `True` for local directory),
#               the filename will automatically be generated from the plot title.
#   
#               An iterable of multiple paths may be provided. In this case
#               the save command will be repeated for each element.
#           export_format: str, optional
#               If the format is not indicated in the file name, specify a format.
#   
#               An iterable of multiple formats may be provided. In this case
#               the save command will be repeated for each element.
#           print_confirm: bool, optional
#               Print a confirmation message where the file has been saved.
#               Default: True
#   
#           Returns
#           -------
#           pathlib.Path
#               Path to the exported file.
#           """
#           # input verification
#           if isinstance(path, bool):
#               if path:
#                   path = Path()
#               else:
#                   return
#           else:
#               path = Path(path)
#   
#           # auto-generate filename
#           if path.is_dir():
#               filename = self.title
#               if filename is None or str(filename) == "":
#                   filename = "interplot_figure"
#               else:
#                   filename = str(filename)
#   
#               for key, value in conf.EXPORT_REPLACE.items():
#                   filename = re.sub(key, value, filename)
#               filename += "." + (
#                   conf.EXPORT_FORMAT if export_format is None else export_format
#               )
#               path = path / filename
#   
#           # PLOTLY
#           if self.interactive:
#   
#               # HTML
#               if str(path)[-5:] == ".html":
#                   if html_no_fig_size:
#                       fig_size = self.fig_size
#                       self.update(fig_size=(None, None))
#   
#                   self.fig.write_html(
#                       path,
#                       config=(
#                           conf.PTY_CONFIG
#                           if self.save_config is None
#                           else self.save_config
#                       ),
#                       **kwargs,
#                   )
#   
#                   if html_no_fig_size:
#                       self.update(fig_size=fig_size)
#   
#               # image
#               else:
#                   scale = self.dpi / 100.0
#                   self.fig.write_image(
#                       path,
#                       scale=scale,
#                       **kwargs,
#                   )
#   
#           # MATPLOTLIB
#           else:
#               self.fig.savefig(
#                   path,
#                   facecolor="white",
#                   bbox_inches="tight",
#                   **kwargs,
#               )
#   
#           if print_confirm:
#               print("saved figure at {}".format(str(path)))
#   
#           return path
#   
#       def show(self):
#           """Show the plot."""
#           if CALLED_FROM_NOTEBOOK:
#               if self.interactive:
#                   init_notebook_mode()
#                   display_html(self.JS_RENDER_WARNING, raw=True)
#                   return self.fig.show(
#                       config=(
#                           conf.PTY_CONFIG
#                           if self.save_config is None
#                           else self.save_config
#                       )
#                   )
#               display_png(self._repr_png_(), raw=True)
#               return
#   
#           if self.interactive:
#               return self.fig.show(
#                   config=(
#                       conf.PTY_CONFIG if self.save_config is None else self.save_config
#                   )
#               )
#           return self.fig.show()
#   
#       def close(self):
#           """Close the plot."""
#           if not self.interactive:
#               plt.close(self.fig)
#   
#           else:
#               warn("close() is not supported for interactive plots.")
#   
#       def _repr_mimebundle_(self, *args, **kwargs):
#           if self.interactive:
#               return self.fig._repr_mimebundle_(*args, **kwargs)
#   
#           else:
#               raise NotImplementedError
#   
#       def _repr_html_(self):
#           if self.interactive:
#               init_notebook_mode()
#               return self.JS_RENDER_WARNING + self.fig._repr_html_()
#           raise NotImplementedError
#   
#       def _repr_png_(self):
#           if self.interactive:
#               raise NotImplementedError
#           bio = BytesIO()
#           self.fig.savefig(bio, format="png")
#           bio.seek(0)
#           return bio.read()
#   
#   
#   def magic_plot(core, doc_decorator=None):
#       """
#       Plot generator wrapper.
#   
#       Your function feeds the data, the wrapper gives control over the plot
#       to the user.
#   
#       Parameters
#       ----------
#       doc_decorator: str, optional
#           Append the docstring with the decorated parameters.
#   
#           By default, the global variable `_DOCSTRING_DECORATOR` will be used.
#   
#       Examples
#       --------
#       >>> @interplot.magic_plot
#       ... def plot_lines(samples=100, n=10, label="sigma={0}, mu={1}", fig=None):
#       ...     \"\"\"
#       ...     Plot Gaussian noise.
#       ...
#       ...     The function must accept the `fig` parameter from the decorator.
#       ...     \"\"\"
#       ...     for i in range(1, n+1):
#       ...         fig.add_line(
#       ...             np.random.normal(i*10,i,samples),
#       ...             label=label.format(i, i*10),
#       ...         )
#   
#       >>> plot_lines(samples=200, title="Normally distributed Noise")
#   
#       .. raw:: html
#           :file: ../source/plot_examples/gauss_plot_pty.html
#   
#       >>> plot_lines(
#       ...     samples=200, interactive=False, title="Normally distributed Noise")
#   
#       .. image:: plot_examples/gauss_plot_mpl.png
#           :alt: [matplotlib plot "Normally distributed Noise]
#       """
#       doc_decorator = (
#           conf._DOCSTRING_DECORATOR if doc_decorator is None else doc_decorator
#       )
#   
#       def wrapper(
#           *args,
#           fig=None,
#           skip_post_process=False,
#           interactive=None,
#           rows=1,
#           cols=1,
#           title=None,
#           xlabel=None,
#           ylabel=None,
#           xlim=None,
#           ylim=None,
#           xlog=False,
#           ylog=False,
#           shared_xaxes=False,
#           shared_yaxes=False,
#           column_widths=None,
#           row_heights=None,
#           fig_size=None,
#           dpi=None,
#           legend_loc=None,
#           legend_title=None,
#           legend_togglegroup=None,
#           color_cycle=None,
#           save_fig=None,
#           save_format=None,
#           save_config=None,
#           global_custom_func=None,
#           mpl_custom_func=None,
#           pty_custom_func=None,
#           pty_update_layout=None,
#           **kwargs,
#       ):
#           # init Plot
#           fig = Plot.init(
#               fig,
#               interactive=interactive,
#               rows=rows,
#               cols=cols,
#               title=title,
#               xlabel=xlabel,
#               ylabel=ylabel,
#               xlim=xlim,
#               ylim=ylim,
#               xlog=xlog,
#               ylog=ylog,
#               shared_xaxes=shared_xaxes,
#               shared_yaxes=shared_yaxes,
#               column_widths=column_widths,
#               row_heights=row_heights,
#               fig_size=fig_size,
#               dpi=dpi,
#               legend_loc=legend_loc,
#               legend_title=legend_title,
#               legend_togglegroup=legend_togglegroup,
#               color_cycle=color_cycle,
#               save_fig=save_fig,
#               save_format=save_format,
#               save_config=save_config,
#               global_custom_func=global_custom_func,
#               mpl_custom_func=mpl_custom_func,
#               pty_custom_func=pty_custom_func,
#               pty_update_layout=pty_update_layout,
#           )
#   
#           # execute core method
#           core(*args, fig=fig, **kwargs)
#   
#           # post-processing
#           if not skip_post_process:
#               fig.post_process()
#   
#           # return
#           return fig
#   
#       # rewrite _DOCSTRING_DECORATOR
#       wrapper.__doc__ = (
#           _rewrite_docstring(
#               core.__doc__,
#               doc_decorator,
#           )
#           + "\n"
#       )
#   
#       return wrapper
#   
#   
#   def magic_plot_preset(doc_decorator=None, **kwargs_preset):
#       """
#       Plot generator wrapper, preconfigured.
#   
#       Your function feeds the data, the wrapper gives control over the plot
#       to the user.
#   
#       Parameters
#       ----------
#       doc_decorator: str, optional
#           Append the docstring with the decorated parameters.
#   
#           By default, the global variable `conf._DOCSTRING_DECORATOR`
#           will be used.
#       strict_preset: bool, default: False
#           Prevents overriding the preset upon calling the decorated function.
#       **kwargs_preset: dict
#           Define presets for any keyword arguments accepted by `Plot`.
#   
#       Examples
#       --------
#       >>> @interplot.magic_plot_preset(
#       ...     title="Data view",
#       ...     interactive=False,
#       ...     strict_preset=False,
#       ... )
#       ... def line(
#       ...     data,
#       ...     fig,
#       ...     **kwargs,
#       ... ):
#       ...     fig.add_line(data)
#       ...
#       ... line([0,4,6,7], xlabel="X axis")
#   
#       [matplotlib figure, "Data view"]
#       """
#       strict_preset = kwargs_preset.get("strict_preset", False)
#       if "strict_preset" in kwargs_preset:
#           del kwargs_preset["strict_preset"]
#   
#       def decorator(core):
#   
#           def inner(*args_inner, **kwargs_inner):
#               # input clash check
#               # decorator is set to strict presets
#               if strict_preset:
#                   for kwarg in kwargs_inner:
#                       if kwarg in kwargs_preset:
#                           raise ValueError(
#                               "Keyword argument '" + kwarg + "' cannot be set.\n"
#                               "Overriding keyword arguments was deactivated with"
#                               " strict_preset=True in the decorator function."
#                           )
#   
#               # default behaviour: presets can be overridden
#               else:
#                   for kwarg in kwargs_inner:
#                       if kwarg in kwargs_preset:
#                           del kwargs_preset[kwarg]
#   
#               return magic_plot(core, doc_decorator=doc_decorator)(
#                   *args_inner,
#                   **kwargs_inner,
#                   **kwargs_preset,
#               )
#   
#           # rewrite DOCSTRING_DECORATOR
#           inner.__doc__ = _rewrite_docstring(
#               core.__doc__,
#               doc_decorator,
#               kwargs_remove=kwargs_preset if strict_preset else (),
#           )
#           return inner
#   
#       return decorator
#   
#   
#   @magic_plot
#   @wraps(Plot.add_line)
#   def line(
#       *args,
#       fig,
#       **kwargs,
#   ):
#       fig.add_line(*args, **kwargs)
#   
#   
#   @magic_plot
#   @wraps(Plot.add_scatter)
#   def scatter(
#       *args,
#       fig,
#       **kwargs,
#   ):
#       fig.add_scatter(*args, **kwargs)
#   
#   
#   @magic_plot
#   @wraps(Plot.add_linescatter)
#   def linescatter(
#       *args,
#       fig,
#       **kwargs,
#   ):
#       fig.add_linescatter(*args, **kwargs)
#   
#   
#   @magic_plot
#   @wraps(Plot.add_bar)
#   def bar(
#       *args,
#       fig,
#       **kwargs,
#   ):
#       fig.add_bar(*args, **kwargs)
#   
#   
#   @magic_plot
#   @wraps(Plot.add_fill)
#   def fill(
#       *args,
#       fig,
#       **kwargs,
#   ):
#       fig.add_fill(*args, **kwargs)
#   
#   
#   @magic_plot
#   @wraps(Plot.add_text)
#   def text(
#       *args,
#       fig,
#       **kwargs,
#   ):
#       fig.add_text(*args, **kwargs)
#   
#   
#   @magic_plot
#   @wraps(Plot.add_image)
#   def image(
#       *args,
#       fig,
#       **kwargs,
#   ):
#       fig.add_image(*args, **kwargs)
#   
#   
#   @magic_plot
#   @wraps(Plot.add_hist)
#   def hist(
#       *args,
#       fig,
#       **kwargs,
#   ):
#       fig.add_hist(*args, **kwargs)
#   
#   
#   @magic_plot
#   @wraps(Plot.add_boxplot)
#   def boxplot(
#       *args,
#       fig,
#       **kwargs,
#   ):
#       fig.add_boxplot(*args, **kwargs)
#   
#   
#   @magic_plot
#   @wraps(Plot.add_heatmap)
#   def heatmap(
#       *args,
#       fig,
#       **kwargs,
#   ):
#       fig.add_heatmap(*args, **kwargs)
#   
#   
#   @magic_plot
#   @wraps(Plot.add_regression)
#   def regression(
#       *args,
#       fig,
#       **kwargs,
#   ):
#       fig.add_regression(*args, **kwargs)
#   
#   
#   class ShowDataArray(NotebookInteraction):
#       """
#       Automatically display a `xarray.DataArray` in a Jupyter notebook.
#   
#       If the DataArray has more than two dimensions, provide default
#       sel or isel selectors to reduce to two dimensions.
#   
#       Parameters
#       ----------
#       data: xarray.DataArray
#       default_sel: dict, optional
#           Select a subset of a the DataArray by label.
#           Can be a slice or the type of the dimension.
#       default_isel: dict, optional
#           Select a subset of a the DataArray by integer count.
#           Can be a integer slice or an integer.
#       """
#   
#       def __init__(
#           self,
#           data,
#           default_sel=None,
#           default_isel=None,
#       ):
#           self.data = data
#           self.default_sel = default_sel
#           self.default_isel = default_isel
#   
#       @magic_plot
#       def _plot_core(
#           self,
#           data,
#           sel=None,
#           isel=None,
#           fig=None,
#           **kwargs,
#       ):
#           sel = {} if sel is None else sel
#           isel = {} if isel is None else isel
#           fig.add_line(data.sel(**sel).isel(**isel), **kwargs)
#   
#       def plot(
#           self,
#           *args,
#           sel=None,
#           isel=None,
#           **kwargs,
#       ):
#           """
#           Show the DataArray.
#   
#           Parameters
#           ----------
#           sel: dict, optional
#               Select a subset of a the DataArray by label.
#               Can be a slice or the type of the dimension.
#               If None, default_sel will be used.
#           isel: dict, optional
#               Select a subset of a the DataArray by integer count.
#               Can be a integer slice or an integer.
#               If None, default_isel will be used.
#   
#           Returns
#           -------
#           Plot.fig
#           """
#           sel = self.default_sel if sel is None else sel
#           isel = self.default_isel if isel is None else isel
#           return self._plot_core(self.data, *args, sel=sel, isel=isel, **kwargs)
#   
#   
#   class ShowDataset(ShowDataArray):
#       """
#       Automatically display a `xarray.Dataset` in a Jupyter notebook.
#   
#       Provide a default variable to display from the Dataset for
#       automatic display.
#       If the Dataset has more than two dimensions, provide default
#       sel or isel selectors to reduce to two dimensions.
#   
#       Parameters
#       ----------
#       data: xarray.DataArray
#       default_var: str, optional
#           Select the variable of the Dataset to display by label.
#       default_sel: dict, optional
#           Select a subset of a the Dataset by label.
#           Can be a slice or the type of the dimension.
#       default_isel: dict, optional
#           Select a subset of a the Dataset by integer count.
#           Can be a integer slice or an integer.
#       """
#   
#       def __init__(
#           self,
#           data,
#           default_var=None,
#           default_sel=None,
#           default_isel=None,
#       ):
#           self.data = data
#           self.default_var = default_var
#           self.default_sel = default_sel
#           self.default_isel = default_isel
#   
#       def plot(
#           self,
#           *args,
#           var=None,
#           sel=None,
#           isel=None,
#           **kwargs,
#       ):
#           """
#           Show a variable of the Dataset.
#   
#           Parameters
#           ----------
#           var: str, optional
#               Select the variable of the Dataset to display by label.
#               If None, default_var will be used.
#           sel: dict, optional
#               Select a subset of a the DataArray by label.
#               Can be a slice or the type of the dimension.
#               If None, default_sel will be used.
#           isel: dict, optional
#               Select a subset of a the DataArray by integer count.
#               Can be a integer slice or an integer.
#               If None, default_isel will be used.
#   
#           Returns
#           -------
#           Plot.fig
#           """
#   
#           var = self.default_var if var is None else var
#           sel = self.default_sel if sel is None else sel
#           isel = self.default_isel if isel is None else isel
#           return super()._plot_core(self.data[var], *args, sel=sel, isel=isel, **kwargs)
#   
###########################
#### interplot/conf.py ####
###########################
#   """
#   Modify the default behavior of the `interplot` package.
#   
#   Example:
#   
#   .. code-block:: python
#   
#      >>> # default behavior
#      ... interplot.line(x, y)
#   
#   .. raw:: html
#        :file: ../source/plot_examples/default_sin.html
#   
#   
#   .. code-block:: python
#   
#      >>> # modify default behavior
#      ... interplot.conf.INTERACTIVE = False
#      ... interplot.conf.COLOR_CYCLE[0] = "#000000"
#      ... interplot.conf.DPI = 150
#      ... interplot.conf.MPL_FIG_SIZE = (400, 400)
#      ...
#      ... # user-modified default behavior
#      ... interplot.line(x, y)
#   
#   
#   .. image:: plot_examples/default_sin.png
#       :alt: [matplotlib plot of a sinus curve]
#   """
#   
#   INTERACTIVE = True
#   """
#   Generate a `plotly` figure by default.
#   """
#   
#   COLOR_CYCLE = [  # optimised for color vision deficiencies
#       "#006BA4",
#       "#FF800E",
#       "#ABABAB",
#       "#595959",
#       "#5F9ED1",
#       "#C85200",
#       "#898989",
#       "#A2C8EC",
#       "#FFBC79",
#       "#CFCFCF",
#   ]
#   """
#   Colors to be cycled through by default.
#   
#   This default cycle is optimised for color vision deficiencies.
#   """
#   
#   DPI = 100
#   """
#   Figure resolution in dots per inch.
#   """
#   
#   
#   PTY_FIG_SIZE = (None, 500)  # px, None for responsive
#   """
#   Default figure size for the `plotly` backend, in px.
#   Use `None` for adaptive size.
#   """
#   
#   MPL_FIG_SIZE = (700, 450)  # px
#   """
#   Default figure size for the `matplotlib` backend, in px.
#   """
#   
#   
#   EXPORT_FORMAT = "png"
#   """
#   Default export format.
#   """
#   
#   EXPORT_REPLACE = {
#       "\n": "_",  # replace line breaks \\n with underscores
#       r"<\s*br\s*/?\s*>": "_",  # replace line breaks <br> with underscores
#       r"[ ]?[/|\\][ ]?": "_",  # replace slashes with underscores
#       " ": "-",  # replace spaces with dashes
#       "<.*?>": "",  # remove html tags
#       "[@!?,:;+*%&()=#|'\"<>]": "",  # remove special characters
#       r"\\": "_",  # replace backslashes with underscores
#       r"\s": "_",  # replace any special whitespace with underscores
#       # "^.*$": lambda s: s.group().lower()  # lowercase everything
#   }
#   """
#   Replace characters in the figure title to use as default filename.
#   
#   Use a dictionary with regex patterns as keys
#   and the (regex) replacement as values.
#   
#   Default
#   -------
#       - line breaks with underscores
#       - slashes with underscores
#       - spaces with dashes
#       - special characters are removed
#       - backslashes with underscores
#       - any special whitespace with underscores
#   
#   Optional
#   --------
#       - lowercase everything
#           ```ip.conf.EXPORT_REPLACE["^.*$"] = lambda s: s.group().lower()```
#   """
#   
#   
#   def GLOBAL_CUSTOM_FUNC(fig):
#       """
#       Apply a custom function on figures on finishing the plot.
#   
#       Will be called in `interplot.Plot.post_process()`.
#   
#       When overriding this function, make sure it accepts `fig`.
#       There is no need to return `fig`.
#   
#       Parameters
#       ----------
#       fig: interplot.Plot instance
#   
#       Examples
#       --------
#       >>> def global_custom_func(fig):
#       ...     # do your customisations
#       ...     fig.add_text(0, 0, "watermark", color="gray")
#       ...
#       ...     # also include default function
#       ...     conf.GLOBAL_CUSTOM_FUNC(fig)
#       ...
#       ... fig = interplot.Plot(
#       ...     global_custom_func=global_custom_func
#       ... )
#       ... fig.add_line(x, y)
#       ... fig.post_process()  # global_custom_func will be executed here
#       ... fig.show()
#       """
#       pass
#   
#   
#   def MPL_CUSTOM_FUNC(fig, ax):
#       """
#       Apply a custom function on matplotlib figures on finishing the plot.
#   
#       Will be called in `interplot.Plot.post_process()`.
#   
#       When overriding this function, make sure it accepts `fig` and `ax`
#       and returns `fig` and `ax`.
#   
#       Parameters
#       ----------
#       fig : matplotlib.figure.Figure
#           The figure object.
#       ax : matplotlib.axes.Axes
#           The axes object.
#   
#       Returns
#       -------
#       fig : matplotlib.figure.Figure
#           The modified figure object.
#       ax : matplotlib.axes.Axes
#           The modified axes object.
#   
#       Examples
#       --------
#       >>> def mpl_custom_func(fig, ax):
#       ...     # do your customisations
#       ...     fig.do_stuff()
#       ...     ax[0, 0].do_more()
#       ...
#       ...     # also include default function
#       ...     fig, ax = conf.MPL_CUSTOM_FUNC(fig, ax)
#       ...
#       ...     return fig, ax
#       ...
#       ... fig = interplot.Plot(
#       ...     interactive=False,
#       ...     mpl_custom_func=mpl_custom_func
#       ... )
#       ... fig.add_line(x, y)
#       ... fig.post_process()  # mpl_custom_func will be executed here
#       ... fig.show()
#       """
#       return fig, ax
#   
#   
#   def PTY_CUSTOM_FUNC(fig):
#       """
#       Apply a custom function on plotly figures on finishing the plot.
#   
#       Will be called in `interplot.Plot.post_process()`.
#   
#       When overriding this function, make sure it accepts `fig`
#       and returns `fig`.
#   
#       Parameters
#       ----------
#       fig : plotly.graph_objects.Figure
#           The figure object.
#   
#       Returns
#       -------
#       fig : plotly.graph_objects.Figure
#           The modified figure object.
#   
#       Examples
#       --------
#       >>> def pty_custom_func(fig):
#       ...     # do your customisations
#       ...     fig.do_stuff()
#       ...
#       ...     # also include default function
#       ...     fig = conf.PTY_CUSTOM_FUNC(fig)
#       ...
#       ...     return fig
#       ...
#       ... fig = interplot.Plot(
#       ...     interactive=True,
#       ...     pty_custom_func=pty_custom_func
#       ... )
#       ... fig.add_line(x, y)
#       ... fig.post_process()  # pty_custom_func will be executed here
#       ... fig.show()
#       """
#       return fig
#   
#   
#   PTY_LEGEND_TOGGLEGROUP = True
#   """
#   PLOTLY ONLY.
#   If True, elements with the same legend group will be toggled together
#   when clicking on a legend item.
#   """
#   
#   
#   PTY_UPDATE_LAYOUT = dict()
#   """
#   PLOTLY ONLY.
#   
#   Pass keyword arguments to plotly's
#   `plotly.graph_objects.Figure.update_layout(**pty_update_layout)`
#   
#   Default: None
#   """
#   
#   
#   PTY_CONFIG = dict(
#       displayModeBar=True,
#       displaylogo=False,
#   )
#   """
#   Modify the layout of the plotly figure.
#   
#   See https://plotly.com/python/reference/layout/ for reference.
#   """
#   
#   
#   PTY_LINE_STYLES = {
#       "-": "solid",
#       "--": "dash",
#       "-.": "dashdot",
#       ":": "dot",
#       "solid": "solid",
#       "dashed": "dash",
#       "dashdot": "dashdot",
#       "dotted": "dot",
#   }
#   """
#   Mapping for line styles for `plotly`.
#   """
#   
#   MPL_LINE_STYLES = {value: key for key, value in PTY_LINE_STYLES.items()}
#   """
#   Mapping for line styles for `matplotlib`.
#   """
#   
#   PTY_MARKERS = {
#       ".": "circle",
#       "s": "square",
#       "D": "diamond",
#       "P": "cross",
#       "X": "x",
#       "^": "triangle-up",
#       "v": "triangle-down",
#       "<": "triangle-left",
#       ">": "triangle-right",
#       "triangle-ne": "triangle-ne",
#       "triangle-se": "triangle-se",
#       "triangle-sw": "triangle-sw",
#       "triangle-nw": "triangle-nw",
#       "p": "pentagon",
#       "h": "hexagon",
#       "H": "hexagon2",
#       "8": "octagon",
#       "*": "star",
#       "hexagram": "hexagram",
#       "star-triangle-up": "star-triangle-up",
#       "star-triangle-down": "star-triangle-down",
#       "star-square": "star-square",
#       "star-diamond": "star-diamond",
#       "d": "diamond-tall",
#       "diamond-wide": "diamond-wide",
#       "hourglass": "hourglass",
#       "bowtie": "bowtie",
#       "circle-cross": "circle-cross",
#       "circle-x": "circle-x",
#       "square-cross": "square-cross",
#       "square-x": "square-x",
#       "diamond-cross": "diamond-cross",
#       "diamond-x": "diamond-x",
#       "+": "cross-thin",
#       "x": "x-thin",
#       "asterisk": "asterisk",
#       "hash": "hash",
#       "2": "y-up",
#       "1": "y-down",
#       "3": "y-left",
#       "4": "y-right",
#       "_": "line-ew",
#       "|": "line-ns",
#       "line-ne": "line-ne",
#       "line-nw": "line-nw",
#       6: "arrow-up",
#       7: "arrow-down",
#       4: "arrow-left",
#       5: "arrow-right",
#       "arrow-bar-up": "arrow-bar-up",
#       "arrow-bar-down": "arrow-bar-down",
#       "arrow-bar-left": "arrow-bar-left",
#       "arrow-bar-right": "arrow-bar-right",
#       "arrow": "arrow",
#       "arrow-wide": "arrow-wide",
#   }
#   """
#   Mapping for marker styles for `plotly`.
#   """
#   
#   PTY_MARKERS_LIST = list(PTY_MARKERS.values())
#   """
#   Possible line styles for `plotly`.
#   """
#   
#   MPL_MARKERS = {value: key for key, value in PTY_MARKERS.items()}
#   """
#   Mapping for marker styles for `matplotlib`.
#   """
#   MPL_MARKERS.update(
#       {  # next best matches
#           "triangle-nw": "^",
#           "triangle-ne": ">",
#           "triangle-se": "v",
#           "triangle-sw": "<",
#           "hexagram": "*",
#           "star-triangle-up": "^",
#           "star-triangle-down": "v",
#           "star-square": "s",
#           "star-diamond": "D",
#           "diamond-wide": "D",
#           "hourglass": "d",
#           "bowtie": "D",
#           "circle-cross": "+",
#           "circle-x": "x",
#           "cross-thin": "+",
#           "square-cross": "s",
#           "square-x": "s",
#           "diamond-cross": "D",
#           "diamond-x": "D",
#           "x-thin": "x",
#           "hash": "*",
#           "asterisk": "*",
#           "line-ne": "|",
#           "line-nw": "_",
#           "arrow-bar-up": 6,
#           "arrow-bar-down": 7,
#           "arrow-bar-left": 4,
#           "arrow-bar-right": 5,
#           "arrow": 6,
#           "arrow-wide": 6,
#       }
#   )
#   MPL_MARKERS_LIST = list(MPL_MARKERS.values())
#   """
#   Possible line styles for `matplotlib`.
#   """
#   
#   _REWRITE_DOCSTRING = True
#   
#   _DOCSTRING_DECORATOR = """
#       interactive: bool, default: True
#           Display an interactive plotly line plot
#           instead of the default matplotlib figure.
#       rows, cols: int, default: 1
#           Create a grid with `n` rows and `m` columns.
#       title: str, default: None
#           Plot title.
#       xlabel, ylabel: str or str tuple, default: None
#           Axis labels.
#   
#           Either one title for the entire axis or one for each row/column.
#       xlim, ylim: tuple of 2 numbers or nested, default: None
#           Axis range limits.
#   
#           In case of multiple rows/cols provide either:
#               - a tuple
#               - a tuple for each row
#               - a tuple for each row containing a tuple for each column.
#       xlog, ylog: bool or bool tuple, default: False
#           Logarithmic scale for the axis.
#   
#           Either one boolean for the entire axis or one for each row/column.
#       shared_xaxes, shared_yaxes: str, default: None
#           Define how multiple subplots share there axes.
#   
#           Options:
#               - "all" or True
#               - "rows"
#               - "columns" or "cols"
#               - None or False
#       column_widths, row_heights: tuple/list, default: None
#           Ratios of the width/height dimensions in each column/row.
#           Will be normalised to a sum of 1.
#       fig_size: tuple of 2x float, optional
#           Width, height in pixels.
#   
#           Default behavior defined by `conf.MPL_FIG_SIZE`
#           and `conf.PTY_FIG_SIZE`.
#           Plotly allows None for responsive size adapting to viewport.
#       dpi: int, default: 100
#           Plot resolution.
#       legend_loc: str, optional
#           MATPLOTLIB ONLY.
#   
#           Default:
#               - In case of 1 line: None
#               - In case of >1 line: "best" (auto-detect)
#       legend_title: str, default: None
#           MPL: Each subplot has its own legend, so a 2d list in the shape of
#           the subplots may be provided.
#   
#           PTY: Just provide a `str`.
#       legend_togglegroup: bool, default: False
#           PLOTLY ONLY.
#   
#           Whether legend items with the same group will be toggled together
#           when clicking on a legend item.
#       color_cycle: list, optional
#           Cycle through colors in the provided list.
#   
#           If left `None`, the default color cycle
#           `conf.COLOR_CYCLE` will be used.
#       save_fig: str or pathlib.Path, default: None
#           Provide a path to export the plot.
#   
#           Possible formats: png, jpg, svg, html, ...
#   
#           The figure will only be saved on calling the instance's
#           `.post_process()`.
#   
#           If a directory (or `True` for local directory) is provided,
#           the filename will be automatically generated based on the title.
#   
#           An iterable of multiple paths / filenames may be provided. In this case
#           the save command will be repeated for each element.
#       save_format: str, default: None
#           Provide a format for the exported plot, if not declared in `save_fig`.
#   
#           An iterable of multiple formats may be provided. In this case
#           the save command will be repeated for each element.
#       save_config: dict, default: None
#           Plotly only.
#   
#           Pass config options to
#           `plotly.graph_objects.Figure.write_html(config=save_config)`.
#       global_custom_func: function, default: None
#           Pass a function reference to further style the figures.
#   
#           >>> def global_custom_func(fig):
#           ...     # do your customisations
#           ...     fig.add_text(0, 0, "watermark", color="gray")
#           ...
#           ...     # also include default function
#           ...     conf.GLOBAL_CUSTOM_FUNC(fig)
#           ...
#           ... fig = interplot.Plot(
#           ...     global_custom_func=global_custom_func
#           ... )
#           ... fig.add_line(x, y)
#           ... fig.post_process()  # global_custom_func will be executed here
#           ... fig.show()
#       mpl_custom_func: function, default: None
#           MATPLOTLIB ONLY.
#   
#           Pass a function reference to further style the matplotlib graphs.
#           Function must accept `fig, ax` and return `fig, ax`.
#   
#           Note: `ax` always has `row` and `col` coordinates, even if the
#           plot is just 1x1.
#   
#           >>> def mpl_custom_func(fig, ax):
#           ...     # do your customisations
#           ...     fig.do_stuff()
#           ...     ax[0, 0].do_more()
#           ...
#           ...     # also include default function
#           ...     fig, ax = conf.MPL_CUSTOM_FUNC(fig, ax)
#           ...
#           ...     return fig, ax
#           ...
#           ... fig = interplot.Plot(
#           ...     interactive=False,
#           ...     mpl_custom_func=mpl_custom_func
#           ... )
#           ... fig.add_line(x, y)
#           ... fig.post_process()  # mpl_custom_func will be executed here
#           ... fig.show()
#       pty_custom_func: function, default: None
#           PLOTLY ONLY.
#   
#           Pass a function reference to further style the plotly graphs.
#           Function must accept `fig` and return `fig`.
#   
#           >>> def pty_custom_func(fig):
#           ...     # do your customisations
#           ...     fig.do_stuff()
#           ...
#           ...     # also include default function
#           ...     fig = conf.PTY_CUSTOM_FUNC(fig)
#           ...
#           ...     return fig
#           ...
#           ... fig = interplot.Plot(
#           ...     interactive=True,
#           ...     pty_custom_func=pty_custom_func
#           ... )
#           ... fig.add_line(x, y)
#           ... fig.post_process()  # pty_custom_func will be executed here
#           ... fig.show()
#       pty_update_layout: dict, default: None
#           PLOTLY ONLY.
#   
#           Pass keyword arguments to plotly's
#           `plotly.graph_objects.Figure.update_layout(**pty_update_layout)`
#   
#           Default: None
#   
#       Returns
#       -------
#       `interplot.Plot` instance
#   """
#   
###############################
#### interplot/__init__.py ####
###############################
#   """Create matplotlib/plotly hybrid plots with a few lines of code."""
#   
#   __all__ = [  # noqa F405
#       "conf",
#       "arraytools",
#       "iter",
#       "plot",
#       "debug",
#   ]
#   
#   from . import arraytools
#   from . import debug
#   from .iter import *  # noqa F403
#   from .plot import *  # noqa F403
#   
#   import sys
#   
#   if sys.version_info >= (3, 8):
#       import importlib.metadata
#   
#       try:
#           __version__ = importlib.metadata.version(__name__)
#   
#       except importlib.metadata.PackageNotFoundError:
#           pass
#   
#   else:
#       import pkg_resources
#   
#       try:
#           __version__ = pkg_resources.get_distribution(__name__).version
#   
#       except pkg_resources.DistributionNotFound:
#           pass
#   
############################
#### interplot/debug.py ####
############################
#   """
#   A small tool to eavesdrop on function calls and print or log them.
#   
#   Examples
#   --------
#   >>> @interplot.debug.wiretap
#   ... def func(foo):
#   ...     return foo + foo
#   ... interplot.debug.start_logging(save_to_log=True, verbose=True)
#   ... func("bar")
#   Wiretap log: {
#       "time": "2025-11-18 09:13:17.387600",
#       "function": "<function func at 0x323088680>",
#       "args": [
#           "bar"
#       ],
#       "kwargs": {},
#       "result": "barbar"
#   }
#   'barbar'
#   >>> interplot.debug.get_log(-1)["args"]  # get the last log entry
#   ('bar',)
#   """
#   
#   from functools import wraps
#   
#   import datetime as dt
#   
#   import json
#   
#   
#   _active = False
#   """Whether to watch for events."""
#   _save_to_log = True
#   """Whether to log the events."""
#   _verbose = True
#   """Whether to print the events to the output."""
#   
#   
#   log = []
#   """The logged events."""
#   
#   
#   def start_logging(save_to_log=None, verbose=None):
#       """
#       Start logging function calls.
#   
#       Parameters
#       ----------
#       save_to_log: bool, optional
#           Data will be accessible at `interplot.debug.log`.
#   
#           If undefined, the last setting will be used.
#   
#           By default, save_to_log is turned on once logging is activated.
#       verbose: bool, optional
#           Print payload to output on every call.
#   
#           By default, verbose is turned on once logging is activated.
#       """
#       global _active
#       _active = True
#   
#       if save_to_log is not None:
#           global _save_to_log
#           _save_to_log = save_to_log
#   
#       if verbose is not None:
#           global _verbose
#           _verbose = verbose
#   
#   
#   def stop_logging():
#       """Stop logging function calls."""
#       global _active
#       _active = False
#   
#   
#   def get_log(index=None):
#       """Get the logged events."""
#       global log
#   
#       if index is None:
#           return log
#   
#       return log[index]
#   
#   
#   def clear_log():
#       """Clear the log."""
#       global log
#       log = []
#   
#   
#   def wiretap(core):
#       """
#       Decorator to log input and output of the decorated function.
#   
#       Examples
#       --------
#       >>> @interplot.debug.wiretap
#       ... def func(foo):
#       ...     return foo + foo
#       """
#   
#       @wraps(core)
#       def inner(*args, core=core, **kwargs):
#           global _active, _save_to_log, _verbose
#   
#           if not _active:
#               return core(*args, **kwargs)
#   
#           res = core(*args, **kwargs)
#   
#           entry = dict(
#               time=dt.datetime.now(),
#               function=str(core),
#               args=args,
#               kwargs=kwargs,
#               result=res,
#           )
#   
#           if _verbose:
#               print("Wiretap log:", json.dumps(entry, default=str, indent=4))
#           if _save_to_log:
#               log.append(entry)
#   
#           return res
#   
#       return inner
#   
#############
#### END ####
#############
