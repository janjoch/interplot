"""Create matplotlib/plotly hybrid plots with a few lines of code."""

import pkg_resources

__all__ = [  # noqa F405
    "conf",
    "arraytools",
    "iter",
    "plot",
]

from . import arraytools
from .iter import *  # noqa F403
from .plot import *  # noqa F403

try:  # try except because of sphinx build --> DistributionNotFound Error
    __version__ = pkg_resources.get_distribution("interplot").version

except Exception:
    __version__ = "not detected!"
