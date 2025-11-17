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

import sys

if sys.version_info >= (3, 8):
    import importlib.metadata

    try:
        __version__ = importlib.metadata.version(__name__)

    except importlib.metadata.PackageNotFoundError:
        pass

else:
    import pkg_resources

    try:
        __version__ = pkg_resources.get_distribution(__name__).version

    except pkg_resources.DistributionNotFound:
        pass
