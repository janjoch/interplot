"""Janosch's small Python code snippets making life a bit easier."""

import pkg_resources

__all__ = [  # noqa F405
    "accelerate",
    "arraytools",
    "color",
    "convert",
    "datetimeparser",
    "gauss",
    "iter",
    "plot",
]

from . import *  # noqa F403

try:  # try except because of sphinx build --> DistributionNotFound Error
    __version__ = pkg_resources.get_distribution("toolbox").version

except Exception:
    __version__ = "not detected!"
