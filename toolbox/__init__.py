"""Janosch's small Python code snippets making life a bit easier."""

import pkg_resources
from pathlib import Path

__all__ = [
    "accelerate",
    "arraytools",
    "color",
    "convert",
    "datetimeparser",
    "gauss",
    "iter",
]

from . import *
#from .toolbox import *

try:  # try except because of sphinx build --> DistributionNotFound Error
    __version__ = pkg_resources.get_distribution("toolbox").version

except Exception:
    __version__ = "not detected!"
