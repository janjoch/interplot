"""Gaussian probability density function."""


import math

import numpy as np


def gauss(x, A, mu=None, sigma=None):
    if mu is None:
        A, mu, sigma = A
    return (
        A
        * np.exp(-np.power((x - mu) / sigma, 2) / 2)
        / (sigma * math.sqrt(2 * math.pi))
    )
