"""Tools to iterate python objects."""


import numpy as np


ITERABLE_TYPES = (tuple, list, dict, np.ndarray)


def zip_smart(*iterables, iterable_types=None, strict=True):
    """
    Iterate over several iterables in parallel,
    producing tuples with an item from each one.

    Like Python's builtin zip() function,
    but if an argument is not iterable, it will be repeated each iteration.

    Parameters
    ----------
    *iterables: misc
        Elements to iterate or repeat.
    iterable_types: tuple of types
        If iterable is one of these types, hand to zip() directly without
        repeating.
        Default: (tuple, list, np.ndarray)
    strict: bool, optional
        Fail if iterables are not the same length. The default is True.

    Returns
    -------
    zip object
        Use it as you would use zip()
    """
    iterable_types = iterable_types or ITERABLE_TYPES
    iterables = list(iterables)
    maxlen = 1
    for arg in iterables:
        if(isinstance(arg, iterable_types)):
            if(len(arg) > maxlen):
                maxlen = len(arg)
    iterables = [
        arg
        if isinstance(arg, iterable_types)
        else (arg, ) * maxlen
        for arg
        in iterables
    ]
    return zip(*iterables, strict=strict)


def sum_nested(inp):
    """Add up all values in dicts, lists or tuples.

    Nested structures are added up recursively.
    """
    if(type(inp) is dict):
        inp = [elem for key, elem in inp.items()]

    if(type(inp) is list or type(inp) is tuple):
        val = 0
        for elem in inp:
            if(
                type(elem) is dict
                or type(elem) is list
                or type(elem) is tuple
            ):
                val += sum_nested(elem)

            else:
                val += elem

        return(val)

    return(inp)
