"""Tools to iterate Python objects."""

from warnings import warn

from numpy import ndarray as np_ndarray
from pandas.core.series import Series as pd_Series


ITERABLE_TYPES = (tuple, list, dict, np_ndarray, pd_Series)
CUSTOM_DIGESTION = ((dict, (lambda dct: [elem for _, elem in dct.items()])),)


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
    iterable_types: tuple of types, optional
        If iterable is one of these types, hand to zip() directly without
        repeating.
        Default: (tuple, list, np.ndarray, pandas.Series)
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
        if isinstance(arg, iterable_types):
            if len(arg) > maxlen:
                maxlen = len(arg)
    iterables = [
        arg if isinstance(arg, iterable_types) else (arg,) * maxlen
        for arg
        in iterables
    ]
    try:
        return zip(*iterables, strict=strict)

    # strict mode not implemented in Python<3.10
    except TypeError:
        if strict:
            warn(
                "zip's strict mode not supported in Python<3.10.\n\n"
                "Falling back to non-strict mode."
            )
        return zip(*iterables)


def sum_nested(
    inp,
    iterable_types=None,
    depth=-1,
    custom_digestion=None,
):
    """
    Add up all values in iterable objects.

    Nested structures are added up recursively.
    Dictionaries are

    Parameters
    ----------
    inp: iterable
        Object to iterate over.
        If it is not a iterable type, the object itself is returned.
    iterable_types: tuple of types, optional
        If iterable is one of these types, hand to zip() directly without
        repeating.
        Default: (tuple, list, np.ndarray, pandas.Series)
    depth: int, optional
        Maximum depth to recurse.
        Set to -1 to recurse infinitely.
        Default -1.
    custom_digestion: tuple of tuples, optional
        Each element of the tuple must be a tuple of the following structure:
            (
                type or tuple of types,
                lambda function to digest the elements,
            )
        The result of the lambda function will then be treated
        like the new type.
        By default, toolbox.iter.CUSTOM_DIGESTION will be used:
            Dicts will be digested to a list of their values.

    Returns
    -------
    sum
    """
    # input verification
    depth = -1 if depth is None else depth
    iterable_types = iterable_types or ITERABLE_TYPES
    custom_digestion = custom_digestion or CUSTOM_DIGESTION

    # custom digestion
    for type_, lambda_ in custom_digestion:
        if isinstance(inp, type_):
            inp = lambda_(inp)

    # if is not iterable, return as-is
    if not isinstance(inp, ITERABLE_TYPES):
        return inp

    # check recursion level
    if depth is None or depth == 0:
        raise TypeError(
            (
                "Iterable type detected, but recursion has reached "
                "its maximum depth.\n\n"
                "Element:\n{}\n\n"
                "Type:\n{}"
            ).format(str(inp), str(type(inp)))
        )

    # iterate
    val = 0
    for elem in inp:

        # sum_nested only returns non-iterable types
        val += sum_nested(
            elem,
            iterable_types=iterable_types,
            depth=depth - 1,
            custom_digestion=custom_digestion,
        )

    return val
