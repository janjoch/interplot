"""Tools to iterate Python objects."""

from warnings import warn
from datetime import datetime

from numpy import ndarray as np_ndarray
from pandas.core.series import Series as pd_Series


ITERABLE_TYPES = (tuple, list, dict, np_ndarray, pd_Series, range)
CUSTOM_DIGESTION = ((dict, (lambda dct: [elem for _, elem in dct.items()])),)

MUTE_STRICT_ZIP_WARNING = False


class NoZip:
    """Avoid iteration in zip() and zip_smart()"""
    def __init__(self, iterable):
        """
        Avoid iteration of an iterable data type in the zip function.

        Class allows iteration and subscription.

        Call the instance to release the original variable.

        Parameters
        ----------
        iterable
            Iterable variable which should be "hidden".
        """
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def __getitem__(self, item):
        return self.iterable[item]

    def __call__(self):
        return self.release()

    def __repr__(self):
        return "NoZip({})".format(self.iterable.__repr__())

    def release(self):
        """Return the original iterable variable."""
        return self.iterable


def _repeat(arg, iterable_types, maxlen, unpack_nozip):
    """
    If arg is not an instance of iterable_types, repeat maxlen times.

    Unpacks NoZip instances by default.
    """
    if isinstance(arg, iterable_types):
        return arg
    if unpack_nozip and isinstance(arg, NoZip):
        arg = arg()
    return (arg,) * maxlen


def zip_smart(*iterables, iterable_types=None, unpack_nozip=True, strict=True):
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
    unpack_nozip: bool, optional
        Unpack a NoZip-wrapped iterable.
        Default: True
    strict: bool, optional
        Fail if iterables are not the same length.
        Not supported in Python < 3.10.
        Default: True

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
        _repeat(arg, iterable_types, maxlen, unpack_nozip=unpack_nozip)
        for arg
        in iterables
    ]
    try:
        return zip(*iterables, strict=strict)

    # strict mode not implemented in Python<3.10
    except TypeError:
        if strict:
            if not MUTE_STRICT_ZIP_WARNING:
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
    In dictionaries, only the values are used.

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


def filter_nozip(iterable, no_iter_types=None, recursive=False, length=2):
    # input validation
    no_iter_types = (
        (float, int, datetime)
        if no_iter_types is None
        else no_iter_types
    )

    # non-iterable
    if not isinstance(iterable, ITERABLE_TYPES):
        return iterable

    # catch forbidden iterable
    if isinstance(iterable, ITERABLE_TYPES) and len(iterable) == length:
        all_allowed = True
        for elem in iterable:
            if not isinstance(elem, no_iter_types):
                all_allowed = False
                break
        if all_allowed:
            return NoZip(iterable)

    # otherwise recursively
    if recursive:
        return [filter_nozip(i, no_iter_types, length) for i in iterable]

    # no hit
    return iterable
