"""
A small tool to eavesdrop on function calls and print or log them.

Examples
--------
>>> @interplot.debug.wiretap
... def func(foo):
...     return foo + foo
... interplot.debug.start_logging(save_to_log=True, verbose=True)
... func("bar")
Wiretap log: {
    "time": "2025-11-18 09:13:17.387600",
    "function": "<function func at 0x323088680>",
    "args": [
        "bar"
    ],
    "kwargs": {},
    "result": "barbar"
}
'barbar'
>>> interplot.debug.get_log(-1)["args"]  # get the last log entry
('bar',)
"""

from functools import wraps

import datetime as dt

import json


_active = False
"""Whether to watch for events."""
_save_to_log = True
"""Whether to log the events."""
_verbose = True
"""Whether to print the events to the output."""


log = []
"""The logged events."""


def start_logging(save_to_log=None, verbose=None):
    """
    Start logging function calls.

    Parameters
    ----------
    save_to_log: bool, optional
        Data will be accessible at `interplot.debug.log`.

        If undefined, the last setting will be used.

        By default, save_to_log is turned on once logging is activated.
    verbose: bool, optional
        Print payload to output on every call.

        By default, verbose is turned on once logging is activated.
    """
    global _active
    _active = True

    if save_to_log is not None:
        global _save_to_log
        _save_to_log = save_to_log

    if verbose is not None:
        global _verbose
        _verbose = verbose


def stop_logging():
    """Stop logging function calls."""
    global _active
    _active = False


def get_log(index=None):
    """Get the logged events."""
    global log

    if index is None:
        return log

    return log[index]


def clear_log():
    """Clear the log."""
    global log
    log = []


def wiretap(core):
    """
    Decorator to log input and output of the decorated function.

    Examples
    --------
    >>> @interplot.debug.wiretap
    ... def func(foo):
    ...     return foo + foo
    """

    @wraps(core)
    def inner(*args, core=core, **kwargs):
        global _active, _save_to_log, _verbose

        if not _active:
            return core(*args, **kwargs)

        res = core(*args, **kwargs)

        entry = dict(
            time=dt.datetime.now(),
            function=str(core),
            args=args,
            kwargs=kwargs,
            result=res,
        )

        if _verbose:
            print("Wiretap log:", json.dumps(entry, default=str, indent=4))
        if _save_to_log:
            log.append(entry)

        return res

    return inner
