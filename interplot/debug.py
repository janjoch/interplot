from functools import wraps

import datetime as dt

import json


PRINT = False
logging = False


log = []


def start_logging(verbose=True):
    """Start logging function calls."""
    global logging
    logging = True
    if verbose:
        global PRINT
        PRINT = True


def stop_logging():
    """Stop logging function calls."""
    global logging
    logging = False


def reset_log():
    """Reset the log."""
    global log
    log = []


def wiretap(core):
    """Decorator to wiretap functions for debugging."""

    @wraps(core)
    def wrapper(*args, core=core, **kwargs):
        """
        Wrapper function for a method.

        """
        if not logging and not PRINT:
            return core(*args, **kwargs)

        entry = dict(
            time=dt.datetime.now(),
            function=str(core),
            args=args,
            kwargs=kwargs,
        )

        res = core(*args, **kwargs)

        if PRINT:
            print(json.dumps(entry, default=str, indent=4))
        if logging:
            log.append(entry)

        return res
    
    return wrapper

