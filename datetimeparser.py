# -*- coding: utf-8 -*-
"""
Parse str timestamps to datetime.datetime.

Source: https://github.com/Janjoch/toolbox

Examples:
    
```
>>>import datetimeparser as dtp

>>> dtp.dmy("31.12.2023")
datetime.datetime(2023, 12, 31, 0, 0)

>>> dtp.dmy("1.2.23 18:40:59.123456")
datetime.datetime(2023, 2, 1, 18, 40, 59, 123456)

>>> dtp.dmy("1.2.23 18:40:59.123", microsecond_shift=3)
datetime.datetime(2023, 2, 1, 18, 40, 59, 123000)
```
"""

import datetime as dt
import re

import numpy as np

def generic(
    time_str,
    pattern,
    order=None,
    swap=(),
    start=(),
    end=(),
    microsecond_shift=None,
    auto_year_complete=2000,
    tzinfo=None,
):
    """
    Parse str timestamps to datetime.datetime.

    Source: https://github.com/Janjoch/toolbox
    
    Parameters
    ----------
    time_str: str
        String to Parse.
    pattern: str
        Regular expression pattern.
    order: list, optional
        If numbers are not in order (starting with year),
        define the order in which they are found in the pattern.
        Overrides swap.
    swap: tuple of tuples, 2 ints, optional
        Swap two numbers to bring them in order (starting with year).
        Is overridden by order.
    start: tuple, optional
        Numbers to add for datetime generation at the beginning:
        Year, month, ...
    end: tuple, optional
        Numbers to add for datetime generation at the end:
        ... seconds, microseconds, ...
    microsecond_shift: int, optional
        Number of decimal places to shift sub-second value right
        to get microseconds.
        Example: To convert ms to us, use microseconds_shift=3.
    auto_year_complete: int, optional
        If the parsed year is below 100, add a number of years.
    tzinfo: datetime.tzinfo, optional
        Pass time zone information to datetime.datetime

    Returns
    -------
    datetime.datetime if the pattern was able to match
    """
    match = re.match(pattern, time_str)
    if(match):
        ints = np.array(match.groups(default=0), dtype=int)
    else:
        return False
    if(order is None):
        for s in swap:
            ints[s[0]], ints[s[1]] = ints[s[1]], ints[s[0]]
    else:
        ints = ints[order]
    ints = list(start) + list(ints) + list(end)
    if(microsecond_shift):
        if(len(ints) >= 7):
            ints[6] = ints[6] * 10**microsecond_shift
        else:
            print(
                "Warning: no sub-second information is present in the pattern,"
                " but a microsecond shift was provided."
            )
    if(ints[0] < 100):
        ints[0] = ints[0] + auto_year_complete
    return dt.datetime(*ints, tzinfo=tzinfo)

def ymd(
    time_str,
    start=r".*?",
    end=r".*",
    space=r"[\s_]",
    date_delimiter=r"[-.]",
    time_delimiter=r"[:.-]",
    microsecond_shift=None,
    auto_year_complete=2000,
):
    r"""
    Parse timestamps to datetime.datetime.

    Input accepts year-month-day[ -hour-minute[-second[-microsecond]]]

    Input format examples:
        23.12.31
        2023.12.31
        2023.12.31 11:30
        2023.12.31 11:30:59
        2023.12.31 11:30:59.123456
        2023-12-31_11-30-59-123456

    Output Format datetime.datetime

    Source: https://github.com/Janjoch/toolbox

    Parameters
    ----------
    time_str: str
        String to Parse.
    start, end: str, optional
        Regex patterns that precedes / follows the timestamp.
        Default: [ \t\n\r\f\v] and more
    space: str, optional
        Regex pattern in between date and time.
    date_delimiter, time_delimiter: str, optional
        Regex patterns in between date or time integers respectively.
    microsecond_shift: int, optional
        Number of decimal places to shift sub-second value right
        to get microseconds.
        Example: To convert ms to us, use microseconds_shift=3.
    auto_year_complete: int, optional
        If the parsed year is below 100, add a number of years.

    Returns
    -------
    datetime.datetime
    """
    return generic(
        time_str=time_str,
        pattern=(
            r'{0}([0-9]+)'
            r'{1}([0-9]+)'
            r'{1}([0-9]+)'
            r'(?:{2}([0-9]+)'
            r'{3}([0-9]+))?'
            r'(?:{3}([0-9]+))?'
            r'(?:{3}([0-9]+))?{4}'
        ).format(start, date_delimiter, space, time_delimiter, end),
        microsecond_shift=microsecond_shift,
        auto_year_complete=auto_year_complete,
    )

def dmy(
    time_str,
    start=r".*?",
    end=r".*",
    space=r"[\s_]",
    date_delimiter=r"[-.]",
    time_delimiter=r"[:.-]",
    microsecond_shift=None,
    auto_year_complete=2000,
):
    r"""
    Parse timestamps to datetime.datetime.

    Input accepts day-month[-year][ -hour-minute[-second[-microsecond]]]

    Input format examples:
        31.12
        31.12.23
        31.12.2023
        31.12.2023 11:30
        31.12.2023 11:30:59
        31.12.2023 11:30:59.123456
        31-12-2023_11-30-59-123456

    Output Format datetime.datetime

    Source: https://github.com/Janjoch/toolbox

    Parameters
    ----------
    time_str: str
        String to Parse.
    start, end: str, optional
        Regex patterns that precedes / follows the timestamp.
        Default: [ \t\n\r\f\v] and more
    space: str, optional
        Regex pattern in between date and time.
    date_delimiter, time_delimiter: str, optional
        Regex patterns in between date or time integers respectively.
    microsecond_shift: int, optional
        Number of decimal places to shift sub-second value right
        to get microseconds.
        Example: To convert ms to us, use microseconds_shift=3.
    auto_year_complete: int, optional
        If the parsed year is below 100, add a number of years.

    Returns
    -------
    datetime.datetime
    """
    return generic(
        time_str=time_str,
        pattern=(
            r'{0}([0-9]+)'
            r'{1}([0-9]+)'
            r'(?:{1}([0-9]+))?'
            r'(?:{2}([0-9]+)'
            r'{3}([0-9]+))?'
            r'(?:{3}([0-9]+))?'
            r'(?:{3}([0-9]+))?{4}'
        ).format(start, date_delimiter, space, time_delimiter, end),
        swap=((0,2),),
        microsecond_shift=microsecond_shift,
        auto_year_complete=auto_year_complete,
    )

def mdy(
    time_str,
    start=r".*?",
    end=r".*",
    space=r"[\s_]",
    date_delimiter=r"[-./]",
    time_delimiter=r"[:./-]",
    microsecond_shift=None,
    auto_year_complete=2000,
):
    r"""
    Parse timestamps to datetime.datetime.

    Input accepts month-day[-year][ -hour-minute[-second[-microsecond]]]

    Input format examples:
        12/31
        12/31/23
        12/31/2023
        12/31/2023 11:30
        12/31/2023 11:30:59
        12/31/2023 11:30:59.123456
        12-31-2023_11-30-59-123456

    Output Format datetime.datetime

    Source: https://github.com/Janjoch/toolbox

    Parameters
    ----------
    time_str: str
        String to Parse.
    start, end: str, optional
        Regex patterns that precedes / follows the timestamp.
        Default: [ \t\n\r\f\v] and more
    space: str, optional
        Regex pattern in between date and time.
    date_delimiter, time_delimiter: str, optional
        Regex patterns in between date or time integers respectively.
    microsecond_shift: int, optional
        Number of decimal places to shift sub-second value right
        to get microseconds.
        Example: To convert ms to us, use microseconds_shift=3.
    auto_year_complete: int, optional
        If the parsed year is below 100, add a number of years.

    Returns
    -------
    datetime.datetime
    """
    return generic(
        time_str=time_str,
        pattern=(
            r'{0}([0-9]+)'
            r'{1}([0-9]+)'
            r'(?:{1}([0-9]+))?'
            r'(?:{2}([0-9]+)'
            r'{3}([0-9]+))?'
            r'(?:{3}([0-9]+))?'
            r'(?:{3}([0-9]+))?{4}'
        ).format(start, date_delimiter, space, time_delimiter, end),
        swap=((0,1),(0,2),),
        microsecond_shift=microsecond_shift,
        auto_year_complete=auto_year_complete,
    )
