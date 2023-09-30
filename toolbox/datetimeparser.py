"""
Parse str timestamps to datetime.datetime.

Source: https://github.com/janjoch/toolbox

Examples:

```
>>> import toolbox.datetimeparser as dtp

>>> dtp.ymd("2023-05-08T14:30:02Z")
datetime.datetime(2023, 5, 8, 14, 30, 2)

>>> dtp.dmy("31.12.2023")
datetime.datetime(2023, 12, 31, 0, 0)

>>> dtp.dmy("31.12.23 14:30:02.123", microsecond_shift=3)
datetime.datetime(2023, 12, 31, 14, 30, 2, 123000)

>>> dtp.dmy("The moonlanding happened on 20.07.1969 20:17:40")
datetime.datetime(1969, 7, 20, 20, 17, 40)

>>> dtp.time("It is now 14:30:12")
datetime.time(14, 30, 12)

>>> dtp.iso_tight("20230508T143002Z")
datetime.datetime(2023, 5, 8, 14, 30, 2)
```
"""

import datetime as dt
import re
from warnings import warn

import numpy as np


AUTO_YEAR_COMPLETE = 2000
AUTO_YEAR_THRESHOLD = 100

REGEX_START = r".*?"
REGEX_END = r".*"
DELIMITER_SPACE = r"[\sTt_]"
DELIMITER_DATE = r"[-./]"
DELIMITER_TIME = r"[:.-]"

REGEX_TIME = (
    r"(?:{2}([0-9]{{1,2}})"  # hour
    r"{3}([0-9]{{1,2}}))?"  # minute
    r"(?:{3}([0-9]{{1,2}}))?"  # second
    r"(?:{3}([0-9]{{1,6}}))?{4}"  # sub-second
)


class DateTimeParsingError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "Error: %s" % self.value


def generic(
    time_str,
    pattern,
    order=None,
    swap=(),
    start=(),
    end=(),
    microsecond_shift=None,
    auto_year_complete=None,
    tzinfo=None,
):
    """
    Parse str timestamps to datetime.datetime.

    Source: https://github.com/janjoch/toolbox

    Parameters
    ----------
    time_str: str
        String to parse.
    pattern: str
        Regular expression pattern.
    order: tuple or list, optional
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
        Default: 2000
        Provide 0 to deactivate.
    tzinfo: datetime.tzinfo, optional
        Pass time zone information to datetime.datetime.

    Returns
    -------
    datetime.datetime if the pattern was able to match
    """
    # input verification
    auto_year_complete = (
        AUTO_YEAR_COMPLETE
        if auto_year_complete is None
        else auto_year_complete
    )

    # parse time_str
    match = re.match(pattern, time_str)
    if match:
        ints = np.array(match.groups(default=0), dtype=int)
    else:
        raise DateTimeParsingError(
            ("No timestamp found.\n\ntime_str:\n{}" "\n\npattern:\n{}").format(
                time_str, pattern
            )
        )

    # sort numbers year...miliseconds
    if order is None:
        for s in swap:
            ints[s[0]], ints[s[1]] = ints[s[1]], ints[s[0]]
    else:
        order = list(order)
        if len(ints) != len(order):
            warn(
                "The length of the order specification does not match"
                " the length of the detected numbers.\n"
                "This may lead to errors or incomplete timestamps."
            )
        ints = ints[order]
    ints = list(start) + list(ints) + list(end)

    # convert to microseconds
    if microsecond_shift:
        if len(ints) >= 7:
            ints[6] = ints[6] * 10**microsecond_shift
        else:
            warn(
                "No sub-second information is present in the pattern,"
                " but a microsecond shift was provided."
            )

    # abbrevated year completion
    if ints[0] < AUTO_YEAR_THRESHOLD:
        ints[0] = ints[0] + auto_year_complete

    return dt.datetime(*ints, tzinfo=tzinfo)


def ymd(
    time_str,
    regex_start=None,
    regex_end=None,
    delimiter_date=None,
    delimiter_space=None,
    delimiter_time=None,
    microsecond_shift=None,
    auto_year_complete=None,
    tzinfo=None,
):
    r"""
    Parse year-month-day date-/timestamps to datetime.datetime.

    Input accepts [year-]month-day[ -hour-minute[-second[-microsecond]]]

    Input format examples:
        23.12.31
        2023.12.31
        2023.12.31 23:59
        2023.12.31 23:59:59
        2023.12.31 23:59:59.123456
        2023-12-31T23:59:59.123456Z (ISO 8601)

    Output Format datetime.datetime

    Source: https://github.com/janjoch/toolbox

    Parameters
    ----------
    time_str: str
        String to parse.
    regex_start: str, optional
        Regex patterns that precedes the timestamp.
        Default: .*?
    regex_end: str, optional
        Regex patterns that follows the timestamp.
        Default: .*
    delimiter_date: str, optional
        Regex patterns in between date integers.
        Default: [-./]
    delimiter_space: str, optional
        Regex pattern in between date and time.
        Default: [\sTt_]
    delimiter_time: str, optional
        Regex patterns in between time integers.
        Default: [-./]
    microsecond_shift: int, optional
        Number of decimal places to shift sub-second value right
        to get microseconds.
        Example: To convert ms to us, use microseconds_shift=3.
    auto_year_complete: int, optional
        If the parsed year is below 100, add a number of years.
    tzinfo: datetime.tzinfo, optional
        Pass time zone information to datetime.datetime.

    Returns
    -------
    datetime.datetime
    """
    # input verification
    regex_start = (
        REGEX_START
        if regex_start is None
        else regex_start
    )
    regex_end = (
        REGEX_END
        if regex_end is None
        else regex_end
    )
    delimiter_date = (
        DELIMITER_DATE
        if delimiter_date is None
        else delimiter_date
    )
    delimiter_space = (
        DELIMITER_SPACE
        if delimiter_space is None
        else delimiter_space
    )
    delimiter_time = (
        DELIMITER_TIME
        if delimiter_time is None
        else delimiter_time
    )

    return generic(
        time_str=time_str,
        pattern=(
            r"{0}(?:([0-9]{{2,4}}){1})?"  # year
            r"([0-9]{{1,2}})"  # month
            r"{1}([0-9]{{1,2}})"  # day
            + REGEX_TIME
        ).format(
            regex_start,
            delimiter_date,
            delimiter_space,
            delimiter_time,
            regex_end,
        ),
        microsecond_shift=microsecond_shift,
        auto_year_complete=auto_year_complete,
        tzinfo=tzinfo,
    )


def dmy(
    time_str,
    regex_start=None,
    regex_end=None,
    delimiter_date=None,
    delimiter_space=None,
    delimiter_time=None,
    microsecond_shift=None,
    auto_year_complete=None,
    tzinfo=None,
):
    r"""
    Parse day-month-year date-/timestamps to datetime.datetime.

    Input accepts day-month[-year][ -hour-minute[-second[-microsecond]]]

    Input format examples:
        31.12
        31.12.23
        31.12.2023
        31.12.2023 23:59
        31.12.2023 23:59:59
        31.12.2023 23:59:59.123456
        31-12-2023_23-59-59-123456Z

    Output Format datetime.datetime

    Source: https://github.com/janjoch/toolbox

    Parameters
    ----------
    time_str: str
        String to parse.
    regex_start: str, optional
        Regex patterns that precedes the timestamp.
        Default: .*?
    regex_end: str, optional
        Regex patterns that follows the timestamp.
        Default: .*
    delimiter_date: str, optional
        Regex patterns in between date integers.
        Default: [-./]
    delimiter_space: str, optional
        Regex pattern in between date and time.
        Default: [\sTt_]
    delimiter_time: str, optional
        Regex patterns in between time integers.
        Default: [-./]
    microsecond_shift: int, optional
        Number of decimal places to shift sub-second value right
        to get microseconds.
        Example: To convert ms to us, use microseconds_shift=3.
    auto_year_complete: int, optional
        If the parsed year is below 100, add a number of years.
    tzinfo: datetime.tzinfo, optional
        Pass time zone information to datetime.datetime.

    Returns
    -------
    datetime.datetime
    """
    # input verification
    regex_start = (
        REGEX_START
        if regex_start is None
        else regex_start
    )
    regex_end = (
        REGEX_END
        if regex_end is None
        else regex_end
    )
    delimiter_date = (
        DELIMITER_DATE
        if delimiter_date is None
        else delimiter_date
    )
    delimiter_space = (
        DELIMITER_SPACE
        if delimiter_space is None
        else delimiter_space
    )
    delimiter_time = (
        DELIMITER_TIME
        if delimiter_time is None
        else delimiter_time
    )

    return generic(
        time_str=time_str,
        pattern=(
            r"{0}([0-9]{{1,2}})"  # day
            r"{1}([0-9]{{1,2}})"  # month
            r"(?:{1}([0-9]{{2,4}}))?"  # year
            + REGEX_TIME
        ).format(
            regex_start,
            delimiter_date,
            delimiter_space,
            delimiter_time,
            regex_end,
        ),
        swap=((0, 2),),
        microsecond_shift=microsecond_shift,
        auto_year_complete=auto_year_complete,
        tzinfo=tzinfo,
    )


def mdy(
    time_str,
    regex_start=None,
    regex_end=None,
    delimiter_date=None,
    delimiter_space=None,
    delimiter_time=None,
    microsecond_shift=None,
    auto_year_complete=None,
    tzinfo=None,
):
    r"""
    Parse month-day-year date-/timestamps to datetime.datetime.

    Input accepts month-day[-year][ -hour-minute[-second[-microsecond]]]

    Input format examples:
        12/31
        12/31/23
        12/31/2023
        12/31/2023 23:59
        12/31/2023 23:59:59
        12/31/2023 23:59:59.123456
        12-31-2023_23-59-59-123456Z

    Output Format datetime.datetime

    Source: https://github.com/janjoch/toolbox

    Parameters
    ----------
    time_str: str
        String to parse.
    regex_start: str, optional
        Regex patterns that precedes the timestamp.
        Default: .*?
    regex_end: str, optional
        Regex patterns that follows the timestamp.
        Default: .*
    delimiter_date: str, optional
        Regex patterns in between date integers.
        Default: [-./]
    delimiter_space: str, optional
        Regex pattern in between date and time.
        Default: [\sTt_]
    delimiter_time: str, optional
        Regex patterns in between time integers.
        Default: [-./]
    microsecond_shift: int, optional
        Number of decimal places to shift sub-second value right
        to get microseconds.
        Example: To convert ms to us, use microseconds_shift=3.
    auto_year_complete: int, optional
        If the parsed year is below 100, add a number of years.
    tzinfo: datetime.tzinfo, optional
        Pass time zone information to datetime.datetime.

    Returns
    -------
    datetime.datetime
    """
    # input verification
    regex_start = (
        REGEX_START
        if regex_start is None
        else regex_start
    )
    regex_end = (
        REGEX_END
        if regex_end is None
        else regex_end
    )
    delimiter_date = (
        DELIMITER_DATE
        if delimiter_date is None
        else delimiter_date
    )
    delimiter_space = (
        DELIMITER_SPACE
        if delimiter_space is None
        else delimiter_space
    )
    delimiter_time = (
        DELIMITER_TIME
        if delimiter_time is None
        else delimiter_time
    )

    return generic(
        time_str=time_str,
        pattern=(
            r"{0}([0-9]+)"  # month
            r"{1}([0-9]+)"  # day
            r"(?:{1}([0-9]+))"  # year
            + REGEX_TIME
        ).format(
            regex_start,
            delimiter_date,
            delimiter_space,
            delimiter_time,
            regex_end,
        ),
        swap=(
            (0, 1),
            (0, 2),
        ),
        microsecond_shift=microsecond_shift,
        auto_year_complete=auto_year_complete,
        tzinfo=tzinfo,
    )


def time(
    time_str,
    date=None,
    regex_start=None,
    regex_end=None,
    delimiter_time=None,
    microsecond_shift=None,
    auto_year_complete=None,
    tzinfo=None,
):
    r"""
    Parse timestamps to datetime.datetime.

    Input accepts hour-minute[-second[-microsecond]]

    Input format examples:
        11:30
        11:30:59
        11:30:59.123456
        11-30-59-123456

    Output Format datetime.datetime

    Source: https://github.com/janjoch/toolbox

    Parameters
    ----------
    time_str: str
        String to parse.
    date: datetime.datetime or list, optional
        Date to complete the timestamp
        A list must provide [year, month, day]
        If no date is provided, the method will return
        a datetime.time object instead of datetime.datetime
        Default: None
    regex_start: str, optional
        Regex patterns that precedes the timestamp.
        Default: .*?
    regex_end: str, optional
        Regex patterns that follows the timestamp.
        Default: .*
    delimiter_time: str, optional
        Regex patterns in between time integers.
        Default: [-./]
    microsecond_shift: int, optional
        Number of decimal places to shift sub-second value right
        to get microseconds.
        Example: To convert ms to us, use microseconds_shift=3.
    auto_year_complete: int, optional
        If the parsed year is below 100, add a number of years.
    tzinfo: datetime.tzinfo, optional
        Pass time zone information to datetime.datetime.

    Returns
    -------
    if date is None:
        datetime.time
    else:
        datetime.datetime
    """
    # input verification
    regex_start = (
        REGEX_START
        if regex_start is None
        else regex_start
    )
    regex_end = (
        REGEX_END
        if regex_end is None
        else regex_end
    )
    delimiter_time = (
        DELIMITER_TIME
        if delimiter_time is None
        else delimiter_time
    )
    return_time = date is None
    if return_time:
        date = [2001, 1, 1]
    elif isinstance(date, dt.datetime):
        date = tuple(date.timetuple())[:3]
    elif len(date) != 3:
        raise ValueError(
            "date parameter must be a list or tuple "
            "containing [year, month, day]"
        )

    result = generic(
        time_str=time_str,
        pattern=REGEX_TIME.format(
            None,
            None,
            regex_start,
            delimiter_time,
            regex_end,
        ),
        start=date,
        microsecond_shift=microsecond_shift,
        auto_year_complete=auto_year_complete,
        tzinfo=tzinfo,
    )

    # no date specified: datetime.time
    if return_time:
        return result.time()

    # date specified: datetime.datetime
    return result


def iso_tight(
    time_str,
    regex_start=r"^",
    regex_end=r"$",
    regex_t="T",
    tzinfo=None,
):
    r"""
    Parse tight ISO 8601 date-/timestamp.

    Format: 20233112T2359[59][Z]

    Output Format datetime.datetime

    Source: https://github.com/janjoch/toolbox

    Parameters
    ----------
    time_str: str
        String to parse.
    regex_start: str, optional
        Regex patterns that precedes the timestamp.
        Default: ^
        (Only matches at the beginning of the string.)
        Provide None to override with wildcard regex.
    regex_t: str, optional
        Replace the ISO T date/time separator with something else.
        Can be any regex pattern.
        Default: T
    regex_end: str, optional
        Regex patterns that follows the timestamp.
        Default: $
        (Only matches at the end of the string.)
        Provide None to override with wildcard regex.
    tzinfo: datetime.tzinfo, optional
        Pass time zone information to datetime.datetime.

    Returns
    -------
    datetime.datetime
    """
    # input verification
    regex_start = (
        REGEX_START
        if regex_start is None
        else regex_start
    )
    regex_end = (
        REGEX_END
        if regex_end is None
        else regex_end
    )

    return generic(
        time_str=time_str,
        pattern=(
            r"{}([0-9]{{4}})([0-9]{{2}})([0-9]{{2}})"
            r"{}([0-9]{{2}})([0-9]{{2}})([0-9]{{2}})?Z?{}"
        ).format(regex_start, regex_t, regex_end),
        tzinfo=tzinfo,
    )
