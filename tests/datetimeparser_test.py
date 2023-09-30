import datetime as dt

import pytest

from toolbox import datetimeparser as dtp


@pytest.mark.parametrize(
    ("input", "datetime"),
    (
        ("23.12.31", dt.datetime(2023, 12, 31)),
        ("2023.12.31", dt.datetime(2023, 12, 31)),
        ("2023.12.31 23:59", dt.datetime(2023, 12, 31, 23, 59)),
        ("2023.12.31 23:59:59", dt.datetime(2023, 12, 31, 23, 59, 59)),
        (
            "2023.12.31 23:59:59.123456",
            dt.datetime(2023, 12, 31, 23, 59, 59, 123456),
        ),
        (
            "2023.12.31 23:59:59.123456Z",
            dt.datetime(2023, 12, 31, 23, 59, 59, 123456),
        ),
        (
            "2023-12-31T23:59:59.123456Z",
            dt.datetime(2023, 12, 31, 23, 59, 59, 123456),
        )
    )
)
def test_ymd(input, datetime):

    assert dtp.ymd(input) == datetime


@pytest.mark.parametrize(
    ("input", "datetime"),
    (
        ("31.12.23", dt.datetime(2023, 12, 31)),
        ("31.12.2023", dt.datetime(2023, 12, 31)),
        ("31.12.2023 23:59", dt.datetime(2023, 12, 31, 23, 59)),
        ("31.12.2023 23:59:59", dt.datetime(2023, 12, 31, 23, 59, 59)),
        (
            "31.12.2023 23:59:59.123456",
            dt.datetime(2023, 12, 31, 23, 59, 59, 123456),
        ),
        (
            "31.12.2023 23:59:59.123456Z",
            dt.datetime(2023, 12, 31, 23, 59, 59, 123456),
        )
    )
)
def test_dmy(input, datetime):

    assert dtp.dmy(input) == datetime


@pytest.mark.parametrize(
    ("input", "datetime"),
    (
        ("12/31/23", dt.datetime(2023, 12, 31)),
        ("12/31/2023", dt.datetime(2023, 12, 31)),
        ("12/31/2023 23:59", dt.datetime(2023, 12, 31, 23, 59)),
        ("12/31/2023 23:59:59", dt.datetime(2023, 12, 31, 23, 59, 59)),
        (
            "12/31/2023 23:59:59.123456",
            dt.datetime(2023, 12, 31, 23, 59, 59, 123456),
        ),
        (
            "12-31-2023_23-59-59-123456Z",
            dt.datetime(2023, 12, 31, 23, 59, 59, 123456),
        )
    )
)
def test_mdy(input, datetime):

    assert dtp.mdy(input) == datetime


def test_iso_tight():

    assert dtp.iso_tight("20230901T2312") == dt.datetime(2023, 9, 1, 23, 12)


def test_iso_tight_no_t():

    assert dtp.iso_tight("202309012312", regex_t="") \
        == dt.datetime(2023, 9, 1, 23, 12)
