import interplot

import pytest

from numpy import array

from pandas import DataFrame, Series


dct = {1: 5, 2: 6}
a = array((1, 2))
s = Series((1, 2))
df = DataFrame(data=array(((1, 2), (3, 4))), columns=("A", "B"))


@pytest.mark.parametrize(
    "input",
    (1, 3.1415, (1, 2, 3), [4, 5, 6], {1: 3, 5: 3.1415}),
)
def test_repeat(input):
    gen = interplot.repeat(input).__iter__()
    _ = next(gen)
    assert next(gen) == input


@pytest.mark.parametrize(
    "in_out",
    (
        ((1, 2), 2),
        ((1, 2, 3), 2),
        ([1, 2], 2),
        ("abc", "abc"),
        (("a", "b"), "b"),
        (dct, 2),
        (range(3), 1),
        (a, 2),
        (s, 2),
        (df, "B"),
    )
)
def test_zip_smart(in_out):
    gen = interplot.zip_smart(
        in_out[0],
        (1, 2, 3),
    )
    _ = next(gen)
    assert next(gen)[0] == in_out[1]


@pytest.mark.parametrize(
    "in_out",
    (
        ((1, 2), (1, 2)),
        ((1, 2, 3), 2),
        ([1, 2], [1, 2]),
        ("abc", "abc"),
        (("a", "b"), "b"),
        (dct, dct),
        (range(3), 1),
        (a, 2),
        (df, "B"),
    )
)
def test_filter_nozip(in_out):
    gen = interplot.zip_smart(
        interplot.filter_nozip(in_out[0]),
        (1, 2, 3),
    )
    _ = next(gen)
    assert next(gen)[0] == in_out[1]
