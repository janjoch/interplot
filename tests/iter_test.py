from toolbox import iter

import pytest

from numpy import array

from pandas import DataFrame, Series


@pytest.mark.parametrize(
    "input",
    (1, 3.1415, (1, 2, 3), [4, 5, 6], {1: 3, 5: 3.1415}),
)
def test_repeat(input):
    gen = iter.repeat(input).__iter__()
    _ = next(gen)
    assert next(gen) == input


# tbc...
def test_zip_smart():
    assert 1 == 1
