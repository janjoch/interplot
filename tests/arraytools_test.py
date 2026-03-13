import numpy as np

import interplot.arraytools as at


def test_downsample_small():
    a = np.arange(123)
    a_ = at.downsample(10, a)
    assert len(a_) == 10
    assert a_[0] == 0
    assert a_[-1] == 117

    assert (a_ == at.downsample_step(10, a)).all()


def test_downsample_step_odd():
    a = np.arange(100)
    a_ = at.downsample_step(20, a)
    assert len(a_) == 20
    assert a_[0] == 0
    assert a_[-1] == 95

    assert (a_ == at.downsample_step(20, a[:-4])).all()


def test_downsample_average_small():
    a = np.arange(100)
    a_ = at.downsample_average(20, a)
    assert len(a_) == 20
    assert a_[0] == 2.
    assert a_[-1] == 97.

    b_ = at.downsample_average(20, a[:-1])
    assert len(b_) == 19
    assert b_[-1] == 92.


def test_downsample_multiple():
    a = np.arange(123)
    b = np.random.normal(size=123)

    a_, b_ = at.downsample(20, a, b)
    assert len(a_) == len(b_)

def test_downsample_step_2d():
    aa = np.random.normal(size=(10,123))
    aa_ = at.downsample(13, aa, axis=1)
    assert aa_.shape == (10, 13)

def test_downsample_average_4d():
    aaaa = np.random.normal(size=(10, 5, 123, 4))
    aaaa_ = at.downsample_average(13, aaaa, axis=2)
    assert aaaa_.shape == (10, 5, 12, 4)
    a_ = at.downsample(13, aaaa[1, 2, :, 3], mode="average")
    assert np.isclose(aaaa_[1, 2, :, 3], a_).all()
