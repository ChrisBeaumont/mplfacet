import numpy as np
from nose.tools import assert_raises

from ..facet import Facet


class TestPickAxes(object):

    def test_single_row(self):
        x = np.array([1, 1, 2, 1, 3])
        f = Facet(x, x)
        assert f._pick_axes([1]) == (0,)
        assert f._pick_axes([2]) == (1,)
        assert f._pick_axes([3]) == (2,)

    def test_2d(self):
        x = np.array([1, 1, 2, 1, 3])
        y = np.array([1, 2, 2, 1, 1])
        f = Facet([x, y], x)
        assert f._pick_axes([1, 1]) == (0, 0)
        assert f._pick_axes([2, 1]) == (1, 0)
        assert f._pick_axes([3, 1]) == (2, 0)
        assert f._pick_axes([1, 2]) == (0, 1)
        assert f._pick_axes([2, 2]) == (1, 1)
        assert f._pick_axes([3, 2]) == (2, 1)


def test_bad_options():

    x = np.array([1, 2, 3])
    y = np.array([1, 1, 2])

    #not enough plots for all facets
    assert_raises(ValueError, Facet, x, x, nrows=0, ncols=0)
    assert_raises(ValueError, Facet, x, x, nrows=0, ncols=1)
    assert_raises(ValueError, Facet, x, x, nrows=1, ncols=1)

    #incompatible 2d layout with 2 facets
    assert_raises(ValueError, Facet, [x, y], x, nrows=3, ncols=1)
    assert_raises(ValueError, Facet, [x, y], x, nrows=3, ncols=3)
    assert_raises(ValueError, Facet, [x, y], x, nrows=2, ncols=2)
    assert_raises(ValueError, Facet, [x, y], x, nrows=4, ncols=2)

    assert_raises(ValueError, Facet, [y, x], x, nrows=1, ncols=3)
    assert_raises(ValueError, Facet, [y, x], x, nrows=3, ncols=3)
    assert_raises(ValueError, Facet, [y, x], x, nrows=2, ncols=2)
    assert_raises(ValueError, Facet, [y, x], x, nrows=2, ncols=4)

def test_bad_key():

    #key and data not same shape
    key = np.array([1, 2, 3, 4])
    data = np.array([1, 2, 3])
    assert_raises(ValueError, Facet, key, data)


    #too many keys
    key = data
    assert_raises(ValueError, Facet, [key, key, key], data)


def test_subplot_dims():

    def check(facet, nrow, ncol):
        nr, nc = facet._subplot_dims()
        assert nr == nrow
        assert nc == ncol

    k1 = np.array([1, 1, 2, 2, 2])
    k2 = np.array([1, 2, 1, 2, 3])
    k3 = np.array([1, 2, 3, 4, 4])

    data = k1 * 0

    check(Facet(k1, data), 1, 2)
    check(Facet(k1, data, ncols=1), 2, 1)
    check(Facet([k1, k2], data), 2, 3)
    check(Facet([k2, k1], data), 3, 2)
    check(Facet(k2, data, ncols=2), 2, 2)
    check(Facet(k2, data, ncols=1), 3, 1)
    check(Facet(k2, data), 1, 3)
    check(Facet(k3, data), 2, 2)

def test_facet_overflow():
    x = np.arange(51)
    assert_raises(ValueError, Facet, x, x)

    x = np.arange(5)
    y = np.arange(11)
    assert_raises(ValueError, Facet, [x, y], x)

def test_label_indexer():
    x = np.arange(5) + 1
    f = Facet(x, x, labeler='zero one two three four five'.split())
    assert f._label([1]) == 'one'
    assert f._label([2]) == 'two'
    assert f._label([3]) == 'three'
    assert f._label([4]) == 'four'
    assert f._label([5]) == 'five'

def test_label_caller():
    x = np.arange(5) + 1
    labeler = lambda x: 'a b c d e'.split()[x - 1]
    f = Facet(x, x, labeler=labeler)
    assert f._label([1]) == 'a'
    assert f._label([2]) == 'b'
    assert f._label([3]) == 'c'
    assert f._label([4]) == 'd'
    assert f._label([5]) == 'e'

def test_empty_facet():
    x = np.array([])
    assert_raises(ValueError, Facet, x, x)
