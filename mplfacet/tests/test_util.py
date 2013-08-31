import numpy as np

from ..util import groupby


def check_groupby(*arrs):
    ct = 0
    last = None
    for key, inds in groupby(*arrs):
        #keys should be sorted
        if last is not None:
            assert key > last
        last = key

        #groups should match key
        ct += inds[0].size
        for a, k in zip(arrs, key):
            assert (np.asarray(a)[inds] == k).all()

    #all items returned
    assert ct == np.size(arrs[0])


class TestGroupBy(object):

    def test_1d_sort(self):
        x = np.array([1, 1, 2, 3, 4])
        check_groupby(x)

    def test_1d_single(self):
        check_groupby(np.array([1, 1, 1]))

    def test_unique(self):
        check_groupby(np.array([1, 2, 3]))

    def test_unsorted(self):
        check_groupby(np.array([3, 1, 1, 8, 3]))

    def test_2d(self):
        check_groupby(np.array([[1, 2], [1, 1]]))

    def test_twoarr(self):
        check_groupby(np.array([1, 1, 3]),
                      np.array([1, 1, 1]))

    def test_threarr(self):
        check_groupby(np.array([1, 1, 3, 1, 3]),
                      np.array([1, 1, 1, 2, 3]),
                      np.array([2, 1, 3, 3, 2]))

    def test_big(self):
        x = np.random.randint(0, 3, 1000).reshape(10, 10, 10)
        y = np.random.randint(0, 3, 1000).reshape(10, 10, 10)
        check_groupby(x, y)

    def test_arraylike(self):
        check_groupby([1, 1, 2, 3, 1, 2, 3])
