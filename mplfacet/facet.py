from collections import namedtuple
from functools import wraps, partial

import numpy as np
from matplotlib.axes import Axes

from .util import subplots, groupby


FacetItem = namedtuple('FacetItem', 'axes data key label')


def _axeswrap(self, key):
    target = getattr(self.axes, key)

    result = wraps(target)(partial(target, *self.data))
    hdr = "Faceted Wrapper for %s\n\n" % target.__name__
    result.__doc__ = hdr + result.__doc__
    return result
FacetItem.__getattr__ = _axeswrap


class Facet(object):
    def __init__(self, keys, data, labeler=None,
                 xlabel=None, ylabel=None, **subplot_opts):
        """
        Create a new facet object

        Parameters
        ----------
        keys : array-like, or list of two array-likes
          The data to use for faceting. Each array
          should have the same shape as the data array(s).
          Data will partitioned into groups with equal elements
          in keys

        data : array-like, or sequence of array-like
          The data to facet. These will be passed as the first
          arguments to plot calls

        labeler : function, dict, or array
          A mapping from values in key to strings, labeling each group.
          If provided, labels will be used to set plot titles.  If a
          function, will use result from labeler(key) for label
          Otherwise, will use the result from labeler[key] for
          label. Key will an object if faceting on one
          item. Otherwise, it will be a 2-tuple

        xlabel : str (optional)
          X axis label

        ylabel : str (optional)
          Y axis label

        Extra keywords will be passed to `~matplotlib.pyplot.subplots`.

        Examples
        --------
        #make a faceted scatterplot of x and y, grouped into subsets
        #with equal key values
        Facet(key, [x, y]).scatter()

        #make a faceted histogram
        Facet(key, x).hist(bins=np.linspace(0, 5, 10))

        key = np.random.randint(0, 3, 300)
        x = np.linspace(0, 5, 500)
        y = np.sin(x) + key
        f = Facet(key, [x, y]).scatter()

        #make a faceted grid, based on two variables
        f = Facet(k1, k2, [x, y]).scatter()
        """
        if isinstance(data, np.ndarray):
            self.data = (data,)
        else:
            self.data = data

        shp = np.shape(self.data[0])
        if np.shape(keys) == shp:  # one key provided
            self._keys = [keys]
        elif len(keys) == 2 and np.shape(keys[0]) == np.shape(keys[1]) == shp:
            #2 keys provided
            self._keys = keys
        else:
            raise ValueError("Keys must be an array shaped like the data, "
                             "or a list of two such arrays")

        nfacet = np.product([np.unique(k).size for k in self._keys])
        if nfacet > 50:
            raise ValueError("Too many facets to plot (limit=50): %i" % nfacet)

        if nfacet == 0:
            raise ValueError("No data to facet!")

        self._key_index = [dict((k, i) for i, k in enumerate(np.unique(key)))
                           for key in self._keys]

        self._labeler = labeler
        self._xlabel = xlabel
        self._ylabel = ylabel

        self.subplot_opts = subplot_opts.copy()
        nr, nc = self._subplot_dims()
        self.subplot_opts['nrows'] = nr
        self.subplot_opts['ncols'] = nc

    def _subplot_dims(self):
        """Determine size of subplot grid, given possible
        constraints on nrows, ncols"""
        nrows = self.subplot_opts.get('nrows', None)
        ncols = self.subplot_opts.get('ncols', None)

        #if 2 keys provided, rows and cols are fixed
        if len(self._keys) == 2:
            nr = len(self._key_index[0])
            nc = len(self._key_index[1])

            if ((nrows is not None and nrows != nr) or
                (ncols is not None and ncols != nc)):
                raise ValueError("Two keys specified: (nrows, ncols) must be "
                                 "(%i, %i) " % (nr, nc))
            return nr, nc

        sz = len(self._key_index[0])

        #if 1 key provided, just need nrows * ncols >= nfacets
        if nrows is None:
            if ncols is None:
                nrows = max(1, np.int(np.sqrt(sz)))
            else:
                nrows = np.int(np.ceil(1. * sz / ncols))
        if ncols is None:
            ncols = np.int(np.ceil(1. * sz / nrows))
        if nrows * ncols < sz:
            raise ValueError("nrows (%i) and ncols (%i) not big enough "
                             "to plot %i facets" % (nrows, ncols, sz))
        return nrows, ncols

    @classmethod
    def from_labeled_array(cls, x, facet, data, **opts):
        """
        Build a Facet object from a Pandas DataFrame

        Keyword arguments are passed to __init__

        Parameters
        ----------
        df : DataFrame instance
          The data to use

        facet : str or list of 2 str
          The name(s) of columns to use as facet keys

        data : str or list of str
          The name(s) of columns to use as data

        Returns
        -------
        facet : Facet instance
        """
        if isinstance(facet, basestring):
            facet = [facet]
        if isinstance(data, basestring):
            data = [data]

        def labeler(key):
            return ', '.join('%s: %s' % (f, k) for f, k in zip(facet, key))

        opts.setdefault('labeler', labeler)

        facet_val = [np.asarray(x[f]) for f in facet]
        data_val = [np.asarray(x[f]) for f in data]

        return cls(facet_val, data_val, **opts)

    @property
    def _subplots(self):
        opts = self.subplot_opts
        opts.setdefault('tight_layout', True)
        opts.setdefault('sharex', True)
        opts.setdefault('sharey', True)

        num = np.product([len(i) for i in self._key_index])
        fig, subs = subplots(num=num, **opts)
        if len(self._keys) == 1:
            subs = subs.ravel()
        return subs

    def _dispatch(self, func, *args, **kwargs):
        """ Repeatedly call an axes function on each facet

        Each method is automatically passed the faceted data,
        so this only works for plot methods whose first arguments
        are data arrays (e.g. axes.plot(x, y))


        Extra args and kwargs are passed to the axes call
        after the data arguments

        Parameters
        ----------
        func : str
            Name of an axes method to cal
        """
        for item in self:
            a = item.data
            a.extend(args)
            method = getattr(item.axes, func)
            method(*a, **kwargs)
            item.axes.set_title(item.label)

        f = item.axes.figure
        textopts = dict(size='large')
        if self._xlabel is not None:
            f.text(.5, 0, self._xlabel, ha='center', **textopts)
        if self._ylabel is not None:
            f.text(0, .5, self._ylabel, va='center', rotation='vertical',
                   **textopts)

    def __getattr__(self, method):
        """
        All axes plot methods are available as atttributes.
        Each is repeatedly called with each facet

        Examples
        --------
        key = np.array([1, 1, 2, 2])
        data = np.array([1, 2, 2, 3])
        f = Facet(key, data).plot()
        """
        try:
            target = getattr(Axes, method)
        except AttributeError:
            raise AttributeError("%s is not a valid Axes plot method" % method)

        hdr = "\nFaceted wrapper for Axes.%s\n\n" % method
        result = wraps(target)(partial(self._dispatch, method))
        result.__doc__ = hdr + result.__doc__
        return result

    def _pick_axes(self, key):
        #given a key value, return the index of the approrpriate axes
        #in the subplots array
        return tuple(i[k] for k, i in zip(key, self._key_index))

    def _label(self, key):
        """
        Given a facet key, return a label
        """
        if len(key) == 1:
            key = key[0]
        else:
            key = tuple(key)

        if self._labeler is None:
            return str(key)

        #callable
        if hasattr(self._labeler, '__call__'):
            return self._labeler(key)

        #indexable
        return self._labeler[key]

    def __iter__(self):
        """
        Iterate over each facet, returning data for custom plotting

        Yields
        ------
        FacetItem

        Each facet item has four attributes:
          * axes : An axes object
          * data : A tuple of the faceted data
          * key : The value(s) of the key used for the facet
          * label : A label for key

        In addition, calling methods like `scatter` or other
        axes.plot methods on a FacetItem will automatically
        pass the faceted data to the appropriate axes
        """
        axes = self._subplots
        used = []
        for k, ind in groupby(*self._keys):
            a = axes[self._pick_axes(k)]
            used.append(a)
            data = [d[ind] for d in self.data]
            label = self._label(k)
            yield FacetItem(axes=a, data=data, key=k, label=label)
