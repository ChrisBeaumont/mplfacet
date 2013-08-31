import warnings

import numpy as np
import matplotlib.pyplot as plt


def groupby(*arrs):
    """Iterate over unique tuples across a series of arrays

    Parameters
    ----------
    arrs : List of array-like (N dimensional)
        Arrays of the same shape

    Yields
    ------
    key, indices

    key : Tuple of values
    indices : tuple of N index arrays (one for each dimension
              of the input arrays)

    For each (key, indices) pair, arrs[i][indices] == key[i]

    Examples
    --------
    In [3]: for k, ind in groupby([1, 1, 3, 1, 2]):
       ...:     print k, ind
     [1] (array([0, 1, 3]),)
     [2] (array([4]),)
     [3] (array([2]),)
    """
    shp = np.shape(arrs[0])
    for a in arrs:
        if np.shape(a) != shp:
            raise ValueError("All inputs must have the same shape")

    f = [np.ravel(a) for a in arrs]
    lsort = np.lexsort(f[::-1])

    switch = np.zeros(lsort.size + 1, dtype=np.bool)
    for ff in f:
        switch[:-1] |= (ff[lsort] != np.roll(ff[lsort], 1))

    #items between successive Trues in switch are equal
    #when ordered by lsort
    switch[0] = False
    switch[-1] = True

    start = 0
    for stop in np.where(switch)[0]:
        key = [ff[lsort[start]] for ff in f]
        ind = lsort[start:stop]
        #all items in f[ind]] are equal to key
        yield key, np.unravel_index(ind, np.asarray(arrs[0]).shape)
        start = stop


def subplots(nrows=1, ncols=1, num=None, sharex=False,
              sharey=False, squeeze=True, subplot_kw=None, **fig_kw):
    """
    Create a figure with a set of subplots already made.

    This utility wrapper makes it convenient to create common layouts of
    subplots, including the enclosing figure object, in a single call.

    Keyword arguments:

      *nrows* : int
        Number of rows of the subplot grid.  Defaults to 1.

      *ncols* : int
        Number of columns of the subplot grid.  Defaults to 1.

      *num* : int
        Number of total axes. If less than nrows * ncols,
        then only the first num axes (from left to right, top to bottom)
        will be created.

      *sharex* : string or bool
        If *True*, the X axis will be shared amongst all subplots.  If
        *True* and you have multiple rows, the x tick labels on all but
        the last row of plots will have visible set to *False*
        If a string must be one of "row", "col", "all", or "none".
        "all" has the same effect as *True*, "none" has the same effect
        as *False*.
        If "row", each subplot row will share a X axis.
        If "col", each subplot column will share a X axis and the x tick
        labels on all but the last row will have visible set to *False*.

      *sharey* : string or bool
        If *True*, the Y axis will be shared amongst all subplots. If
        *True* and you have multiple columns, the y tick labels on all but
        the first column of plots will have visible set to *False*
        If a string must be one of "row", "col", "all", or "none".
        "all" has the same effect as *True*, "none" has the same effect
        as *False*.
        If "row", each subplot row will share a Y axis.
        If "col", each subplot column will share a Y axis and the y tick
        labels on all but the last row will have visible set to *False*.

      *squeeze* : bool
        If *True*, extra dimensions are squeezed out from the
        returned axis object:

        - if only one subplot is constructed (nrows=ncols=1), the
        resulting single Axis object is returned as a scalar.

        - for Nx1 or 1xN subplots, the returned object is a 1-d numpy
        object array of Axis objects are returned as numpy 1-d
        arrays.

        - for NxM subplots with N>1 and M>1 are returned as a 2d
        array.

        If *False*, no squeezing at all is done: the returned axis
        object is always a 2-d array containing Axis instances, even if it
        ends up being 1x1.

      *subplot_kw* : dict
        Dict with keywords passed to the
        :meth:`~matplotlib.figure.Figure.add_subplot` call used to
        create each subplots.

      *fig_kw* : dict
        Dict with keywords passed to the :func:`figure` call.  Note that all
        keywords not recognized above will be automatically included here.

    Returns:

      fig, ax : tuple

        - *fig* is the :class:`matplotlib.figure.Figure` object

        - *ax* can be either a single axis object or an array of axis
        objects if more than one subplot was created.  The dimensions
        of the resulting array can be controlled with the squeeze
        keyword, see above.

    Examples::

        x = np.linspace(0, 2*np.pi, 400)
        y = np.sin(x**2)

        # Just a figure and one subplot
        f, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title('Simple plot')

        # Two subplots, unpack the output array immediately
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.plot(x, y)
        ax1.set_title('Sharing Y axis')
        ax2.scatter(x, y)

        # Four polar axes
        plt.subplots(2, 2, subplot_kw=dict(polar=True))

        # Share a X axis with each column of subplots
        plt.subplots(2, 2, sharex='col')

        # Share a Y axis with each row of subplots
        plt.subplots(2, 2, sharey='row')

        # Share a X and Y axis with all subplots
        plt.subplots(2, 2, sharex='all', sharey='all')
        # same as
        plt.subplots(2, 2, sharex=True, sharey=True)
    """
    # for backwards compatibility
    if isinstance(sharex, bool):
        if sharex:
            sharex = "all"
        else:
            sharex = "none"
    if isinstance(sharey, bool):
        if sharey:
            sharey = "all"
        else:
            sharey = "none"
    share_values = ["all", "row", "col", "none"]
    num = num or nrows * ncols

    if sharex not in share_values:
        # This check was added because it is very easy to type subplots(1, 2, 1)
        # when subplot(1, 2, 1) was intended. In most cases, no error will
        # ever occur, but mysterious behavior will result because what was
        # intended to be the subplot index is instead treated as a bool for
        # sharex.
        if isinstance(sharex, int):
            warnings.warn("sharex argument to subplots() was an integer."
                          " Did you intend to use subplot() (without 's')?")

        raise ValueError("sharex [%s] must be one of %s" %
                         (sharex, share_values))
    if sharey not in share_values:
        raise ValueError("sharey [%s] must be one of %s" %
                         (sharey, share_values))
    if subplot_kw is None:
        subplot_kw = {}

    fig = plt.figure(**fig_kw)

    # Create empty object array to hold all axes.  It's easiest to make it 1-d
    # so we can just append subplots upon creation, and then
    nplots = nrows * ncols
    axarr = np.empty(nplots, dtype=object)

    # Create first subplot separately, so we can share it if requested
    ax0 = fig.add_subplot(nrows, ncols, 1, **subplot_kw)
    #if sharex:
    #    subplot_kw['sharex'] = ax0
    #if sharey:
    #    subplot_kw['sharey'] = ax0
    axarr[0] = ax0

    r, c = np.mgrid[:nrows, :ncols]
    r = r.flatten() * ncols
    c = c.flatten()
    lookup = {
        "none": np.arange(nplots),
        "all": np.zeros(nplots, dtype=int),
        "row": r,
        "col": c,
        }
    sxs = lookup[sharex]
    sys = lookup[sharey]

    # Note off-by-one counting because add_subplot uses the MATLAB 1-based
    # convention.
    for i in range(1, min(num, nplots)):
        if sxs[i] == i:
            subplot_kw['sharex'] = None
        else:
            subplot_kw['sharex'] = axarr[sxs[i]]
        if sys[i] == i:
            subplot_kw['sharey'] = None
        else:
            subplot_kw['sharey'] = axarr[sys[i]]
        axarr[i] = fig.add_subplot(nrows, ncols, i + 1, **subplot_kw)

    # returned axis array will be always 2-d, even if nrows=ncols=1
    axarr = axarr.reshape(nrows, ncols)

    # turn off redundant tick labeling
    for k in range(nplots):
        i, j = k / ncols, k % ncols
        ax = axarr[i, j]

        if ax is None:
            continue
        if sharex in ['col', 'all'] and (i < (nrows - 1) and
                                         (axarr[i + 1, j] is not None)):
            #hide x axis if there's a plot below
            [label.set_visible(False) for label in ax.get_xticklabels()]
            ax.xaxis.offsetText.set_visible(False)
        if sharey in ['row', 'all'] and j > 0:
            #hide y axis if there's a plot leftward
            [label.set_visible(False) for label in ax.get_yticklabels()]
            ax.yaxis.offsetText.set_visible(False)

    if squeeze:
        # Reshape the array to have the final desired dimension (nrow,ncol),
        # though discarding unneeded dimensions that equal 1.  If we only have
        # one subplot, just return it instead of a 1-element array.
        if nplots == 1:
            ret = fig, axarr[0, 0]
        else:
            ret = fig, axarr.squeeze()
    else:
        # returned axis array will be always 2-d, even if nrows=ncols=1
        ret = fig, axarr.reshape(nrows, ncols)

    return ret
