mplfacet
========

mplfacet is a utility to make faceted plots in Matplotlib with less boilerplate code.

Say you wanted to compare the height distribution of males and females, given arrays for `gender` and `height`. Normally, you would have to write code like this:

```
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

axes[0].hist(height[gender == 'Male'])
axes[0].set_title('Male')
axes[1].hist(height[gender == 'Female])
axes[1].set_title('Female')
```

with `mplfacet`, you can achieve the same result with

```
Facet(gender, height).hist()
```

For more usage examples, see the [guide notebook](http://nbviewer.ipython.org/urls/raw.github.com/ChrisBeaumont/mplfacet/master/guide.ipynb)

### Note
A recently fixed [matplotlib bug](https://github.com/matplotlib/matplotlib/issues/2356) can lead to bad default `xlimit/ylimit` values for facets. Updating to the most recent developer version of matplotlib fixes this.