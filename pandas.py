import gc
from collections import OrderedDict as odict
from itertools import count
from typing import List, Tuple, Callable

import numpy as np
import scipy as sp
import pandas as pd



palette_dtypes = {np.dtype('object'): '#fd7272',
                  np.dtype('float32'): '#00b894',
                  np.dtype('float64'): '#60a3bc',
                  np.dtype('int64'): '#ff9f1a',
                  np.dtype('int32'): '#fed330'}



def frame_info(frame: pd.DataFrame,
               n_samples: int=33,               # MAGIC: 33
               styling: bool=True,
               before_styling: Callable=lambda df: df):

    # compute values
    nrow, ncol = frame.shape
    num_nan = frame.isna().sum(axis=0)
    num_notnan = frame.count()
    frac_nan = num_nan/nrow
    num_unique = frame.nunique()
    frac_unique = num_unique/num_notnan

    max10k = np.min([10_000, nrow//10])
    n_samples = np.min([1_000, max10k, n_samples])

    # build dataframe
    res = pd.DataFrame(odict(dtype=frame.dtypes,
                             samples=frame.sample(max10k).agg(lambda x:list(x.unique()[:n_samples])),
                             frac_nan=frac_nan,
                             num_nan=num_nan,
                             num_notnan=num_notnan,
                             frac_unique=frac_unique,
                             num_unique=num_unique))

    gc.collect()
    print(f'shape = {frame.shape}')

    if not styling:
        return res

    # for instance, one can sort_values() before styling
    res = before_styling(res)

    bg_dtypes = lambda val: f'background-color: {palette_dtypes.get(val, "#9b59b6")}'
    fg_frac_nan = lambda val: f'color: {"red" if val > .5 else "black"}'
    fg_frac_unique = lambda val: f'color: {"red" if val < .1 else "black"}'
    return (res.style
               .bar(subset=['frac_nan'], color='#8395a7', width=100*frac_nan.max())
               .bar(subset=['frac_unique'], color='#cad3c8', width=100*frac_unique.max())
               .applymap(bg_dtypes, subset=['dtypes'])
               .applymap(fg_frac_nan, subset=['frac_nan'])
               .applymap(fg_frac_unique, subset=['frac_unique']))



def qqplot(data):

    def make_plot(series: pd.Series,
                  axes, #: matplotlib.axes.Axes
                  title: str):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html
        (x, y), (m, b, r) = sp.stats.probplot(series.dropna())
        color = palette_dtypes.get(series.dtype, '#9b59b6')
        axes.plot(x, y, color=color, marker='|', linestyle='')
        axes.plot(x, m*x+b, color='#4b4b4b', linestyle='-', linewidth=.9)
        axes.set_title(title, fontsize=10)

    def plot_frame(frame: pd.DataFrame):
        # setup figure size, number of rows/cols
        nrow, ncol = frame.shape
        fig_ncol = np.min([5, ncol])                          # magic constant: 5
        fig_nrow = (ncol % fig_ncol != 0) + ncol // fig_ncol
        fig_size = (19, 3.6 * fig_nrow)                       # magic constants: 19, 3.6
        fig, ax = plt.subplots(fig_nrow, fig_ncol, figsize=fig_size)
        # make qqplots in figure/axes
        g = ((p, q) for p in count(0) for q in range(fig_ncol))
        for (r, c), column in zip(g, frame.columns):
            axes = ax[r][c] if fig_nrow > 1 else ax[c]        # Axes is 1D-array if fig_nrow == 1
            make_plot(frame[column], axes, column)

    def plot_series(series: pd.Series):
        ax = plt.subplot()
        make_plot(series, ax, series.name)

    if isinstance(data, pd.DataFrame):
        # exclude np.dtype('object') columns
        frame = data.select_dtypes(exclude='object')
        plot_frame(frame)
    elif isinstance(data, pd.Series):
        plot_series(data)
    else:
        raise TypeError(f'Unknow object: {data}')

    gc.collect()
