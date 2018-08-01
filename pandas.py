import gc
from collections import OrderedDict as odict
from itertools import count
from typing import List, Tuple, Callable

import numpy as np
import scipy as sp
import pandas as pd
from tqdm import tqdm



def csv2df(path, **kw):
    df = pd.read_csv(path, **kw)
    df.info()
    print('shape: {}'.format(df.shape))
    return df



palette_dtypes = {np.dtype('object'): '#fd7272',
                  np.dtype('float32'): '#00b894',
                  np.dtype('float64'): '#60a3bc',
                  np.dtype('int64'): '#ff9f1a',
                  np.dtype('int32'): '#fed330'}


def frame_info(frame: pd.DataFrame,
               n_samples: int=33,
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
    res = pd.DataFrame(odict(dtypes=frame.dtypes,
                             samples=frame.sample(max10k).agg(lambda x:list(x.unique()[:n_samples])),
                             frac_nan=frac_nan,
                             num_nan=num_nan,
                             num_notnan=num_notnan,
                             frac_unique=frac_unique,
                             num_unique=num_unique))

    gc.collect()
    print('shape = {}'.format(frame.shape))

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



def qqplot(frame: pd.DataFrame,
           dtypes_include: Tuple[str]=('int32', 'int64', 'float32', 'float64'),
           dtypes_exclude: Tuple[str]=('object',)):

    # include/exclude columns by dtypes
    columns = frame.select_dtypes(include=dtypes_include, exclude=dtypes_exclude).columns
    # how many columns will be plotted
    ncol = len(columns)

    # setup figure size, number of rows/cols
    fig_cols = 5
    fig_rows = (ncol % fig_cols != 0) + ncol // fig_cols
    fig_size = (19, 3.6 * fig_rows)
    fig, ax = plt.subplots(fig_rows, fig_cols, figsize=fig_size)

    # fill qqplots into figure
    g = ((p, q) for p in count(0) for q in range(fig_cols))
    for (r, c), column in tqdm(zip(g, columns), total=ncol):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html
        (x, y), (m, b, _) = sp.stats.probplot(frame[column].dropna())
        color = palette_dtypes.get(frame[column].dtype, '#9b59b6')
        ax[r][c].plot(x, y, color=color, marker='|', linestyle='')
        ax[r][c].plot(x, m*x+b, color='#4b4b4b', linestyle='-', linewidth=.9)
        ax[r][c].set_title(column, fontsize=10)

    gc.collect()
