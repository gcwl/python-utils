import gc
from collections import OrderedDict as odict
from typing import List, Tuple, Callable

import numpy as np
import pandas as pd

from IPython.display import display


def csv2df(path, **kw):
    df = pd.read_csv(path, **kw)
    df.info()
    print('shape: {}'.format(df.shape))
    return df


def frame_info(frame: pd.DataFrame,
               n_samples: int=10,
               styling: bool=True,
               before_styling: Callable=lambda df: df):

    # compute values of interest
    nrow, ncol = frame.shape
    num_nan = frame.isna().sum(axis=0)
    num_notnan = frame.count()
    frac_nan = num_nan/nrow
    num_unique = frame.nunique()
    frac_unique = num_unique/num_notnan

    max10k = np.min([10_000, nrow//10])
    n_samples = np.min([1_000, max10k, n_samples])

    # build dataframe of interest
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

