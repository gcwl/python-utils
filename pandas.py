import gc
from collections import OrderedDict as odict

import numpy as np
import pandas as pd

from IPython.display import display


def csv2df(path, verbose=True, **kw):
    df = pd.read_csv(path, **kw)
    if verbose:
        print('shape = {}'.format(df.shape))
        display(df.head(10))
        df.info()
        display(df.describe())
    return df


def frame_info(df, return_frame=False):
    nr, nc = df.shape
    isna = df.isna().sum(axis=0)
    count = df.count()
    nunique = df.nunique()
    nsamples = np.min([1000, nr//10])

    res = pd.DataFrame(odict(dtypes=df.dtypes,
                             samples=df.sample(nsamples).agg(lambda c:list(c.unique()[:10])),
                             frac_isna=isna/nr,
                             num_isna=isna,
                             num_notna=count,
                             frac_unique=nunique/count,
                             num_unique=nunique))

    gc.collect()
    print(df.shape)

    if return_frame:
        return res

    def color_dtypes(value: str) -> str:
        palette = {np.dtype('object'): '#ff6b81',
                   np.dtype('float64'): '#34e7e4',
                   np.dtype('int64'): '#ffdd59'}
        return 'background-color: {}'.format(palette.get(value, '#2f3542'))

    return (res.sort_values(by='frac_isna', ascending=False)
            .style
            .applymap(color_dtypes, subset=['dtypes'])
            .bar(subset=['frac_unique', 'frac_isna'],
                       color='#778ca3'))

