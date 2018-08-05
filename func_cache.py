from functools import wraps, partial
from pathlib import Path
import pickle
import joblib
import hashlib

import pandas as pd

from timer import Timer

               
def hash_args(obj, h=None):
    h = h or hashlib.sha256()
    if isinstance(obj, dict):
        for k, v in sorted(obj.items()):
            h.update(pickle.dumps(k))
            hash_args(v, h)
    elif isinstance(obj, (list, tuple, set)):
        for e in obj:
            hash_args(e, h)
    else:
        h.update(pickle.dumps(obj))
    return h.hexdigest()



def func_cache(f, cache_dir='./cache', return_path=False):

    @wraps(f)
    def wrapper(*args, **kw):
        
        # name of wrapped func
        func_name = f.__name__

        # compute checksum (filename)
        with Timer(f'<{func_name}> Compute checksum', LOG.info):
            h = hashlib.sha256()
            hash_args(args, h)
            hash_args(kw, h)
            checksum = h.hexdigest()

        # make dir/path
        dirpath = f'{cache_dir}/FUNC.{func_name}'
        Path(dirpath).mkdir(exist_ok=True)
        filepath = f'{dirpath}/{checksum}.pkl'
        
        # check if cache exists
        if Path(filepath).is_file():
            with Timer(f'<{func_name}> Load result from cache ({filepath})', LOG.info):
                return filepath if return_path else joblib.load(filepath)
        
        # compute value, and save result as feather format
        with Timer(f'<{func_name}> Compute result', LOG.info):
            val = f(*args, **kw)
        with Timer(f'<{func_name}> Save result to cache ({filepath})', LOG.info):
            joblib.dump(val, filepath)
        if return_path:
            del val
            return filepath
        return val

    return wrapper


@func_cache
def read_csv(*args, **kw):
    return pd.read_csv(*args, **kw)


@func_cache
def one_hot(frame, columns, dummy_na=True, prefix_sep='=', **kw):
    df = pd.get_dummies(frame, columns=columns, dummy_na=dummy_na, prefix_sep=prefix_sep, **kw)
    cols_onehot = set(df.columns) - set(frame.columns)
    return df, sorted(cols_onehot)
