import sys
from contextlib import contextmanager


# https://stackoverflow.com/a/13250224
@contextmanager
def Shhh(out=None, err=None):
    '''Redirect std{out,err}'''
    saved = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        yield
    finally:
        sys.stdout, sys.stderr = saved
