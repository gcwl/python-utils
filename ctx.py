import sys
import gc
from contextlib import contextmanager


@contextmanager
def Ctx():
    # https://stackoverflow.com/a/21795428
    saved_ctx = dict(**sys.modules[__name__].__dict__)
    try:
        yield
    finally:
        # delete variables created in the lifttime of this context
        new_vars = set(sys.modules[__name__].__dict__) - set(saved_ctx)
        for x in new_vars:
            del sys.modules[__name__].__dict__[x]
        # restore saved context
        sys.modules[__name__].__dict__.update(saved_ctx)
        gc.collect()
