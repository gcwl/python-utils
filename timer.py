import time
from contextlib import contextmanager


@contextmanager
def Timer(message=None, stream=print):
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        stream(f'{message or "Time elapsed"}: {dt:.4f}s')


if __name__ == '__main__':

    def timer_nested():
        with Timer('test1'):
            time.sleep(.2)
            with Timer('test2'):
                time.sleep(.7)
                with Timer():
                    time.sleep(.1)

    def timer_stream():
        import logging
        logging.basicConfig(format='%(asctime)s|%(process)d|%(module)s.%(name)s.%(funcName)s:%(lineno)s|%(levelname)s|%(message)s')
        logger = logging.getLogger(__name__)
        with Timer(stream=logger.error):
            time.sleep(.3)

    timer_nested()
    timer_stream()