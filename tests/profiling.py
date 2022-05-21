"""Module with useful functions for profiling"""

import time


def timef(method):
    "Decorator to time a function"

    def timed(*args, **kw):
        t_start = time.time()
        result = method(*args, **kw)
        t_end = time.time()
        print(f"{method.__name__}  {t_end - t_start:.2f} s")
        return result

    return timed
