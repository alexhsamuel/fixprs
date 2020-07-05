from   time import perf_counter
import itertools
import logging
import numpy as np


class timing:
    """
    Context manager that measures wall clock time between entry and exit.
    """

    def __init__(self, name, timer=perf_counter):
        self.__name = name
        self.__timer = timer
        self.__start = self.__end = self.__elapsed = None
        self.__end = None


    def __enter__(self):
        self.__start = self.__timer()
        return self
        

    def __exit__(self, *exc):
        self.__end = self.__timer()


    @property
    def name(self):
        return self.__name


    @property
    def start(self):
        return self.__start


    @property
    def end(self):
        return self.__end


    @property
    def elapsed(self):
        return self.__end - self.__start



def _benchmark(fn, s, n, *, quantile=0.05):
    # Loop pedestal calculation.
    null = lambda: None
    samples = []
    for _ in range(s):
        start = perf_counter()
        for _ in range(n):
            null()
        samples.append(perf_counter() - start)
    pedestal = np.percentile(samples, 100 * quantile, interpolation="nearest")

    logging.debug("pedestal={:.0f} ns".format(pedestal / n / 1e-9))
    
    samples = []
    for _ in range(s):
        start = perf_counter()
        for _ in range(n):
            fn()
        samples.append(perf_counter() - start)
    
    result = np.percentile(samples, 100 * quantile, interpolation="nearest")
    return (result - pedestal) / n


def benchmark(fn, *, quantile=0.05):
    MIN_SAMPLE_TIME = 1E-3
    TARGET_TIME = 0.2

    # Estimate parameters.
    for scale in itertools.count():
        n = 10 ** scale
        start = perf_counter()
        for _ in range(n):
            fn()
        elapsed = perf_counter() - start
        if elapsed >= MIN_SAMPLE_TIME:
            break

    s = max(5, min(100, int(TARGET_TIME / elapsed)))

    logging.debug("calibration: n={} s={}".format(n, s))

    return _benchmark(fn, s, n, quantile=quantile)


