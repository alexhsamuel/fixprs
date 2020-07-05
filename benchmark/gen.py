import argparse
from   functools import partial
import numpy as np
import pandas as pd
import random
import sys

#-------------------------------------------------------------------------------

def _ints(value, shape):
    if callable(value):
        # It's a random function.
        return value(shape)
    else:
        # Assume it's a constant.
        result = np.empty(shape, dtype=int)
        result[:] = value
        return result


def cumsum(gen):
    return lambda n: gen(n).cumsum()


def normal(mu=0, sigma=1):
    return partial(np.random.normal, mu, sigma)


def uniform(lo=0, hi=1):
    return partial(np.random.uniform, lo, hi)


def uniform_int(lo, hi):
    return partial(np.random.randint, lo, hi)


def word(length, upper=False):
    """
    @param length
      Fixed string length, or a random function to generate it.
    """
    letters = "abcdefghijklmnopqrstuvwxyz"
    if upper:
        letters = letters.upper()

    def gen(shape):
        lengths = _ints(length, shape)
        field_length = lengths.max()
        dtype = "U{}".format(field_length)

        result = np.empty(shape, dtype=dtype)
        flat = result.ravel()
        for i, l in enumerate(lengths):
            flat[i] = "".join( random.choice(letters) for _ in range(l) )
        return result

    return gen


def choice(choices):
    return partial(np.random.choice, choices)


def dataframe(**kw_args):
    def gen(length):
        columns = { n: g(length) for n, g in kw_args.items() }
        return pd.DataFrame.from_dict(columns)

    return gen


#-------------------------------------------------------------------------------

GEN0 = dataframe(
    id      =cumsum(uniform_int(1, 10)),
    name    =word(8, upper=True),
    normal0 =normal(),
    normal1 =normal(),
    normal2 =normal(),
    normal3 =normal(),
    normal4 =normal(),
    normal5 =normal(),
    normal6 =normal(),
    normal7 =normal(),
    uniform =uniform(),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("length", metavar="LEN", type=int)
    args = parser.parse_args()

    df = GEN0(args.length)
    df.index.name = "index"
    df.to_csv(sys.stdout)


if __name__ == "__main__":
    main()


