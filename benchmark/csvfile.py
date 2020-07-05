import argparse
import csv
import logging
import numpy as np
import pandas as pd
from   pathlib import Path

import fixprs
import gen
from   timing import benchmark

#-------------------------------------------------------------------------------

def load_csv_arr(path, dtype):
    """
    Loads a CSV file to a dict of U arrays using the `csv` module.
    """
    with open(path) as file:
        reader = csv.reader(file)
        names = next(reader)
        arrs = ( np.array(a, dtype=dtype) for a in zip(*reader) )
        return dict(zip(names, arrs))


def run(path):
    yield "csv", "load_csv_arr dtype=object", benchmark(lambda: load_csv_arr(path, object))
    yield "csv", "load_csv_arr dtype=S20", benchmark(lambda: load_csv_arr(path, "S20"))
    yield "np", "loadtxt str-recarray", benchmark(
        lambda: np.loadtxt(path, skiprows=1, delimiter=",", dtype=np.dtype([ (str(n), str) for n in range(12) ])))
    # FIXME: This returns O arrays of str.
    yield "pd", "load_csv dtype=str", benchmark(lambda: pd.read_csv(path, dtype=str))
    # FIXME: This returns S arrays.
    yield "fixprs", "parse_file", benchmark(lambda: fixprs.parse_file(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("length", metavar="LEN", type=int)
    args = parser.parse_args()
    length = args.length

    path = Path(__file__).parent / f"gen/csvfile-{length}.csv"
    if not path.exists():
        logging.info(f"generating: {path}")
        df = gen.GEN0(args.length)
        df.to_csv(path)

    for lib, name, elapsed in run(path):
        print(f"{lib:8s} {name:40s} {elapsed / length / 1e-6:8.3f} µs")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

