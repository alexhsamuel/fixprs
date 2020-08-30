#!/usr/bin/env python

import argparse
import logging
import os
from   pathlib import Path
import time

import pandas as pd

import fixprs.ext2
import gen

#-------------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"

SCHEMAS = {
    "4icol": gen.dataframe(
        label   =gen.word(8, upper=True),
        val0    =gen.uniform_int(1,     100),
        val1    =gen.uniform_int(1,   10000),
        val2    =gen.uniform_int(1, 1000000),
    ),
}

def get_data_file(schema_name, length):
    path = DATA_DIR / f"{schema_name}-{length}.csv"
    if not path.is_file():
        logging.info(f"generating: {path}")
        df = SCHEMAS[schema_name](length)
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(path, index=False)
    return path


#-------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
args = parser.parse_args()

path = get_data_file("4icol", 1000000)

for _ in range(3):
    t0 = time.perf_counter()
    with open(path, "rb") as file:
        res = fixprs.ext2.parse_buffer(file.read())
    elapsed = time.perf_counter() - t0
    print("fixprs", elapsed)

    t0 = time.perf_counter()
    df = pd.read_csv(path, dtype={
        "label" : str,
        "val0"  : int,
        "val1"  : int,
        "val2"  : int,
    })
    elapsed = time.perf_counter() - t0
    print("pd.read_csv", elapsed)

