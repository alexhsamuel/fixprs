import numpy as np
from   numpy.testing import assert_array_equal as arr_eq
from   pathlib import Path

import fixprs

DIR = Path(__file__).parent

#-------------------------------------------------------------------------------

def test_load_basic0():
    a = fixprs.load_file(DIR / "basic0.csv")
    assert list(a) == ["idx", "name", "val"]
    arr_eq(a["idx"], [3, 5, 6, 11])
    # FIXME: Should return str!
    arr_eq(np.char.decode(a["name"]), ["foo", "bar", "baz", "bif"])
    arr_eq(a["val"], [3.14159, 2.71828, 0, -1])


def test_basic0():
    a = fixprs.parse_file(DIR / "basic0.csv")
    assert list(a) == ["idx", "name", "val"]

    # FIXME: Should return strs!
    a = { n: np.char.decode(v) for n, v in a.items() }

    arr_eq(a["idx"], ["3", "5", "6", "11"])
    arr_eq(a["name"], ["foo", "bar", "baz", "bif"])
    arr_eq(a["val"], ["3.14159", "2.71828", "0", "-1"])


