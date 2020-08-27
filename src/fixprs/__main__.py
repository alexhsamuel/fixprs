import argparse
from   pathlib import Path

import fixprs.ext2


parser = argparse.ArgumentParser()
parser.add_argument("path", metavar="PATH", type=Path)
args = parser.parse_args()

with open(args.path, "rb") as file:
    data = file.read()

arrs = fixprs.ext2.parse_buffer(data)
print(f"got {len(arrs)} arrs")

for arr in arrs:
    print(f"{str(arr.dtype)[: 20]:20s} {str(arr)[: 58]}")

