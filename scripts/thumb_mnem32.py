#!/usr/bin/env python3
"""Print the mnemonic of every first halfword in a range, one row per row.

The table is read from the references, so it is the truth for these two
tools and not a recollection of the architecture manual.  Where the
second halfword changes the answer, the row says so.

Usage:
    thumb_mnem32.py 0xf000 0xf1ff --hw2 0x0123
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_ref as ref

SCRATCH = "/tmp/occ-thumb-mnem32"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lo", type=lambda s: int(s, 0))
    ap.add_argument("hi", type=lambda s: int(s, 0))
    ap.add_argument("--hw2", default="0x0123")
    ap.add_argument("--objdump", action="store_true")
    args = ap.parse_args()

    hw2 = int(args.hw2, 0)
    hw1s = list(range(args.lo, args.hi + 1))
    pairs = [(h, hw2) for h in hw1s]
    words = [(hw2 << 16) | h for h, _ in pairs]
    table = ref.objdump_isolated_words32(words, SCRATCH) if args.objdump else {}

    for i, hw1 in enumerate(hw1s):
        addr = ref.iso32_addr(i)
        cs = ref.capstone_read((hw2 << 16) | hw1, addr, 4)
        if args.objdump:
            od = table.get(addr)
            same = "" if (cs and od and cs.replace(" ", "") == od.replace("\t", "").replace(" ", "")) else "  <<"
            print(f"0x{hw1:04x}  {cs!r}\n     {od!r}{same}")
        else:
            print(f"0x{hw1:04x}  {cs!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
