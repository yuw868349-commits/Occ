#!/usr/bin/env python3
"""Read one class of the 32-bit Thumb space in full and print what it holds.

The class is a range of first halfwords.  Every first halfword in the
range is read behind a set of second halfwords, and the readings are
printed so that the fields of the class can be read off.

Usage:
    thumb_class32.py 0xf040 0xf05f --hw2 0x0123
    thumb_class32.py 0xf040 0xf05f --hw2 0x0123,0x0234
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_ref as ref

SCRATCH = "/tmp/occ-thumb-class32"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lo", type=lambda s: int(s, 0))
    ap.add_argument("hi", type=lambda s: int(s, 0))
    ap.add_argument("--hw2", default="0x0123")
    ap.add_argument("--objdump", action="store_true")
    args = ap.parse_args()

    hw2s = [int(x, 0) for x in args.hw2.split(",")]
    hw1s = list(range(args.lo, args.hi + 1))
    pairs = [(h, w) for h in hw1s for w in hw2s]
    words = [(w << 16) | h for h, w in pairs]
    table = ref.objdump_isolated_words32(words, SCRATCH) if args.objdump else {}

    for i, (hw1, hw2) in enumerate(pairs):
        addr = ref.iso32_addr(i)
        cs = ref.capstone_read((hw2 << 16) | hw1, addr, 4)
        od = table.get(addr)
        if args.objdump:
            print(f"0x{hw1:04x}/0x{hw2:04x}  {cs!r}\n    {od!r}")
        else:
            print(f"0x{hw1:04x}/0x{hw2:04x}  {cs!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
