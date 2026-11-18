#!/usr/bin/env python3
"""Read one class of the wide-immediate space and print what it holds.

The class with a wide immediate holds several different instructions,
each with its own layout, and the layouts differ in ways that a
description of the class does not say: the bit that tells the shift
type from the shift amount, the bit that must be clear, the number of
bits of the saturation value.  This script reads the class behind a
grid of second halfwords and prints the readings, so that each of those
can be read off.

Usage:
    thumb_wide32.py 0xf300 0xf3ff
    thumb_wide32.py 0xf360 0xf36f --probes 0x0000,0x0007,0x001f
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_ref as ref

SCRATCH = "/tmp/occ-thumb-wide32"

# The second halfwords the class is read behind: the ones that put a
# value in each field on its own.
PROBES = (0x0000, 0x0001, 0x0007, 0x0008, 0x000f, 0x0010, 0x001f, 0x0020,
          0x0040, 0x0060, 0x0080, 0x00a0, 0x00c0, 0x00e0, 0x00f0, 0x0100,
          0x1000, 0x2000, 0x4000, 0x7000, 0x701f, 0x7fff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lo", type=lambda s: int(s, 0))
    ap.add_argument("hi", type=lambda s: int(s, 0))
    ap.add_argument("--probes", default=None)
    args = ap.parse_args()

    probes = ([int(x, 0) for x in args.probes.split(",")]
              if args.probes else list(PROBES))
    hw1s = list(range(args.lo, args.hi + 1))
    pairs = [(h, w) for h in hw1s for w in probes]
    words = [(w << 16) | h for h, w in pairs]
    table = ref.objdump_isolated_words32(words, SCRATCH)

    for i, (hw1, hw2) in enumerate(pairs):
        addr = ref.iso32_addr(i)
        cs = ref.capstone_read((hw2 << 16) | hw1, addr, 4)
        od = table.get(addr)
        # The two references must agree; where they do not, the
        # difference is printed so that it is not lost.
        if cs is not None and od is not None:
            same = "" if cs.replace(" ", "") == od.replace("\t", "").replace(" ", "") else "   <<"
        else:
            same = ""
        text = cs if cs is not None else (od if od is not None else "-")
        print(f"0x{hw1:04x}/0x{hw2:04x}  {text}{same}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
