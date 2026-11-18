#!/usr/bin/env python3
"""Print the form of every encoding in a range, one row per first halfword.

Each row holds the first halfword and, next to it, what the reference
made of it behind each of the second halfwords.  A row whose readings all
say the same thing holds one form; a row whose readings differ holds
several, and the second halfword is what tells them apart.

Usage:
    thumb_form32.py 0xf300 0xf3ff
    thumb_form32.py 0xf300 0xf3ff --probes 0x0000,0x0020,0x0040
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_ref as ref

SCRATCH = "/tmp/occ-thumb-form32"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lo", type=lambda s: int(s, 0))
    ap.add_argument("hi", type=lambda s: int(s, 0))
    ap.add_argument("--probes", default="0x0000,0x0020,0x0040,0x0080,0x0100,"
                                       "0x0200,0x0400,0x0800,0x1000,0x2000,"
                                       "0x4000,0x8000")
    ap.add_argument("--objdump", action="store_true")
    args = ap.parse_args()

    probes = [int(x, 0) for x in args.probes.split(",")]
    hw1s = list(range(args.lo, args.hi + 1))
    pairs = [(h, w) for h in hw1s for w in probes]
    words = [(w << 16) | h for h, w in pairs]
    table = ref.objdump_isolated_words32(words, SCRATCH) if args.objdump else {}

    for i, (hw1, hw2) in enumerate(pairs):
        addr = ref.iso32_addr(i)
        cs = ref.capstone_read((hw2 << 16) | hw1, addr, 4)
        od = table.get(addr)
        text = cs if cs is not None else (od if od is not None else "-")
        print(f"0x{hw1:04x}/0x{hw2:04x}  {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
