#!/usr/bin/env python3
"""Read a region of the 32-bit Thumb space and print what it holds.

The map says which class an encoding belongs to; this says what the class
looks like: which bits hold the destination, which hold the source, and
which choose between the forms.  The answer is read off the references
rather than assumed, so it is the truth for these two tools.

The output is one line per first halfword, with the reading of each probe
second halfword next to it.

Usage:
    thumb_region32.py 0xe800 0xe8ff --probes 0x0000,0x0055,0x00ff,0x0ff0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_ref as ref

SCRATCH = "/tmp/occ-thumb-region32"


def parse_probes(text):
    out = []
    for piece in text.split(","):
        piece = piece.strip()
        if piece:
            out.append(int(piece, 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lo", type=lambda s: int(s, 0))
    ap.add_argument("hi", type=lambda s: int(s, 0))
    ap.add_argument("--probes", default="0x0000,0x0055,0x00ff,0x0ff0,0xff00")
    ap.add_argument("--objdump", action="store_true",
                    help="read llvm-objdump as well as capstone")
    args = ap.parse_args()

    probes = parse_probes(args.probes)
    pairs = []
    for hw1 in range(args.lo, args.hi + 1):
        for hw2 in probes:
            pairs.append((hw1, hw2))
    words = [(hw2 << 16) | hw1 for hw1, hw2 in pairs]
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
