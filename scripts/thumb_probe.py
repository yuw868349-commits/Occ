#!/usr/bin/env python3
"""Read a list of 32-bit Thumb encodings through both references.

This is the working tool: it is what the layout of each class is read
off, one encoding at a time.  Both references are read at the same
address, so a branch means the same thing on both sides.

Usage:
    thumb_probe.py 0xea00 0x0000 0xea4f 0x0123
    thumb_probe.py --words 0xe92d0000 0xe8bd1000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_ref as ref

SCRATCH = "/tmp/occ-thumb-probe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("encodings", nargs="+",
                    help="halfword pairs, as 0xea00 0x0000")
    ap.add_argument("--words", nargs="+", default=None,
                    help="whole words, as 0xe92d0000 (hw1 in the low half)")
    ap.add_argument("--addr", type=lambda s: int(s, 0), default=0x1000)
    args = ap.parse_args()

    pairs = []
    for i in range(0, len(args.encodings) - 1, 2):
        pairs.append((int(args.encodings[i], 0), int(args.encodings[i + 1], 0)))
    for w in args.words or []:
        w = int(w, 0)
        pairs.append((w & 0xFFFF, (w >> 16) & 0xFFFF))

    words = [(hw2 << 16) | hw1 for hw1, hw2 in pairs]
    table = ref.objdump_isolated_words32(words, SCRATCH)

    for i, (hw1, hw2) in enumerate(pairs):
        addr = ref.iso32_addr(i)
        cs = ref.capstone_read((hw2 << 16) | hw1, addr, 4)
        od = table.get(addr)
        mark = "" if (cs and od) else "   "
        print(f"0x{hw1:04x}/0x{hw2:04x}  cs={cs!r}")
        print(f"               od={od!r}{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
