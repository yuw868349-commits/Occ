#!/usr/bin/env python3
"""Read the immediate of one T32 class, field by field.

The twelve-bit immediate of the modified-immediate class is spread over
the two halfwords: a bit of the first, three of the second, and eight
more of the second.  The value it stands for is not the twelve bits
themselves, and how it is turned into a thirty-two-bit value is the
thing this script measures.

The output is one line per reading: the encoding, and the value the
references give for it.  Nothing is assumed about the rule; the rule is
read off these lines.

Usage:
    thumb_imm.py 0xf019            # the class, by its first halfword
    thumb_imm.py 0xf019 --wide 0x1234
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_ref as ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hw1", type=lambda s: int(s, 0))
    ap.add_argument("--i", type=int, default=None,
                    help="the value of the bit of the first halfword that "
                         "belongs to the immediate")
    ap.add_argument("--imm3", type=int, default=None)
    ap.add_argument("--imm8", type=int, default=None)
    args = ap.parse_args()

    rows = []
    if args.i is not None and args.imm3 is not None and args.imm8 is not None:
        # One reading at a time: vary the three fields on their own.
        for i in (0, 1):
            for imm3 in range(8):
                for imm8 in (0x00, 0x3E, 0xFF):
                    hw1 = args.hw1 & ~(1 << 10) | (i << 10)
                    hw2 = (imm3 << 12) | (imm8 & 0xFF)
                    rows.append((hw1, hw2))
    else:
        rows.append((args.hw1, 0x1234))

    pairs = list(rows)
    words = [(hw2 << 16) | hw1 for hw1, hw2 in pairs]
    for i, (hw1, hw2) in enumerate(pairs):
        addr = ref.iso32_addr(i)
        cs = ref.capstone_read((hw2 << 16) | hw1, addr, 4)
        print(f"0x{hw1:04x}/0x{hw2:04x}  {cs!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
