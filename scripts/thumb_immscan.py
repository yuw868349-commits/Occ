#!/usr/bin/env python3
"""Print the value that every combination of the modified immediate gives.

The class holds twelve bits: one in the first halfword, three in the
second, and eight more in the second.  This script reads all four
thousand and ninety-six combinations through both references and prints
the value each one gives, laid out so that the rule is visible.

Usage:
    thumb_immscan.py                 # the whole grid, at one byte value
    thumb_immscan.py --byte 0x3e
    thumb_immscan.py --rows 0,2,4,6  # only these imm3 values
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_ref as ref
from thumb_immfit import parse_value

SCRATCH = "/tmp/occ-thumb-immscan"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--byte", type=lambda s: int(s, 0), default=0x3E)
    ap.add_argument("--rows", default=None,
                    help="comma-separated imm3 values to show")
    args = ap.parse_args()

    rows = []
    for i in (0, 1):
        for imm3 in range(8):
            for imm8 in range(256):
                hw1 = 0xF019 & ~(1 << 10) | (i << 10)
                hw2 = (imm3 << 12) | imm8
                rows.append((hw1, hw2, (i << 11) | (imm3 << 8) | imm8))

    words = [(hw2 << 16) | hw1 for hw1, hw2, _ in rows]
    table = ref.objdump_isolated_words32(words, SCRATCH)

    wanted = None
    if args.rows:
        wanted = set(int(x, 0) for x in args.rows.split(","))

    for i, (hw1, hw2, imm12) in enumerate(rows):
        if imm12 & 0xFF != args.byte:
            continue
        imm3 = (imm12 >> 8) & 7
        if wanted is not None and imm3 not in wanted:
            continue
        addr = ref.iso32_addr(i)
        cs = ref.capstone_read((hw2 << 16) | hw1, addr, 4)
        od = table.get(addr)
        cv = parse_value(cs) if cs else None
        ov = parse_value(od) if od else None
        flag = "" if cv == ov else "   <-- the two references differ"
        print(f"  i={imm12 >> 11} imm3={imm3} imm8=0x{imm12 & 0xFF:02x} "
              f"imm12=0x{imm12:03x}  capstone="
              f"{'None' if cv is None else f'0x{cv:08x}'}  "
              f"objdump={'None' if ov is None else f'0x{ov:08x}'}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
