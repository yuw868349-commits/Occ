#!/usr/bin/env python3
"""Print the reference's reading of one region of the 16-bit space, as a table.

Some regions of the space are laid out as a two-dimensional field — the
condition in one field, the mask in another — and the meaning of the
combination is not something that can be written down from memory: it has
to be read off the table.

So the region is read with both references and printed as a grid: the rows
are one field, the columns the other, and every cell says what the
instruction is.

Usage:
    thumb_grid.py 0xbf00 0xbfff
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capstone as cs

ADDR = 0x1000


def read(hw):
    md = cs.Cs(cs.CS_ARCH_ARM, cs.CS_MODE_THUMB)
    code = hw.to_bytes(2, "little")
    for i in md.disasm(code, ADDR, count=1):
        if i.size != 2:
            return None
        return f"{i.mnemonic} {i.op_str}".strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lo", type=lambda s: int(s, 16))
    ap.add_argument("hi", type=lambda s: int(s, 16))
    ap.add_argument("--row", type=int, default=4,
                    help="the bit at the bottom of the row field")
    ap.add_argument("--rowlen", type=int, default=4)
    ap.add_argument("--col", type=int, default=0,
                    help="the bit at the bottom of the column field")
    ap.add_argument("--collen", type=int, default=4)
    args = ap.parse_args()

    lo, hi = args.lo, args.hi
    rwidth = 1 << args.rowlen
    cwidth = 1 << args.collen
    rmask = (rwidth - 1) << args.row
    cmask = (cwidth - 1) << args.col

    print(f"rows: bits[{args.row + args.rowlen - 1}:{args.row}]   "
          f"columns: bits[{args.col + args.collen - 1}:{args.col}]")
    print()
    for r in range(rwidth):
        cells = []
        for c in range(cwidth):
            hw = (lo & ~(rmask | cmask)) | (r << args.row) | (c << args.col)
            if hw < lo or hw > hi:
                cells.append("-")
                continue
            t = read(hw)
            cells.append(t if t else "(undef)")
        print(f"row {r:2d} (0x{r:02x})  " +
              " | ".join(f"{c:>20s}" for c in cells))


if __name__ == "__main__":
    main()
