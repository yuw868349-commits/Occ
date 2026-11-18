#!/usr/bin/env python3
"""Measure the displacement of each branch form one bit at a time.

The models in `thumb_branch32.py` score whole readings against each
other, which says which model is right but not which bit does what.  This
script answers the second question directly: it holds one encoding fixed
and flips one bit at a time, and prints how the displacement moves.  A
bit that moves the displacement by a known amount is a bit of the
displacement, and a bit that does not move it is not.

Usage:
    thumb_branch_bits.py [--form bw|bl|cond|blx]
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_ref as ref

SCRATCH = "/tmp/occ-thumb-diff/branchbits"
IMM_RE = re.compile(r"@\s*imm\s*=\s*(#-?0x[0-9a-f]+|#-?\d+)")

BASE = {
    # hw1, hw2, the name of each bit of the displacement
    "bw": (0xF000, 0x9000, "b.w"),
    "bl": (0xF000, 0xD000, "bl"),
    "cond": (0xF000, 0x8000, "beq.w"),
    "blx": (0xF000, 0xC000, "blx"),
}


def imm_of(text):
    if text is None:
        return None
    m = IMM_RE.search(text)
    if not m:
        return None
    body = m.group(1)[1:]
    neg = body.startswith("-")
    if neg:
        body = body[1:]
    value = int(body, 16) if body.startswith("0x") else int(body)
    return -value if neg else value


def read(cases):
    words = [c[0] for c in cases]
    table = ref.objdump_isolated_words32(words, SCRATCH)
    out = []
    for i in range(len(cases)):
        out.append(imm_of(table.get(ref.iso32_addr(i))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", default="all")
    args = ap.parse_args()

    for form in ("bw", "bl", "cond", "blx"):
        if args.form not in ("all", form):
            continue
        hw1, hw2, name = BASE[form]
        cases = [((hw2 << 16) | hw1, "base")]
        for bit in range(16):
            cases.append((((hw2 ^ (1 << bit)) << 16) | hw1, f"hw2[{bit}]"))
        for bit in range(10):
            cases.append((((hw2 << 16) | (hw1 ^ (1 << bit))), f"hw1[{bit}]"))
        values = read(cases)
        base = values[0]
        print(f"{name}: base displacement {base}")
        for i, (_, label) in enumerate(cases):
            if i == 0:
                continue
            value = values[i]
            if value is None:
                print(f"  {label:10s} undefined")
                continue
            print(f"  {label:10s} {value:12d}  (moves by {value - base:+d})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
