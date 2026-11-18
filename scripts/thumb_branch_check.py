#!/usr/bin/env python3
"""Cross-check the branch models against capstone's absolute destinations.

llvm-objdump reports the displacement of a branch, which settles the layout
of the field.  capstone reports the *destination*, which settles a
different question: whether the destination is computed in the thirty-two
bit space the architecture has, or in a wider one.

The two are read at a known address, and the destination capstone prints is
compared against the destination the model predicts, both reduced to the
thirty-two bits the architecture works in.

Usage:
    thumb_branch_check.py [--form NAME]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_ref as ref

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thumb_branch32 import (cases_blx, cases_conditional, cases_wide,
                            sign_extend)

ADDR = 0x1000
MASK = 0xFFFFFFFF


def capstone_target(text):
    """The destination capstone printed, or None."""
    if text is None:
        return None
    for part in text.split("#"):
        part = part.strip()
        if part.startswith("0x"):
            return int(part, 16)
    return None


def models_for(form, fields):
    """The displacement models, by form."""
    if form == "cond":
        s, cond, j1, j2, imm6, imm11 = fields
        base = (imm6 << 12) | (imm11 << 1)
        return {
            "j2_first_raw": sign_extend(base | (s << 20) | (j2 << 19) | (j1 << 18), 21),
        }
    if form == "blx":
        s, j1, j2, imm10, imm10l, h = fields
        i1 = 1 ^ (j1 ^ s)
        i2 = 1 ^ (j2 ^ s)
        return {
            "25_i1_first": sign_extend(
                (s << 24) | (i1 << 23) | (i2 << 22) | (imm10 << 12) | (imm10l << 2), 25),
            "24_i1_first": sign_extend(
                (s << 23) | (i1 << 22) | (i2 << 21) | (imm10 << 11) | (imm10l << 1), 24),
        }
    s, j1, j2, imm10, imm11 = fields
    i1 = 1 ^ (j1 ^ s)
    i2 = 1 ^ (j2 ^ s)
    base = (imm10 << 12) | (imm11 << 1)
    return {
        "25_i1_first": sign_extend(base | (s << 24) | (i1 << 23) | (i2 << 22), 25),
        "25_i2_first": sign_extend(base | (s << 24) | (i2 << 23) | (i1 << 22), 25),
    }


def check(form, cases):
    words = [c[0] for c in cases]
    scores = {}
    total = 0
    for i, (_, fields) in enumerate(cases):
        text = ref.capstone_read(words[i], ADDR, 4)
        target = capstone_target(text)
        if target is None:
            continue
        total += 1
        pc = ADDR + 4
        for label, amount in models_for(form, fields).items():
            # The destination the architecture computes, in the thirty-two
            # bit space it works in.
            want = (pc + amount) & MASK
            scores[label] = scores.get(label, 0) + (1 if want == target else 0)
    print(f"{form}: {total} readings")
    for label in sorted(scores, key=lambda k: -scores[k]):
        print(f"  {label:20s} {scores[label]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", default="all")
    args = ap.parse_args()
    if args.form in ("all", "bw"):
        check("bw", cases_wide(0x9000))
    if args.form in ("all", "bl"):
        check("bl", cases_wide(0xD000))
    if args.form in ("all", "cond"):
        check("cond", cases_conditional())
    if args.form in ("all", "blx"):
        check("blx", cases_blx())
    return 0


if __name__ == "__main__":
    sys.exit(main())
