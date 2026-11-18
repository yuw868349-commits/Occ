#!/usr/bin/env python3
"""Read the displacement of every 32-bit Thumb branch form out of objdump.

llvm-objdump prints the displacement of a branch as an `@ imm = #0x..`
annotation.  That annotation is the displacement itself, so it can be
compared against a model without any rewriting of addresses, which makes
it the most direct reading of the architecture that is available.

The four forms do not share a layout, so each is read on its own:

    B<cond>.w   11110 S cond(4) imm6 | 10 J1 0 J2 imm11
    B.w         11110 S imm10      | 10 J1 1 J2 imm11
    BL          11110 S imm10      | 11 J1 1 J2 imm11
    BLX         11110 S imm10H     | 11 J1 0 J2 imm10L

The model is not assumed here.  For each form the script tries every
candidate it can think of and reports how many readings each one gets
right, so that the layout is a measurement rather than a memory.

Usage:
    thumb_branch32.py [--form NAME]
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_ref as ref

SCRATCH = "/tmp/occ-thumb-diff/branch32"

IMM_RE = re.compile(r"@\s*imm\s*=\s*(#-?0x[0-9a-f]+|#-?\d+)")


def imm_of(text):
    """The displacement objdump printed, or None."""
    if text is None:
        return None
    m = IMM_RE.search(text)
    if not m:
        return None
    body = m.group(1)[1:]
    return int(body, 16) if "0x" in body else int(body, 10)


def sign_extend(value, width):
    """The value as a signed number of that width."""
    return value - (1 << width) if value & (1 << (width - 1)) else value


def cases_conditional():
    """B<cond>.w: 11110 S cond(4) imm6 | 10 J1 0 J2 imm11."""
    out = []
    for s in (0, 1):
        for cond in (0x0, 0xE):
            for j1 in (0, 1):
                for j2 in (0, 1):
                    for imm6 in (0x00, 0x01, 0x20, 0x3F):
                        for imm11 in (0x000, 0x001, 0x400, 0x7FF):
                            hw1 = 0xF000 | (s << 10) | (cond << 6) | imm6
                            hw2 = 0x8000 | (j1 << 13) | (j2 << 11) | imm11
                            out.append(((hw2 << 16) | hw1,
                                        (s, cond, j1, j2, imm6, imm11)))
    return out


def cases_wide(prefix):
    """B.w and BL: 11110 S imm10 | 1x J1 1 J2 imm11."""
    out = []
    for s in (0, 1):
        for j1 in (0, 1):
            for j2 in (0, 1):
                for imm10 in (0x000, 0x001, 0x200, 0x3FF):
                    for imm11 in (0x000, 0x001, 0x400, 0x7FF):
                        hw1 = 0xF000 | (s << 10) | imm10
                        hw2 = prefix | (j1 << 13) | (j2 << 11) | imm11
                        out.append(((hw2 << 16) | hw1, (s, j1, j2, imm10, imm11)))
    return out


def cases_blx():
    """BLX: 11110 S imm10H | 11 J1 0 J2 imm10L H."""
    out = []
    for s in (0, 1):
        for j1 in (0, 1):
            for j2 in (0, 1):
                for imm10 in (0x000, 0x001, 0x200, 0x3FF):
                    for imm10l in (0x000, 0x001, 0x200, 0x3FF):
                        for h in (0, 1):
                            hw1 = 0xF000 | (s << 10) | imm10
                            hw2 = 0xC000 | (j1 << 13) | (j2 << 11) | (imm10l << 1) | h
                            out.append(((hw2 << 16) | hw1,
                                        (s, j1, j2, imm10, imm10l, h)))
    return out


# The models to try.  Each is a function from the fields to a displacement.
#
# The question each one answers is which of the two jump bits comes first
# in the field, and whether the jump bits are folded with the sign bit.
def wide_models(s, j1, j2, imm10, imm11):
    i1 = 1 ^ (j1 ^ s)
    i2 = 1 ^ (j2 ^ s)
    base = (imm10 << 12) | (imm11 << 1)
    yield "i1_first_folded", sign_extend(base | (s << 24) | (i1 << 23) | (i2 << 22), 25)
    yield "i2_first_folded", sign_extend(base | (s << 24) | (i2 << 23) | (i1 << 22), 25)
    yield "j1_first_raw", sign_extend(base | (s << 24) | (j1 << 23) | (j2 << 22), 25)
    yield "j2_first_raw", sign_extend(base | (s << 24) | (j2 << 23) | (j1 << 22), 25)


def conditional_models(s, cond, j1, j2, imm6, imm11):
    base = (imm6 << 12) | (imm11 << 1)
    yield "j2_first_raw", sign_extend(base | (s << 20) | (j2 << 19) | (j1 << 18), 21)
    yield "j1_first_raw", sign_extend(base | (s << 20) | (j1 << 19) | (j2 << 18), 21)
    i1 = 1 ^ (j1 ^ s)
    i2 = 1 ^ (j2 ^ s)
    yield "i2_first_folded", sign_extend(base | (s << 20) | (i2 << 19) | (i1 << 18), 21)
    yield "i1_first_folded", sign_extend(base | (s << 20) | (i1 << 19) | (i2 << 18), 21)


def blx_models(s, j1, j2, imm10, imm10l, h):
    i1 = 1 ^ (j1 ^ s)
    i2 = 1 ^ (j2 ^ s)
    # The target of a call to a halfword is a halfword, so the low bit of
    # the displacement is clear, and the H bit says how wide the field is.
    yield "24_i1_first", sign_extend(
        (s << 23) | (i1 << 22) | (i2 << 21) | (imm10 << 11) | (imm10l << 1), 24)
    yield "25_i1_first", sign_extend(
        (s << 24) | (i1 << 23) | (i2 << 22) | (imm10 << 12) | (imm10l << 2), 25)
    yield "24_i2_first", sign_extend(
        (s << 23) | (i2 << 22) | (i1 << 21) | (imm10 << 11) | (imm10l << 1), 24)
    yield "24_j2_first", sign_extend(
        (s << 23) | (j2 << 22) | (j1 << 21) | (imm10 << 11) | (imm10l << 1), 24)
    # The H bit turns a displacement of ten bits into one of twenty-four.
    if h == 0:
        yield "h0_low_only", sign_extend(
            (s << 23) | (i1 << 22) | (i2 << 21) | (imm10l << 1), 24)
    else:
        yield "h1_both", sign_extend(
            (s << 23) | (i1 << 22) | (i2 << 21) | (imm10 << 11) | (imm10l << 1), 24)


def run(name, cases, models, isolated=True):
    words = [c[0] for c in cases]
    if isolated:
        table = ref.objdump_isolated_words32(words, SCRATCH)
    else:
        table = ref.objdump_read(words, 4, SCRATCH)
    scores = {}
    total = 0
    misses = []
    for i, (_, fields) in enumerate(cases):
        want = imm_of(table.get(ref.iso32_addr(i)))
        if want is None:
            continue
        total += 1
        for label, value in models(*fields):
            scores[label] = scores.get(label, 0) + (1 if value == want else 0)
        if len(misses) < 6:
            best = None
            for label, value in models(*fields):
                if value == want:
                    best = label
            misses.append((fields, want, best))
    print(f"{name}: {total} readings")
    for label in sorted(scores, key=lambda k: -scores[k]):
        print(f"  {label:20s} {scores[label]}")
    for fields, want, best in misses:
        print(f"    miss {fields} want={want} best={best}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", default="all")
    args = ap.parse_args()
    if args.form in ("all", "bw"):
        run("B.w", cases_wide(0x9000), wide_models)
    if args.form in ("all", "bl"):
        run("BL", cases_wide(0xD000), wide_models)
    if args.form in ("all", "cond"):
        run("B<cond>.w", cases_conditional(), conditional_models)
    if args.form in ("all", "blx"):
        run("BLX", cases_blx(), blx_models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
