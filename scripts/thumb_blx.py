#!/usr/bin/env python3
"""Read the rules of the BLX form out of objdump and capstone.

The call to a halfword-aligned target has a bit in the second halfword
that says whether the target is halfword-aligned or word-aligned, and the
architecture only defines the call when that bit is clear.  Whether that
is so is a question about the encoding, and it is measured here rather
than assumed: every value of that bit is read through both references and
counted.

Usage:
    thumb_blx.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_ref as ref

SCRATCH = "/tmp/occ-thumb-diff/blx"


def main():
    # Every combination of the fields that the form has, with the bit in
    # question both ways.
    cases = []
    for s in (0, 1):
        for j1 in (0, 1):
            for j2 in (0, 1):
                for h in (0, 1):
                    for imm10 in (0x000, 0x123, 0x3FF):
                        for imm10l in (0x000, 0x045, 0x3FF):
                            hw1 = 0xF000 | (s << 10) | imm10
                            hw2 = 0xC000 | (j1 << 13) | (j2 << 11) | (imm10l << 1) | h
                            cases.append(((hw2 << 16) | hw1, (s, j1, j2, h, imm10, imm10l)))
    words = [c[0] for c in cases]
    table = ref.objdump_isolated_words32(words, SCRATCH)

    defined = {}
    undefined = {}
    for i, (_, fields) in enumerate(cases):
        addr = ref.iso32_addr(i)
        text = table.get(addr)
        s, j1, j2, h, imm10, imm10l = fields
        if text is None:
            undefined[h] = undefined.get(h, 0) + 1
        else:
            defined[h] = defined.get(h, 0) + 1
            if not text.startswith("blx"):
                print(f"  a defined call that is not a call: {fields} -> {text}")
    print("the bit at hw2[0]:")
    for h in (0, 1):
        print(f"  h={h}: defined {defined.get(h, 0)}, undefined {undefined.get(h, 0)}")

    # The displacement of the calls that are defined, against the model.
    import re
    good = 0
    total = 0
    for i, (_, fields) in enumerate(cases):
        addr = ref.iso32_addr(i)
        text = table.get(addr)
        if text is None:
            continue
        m = re.search(r"@\s*imm\s*=\s*(#-?0x[0-9a-f]+)", text)
        if not m:
            continue
        body = m.group(1)[1:]
        neg = body.startswith("-")
        if neg:
            body = body[1:]
        want = int(body, 16) if body.startswith("0x") else int(body)
        if neg:
            want = -want
        s, j1, j2, h, imm10, imm10l = fields
        i1 = 1 ^ (j1 ^ s)
        i2 = 1 ^ (j2 ^ s)
        v = (s << 24) | (i1 << 23) | (i2 << 22) | (imm10 << 12) | (imm10l << 2)
        if v & (1 << 24):
            v -= 1 << 25
        total += 1
        if v == want:
            good += 1
    print(f"displacement model: {good}/{total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
