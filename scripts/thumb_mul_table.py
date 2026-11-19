#!/usr/bin/env python3
"""Read the T32 multiply and divide family out of the reference.

The family is two layouts in one: the second halfword's top nibble says
which layout, and each layout has its own way of naming the sub-form.
The names and the operand orders are read out of `llvm-objdump` rather
than taken from a table in a manual, because the manual's table is
spelled differently from what a disassembler prints.

Usage:
    thumb_mul_table.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import thumb_ref as ref  # noqa: E402

DECODER = os.path.join(HERE, "..", "target", "release", "examples", "thumb_decode")


def ours(encodings):
    payload = "".join(f"{a:04x} {b:04x} 0\n" for a, b in encodings)
    res = subprocess.run([DECODER], input=payload, capture_output=True, text=True)
    out = []
    for line in res.stdout.splitlines():
        parts = line.split(" ", 2)
        out.append(parts[2] if len(parts) > 2 else "")
    return out


def theirs(encodings):
    words = [((b << 16) | a) for a, b in encodings]
    table = ref.objdump_isolated_words32(words, "/tmp/occ-thumb-diff/multable")
    out = []
    for i in range(len(encodings)):
        text = table.get(ref.iso32_addr(i))
        if text is None:
            out.append(None)
            continue
        text = text.split("<")[0].strip()
        out.append(text.replace("\t", " "))
    return out


def main():
    encodings = []
    keys = []
    # The layout with the top nibble clear: the sub-form is in bits 5:4.
    for op in range(16):
        for variant in range(4):
            encodings.append((0xFB00 | (op << 4) | 0x1, (variant << 4) | 0x3))
            keys.append(("A", op, variant))
    # The layout with the top nibble set: the sub-form is in bits 7:4.
    for op in range(16):
        for sub in range(16):
            encodings.append((0xFB00 | (op << 4) | 0x1, 0xF000 | (sub << 4) | 0x3))
            keys.append(("B", op, sub))

    mine = ours(encodings)
    theirs_ = theirs(encodings)
    for key, m, t in zip(keys, mine, theirs_):
        layout, op, sub = key
        if t is None:
            continue
        name = t.split(" ")[0]
        agree = "=" if m == t else "!"
        print(f"{layout} op={op:04b} sub={sub:04b} {name:10s} {agree} {t}")


if __name__ == "__main__":
    sys.exit(main())
