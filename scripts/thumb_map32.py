#!/usr/bin/env python3
"""Build the full map of the 32-bit Thumb space.

The map is from the first halfword of an encoding to the class it opens.
The first halfword of a 32-bit Thumb instruction is one of `11101`,
`11110` or `11111`, so the space opens at `0xe800` and there are 7680
first halfwords in it.

The class of a first halfword is not always decided by the first halfword
alone: some classes share the first halfword and are told apart by the
second one.  So the map is built by reading the first halfword behind a
set of second halfwords that between them exercise every bit, and the
class of a first halfword is the set of classes that are met.

The reading is done with capstone, which is quick enough to cover the
whole space; `thumb_diff32.py` is what settles the spelling against the
second reference.

Usage:
    thumb_map32.py [--json PATH]
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_ref as ref

# The second halfwords the first halfword is read behind.  Every bit of
# the second halfword is exercised by one of them.
PROBES = (0x0000, 0xFFFF, 0x5555, 0xAAAA, 0x0F0F, 0xF0F0, 0x00FF, 0xFF00,
          0x0F00, 0xF000, 0x00F0, 0xF00F, 0x8000, 0x0800, 0x0080, 0x000F)


def shape_of(text):
    """The shape of a reading: the mnemonic and the kind of each operand."""
    if text is None:
        return None
    parts = text.split(None, 1)
    mnemonic = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    kinds = []
    for piece in rest.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if piece.startswith("#"):
            kinds.append("i")
        elif piece.startswith("["):
            kinds.append("m")
        else:
            kinds.append("r")
    return " ".join([mnemonic] + kinds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="/tmp/occ-thumb-table/map32.json")
    args = ap.parse_args()

    # Every encoding is read once per probe.
    encodings = []
    for hw1 in range(0xE800, 0x10000):
        for hw2 in PROBES:
            encodings.append(((hw2 << 16) | hw1, hw1, hw2))

    md = __import__("capstone").Cs(__import__("capstone").CS_ARCH_ARM,
                                   __import__("capstone").CS_MODE_THUMB)
    md.detail = False

    # Read them all in one pass: capstone is a library, so this is fast.
    readings = collections.defaultdict(set)
    for w, hw1, hw2 in encodings:
        code = w.to_bytes(4, "little")
        for i in md.disasm(code, 0x1000, count=1):
            if i.size == 4:
                readings[hw1].add(shape_of(f"{i.mnemonic} {i.op_str}".strip()))
            break

    table = {}
    for hw1 in range(0xE800, 0x10000):
        table[hw1] = sorted(readings.get(hw1, ()))

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({str(k): v for k, v in table.items()}, f)

    # A report of the classes.
    classes = collections.defaultdict(list)
    for hw1, shapes in table.items():
        classes[tuple(shapes)].append(hw1)
    print(f"{len(table)} first halfwords, {len(classes)} classes")
    ranked = sorted(classes.items(), key=lambda kv: -len(kv[1]))
    for shapes, hw1s in ranked:
        lo, hi = min(hw1s), max(hw1s)
        print(f"  {len(hw1s):5d}  0x{lo:04x}-0x{hi:04x}  {' | '.join(shapes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
