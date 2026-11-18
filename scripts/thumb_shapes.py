#!/usr/bin/env python3
"""Summarise a solved layout table as a set of instruction shapes.

The raw layout table has one entry per *shape*, and the shapes differ
only in which register a memory reference happens to name — so the same
kind of instruction appears many times over.  This groups them by the
shape that does not depend on the register: the mnemonic, the kinds of
the operands, and the base register class.

The result is the list of instruction shapes the decoder has to
implement, each with the bits that feed each of its operands.  It is the
reading of the architecture manual that the decoder is written from.

Usage: thumb_shapes.py PATH [--space 16]
"""
import argparse
import collections
import json
import re
import sys


def operand_ranges(bits, nbits):
    """The bit ranges that feed one operand, as a list of (hi, lo, weight).

    The measurement gives a weight per bit; a run of bits whose weights
    are consecutive powers of two is one field, and the weight tells where
    in the field the bit sits.
    """
    used = []
    for bit, info in bits.items():
        if "feeds" not in info:
            continue
        k, w, c = info["feeds"][0]
        used.append((int(bit), k, w, c))
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    with open(args.path) as f:
        table = json.load(f)

    # Group the shapes: the mnemonic and the operand kinds, with the
    # memory base's register number replaced by its class.
    groups = collections.defaultdict(list)
    for name, entry in table.items():
        shape = entry["shape"]
        mn = shape[0]
        ops = shape[1:]
        base = []
        for op in ops:
            m = re.match(r"m:([^:]+):(.):(\d+)$", op)
            if m:
                regname, hasindex, lshift = m.groups()
                # The base register: keep whether it is the stack pointer
                # or the program counter, but not which plain register.
                if regname in ("sp", "pc"):
                    base.append(f"m:{regname}:{hasindex}:{lshift}")
                else:
                    base.append(f"m:rN:{hasindex}:{lshift}")
            else:
                base.append(op)
        key = (mn, tuple(base))
        groups[key].append(entry)

    print(f"{len(groups)} instruction shapes")
    for (mn, ops), entries in sorted(groups.items(), key=lambda kv: -sum(
            e["count"] for e in kv[1])):
        total = sum(e["count"] for e in entries)
        firsts = sorted(e["first"] for e in entries)
        print()
        print(f"=== {mn}  {' '.join(ops)}   total={total}")
        print(f"    first encodings: " +
              ", ".join(f"0x{x:04x}" for x in firsts[:12]))
        # Union of the bits across the group.
        union = collections.defaultdict(collections.Counter)
        for e in entries:
            for bit, info in e["bits"].items():
                if "feeds" in info:
                    k, w, c = info["feeds"][0]
                    union[int(bit)][(k, w)] += 1
                else:
                    union[int(bit)]["form"] += 1
        for bit in sorted(union):
            top = union[bit].most_common(1)[0]
            if top[0] == "form":
                names = collections.Counter()
                for e in entries:
                    info = e["bits"].get(str(bit))
                    if info and "names" in info:
                        for n, c in info["names"].items():
                            names[n] += c
                sel = ", ".join(sorted(names))
                print(f"    b{bit:2d} -> selector: {sel}")
            else:
                (k, w) = top[0]
                print(f"    b{bit:2d} -> op{k}, weight {w}")


if __name__ == "__main__":
    sys.exit(main())
