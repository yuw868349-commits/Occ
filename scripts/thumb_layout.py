#!/usr/bin/env python3
"""Solve the layout of every Thumb class by measurement.

A decoder is written by *reading the architecture manual*, and a manual
describes an encoding in prose: "the destination is in bits 2 to 0".  A
prose description is easy to get subtly wrong, and a wrong field shows up
much later as a wrong decompilation — long after the mistake was made.

This script does not read the manual.  It takes the encoding space apart and
watches what moves.  For each class of instruction it flips one bit at a
time and compares the two readings: the operand that changed is the operand
that bit feeds, and the size of the change is the bit's weight inside that
operand.

The answer is a table, per class, of

* which mnemonic each bit selects between,
* which operand each bit feeds, and
* what weight the bit has there.

That table is what the decoder is then written from.  Nothing in the
decoder's field layouts is a matter of memory.

The reading is capstone's *structural* one — its operand list says which
operand is a register, which is an immediate, and which is a memory
reference — and that is what makes the measurement possible.

Usage:
    thumb_layout.py --space 16 --out /tmp/occ-thumb-table/l16.json
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import capstone as cs

REGID = {f"r{i}": i for i in range(13)}
REGID["sp"] = 13
REGID["lr"] = 14
REGID["pc"] = 15


def read16(hw):
    """Read one halfword with capstone, structurally."""
    md = cs.Cs(cs.CS_ARCH_ARM, cs.CS_MODE_THUMB)
    md.detail = True
    code = hw.to_bytes(2, "little")
    for i in md.disasm(code, 0, count=1):
        if i.size != 2:
            return None
        ops = []
        for o in i.operands:
            if o.type == cs.arm.ARM_OP_REG:
                ops.append(("r", REGID.get(i.reg_name(o.reg)), i.reg_name(o.reg)))
            elif o.type == cs.arm.ARM_OP_IMM:
                ops.append(("i", o.imm))
            elif o.type == cs.arm.ARM_OP_MEM:
                index = o.mem.index
                ops.append(("m", i.reg_name(o.mem.base),
                            -1 if index == 0 else REGID.get(i.reg_name(index), -2),
                            o.mem.disp, o.mem.lshift))
            else:
                return None
        return (i.mnemonic, ops)
    return None


def shape_of(reading):
    """The shape of a reading: the mnemonic and the kinds of the operands.

    Two encodings with the same shape are the same *kind* of instruction,
    and they are the unit the layout table is keyed on.
    """
    mn, ops = reading
    out = [mn]
    for op in ops:
        if op[0] == "r":
            out.append("r")
        elif op[0] == "i":
            out.append("i")
        else:
            out.append(f"m:{op[1]}:{'i' if op[2] != -1 else '-'}:{op[4]}")
    return tuple(out)


def change(before, after):
    """Describe what one flipped bit did.

    The return value is a list of (operand index, weight) pairs, where the
    weight is the amount the operand's value changed by — which for a
    register is the bit's value inside the register number, and for an
    immediate is the bit's value inside the number.

    The string "form" means the reading changed in a way that is not a
    field change: the mnemonic moved, or the number of operands moved, or
    a memory reference changed its base.  Those are the bits that choose
    between forms, and they are recorded separately.
    """
    if before is None or after is None:
        return None
    if before[0] != after[0] or len(before[1]) != len(after[1]):
        return "form"
    out = []
    for k, (a, b) in enumerate(zip(before[1], after[1])):
        if a[0] != b[0]:
            return "form"
        if a[0] == "r":
            if a[1] == b[1]:
                continue
            if a[1] is None or b[1] is None:
                return "form"
            delta = a[1] ^ b[1]
            if delta & (delta - 1):
                return "form"
            out.append((k, delta))
        elif a[0] == "i":
            d = b[1] - a[1]
            if d == 0:
                continue
            if d < 0 or (d & (d - 1)):
                return "form"
            out.append((k, d))
        else:
            if a[1] != b[1]:
                return "form"
            if a[2] != b[2]:
                if a[2] == -1 or b[2] == -1:
                    return "form"
                delta = a[2] ^ b[2]
                if delta & (delta - 1):
                    return "form"
                out.append((k, delta))
                continue
            if a[3] != b[3]:
                d = b[3] - a[3]
                if d == 0 or d < 0 or (d & (d - 1)):
                    return "form"
                out.append((k, d))
                continue
            if a[4] != b[4]:
                return "form"
    return out


def solve(members, read, nbits, samples=8):
    """Work out the layout of one class from a set of its encodings.

    Each bit of the encoding is flipped in each sample, and the changes
    are collected.  A bit that always feeds the same operand with the same
    weight is a field bit; a bit that changes the mnemonic or the number of
    operands is a selector.
    """
    feeds = collections.defaultdict(collections.Counter)
    names = collections.defaultdict(collections.Counter)
    forms = collections.Counter()
    members = sorted(members)
    step = max(1, len(members) // samples)
    for base in members[::step][:samples]:
        before = read(base)
        if before is None:
            continue
        for bit in range(nbits):
            after = read(base ^ (1 << bit))
            ch = change(before, after)
            if ch is None:
                continue
            if ch == "form":
                forms[bit] += 1
                if after is not None:
                    names[bit][after[0]] += 1
                continue
            for k, w in ch:
                feeds[bit][(k, w)] += 1
    return feeds, names, forms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", type=int, default=16)
    ap.add_argument("--out", default="/tmp/occ-thumb-table/l16.json")
    args = ap.parse_args()

    if args.space == 16:
        read = read16
        nbits = 16
        classes = collections.defaultdict(list)
        for hw in range(0x10000):
            r = read(hw)
            if r is not None:
                classes[shape_of(r)].append(hw)
    else:
        raise SystemExit("only the 16-bit space is enumerated in full")

    out = {}
    for shape, members in sorted(classes.items(), key=lambda kv: -len(kv[1])):
        feeds, names, forms = solve(members, read, nbits)
        entry = {"shape": list(shape), "count": len(members),
                 "first": members[0], "bits": {}}
        for bit in range(nbits):
            f = feeds.get(bit)
            n = names.get(bit)
            if f:
                entry["bits"][str(bit)] = {
                    "feeds": [[k, w, c] for (k, w), c in f.most_common()]}
            elif n:
                entry["bits"][str(bit)] = {"names": dict(n.most_common())}
            elif forms.get(bit):
                entry["bits"][str(bit)] = {"form": forms[bit]}
        out[" ".join(shape)] = entry

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}: {len(out)} classes")


if __name__ == "__main__":
    main()
