#!/usr/bin/env python3
"""Measure the layout of every Thumb class, completely.

A decoder is written by *reading the architecture manual*, and a manual
describes an encoding in prose: "the destination is in bits 2 to 0".  A
prose description is easy to get subtly wrong, and a wrong field shows up
much later as a wrong decompilation — long after the mistake was made.

This script does not read the manual.  It takes the encoding space apart
and watches what moves.  For each class of instruction it flips one bit
at a time and compares the two readings.  The operand that changed is the
operand that bit feeds, and the size of the change is the bit's weight
inside that operand.

What is measured is the *structure*, not the text: which operand is a
register, which is an immediate, which is a memory reference, and — for a
memory reference — which is the base register, which the index register
and which the displacement.  A branch's destination is an address, so the
displacement is recovered from it by subtracting the program counter,
which is the same number for every reading of the same encoding.

The result is a table, per class, of

* the mnemonic,
* for each operand, the bits that feed it and their weights,
* and for the bits that choose between forms, the mnemonics they choose
  between.

That table is what the decoder is written from.  Nothing in the decoder's
field layouts is a matter of memory.

Usage:
    thumb_measure.py --space 16 --out /tmp/occ-thumb-table/m16.json
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

# The address every halfword is read at.  It is not the address the
# halfword would really live at; it is a fixed one, so that a branch's
# destination can be turned back into a displacement.
ADDR = 0x1000


class Operand:
    """One operand, decomposed into the parts a field can feed."""

    __slots__ = ("kind", "reg", "base", "index", "disp", "lshift")

    def __init__(self, kind, reg=-1, base=-1, index=-1, disp=0, lshift=0):
        self.kind = kind
        self.reg = reg
        self.base = base
        self.index = index
        self.disp = disp
        self.lshift = lshift

    def key(self):
        if self.kind == "r":
            return "r"
        if self.kind == "i":
            return "i"
        return f"m:{self.base_name()}:{'i' if self.index >= 0 else '-'}:{self.lshift}"

    def base_name(self):
        if self.base in (13, 14, 15):
            return {13: "sp", 14: "lr", 15: "pc"}[self.base]
        return "rN"


def read16(hw):
    """Read one halfword with capstone, structurally."""
    md = cs.Cs(cs.CS_ARCH_ARM, cs.CS_MODE_THUMB)
    md.detail = True
    code = hw.to_bytes(2, "little")
    for i in md.disasm(code, ADDR, count=1):
        if i.size != 2:
            return None
        ops = []
        for o in i.operands:
            if o.type == cs.arm.ARM_OP_REG:
                ops.append(Operand("r", reg=REGID.get(i.reg_name(o.reg), -1)))
            elif o.type == cs.arm.ARM_OP_IMM:
                ops.append(Operand("i", disp=o.imm))
            elif o.type == cs.arm.ARM_OP_MEM:
                index = o.mem.index
                ops.append(Operand(
                    "m",
                    base=REGID.get(i.reg_name(o.mem.base), -1),
                    index=-1 if index == 0 else REGID.get(i.reg_name(index), -2),
                    disp=o.mem.disp,
                    lshift=o.mem.lshift))
            else:
                return None
        return (i.mnemonic, ops)
    return None


def shape_of(reading):
    """The shape of a reading: the mnemonic and the kinds of the operands."""
    mn, ops = reading
    return tuple([mn] + [op.key() for op in ops])


def delta_of(a, b):
    """The change in one operand, as (role, weight).

    The role says which part of the operand moved — the register number,
    the displacement, the base register, the index register — and the
    weight is the size of the move.  For a register number the weight is
    the bit's value inside the number; for a displacement it is the
    bit's value inside the number.

    The string "form" means the operand changed in a way that is not a
    field change: it became a different kind of thing.  Those bits
    choose between forms, and they are recorded separately.
    """
    if a.kind != b.kind:
        return "form"
    if a.kind == "r":
        if a.reg == b.reg:
            return None
        if a.reg < 0 or b.reg < 0:
            return "form"
        d = a.reg ^ b.reg
        return ("reg", d) if (d & (d - 1)) == 0 else "form"
    if a.kind == "i":
        d = b.disp - a.disp
        if d == 0:
            return None
        if d < 0 or (d & (d - 1)):
            return "form"
        return ("imm", d)
    # A memory reference.
    if a.base != b.base:
        if a.base < 0 or b.base < 0:
            return "form"
        d = a.base ^ b.base
        return ("base", d) if (d & (d - 1)) == 0 else "form"
    if a.index != b.index:
        if a.index < 0 or b.index < 0:
            return "form"
        d = a.index ^ b.index
        return ("index", d) if (d & (d - 1)) == 0 else "form"
    if a.disp != b.disp:
        d = b.disp - a.disp
        if d == 0 or d < 0 or (d & (d - 1)):
            return "form"
        return ("disp", d)
    if a.lshift != b.lshift:
        return "form"
    return None


def compare(before, after):
    """Describe what one flipped bit did to a whole reading."""
    if before is None or after is None:
        return None
    if before[0] != after[0] or len(before[1]) != len(after[1]):
        return "form"
    out = []
    for k, (a, b) in enumerate(zip(before[1], after[1])):
        d = delta_of(a, b)
        if d is None:
            continue
        if d == "form":
            return "form"
        out.append((k,) + d)
    return out


def solve(members, read, nbits, samples=10):
    """Work out the layout of one class from a set of its encodings."""
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
            ch = compare(before, after)
            if ch is None:
                continue
            if ch == "form":
                forms[bit] += 1
                if after is not None:
                    names[bit][after[0]] += 1
                continue
            for k, role, w in ch:
                feeds[bit][(k, role, w)] += 1
    return feeds, names, forms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", type=int, default=16)
    ap.add_argument("--out", default="/tmp/occ-thumb-table/m16.json")
    args = ap.parse_args()

    if args.space != 16:
        raise SystemExit("only the 16-bit space is enumerated in full")

    read = read16
    nbits = 16
    classes = collections.defaultdict(list)
    for hw in range(0x10000):
        r = read(hw)
        if r is not None:
            classes[shape_of(r)].append(hw)

    out = {}
    for shape, members in classes.items():
        feeds, names, forms = solve(members, read, nbits)
        entry = {"shape": list(shape), "count": len(members),
                 "first": members[0], "bits": {}}
        for bit in range(nbits):
            f = feeds.get(bit)
            n = names.get(bit)
            if f:
                entry["bits"][str(bit)] = {
                    "feeds": [[k, role, w, c] for (k, role, w), c in f.most_common()]}
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
