#!/usr/bin/env python3
"""Measure the layout of the 32-bit Thumb classes.

The 16-bit space could be read in full, so the layout of every one of its
65536 encodings was read out of the references and nothing in the decoder
is a matter of memory.  The 32-bit space is four times the size of the
address space of a halfword — 2**32 encodings — so it cannot be read that
way, and this script measures it instead.

The method is the same one the 16-bit space was measured with: take the
encoding space apart and watch what moves.  For each class of instruction
it flips one bit at a time and compares the two readings.  The operand
that changed is the operand that bit feeds, and the size of the change is
the bit's weight inside that operand.

What is measured is the *structure*, not the text: which operand is a
register, which is an immediate, which is a memory reference, and — for a
memory reference — which is the base register, which the index register
and which the displacement.  A branch's destination is an address, so the
displacement is recovered from it by subtracting the program counter.

The classes are found by reading the first halfword on its own: the
first halfword of a 32-bit instruction says which class the instruction
belongs to, and the low bits of it are the operands.  Every one of the
7680 first halfwords that can open a 32-bit instruction is read with
several second halfwords behind it, and the ones that read the same way
are put in one class.

Usage:
    thumb_measure32.py --out /tmp/occ-thumb-table/m32.json
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import capstone as cs

from thumb_measure import Operand, compare, delta_of, shape_of

# The address every encoding is read at.  It is not the address the
# encoding would really live at; it is a fixed one, so that a branch's
# destination can be turned back into a displacement.
ADDR = 0x1000

# The second halfwords used to look behind a first halfword.  A class is
# the set of first halfwords that read the same way behind all of them.
PROBES = (0x0000, 0xFFFF, 0x1234, 0x8888, 0x00F0)

REGID = {f"r{i}": i for i in range(13)}
REGID["sp"] = 13
REGID["lr"] = 14
REGID["pc"] = 15


def read32(hw1, hw2):
    """Read one 32-bit encoding with capstone, structurally."""
    md = cs.Cs(cs.CS_ARCH_ARM, cs.CS_MODE_THUMB)
    md.detail = True
    code = ((hw2 << 16) | hw1).to_bytes(4, "little")
    for i in md.disasm(code, ADDR, count=1):
        if i.size != 4:
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


def solve(pairs, samples=12):
    """Work out the layout of one class from a set of its encodings."""
    feeds = collections.defaultdict(collections.Counter)
    names = collections.defaultdict(collections.Counter)
    forms = collections.Counter()
    pairs = sorted(pairs)
    step = max(1, len(pairs) // samples)
    for hw1, hw2 in pairs[::step][:samples]:
        before = read32(hw1, hw2)
        if before is None:
            continue
        for bit in range(32):
            if bit < 16:
                after = read32(hw1 ^ (1 << bit), hw2)
                where = bit
            else:
                after = read32(hw1, hw2 ^ (1 << (bit - 16)))
                where = bit
            ch = compare(before, after)
            if ch is None:
                continue
            if ch == "form":
                forms[where] += 1
                if after is not None:
                    names[where][after[0]] += 1
                continue
            for k, role, w in ch:
                feeds[where][(k, role, w)] += 1
    return feeds, names, forms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/occ-thumb-table/m32.json")
    ap.add_argument("--min-class", type=int, default=1,
                    help="the smallest class worth reporting on its own")
    args = ap.parse_args()

    # The classes: the first halfwords that read the same way.
    classes = collections.defaultdict(list)
    for hw1 in range(0xE800, 0x10000):
        seen = set()
        for hw2 in PROBES:
            r = read32(hw1, hw2)
            if r is None:
                continue
            seen.add(shape_of(r))
        for shape in seen:
            classes[shape].append(hw1)

    out = {}
    for shape, hw1s in sorted(classes.items(), key=lambda kv: -len(kv[1])):
        if len(hw1s) < args.min_class:
            continue
        # The encodings of the class: the first halfwords that read this
        # way, each behind the second halfwords that read this way too.
        pairs = []
        for hw1 in hw1s:
            for hw2 in PROBES:
                r = read32(hw1, hw2)
                if r is not None and shape_of(r) == shape:
                    pairs.append((hw1, hw2))
        feeds, names, forms = solve(pairs)
        entry = {"shape": list(shape), "count": len(hw1s),
                 "first": hw1s[0], "bits": {}}
        for bit in range(32):
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
    for shape, entry in sorted(out.items(), key=lambda kv: -kv[1]["count"])[:30]:
        print(f"  {entry['count']:5d}  {shape}")


if __name__ == "__main__":
    main()
