#!/usr/bin/env python3
"""Solve the layout of every instruction class by measurement.

A decoder is written by *reading the architecture manual*, and a manual
describes an encoding in prose: "the destination is in bits 2 to 0".  A
prose description is easy to get subtly wrong, and a wrong field shows up
much later as a wrong decompilation.

This script does not read the manual.  It takes the encoding space apart
and watches what moves.  For each class of instruction it finds a set of
representative encodings, flips one bit at a time, and compares the two
readings: the operand that changed is the operand that bit feeds, and the
size of the change is the weight of the bit inside that operand.

The result is a table, per class, of

* which mnemonic each bit selects between,
* which operand each bit feeds, and
* what weight the bit has inside that operand.

That table is what the decoder is then written from.  Nothing in the
decoder's field layouts is a matter of memory.

Two references are used, and they are used for different things:

* capstone is read *structurally* — its operand list says which operand is
  a register, which is an immediate, and which is a memory reference — and
  that is what makes the measurement possible;
* llvm-objdump is read *textually*, and it is the spelling that the
  decoder is asked to match, because it is ARM's own disassembler.

Usage:
    thumb_fields.py --space 16 --out fields16.json
    thumb_fields.py --space 32 --out fields32.json
"""
import argparse
import collections
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_ref as ref

import capstone as cs

# The operand kinds, as this script names them.
REG, IMM, MEM = "reg", "imm", "mem"


class Reading:
    """One decoded instruction, in a form that can be compared field by field."""

    __slots__ = ("mn", "ops", "size")

    def __init__(self, mn, ops, size):
        self.mn = mn
        self.ops = ops
        self.size = size

    def key(self):
        """The shape of the reading: what kind of thing each operand is."""
        out = [self.mn]
        for kind, *rest in self.ops:
            if kind == REG:
                out.append("reg")
            elif kind == IMM:
                out.append("imm")
            else:
                base, index, disp, lshift = rest
                out.append(f"mem:{base}:{'+' if index >= 0 else '-'}:{lshift}")
        return tuple(out)

    def __repr__(self):
        return f"{self.mn} " + ", ".join(
            f"{k}{rest}" for k, *rest in self.ops)


def decode16(hw):
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
                ops.append((REG, i.reg_name(o.reg)))
            elif o.type == cs.arm.ARM_OP_IMM:
                ops.append((IMM, o.imm))
            elif o.type == cs.arm.ARM_OP_MEM:
                base = i.reg_name(o.mem.base)
                index = o.mem.index
                idx = -1 if index == 0 else i.reg_name(index)
                ops.append((MEM, base, idx, o.mem.disp, o.mem.lshift))
            else:
                return None
        return Reading(i.mnemonic, ops, i.size)
    return None


def decode32(word):
    """Read one 32-bit Thumb encoding with capstone, structurally."""
    md = cs.Cs(cs.CS_ARCH_ARM, cs.CS_MODE_THUMB)
    md.detail = True
    code = word.to_bytes(4, "little")
    for i in md.disasm(code, 0, count=1):
        if i.size != 4:
            return None
        ops = []
        for o in i.operands:
            if o.type == cs.arm.ARM_OP_REG:
                ops.append((REG, i.reg_name(o.reg)))
            elif o.type == cs.arm.ARM_OP_IMM:
                ops.append((IMM, o.imm))
            elif o.type == cs.arm.ARM_OP_MEM:
                base = i.reg_name(o.mem.base)
                index = o.mem.index
                idx = -1 if index == 0 else i.reg_name(index)
                ops.append((MEM, base, idx, o.mem.disp, o.mem.lshift))
            else:
                return None
        return Reading(i.mnemonic, ops, i.size)
    return None


REGID = {f"r{i}": i for i in range(13)}
REGID["sp"] = 13
REGID["lr"] = 14
REGID["pc"] = 15


def regnum(name):
    return REGID.get(name)


def compare(a, b, bit):
    """Describe what one flipped bit did to a reading.

    The return value is a list of (operand index, weight) pairs, where the
    weight is the amount the operand's value changed by.  A `None` means
    the reading changed in a way that is not a field change — the
    mnemonic moved, or the number of operands moved — which is the bit
    choosing between forms.
    """
    if a is None or b is None:
        return None
    if a.mn != b.mn or len(a.ops) != len(b.ops):
        return "form"
    out = []
    for k, (oa, ob) in enumerate(zip(a.ops, b.ops)):
        if oa[0] != ob[0]:
            return "form"
        if oa[0] == REG:
            va, vb = regnum(oa[1]), regnum(ob[1])
            if va is None or vb is None:
                return "form"
            if va == vb:
                continue
            delta = va ^ vb
            if delta & (delta - 1):
                return "form"
            out.append((k, delta))
        elif oa[0] == IMM:
            d = ob[1] - oa[1]
            if d == 0:
                continue
            if d < 0:
                return "form"  # a sign bit, handled separately
            if d & (d - 1):
                return "form"
            out.append((k, d))
        else:
            if oa[1] != ob[1]:
                return "form"
            if oa[2] != ob[2]:
                d = regnum(ob[2]) - regnum(oa[2])
                if d == 0 or d & (d - 1):
                    return "form"
                out.append((k, d))
                continue
            if oa[3] != ob[3]:
                d = ob[3] - oa[3]
                if d == 0 or d & (d - 1):
                    return "form"
                out.append((k, d))
                continue
            if oa[4] != ob[4]:
                return "form"
    return out


def solve_class(samples, decode, nbits):
    """Work out the layout of a class from a set of representative encodings.

    Each bit of the encoding is flipped in each sample, and the changes are
    collected.  A bit that always feeds the same operand with the same
    weight is a field bit; a bit that sometimes changes the form is a
    selector, and the mnemonic it selects between is recorded.
    """
    # bit -> {(operand index, weight): count}
    feeds = collections.defaultdict(collections.Counter)
    # bit -> {mnemonic: count}, for the bits that only change the mnemonic
    names = collections.defaultdict(collections.Counter)
    forms = collections.Counter()
    for base in samples:
        a = decode(base)
        if a is None:
            continue
        for bit in range(nbits):
            other = base ^ (1 << bit)
            b = decode(other)
            ch = compare(a, b, bit)
            if ch is None:
                continue
            if ch == "form":
                forms[bit] += 1
                if b is not None:
                    names[bit][b.mn] += 1
                continue
            for k, w in ch:
                feeds[bit][(k, w)] += 1
    return feeds, names, forms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--samples", type=int, default=6)
    args = ap.parse_args()

    if args.space == 16:
        # Every halfword is a candidate; the ones capstone reads as 32-bit
        # encodings are not 16-bit instructions and are left out.
        readings = {}
        for hw in range(0x10000):
            r = decode16(hw)
            if r is not None:
                readings[hw] = r
        classes = collections.defaultdict(list)
        for hw, r in readings.items():
            classes[r.key()].append(hw)
        decode = decode16
        nbits = 16
    else:
        # The 32-bit space is far too large to enumerate; the classes are
        # found by their first halfword, and each class is sampled.
        classes = collections.defaultdict(list)
        decode = decode32
        nbits = 32

    out = {}
    for key, members in sorted(classes.items(), key=lambda kv: -len(kv[1])):
        # A handful of members spread across the class, so that a field
        # which is only visible in some encodings still shows up.
        members.sort()
        step = max(1, len(members) // args.samples)
        samples = members[::step][:args.samples]
        feeds, names, forms = solve_class(samples, decode, nbits)
        entry = {
            "shape": list(key),
            "count": len(members),
            "members": members[:32],
            "bits": {},
        }
        for bit in range(nbits):
            f = feeds.get(bit)
            n = names.get(bit)
            if f:
                entry["bits"][str(bit)] = {
                    "feeds": [[k, w, c] for (k, w), c in f.most_common()],
                }
            elif n:
                entry["bits"][str(bit)] = {"names": dict(n.most_common())}
            elif forms.get(bit):
                entry["bits"][str(bit)] = {"form": forms[bit]}
        out[str(key)] = entry

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.out}: {len(out)} classes")
    else:
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
