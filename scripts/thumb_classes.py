#!/usr/bin/env python3
"""Print the members of the measured classes, as ranges.

The measurement table says what a class looks like — the mnemonic and the
kinds of its operands — and how many encodings belong to it.  What it does
not say is *which* encodings those are, and a decoder is written from the
ranges, not from a count.

So the space is read again, this time grouped by class, and each class is
printed as the ranges of encodings it covers.  A class that is one
contiguous range is the usual case; a class that is two ranges is a class
with a hole in it, and the hole is where the undefined encodings live.

Usage:
    thumb_classes.py [substring ...]
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capstone as cs

REGID = {f"r{i}": i for i in range(13)}
REGID["sp"] = 13
REGID["lr"] = 14
REGID["pc"] = 15
ADDR = 0x1000


class Operand:
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


def ranges(members):
    """The members as a list of inclusive ranges."""
    out = []
    for m in sorted(members):
        if out and m == out[-1][1] + 1:
            out[-1][1] = m
        else:
            out.append([m, m])
    return out


def fmt(r):
    if r[0] == r[1]:
        return f"0x{r[0]:04x}"
    return f"0x{r[0]:04x}-0x{r[1]:04x}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="*", default=None)
    args = ap.parse_args()

    classes = collections.defaultdict(list)
    for hw in range(0x10000):
        r = read16(hw)
        if r is None:
            continue
        mn, ops = r
        classes[(mn, tuple(op.key() for op in ops))].append(hw)

    print(f"{len(classes)} classes")
    for (mn, keys), members in sorted(classes.items(),
                                      key=lambda kv: min(kv[1])):
        label = " ".join([mn] + list(keys))
        if args.patterns and not any(p in label for p in args.patterns):
            continue
        rs = ranges(members)
        print(f"{label:24s} n={len(members):5d}  "
              + " ".join(fmt(r) for r in rs))


if __name__ == "__main__":
    main()
