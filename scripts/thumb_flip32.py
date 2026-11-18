#!/usr/bin/env python3
"""Flip one bit of a 32-bit Thumb encoding and watch what moves.

This is the measurement, in a form that can be read.  The first halfword
of a 32-bit Thumb instruction says which class the instruction belongs
to, and the bits of the two halfwords say what the instruction does.
Flipping a bit and reading the encoding again says which field that bit
belongs to.

Usage:
    thumb_flip32.py 0xea00 0x0000 [--half 2] [--base capstone|objdump]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_ref as ref

SCRATCH = "/tmp/occ-thumb-flip32"


def readings(pairs, base):
    words = [(hw2 << 16) | hw1 for hw1, hw2 in pairs]
    if base == "capstone":
        out = []
        for i, (hw1, hw2) in enumerate(pairs):
            out.append(ref.capstone_read((hw2 << 16) | hw1, ref.iso32_addr(i), 4))
        return out
    table = ref.objdump_isolated_words32(words, SCRATCH)
    return [table.get(ref.iso32_addr(i)) for i in range(len(pairs))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hw1", type=lambda s: int(s, 0))
    ap.add_argument("hw2", type=lambda s: int(s, 0))
    ap.add_argument("--half", type=int, default=2, choices=(1, 2))
    ap.add_argument("--base", default="capstone", choices=("capstone", "objdump"))
    args = ap.parse_args()

    pairs = [(args.hw1, args.hw2)]
    for bit in range(16):
        if args.half == 2:
            pairs.append((args.hw1, args.hw2 ^ (1 << bit)))
        else:
            pairs.append((args.hw1 ^ (1 << bit), args.hw2))
    got = readings(pairs, args.base)

    print(f"base: {got[0]!r}")
    for i in range(1, len(pairs)):
        bit = i - 1
        print(f"  bit{bit:2d} flip -> {got[i]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
