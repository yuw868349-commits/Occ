#!/usr/bin/env python3
"""Fit the rule that turns a twelve-bit Thumb immediate into a value.

The modified-immediate class holds twelve bits: one in the first
halfword, three in the second, and eight more in the second.  What those
twelve bits are worth is a rule, and the rule is what this script
checks: it reads every one of the four thousand and ninety-six
combinations through both references and compares the reading against a
candidate rule.

The candidate is given on the command line, so a rule can be changed and
re-tested without editing anything.

Usage:
    thumb_immfit.py
    thumb_immfit.py --rule 'candidate'
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_ref as ref

SCRATCH = "/tmp/occ-thumb-immfit"


def candidate(imm12):
    """The rule as this project currently believes it."""
    if imm12 & 0xC00 == 0:
        byte = imm12 & 0xFF
        return [
            byte,
            byte * 0x0001_0001,
            byte * 0x0100_0100,
            byte * 0x0101_0101,
        ][(imm12 >> 8) & 3]
    unrotated = 0x80 | (imm12 & 0x7F)
    rot = (imm12 >> 7) & 0x1F
    return ((unrotated >> rot) | (unrotated << (32 - rot))) & 0xFFFFFFFF


def parse_value(text):
    """The immediate in a reading, as a number.

    Both references print the value after a `#`; the number is in hex with
    a `0x` in front of it, but a small value may be printed in decimal.
    """
    if "#" not in text:
        return None
    body = text.rsplit("#", 1)[1].strip()
    body = body.split()[0] if body else ""
    if body.startswith("0x") or body.startswith("0X"):
        return int(body, 16)
    if body.startswith("-"):
        return -int(body[1:], 10)
    return int(body, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    rows = []
    for i in (0, 1):
        for imm3 in range(8):
            for imm8 in range(256):
                hw1 = 0xF019 & ~(1 << 10) | (i << 10)
                hw2 = (imm3 << 12) | imm8
                rows.append((hw1, hw2, (i << 11) | (imm3 << 8) | imm8))

    words = [(hw2 << 16) | hw1 for hw1, hw2, _ in rows]
    table = ref.objdump_isolated_words32(words, SCRATCH)

    bad = 0
    shown = 0
    for i, (hw1, hw2, imm12) in enumerate(rows):
        addr = ref.iso32_addr(i)
        want = candidate(imm12)
        readings = [
            ("capstone", ref.capstone_read((hw2 << 16) | hw1, addr, 4)),
            ("objdump", table.get(addr)),
        ]
        for who, text in readings:
            if text is None:
                continue
            value = parse_value(text)
            if value is None:
                continue
            if value != want:
                bad += 1
                if shown < args.limit:
                    shown += 1
                    print(f"  {who:8s} 0x{hw1:04x}/0x{hw2:04x} imm12=0x{imm12:03x} "
                          f"reference=0x{value:08x} rule=0x{want:08x}")
    total = len(rows)
    print(f"{total} combinations, {bad} disagree")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
