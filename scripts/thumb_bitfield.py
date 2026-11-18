#!/usr/bin/env python3
"""Fit the rule that gives the bit-field operations their two numbers.

The insert and the clear of a field take two numbers: where the field
starts and how wide it is.  The encoding holds a five-bit width and a
five-bit end position, and the start is derived from the two.  Which
derivation it is, and which of the two numbers the encoding holds
directly, is what this script fits: it reads every combination of the
relevant fields and compares the reading against a candidate rule.

Usage:
    thumb_bitfield.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_ref as ref

SCRATCH = "/tmp/occ-thumb-bitfield"

# The class: the insert of a field.  The clear is the same class with
# the source register field set to the program counter.
CLASS = 0xF360


def read(text):
    """The two numbers in a reading."""
    if not text:
        return None
    m = re.search(r"#(-?0x[0-9a-f]+|-?\d+)\s*,\s*#(0x[0-9a-f]+|\d+)", text)
    if not m:
        return None
    def num(s):
        body = s[1:] if s.startswith("-") else s
        return int(body, 16) if body.startswith("0x") else int(body, 10)
    return num(m.group(1)), num(m.group(2))


def candidate(imm3, imm2, imm5):
    """The rule this project currently believes.

    The encoding holds a five-bit width and a five-bit end position, and
    the start of the field is derived from the two.
    """
    msb = (imm3 << 2) | imm2
    width = imm5 + 1
    lsb = msb + 1 - width
    if lsb < 0 or msb > 31:
        return None
    return lsb, width


def candidate_b(imm3, imm2, imm5):
    """The other rule: the five low bits are the start of the field."""
    lsb = imm5
    width = ((imm3 << 2) | imm2) + 1 - lsb
    if width < 1 or lsb > 31:
        return None
    return lsb, width


def candidate_c(imm3, imm2, imm5):
    """The rule the measurement says.

    The five high bits are the end of the field and the five low bits
    say how wide it is, counted from that end.
    """
    msb = (imm3 << 2) | imm2
    if imm5 <= msb:
        return imm5, 1
    return msb, imm5 - msb + 1


def main():
    pairs = []
    for imm3 in range(8):
        for imm2 in range(4):
            for imm5 in range(32):
                hw2 = (imm3 << 12) | (imm2 << 6) | imm5
                pairs.append((hw1 := CLASS, hw2, imm3, imm2, imm5))
    words = [(hw2 << 16) | hw1 for hw1, hw2, _, _, _ in pairs]
    table = ref.objdump_isolated_words32(words, SCRATCH)

    bad = 0
    shown = 0
    rules = {"a": candidate, "b": candidate_b, "c": candidate_c}
    scores = {k: 0 for k in rules}
    for i, (hw1, hw2, imm3, imm2, imm5) in enumerate(pairs):
        addr = ref.iso32_addr(i)
        cs = ref.capstone_read((hw2 << 16) | hw1, addr, 4)
        od = table.get(addr)
        got = read(cs) or read(od)
        for key, rule in rules.items():
            want = rule(imm3, imm2, imm5)
            if (got is None) != (want is None):
                continue
            if got is not None and got != want:
                continue
            scores[key] += 1
        want = candidate_c(imm3, imm2, imm5)
        if got is None:
            if want is not None:
                bad += 1
                if shown < 30:
                    shown += 1
                    print(f"  0x{hw2:04x} imm3={imm3} imm2={imm2} imm5={imm5:2d} "
                          f"reference=None rule={want}")
            continue
        if want is None:
            bad += 1
            if shown < 30:
                shown += 1
                print(f"  0x{hw2:04x} imm3={imm3} imm2={imm2} imm5={imm5:2d} "
                      f"reference={got} rule=None")
            continue
        if got != want:
            bad += 1
            if shown < 30:
                shown += 1
                print(f"  0x{hw2:04x} imm3={imm3} imm2={imm2} imm5={imm5:2d} "
                      f"reference={got} rule={want}")
    print(f"{len(pairs)} combinations, {bad} disagree")
    print("  each rule agrees with the reference on:")
    for key, score in scores.items():
        print(f"    rule {key}: {score}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
