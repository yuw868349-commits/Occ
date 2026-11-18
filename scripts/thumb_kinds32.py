#!/usr/bin/env python3
"""Count how many distinct instruction classes a region of the space holds.

The mnemonic of an encoding is read through both references, and the
different mnemonics are counted.  A region that holds a dozen mnemonics
is a dozen classes of instruction; a region that holds a hundred is a
hundred, and that is the measure of the work.

Usage:
    thumb_kinds32.py 0xec00 0xefff
    thumb_kinds32.py 0xfc00 0xffff --top 40
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_ref as ref

SCRATCH = "/tmp/occ-thumb-kinds32"

# The second halfwords each first halfword is read behind.
PROBES = (0x0000, 0xFFFF, 0x5555, 0xAAAA, 0x0F0F, 0xF0F0, 0x00FF, 0xFF00)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lo", type=lambda s: int(s, 0))
    ap.add_argument("hi", type=lambda s: int(s, 0))
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    hw1s = list(range(args.lo, args.hi + 1))
    pairs = [(h, w) for h in hw1s for w in PROBES]
    words = [(w << 16) | h for h, w in pairs]
    table = ref.objdump_isolated_words32(words, SCRATCH)

    kinds = collections.Counter()
    per_hw1 = collections.defaultdict(set)
    for i, (hw1, hw2) in enumerate(pairs):
        addr = ref.iso32_addr(i)
        cs = ref.capstone_read((hw2 << 16) | hw1, addr, 4)
        od = table.get(addr)
        for text in (cs, od):
            if not text or text.startswith("<unknown>"):
                continue
            head = text.split()[0].lower()
            kinds[head] += 1
            per_hw1[hw1].add(head)

    print(f"{len(hw1s)} first halfwords, {len(pairs)} encodings")
    print(f"{len(kinds)} distinct mnemonics")
    print("the most common:")
    for name, count in kinds.most_common(args.top):
        print(f"    {name:12s} {count}")
    multi = sum(1 for v in per_hw1.values() if len(v) > 1)
    print(f"{multi} first halfwords hold more than one mnemonic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
