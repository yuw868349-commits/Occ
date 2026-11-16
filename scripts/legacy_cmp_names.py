#!/usr/bin/env python3
"""The names the legacy SSE comparison instructions take.

The condition is part of the name: `cmpsd` and `cmpnlesd` are different
instructions. Both disassemblers are asked, and only the answers they
agree on are recorded.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llvm_disasm as llvm  # noqa: E402
import truth_probe as tp  # noqa: E402

# The legacy encodings: 0f c2 is the packed comparison, 0f c6 the shuffle.
# The mandatory prefix picks the precision: 66 = double, f3 = single.
PREFIXES = {0: None, 1: b"\x66", 2: b"\xf3", 3: b"\xf2"}


def main():
    blobs = []
    tags = []
    for op in (0xc2,):
        for pp in range(4):
            for cond in range(16):
                prefix = PREFIXES[pp]
                blob = (prefix or b"") + bytes([0x0f, op, 0xc1, cond])
                blobs.append(blob)
                tags.append((op, pp, cond))
    pairs = tp.gnu_disassemble_len(blobs)
    ll = llvm.disassemble(blobs)

    for tag, blob, (gtext, glen), (ltext, llen) in zip(tags, blobs, pairs, ll):
        gname = tp.mnemonic_of(gtext)
        lname = tp.mnemonic_of(ltext)
        mark = "  " if (gname == lname and glen == llen) else "!!"
        print(f"{mark} op={tag[0]:02x} pp={tag[1]} cond={tag[2]:2d} "
              f"gnu[{glen}]={gtext[:44]:<44} llvm[{llen}]={lname}")


if __name__ == "__main__":
    main()
