#!/usr/bin/env python3
"""Build the comparison tables from a real disassembler.

Some instructions are named after the condition they test: `vcmpeqps` and
`vcmpneqps` are the same opcode with a different immediate byte, and a
listing that calls them both `vcmpps` has lost the only thing that tells
them apart. There are eight conditions and sixteen families, and the
machine has one spelling for each pair.

This script finds the families by building the encodings and watching the
name change with the immediate byte, then reads out the eight names of
each family.

Usage: compare_tables.py [--rust]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llvm_disasm as llvm

# The number of conditions the immediate byte selects.
CONDITIONS = 8


def evex(map_no, opcode, pp=0, w=0, ll=2, vvvv=0, mask=0, z=0, bcast=0,
         modrm=0xc1):
    """An EVEX encoding of `opcode` in `map_no`."""
    p0 = 0xf0 | (map_no & 0x07)
    p1 = (w << 7) | ((~vvvv & 0x0f) << 3) | 0x04 | (pp & 3)
    p2 = (z << 7) | (ll << 5) | (bcast << 4) | 0x08 | (mask & 0x07)
    return bytes([0x62, p0, p1, p2, opcode, modrm])


def candidates():
    """Every combination of fields an EVEX comparison could use."""
    for map_no in (1, 2, 3, 5, 6):
        for opcode in range(256):
            for pp in range(4):
                for w in (0, 1):
                    yield (map_no, opcode, pp, w)


def find_families():
    """The families whose name changes with the immediate byte.

    A family is one where the instruction with one immediate byte and the
    instruction with the next are both the same length and both decode:
    only then is the immediate byte part of the instruction.
    """
    jobs = list(candidates())
    blobs = []
    for map_no, opcode, pp, w in jobs:
        for imm in (0, 1):
            blobs.append(evex(map_no, opcode, pp=pp, w=w) + bytes([imm]))
    decoded = llvm.disassemble(blobs)

    families = []
    for i, (map_no, opcode, pp, w) in enumerate(jobs):
        text0, len0 = decoded[2 * i]
        text1, len1 = decoded[2 * i + 1]
        if not len0 or len0 != len1:
            continue
        if not text0 or not text1:
            continue
        name0 = text0.split("\t")[0]
        name1 = text1.split("\t")[0]
        if name0 == name1 or name0 in llvm.UNKNOWN or name1 in llvm.UNKNOWN:
            continue
        families.append((map_no, opcode, pp, w))
    return families


def names_of(map_no, opcode, pp, w):
    """The eight names of one family."""
    blobs = [evex(map_no, opcode, pp=pp, w=w) + bytes([imm])
             for imm in range(CONDITIONS)]
    out = []
    for text, _length in llvm.disassemble(blobs):
        if not text:
            return None
        name = text.split("\t")[0]
        if name in llvm.UNKNOWN:
            return None
        out.append(name)
    return out


def collect():
    """Every family and its names."""
    out = []
    for map_no, opcode, pp, w in find_families():
        names = names_of(map_no, opcode, pp, w)
        if names is None:
            continue
        out.append((map_no, opcode, pp, w, names))
    return out


def emit_rust(families):
    """The table as Rust source."""
    print("/// The comparison instructions, one row per family.")
    print("///")
    print("/// The name of each instruction is the condition it tests, so the")
    print("/// immediate byte selects the name. Every family was read out of a")
    print("/// disassembler; the script that does this is")
    print("/// `scripts/compare_tables.py`.")
    print("pub const CMP_OPS: &[CmpOp] = &[")
    for map_no, opcode, pp, w, names in families:
        listed = ", ".join('"' + n + '"' for n in names)
        print(f"    cmp_op(0x{opcode:02x}, {map_no}, {pp}, "
              f"{'true' if w else 'false'}, [{listed}]),")
    print("];")


if __name__ == "__main__":
    families = collect()
    if "--rust" in sys.argv:
        emit_rust(families)
    else:
        for map_no, opcode, pp, w, names in families:
            print(f"  map={map_no} op=0x{opcode:02x} pp={pp} w={w}")
            for i, n in enumerate(names):
                print(f"      {i}: {n}")
