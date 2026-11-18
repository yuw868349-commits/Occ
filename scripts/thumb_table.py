#!/usr/bin/env python3
"""Read the whole 16-bit Thumb space, structurally, and cache it.

This is the raw material for the decoder's layout tables.  Every one of the
65536 halfwords is read with capstone in *detail* mode, so the answer is a
list of typed operands — a register, an immediate, a memory reference —
rather than a line of text.  Text cannot be diffed field by field; a typed
operand can.

llvm-objdump is read at the same time, because it is the spelling the
decoder is asked to match, and because it is ARM's own disassembler: where
it and capstone disagree, the disagreement is worth knowing about.

The cache is written once and read many times, because the layout solver
asks questions of this table hundreds of times.

Usage: thumb_table.py [--out PATH]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_ref as ref

import capstone as cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/occ-thumb-table/t16.json")
    ap.add_argument("--scratch", default="/tmp/occ-thumb-table/od")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # The whole space in one objdump run, at a known address per halfword.
    # Each halfword is read on its own, so that the state an `it` block
    # leaves behind cannot reach the next one.
    words = list(range(0x10000))
    table = ref.objdump_isolated_space(words, args.scratch)

    md = cs.Cs(cs.CS_ARCH_ARM, cs.CS_MODE_THUMB)
    md.detail = True

    rows = []
    for hw in range(0x10000):
        code = hw.to_bytes(2, "little")
        reading = None
        for i in md.disasm(code, 0, count=1):
            if i.size != 2:
                break
            ops = []
            for o in i.operands:
                if o.type == cs.arm.ARM_OP_REG:
                    ops.append(["r", i.reg_name(o.reg)])
                elif o.type == cs.arm.ARM_OP_IMM:
                    ops.append(["i", o.imm])
                elif o.type == cs.arm.ARM_OP_MEM:
                    index = o.mem.index
                    ops.append(["m", i.reg_name(o.mem.base),
                                -1 if index == 0 else i.reg_name(index),
                                o.mem.disp, o.mem.lshift])
                else:
                    ops.append(["?", 0])
            reading = [i.mnemonic, ops]
            break
        rows.append({
            "hw": hw,
            "cs": reading,
            "cs_raw": capstone_text(hw),
            "od_raw": table.get(ref.iso_addr(hw)),
        })

    with open(args.out, "w") as f:
        json.dump(rows, f)
    print(f"wrote {args.out}: {len(rows)} halfwords")


def capstone_text(hw):
    md = cs.Cs(cs.CS_ARCH_ARM, cs.CS_MODE_THUMB)
    md.detail = False
    code = hw.to_bytes(2, "little")
    for i in md.disasm(code, 0, count=1):
        if i.size == 2:
            return f"{i.mnemonic} {i.op_str}".strip()
    return None


if __name__ == "__main__":
    main()
