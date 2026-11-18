#!/usr/bin/env python3
"""Print both references' readings of one region, halfword by halfword.

Some regions of the space hold a family of instructions whose members are
picked out by a field, and the field's meaning is not the same for every
member.  Both references are asked, side by side, so that a disagreement
between them is visible rather than hidden.

Usage:
    thumb_region.py 0xb640 0xb6ff
"""
import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capstone as cs

OBJCOPY = "/usr/lib/llvm-23/bin/llvm-objcopy"
OBJDUMP = "/usr/lib/llvm-23/bin/llvm-objdump"
TRIPLE = "thumbv7-linux-gnueabi"


def objdump_table(words, scratch):
    stream = []
    for w in words:
        stream.append(w)
        stream.append(0xBF00)
    blob = b"".join(w.to_bytes(2, "little") for w in stream)
    subprocess.run(["rm", "-rf", scratch], check=False)
    subprocess.run(["mkdir", "-p", scratch], check=True)
    raw = f"{scratch}/r.bin"
    with open(raw, "wb") as f:
        f.write(blob)
    obj = f"{scratch}/a.o"
    subprocess.run(
        [OBJCOPY, "--input-target=binary", "--output-target=elf32-littlearm",
         "--rename-section", ".data=.text,code", raw, obj],
        check=True, capture_output=True)
    res = subprocess.run(
        [OBJDUMP, "-d", f"--triple={TRIPLE}", obj],
        check=True, capture_output=True, text=True).stdout
    table = {}
    for line in res.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\s+[0-9a-f ]+\s*(.*)$", line)
        if m:
            table[int(m.group(1), 16)] = m.group(2).strip()
    return table


def capstone_read(hw, addr):
    md = cs.Cs(cs.CS_ARCH_ARM, cs.CS_MODE_THUMB)
    code = hw.to_bytes(2, "little")
    for i in md.disasm(code, addr, count=1):
        if i.size != 2:
            return None
        return f"{i.mnemonic} {i.op_str}".strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lo", type=lambda s: int(s, 16))
    ap.add_argument("hi", type=lambda s: int(s, 16))
    ap.add_argument("--scratch", default="/tmp/occ-thumb-region")
    args = ap.parse_args()

    words = list(range(args.lo, args.hi + 1))
    table = objdump_table(words, args.scratch)
    for i, hw in enumerate(words):
        addr = i * 4
        c = capstone_read(hw, addr)
        o = table.get(addr)
        flag = "" if (c == o or (c is None and o is None)
                      or (o is not None and o.startswith("<unknown>")
                          and c is None)) else "  <-- DIFFER"
        print(f"0x{hw:04x}  capstone: {str(c):<22s} objdump: {o}{flag}")


if __name__ == "__main__":
    main()
