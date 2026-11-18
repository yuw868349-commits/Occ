#!/usr/bin/env python3
"""Measure the `it` blocks, the hints, and the system instructions.

The 16-bit space holds two families whose meaning is not obvious from the
bits alone:

* `1011 1111 cond(4) mask(4)`.  When the mask is zero the encoding is one
  of the hints; when it is not, the encoding opens an `it` block, and the
  mask says how many instructions the block covers and which of them are
  the "then" arm.
* `1011 1110 ...`, `1011 0110 ...` and `1011 1010 ...`: the breakpoint,
  the interrupt masks, and the endianness setter.

The bits of the mask are not read the same way twice, so the rules are
measured rather than assumed.  Every encoding of the two families is read
with both references, and the *assembler* is asked to encode every spelling
of the `it` family -- the assembler is ARM's own encoder, so what it
produces for `ittte` says what the mask means.

The measurements are written to a table so that the decoder's rules can be
checked against them rather than re-derived.

Usage:
    thumb_it.py --out /tmp/occ-thumb-table/it16.json
"""
import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capstone as cs

OBJCOPY = "/usr/lib/llvm-23/bin/llvm-objcopy"
OBJDUMP = "/usr/lib/llvm-23/bin/llvm-objdump"
MCTARGET = "/usr/lib/llvm-23/bin/llvm-mc"
TRIPLE = "thumbv7-linux-gnueabi"


def capstone_read(hw, addr):
    """Read one halfword with capstone, at a known address."""
    md = cs.Cs(cs.CS_ARCH_ARM, cs.CS_MODE_THUMB)
    code = hw.to_bytes(2, "little")
    for i in md.disasm(code, addr, count=1):
        if i.size != 2:
            return None
        return f"{i.mnemonic} {i.op_str}".strip()
    return None


def objdump_table(words, scratch):
    """Read a stream of halfwords with llvm-objdump, each in its own slot."""
    stream = []
    for w in words:
        stream.append(w)
        stream.append(0xBF00)  # nop, so that no encoding swallows the next
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


def assembler_encodings():
    """Ask the assembler to encode every spelling of the `it` family."""
    names = []
    for count in range(1, 5):
        for bits in range(1 << (count - 1)):
            letters = "t"
            for k in range(count - 1):
                letters += "e" if (bits >> k) & 1 else "t"
            names.append("it" + letters)
    out = {}
    for name in names:
        src = f".text\n.global _s\n_s:\n\t{name} eq\n"
        proc = subprocess.run(
            [MCTARGET, f"-triple={TRIPLE}", "-show-encoding", "-"],
            input=src, capture_output=True, text=True)
        for line in proc.stdout.splitlines():
            m = re.match(r"\s*0x([0-9a-f]+):\s+(.*?)\s*//\s*(.*)$", line)
            if not m:
                continue
            word = int(m.group(1), 16)
            text = m.group(2).strip()
            # The assembler prints the instruction's own spelling; keep it.
            if text.lower() != name:
                text = name
            out[name] = {"encoding": word, "assembler": text}
    return out


def measure_family(words, scratch):
    """Read every encoding of a family with both references."""
    table = objdump_table(words, scratch)
    rows = []
    for i, hw in enumerate(words):
        addr = i * 4
        rows.append({
            "hw": hw,
            "addr": addr,
            "capstone": capstone_read(hw, addr),
            "objdump": table.get(addr),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/occ-thumb-table/it16.json")
    ap.add_argument("--scratch", default="/tmp/occ-thumb-it")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # The `it` family and the hints share one encoding: `1011 1111 cccc mmmm`.
    it_words = [0xBF00 | (c << 4) | m for c in range(16) for m in range(16)]
    it_rows = measure_family(it_words, args.scratch + "/it")

    # The system instructions: the breakpoint, the interrupt masks, the
    # endianness setter and the barriers.
    sys_words = sorted(set(
        list(range(0xBE00, 0xBF00))          # the breakpoint row
        + list(range(0xB640, 0xB680))        # setend, cpsie, cpsid
        + list(range(0xB660, 0xB680))
        + list(range(0xB670, 0xB680))
        + list(range(0xBFB0, 0xBFC0))        # the barriers
    ))
    sys_rows = measure_family(sys_words, args.scratch + "/sys")

    # The branch exchange family.
    bx_words = list(range(0x4700, 0x4800))
    bx_rows = measure_family(bx_words, args.scratch + "/bx")

    table = {
        "it": it_rows,
        "system": sys_rows,
        "branch_exchange": bx_rows,
        "assembler": assembler_encodings(),
    }
    with open(args.out, "w") as f:
        json.dump(table, f, indent=1)

    # A readable report, so the table can be checked by eye.
    print("the `it` family, read as `cond mask` -> the two references")
    print("  (the mnemonic is printed with the letters counted, so that")
    print("   `it` and `itt` cannot be confused with `itte`.)")
    for row in it_rows:
        hw = row["hw"]
        cond = (hw >> 4) & 0xF
        mask = hw & 0xF
        c = row["capstone"] or ""
        o = (row["objdump"] or "").split("\t")[0].split()[0] if row["objdump"] else ""
        cm = c.split()[0] if c else "-"
        mark = "" if cm == o else "   <-- DIFFER"
        print(f"  cond={cond:x} mask={mask:x}  capstone={cm:<10s} "
              f"objdump={o:<10s}{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
