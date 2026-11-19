#!/usr/bin/env python3
"""Read every encoding of the T32 `0xFA` class and record what it means.

The class is the one whose first halfword is `1111 1010`: the register
shifts, the extends, and the parallel additions and subtractions.  It is
read from the second halfword as well, so a mistake in one field shows up
as a different instruction rather than as a wrong operand, and the only
way to find it is to read every encoding of the class and compare.

Two references are available.  `llvm-objdump` and `arm-none-eabi-objdump`
are separate implementations of the same architecture manual, and where
they differ the encoding is either undefined or unpredictable, so both
have to be read to tell the two apart.

Usage:
    t32_fa_scan.py [--out table.txt] [--rn 3] [--objdump llvm|gnu]
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

LLVM = "llvm-objdump"
GNU = "arm-none-eabi-objdump"


def llvm_listing(path):
    """The instructions of a file, as `llvm-objdump` reads them."""
    res = subprocess.run(
        [LLVM, "-d", "--triple=thumbv7-linux-gnueabi", path],
        check=True, capture_output=True, text=True).stdout
    out = []
    for line in res.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\s+([0-9a-f ]+?)\s{2,}(\S.*)$", line)
        if not m:
            continue
        text = m.group(3).strip()
        text = re.sub(r"\s*@\s*imm\s*=\s*#\S+", "", text).strip()
        out.append(text)
    return out


def gnu_listing(path):
    """The instructions of a file, as `arm-none-eabi-objdump` reads them."""
    res = subprocess.run(
        [GNU, "-d", path], check=True, capture_output=True, text=True).stdout
    out = []
    for line in res.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\s+([0-9a-f]{4})\s+([0-9a-f]{4})\s*(.*)$", line)
        if not m:
            continue
        text = m.group(4).strip()
        if "UNDEFINED" in text:
            text = "<undefined>"
        out.append(text)
    return out


def generate(tmp, op, rn, hw2_values):
    """An object holding every encoding of the class for one operation."""
    lines = [".syntax unified", ".thumb", ".text", ".global fa", ".type fa, %function",
             "fa:"]
    hw1 = 0xFA00 | (op << 4) | rn
    for hw2 in hw2_values:
        # The first halfword of a T32 instruction is the one at the lower
        # address, and a little-endian word holds it in its low half.
        lines.append(f"  .word {((hw2 << 16) | hw1):#010x}")
    lines.append(".size fa, . - fa")
    src = os.path.join(tmp, "fa.s")
    with open(src, "w") as f:
        f.write("\n".join(lines) + "\n")
    obj = os.path.join(tmp, "fa.o")
    subprocess.run(
        ["clang", "--target=armv7-linux-gnueabi", "-mthumb", "-c", src, "-o", obj],
        check=True)
    return obj, hw1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--op", type=int, default=None,
                    help="scan a single operation of the first halfword")
    ap.add_argument("--rn", type=int, default=3)
    ap.add_argument("--objdump", default="llvm", choices=["llvm", "gnu"])
    ap.add_argument("--hw2-lo", type=lambda v: int(v, 0), default=0,
                    help="the lowest second halfword to read")
    ap.add_argument("--hw2-hi", type=lambda v: int(v, 0), default=0x10000,
                    help="the highest second halfword to read, exclusive")
    args = ap.parse_args()
    listing = llvm_listing if args.objdump == "llvm" else gnu_listing
    ops = [args.op] if args.op is not None else list(range(16))
    hw2_values = list(range(args.hw2_lo, args.hw2_hi))
    lines = []
    with tempfile.TemporaryDirectory() as tmp:
        for op in ops:
            obj, hw1 = generate(tmp, op, args.rn, hw2_values)
            ref = listing(obj)
            if len(ref) != len(hw2_values):
                print(f"length differs: reference {len(ref)}, expected {len(hw2_values)}")
                return 1
            for hw2, text in zip(hw2_values, ref):
                lines.append(f"{hw1:#06x} {hw2:#06x}  {text}")
    out = "\n".join(lines)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out + "\n")
        print(f"wrote {len(lines)} lines to {args.out}")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
