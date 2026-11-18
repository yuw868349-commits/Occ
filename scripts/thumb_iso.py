#!/usr/bin/env python3
"""Read encodings with llvm-objdump, with nothing carried over from before.

`llvm-objdump` keeps the state of an `it` block while it walks the
instructions: the ones that follow an `it` are printed with the block's
condition attached to them, whether or not they were meant to be there.
Reading a whole space in one stream therefore reports the *block's* view
of the instructions after an `it`, not their own.

Putting each encoding in a section of its own does not fix it: the state
is carried across sections as well, so an `it` in one section still
reaches into the next.

What does fix it is padding.  An `it` block covers at most four
instructions, so four `nop`s before the encoding under test mean that
whatever block was open has been consumed and closed, and the encoding is
read on its own.  The padding is in its own section for the same reason:
the state has to be broken before the encoding, not merely separated from
the encoding after it.

The addresses the padding introduces do not matter, because this is for
the families that have no branch destination in them.

Usage:
    thumb_iso.py --range 0xbf00:0xbf20
    thumb_iso.py --words 0xbe00,0xbe80
"""
import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capstone as cs

MCTARGET = "/usr/lib/llvm-23/bin/llvm-mc"
OBJDUMP = "/usr/lib/llvm-23/bin/llvm-objdump"
TRIPLE = "thumbv7-linux-gnueabi"

# An `it` block covers at most four instructions, so four `nop`s are
# enough to close any block that was open.
PAD = 4
NOP = 0xBF00


def capstone_read(hw, addr=0):
    """Read one halfword with capstone, at a known address.

    `count=1` stops the decoder at the end of the instruction, so there
    is no state carried over from anything that came before it.
    """
    md = cs.Cs(cs.CS_ARCH_ARM, cs.CS_MODE_THUMB)
    md.detail = False
    code = hw.to_bytes(2, "little")
    for i in md.disasm(code, addr, count=1):
        if i.size != 2:
            return None
        return f"{i.mnemonic} {i.op_str}".strip()
    return None


def objdump_isolated(words, scratch, pad=PAD):
    """Read each word with llvm-objdump, with the state pushed clean."""
    os.makedirs(scratch, exist_ok=True)
    stream = []
    for w in words:
        stream += [NOP] * pad + [w]

    src = []
    for i, w in enumerate(stream):
        src.append(f'.section .text.s{i},"ax",%progbits\n.inst 0x{w:04x}\n')
    asm = f"{scratch}/a.s"
    with open(asm, "w") as f:
        f.write("\n".join(src))
    obj = f"{scratch}/a.o"
    proc = subprocess.run(
        [MCTARGET, f"-triple={TRIPLE}", "-filetype=obj", asm, "-o", obj],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"llvm-mc failed: {proc.stderr[:800]}")
    res = subprocess.run(
        [OBJDUMP, "-d", f"--triple={TRIPLE}", obj],
        check=True, capture_output=True, text=True).stdout

    # Every section holds one instruction, and the sections come back in
    # the order they were written.
    readings = []
    for line in res.splitlines():
        m = re.match(r"\s+[0-9a-f]+:\s+[0-9a-f ]+\s*(.*)$", line)
        if m:
            readings.append(m.group(1).strip())
    if len(readings) != len(stream):
        raise SystemExit(f"read {len(readings)} readings for {len(stream)} words")

    # The padding goes back out; the encoding's own reading is what is
    # wanted.
    return [readings[i * (pad + 1) + pad] for i in range(len(words))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", default="")
    ap.add_argument("--range", default=None,
                    help="read every halfword in this range, as start:end")
    ap.add_argument("--scratch", default="/tmp/occ-thumb-iso")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.range:
        a, b = (int(x, 0) for x in args.range.split(":"))
        words = list(range(a, b))
    else:
        words = [int(w, 0) for w in args.words.split(",") if w.strip()]

    od = objdump_isolated(words, args.scratch)
    for w, o in zip(words, od):
        c = capstone_read(w)
        if args.quiet:
            continue
        mark = "" if (c or "").split(" ")[0] == o.split(" ")[0] else "   <-- DIFFER"
        print(f"  {w:04x}  capstone={c!r:<28s} objdump={o!r}{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
