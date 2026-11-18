#!/usr/bin/env python3
"""Ask the assembler what a compare-and-branch encoding means.

The decoder and the two references disagree about *which way* a compare-and-
branch goes, and the references disagree with each other about how the
offset field is signed.  None of the three is a proof: the decoder may be
wrong, and the two references may share a mistake.

The assembler is a fourth implementation, and the only one that has to
*produce* the encoding from a known intent.  So it is asked the question
directly: put an instruction at a known distance from a label, and read the
bits it chose.  The distance is what was asked for; the bits are what the
architecture says that distance is.

For every distance from the smallest to the largest the 16-bit form can
reach, the encoding is produced and printed, so the offset field's meaning
can be read off the table without anyone having to remember it.

Usage:
    thumb_asm_cbz.py
"""
import os
import re
import subprocess
import sys
import tempfile

MCTARGET = "/usr/lib/llvm-23/bin/llvm-mc"
OBJDUMP = "/usr/lib/llvm-23/bin/llvm-objdump"
TRIPLE = "thumbv7-linux-gnueabi"


def assemble(distance):
    """Ask for a `cbz` whose target is `distance` bytes ahead.

    The instruction sits at offset 8 in the text, and the label at offset
    8 + 4 + distance, so that the offset the assembler must encode is
    exactly `distance` — the program counter of a Thumb instruction reads
    as its own address plus four.
    """
    src = f"""
.text
        .space 8
        cbz r0, tgt
        .space {distance}
tgt:
        .word 0
"""
    with tempfile.TemporaryDirectory() as d:
        s = f"{d}/a.s"
        o = f"{d}/a.o"
        with open(s, "w") as f:
            f.write(src)
        r = subprocess.run(
            [MCTARGET, f"-triple={TRIPLE}", "-filetype=obj", s, "-o", o],
            capture_output=True, text=True)
        if r.returncode != 0:
            return None, r.stderr.strip().splitlines()[:2]
        d2 = subprocess.run(
            [OBJDUMP, "-d", f"--triple={TRIPLE}", o],
            capture_output=True, text=True).stdout
        for line in d2.splitlines():
            m = re.match(r"\s+([0-9a-f]+):\s+([0-9a-f ]+)\s*(.*)$", line)
            if m and int(m.group(1), 16) == 8:
                raw = m.group(2).replace(" ", "")
                return int(raw[:4], 16), m.group(3)
    return None, None


def main():
    print("distance  encoding   objdump")
    print("-------  --------   -------")
    rows = []
    for d in list(range(0, 34, 2)) + list(range(36, 132, 2)):
        enc, text = assemble(d)
        if enc is None:
            print(f"{d:7d}  <rejected>  {text}")
            continue
        rows.append((d, enc, text))
        print(f"{d:7d}  0x{enc:04x}     {text}")
    if rows:
        print()
        print(f"the 16-bit form reaches {max(d for d, _, _ in rows)} bytes ahead")


if __name__ == "__main__":
    main()
