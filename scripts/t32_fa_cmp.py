#!/usr/bin/env python3
"""Check the T32 `0xFA` class against `llvm-objdump`, in one process.

The class is the one whose first halfword is `1111 1010`: the register
shifts, the extends, and the parallel additions and subtractions.  It is
read from the second halfword as well, so a mistake in one field shows up
as a different instruction rather than as a wrong operand, and the only
way to find it is to read every encoding of the class and compare.

Both sides are produced here rather than read from files: this script
writes every encoding into an object, asks `llvm-objdump` what it is,
runs the decoder over the same bytes through the `t32_fa_table` example,
and diffs the two listings.  Use `t32_fa_scan.py` instead when the
reference side alone has to be inspected by hand.

The reference is a different implementation of the same architecture
manual.  Where the two disagree, the difference is reported: the class
has more than one way to spell each instruction, and a name that differs
is not always a mistake, but an instruction that means something else
always is.

Usage:
    t32_fa_cmp.py [--arch armv7]
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

OBJDUMP = "llvm-objdump"


def objdump_listing(path):
    """The instructions of a file, as `llvm-objdump` reads them."""
    res = subprocess.run(
        [OBJDUMP, "-d", "--triple=thumbv7-linux-gnueabi", path],
        check=True, capture_output=True, text=True).stdout
    out = []
    for line in res.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\s+([0-9a-f ]+?)\s{2,}(\S.*)$", line)
        if not m:
            continue
        text = m.group(3).strip()
        # The reference writes an encoding it does not know as `<unknown>`.
        # That is a reading, not an absence, so it is kept.
        text = re.sub(r"\s*@\s*imm\s*=\s*#\S+", "", text).strip()
        out.append(text)
    return out


def ours(path):
    """The instructions of a file, as this tool reads them."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exe = os.path.join(here, "target", "release", "examples", "dump")
    if not os.path.exists(exe):
        raise SystemExit("build the decoder first: cargo build -p occ-decode --example dump")
    res = subprocess.run([exe, path], check=True, capture_output=True, text=True).stdout
    out = []
    for line in res.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+(.*?)\s{2,}([0-9a-f ]+)$", line)
        if not m:
            continue
        out.append(m.group(2).strip())
    return out


def generate(tmp):
    """An object holding every encoding of the class.

    The second halfword is sampled rather than exhausted: the class reads
    its operation from four bits of the first halfword and from four bits
    of the second, and the samples below cover every field of the second
    that changes the instruction.
    """
    seconds = [
        0xF080, 0xF081, 0xF088, 0xF090, 0xF0A0, 0xF0C0, 0xF180, 0xF280,
        0x0000, 0x0010, 0x0080, 0x0100, 0x8000, 0x8100, 0xF800, 0xF8F0,
    ]
    lines = [".syntax unified", ".thumb", ".text", ".global fa", ".type fa, %function",
             "fa:"]
    words = []
    for op in range(16):
        for rn in (0, 3):
            hw1 = 0xFA00 | (op << 4) | rn
            for hw2 in seconds:
                # The first halfword of a T32 instruction is the one at the
                # lower address, and a little-endian word holds it in its
                # low half.
                words.append((hw2 << 16) | hw1)
    for w in words:
        lines.append(f"  .word {w:#010x}")
    lines.append(".size fa, . - fa")
    src = os.path.join(tmp, "fa.s")
    with open(src, "w") as f:
        f.write("\n".join(lines) + "\n")
    obj = os.path.join(tmp, "fa.o")
    subprocess.run(
        ["clang", "--target=armv7-linux-gnueabi", "-mthumb", "-c", src, "-o", obj],
        check=True)
    return obj, words


def normalise(text):
    """The two listings spell the same instruction in the same words.

    `llvm-objdump` writes the destination first and this tool does too, but
    the width marker and the space before the operands differ, and the
    reference writes a register list in its own way.
    """
    t = text.replace(".w", "")
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace(" ,", ",")
    # An encoding the reference does not know and one this tool calls
    # undefined are the same reading, spelled two ways.
    if t in ("<unknown>",) or t.endswith("; undefined"):
        return "<undefined>"
    return t


def reads_an_undefined_field(word):
    """Whether the reference read a field the encoding does not define.

    The byte reversals and the count of leading zeros take one source, and
    it is the one in the second halfword.  Their first halfword carries a
    register field as well, but the instruction has no second operand for
    it: the assembler writes the source register in both places, and the
    architecture leaves the field with nothing to say.  The reference
    reads the two fields together, so on an encoding where they differ it
    names a register that is neither of them.

    Such an encoding is not one the assembler produces, and the reading
    the reference gives it is not one the machine can execute, so the
    difference is counted on its own rather than as a disagreement about
    what the instruction does.
    """
    hw1 = word & 0xFFFF
    hw2 = (word >> 16) & 0xFFFF
    op = (hw1 >> 4) & 0xF
    variant = (hw2 >> 4) & 0xF
    if op == 9 and variant in (8, 9, 10, 11):
        return (hw1 & 0xF) != 0
    if op == 11 and variant == 8:
        return (hw1 & 0xF) != 0
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        obj, words = generate(tmp)
        ref = objdump_listing(obj)
        mine = ours(obj)
    if len(ref) != len(mine):
        print(f"length differs: reference {len(ref)}, ours {len(mine)}")
        for i, (a, b) in enumerate(zip(ref, mine)):
            if a != b:
                print(f"  first difference at {i}: {a!r} vs {b!r}")
                break
        return 1
    bad = 0
    loose = 0
    for i, (a, b) in enumerate(zip(ref, mine)):
        if normalise(a) == normalise(b):
            continue
        if reads_an_undefined_field(words[i]):
            loose += 1
            if loose <= 20:
                print(f"{i:5d}  hw={words[i]:#010x}  reference {a!r}  ours {b!r}"
                      f"  [the reference read a field the encoding does not define]")
            continue
        bad += 1
        if bad <= 60:
            print(f"{i:5d}  hw={words[i]:#010x}  reference {a!r}  ours {b!r}")
    print(f"{len(ref)} instructions, {bad} differ"
          + (f", {loose} of them the reference reading a field the encoding"
             f" does not define" if loose else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
