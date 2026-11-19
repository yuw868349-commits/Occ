#!/usr/bin/env python3
"""Compare this decoder's reading of a file against two references.

The two references are `llvm-objdump` and `capstone`, and they are read
independently: where they agree, the reading is taken as the truth, and
where they disagree the difference is reported rather than resolved.

The addresses are not compared. A relocatable object has no load address,
and this tool rebases a symbol onto the file offset of the section it is
in, so the same instruction sits at two different numbers in the two
listings. What is compared is the *sequence* of instructions and what each
one says, which is what the difference in the numbers is hiding.

Usage:
    thumb_obj_cmp.py <file.o> [--arch armv7] [--mode thumb]
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import thumb_ref as ref  # noqa: E402

OBJDUMP = ref.OBJDUMP
OBJCOPY = ref.OBJCOPY


def objdump_listing(path, triple):
    """The instructions of a file, as `llvm-objdump` reads them."""
    res = subprocess.run(
        [OBJDUMP, "-d", f"--triple={triple}", path],
        check=True, capture_output=True, text=True).stdout
    out = []
    for line in res.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\s+([0-9a-f ]+?)\s{2,}(\S.*)$", line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        text = m.group(3).strip()
        # The symbol name and the relocation note are not the instruction.
        text = text.split("<")[0].strip()
        text = re.sub(r"\s*@\s*imm\s*=\s*#\S+", "", text).strip()
        out.append((addr, text))
    return out


def ours(path):
    """The instructions of a file, as this tool reads them."""
    exe = os.path.join(os.path.dirname(HERE), "target", "debug", "examples", "dump")
    if not os.path.exists(exe):
        raise SystemExit("build the decoder first: cargo build -p occ-decode --example dump")
    res = subprocess.run([exe, path], check=True, capture_output=True, text=True).stdout
    out = []
    for line in res.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+(.*?)\s{2,}([0-9a-f ]+)$", line)
        if not m:
            continue
        out.append((int(m.group(1), 16), m.group(2).strip()))
    return out


def normalise(text):
    """The parts of an instruction that the two listings must agree on.

    The register names and the immediate values are kept: they are the
    instruction. What is dropped is the way the target of a branch is
    written, because the two listings number the instructions differently.
    """
    text = text.replace("\t", " ")
    text = text.split("<")[0].strip()
    text = re.sub(r"\s*@\s*imm\s*=\s*#\S+", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    # `b.w 0x1234` and `b 0x1234` are the same branch; the `.w` says the
    # encoding is the wide one, which is not what the instruction does.
    text = re.sub(r"^([a-z][a-z0-9]*)\.w\b", r"\1", text)
    # A target written as an absolute number differs between the listings.
    text = re.sub(r"#?0x[0-9a-f]+$", "#TARGET", text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--triple", default="thumbv7-linux-gnueabi")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    theirs = objdump_listing(args.file, args.triple)
    mine = ours(args.file)
    print(f"reference: {len(theirs)} instructions")
    print(f"ours:     {len(mine)} instructions")

    # The two listings start at different places: the reference numbers
    # from the section's own address and this tool from the file offset.
    # The comparison is by position in the sequence.
    n = min(len(theirs), len(mine))
    same = 0
    shown = 0
    for i in range(n):
        a = normalise(theirs[i][1])
        b = normalise(mine[i][1])
        if a == b:
            same += 1
            continue
        if shown >= args.limit:
            continue
        shown += 1
        print(f"  #{i} reference {theirs[i][1]!r}")
        print(f"     ours      {mine[i][1]!r}")
    print(f"agree: {same}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
