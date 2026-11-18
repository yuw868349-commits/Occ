#!/usr/bin/env python3
"""Check the Thumb decoder against two independent references.

The decoder is only worth believing if it agrees with something that was
written independently of it.  Two such things exist here: capstone, and
llvm-objdump, which is ARM's own disassembler.  Both are read at the same
known address, and both are reduced to the same canonical form, so a
difference between them is a difference in the *architecture* rather than
in the spelling.

Three things are reported separately, because they matter separately:

* **A class the decoder gets wrong.**  The decoder read the instruction and
  named it differently from both references.  This is a bug.
* **A class the references disagree about.**  capstone and objdump read the
  same encoding differently.  This is not the decoder's fault, but it does
  mean the encoding is one the two profiles read differently, and the
  difference is reported so it can be settled on purpose.
* **A class only one reference defines.**  Usually a vector instruction or
  a coprocessor access, which this decoder does not cover, or a 32-bit
  encoding that has been read in a 16-bit slot.

Usage: thumb_diff.py [--wide N] [--bin PATH]
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_canon as canon
import thumb_ref as ref

# The decoder under test: a small program that reads a list of encodings
# and prints what it makes of each one.
DECODER = "target/release/examples/thumb_decode"


def run_decoder(entries):
    """Read the encodings through the decoder and collect its answers.

    Each encoding is read at the address the references read it at, so
    that a branch's absolute destination means the same thing on both
    sides.
    """
    payload = "\n".join(f"{hw1:04x} {hw2:04x} {at:x}" for hw1, hw2, at in entries)
    proc = subprocess.run([DECODER], input=payload, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr[:2000])
        raise SystemExit(f"decoder failed with status {proc.returncode}")
    out = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        hw1, hw2, text = line.split(" ", 2)
        out.append(text)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="target/release/examples/thumb_decode")
    ap.add_argument("--scratch", default="/tmp/occ-thumb-diff")
    args = ap.parse_args()

    if not os.path.exists(args.bin):
        raise SystemExit(f"build the decoder first: cargo build --release --example thumb_decode")

    rows = ref.read_16bit_space(args.scratch + "/od16")
    entries = [(hw, 0, ref.iso_addr(hw)) for hw, _, _ in rows]
    mine = run_decoder(entries)

    wrong = 0
    refsplit = 0
    onlyref = 0
    undefined = 0
    wide = 0
    total = 0
    examples = []

    for (hw, cs_raw, od_raw), decoded in zip(rows, mine):
        # The halfwords from `11101` up are the prefixes of the 32-bit
        # set: they are not 16-bit instructions at all.  The decoder reads
        # them as the start of a wide one, and the references do too, but
        # with a different second halfword behind them, so the three sides
        # are not reading the same instruction and there is nothing to
        # compare.  They are counted on their own.
        if hw >> 11 in (0b11101, 0b11110, 0b11111):
            wide += 1
            continue
        pc = ref.iso_pc(hw)
        cs = canon.canonical(cs_raw, pc)
        od = canon.canonical(od_raw, pc)
        mine_c = canon.canonical(decoded, pc)
        total += 1

        if cs is not None and od is not None:
            if cs != od:
                refsplit += 1
                continue
            if mine_c != cs:
                wrong += 1
                if len(examples) < 40:
                    examples.append((hw, cs_raw, od_raw, decoded, cs, mine_c))
            continue

        if cs is None and od is None:
            # Neither reference reads an instruction here.  The decoder
            # has to agree: claiming an instruction the architecture does
            # not define is a bug, and saying so too is correct.
            if mine_c is not None:
                wrong += 1
                if len(examples) < 40:
                    examples.append((hw, cs_raw, od_raw, decoded, None, mine_c))
            else:
                undefined += 1
            continue

        # Exactly one reference defines the class.  The decoder has to
        # agree with the one that does.
        onlyref += 1
        want = cs if cs is not None else od
        if mine_c != want:
            wrong += 1
            if len(examples) < 40:
                examples.append((hw, cs_raw, od_raw, decoded, want, mine_c))

    print(f"16-bit space: {total} halfwords")
    print(f"  decoder disagrees with the references : {wrong}")
    print(f"  the references disagree with each other: {refsplit}")
    print(f"  only one reference defines the class  : {onlyref}")
    print(f"  neither reference defines an instruction: {undefined}")
    print(f"  the halfwords that open a 32-bit one  : {wide}")
    for hw, cs_raw, od_raw, decoded, cs, mine_c in examples:
        print(f"  hw=0x{hw:04x}")
        print(f"    capstone: {cs_raw}")
        print(f"    objdump : {od_raw}")
        print(f"    decoder : {decoded}")
        print(f"    want    : {cs}    got: {mine_c}")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
