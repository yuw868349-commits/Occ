#!/usr/bin/env python3
"""Check the 32-bit Thumb decoder against two independent references.

The 16-bit space could be read in full; the 32-bit space is four times the
size of the address space of a halfword and cannot be.  So the space is
covered by a *sample* that is built from its structure rather than at
random:

* every one of the 7680 first halfwords that can open a 32-bit
  instruction, each behind several second halfwords, so that every class
  is met and the classes that share a first halfword are told apart;
* every bit of the second halfword, flipped on its own, for every first
  halfword, so that every field of every class is exercised at least
  once;
* every bit of the first halfword, flipped on its own, so that the
  classes that are chosen by a low bit are told apart;
* a random sample, so that the encodings the structure did not think of
  are not missed.

The sample is a compromise, and it is stated as one: it cannot prove the
decoder right over the whole space the way the 16-bit test does.  What it
does is catch every layout that is wrong in a way that shows up in more
than one place, which is every layout that is wrong.

Usage:
    thumb_diff32.py [--sample N] [--bin PATH]
"""
import argparse
import collections
import os
import random
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_canon as canon
import thumb_ref as ref

# The decoder under test: a small program that reads a list of encodings
# and prints what it makes of each one.
DECODER = "target/release/examples/thumb_decode"

# The second halfwords a first halfword is read behind when the classes
# are found.  They are the ones that exercise the fields: all clear, all
# set, and the two that alternate.
PROBES = (0x0000, 0xFFFF, 0x5555, 0xAAAA, 0x0F0F, 0xF0F0, 0x00FF, 0xFF00)


def sample_words(random_count):
    """The encodings the test reads.

    The first halfword of a 32-bit instruction is `11101`, `11110` or
    `11111`, so the space opens at `0xe800`.
    """
    words = []
    seen = set()

    def add(hw1, hw2):
        key = (hw1, hw2)
        if key in seen:
            return
        seen.add(key)
        words.append(((hw2 << 16) | hw1, hw1, hw2))

    for hw1 in range(0xE800, 0x10000):
        for hw2 in PROBES:
            add(hw1, hw2)
    # Every bit of the second halfword on its own.
    for hw1 in range(0xE800, 0x10000):
        for bit in range(16):
            add(hw1, 1 << bit)
            add(hw1, 0xFFFF ^ (1 << bit))
    # Every bit of the first halfword on its own.
    #
    # Only the bits below the prefix are flipped.  The prefix is the top
    # five bits, and it is what says the encoding is a 32-bit one at
    # all: flipping one of them turns the sample into a 16-bit
    # instruction, which the references would read as two separate
    # instructions and this decoder would read as one wide one.  The
    # sample would then measure the width logic rather than the
    # layout, and every such entry would count as a difference.
    for hw1 in range(0xE800, 0x10000):
        for bit in range(11):
            add(hw1 ^ (1 << bit), 0x0000)
            add(hw1 ^ (1 << bit), 0x1234)
    rng = random.Random(20240917)
    for _ in range(random_count):
        add(rng.randrange(0xE800, 0x10000), rng.randrange(0x10000))
    return words


def run_decoder(entries):
    """Read the encodings through the decoder and collect its answers."""
    payload = "\n".join(f"{hw1:04x} {hw2:04x} {at:x}" for _, hw1, hw2, at in entries)
    proc = subprocess.run([DECODER], input=payload, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr[:2000])
        raise SystemExit(f"decoder failed with status {proc.returncode}")
    out = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        _, _, text = line.split(" ", 2)
        out.append(text)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=DECODER)
    ap.add_argument("--scratch", default="/tmp/occ-thumb-diff/iso32")
    ap.add_argument("--random", type=int, default=200_000,
                    help="how many encodings to read at random")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--only", default=None,
                    help="only report groups whose reference reading matches this")
    ap.add_argument("--region", default=None,
                    help="only count differences whose first halfword is in this range, "
                         "as 0xf000-0xf7ff")
    args = ap.parse_args()

    if not os.path.exists(args.bin):
        raise SystemExit(f"build the decoder first: cargo build --release --example thumb_decode")

    words = sample_words(args.random)
    print(f"reading {len(words)} encodings")
    table = ref.objdump_isolated_words32([w for w, _, _ in words], args.scratch)

    entries = []
    for i, (w, hw1, hw2) in enumerate(words):
        entries.append((w, hw1, hw2, ref.iso32_addr(i)))
    mine = run_decoder(entries)

    wrong = 0
    refsplit = 0
    onlyref = 0
    undefined = 0
    groups = collections.OrderedDict()
    total = 0

    lo = hi = None
    if args.region:
        parts = args.region.split("-")
        lo = int(parts[0], 0)
        hi = int(parts[1], 0) if len(parts) > 1 else lo

    def in_region(hw1):
        return lo is None or lo <= hw1 <= hi

    def kind(text):
        """The difference without the registers and the values in it.

        Two encodings that differ only in which register they name are the
        same kind of difference, and the report has to say so or it is a
        list of hundreds of thousands of lines.
        """
        if text is None:
            return None
        out = re.sub(r"\br\d+\b|\b(?:sb|sl|fp|ip|sp|lr|pc)\b", "R", text)
        out = re.sub(r"#-?0x[0-9a-f]+|#-?\d+", "#I", out)
        out = re.sub(r"<[^>]*>", "<D>", out)
        return out

    def record(want, got, hw1, hw2):
        """Note one difference, and keep a few examples of it."""
        key = (kind(want), kind(got))
        g = groups.setdefault(key, [0, []])
        g[0] += 1
        if len(g[1]) < 3:
            g[1].append((hw1, hw2))

    for i, (w, hw1, hw2) in enumerate(words):
        addr = ref.iso32_addr(i)
        pc = ref.iso32_pc(i)
        cs = canon.canonical(ref.capstone_read(w, addr, 4), pc)
        od = canon.canonical(table.get(addr), pc)
        mine_c = canon.canonical(mine[i], pc)
        total += 1

        if cs is not None and od is not None:
            if cs != od:
                refsplit += 1
                continue
            if mine_c != cs:
                wrong += 1
                if in_region(hw1):
                    record(cs, mine_c, hw1, hw2)
            continue

        if cs is None and od is None:
            if mine_c is not None:
                wrong += 1
                if in_region(hw1):
                    record("<undefined>", mine_c, hw1, hw2)
            else:
                undefined += 1
            continue

        onlyref += 1
        want = cs if cs is not None else od
        if mine_c != want:
            wrong += 1
            if in_region(hw1):
                record(want, mine_c, hw1, hw2)

    # Where the differences are: the top byte of the first halfword is
    # the class of the encoding.
    #
    # Four outcomes per encoding, and they are worth telling apart:
    #
    #   both   the references define it and the decoder agrees
    #   differ  the references define it and the decoder says something else
    #   made    the references say it is not an instruction and the decoder
    #           says something anyway
    #   none   the references say it is not an instruction, and so does the
    #           decoder
    regions = collections.Counter()
    totals = collections.Counter()
    for i, (w, hw1, hw2) in enumerate(words):
        addr = ref.iso32_addr(i)
        pc = ref.iso32_pc(i)
        cs = canon.canonical(ref.capstone_read(w, addr, 4), pc)
        od = canon.canonical(table.get(addr), pc)
        mine_c = canon.canonical(mine[i], pc)
        top = hw1 >> 8
        totals[top] += 1
        if cs is not None and od is not None and cs != od:
            # The references disagree; neither is the truth on its own.
            regions[top + 0x100] += 1
            continue
        want = cs if cs is not None else od
        if want is None:
            if mine_c is not None:
                regions[top + 0x200] += 1
            else:
                regions[top + 0x300] += 1
        elif mine_c != want:
            regions[top + 0x400] += 1
        else:
            regions[top + 0x500] += 1

    print(f"32-bit sample: {total} encodings")
    print(f"  decoder disagrees with the references : {wrong}")
    print(f"  the references disagree with each other: {refsplit}")
    print(f"  only one reference defines the class  : {onlyref}")
    print(f"  neither reference defines an instruction: {undefined}")
    print(f"  {len(groups)} distinct kinds of difference")
    print("  where they are (top byte of the first halfword)")
    print("    agree / differ / made / none / split")
    for top in sorted(totals):
        if top >= 0x100:
            continue
        split = regions[top + 0x100]
        differ = regions[top + 0x400]
        made = regions[top + 0x200]
        none = regions[top + 0x300]
        agree = regions[top + 0x500]
        if not (split or differ or made):
            continue
        print(f"    0x{top:02x}xx  {agree:7d} {differ:7d} {made:7d} {none:7d} {split:7d}")
    ranked = sorted(groups.items(), key=lambda kv: -kv[1][0])
    for (want, got), (count, examples) in ranked[:args.limit]:
        if args.only and args.only not in want:
            continue
        print()
        print(f"  want: {want}")
        print(f"  got : {got}")
        print(f"  {count} encodings")
        for hw1, hw2 in examples:
            print(f"    hw1=0x{hw1:04x} hw2=0x{hw2:04x}")
    return 1 if wrong else 0

if __name__ == "__main__":
    sys.exit(main())
