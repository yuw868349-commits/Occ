#!/usr/bin/env python3
"""Group the differences between the decoder and the references.

The differential test reports every halfword that the decoder reads
differently, and there are tens of thousands of them.  Printing them one by
one hides the shape of the problem: almost always a whole *class* of
encodings is wrong in the same way, and the fix is one edit that covers the
class.

So the differences are grouped by the pair of readings — what the
references say against what the decoder says — and the groups are printed
by size, with a couple of examples each.  The group that is worth fixing
first is the largest one that is not already known to be out of scope.
"""
import argparse
import collections
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thumb_canon as canon
import thumb_ref as ref

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
        _, _, text = line.split(" ", 2)
        out.append(text)
    return out


def shape(text):
    """Reduce a reading to its shape: the class without the field values.

    A group keyed by the whole reading is a group per halfword, because
    the register numbers differ.  What is wanted is the *pattern* of the
    difference — the same class of encoding, read the same wrong way.
    """
    if text is None:
        return None
    t = re.sub(r"\br(\d+)\b", "rN", text)
    t = re.sub(r"#0x[0-9a-f]+", "#N", t)
    t = re.sub(r"\bsp\b", "rN", t)
    t = re.sub(r"\blr\b", "rN", t)
    t = re.sub(r"\bpc\b", "rN", t)
    t = re.sub(r"\{[^}]*\}", "{N}", t)
    return t


def group(rows, mine, limit=40, only=None):
    """Group the disagreements by the pair of readings."""
    groups = collections.OrderedDict()
    for (hw, cs_raw, od_raw), decoded in zip(rows, mine):
        # The halfwords from `11101` up are the prefixes of the 32-bit
        # set: they are not 16-bit instructions at all, and the decoder
        # is right to read them as the start of a wide one.
        if hw >> 11 in (0b11101, 0b11110, 0b11111):
            continue
        pc = ref.iso_pc(hw)
        cs = canon.canonical(cs_raw, pc)
        od = canon.canonical(od_raw, pc)
        mine_c = canon.canonical(decoded, pc)
        if cs is not None and od is not None:
            if cs != od:
                continue  # the references disagree; settled elsewhere
            if mine_c == cs:
                continue
            key = (shape(cs), shape(mine_c))
        else:
            if mine_c is not None and cs is None and od is None:
                key = ("<undefined>", shape(mine_c))
            else:
                continue
        if only and not re.search(only, key[0]):
            continue
        g = groups.setdefault(key, [])
        if len(g) < 4:
            g.append((hw, cs_raw, od_raw, decoded))
    return groups


def by_mnemonic(rows, mine):
    """Count the disagreements by the operation the references name.

    The grouping by *pair* of readings is too fine: an alias that is
    written one way in one class and another way in another class produces
    one group per class, and the size of a group then says nothing about
    how much work the fix is.  This counts by the reference's own
    operation, which is the unit the decoder's tables are keyed on.
    """
    counts = collections.Counter()
    total = 0
    for (hw, cs_raw, od_raw), decoded in zip(rows, mine):
        if hw >> 11 in (0b11101, 0b11110, 0b11111):
            continue
        pc = ref.iso_pc(hw)
        cs = canon.canonical(cs_raw, pc)
        od = canon.canonical(od_raw, pc)
        mine_c = canon.canonical(decoded, pc)
        if cs is not None and od is not None:
            if cs != od or mine_c == cs:
                continue
        else:
            if not (mine_c is not None and cs is None and od is None):
                continue
        total += 1
        want = cs if cs is not None else "<undefined>"
        op = want.split("|")[-1].split()[0]
        counts[op] += 1
    return counts, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=DECODER)
    ap.add_argument("--scratch", default="/tmp/occ-thumb-diff")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--only", default=None,
                    help="only groups whose reference reading matches this regex")
    ap.add_argument("--width", type=int, default=16)
    args = ap.parse_args()

    rows = ref.read_16bit_space(args.scratch + "/od16")
    entries = [(hw, 0, ref.iso_addr(hw)) for hw, _, _ in rows]
    mine = run_decoder(entries)

    counts, total = by_mnemonic(rows, mine)
    undef = counts.pop("<undefined>", 0)
    print(f"{total} halfwords differ from the references")
    print(f"  of which the references do not define the class at all: {undef}")
    print(f"  the rest, by the operation the references name:")
    for op, n in counts.most_common(args.limit):
        print(f"    {n:6d}  {op}")

    groups = group(rows, mine, only=args.only)
    print()
    print(f"{len(groups)} distinct kinds of difference (shown up to {args.limit})")
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    for (want, got), examples in ranked[:args.limit]:
        print()
        print(f"  want: {want}")
        print(f"  got : {got}")
        for hw, cs_raw, od_raw, decoded in examples:
            print(f"    hw=0x{hw:04x}  capstone: {cs_raw!r}  objdump: {od_raw!r}"
                  f"  decoder: {decoded!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
