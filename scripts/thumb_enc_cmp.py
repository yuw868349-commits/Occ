#!/usr/bin/env python3
"""Compare this decoder's reading of named encodings against the references.

The encodings are given on the input as pairs of halfwords, one per line.
Each is read by `llvm-objdump` and by this decoder, and the two readings
are put side by side.

The addresses are the same for both, so the branch targets are directly
comparable — unlike a comparison of two listings of a file, where the two
number the instructions differently.

Usage:
    thumb_enc_cmp.py [--triple thumbv7-linux-gnueabi] [--only-pattern re]
        reads the encodings from the input
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import thumb_ref as ref  # noqa: E402

DECODER = os.path.join(HERE, "..", "target", "release", "examples", "thumb_decode")


def ours(encodings, at=0):
    """What this decoder says about each encoding."""
    payload = "".join(f"{a:04x} {b:04x} {at:x}\n" for a, b in encodings)
    res = subprocess.run([DECODER], input=payload, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"the decoder failed: {res.stderr}")
    out = []
    for line in res.stdout.splitlines():
        parts = line.split(" ", 2)
        out.append(parts[2] if len(parts) > 2 else "")
    return out


def theirs(encodings, triple, at=0):
    """What the reference says about each encoding."""
    words = [((b << 16) | a) for a, b in encodings]
    table = ref.objdump_isolated_words32(words, "/tmp/occ-thumb-diff/enccmp")
    out = []
    for i in range(len(encodings)):
        text = table.get(ref.iso32_addr(i))
        out.append(text)
    return out


def clean(text):
    """The part of a reading that is the instruction."""
    if text is None:
        return None
    text = text.split("<")[0].strip()
    text = re.sub(r"\s*@\s*imm\s*=\s*#\S+", "", text).strip()
    return text.replace("\t", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triple", default="thumbv7-linux-gnueabi")
    ap.add_argument("--pattern", default=None, help="only show differences whose text matches")
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    encodings = []
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        a = int(parts[0], 16)
        b = int(parts[1], 16) if len(parts) > 1 else 0
        encodings.append((a, b))

    mine = ours(encodings)
    theirs_ = theirs(encodings, args.triple)
    same = 0
    shown = 0
    for (a, b), m, t in zip(encodings, mine, theirs_):
        c = clean(t)
        if c is None:
            # The reference does not define the encoding. The only thing
            # this decoder may say about it is that it is not an
            # instruction.
            if m is not None and "undefined" in m:
                same += 1
                continue
        elif c == m:
            same += 1
            continue
        if args.pattern and not re.search(args.pattern, m or ""):
            continue
        if shown >= args.limit:
            continue
        shown += 1
        print(f"  {a:04x} {b:04x}")
        print(f"    reference: {c!r}")
        print(f"    ours:      {m!r}")
    print(f"agree: {same}/{len(encodings)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
