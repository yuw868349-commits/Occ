#!/usr/bin/env python3
"""Print the measured layout of one 32-bit Thumb class, readably.

The measurement table is a machine-readable record of what each bit of an
encoding feeds.  This prints it in the form the decoder is written from:
which bits hold the destination, the source, the immediate, and what the
bits that choose between forms choose between.

Usage:
    thumb_layout32.py "and.w r r r"
    thumb_layout32.py --list "add"
"""
import argparse
import json
import sys

TABLE = "/tmp/occ-thumb-table/m32.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shape", nargs="?", default="")
    ap.add_argument("--list", default="")
    ap.add_argument("--table", default=TABLE)
    args = ap.parse_args()
    m = json.load(open(args.table))

    if args.list:
        for shape, e in sorted(m.items(), key=lambda kv: -kv[1]["count"]):
            if args.list in shape:
                print(f"{e['count']:5d}  first=0x{e['first']:04x}  {shape}")
        return 0

    hits = [(s, e) for s, e in m.items() if s == args.shape]
    if not hits:
        print(f"no class named {args.shape!r}; candidates:")
        for shape, e in sorted(m.items(), key=lambda kv: -kv[1]["count"])[:20]:
            print(f"  {e['count']:5d}  {shape}")
        return 1
    for shape, e in hits:
        print(f"--- {shape}   count={e['count']}  first=0x{e['first']:04x}")
        feeds = {}
        names = {}
        forms = {}
        for bit, b in e["bits"].items():
            if "feeds" in b:
                for k, role, w, c in b["feeds"]:
                    feeds.setdefault((k, role), []).append((int(bit), w))
            elif "names" in b:
                names[int(bit)] = b["names"]
            else:
                forms[int(bit)] = b["form"]
        for (k, role), bs in sorted(feeds.items()):
            bs.sort()
            desc = ", ".join(f"bit{b}" for b, w in bs)
            print(f"    operand {k} ({role}): {desc}")
        for bit, n in sorted(names.items()):
            print(f"    bit {bit}: chooses between {sorted(n)}")
        for bit, c in sorted(forms.items()):
            print(f"    bit {bit}: changes the form ({c})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
