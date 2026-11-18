#!/usr/bin/env python3
"""Print a solved layout table in a form that can be read.

The table is a machine-readable record of what was measured.  This prints
it in a shape a person can check against the architecture manual, so that
a measurement that came out wrong is visible as wrong rather than quietly
built into the decoder.

Usage: thumb_show.py PATH [--only SUBSTRING]
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--only", default=None)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    with open(args.path) as f:
        table = json.load(f)

    for name, entry in sorted(table.items(), key=lambda kv: -kv[1]["count"]):
        if args.only and args.only not in name:
            continue
        shape = entry["shape"]
        mn = shape[0]
        ops = shape[1:]
        print()
        print(f"{mn}  {' '.join(ops)}   count={entry['count']}  first=0x{entry['first']:04x}")
        # The fields: which bit feeds which operand, and with what weight.
        fields = collections.defaultdict(dict)
        selectors = {}
        for bit, info in entry["bits"].items():
            if "feeds" in info:
                k, w, c = info["feeds"][0]
                fields[k][int(bit)] = w
            elif "names" in info:
                selectors[int(bit)] = info["names"]
        for k in sorted(fields):
            bits = sorted(fields[k].items())
            parts = ", ".join(f"b{b}:{w}" for b, w in bits)
            print(f"    op{k} <- {parts}")
        if selectors:
            for b in sorted(selectors):
                names = selectors[b]
                if len(names) == 1:
                    print(f"    bit{b} selects: {list(names)[0]}")
                else:
                    print(f"    bit{b} selects: " +
                          ", ".join(f"{n}" for n in names))


if __name__ == "__main__":
    import collections
    sys.exit(main())
