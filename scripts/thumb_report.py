#!/usr/bin/env python3
"""Read the measured layout table and print what the decoder should do.

This turns the machine-readable measurement into something a person can
check against the architecture manual, so that a measurement which came
out wrong is visible as wrong rather than quietly built into the decoder.

Usage: thumb_report.py PATH [--only SUBSTRING]
"""
import argparse
import collections
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--only", default=None)
    ap.add_argument("--width", type=int, default=100)
    args = ap.parse_args()

    with open(args.path) as f:
        table = json.load(f)

    for name, entry in sorted(table.items(), key=lambda kv: -kv[1]["count"]):
        if args.only and args.only not in name:
            continue
        shape = entry["shape"]
        print()
        print(f"{shape[0]}  {' '.join(shape[1:])}   count={entry['count']}  "
              f"first=0x{entry['first']:04x}")

        # The fields, grouped by the operand they feed.
        fields = collections.defaultdict(list)
        selectors = {}
        for bit, info in entry["bits"].items():
            if "feeds" in info:
                k, role, w, c = info["feeds"][0]
                fields[(k, role)].append((int(bit), w))
            elif "names" in info:
                selectors[int(bit)] = info["names"]
        for (k, role) in sorted(fields):
            bits = sorted(fields[(k, role)])
            text = ", ".join(f"b{b}:{w}" for b, w in bits)
            print(f"    op{k} {role:5s} <- {text}")
        for b in sorted(selectors):
            print(f"    b{b:2d}  selector: {', '.join(sorted(selectors[b]))}")


if __name__ == "__main__":
    sys.exit(main())
