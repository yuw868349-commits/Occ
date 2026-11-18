#!/usr/bin/env python3
"""Print the mnemonic table of a range of first halfwords, compressed.

The rows of the table are runs of first halfwords that read the same way,
so the table is a list of ranges rather than a list of encodings.

Usage:
    thumb_mnem32.py 0xf000 0xf1ff --hw2 0x0123 | thumb_group.py
"""
import re
import sys

def main():
    prev = None
    start = None
    last = None
    rows = []

    def flush():
        if prev is not None:
            rows.append((start, last, prev))

    for line in sys.stdin:
        line = line.rstrip()
        if not line.startswith("0x"):
            continue
        addr = line.split()[0]
        m = re.search(r"'([a-z0-9_.]+)", line)
        m = m.group(1) if m else "None"
        if m != prev:
            flush()
            prev = m
            start = addr
        last = addr
    flush()
    for a, b, m in rows:
        print(f"{a}-{b}  {m}")


if __name__ == "__main__":
    main()
