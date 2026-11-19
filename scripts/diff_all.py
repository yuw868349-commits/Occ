#!/usr/bin/env python3
"""Run the cross-check over every binary on the system.

One binary is not a proof. The point of this is to find the instruction
family that one binary does not contain, so the list has to be as wide as
the machine allows.
"""
import os
import subprocess
import sys

DIFF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "objdump_diff.py")


def candidates():
    """The binaries worth checking: the ones large enough to hold code."""
    seen = set()
    roots = ["/bin", "/sbin", "/usr/bin", "/usr/sbin", "/usr/lib",
             "/usr/libexec", "/lib"]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                path = os.path.join(dirpath, name)
                if path in seen:
                    continue
                seen.add(path)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if st.st_size < 4096:
                    continue
                if not os.access(path, os.R_OK):
                    continue
                yield path


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    bad = 0
    total = 0
    checked = 0
    for path in candidates():
        proc = subprocess.run(
            [sys.executable, DIFF, path],
            capture_output=True, text=True,
        )
        out = proc.stdout
        if "instruction(s) compared" not in out:
            continue
        checked += 1
        total += 1
        lines = out.splitlines()
        head = lines[0]
        lbad = 0
        mbad = 0
        for line in lines:
            if "length mismatches:" in line:
                lbad = int(line.split(":")[1])
            if "mnemonic mismatches:" in line:
                mbad = int(line.split(":")[1])
        if lbad or mbad:
            bad += 1
            print(f"FAIL {path}")
            for line in lines[1:16]:
                print("   ", line)
            if bad >= limit:
                print("... stopping")
                break
    print(f"checked {checked} binaries, {bad} with differences")


if __name__ == "__main__":
    main()
