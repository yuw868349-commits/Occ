#!/usr/bin/env python3
"""Compare instruction boundaries with objdump on real code.

The decoder's job is to find where each instruction starts and ends. A single
byte of over- or under-read shifts everything after it, and the listing still
looks plausible, so the check has to be against an independent decoder rather
than against expectations.

Usage: x86_verify.py [binary ...]
"""
import re
import subprocess
import sys

DECODER = "target/debug/examples/disasm"

# objdump prints one line per instruction: "  401136:\tadd    eax,edx".
LINE = re.compile(r"^\s*([0-9a-f]+):\t")


def objdump(path):
    """Instruction addresses grouped by function, in address order."""
    proc = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", "-M", "intel", path],
        capture_output=True,
        text=True,
    )
    # objdump exits non-zero on a stripped image with nothing to disassemble; the
    # partial output is still usable.
    if proc.returncode != 0 and not proc.stdout:
        return []
    out = proc.stdout
    funcs, cur = [], None
    for line in out.splitlines():
        m = re.match(r"^([0-9a-f]+) <(.+)>:$", line.strip())
        if m:
            cur = (m.group(2), int(m.group(1), 16), [])
            funcs.append(cur)
            continue
        m = LINE.match(line)
        if m and cur is not None:
            cur[2].append(int(m.group(1), 16))
    return [f for f in funcs if f[2]]


def ours(path, start, count):
    out = subprocess.run(
        [DECODER, path, hex(start), str(count)],
        capture_output=True,
        text=True,
    )
    addrs = []
    for line in out.stdout.splitlines():
        m = re.match(r"^\s*([0-9a-f]+)\s", line)
        if m:
            addrs.append(int(m.group(1), 16))
    return addrs


def check(path, limit=400):
    funcs = objdump(path)
    bad = 0
    compared = 0
    insns = 0
    for name, entry, addrs in funcs[:limit]:
        want = addrs
        got = ours(path, entry, len(want) + 2)
        insns += len(want)
        compared += 1
        # Stop at the first address objdump has that we do not: after that the
        # streams have diverged and every later address is noise.
        for i, a in enumerate(want):
            if i >= len(got) or got[i] != a:
                print(f"  MISMATCH {name}+{i} at {a:#x}: decoder says "
                      f"{got[i] if i < len(got) else 'end'}")
                bad += 1
                break
    return compared, insns, bad


if __name__ == "__main__":
    paths = sys.argv[1:] or ["/bin/ls"]
    total_bad = 0
    for p in paths:
        n, ins, bad = check(p)
        total_bad += bad
        print(f"{p}: {n} function(s), {ins} instruction(s), {bad} mismatch(es)")
    print("all functions agree with objdump" if total_bad == 0 else f"{total_bad} mismatch(es)")
    sys.exit(1 if total_bad else 0)
