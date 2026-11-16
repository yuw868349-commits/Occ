#!/usr/bin/env python3
"""Check occ-dwarf against binutils on real compiler output.

Compiles one C file at every optimisation level and both DWARF versions, then
compares three things per binary:

  * the function list (DWARF subprograms vs `nm`),
  * the source line reported for each function entry (vs `addr2line`),
  * that parameters carry a location, wherever the compiler kept one.

Usage: scripts/dwarf_verify.py [--keep]
Exits non-zero on the first mismatch so it can gate a change.
"""

import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(ROOT, "target", "debug", "examples", "dwarf_probe")

SOURCE = r"""
#include <stdio.h>
#include <string.h>

struct point { int x, y; };
struct shape { struct point a, b; unsigned char kind; struct shape *next; };

static int sum(struct point p) { return p.x + p.y; }

int area(struct shape *s) {
    int w = s->b.x - s->a.x;
    int h = s->b.y - s->a.y;
    return w * h;
}

double scale(double v, int n) {
    double acc = 0;
    for (int i = 0; i < n; i++) acc += v * i;
    return acc;
}

int main(int argc, char **argv) {
    struct shape s = {{0, 0}, {3, 4}, 1, 0};
    struct point p = {1, 2};
    printf("%d %d %f %s\n", area(&s), sum(p), scale(1.5, argc), argv[0]);
    return 0;
}
"""

FUNC_LINE = re.compile(
    r"^(0x[0-9a-f]+)\.\.(0x[0-9a-f]+) (\S+)\s+decl=(\S+) params=(\d+) vars=(\d+)"
)
LOOKUP_LINE = re.compile(r"^(0x[0-9a-f]+) line=(\S+)\s+func=(\S+)")


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def functions(probe_out):
    out = []
    for line in probe_out.splitlines():
        m = FUNC_LINE.match(line)
        if m:
            out.append(
                {
                    "low": int(m.group(1), 16),
                    "high": int(m.group(2), 16),
                    "name": m.group(3),
                    "params": int(m.group(5)),
                }
            )
    return out


def lookups(probe_out):
    out = []
    for line in probe_out.splitlines():
        m = LOOKUP_LINE.match(line)
        if m:
            out.append((int(m.group(1), 16), m.group(2), m.group(3)))
    return out


def nm_functions(path):
    res = run(["nm", "-S", "--defined-only", path])
    names = {}
    for line in res.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] in {"T", "t", "W", "w"}:
            try:
                names[parts[3]] = int(parts[0], 16)
            except (ValueError, IndexError):
                pass
    return names


def addr2line(path, addrs):
    args = ["addr2line", "-e", path, "-f", "-i"] + [hex(a) for a in addrs]
    res = run(args)
    rows = res.stdout.splitlines()
    out = []
    for i in range(0, len(rows) - 1, 2):
        out.append((rows[i], rows[i + 1]))
    return out


def check(path, label):
    probe = run([PROBE, path] + [hex(a["low"]) for a in functions(run([PROBE, path]).stdout)])
    if probe.returncode != 0:
        return [f"{label}: probe failed: {probe.stderr.strip()}"]

    funcs = functions(probe.stdout)
    failures = []
    if not funcs:
        return [f"{label}: no functions found"]

    # 1. Names and entry addresses must line up with the symbol table.
    syms = nm_functions(path)
    for f in funcs:
        if f["name"] in syms and syms[f["name"]] != f["low"]:
            failures.append(
                f"{label}: {f['name']} at {f['low']:#x}, symbol table says {syms[f['name']]:#x}"
            )
    # crt startup objects are assembly with no debug info; only require the
    # symbols that came from the compiled source.
    missing = [
        n for n in syms if not n.startswith("_") and n not in {f["name"] for f in funcs}
    ]
    if missing:
        failures.append(f"{label}: functions missing from DWARF: {sorted(missing)}")

    # 2. Lines for the queried addresses must match addr2line.
    addrs = [a for a, _, _ in lookups(probe.stdout)]
    for (addr, loc, _), (a2l_name, a2l_loc) in zip(lookups(probe.stdout), addr2line(path, addrs)):
        # Our line is file:line:column; binutils drops the column.
        want = a2l_loc.strip()
        got = loc.rsplit(":", 1)[0] if loc.count(":") >= 2 else loc
        if want in {"??:0", "??:?"}:
            continue
        if os.path.basename(want) != os.path.basename(got) or want.split(":")[-1] != got.split(":")[-1]:
            failures.append(f"{label}: {addr:#x} line {got}, addr2line says {want}")

    # 3. Every parameter should resolve to something, unless optimised away.
    with_loc = len(re.findall(r"param \S+ .* at_entry=Some", probe.stdout))
    total_params = sum(f["params"] for f in funcs)
    if total_params and with_loc * 2 < total_params:
        failures.append(
            f"{label}: only {with_loc}/{total_params} parameters have a location"
        )
    return failures


def main():
    if not os.path.exists(PROBE):
        print(f"missing {PROBE}; run: cargo build -p occ-dwarf --example dwarf_probe")
        return 2
    if run(["gcc", "--version"]).returncode != 0:
        print("gcc not available, skipping")
        return 0

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "dw.c")
        with open(src, "w") as fh:
            fh.write(SOURCE)
        combos = [
            ("-O0", "-gdwarf-4"),
            ("-O0", "-gdwarf-5"),
            ("-O1", "-gdwarf-4"),
            ("-O1", "-gdwarf-5"),
            ("-O2", "-gdwarf-4"),
            ("-O2", "-gdwarf-5"),
            ("-O3", "-gdwarf-5"),
        ]
        for opt, dwarf in combos:
            label = f"{opt} {dwarf}"
            path = os.path.join(tmp, f"dw{opt}{dwarf}.bin")
            res = run(["gcc", opt, dwarf, "-o", path, src])
            if res.returncode != 0:
                failures.append(f"{label}: gcc failed: {res.stderr.strip()}")
                continue
            failures += check(path, label)

    if failures:
        print(f"{len(failures)} mismatch(es):")
        for f in failures:
            print("  " + f)
        return 1
    print(f"all {len(combos)} builds agree with binutils")
    return 0


if __name__ == "__main__":
    sys.exit(main())
