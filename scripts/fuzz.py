#!/usr/bin/env python3
"""Mutation fuzzer for the container parsers.

The mutator lives in `crates/occ-ffi/examples/fuzz_probe.rs` because doing it
here would mean a file write per iteration, and a file write costs more than the
parse it is trying to test. This script owns the parts Python is good at:
choosing seeds, sizing batches, surviving a hang, and saying what happened.

What counts as a failure:

  * a panic (the process would print `panicked` and exit 101),
  * an abort (exit 134),
  * a segfault (exit 139),
  * a batch that does not finish inside the timeout — which is a hang until
    proven otherwise, so it is reported, not retried quietly.

A rejected input is not a failure. The parser is supposed to refuse most of
what this produces; that is the bounds checking working.

Usage:

    python3 scripts/fuzz.py                     # 10 million load-depth runs
    python3 scripts/fuzz.py --total 2000000
    python3 scripts/fuzz.py --depth deep --total 20000
    python3 scripts/fuzz.py --seeds DIR --out DIR
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(ROOT, "target", "release", "examples", "fuzz_probe")


def macho_seed():
    """A minimal Mach-O 64 executable, so the Mach-O parser gets a real header.

    Written here rather than checked in: it is 104 bytes of magic numbers, and a
    binary blob in the repository would need a paragraph explaining it. The code
    is the explanation.
    """
    header = struct.pack(
        "<IiiIIIII",
        0xFEEDFACF,  # MH_MAGIC_64
        0x01000007,  # CPU_TYPE_X86_64
        3,  # CPU_SUBTYPE_X86_64_ALL
        2,  # MH_EXECUTE
        1,  # ncmds
        72,  # sizeofcmds
        0,  # flags
        0,  # reserved
    )
    segname = b"__TEXT" + b"\0" * 10
    segment = struct.pack(
        "<II16sQQQQiiII",
        0x19,  # LC_SEGMENT_64
        72,  # cmdsize
        segname,
        0x100000000,  # vmaddr
        0x1000,  # vmsize
        0,  # fileoff
        0x1000,  # filesize
        7,  # maxprot
        5,  # initprot
        0,  # nsects
        0,  # flags
    )
    return header + segment + b"\x90" * 64


def wasm_seed():
    """A minimal module with one empty section, if the project has no fixture."""
    return b"\0asm\x01\0\0\0" + b"\x00\x01\x00"


def elf_seed():
    """A tiny ELF header, so a fuzz run works on a machine with no /bin/true."""
    # e_ident, ET_EXEC, EM_X86_64, version, entry, phoff, shoff, flags, ehsize
    ident = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    return ident + struct.pack("<HHIQQQIHHHHHH", 2, 62, 1, 0x400000, 0, 0, 0, 64, 56, 0, 64, 0, 0)


def collect_seeds(dest, extra=None):
    """Gather one seed per container family.

    The five families are the ones `probe_format` can return: ELF, PE, Mach-O,
    WASM and `ar`. A fuzzer that only sees ELF is testing one parser and calling
    it five.
    """
    os.makedirs(dest, exist_ok=True)
    written = []

    def put(name, data):
        path = os.path.join(dest, name)
        with open(path, "wb") as f:
            f.write(data)
        written.append(path)

    put("seed.macho", macho_seed())
    put("seed.wasm", wasm_seed())
    put("seed.elf", elf_seed())

    fixtures = os.path.join(ROOT, "tests", "fixtures")
    if os.path.isdir(fixtures):
        for name in sorted(os.listdir(fixtures)):
            src = os.path.join(fixtures, name)
            if not os.path.isfile(src):
                continue
            size = os.path.getsize(src)
            if size == 0 or size > 400_000:
                continue
            dst = os.path.join(dest, "fixture-" + name.replace(os.sep, "_"))
            shutil.copyfile(src, dst)
            written.append(dst)

    for path in ("/bin/true", "/bin/sh"):
        if os.path.isfile(path):
            dst = os.path.join(dest, "system-" + os.path.basename(path))
            shutil.copyfile(path, dst)
            written.append(dst)

    # An `ar` archive: the member table is a second parser behind the first, and
    # it is the one with the most arithmetic on offsets.
    for path in (
        "/usr/lib/x86_64-linux-gnu/libz.a",
        "/usr/lib/x86_64-linux-gnu/libm.a",
        "/usr/lib/x86_64-linux-gnu/libc.a",
    ):
        if os.path.isfile(path) and os.path.getsize(path) < 3_000_000:
            dst = os.path.join(dest, "archive-" + os.path.basename(path))
            shutil.copyfile(path, dst)
            written.append(dst)
            break

    if extra:
        for path in extra:
            if os.path.isfile(path):
                dst = os.path.join(dest, "extra-" + os.path.basename(path))
                shutil.copyfile(path, dst)
                written.append(dst)

    return written


def run_batch(seeds, iterations, depth, rng, out_dir, start_at, timeout):
    cmd = [
        PROBE,
        "--seeds",
        seeds,
        "--iterations",
        str(iterations),
        "--depth",
        depth,
        "--rng",
        str(rng),
        "--out",
        out_dir,
        "--start-at",
        str(start_at),
    ]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {
            "timeout": True,
            "seconds": time.time() - started,
            "runs": iterations,
            "start_at": start_at,
        }
    elapsed = time.time() - started
    if proc.returncode != 0:
        return {
            "crashed": True,
            "code": proc.returncode,
            "seconds": elapsed,
            "stderr": proc.stderr[-2000:],
            "start_at": start_at,
            "runs": iterations,
        }
    line = ""
    for candidate in reversed(proc.stdout.strip().splitlines()):
        if candidate.startswith("{"):
            line = candidate
            break
    if not line:
        return {"crashed": True, "code": 0, "seconds": elapsed, "stderr": "no JSON report"}
    report = json.loads(line)
    report["seconds"] = elapsed
    return report


def main():
    ap = argparse.ArgumentParser(description="mutation fuzzer for occ's container parsers")
    ap.add_argument("--total", type=int, default=10_000_000, help="total iterations")
    ap.add_argument("--batch", type=int, default=200_000, help="iterations per probe process")
    ap.add_argument("--depth", choices=("load", "deep"), default="load")
    ap.add_argument("--seeds", help="use this directory instead of collecting one")
    ap.add_argument(
        "--out", default=os.path.join(tempfile.gettempdir(), ".occ-fuzz-failures")
    )
    ap.add_argument("--seed-file", type=int, default=0x0CC0FFEE, help="RNG seed")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per batch")
    ap.add_argument("--max-seed-bytes", type=int, default=512 * 1024)
    ap.add_argument("--keep-seeds", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(PROBE):
        print(
            "fuzz.py: build the probe first: cargo build --release -p occ-ffi --example fuzz_probe",
            file=sys.stderr,
        )
        return 2

    tmp = None
    seeds = args.seeds
    if not seeds:
        # The directory is hidden and carries a name of its own, so that
        # cleaning the temporary directory of the project does not delete the
        # seeds while the fuzzer is running.
        tmp = tempfile.mkdtemp(prefix=".occ-fuzz-seeds-")
        seeds = tmp
        got = collect_seeds(seeds)
        print(f"seeds: {len(got)} file(s) in {seeds}", file=sys.stderr)
        for path in got:
            print(f"  {os.path.basename(path):32} {os.path.getsize(path):>10} B", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    total = args.total
    done = 0
    loaded = 0
    rejected = 0
    failures = 0
    crashed = 0
    timeouts = 0
    seconds = 0.0
    rng = args.seed_file
    started = time.time()
    problems = []

    try:
        while done < total:
            n = min(args.batch, total - done)
            report = run_batch(
                seeds, n, args.depth, rng, args.out, done, args.timeout
            )
            seconds += report.get("seconds", 0.0)

            if report.get("timeout"):
                timeouts += 1
                problems.append(f"timeout after {report['seconds']:.0f}s at start-at={done}")
                # Narrow the window instead of giving up: the hang is inside
                # some sub-range, and finding which one is the whole point.
                if n > 1:
                    print(
                        f"  timeout at {done}..{done + n}; narrowing", file=sys.stderr
                    )
                    total_batches = n
                    args.batch = max(1, n // 10)
                    # Advance by the narrowed batch so the same range is retried.
                    total = min(total, done + args.batch)
                    continue
                done += n
                continue

            if report.get("crashed"):
                crashed += 1
                problems.append(
                    f"probe exited {report.get('code')} at start-at={done}: "
                    f"{report.get('stderr', '')[:300]}"
                )
                done += n
                rng = (rng * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
                continue

            loaded += report.get("loaded", 0)
            rejected += report.get("rejected", 0)
            new = report.get("failures", 0)
            failures += new
            done += report.get("runs", n)
            rng = (rng * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF

            rate = done / max(0.001, time.time() - started)
            print(
                f"  {done:>12,} / {total:,} · {loaded:,} loaded · {rejected:,} rejected · "
                f"{failures} panic · {rate:,.0f}/s",
                file=sys.stderr,
            )
            if new:
                print(f"    {new} new failure(s); artefacts in {args.out}", file=sys.stderr)

            if report.get("first_panic"):
                problems.append(f"panic: {report['first_panic']}")
    finally:
        if tmp and not args.keep_seeds:
            shutil.rmtree(tmp, ignore_errors=True)

    wall = time.time() - started
    print()
    print(f"runs          {done:,}")
    print(f"loaded        {loaded:,}")
    print(f"rejected      {rejected:,}")
    print(f"panics        {failures}")
    print(f"crashes       {crashed}")
    print(f"timeouts      {timeouts}")
    print(f"wall          {wall:.1f}s ({done / max(0.001, wall):,.0f}/s)")

    seen = []
    for p in problems:
        if p not in seen:
            seen.append(p)
    if seen:
        print("\nproblems")
        for p in seen[:20]:
            print(f"  {p}")
    if failures or crashed or timeouts:
        print(f"\nartefacts: {args.out}")
        return 1
    print("\nno panic, no crash, no hang")
    return 0


if __name__ == "__main__":
    sys.exit(main())
