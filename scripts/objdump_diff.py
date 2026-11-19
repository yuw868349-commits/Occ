#!/usr/bin/env python3
"""Cross-check the decoder against objdump, instruction by instruction.

Two things are checked, and they are not equally important:

* **Length.** Every instruction must occupy exactly the bytes objdump says it
  does. A single byte of over- or under-read moves every following instruction
  and turns the whole listing into noise. This is the hard check: any
  difference here is a bug.

* **Mnemonic.** The instruction at an address must be the same instruction.
  A wrong mnemonic with the right length is a wrong answer, but it does not
  break the stream, so it is reported separately and does not stop the run.

Usage: objdump_diff.py [binary ...]
"""
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter

DECODER = "target/release/examples/disasm_all"

# objdump prints "  401136:\t89 d8                \tmov    ebx,edi".
OBJ = re.compile(r"^\s*([0-9a-f]+):\t((?:[0-9a-f]{2} )+)\s*(.*?)\s*$")

# Prefixes objdump prints in front of the mnemonic. They are part of the
# instruction's encoding, not of its name.
#
# The segment registers are here because objdump prints them in front of the
# name whenever it cannot fold them into a memory operand: `cs nop WORD PTR
# [rax]` carries a segment override that changes nothing about the
# instruction. `notrack` is the Intel CET prefix for an indirect branch.
PREFIXES = {
    "lock", "rep", "repz", "repnz", "repe", "repne", "bnd", "data16",
    "addr32", "rex", "cs", "ds", "es", "fs", "gs", "ss", "notrack",
}


def is_prefix(word):
    """Whether the word objdump printed is a prefix, not a name.

    objdump prints the REX byte as `rex.WXB` when the instruction does not
    read all of its bits, and the set of letters depends on the
    instruction. Every one of them is the same prefix, so the test is on
    the stem.
    """
    return word in PREFIXES or word.startswith("rex.")

# objdump gives some instructions an alias that names a different operation
# from the one the encoding defines. `66 90` is `xchg ax,ax`, which is the
# two-byte `nop`; the decoder calls it what it is. The two names are the
# same instruction, so the difference is counted on its own rather than as a
# failure.
ALIASES = {
    ("nop", "xchg"),
    # The string instructions: objdump puts the element width in the
    # operand and keeps one name for all of them, the decoder names each
    # width. The operands say the same thing either way.
    ("movsb", "movs"),
    ("stosb", "stos"),
    ("lodsb", "lods"),
    ("scasb", "scas"),
    ("cmpsb", "cmps"),
    # The stack-pointer-width forms: in 64-bit mode the push and pop of the
    # flags register move 64 bits and the manuals call them `pushfq` and
    # `popfq`, while objdump keeps the 32-bit spelling.
    ("pushfq", "pushf"),
    ("popfq", "popf"),
}


def objdump(path):
    """(address, length, mnemonic) for every instruction objdump decoded."""
    proc = subprocess.run(
        ["objdump", "-d", "-M", "intel", path],
        capture_output=True,
        text=True,
    )
    out = []
    for line in proc.stdout.splitlines():
        m = OBJ.match(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        length = len(m.group(2).split())
        words = m.group(3).split()
        if not words:
            # objdump wraps the bytes of a long instruction onto a second
            # line and prints nothing but bytes there. The bytes belong to
            # the instruction above, so they belong to its length.
            if out:
                prev_addr, prev_len, prev_mnemonic = out[-1]
                out[-1] = (prev_addr, prev_len + length, prev_mnemonic)
            continue
        while words and is_prefix(words[0].lower()):
            words.pop(0)
        mnemonic = words[0].lower() if words else "(bad)"
        # A `.byte` line is objdump admitting that it cannot decode the
        # instruction. It says nothing about the name of the instruction, so
        # it cannot be compared.
        if mnemonic.startswith("."):
            continue
        out.append((addr, length, mnemonic))
    return out


def ours(path, count):
    """(address, length, mnemonic) for our own decode of the same text.

    The `--symbols` flag makes the decoder restart at every function entry
    the symbol table names, which is what `objdump` does. Without it the two
    sides disagree about where the padding between two functions ends and
    the code begins, and every instruction after that is compared at the
    wrong address.
    """
    proc = subprocess.run(
        [DECODER, path, "--symbols", "--tsv"],
        capture_output=True,
        text=True,
    )
    out = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        mnemonic = parts[2].lower()
        # A byte the decoder cannot read is a byte objdump cannot read
        # either. The two spellings name the same fact, and the one we use
        # is the one a disassembly listing shows.
        if mnemonic.startswith("."):
            mnemonic = "(bad)"
        out.append((int(parts[0], 16), int(parts[1]), mnemonic))
    return out


def is_relocatable(path):
    """Whether the file is an ET_REL object.

    A relocatable object has no load address: `sh_addr` is zero for every
    section, and `objdump` therefore restarts its address column at zero for
    each section. The decoder gives each section its file offset instead, so
    that addresses stay unique and a symbol's `st_value` can be rebased onto
    one. The two listings describe the same instructions in the same order but
    do not agree on addresses, so they are compared by position.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(20)
    except OSError:
        return False
    return (
        head[:4] == b"\x7fELF"
        and len(head) >= 18
        and int.from_bytes(head[16:18], "little") == 1
    )


def check_in_order(theirs, mine):
    """Compare two listings position by position, ignoring addresses."""
    length_bad = 0
    mnemonic_bad = 0
    alias_bad = 0
    compared = 0
    len_examples = []
    mne_examples = []
    len_hist = Counter()
    mne_hist = Counter()

    for i in range(max(len(theirs), len(mine))):
        if i >= len(theirs):
            length_bad += 1
            len_hist["extra"] += 1
            if len(len_examples) < 12:
                len_examples.append(f"#{i}: decoded past the listing ({mine[i][2]})")
            continue
        if i >= len(mine):
            length_bad += 1
            len_hist["missing"] += 1
            if len(len_examples) < 12:
                len_examples.append(f"#{i}: not decoded (objdump: {theirs[i][2]})")
            continue
        ta, tl, tm = theirs[i]
        ma, ml, mm = mine[i]
        compared += 1
        if tl != ml:
            length_bad += 1
            len_hist[f"{ml}!={tl}"] += 1
            if len(len_examples) < 12:
                len_examples.append(
                    f"#{i} {ta:#x}: ours {ml} byte(s), objdump {tl} ({tm} / {mm})"
                )
            continue
        if mm != tm:
            if (mm, tm) in ALIASES:
                alias_bad += 1
                continue
            mnemonic_bad += 1
            mne_hist[f"{mm}!={tm}"] += 1
            if len(mne_examples) < 12:
                mne_examples.append(f"#{i} {ta:#x}: ours {mm}, objdump {tm}")

    return (compared, length_bad, mnemonic_bad, alias_bad, len_hist, mne_hist,
            len_examples, mne_examples)


def check(path):
    theirs = objdump(path)
    if not theirs:
        return 0, 0, 0, 0, Counter(), Counter(), [], []
    mine = ours(path, len(theirs) * 2)
    if is_relocatable(path):
        return check_in_order(theirs, mine)
    by_addr = {a: (l, m) for a, l, m in mine}

    length_bad = 0
    mnemonic_bad = 0
    alias_bad = 0
    compared = 0
    len_examples = []
    mne_examples = []
    len_hist = Counter()
    mne_hist = Counter()

    for addr, length, mnemonic in theirs:
        got = by_addr.get(addr)
        if got is None:
            # We did not decode an instruction here at all. That is a length
            # bug somewhere upstream, so count it and move on.
            length_bad += 1
            len_hist["missing"] += 1
            if len(len_examples) < 12:
                len_examples.append(f"{addr:#x}: not decoded (objdump: {mnemonic})")
            continue
        compared += 1
        our_len, our_mnemonic = got
        if our_len != length:
            length_bad += 1
            len_hist[f"{our_len}!={length}"] += 1
            if len(len_examples) < 12:
                len_examples.append(
                    f"{addr:#x}: ours {our_len} byte(s), objdump {length} "
                    f"({mnemonic} / {our_mnemonic})"
                )
            continue
        if our_mnemonic != mnemonic:
            if (our_mnemonic, mnemonic) in ALIASES:
                alias_bad += 1
                continue
            mnemonic_bad += 1
            mne_hist[f"{our_mnemonic}!={mnemonic}"] += 1
            if len(mne_examples) < 12:
                mne_examples.append(
                    f"{addr:#x}: ours {our_mnemonic}, objdump {mnemonic}"
                )
    return (compared, length_bad, mnemonic_bad, alias_bad, len_hist, mne_hist,
            len_examples, mne_examples)


def archive_members(path):
    """Every member of an `ar` archive, or `None` when the file is not one.

    `objdump -d` walks an archive's members and restarts its addresses at zero
    for each, so comparing an archive as if it were one image compares every
    member against the wrong address. Writing each member out and checking it
    on its own puts both sides on the same bytes.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != b"!<arch>\n":
        return None

    out = []
    long_names = b""
    pos = 8
    while pos + 60 <= len(data):
        head = data[pos:pos + 60]
        raw = head[0:16].decode("latin-1").rstrip()
        try:
            size = int(head[48:58].decode("latin-1").strip() or "0")
        except ValueError:
            break
        body = data[pos + 60:pos + 60 + size]
        pos += 60 + size
        if pos & 1:
            pos += 1
        if raw == "/":
            continue
        if raw == "//":
            long_names = body
            continue
        if raw.startswith("/") and raw[1:].isdigit():
            rest = long_names[int(raw[1:]):]
            end = len(rest)
            for i, ch in enumerate(rest):
                if ch in b"/\n":
                    end = i
                    break
            name = rest[:end].decode("latin-1")
        elif raw.startswith("#1/"):
            n = int(raw[3:])
            name = body[:n].decode("latin-1").rstrip("\0")
            body = body[n:]
        else:
            name = raw.rstrip("/")
        out.append((name, body))
    return out


def check_member(body):
    """Check one member's bytes, which live in a temporary file."""
    with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as fh:
        fh.write(body)
        tmp = fh.name
    try:
        return check(tmp)
    finally:
        os.unlink(tmp)


if __name__ == "__main__":
    paths = sys.argv[1:] or ["/bin/ls"]
    worst = 0

    def report(label, compared, lbad, mbad, abad, lhist, mhist, lex, mex):
        print(f"{label}: {compared} instruction(s) compared")
        print(f"  length mismatches: {lbad}")
        print(f"  mnemonic mismatches: {mbad}")
        if abad:
            print(f"  alias differences (the same instruction, another name): {abad}")
        for line in lex:
            print(f"    len  {line}")
        for line in mex:
            print(f"    mne  {line}")
        if lhist:
            top = lhist.most_common(8)
            print("  length histogram: " + ", ".join(f"{k} x{v}" for k, v in top))
        if mhist:
            top = mhist.most_common(8)
            print("  mnemonic histogram: " + ", ".join(f"{k} x{v}" for k, v in top))

    for p in paths:
        members = archive_members(p)
        if members is None:
            compared, lbad, mbad, abad, lhist, mhist, lex, mex = check(p)
            report(p, compared, lbad, mbad, abad, lhist, mhist, lex, mex)
            worst += lbad
            continue
        # An archive is checked member by member, and the totals are summed:
        # one broken member is a broken archive, and the report says which.
        compared = lbad = mbad = abad = 0
        lhist, mhist = Counter(), Counter()
        lex, mex = [], []
        for name, body in members:
            if not body:
                continue
            c, l, m, a, lh, mh, lx, mx = check_member(body)
            compared += c
            lbad += l
            mbad += m
            abad += a
            lhist.update(lh)
            mhist.update(mh)
            for line in lx:
                if len(lex) < 12:
                    lex.append(f"{name}: {line}")
            for line in mx:
                if len(mex) < 12:
                    mex.append(f"{name}: {line}")
        print(f"{p}: {len(members)} member(s)")
        report("  total", compared, lbad, mbad, abad, lhist, mhist, lex, mex)
        worst += lbad
    if worst:
        print(f"{worst} length mismatch(es)")
        sys.exit(1)
    print("every instruction has the length objdump says")
