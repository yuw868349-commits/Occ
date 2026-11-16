#!/usr/bin/env python3
"""Ask both disassemblers what an encoding means, and say where they differ.

Two independent decoders are used throughout this project, and they are
used for different things:

* `llvm-objdump` knows every instruction, including the AVX-512 set and the
  BMI family, so it is the source the opcode tables are built from.

* GNU `objdump` is what the cross-check compares against, because that is
  the tool whose output a reader of the listing would compare with.

They do not always agree. GNU `objdump` 2.42 prints a placeholder instead of
the name for some of the 128-bit and 256-bit AVX-512 forms, and it rejects
some encodings that LLVM accepts. Where they differ, the difference has to
be understood rather than papered over, so this script prints both answers
side by side and lets the caller decide which one the decoder should follow.

The encodings go to the disassemblers in one pass, each in a slot of its
own. One encoding that cannot be decoded cannot shift the ones after it,
and the address of each decoded instruction says which slot it belongs to.

Usage: truth_probe.py <map> [--all]
        map 1, 2 or 3 selects the VEX opcode map; 4 selects EVEX.
"""
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llvm_disasm as llvm

SLOT = 16

# The names a disassembler prints for a form it cannot decode.
UNKNOWN = ("<unknown>", "(bad)", "")

OBJ = re.compile(r"^\s*([0-9a-f]+):\t((?:[0-9a-f]{2} )+)\s*\t(.*)$")
# objdump prints the bytes of a long instruction on a second line and
# nothing else there. The bytes belong to the instruction above.
CONT = re.compile(r"^\s*([0-9a-f]+):\t((?:[0-9a-f]{2} )+)\s*$")
# objdump prints the bytes of a long instruction on a second line and
# nothing else there. The bytes belong to the instruction above.
CONT = re.compile(r"^\s*([0-9a-f]+):\t((?:[0-9a-f]{2} )+)\s*$")


def gnu_batch(blobs):
    """What GNU objdump says about each blob, decoded in one pass.

    A disassembler that cannot decode an instruction does not always
    consume the whole of it, and then the instructions that follow are no
    longer on the boundaries between the blobs. Only the line that starts
    exactly at the beginning of a slot says what that slot holds, so the
    rest are thrown away.
    """
    return [text for text, _ in gnu_batch_len(blobs)]


def gnu_batch_len(blobs):
    """What GNU objdump says about each blob, with the length it consumed.

    The same rules as `gnu_batch`, but the number of bytes the instruction
    occupies is kept as well: a length is a fact of its own, and it is the
    fact that a wrong decoder gets wrong first.

    objdump prints an instruction of more than seven bytes on two lines,
    the second one holding nothing but bytes. Those bytes belong to the
    instruction above, so they are added to it.
    """
    if not blobs:
        return []
    lines = [".text"]
    for blob in blobs:
        lines.append(".align 16")
        lines.append(".byte " + ",".join(str(x) for x in blob))
    with tempfile.NamedTemporaryFile(suffix=".s", delete=False,
                                     mode="w") as f:
        f.write("\n".join(lines) + "\n")
        src = f.name
    obj = src[:-2] + ".o"
    try:
        build = subprocess.run(["as", "--64", src, "-o", obj],
                               capture_output=True, text=True)
        if build.returncode != 0:
            raise RuntimeError("as failed: " + build.stderr[:400])
        proc = subprocess.run(
            ["objdump", "-d", "-M", "intel", obj],
            capture_output=True, text=True)
    finally:
        for path in (src, obj):
            if os.path.exists(path):
                os.unlink(path)

    first = {}
    last = None
    for line in proc.stdout.splitlines():
        m = OBJ.match(line)
        if not m:
            # The bytes of an instruction that did not fit on one line
            # belong to the instruction above.
            c = CONT.match(line)
            if c and last is not None:
                prev_text, prev_len = first[last]
                first[last] = (prev_text, prev_len + len(c.group(2).split()))
            continue
        addr = int(m.group(1), 16)
        text = m.group(3).strip()
        length = len(m.group(2).split())
        index = addr // SLOT
        if addr % SLOT != 0 or index in first:
            last = None
            continue
        first[index] = (text, length)
        last = index
    return [first.get(i, ("", 0)) for i in range(len(blobs))]

def gnu_disassemble(blobs):
    """What GNU objdump says about each blob, exactly.

    The blobs are decoded in one pass, and the ones whose slot the
    disassembler lost track of are decoded again on their own: a blob
    decoded by itself cannot be shifted by its neighbours.
    """
    texts = gnu_batch(blobs)
    out = []
    for blob, text in zip(blobs, texts):
        if text:
            out.append(text)
        else:
            out.append(gnu_batch([blob])[0])
    return out


def gnu_disassemble_len(blobs):
    """What GNU objdump says about each blob, with the length it consumed.

    The same rules as `gnu_disassemble`, but the length is kept.
    """
    pairs = gnu_batch_len(blobs)
    out = []
    for blob, (text, length) in zip(blobs, pairs):
        if text:
            out.append((text, length))
        else:
            out.append(gnu_batch_len([blob])[0])
    return out


def mnemonic_of(text):
    """The name of the instruction in a disassembler's output."""
    if not text:
        return ""
    parts = text.split("\t")
    name = parts[0].strip().lower()
    if name in UNKNOWN:
        return ""
    # The rounding mode and the suppressed exceptions are decoration on the
    # name; they are not part of it.
    return name.split(" ")[0]


def build_vex(map_no, opcode, pp, vvvv=1, l=0, w=0, modrm=0xc1, two=False):
    """A VEX encoding."""
    if two:
        b1 = 0xe0 | ((~vvvv & 0x0f) << 3) | (l << 2) | (pp & 3)
        enc = bytes([0xc5, b1, opcode])
    else:
        b1 = 0xe0 | (map_no & 0x1f)
        b2 = (w << 7) | ((~vvvv & 0x0f) << 3) | (l << 2) | (pp & 3)
        enc = bytes([0xc4, b1, b2, opcode])
    if modrm is not None:
        enc += bytes([modrm])
    if map_no == 3:
        enc += b"\x00"
    return enc


def build_evex(map_no, opcode, pp, vvvv=1, ll=2, w=0, mask=0, z=0, bcast=0,
               modrm=0xc1):
    """An EVEX encoding.

    The three payload bytes are, measured against the assembler:

        P0 = R̄ X̄ B̄ R̄' 0 mmm
        P1 = W ~vvvv 1 pp
        P2 = z L'L b V̄' aaa

    Every extension bit is stored inverted, and the third operand is stored
    as the complement of its number.
    """
    p0 = 0xf0 | (map_no & 0x07)
    p1 = (w << 7) | ((~vvvv & 0x0f) << 3) | 0x04 | (pp & 3)
    p2 = (z << 7) | (ll << 5) | (bcast << 4) | 0x08 | (mask & 0x07)
    enc = bytes([0x62, p0, p1, p2, opcode, modrm])
    if map_no == 3:
        enc += b"\x00"
    return enc


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "1"
    map_no = int(which)
    show_all = "--all" in sys.argv

    if map_no == 4:
        jobs = []
        for em in (1, 2, 3):
            for opcode in range(256):
                for pp in range(4):
                    for w in (0, 1):
                        for ll in (0, 1, 2):
                            jobs.append((em, opcode, pp, w, ll))
        blobs = [build_evex(m, o, pp, ll=ll, w=w, modrm=0xc1)
                 for m, o, pp, w, ll in jobs]
    else:
        jobs = []
        for opcode in range(256):
            for pp in range(4):
                for w in (0, 1):
                    for l in (0, 1):
                        jobs.append((opcode, pp, w, l))
        blobs = [build_vex(map_no, o, pp, w=w, l=l, modrm=0xc1)
                 for o, pp, w, l in jobs]

    gnu = gnu_disassemble(blobs)
    ll = llvm.disassemble(blobs)

    diffs = 0
    for job, gtext, (ltext, _length) in zip(jobs, gnu, ll):
        gname = mnemonic_of(gtext)
        lname = mnemonic_of(ltext)
        if gname == lname:
            if show_all and gname:
                print(f"  {job} both={gname}")
            continue
        diffs += 1
        print(f"  {job}  gnu={gname or '(none)':<16} llvm={lname or '(none)'}")
    print(f"{diffs} difference(s) out of {len(jobs)}")


if __name__ == "__main__":
    main()
