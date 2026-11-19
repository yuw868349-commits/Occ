#!/usr/bin/env python3
"""Ask a disassembler what an encoding means, in bulk.

Two properties make this usable for building opcode tables:

* Every encoding is put in a slot of its own, wide enough that no
  instruction can run past it. One encoding that the disassembler does
  not recognise therefore cannot shift the ones after it, and the address
  of each decoded instruction says which slot it belongs to.

* The whole buffer is decoded in one pass. The tables have thousands of
  entries, and asking the disassembler about them one at a time would
  take hours.

The truth comes from `llvm-objdump`. Its AVX-512 support is complete;
an older GNU `objdump` prints a placeholder instead of the name for the
128-bit and 256-bit forms, and prints nothing at all for the mask
registers.
"""
import os
import re
import subprocess
import tempfile

# The slot width. No x86 instruction is longer than fifteen bytes, so a
# sixteen-byte slot cannot be overrun.
SLOT = 16

# The names the disassembler prints for a form it cannot decode.
UNKNOWN = ("<unknown>", "(bad)", "")

# One decoded instruction: the text, and the number of bytes it took.
LINE = re.compile(r"^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} )+)\s*\t(.*)$")


def disassemble(blobs):
    """The first instruction of every blob, in order.

    The result is a list of `(text, length)` pairs. An entry that the
    disassembler could not decode has an empty text and a zero length.
    """
    if not blobs:
        return []
    # The assembler is what turns a byte buffer into an object file the
    # disassembler will read, so the buffer becomes one `.byte` directive
    # per slot, each one aligned to a slot boundary.
    lines = [".text"]
    for blob in blobs:
        lines.append(".align 16")
        lines.append(".byte " + ",".join(str(b) for b in blob))
    with tempfile.NamedTemporaryFile(suffix=".s", delete=False,
                                     mode="w") as f:
        f.write("\n".join(lines) + "\n")
        src = f.name
    obj = src[:-2] + ".o"
    try:
        build = subprocess.run(
            ["llvm-mc", "-triple=x86_64-pc-linux-gnu", "-filetype=obj",
             src, "-o", obj],
            capture_output=True, text=True,
        )
        if build.returncode != 0:
            raise RuntimeError("llvm-mc failed: " + build.stderr[:400])
        proc = subprocess.run(
            ["llvm-objdump", "-d", "--x86-asm-syntax=intel", obj],
            capture_output=True, text=True,
        )
    finally:
        for path in (src, obj):
            if os.path.exists(path):
                os.unlink(path)

    first = {}
    for line in proc.stdout.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        index = int(m.group(1), 16) // SLOT
        if index in first:
            continue
        first[index] = (m.group(3).strip(), len(m.group(2).split()))
    return [first.get(i, ("", 0)) for i in range(len(blobs))]


def split_operands(operands):
    """The operands, split on the commas that are not inside brackets.

    The braces of a mask register are decoration: `zmm0{k1}{z}` names one
    operand, not three. The comment the disassembler appends is not an
    operand either.
    """
    if not operands:
        return []
    cleaned = re.sub(r"\{[^}]*\}", "", operands)
    cleaned = re.sub(r"\s*#.*$", "", cleaned)
    cleaned = cleaned.replace("{vex}", "")
    if not cleaned.strip():
        return []
    parts, depth, current = [], 0, []
    for ch in cleaned:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    parts.append("".join(current).strip())
    return [p for p in parts if p]


def operand_kind(text):
    """What an operand is: a vector register, a general-purpose one, a mask
    register, a memory reference, or an immediate."""
    t = text.strip().lower()
    if re.fullmatch(r"-?0x[0-9a-f]+|-?\d+", t):
        return "imm"
    if re.fullmatch(r"(xmm|ymm|zmm|mm)\d+", t):
        return "vec"
    if re.fullmatch(r"k\d+", t):
        return "mask"
    if "ptr" in t or "[" in t or "(" in t:
        return "mem"
    return "gp"


def describe(body):
    """The mnemonic, the operand kinds and the immediate flag.

    The result is `None` when the disassembler could not decode the
    encoding.
    """
    if not body:
        return None
    parts = body.split("\t")
    mnemonic = parts[0].strip().lower()
    if mnemonic in UNKNOWN:
        return None
    operands = parts[1] if len(parts) > 1 else ""
    # The rounding mode and the suppressed exceptions are decoration on
    # the mnemonic; they are not part of it.
    mnemonic = mnemonic.split(" ")[0]
    kinds = [operand_kind(p) for p in split_operands(operands)]
    has_imm = "imm" in kinds
    return mnemonic, kinds, has_imm, operands
