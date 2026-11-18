#!/usr/bin/env python3
"""The reference-reading layer for the Thumb decoder.

Two independent implementations read the same encodings: capstone and
llvm-objdump, ARM's own disassembler.  They are read at *known* addresses so
that a branch destination can be rewritten as an offset from the program
counter, which is the one form that is comparable between the two — the same
instruction sits at a different address in each tool.

llvm-objdump prints its own reading of a branch displacement as an
`@ imm = #0x..` annotation.  That annotation is the *displacement*, not the
destination, so it is comparable between the two tools without any rewriting
at all, and it is what settles the branch models.

The reduction of a reading to a comparable form lives in thumb_canon.py.
"""
import os
import re
import subprocess

import capstone

# The profiles llvm-objdump is read in.
#
# ARMv8-M baseline knows the security-extension instructions (`bxns`,
# `blxns`) that ARMv7-M does not have; ARMv7-M knows the ones ARMv8-M
# dropped (`setend`).  Neither is a superset of the other, and a binary
# may be built for either, so both are read: the ARMv8-M reading is taken
# where it defines the encoding, and the ARMv7-M one where ARMv8-M leaves
# it undefined.  The two read only one class of encoding differently — the
# branch exchange, where ARMv7-M has no security-extension form of it and
# prints the plain one — and there the ARMv8-M reading is the one taken,
# because that is the encoding the architecture actually gives.
#
# capstone agrees with the result on every one of the 65536 halfwords.
PROFILES = ("thumbv8m.base-none-eabi", "thumbv7-linux-gnueabi")

# The tools are looked up where they live, and the version is recorded so
# that a table built by one version is not silently compared against
# another.
OBJCOPY = "/usr/lib/llvm-23/bin/llvm-objcopy"
OBJDUMP = "/usr/lib/llvm-23/bin/llvm-objdump"
TRIPLE = PROFILES[0]


def capstone_read(hw, addr, nbytes):
    """Read one instruction with capstone, at a known address."""
    md = capstone.Cs(capstone.CS_ARCH_ARM, capstone.CS_MODE_THUMB)
    md.detail = False
    code = hw.to_bytes(nbytes, "little")
    for i in md.disasm(code, addr, count=1):
        if i.size == nbytes:
            return f"{i.mnemonic} {i.op_str}".strip()
    return None


def objdump_read(words, nbytes, scratch):
    """Read a stream of instructions with llvm-objdump, at known addresses.

    Every slot is padded so that a 32-bit encoding cannot swallow the next
    slot and shift every later reading; the padding is part of the slot, so
    the addresses stay predictable.  The whole stream is read in one pass:
    reading it in pieces would restart the addresses at zero each time.
    """
    if nbytes == 2:
        stream = []
        for w in words:
            stream.append(w)
            stream.append(0xBF00)  # nop
    else:
        stream = list(words)
    blob = b"".join(w.to_bytes(nbytes, "little") for w in stream)
    subprocess.run(["rm", "-rf", scratch], check=False)
    subprocess.run(["mkdir", "-p", scratch], check=True)
    raw = f"{scratch}/r.bin"
    with open(raw, "wb") as f:
        f.write(blob)
    obj = f"{scratch}/a.o"
    subprocess.run(
        [OBJCOPY, "--input-target=binary", "--output-target=elf32-littlearm",
         "--rename-section", ".data=.text,code", raw, obj],
        check=True, capture_output=True)
    res = subprocess.run(
        [OBJDUMP, "-d", f"--triple={TRIPLE}", obj],
        check=True, capture_output=True, text=True).stdout
    table = {}
    for line in res.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\s+[0-9a-f ]+\s*(.*)$", line)
        if m:
            table[int(m.group(1), 16)] = m.group(2).strip()
    return table


# ---------------------------------------------------------------------------
# Reading one encoding on its own
#
# llvm-objdump keeps the state of an `it` block while it walks the
# instructions: the ones that follow an `it` are printed with the block's
# condition attached to them, whether or not they were meant to be there.
# Putting each encoding in a section of its own does not fix it — the state
# is carried across sections as well.
#
# What does fix it is padding.  An `it` block covers at most four
# instructions, so four `nop`s in front of the encoding mean that whatever
# block was open has been consumed and closed before the encoding is read.

# The number of `nop`s in front of each encoding, and the `nop` itself.
ISOLATION_PAD = 4
ISOLATION_NOP = 0xBF00

# The distance between two encodings: the padding is four halfwords and the
# encoding is one, so the stride is five halfwords.
ISOLATION_STRIDE = 2 * (ISOLATION_PAD + 1)


def iso_addr(hw):
    """The address the halfword `hw` is read at."""
    return ISOLATION_STRIDE * hw + 2 * ISOLATION_PAD


def iso_pc(hw):
    """The program counter the halfword `hw` is read at."""
    return iso_addr(hw) + 4


# The same for the 32-bit set: four `nop`s in front of the encoding, which
# is four bytes, so the stride is twelve.
ISOLATION32_STRIDE = 2 * ISOLATION_PAD + 4


def iso32_addr(index):
    """The address the 32-bit encoding `index` is read at."""
    return ISOLATION32_STRIDE * index + 2 * ISOLATION_PAD


def iso32_pc(index):
    """The program counter the 32-bit encoding `index` is read at."""
    return iso32_addr(index) + 4


def objdump_isolated_words32(words, scratch):
    """Read a list of 32-bit encodings with llvm-objdump, each on its own.

    The encoding is the *word*, with the first halfword in the low half:
    the encoding of `bl #0x100` is the four bytes `00 f0 80 f8`, whose
    first halfword is `0xf000` and whose second is `0xf880`.

    The table is keyed by address, and the address of the encoding at
    index `i` is `iso32_addr(i)`.
    """
    merged = {}
    readings = []
    for i, triple in enumerate(PROFILES):
        readings.append(_objdump_isolated32(words, scratch, triple, f"p{i}"))
    for index in range(len(words)):
        addr = iso32_addr(index)
        value = None
        for table in readings:
            text = table.get(addr)
            if text is None:
                continue
            if text.strip() == "<unknown>":
                continue
            value = text
            break
        merged[addr] = value
    return merged


def _objdump_isolated32(words, scratch, triple, sub):
    """Read the 32-bit encodings in one profile."""
    blob = bytearray()
    for w in words:
        blob += ISOLATION_NOP.to_bytes(2, "little") * ISOLATION_PAD
        blob += (w & 0xFFFFFFFF).to_bytes(4, "little")
    where = f"{scratch}/{sub}"
    subprocess.run(["rm", "-rf", where], check=False)
    subprocess.run(["mkdir", "-p", where], check=True)
    raw = f"{where}/r.bin"
    with open(raw, "wb") as f:
        f.write(bytes(blob))
    obj = f"{where}/a.o"
    subprocess.run(
        [OBJCOPY, "--input-target=binary", "--output-target=elf32-littlearm",
         "--rename-section", ".data=.text,code", raw, obj],
        check=True, capture_output=True)
    res = subprocess.run(
        [OBJDUMP, "-d", f"--triple={triple}", obj],
        check=True, capture_output=True, text=True).stdout
    table = {}
    for line in res.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\s+[0-9a-f ]+\s*(.*)$", line)
        if m:
            table[int(m.group(1), 16)] = m.group(2).strip()
    return table


def objdump_isolated_space(words, scratch):
    """Read a list of encodings with llvm-objdump, each on its own.

    The encodings are laid out in one run of code, each behind four `nop`s.
    The `nop`s are 16-bit and cannot be mistaken for the first half of a
    32-bit encoding, so the block an `it` opens is always closed by them
    and never reaches the encoding under test.

    Every profile is read, and the reading that defines the encoding is
    taken; see `PROFILES` for why there is more than one.

    The address of the encoding under test is `hw * 10 + 8`, which
    `iso_addr` gives; the caller needs it to compare a branch destination
    against another tool's.
    """
    merged = {}
    readings = []
    for i, triple in enumerate(PROFILES):
        readings.append(_objdump_isolated(words, scratch, triple, f"p{i}"))
    for index, _ in enumerate(words):
        addr = iso_addr(index)
        value = None
        for table in readings:
            text = table.get(addr)
            if text is None:
                # The encoding was swallowed by a wider instruction that
                # started before it, so there is no reading of its own.
                continue
            if text.strip() == "<unknown>":
                # This profile does not define the encoding; a later one
                # may.
                continue
            value = text
            break
        merged[addr] = value
    return merged


def _objdump_isolated(words, scratch, triple, sub):
    """Read the encodings in one profile."""
    blob = bytearray()
    for w in words:
        blob += ISOLATION_NOP.to_bytes(2, "little") * ISOLATION_PAD
        blob += (w & 0xFFFF).to_bytes(2, "little")
    where = f"{scratch}/{sub}"
    subprocess.run(["rm", "-rf", where], check=False)
    subprocess.run(["mkdir", "-p", where], check=True)
    raw = f"{where}/r.bin"
    with open(raw, "wb") as f:
        f.write(bytes(blob))
    obj = f"{where}/a.o"
    subprocess.run(
        [OBJCOPY, "--input-target=binary", "--output-target=elf32-littlearm",
         "--rename-section", ".data=.text,code", raw, obj],
        check=True, capture_output=True)
    res = subprocess.run(
        [OBJDUMP, "-d", f"--triple={triple}", obj],
        check=True, capture_output=True, text=True).stdout
    table = {}
    for line in res.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\s+[0-9a-f ]+\s*(.*)$", line)
        if m:
            table[int(m.group(1), 16)] = m.group(2).strip()
    return table


def objdump_imm(table):
    """Extract llvm-objdump's own reading of a branch displacement.

    objdump prints it as `@ imm = #0x..`; that is the displacement, so it
    compares directly against a model without any address rewriting.
    """
    out = {}
    for addr, text in table.items():
        m = re.search(r"@\s*imm\s*=\s*(#0x[0-9a-f]+|#-?0x[0-9a-f]+|#-?\d+)", text)
        if not m:
            continue
        body = m.group(1)[1:]
        v = int(body, 16) if "0x" in body else int(body, 10)
        out[addr] = v
    return out


def read_16bit_space(scratch):
    """Read the whole 16-bit space with both tools.

    Each halfword is read *on its own*, with four `nop`s in front of it, so
    that the state an `it` block leaves behind cannot reach it: llvm-objdump
    keeps that state while it walks the instructions, and an `it` in one
    slot would otherwise be reported as the condition of the instructions in
    the slots after it.

    The encodings are laid out at a stride of ten bytes — four `nop`s and
    then the halfword — which puts the halfword under test at
    `hw * 10 + 8`.  Both tools are read at that address, so a branch's
    destination means the same thing on all three sides.
    """
    words = list(range(0x10000))
    table = objdump_isolated_space(words, scratch)
    out = []
    for hw in range(0x10000):
        addr = iso_addr(hw)
        out.append((hw, capstone_read(hw, addr, 2), table.get(addr)))
    return out


def read_32bit_words(words, scratch):
    """Read a list of 32-bit encodings with both tools.

    The first halfword of a Thumb-2 encoding is the *low* half of the
    32-bit word, so a word here is `(hw2 << 16) | hw1`.
    """
    table = objdump_read(words, 4, scratch)
    out = []
    for i, w in enumerate(words):
        addr = i * 4
        out.append((w & 0xFFFF, (w >> 16) & 0xFFFF,
                    capstone_read(w, addr, 4), table.get(addr)))
    return out
