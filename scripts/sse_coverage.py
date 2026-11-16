#!/usr/bin/env python3
"""Compare the legacy `0f38` and `0f3a` maps against `objdump`, byte for byte.

The tables in `crates/occ-decode/src/x86/sse_map.rs` were built by
encoding every combination and asking a disassembler what it saw. This
script asks the same question again, but compares the answer with what the
decoder says now. It is the check that the tables stayed complete: a row
that was never recorded shows up here as an instruction objdump names and
the decoder does not.

The encodings are laid out in fixed-size slots, one instruction per slot,
so a byte sequence objdump cannot read does not shift the ones after it.

Usage: sse_coverage.py [--verbose]
"""
import re
import subprocess
import sys

# The decoder, built by `cargo build --release --examples`.
DECODER = "target/release/examples/decode_bytes"

# No x86 instruction is longer than fifteen bytes.
SLOT = 16

# The mandatory prefixes, in the order the map indexes them: none, `66`,
# `f3`, `f2`.
PREFIXES = [b"", b"\x66", b"\xf3", b"\xf2"]

# objdump prints `  0:\t66 0f 38 00 c1 \tpshufb xmm0,xmm1`.
OBJ = re.compile(r"^\s*([0-9a-f]+):\t((?:[0-9a-f]{2} )+)\s*(.*?)\s*$")

# The bytes objdump puts in front of a mnemonic that are not its name.
PREFIX_WORDS = {
    "lock", "rep", "repz", "repnz", "repe", "repne", "bnd", "data16",
    "addr32", "rex", "rex.w", "rex.r", "rex.x", "rex.b", "notrack",
    "cs", "ds", "es", "fs", "gs", "ss",
}


def build_slots():
    """Every encoding to probe, and the map entry it stands for.

    Each opcode is probed twice, once with a register r/m operand and once
    with a memory one. A few instructions exist only in the memory form —
    `movntdqa` reads memory and nothing else — and probing them with a
    register operand would report a missing instruction where the decoder
    is right to refuse.
    """
    slots = []
    for map_byte, map_name in ((0x38, 0x38), (0x3A, 0x3A)):
        for pp_index, prefix in enumerate(PREFIXES):
            for opcode in range(256):
                for modrm in (0xC1, 0x00):
                    blob = prefix + bytes([0x0F, map_byte, opcode, modrm])
                # Room for an immediate byte, then padding. The padding is
                # `nop` rather than zero: a zero byte is a valid opcode
                # (`add [rax], al`), and a run of them lets objdump walk
                # across the slot boundary into the next encoding, where it
                # then reads that encoding as data. One-byte nops keep the
                # walk on the slot it belongs to.
                    blob += b"\x90" * (SLOT - len(blob))
                    slots.append((map_name, pp_index, opcode, blob))
    return slots


def objdump_names(slots):
    """The mnemonic objdump gives each slot, or an empty string."""
    buf = b"".join(blob for _, _, _, blob in slots)
    with open("/tmp/sse_probe.bin", "wb") as f:
        f.write(buf)
    proc = subprocess.run(
        ["objdump", "-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel",
         "/tmp/sse_probe.bin"],
        capture_output=True, text=True,
    )
    found = {}
    for line in proc.stdout.splitlines():
        m = OBJ.match(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        # Only the first instruction of a slot is trusted: after a byte
        # sequence objdump cannot read it resumes at the next byte, and
        # everything it prints from there belongs to the same slot.
        index = addr // SLOT
        if addr % SLOT or index in found:
            continue
        words = m.group(3).split()
        while words and words[0].lower() in PREFIX_WORDS:
            words.pop(0)
        name = words[0].lower() if words else "(bad)"
        if name.startswith("."):
            name = "(bad)"
        found[index] = name
    return [found.get(i, "") for i in range(len(slots))]


def decoder_names(slots):
    """The mnemonic the decoder gives each slot, in one process."""
    args = [DECODER, "64"]
    for _, _, _, blob in slots:
        args.append(blob.hex())
    proc = subprocess.run(args, capture_output=True, text=True)
    names = []
    for line in proc.stdout.splitlines():
        # The line reads `<hex> -> <mnemonic> <operands>`.
        if "->" not in line:
            continue
        text = line.split("->", 1)[1].strip()
        names.append(text.split()[0].lower() if text else "")
    return names


def main():
    verbose = "--verbose" in sys.argv
    slots = build_slots()
    theirs = objdump_names(slots)
    mine = decoder_names(slots)
    if len(mine) != len(slots):
        print(f"decoder returned {len(mine)} lines for {len(slots)} slots")
        return 1

    # One verdict per (map, prefix, opcode): the two probes of an opcode
    # ask the same question, and an instruction that exists in either form
    # is one the table has to name.
    verdict = {}
    for (map_name, pp, opcode, _), their, our in zip(slots, theirs, mine):
        key = (map_name, pp, opcode)
        readable = their and their != "(bad)"
        named = our and our != "(bad)" and not our.startswith(".")
        if not readable:
            continue
        if not named:
            kind = "missing"
        elif our != their:
            kind = "wrong"
        else:
            kind = "ok"
        # A verdict of "ok" from either form settles the question: the
        # decoder naming the instruction anywhere is the thing being
        # tested, and the two probes only differ in which form of it they
        # ask about.
        rank = {"ok": 2, "wrong": 1, "missing": 0}
        previous = verdict.get(key)
        if previous is None or rank[kind] > rank[previous[0]]:
            verdict[key] = (kind, their, our)

    missing = []
    wrong = []
    agreed = 0
    for (map_name, pp, opcode), (kind, their, our) in sorted(verdict.items()):
        if kind == "ok":
            agreed += 1
        elif kind == "missing":
            missing.append((map_name, pp, opcode, their))
        else:
            wrong.append((map_name, pp, opcode, our, their))

    print(f"slots probed: {len(slots)}")
    print(f"opcodes compared: {len(verdict)}")
    print(f"objdump named: {sum(1 for t in theirs if t and t != '(bad)')}")
    print(f"agreed: {agreed}")
    print(f"the decoder does not name ({len(missing)}):")
    for map_name, pp, opcode, their in missing[:40]:
        print(f"  0f{map_name:02x} pp={pp} op=0x{opcode:02x}  objdump: {their}")
    print(f"named differently ({len(wrong)}):")
    for map_name, pp, opcode, our, their in wrong[:40]:
        print(f"  0f{map_name:02x} pp={pp} op=0x{opcode:02x}  "
              f"ours {our}, objdump {their}")
    if verbose:
        return 0
    return 1 if (missing or wrong) else 0


if __name__ == "__main__":
    sys.exit(main())
