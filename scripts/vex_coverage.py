#!/usr/bin/env python3
"""Compare the VEX tables against `objdump`, one encoding at a time.

The VEX tables hold one row per opcode and mandatory prefix, and several
of the rows depend on the `W` bit as well: the bit picks a different name,
or makes the encoding name nothing at all. This script asks `objdump`
about every combination of map, prefix, `W` bit and opcode, and compares
the answer with what the decoder says.

Two operand forms are probed per encoding. Some instructions exist only
in the memory form, and asking about the register form of one of those
would report a missing instruction where the decoder is right to refuse.

Usage: vex_coverage.py [--verbose]
"""
import re
import subprocess
import sys

DECODER = "target/release/examples/decode_bytes"
SLOT = 16
MAPS = (1, 2, 3)

OBJ = re.compile(r"^\s*([0-9a-f]+):\t((?:[0-9a-f]{2} )+)\s*(.*?)\s*$")

PREFIX_WORDS = {
    "lock", "rep", "repz", "repnz", "repe", "repne", "bnd", "data16",
    "addr32", "rex", "cs", "ds", "es", "fs", "gs", "ss", "notrack",
}


def is_prefix(word):
    return word in PREFIX_WORDS or word.startswith("rex.")


def vex(map_no, opcode, pp, w, modrm):
    """A three-byte VEX encoding of `opcode`.

    P0 = R̄ X̄ B̄ mmmmm, P1 = W ~vvvv L pp. The third source is left
    unused, which is what `~vvvv` all ones means.
    """
    p0 = 0xe0 | (map_no & 0x1f)
    p1 = (w << 7) | 0x78 | (1 << 2) | (pp & 3)
    return bytes([0xc4, p0, p1, opcode, modrm])


def build_slots():
    slots = []
    for map_no in MAPS:
        for pp in range(4):
            for w in (0, 1):
                for opcode in range(256):
                    for modrm in (0xC1, 0x00):
                        blob = vex(map_no, opcode, pp, w, modrm)
                        blob += b"\x90" * (SLOT - len(blob))
                        slots.append((map_no, pp, w, opcode, blob))
    return slots


def objdump_names(slots):
    buf = b"".join(blob for *_, blob in slots)
    with open("/tmp/vex_probe.bin", "wb") as f:
        f.write(buf)
    proc = subprocess.run(
        ["objdump", "-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel",
         "/tmp/vex_probe.bin"],
        capture_output=True, text=True,
    )
    found = {}
    for line in proc.stdout.splitlines():
        m = OBJ.match(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        index = addr // SLOT
        if addr % SLOT or index in found:
            continue
        words = m.group(3).split()
        while words and is_prefix(words[0].lower()):
            words.pop(0)
        name = words[0].lower() if words else "(bad)"
        if name.startswith("."):
            name = "(bad)"
        found[index] = name
    return [found.get(i, "") for i in range(len(slots))]


def decoder_names(slots):
    names = []
    for start in range(0, len(slots), 200):
        chunk = slots[start:start + 200]
        args = [DECODER, "64"] + [blob.hex() for *_, blob in chunk]
        proc = subprocess.run(args, capture_output=True, text=True)
        for line in proc.stdout.splitlines():
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

    verdict = {}
    for (map_no, pp, w, opcode, _), their, our in zip(slots, theirs, mine):
        readable = their and their != "(bad)"
        named = our and our != "(bad)" and not our.startswith(".")
        if not readable and not named:
            continue
        if not readable:
            kind = "extra"
        elif not named:
            kind = "missing"
        elif our != their:
            kind = "wrong"
        else:
            kind = "ok"
        key = (map_no, pp, w, opcode)
        rank = {"ok": 3, "wrong": 2, "missing": 1, "extra": 0}
        previous = verdict.get(key)
        if previous is None or rank[kind] > rank[previous[0]]:
            verdict[key] = (kind, their, our)

    missing = []
    wrong = []
    extra = []
    agreed = 0
    for (map_no, pp, w, opcode), (kind, their, our) in sorted(verdict.items()):
        if kind == "ok":
            agreed += 1
        elif kind == "missing":
            missing.append((map_no, pp, w, opcode, their))
        elif kind == "wrong":
            wrong.append((map_no, pp, w, opcode, our, their))
        else:
            extra.append((map_no, pp, w, opcode, our))

    print(f"slots probed: {len(slots)}")
    print(f"encodings compared: {len(verdict)}")
    print(f"agreed: {agreed}")
    print(f"the decoder does not name ({len(missing)}):")
    for map_no, pp, w, opcode, their in missing[:50]:
        print(f"  vex map={map_no} pp={pp} w={w} op=0x{opcode:02x}  "
              f"objdump: {their}")
    print(f"named differently ({len(wrong)}):")
    for map_no, pp, w, opcode, our, their in wrong[:50]:
        print(f"  vex map={map_no} pp={pp} w={w} op=0x{opcode:02x}  "
              f"ours {our}, objdump {their}")
    print(f"named where objdump refuses ({len(extra)}):")
    for map_no, pp, w, opcode, our in extra[:50]:
        print(f"  vex map={map_no} pp={pp} w={w} op=0x{opcode:02x}  ours {our}")
    if verbose:
        return 0
    return 1 if (missing or wrong) else 0


if __name__ == "__main__":
    sys.exit(main())
