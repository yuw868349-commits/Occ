#!/usr/bin/env python3
"""Compare the decoder's reading of the T32 `0xFA` class with a reference.

Both files hold one line per encoding, in the shape

    0xfa03 0xf000  lsl.w r0, r3, r0

The two are compared line by line.  A difference in the name of the
instruction is reported separately from a difference in its operands,
because the first says the encoding was read as another instruction and
the second says it was read as the right one with the wrong register.

The two disassemblers do not always write the same thing for the same
instruction.  One of them calls the tenth register `r10` and the
other calls it `sl`, and one of them writes the undefined encoding as
`<undefined>` and the other writes it as the word it could not read.
Those are differences of spelling, not of meaning, so they are put
right before the two readings are compared: what is left is a
difference about what the instruction does.

Usage:
    t32_fa_diff.py ours.txt reference.txt
"""
import re
import sys

# The names the two disassemblers use for the same register.
ALIASES = {
    "r10": "sl",
    "r11": "fp",
    "r12": "ip",
    "r13": "sp",
    "r14": "lr",
    "r15": "pc",
}

# The two ways of writing an encoding that is not an instruction.
UNDEFINED = re.compile(r"^(\.word .*; )?undefined$")


def read(path):
    """The readings in a file, keyed by the encoding."""
    out = {}
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            key, _, text = line.partition("  ")
            out[key.strip()] = text.strip()
    return out


def normalise(text):
    """Put a reading into a shape that can be compared with another.

    The register aliases are put back to their names, the note that
    the reference appends to an unpredictable encoding is taken off,
    and an encoding that is not an instruction is given one name.
    """
    text = text.split("@")[0].strip()
    if UNDEFINED.match(text):
        return "<undefined>"
    parts = text.replace("\t", " ").split(None, 1)
    name = parts[0] if parts else ""
    operands = parts[1] if len(parts) > 1 else ""
    operands = operands.replace(",", " , ")
    for canonical, alias in ALIASES.items():
        operands = re.sub(rf"\b{canonical}\b", alias, operands)
    operands = re.sub(r"\s+", " ", operands).replace(" ,", ",").strip()
    return f"{name} {operands}".strip()


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ours = read(sys.argv[1])
    theirs = read(sys.argv[2])
    if set(ours) != set(theirs):
        missing = set(theirs) - set(ours)
        extra = set(ours) - set(theirs)
        print(f"the files do not cover the same encodings: "
              f"{len(missing)} missing, {len(extra)} extra")
        for key in sorted(missing)[:10]:
            print(f"  missing {key}")
        for key in sorted(extra)[:10]:
            print(f"  extra   {key}")
        return 1

    name_diff = []
    operand_diff = []
    for key in sorted(ours):
        a = normalise(ours[key])
        b = normalise(theirs[key])
        if a == b:
            continue
        a_parts = a.split(None, 1)
        b_parts = b.split(None, 1)
        if a_parts[0] != b_parts[0]:
            name_diff.append((key, ours[key], theirs[key]))
        else:
            operand_diff.append((key, ours[key], theirs[key]))

    total = len(ours)
    print(f"{total} encodings compared")
    print(f"{len(name_diff)} differ in the name of the instruction")
    print(f"{len(operand_diff)} differ in the operands")
    for key, a, b in name_diff[:30]:
        print(f"  {key}  ours: {a!r}  reference: {b!r}")
    for key, a, b in operand_diff[:30]:
        print(f"  {key}  ours: {a!r}  reference: {b!r}")
    return 1 if (name_diff or operand_diff) else 0


if __name__ == "__main__":
    sys.exit(main())
