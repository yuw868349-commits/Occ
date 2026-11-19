#!/usr/bin/env python3
"""The canonical form of a Thumb instruction, for the differential test.

The decoder in this project prints a particular spelling.  To compare it
with two references, both sides have to be reduced to the same shape:
lowercase mnemonics, registers as plain numbers, immediates as `#0x..`,
and branch destinations as the offset from the program counter.

The result is the *class* of an instruction: which operation it is and
which registers it touches.  Two readings that differ only in how they
spell a shift (`lsl #28` against `lsl #0x1c`) are the same instruction,
and the comparison must say so.

The one thing that has to be decided before anything else is what kind of
instruction this is, because a branch destination and an immediate are
both numbers and only one of them is an address.
"""
import re

# The aliases the two references use for the same condition.
CONDS = {
    "eq": "eq", "ne": "ne", "cs": "hs", "cc": "lo", "mi": "mi", "pl": "pl",
    "vs": "vs", "vc": "vc", "hi": "hi", "ls": "ls", "ge": "ge", "lt": "lt",
    "gt": "gt", "le": "le", "al": "", "nv": "nv",
}

# The banked names capstone uses.
BANK = {"sb": "r9", "sl": "r10", "fp": "r11", "ip": "r12"}

# Mnemonics whose last operand is a branch destination rather than a value.
BRANCHY = {"b", "bl", "blx", "bx", "cbz", "cbnz", "bra", "tbb", "tbh"}

# The 16-bit set writes the flag-setting form into the mnemonic; the
# 32-bit set writes it into a suffix.
FLAG_SPELLING = (("movs", "mov"), ("adds", "add"), ("subs", "sub"),
                 ("ands", "and"), ("eors", "eor"), ("bics", "bic"),
                 ("orrs", "orr"), ("adcs", "adc"), ("sbcs", "sbc"),
                 ("rsbs", "rsb"), ("muls", "mul"), ("lsls", "lsl"),
                 ("lsrs", "lsr"), ("asrs", "asr"), ("rors", "ror"))


def canonical(text, pc):
    """Reduce one reading to its class."""
    if text is None:
        return None
    t = text.strip()
    if not t or t.startswith("<unknown>") or t.startswith(".word"):
        return None
    # A reading that says the encoding is not an instruction is not a
    # reading of an instruction.  The two references say so with
    # `<unknown>`, and the decoder says so with a `.short` line that
    # names the halfword; both mean the same thing, and both must be
    # reduced to the same absence.
    if t.startswith(".short") or "; undefined" in t:
        return None

    # objdump prints its annotations after the instruction: a displacement
    # and, for a branch, the address it read.  Everything from the first
    # `@` on is annotation and goes; the destination that came before it
    # stays, because it has to be rewritten before the symbol name is
    # stripped.
    t = t.split("@")[0].strip()
    t = t.lower()
    t = t.replace("\t", " ").replace(",", " , ")

    parts = t.split()
    if not parts:
        return None
    head = parts[0]
    rest = parts[1:]

    # The width suffix comes off first.  A wide instruction is written
    # `beq.w`, and the condition is inside the name rather than after
    # it, so the name has to be stripped before the condition can be
    # read out of it.
    width = ""
    for tail in (".w", ".n"):
        if head.endswith(tail):
            width = tail
            head = head[: -len(tail)]
            break

    # Split the mnemonic into a base, a condition and a width suffix.
    # A condition can be written three ways: as its own word (`beq`), as a
    # suffix on the branch (`b.eq`), or as a prefix on the branch
    # (`beq`), which is what the 16-bit set uses.
    cond = ""
    base = head
    for suffix, name in CONDS.items():
        if base == suffix:
            cond = name
            base = "b"
            break
        if base.endswith("." + suffix):
            cond = name
            base = "b"
            break
        if base.startswith("b" + suffix) and len(base) == len(suffix) + 1:
            cond = name
            base = "b"
            break
        # `bhs` and `blo` are the two-letter spellings of the carry
        # conditions; they are written with the `c` in front.
        if base == "bhs":
            cond = "hs"
            base = "b"
            break
        if base == "blo":
            cond = "lo"
            base = "b"
            break

    # The flag-setting spelling.
    for old, new in FLAG_SPELLING:
        if base == old:
            base = new
            break
        if base.endswith("." + old):
            base = base[: -(len(old) + 1)] + "." + new
            break

    mn = base + width

    # Registers.
    def reg(m):
        return BANK.get(m.group(0), m.group(0))

    rest = [re.sub(r"\b(?:r\d+|sb|sl|fp|ip|sp|lr|pc)\b", reg, r) for r in rest]

    branchy = base in BRANCHY and base != "bx"
    if branchy:
        # A branch destination is an address; it is the only thing that has
        # to be made relative for the two sides to be comparable.
        def dest(m):
            v = int(m.group(1), 16)
            return f"<+{v - pc:+#x}>"

        rest = [re.sub(r"#?0x([0-9a-f]+)\b", dest, r) for r in rest]
    else:
        # Everything that looks like a number is an immediate.
        def num(m):
            s = m.group(0)
            body = s.lstrip("#")
            neg = body.startswith("-")
            if neg:
                body = body[1:]
            v = int(body, 16) if body.startswith("0x") else int(body, 10)
            return ("-" if neg else "") + f"#{v:#x}"

        rest = [re.sub(r"#-?(?:0x[0-9a-f]+|\d+)", num, r) for r in rest]

    # Now the symbol name can go.  The rewritten destination is written in
    # angle brackets too, so it is protected first.
    t = " ".join([mn] + rest)
    # The rewritten destination is written in angle brackets too, so the
    # whole of it is protected before the symbols are stripped.
    t = re.sub(r"<\+[^>]*>", lambda m: m.group(0).replace("<", "\x00", 1)
               .replace(">", "\x01", 1), t)
    t = re.sub(r"<[^>]*>", " ", t)
    t = t.replace("\x00", "<").replace("\x01", ">")
    # objdump prints the address it read after the symbol; drop it.
    t = re.sub(r"\s+0x[0-9a-f]+\s*$", "", t)

    # A register list is a set: its order does not carry meaning.
    def reglist(m):
        inner = m.group(1)
        names = re.findall(r"\b(?:r\d+|sp|lr|pc)\b", inner)
        return "{" + ",".join(sorted(names)) + "}"

    t = re.sub(r"\{([^}]*)\}", reglist, t)

    out = (cond + "|" if cond else "") + mn
    body = t[len(mn):].strip()
    if body:
        out += " " + body
    out = re.sub(r"\s+", " ", out).strip().replace(" , ", ",")
    # The symbol removal leaves the space that separated it behind.
    out = re.sub(r"\s*>$", "", out)
    return out
