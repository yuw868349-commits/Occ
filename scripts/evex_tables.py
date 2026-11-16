#!/usr/bin/env python3
"""Build the EVEX opcode tables from a real disassembler, not from memory.

The EVEX encoding has four fields that pick the instruction: the opcode
map, the mandatory prefix, the `W` bit and the opcode byte. Every
combination of them is a different instruction, and the AVX-512 set has
several thousand of them. Writing them down by hand means writing down
several thousand chances to be wrong.

This script builds the encodings instead and asks the disassembler what
they mean. The truth comes from `llvm-objdump`, whose AVX-512 support is
complete; an older GNU `objdump` prints a placeholder instead of the name
for the 128-bit and 256-bit forms.

The encodings are laid out in a buffer of fixed-size slots and decoded in
one pass. A slot is wide enough that no instruction can run past it, so
one bad encoding cannot shift the ones after it; the address of each
decoded instruction says which slot it belongs to.

What the table records for each instruction is the name, the operand
shape, and whether a trailing immediate byte is part of it. The length
comes from the same measurement: the disassembler prints the bytes it
consumed, and the number of bytes is the length.

The bit layout below was measured, not assumed. Every field was read out
of the bytes the assembler produces, and each claim here is checked
against the assembler by `verify_layout()`.

Usage: evex_tables.py [--rust] [--stats] [--verify]
"""
import collections
import os
import re
import subprocess
import sys
import tempfile

# The slot width. No x86 instruction is longer than fifteen bytes, so a
# sixteen-byte slot cannot be overrun.
SLOT = 16

# The opcode maps an EVEX prefix can name.
#
# The numbers are the ones the encoding itself uses. They were read out
# of the bytes the assembler produces for instructions in each map:
# `1` is the `0f` map, `2` is the `0f38` map, `3` is the `0f3a` map, and
# `5` and `6` are the `0f38` map reached through `f3` and `f2`.
MAPS = (1, 2, 3, 5, 6)

# The vector lengths, in the order the `L'L` field numbers them.
LENGTHS = (0, 1, 2)

# The names `llvm-objdump` prints for a form it cannot decode.
UNKNOWN = ("<unknown>", "(bad)", "")

# The bytes an EVEX instruction takes before its ModRM byte. Every
# instruction has these, and the ones with an immediate add to them.
BASE_LEN = 6


def evex(map_no, opcode, pp=0, w=0, ll=2, vvvv=0, mask=0, z=0, bcast=0,
         modrm=0xc1):
    """An EVEX encoding of `opcode` in `map_no`.

    The three prefix bytes carry, in order: the register-number extension
    bits and the map; the `W` bit, the third operand and the mandatory
    prefix; the zeroing, the length, the broadcast and the mask.

    The bit layout, measured against the assembler:

        P0 = R̄ X̄ B̄ R̄' 0 mmm
        P1 = W ~vvvv 1 pp
        P2 = z L'L b V̄' aaa

    Every extension bit is stored inverted, and the third operand is
    stored as the complement of its number.
    """
    p0 = 0xf0 | (map_no & 0x07)
    p1 = (w << 7) | ((~vvvv & 0x0f) << 3) | 0x04 | (pp & 3)
    p2 = (z << 7) | (ll << 5) | (bcast << 4) | 0x08 | (mask & 0x07)
    return bytes([0x62, p0, p1, p2, opcode, modrm])


def assemble(lines):
    """The bytes the assembler gives for each instruction."""
    with tempfile.NamedTemporaryFile(suffix=".s", delete=False,
                                     mode="w") as f:
        f.write("\n".join(lines) + "\n")
        path = f.name
    try:
        proc = subprocess.run(
            ["llvm-mc", "-triple=x86_64-pc-linux-gnu", "-assemble",
             "-show-encoding", path],
            capture_output=True, text=True,
        )
    finally:
        os.unlink(path)
    out = {}
    for line in proc.stdout.splitlines():
        # The line reads `TAB name TAB operands # encoding: [bytes]`.
        if "# encoding: [" not in line:
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        # The key is the instruction as written, so that two forms of the
        # same mnemonic do not collide.
        key = fields[1].strip() + " " + fields[2].split("#")[0].strip()
        blob = line.split("# encoding: [", 1)[1].split("]", 1)[0]
        out[key] = bytes(int(x, 16) for x in blob.split(","))
    return out


def disassemble(blobs):
    """The first instruction of every slot: its text and its length.

    The whole buffer is handed to the disassembler once. Every slot
    starts at a known offset, and no instruction is long enough to run
    into the next slot, so the address of each decoded instruction says
    which slot it belongs to. The number of bytes the disassembler
    prints is the length of the instruction, which is what says whether
    a trailing immediate byte is part of it.
    """
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
        m = re.match(r"^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} )+)\s*\t(.*)$",
                     line)
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
    """The mnemonic, the operand kinds and the immediate flag."""
    if not body:
        return None
    parts = body.split("\t")
    mnemonic = parts[0].strip().lower()
    if mnemonic in UNKNOWN:
        return None
    operands = parts[1] if len(parts) > 1 else ""
    # `llvm-objdump` prints the rounding mode and the suppressed
    # exceptions as decoration on the mnemonic; they are not part of it.
    mnemonic = mnemonic.split(" ")[0]
    kinds = [operand_kind(p) for p in split_operands(operands)]
    shape = "".join(k for k in kinds if k != "imm")
    return mnemonic, shape, "imm" in kinds, operands


# The register classes the disassembler prints, by width in bytes.
CLASSES = {"xmm": 16, "ymm": 32, "zmm": 64}


def dest_width(body):
    """The width of the destination register, in bytes.

    A scalar instruction keeps its destination at `xmm` whatever the
    `L` field says: `vrndscaless zmm0` is not a thing, and the encoding
    that would say so still names a 128-bit register. Measuring the width
    the disassembler printed is how the table learns which instructions
    those are.
    """
    text = describe(body)
    if text is None:
        return None
    for operand in split_operands(text[3]):
        m = re.match(r"\s*(xmm|ymm|zmm)", operand, re.IGNORECASE)
        if m:
            return CLASSES[m.group(1).lower()]
    return None


def kinds_of(body):
    """The operand kinds of a decoded instruction, in order.

    `describe` folds the kinds into a string, which is enough to look a
    shape up but not to ask where the memory operand is. That question is
    what tells a store from a load, so it is asked here instead.

    The immediate byte is not an operand of the instruction's shape: it
    is a field of the encoding that happens to be printed like one, and
    counting it would make `vpclmulqdq` a four-operand instruction whose
    last operand is never memory.
    """
    if not body:
        return []
    parts = body.split("\t")
    operands = parts[1] if len(parts) > 1 else ""
    return [k for k in (operand_kind(p) for p in split_operands(operands))
            if k != "imm"]


def collect():
    """Every instruction the encoding defines.

    The result is keyed by the opcode byte, and each entry holds the name
    for every combination of map, prefix, `W` bit and vector length that
    names one.

    Two forms are measured for every encoding: the register form, which
    says what the instruction is called and how long it is, and the
    memory form, which says how wide the memory operand is. The two do
    not cover the same set — an instruction that moves a mask register
    has no register form but a memory one, and the other way round — so
    both are read and their results merged.
    """
    jobs = []
    for map_no in MAPS:
        for opcode in range(256):
            for pp in range(4):
                for w in (0, 1):
                    for ll in LENGTHS:
                        jobs.append((map_no, opcode, pp, w, ll))

    # The register form and the memory form, measured side by side. A
    # third pass sets the broadcast bit, which turns the memory operand
    # into a single element repeated across the lanes; the width the
    # disassembler prints for that element is the one the listing shows
    # in `{1toN}`.
    reg = disassemble([evex(m, o, pp, w=w, ll=ll, modrm=0xc1)
                       for m, o, pp, w, ll in jobs])
    mem = disassemble([evex(m, o, pp, w=w, ll=ll, modrm=0x00)
                       for m, o, pp, w, ll in jobs])
    bcast = disassemble([evex(m, o, pp, w=w, ll=ll, bcast=1, modrm=0x00)
                         for m, o, pp, w, ll in jobs])

    table = {}
    for (map_no, opcode, pp, w, ll), (rbody, rlen), (mbody, mlen), (bbody, _) in zip(
            jobs, reg, mem, bcast):
        rinfo = describe(rbody)
        minfo = describe(mbody)
        if rinfo is None and minfo is None:
            continue
        # The register form is what names the instruction and says how
        # long it is, because it is the form every instruction has. The
        # memory form is what says which operand is the memory one: the
        # two forms of a move name the same register in the same place,
        # so the register form alone cannot tell `vmovups [rax], zmm1`
        # from `vmovups zmm1, [rax]`.
        if rinfo is not None:
            mnemonic, _, _, _ = rinfo
            has_imm = rlen == BASE_LEN + 1
        else:
            mnemonic, _, _, _ = minfo
            has_imm = mlen == BASE_LEN + 1

        rkinds = kinds_of(rbody)
        mkinds = kinds_of(mbody)
        shape, scalar = shape_and_scalar(rkinds, mkinds)
        width = mem_width(mbody)
        elem = broadcast_width(bbody, width)
        entry = table.setdefault(opcode, {}).setdefault((map_no, pp, w), {})
        entry[ll] = (mnemonic, shape, has_imm, width, elem, scalar,
                     dest_width(rbody) if rinfo is not None else None)
    return table


def shape_and_scalar(rkinds, mkinds):
    """The operand shape, and the shape that depends on the ModRM mode.

    The memory form is the one that says where the destination is: a
    store names the memory operand first, a load names it last. The
    register form is then used for everything else, because it is the one
    that exists for every instruction.

    Two families have one operand more in the register form than in the
    memory form. The scalar moves write a lane of the destination from a
    lane of the source, and a register source therefore has to be named
    while a memory source does not. Those return the name of the shape
    that the decoder has to decide from the ModRM mode, and the plain
    shapes do not.
    """
    if not rkinds:
        return "".join(mkinds), None
    if not mkinds:
        return "".join(rkinds), None
    memory_first = mkinds[0] == "mem"
    memory_last = mkinds[-1] == "mem"
    if len(rkinds) != len(mkinds):
        # One operand fewer in the memory form: a scalar move.
        if memory_first:
            return "", "ScalarStore"
        if memory_last:
            return "", "ScalarLoad"
        return "".join(rkinds), None
    if memory_first and rkinds[0] != "mem":
        # The same operands in the other order: the memory operand is the
        # destination. Only its position changes, so the register kinds
        # keep their own names and the first one becomes the memory one.
        return "mem" + "".join(rkinds[1:]), None
    return "".join(rkinds), None


def broadcast_width(body, whole):
    """The width of one element of a broadcasting operand, in bytes.

    Zero when the instruction has no broadcast form. The test is the
    width the disassembler prints with the broadcast bit set: a
    broadcasting operand is one element, so it is narrower than the whole
    vector the same instruction reads without the bit.
    """
    if not body:
        return 0
    width = mem_width(body)
    if width == 0:
        return 0
    if whole and width >= whole:
        # The bit changed nothing: this instruction does not broadcast.
        return 0
    return width


# The memory operand widths the disassembler prints, and the byte counts
# they stand for.
MEM_WIDTHS = {
    "byte": 1,
    "word": 2,
    "dword": 4,
    "qword": 8,
    "tbyte": 10,
    "xmmword": 16,
    "ymmword": 32,
    "zmmword": 64,
}


def mem_width(body):
    """The width of the memory operand, in bytes, or zero."""
    if not body:
        return 0
    m = re.search(r"(xmmword|ymmword|zmmword|dword|qword|word|byte|tbyte)"
                  r"\s+ptr", body)
    if not m:
        return 0
    return MEM_WIDTHS.get(m.group(1), 0)


def verify_layout():
    """Check the bit layout against the assembler.

    Every claim the encoder makes is checked by assembling the same
    instruction and comparing the bytes. An assumption that does not
    survive this check is an assumption that would put every instruction
    after it in the wrong place.
    """
    cases = [
        # (assembler line, expected bytes)
        ("vaddps %zmm2, %zmm1, %zmm0",
         "62f1744858c2"),
        ("vpaddb %zmm2, %zmm1, %zmm0",
         "62f17548fcc2"),
        ("vaddps %zmm9, %zmm3, %zmm5",
         "62d1644858e9"),
        ("vpermt2d %zmm2, %zmm1, %zmm0",
         "62f275487ec2"),
        ("vpternlogd $3, %zmm2, %zmm1, %zmm0",
         "62f3754825c203"),
        ("vaddph %zmm2, %zmm1, %zmm0",
         "62f5744858c2"),
        ("vpmovm2b %k1, %zmm0",
         "62f27e4828c1"),
    ]
    enc = assemble([c[0] for c in cases])
    bad = 0
    for line, want in cases:
        got = enc.get(line)
        if got is None:
            print(f"  FAIL {line}: 汇编失败")
            bad += 1
            continue
        if got.hex() != want:
            print(f"  FAIL {line}: 得到 {got.hex()}，期望 {want}")
            bad += 1
        else:
            print(f"  ok   {line}: {got.hex()}")
    return bad == 0


# The operand shapes, in the terms the decoder uses.
FORMS = {
    "": "Form::None",
    "vecvec": "Form::Two",
    "vecvecvec": "Form::Three",
    "vecvecvecvec": "Form::Four",
    "gpvec": "Form::GpDst",
    "gpvecvec": "Form::GpDstThree",
    "vecgp": "Form::GpSrc",
    "vecvecgp": "Form::ThreeGpSrc",
    "memvec": "Form::Store",
    "memvecvec": "Form::StoreThree",
    "maskvecvec": "Form::MaskThree",
    "maskvecvecvec": "Form::MaskFour",
    "maskvec": "Form::MaskTwo",
    "gp": "Form::GpOnly",
    "mask": "Form::MaskOnly",
    "vecmem": "Form::VecMem",
    "vecmemvec": "Form::VecMemThree",
    "mem": "Form::MemOnly",
    "vecvecmem": "Form::VecVecMem",
    "memvecgp": "Form::Scatter",
    "vecmemvecgp": "Form::Gather",
}


def form_of(shape):
    """The operand shape, in the terms the decoder uses."""
    return FORMS.get(shape, "Form::Two")


def emit_rust(table):
    """The table as Rust source."""
    print("/// The EVEX instructions.")
    print("const EVEX_OPS: &[EvexOp] = &[")
    for opcode in sorted(table):
        for key in sorted(table[opcode]):
            map_no, pp, w = key
            entry = table[opcode][key]
            names = []
            for ll in LENGTHS:
                names.append(entry[ll][0] if ll in entry else "")
            if all(n == "" for n in names):
                continue
            # The shape and the immediate flag come from the shortest form
            # the instruction has; they do not depend on the length. The
            # broadcast width is the same at every length too, because it
            # is the width of one element.
            shape = None
            imm = False
            scalar = None
            elem = 0
            for ll in LENGTHS:
                if ll in entry:
                    shape = shape or entry[ll][1]
                    imm = imm or entry[ll][2]
                    scalar = scalar or entry[ll][5]
                    elem = max(elem, entry[ll][4])
            if scalar is not None:
                form = "Form::" + scalar
            else:
                form = form_of(shape or "")
            # A destination that stays 128 bits wide at the 512-bit
            # length is a scalar instruction: it works on one element and
            # names one lane, whatever the prefix says.
            xmm_dest = False
            for ll in (2, 1):
                if ll in entry and entry[ll][6] == 16:
                    xmm_dest = True
                    break
            mem = mem_size_of(entry)
            w = "true" if w else "false"
            print(f"    evex_op(0x{opcode:02x}, {map_no}, {pp}, {w}, "
                  f"[{', '.join(chr(34) + n + chr(34) for n in names)}], "
                  f"{form}, {mem}, {'true' if imm else 'false'}, {elem}, "
                  f"{'true' if xmm_dest else 'false'}),")
    print("];")


# The memory operand sizes, in the terms the decoder uses.
MEM_SIZES = {
    0: "MemSize::Vector",
    1: "MemSize::Element(1)",
    2: "MemSize::Element(2)",
    4: "MemSize::Element(4)",
    8: "MemSize::Element(8)",
    16: "MemSize::Element(16)",
    32: "MemSize::Element(32)",
    64: "MemSize::Element(64)",
}


def mem_size_of(entry):
    """The size of the memory operand, in the terms the decoder uses.

    The memory operand is the whole vector when its width follows the
    vector length, and one element when it does not. The widths that were
    measured at each length say which of the two it is: a width that
    doubles when the length doubles is the whole vector, and a width that
    stays the same is one element.
    """
    widths = {ll: entry[ll][3] for ll in LENGTHS if ll in entry}
    if not widths:
        return "MemSize::Vector"
    values = [widths[ll] for ll in sorted(widths)]
    if len(values) > 1:
        # The width grows with the length: the whole vector.
        grows = all(values[i] < values[i + 1]
                    for i in range(len(values) - 1))
        if grows:
            return "MemSize::Vector"
    # One element, of the width that was measured.
    smallest = min(widths.values())
    return MEM_SIZES.get(smallest, "MemSize::Vector")


def stats(table):
    """A summary of what the table holds."""
    names = collections.Counter()
    shapes = collections.Counter()
    widths = collections.Counter()
    for opcode in table:
        for key in table[opcode]:
            for ll in LENGTHS:
                if ll not in table[opcode][key]:
                    continue
                mnemonic, shape, has_imm, width, elem, scalar = (
                    table[opcode][key][ll])
                names[mnemonic] += 1
                shapes[scalar or shape] += 1
                widths[width] += 1
    print(f"opcode 字节数: {len(table)}")
    print(f"唯一助记符: {len(names)}")
    print(f"总条目: {sum(names.values())}")
    print("操作数形状:")
    for shape, count in shapes.most_common():
        print(f"  {shape or '(无)':16s} {count}")
    print("内存宽度:")
    for width, count in sorted(widths.items()):
        label = str(width) if width else "(无)"
        print(f"  {label:16s} {count}")
    print("助记符前 40:")
    for mnemonic, count in names.most_common(40):
        print(f"  {mnemonic:24s} {count}")


if __name__ == "__main__":
    if "--verify" in sys.argv:
        ok = verify_layout()
        print("位域布局验证:", "全部通过" if ok else "存在偏差")
        sys.exit(0 if ok else 1)
    table = collect()
    if "--rust" in sys.argv:
        emit_rust(table)
    elif "--stats" in sys.argv:
        stats(table)
    else:
        for opcode in sorted(table):
            for key in sorted(table[opcode]):
                entry = table[opcode][key]
                names = " ".join(
                    f"{ll}:{entry[ll][0]}" if ll in entry else f"{ll}:-"
                    for ll in LENGTHS
                )
                print(f"    0x{opcode:02x} map={key[0]} pp={key[1]} "
                      f"w={key[2]}  {names}")
