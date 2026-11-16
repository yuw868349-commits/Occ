#!/usr/bin/env python3
"""Build the gather and scatter tables from a real disassembler.

The gather and scatter instructions are the only ones whose memory
operand is addressed by a vector register rather than by a general-purpose
one. That changes three things the other tables cannot express: the index
register is named at the width of a vector register, the data register is
not the same width as the index one, and the encoding's `L` field gives
the wider of the two rather than the data one.

The tables in `vector_tables.rs` and `evex_tables.rs` are built by
sweeping every opcode with a memory operand that has no SIB byte, which is
exactly the form a VSIB instruction cannot be encoded in: with no index
register there is nothing for the encoding to extend. Those sweeps
therefore report the whole family as naming no instruction, and this
script measures it separately, with a SIB byte present.

What is measured for each encoding:

* the mnemonic, which the `W` bit changes;
* the width of one element of the memory operand, which the listing
  prints as `DWORD PTR` or `QWORD PTR`;
* the width of the index register, which the listing prints as the
  register's name;
* which operand the memory reference is, which says whether the
  instruction gathers or scatters.

Usage: vsib_tables.py [--rust] [--stats]
"""
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llvm_disasm as llvm

# A slot wide enough that no instruction runs past it, so one bad
# encoding cannot shift the ones after it. Sixteen is the widest an x86
# instruction can be.
SLOT = 16

# The opcodes. `0f38` groups the gathers at `90..93` and the scatters at
# `a0..a3`; the difference between the two halves is a single bit, which
# is why they share a shape.
GATHER_OPS = (0x90, 0x91, 0x92, 0x93)
SCATTER_OPS = (0xa0, 0xa1, 0xa2, 0xa3)

# The mandatory prefixes an encoding can carry. Only `66` is used by this
# family, but the others are swept so that the table says so from
# measurement rather than from belief.
PPS = (0, 1, 2, 3)

# The register classes the disassembler prints, by width in bytes.
CLASSES = {"xmm": 16, "ymm": 32, "zmm": 64}

# The pointer sizes the disassembler prints, in bytes.
PTRS = {"BYTE": 1, "WORD": 2, "DWORD": 4, "QWORD": 8, "XMMWORD": 16,
        "YMMWORD": 32, "ZMMWORD": 64}


def vex(op, pp, w, ll):
    """A three-byte VEX encoding of `op`, with a SIB byte for the index.

    The layout was measured elsewhere in this directory and is repeated
    here only so the encodings can be built: `c4`, then the inverted
    `R X B` bits and the map, then `W vvvv L pp`.
    """
    p0 = 0xe0 | 2          # no extension bits, map 2 = `0f38`
    p1 = (w << 7) | (1 << 3) | (ll << 2) | pp
    # modrm 04 = SIB follows; sib 08 = base rax, index rcx, scale 1.
    return bytes([0xc4, p0, p1, op, 0x04, 0x08])


def evex(op, pp, w, ll):
    """An EVEX encoding of `op`, with a SIB byte for the index."""
    p0 = 0xf2 | 2          # no extension bits, map 2 = `0f38`
    p1 = (w << 7) | (0x0f << 3) | 0x04 | pp
    # `V'` must be ones: clearing it extends the index register.
    p2 = (ll << 5) | 0x08 | 1
    return bytes([0x62, p0, p1, p2, op, 0x04, 0x08])


def disassemble(blobs, tool):
    """One line per slot: the bytes the disassembler consumed and its text.

    The whole buffer goes to the disassembler in one pass, in
    fixed-size slots filled with `nop`, so that an encoding it rejects
    cannot move the ones after it. The second channel has its own helper,
    because it reads object files rather than raw bytes.
    """
    if tool == "llvm":
        return llvm.disassemble(blobs)
    data = b"".join(b + b"\x90" * (SLOT - len(b)) for b in blobs)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(data)
        path = f.name
    proc = subprocess.run(
        ["objdump", "-D", "-b", "binary", "-mi386:x86-64", "-Mintel", path],
        capture_output=True, text=True)
    text = {}
    for line in proc.stdout.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} )+)\s*(.*)", line)
        if not m:
            continue
        off = int(m.group(1), 16)
        if off % SLOT:
            continue
        text[off] = (len(m.group(2).split()), m.group(3).strip())
    return [text.get(i * SLOT, (0, "")) for i in range(len(blobs))]


def describe(body):
    """What a listing line says, in the terms the table records.

    Returns the mnemonic, the width of one memory element, the width of
    the index register, and whether the memory operand comes first.
    """
    if not body or "(bad)" in body or "<unknown>" in body:
        return None
    # The two disassemblers put the mnemonic after the bytes in their own
    # way: one separates it with a tab and the other with spaces. Both
    # are reduced to the same shape here.
    text = body.replace("\t", " ").strip().upper()
    mnemonic, _, rest = text.partition(" ")
    operands = [p.strip() for p in rest.split(",") if p.strip()]
    # Operands can contain commas inside the brackets only in the form
    # `[base+index*scale]`, which has none, so a plain split is enough.
    mem_idx = None
    elem = None
    index = None
    data = None
    for n, text in enumerate(operands):
        if "PTR" in text:
            mem_idx = n
            for word in text.replace("[", " ").replace("]", " ").split():
                if word in PTRS:
                    elem = PTRS[word]
            inside = text[text.find("[") + 1:text.find("]")] \
                if "[" in text else ""
            for token in re.split(r"[+\-*]", inside):
                m = re.fullmatch(r"(XMM|YMM|ZMM)\d+", token.strip())
                if m:
                    index = CLASSES[m.group(1).lower()]
            continue
        # A vector register named outside the brackets is a data one: it
        # is where the gathered elements go, or where the scattered ones
        # come from. A write mask is written after it and is not part of
        # the width.
        m = re.fullmatch(r"(XMM|YMM|ZMM)\d+", text.split("{")[0].strip())
        if m:
            data = CLASSES[m.group(1).lower()]
    if mem_idx is None or elem is None or index is None or data is None:
        return None
    return mnemonic.lower(), elem, index, data, mem_idx == 0


def measure(build, lengths, ops):
    """Every encoding of the family, measured by two disassemblers."""
    jobs = [(op, pp, w, ll) for op in ops for pp in PPS for w in (0, 1)
            for ll in lengths]
    blobs = [build(op, pp, w, ll) for op, pp, w, ll in jobs]
    gnu = disassemble(blobs, "gnu")
    llvm = disassemble(blobs, "llvm")
    out = {}
    for (op, pp, w, ll), (glen, gbody), (lbody, llen) in zip(jobs, gnu, llvm):
        g = describe(gbody)
        l = describe(lbody)
        if g is None or l is None:
            continue
        # Both disassemblers have to agree, and both have to have
        # consumed the same bytes: a disagreement about the length would
        # put every following instruction in the wrong place.
        if g != l or glen != llen:
            continue
        out[(op, pp, w, ll)] = g
    return out


def collapse(measured):
    """One row per opcode, prefix and `W` bit.

    The vector length is not part of the key because it changes the
    register widths, not the instruction: the decoder works those out
    from the `L` field and the two element widths. A row therefore has
    to hold for every length that named it, and a row that does not is
    dropped rather than written down half-measured.
    """
    rows = {}
    for (op, pp, w, ll), (name, elem, index, data, mem_first) in measured.items():
        key = (op, pp, w)
        rows.setdefault(key, []).append((ll, name, elem, index, data, mem_first))
    out = {}
    for key, seen in rows.items():
        names = {s[1] for s in seen}
        elems = {s[2] for s in seen}
        firsts = {s[5] for s in seen}
        if len(names) != 1 or len(elems) != 1 or len(firsts) != 1:
            continue
        # The width of one index element is not printed anywhere: what the
        # listing shows is the width of the whole index register, which is
        # the length of the vector scaled by the ratio of the two element
        # widths. Dividing the index register's width out of the data
        # register's leaves that ratio, and the ratio times the data width
        # is the index width.
        #
        # The ratio is read at the longest length the instruction exists
        # at, because at the shortest one both registers are clamped to
        # `xmm` and the two widths say nothing about each other.
        _, _, elem, index, data, _ = max(seen, key=lambda s: s[0])
        out[key] = (names.pop(), elems.pop(), elem * index // data,
                    firsts.pop())
    return out


def main():
    if "--stats" in sys.argv:
        pass
    vex_rows = collapse(measure(vex, (0, 1), GATHER_OPS))
    evex_rows = collapse(measure(evex, (0, 1, 2), GATHER_OPS + SCATTER_OPS))
    if "--stats" in sys.argv:
        print(f"# vex rows: {len(vex_rows)}", file=sys.stderr)
        print(f"# evex rows: {len(evex_rows)}", file=sys.stderr)
        for k in sorted(evex_rows):
            print(f"#  {k} {evex_rows[k]}", file=sys.stderr)
        for k in sorted(vex_rows):
            print(f"#  {k} {vex_rows[k]}", file=sys.stderr)
    if "--rust" not in sys.argv:
        return
    print("//! The gather and scatter tables.")
    print("//!")
    print("//! Generated by `scripts/vsib_tables.py` from what two")
    print("//! disassemblers say about every encoding of the family. The")
    print("//! opcodes are the only ones whose memory operand is addressed")
    print("//! by a vector register, and the width of that register is not")
    print("//! the width of the data register, so the rows record both.")
    print()
    print("use super::vector::Form;")
    print()
    print("/// One gather or scatter opcode.")
    print("pub struct VsibOp {")
    print("    /// The opcode byte.")
    print("    pub op: u8,")
    print("    /// The mandatory prefix.")
    print("    pub pp: u8,")
    print("    /// The mnemonic for the `W` bit clear.")
    print("    pub name: &'static str,")
    print("    /// The mnemonic for the `W` bit set.")
    print("    pub name_w: &'static str,")
    print("    /// The operand shape.")
    print("    pub form: Form,")
    print("    /// The width of one element of the memory operand, in bytes,")
    print("    /// when the `W` bit is clear.")
    print("    pub elem: u8,")
    print("    /// The same, when the `W` bit is set.")
    print("    pub elem_w: u8,")
    print("    /// The width of one element of the index register, in bytes.")
    print("    pub index: u8,")
    print("}")
    print()
    for label, rows, evex_form in (
            ("VEX_VSIB", vex_rows, False),
            ("EVEX_VSIB", evex_rows, True)):
        print(f"const {label}: &[VsibOp] = &[")
        # Group by opcode and prefix so that the two names of one opcode
        # stand next to each other.
        by_op = {}
        for (op, pp, w), (name, elem, index, mem_first) in rows.items():
            by_op.setdefault((op, pp), {})[w] = (name, elem, index, mem_first)
        for (op, pp), ws in sorted(by_op.items()):
            low = ws.get(0) or ws.get(1)
            high = ws.get(1) or ws.get(0)
            form = "Form::Scatter" if low[3] else (
                "Form::Two" if evex_form else "Form::Gather")
            print(f'    vsib(0x{op:02x}, {pp}, "{low[0]}", "{high[0]}", '
                  f"{form}, {low[1]}, {high[1]}, {low[2]}),")
        print("];")
        print()
    print("/// `const` constructor, so the table stays readable.")
    print("#[allow(clippy::too_many_arguments)]")
    print("const fn vsib(")
    print("    op: u8,")
    print("    pp: u8,")
    print("    name: &'static str,")
    print("    name_w: &'static str,")
    print("    form: Form,")
    print("    elem: u8,")
    print("    elem_w: u8,")
    print("    index: u8,")
    print(") -> VsibOp {")
    print("    VsibOp {")
    print("        op,")
    print("        pp,")
    print("        name,")
    print("        name_w,")
    print("        form,")
    print("        elem,")
    print("        elem_w,")
    print("        index,")
    print("    }")
    print("}")
    print()
    print("/// The row for one opcode, in the encoding the caller is reading.")
    print("pub fn lookup(evex: bool, op: u8, pp: u8) -> Option<&'static VsibOp> {")
    print("    let table = if evex { EVEX_VSIB } else { VEX_VSIB };")
    print("    table")
    print("        .iter()")
    print("        .find(|e| e.op == op && e.pp == pp)")
    print("}")


if __name__ == "__main__":
    main()
