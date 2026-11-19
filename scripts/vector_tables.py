#!/usr/bin/env python3
"""Build the VEX opcode tables from a real disassembler, not from memory.

The VEX opcode maps have hundreds of entries, and the mnemonic of each
depends on the opcode byte, the mandatory prefix, the `W` bit and
sometimes the vector length. Writing that down by hand means writing down
several hundred chances to be wrong.

This script builds the encodings instead and asks two independent
disassemblers what they mean. The truth comes from `llvm-objdump`, whose
AVX-512 and BMI support is complete, and from GNU `objdump`, which is the
tool the cross-check compares against. Only the answers the two agree on
are used; the disagreements are printed, because a name that only one of
them gives is not a name that can be relied on.

Three rounds are needed, because no single encoding says everything:

* With a non-zero `vvvv`, an instruction that has three operands is valid
  and one that has two is not.
* With a zero `vvvv`, the opposite holds.
* With a memory operand, the order of the operands says whether the
  instruction stores or loads, which the register form cannot show.

The table has one row for every combination of the opcode byte and the
mandatory prefix, because that is the granularity at which the instruction
is named: `0f 10` is four different instructions — `vmovups`, `vmovupd`,
`vmovss` and `vmovsd` — and each of them has its own operand shape and
its own conditions on the rest of the prefix.

Usage: vector_tables.py <map> [--rust]
        map 1, 2 or 3 selects the VEX opcode map; 4 selects EVEX;
        map 5 and 6 select the legacy `0f38` and `0f3a` maps, which use a
        mandatory prefix byte instead of the VEX prefix.
"""
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llvm_disasm as llvm
import truth_probe as tp

LINE = re.compile(r"^\s*([0-9a-f]+):\t((?:[0-9a-f]{2} )+)\s*(.*?)\s*$")

# objdump prints a mandatory prefix in front of the name when the prefix is
# not part of the instruction's own name. When the instruction that follows
# cannot be decoded, the prefix is printed on a line of its own.
PREFIX_WORDS = {
    "lock", "rep", "repz", "repnz", "repe", "repne", "bnd", "data16",
    "addr32", "notrack", "cs", "ds", "es", "fs", "gs", "ss",
}

# The operand shapes the tables use, in the terms the decoder uses.
#
# The shape is measured from the register form, and the memory form says
# whether the destination is the memory operand. The two are put together
# here, because the shape of the register form alone does not say where
# the destination goes: `vmovups xmm1, xmm2` and `vmovups [rax], xmm1`
# have the same register shape and different destinations.
#
# The first key is the shape of the register form, the second says
# whether the memory form puts the memory operand first.
FORMS = {
    # Two operands, the destination in the `reg` field.
    ("vecvec", False): "Form::Two",
    ("vecvec", True): "Form::Store",
    ("vecvecvec", False): "Form::Three",
    ("vecvecvec", True): "Form::StoreThree",
    ("vecvecvecvec", False): "Form::Four",
    ("vecvecvecvec", True): "Form::StoreFour",
    # A general-purpose register and a vector one.
    ("gpvec", False): "Form::GpDst",
    ("gpvec", True): "Form::GpStore",
    ("gpvecvec", False): "Form::GpDstThree",
    ("gpvecvec", True): "Form::GpStoreThree",
    ("vecgp", False): "Form::GpSrc",
    ("vecgp", True): "Form::GpStore",
    ("vecvecgp", False): "Form::ThreeGpSrc",
    ("vecvecgp", True): "Form::GpStoreThree",
    # Mask registers.
    ("maskmask", False): "Form::MaskStore",
    ("maskmask", True): "Form::MaskStore",
    ("maskgp", False): "Form::MaskGpStore",
    ("maskgp", True): "Form::MaskGpStore",
    ("gpmask", False): "Form::GpMaskSrc",
    ("gpmask", True): "Form::GpMaskSrc",
    ("maskmaskmask", False): "Form::MaskThreeReg",
    ("maskmaskmask", True): "Form::MaskThreeReg",
    # The BMI family: three general-purpose registers.
    ("gpgpgp", False): "Form::GpThree",
    ("gpgpgp", True): "Form::GpThree",
    # The non-destructive destination form.
    ("gpgp", False): "Form::GpVvvvDst",
    ("gpgp", True): "Form::GpVvvvDst",
    # The instruction has no operands of its own.
    ("", False): "Form::None",
    ("", True): "Form::None",
    ("gp", False): "Form::GpOnly",
    ("gp", True): "Form::GpOnly",
    ("mask", False): "Form::MaskOnly",
    ("mask", True): "Form::MaskOnly",
    ("mem", False): "Form::MemOnly",
    ("mem", True): "Form::MemOnly",
}


def build_vex(map_no, opcode, pp, vvvv=1, l=0, w=0, modrm=0xc1, two_byte=False):
    """A VEX encoding of `opcode` in `map_no` with mandatory prefix `pp`.

    The two-byte form cannot name the `0f38` and `0f3a` maps, and it has
    no `W` bit, so it is only used for the `0f` map.

    The bit layout, measured against the assembler:

        C1 = R̄ v̄vvv[3] v̄vvv[2:0] L pp
        C2 = R̄ X̄ B̄ mmmmm[4] mmmmm[3:0]
        C3 = W v̄vvv[3] v̄vvv[2:0] L pp

    Every field that holds a register number is stored inverted.
    """
    if two_byte:
        b1 = 0xe0 | ((~vvvv & 0x0f) << 3) | (l << 2) | (pp & 3)
        enc = bytes([0xc5, b1, opcode])
    else:
        b1 = 0xe0 | (map_no & 0x1f)
        b2 = (w << 7) | ((~vvvv & 0x0f) << 3) | (l << 2) | (pp & 3)
        enc = bytes([0xc4, b1, b2, opcode])
    if modrm is not None:
        enc += bytes([modrm])
    if map_no == 3:
        # Every instruction in the `0f3a` map carries an immediate byte.
        enc += b"\x00"
    return enc


def build_legacy(map_no, opcode, pp, modrm=0xc1):
    """A legacy encoding of `opcode` in `map_no` with mandatory prefix `pp`.

    The mandatory prefix comes first, as a separate byte, and the map is
    selected by the two bytes after `0f`.
    """
    prefix = {0: b"", 1: b"\x66", 2: b"\xf3", 3: b"\xf2"}[pp & 3]
    tail = b"\x38" if map_no == 1 else b"\x3a"
    enc = prefix + b"\x0f" + tail + bytes([opcode, modrm])
    if map_no == 2:
        # Every instruction in the `0f3a` map carries an immediate byte.
        enc += b"\x00"
    return enc


def collect(map_no):
    """Every instruction the map defines, with everything measured about it.

    The result is keyed by the opcode byte and the mandatory prefix. Each
    entry holds, for the `W` bit clear and for it set, the name, the
    operand shape, the immediate flag, and the values of the `L` bit and
    of the `vvvv` field that name an instruction.
    """
    # The `reg` field of the ModRM byte names a few instructions, and the
    # `L` bit and the `vvvv` field decide whether an encoding names one at
    # all, so all of them are swept.
    rows = []
    for opcode in range(256):
        for pp in range(4):
            for w in (0, 1):
                for l in (0, 1):
                    for vvvv in (0, 1):
                        for reg in range(8):
                            rows.append((opcode, pp, w, l, vvvv, reg))

    def build(mem):
        out = []
        for opcode, pp, w, l, vvvv, reg in rows:
            modrm = 0x00 if mem else 0xc0 | (reg << 3) | 1
            out.append(build_vex(map_no, opcode, pp, w=w, l=l, vvvv=vvvv,
                                 modrm=modrm))
        return out

    gnu_reg = tp.gnu_disassemble(build(False))
    ll_reg = llvm.disassemble(build(False))
    gnu_mem = tp.gnu_disassemble(build(True))
    ll_mem = llvm.disassemble(build(True))

    table = {}
    for is_mem, (gnu, ll) in enumerate(
            ((gnu_reg, ll_reg), (gnu_mem, ll_mem))):
        for (opcode, pp, w, l, vvvv, reg), gtext, (ltext, length) in zip(
                rows, gnu, ll):
            gname = tp.mnemonic_of(gtext)
            lname = tp.mnemonic_of(ltext)
            if gname != lname or not gname:
                # The two disagree, or neither knows. Nothing is learned
                # from this encoding.
                continue
            info = llvm.describe(ltext)
            if info is None:
                continue
            mnemonic, kinds, has_imm, _operands = info
            shape = "".join(k for k in kinds if k != "imm")
            entry = table.setdefault((opcode, pp), {}).setdefault(w, {
                "names": set(), "shapes": set(), "imm": False,
                "lens": set(), "vvvvs": set(), "regs": {},
                "memshapes": set(), "memwidths": set(),
            })
            entry["names"].add(mnemonic)
            entry["imm"] = entry["imm"] or has_imm
            entry["lens"].add(l)
            entry["vvvvs"].add(vvvv)
            entry["regs"].setdefault(mnemonic, set()).add(reg)
            if is_mem:
                # The memory form says where the destination is: a shape
                # that starts with `mem` puts the memory operand first,
                # and the instruction stores. It also says how wide the
                # memory operand is.
                entry["memshapes"].add(shape)
                entry["memwidths"].add(mem_width(ltext))
            else:
                entry["shapes"].add(shape)
    return table


def entry_for(entry):
    """The row of the table for one opcode and prefix.

    The result says the name for the `W` bit clear and for it set, the
    operand shape, the immediate flag, the vector length the instruction
    is defined at, whether the `vvvv` field must be zero, and the names
    that the `reg` field selects.
    """
    w0 = entry.get(0)
    w1 = entry.get(1)
    if w0 is None and w1 is None:
        return None
    base = w0 or w1
    # The name: the `W` bit changes it for a few instructions only.
    name = sorted(base["names"])[0]
    name_w = name
    if w0 and w1:
        n0 = sorted(w0["names"])
        n1 = sorted(w1["names"])
        if n0 != n1:
            name = n0[0]
            name_w = n1[0]
    # The shape. The register form says how the operands are arranged; the
    # memory form says whether the memory operand is the destination.
    shapes = sorted(base["shapes"])
    if not shapes:
        # The instruction has no register form. The memory form then says
        # everything there is to say.
        shapes = sorted(base["memshapes"])
    shape = shapes[0] if shapes else ""
    store = False
    for s in base["memshapes"]:
        if s.startswith("mem"):
            store = True
    # The vector length. The instruction exists at every length unless
    # only one of them names it.
    lens = base["lens"] | (w1["lens"] if w1 else set())
    length = "Length::Any"
    if lens == {0}:
        length = "Length::Xmm"
    elif lens == {1}:
        length = "Length::Ymm"
    # The `vvvv` field. Three rules are possible: any value, zero, or a
    # value that is not zero.
    vvvvs = base["vvvvs"] | (w1["vvvvs"] if w1 else set())
    vvvv_zero = False
    if vvvvs == {0}:
        vvvv_zero = True
    # The `reg` field names the instruction for a few of them.
    reg_names = None
    all_names = base["names"]
    if len(all_names) > 1:
        out = []
        for reg in range(8):
            hits = [n for n in all_names
                    if reg in base["regs"].get(n, ())]
            out.append(hits[0] if len(hits) == 1 else "")
        if any(out):
            reg_names = out
    # The size of the memory operand. It is measured from the memory
    # form, which is the only one that has a memory operand, and the
    # width that the disassembler prints says how wide it is.
    mem = mem_size_of(entry, 0)
    mem_w = mem_size_of(entry, 1)
    return (name, name_w, shape, base["imm"], store, length, vvvv_zero,
            reg_names, mem, mem_w)


def form_of(shape, store):
    """The operand shape, in the terms the decoder uses."""
    return FORMS.get((shape, store), "Form::Two")


def emit_rust(table, title):
    """The table as Rust source."""
    print(f"/// {title}")
    print("const TABLE: &[VecOp] = &[")
    for opcode, pp in sorted(table):
        entry = table[(opcode, pp)]
        got = entry_for(entry)
        if not got:
            continue
        (name, name_w, shape, has_imm, store, length, vvvv_zero,
         reg_names, mem, mem_w) = got
        form = form_of(shape, store)
        if reg_names is not None:
            listed = ", ".join(f'"{n}"' for n in reg_names)
            print(f"    vop_reg(0x{opcode:02x}, {pp}, [{listed}], {form}, "
                  f"{mem}, {'true' if has_imm else 'false'}),")
        elif name == name_w:
            if mem != mem_w:
                # The name is the same either way, but the memory operand
                # is not.
                print(f"    vop_wm(0x{opcode:02x}, {pp}, \"{name}\", "
                      f"\"{name}\", {form}, {mem}, {mem_w}, "
                      f"{'true' if has_imm else 'false'}),")
            elif length == "Length::Any" and not vvvv_zero:
                print(f"    vop(0x{opcode:02x}, {pp}, \"{name}\", {form}, "
                      f"{mem}, {'true' if has_imm else 'false'}),")
            else:
                print(f"    vop_c(0x{opcode:02x}, {pp}, \"{name}\", {form}, "
                      f"{mem}, {'true' if has_imm else 'false'}, "
                      f"{length}, {'true' if vvvv_zero else 'false'}),")
        else:
            if mem != mem_w:
                print(f"    vop_wm(0x{opcode:02x}, {pp}, \"{name}\", "
                      f"\"{name_w}\", {form}, {mem}, {mem_w}, "
                      f"{'true' if has_imm else 'false'}),")
            else:
                print(f"    vop_w(0x{opcode:02x}, {pp}, \"{name}\", "
                      f"\"{name_w}\", {form}, {mem}, "
                      f"{'true' if has_imm else 'false'}),")
    print("];")


# The memory operand sizes that the measurement cannot give, by opcode.
#
# The key is the opcode; the value is the size of the operand the
# instruction reads or writes.
MEM_OVERRIDES = {
    0x10: "MemSize::Element(4)",   # vmovss
    0x11: "MemSize::Element(4)",   # vmovss
    0x12: "MemSize::Element(8)",   # vmovhpd
    0x13: "MemSize::Element(8)",   # vmovhpd
    0x16: "MemSize::Element(8)",   # vmovhpd
    0x17: "MemSize::Element(8)",   # vmovhpd
    0x2a: "MemSize::Element(4)",   # vcvtsi2ss
    0x2c: "MemSize::Element(4)",   # vcvttss2si
    0x2d: "MemSize::Element(4)",   # vcvtss2si
    0x51: "MemSize::Element(4)",   # vsqrtss
    0x52: "MemSize::Element(4)",   # vrsqrtss
    0x53: "MemSize::Element(4)",   # vrcpss
    0x5a: "MemSize::Element(8)",   # vcvtss2sd
    0x5b: "MemSize::Element(4)",   # vcvtps2dq
    0x5c: "MemSize::Element(4)",   # vsubss
    0x5d: "MemSize::Element(4)",   # vminss
    0x5e: "MemSize::Element(4)",   # vdivss
    0x5f: "MemSize::Element(4)",   # vmaxss
    0x6e: "MemSize::Element(4)",   # vcvtsi2ss
    0x6f: "MemSize::Element(4)",   # vmovd
    0x7e: "MemSize::Element(4)",   # vmovq
    0x7f: "MemSize::Element(4)",   # vmovd
    0xc2: "MemSize::Element(4)",   # vcmpss
    0xc6: "MemSize::Element(4)",   # vshufps
    0xd6: "MemSize::Element(8)",   # vmovq
    0xe6: "MemSize::Element(8)",   # vcvtsd2ss
}


# The widths the disassembler prints for a memory operand, and the byte
# counts they stand for.
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


def mem_width(text):
    """The width of the memory operand in a disassembler's output.

    The result is the number of bytes, or zero when the instruction has no
    memory operand or the width is not one of the ones the disassembler
    prints.
    """
    if not text:
        return 0
    m = re.search(r"\b(byte|word|dword|qword|tbyte|xmmword|ymmword|zmmword)"
                  r"\s+ptr", text)
    if not m:
        return 0
    return MEM_WIDTHS.get(m.group(1), 0)


def mem_size_of(entry, w):
    """The size of the memory operand, in the terms the decoder uses.

    The width is measured from the memory form, which is the only form
    that has a memory operand. The size is one element when the width does
    not follow the vector length, and the whole vector when it does.
    """
    widths = set()
    for key, v in entry.items():
        if key == w:
            widths |= v["memwidths"]
    widths.discard(0)
    if not widths:
        return "MemSize::Vector"
    if len(widths) == 1:
        w0 = widths.pop()
        # The whole vector is sixteen, thirty-two or sixty-four bytes; a
        # single element is never that wide.
        if w0 in (16, 32, 64):
            return "MemSize::Vector"
        return f"MemSize::Element({w0})"
    # More than one width was measured for this variant of the
    # instruction. The widest one that is a whole vector says the operand
    # is one element.
    return "MemSize::Vector"


def legacy_entry(opcode, pp, slot, map_no):
    """One entry of the legacy table: the name, the shape, the memory size.

    The memory form is the one that can be trusted: every instruction in
    these maps has a memory form, while some of them have no register form
    at all. The register form is what says whether the instruction stores.

    Only the `66` prefix gives the SSE form of these instructions. Without
    it the same bytes are the MMX form, which has different operands and a
    different meaning, and the `f3` and `f2` prefixes are not defined for
    these maps at all.
    """
    if pp != 1:
        return None
    reg_slot = None
    mem_slot = None
    for key in _register_keys(False, 0):
        if key in slot:
            reg_slot = slot[key]
            break
    for key in _register_keys(True, 0):
        if key in slot:
            reg_slot = slot[key]
            break
    for key in _memory_keys(False, 0):
        if key in slot:
            mem_slot = slot[key]
            break
    for key in _memory_keys(True, 0):
        if key in slot:
            mem_slot = slot[key]
            break
    if reg_slot and mem_slot and reg_slot[0] != mem_slot[0]:
        # The register form is not an instruction of this kind.
        reg_slot = None
    source = reg_slot or mem_slot
    if source is None:
        return None
    mnemonic, kinds, has_imm, _, _length = source
    shape = "".join(k for k in kinds if k != "imm")
    store = False
    if mem_slot:
        mkinds = mem_slot[1]
        if mkinds and mkinds[0] == "mem" and shape != "mem":
            store = True
    key = (shape, store)
    form = LEGACY_FORMS.get(key)
    if form is None:
        return None
    # The size of the r/m operand, whether it ends up a register or a
    # memory reference.
    rm = "Size::X"
    if mnemonic.endswith("ss"):
        rm = "Size::D"
    elif mnemonic.endswith("sd"):
        rm = "Size::Q"
    elif mnemonic == "crc32":
        rm = "Size::B"
    elif mnemonic in ("movdiri", "aadd", "aand", "axor", "aor"):
        rm = "Size::Y"
    elif mnemonic in ("pextrb", "pinsrb"):
        rm = "Size::B"
    elif mnemonic == "pextrw":
        rm = "Size::W"
    elif mnemonic in ("pextrd", "extractps", "pinsrd", "movbe"):
        rm = "Size::D"
    # Every instruction in the `0f3a` map carries an immediate byte.
    if map_no == 6:
        has_imm = True
    return (opcode, pp, mnemonic, form, rm, has_imm)


def _register_keys(two, w):
    """The slot keys of the register forms, in the order they are tried."""
    for l in (0, 1):
        for vvvv in (1, 0):
            for reg in range(8):
                yield (vvvv, False, two, w, l, reg)


def _memory_keys(two, w):
    """The slot keys of the memory forms, in the order they are tried."""
    for l in (0, 1):
        for vvvv in (1, 0):
            for reg in range(8):
                yield (vvvv, True, two, w, l, reg)


def emit_legacy_rust(map_no):
    """The legacy table as Rust source."""
    table = collect_legacy(map_no)
    rows = []
    for opcode in sorted(table):
        for pp in range(4):
            slot = table[opcode].get(pp)
            if not slot:
                continue
            got = legacy_entry(opcode, pp, slot, map_no)
            if got:
                rows.append(got)
    print("/// The legacy instructions in the `%s` map." %
          ("0f38" if map_no == 5 else "0f3a"))
    print("const LEGACY_%s: &[SseOp] = &[" % ("0F38" if map_no == 5 else "0F3A"))
    for opcode, pp, mnemonic, form, mem, has_imm in rows:
        print(f'    sse_op(0x{opcode:02x}, {pp}, "{mnemonic}", {form}, {mem}, '
              f'{"true" if has_imm else "false"}),')
    print("];")


def collect_legacy(map_no):
    """The measurement for one of the legacy maps."""
    rows = []
    for opcode in range(256):
        for pp in range(4):
            for l in (0, 1):
                for vvvv in (1, 0):
                    for reg in range(8):
                        rows.append((opcode, pp, l, vvvv, reg))

    def build(mem):
        out = []
        for opcode, pp, l, vvvv, reg in rows:
            modrm = 0x00 if mem else 0xc0 | (reg << 3) | 1
            out.append(build_legacy(map_no - 4, opcode, pp, modrm=modrm))
        return out

    gnu_reg = tp.gnu_disassemble(build(False))
    ll_reg = llvm.disassemble(build(False))
    gnu_mem = tp.gnu_disassemble(build(True))
    ll_mem = llvm.disassemble(build(True))

    table = {}
    for is_mem, (gnu, ll) in enumerate(
            ((gnu_reg, ll_reg), (gnu_mem, ll_mem))):
        for (opcode, pp, l, vvvv, reg), gtext, (ltext, length) in zip(
                rows, gnu, ll):
            gname = tp.mnemonic_of(gtext)
            lname = tp.mnemonic_of(ltext)
            if gname != lname or not gname:
                continue
            info = llvm.describe(ltext)
            if info is None:
                continue
            mnemonic, kinds, has_imm, _operands = info
            key = (vvvv, is_mem, False, 0, l, reg)
            slot = table.setdefault(opcode, {}).setdefault(pp, {})
            slot[key] = (mnemonic, kinds, has_imm, "", length)
    return table


# The operand shapes of the legacy maps, in the terms the decoder uses.
LEGACY_FORMS = {
    ("vecvec", False): "SseForm::VecReg",
    ("vecmem", False): "SseForm::VecMem",
    ("memgp", True): "SseForm::GpMemStore",
    ("gpmem", False): "SseForm::GpRegMem",
    ("gpgp", False): "SseForm::GpReg",
    ("gpvec", True): "SseForm::GpStore",
    ("vecgp", False): "SseForm::VecGp",
    ("vecvecvec", False): "SseForm::VecRegXmm0",
    ("mem", True): "SseForm::MemOnly",
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "1"
    as_rust = "--rust" in sys.argv
    map_no = int(which)

    if map_no in (5, 6):
        if as_rust:
            emit_legacy_rust(map_no)
            return
        table = collect_legacy(map_no)
        for opcode in sorted(table):
            per_pp = table[opcode]
            names = []
            for pp in range(4):
                slot = per_pp.get(pp)
                got = legacy_entry(opcode, pp, slot, map_no) if slot else None
                names.append(f"{pp}:{got[2]}" if got else f"{pp}:-")
            print(f"    0x{opcode:02x}  " + "  ".join(names))
        return

    table = collect(map_no)
    if as_rust:
        title = {1: "The VEX instructions in the `0f` map.",
                 2: "The VEX instructions in the `0f38` map.",
                 3: "The VEX instructions in the `0f3a` map."}[map_no]
        emit_rust(table, title)
    else:
        for opcode, pp in sorted(table):
            got = entry_for(table[(opcode, pp)])
            if not got:
                continue
            name = got[0]
            print(f"    0x{opcode:02x} pp={pp}  {name:<16} "
                  f"shape={got[2]:<14} L={got[5]} vvvv_zero={got[6]}")


if __name__ == "__main__":
    main()
