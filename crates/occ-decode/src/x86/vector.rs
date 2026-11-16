//! VEX and EVEX decoding.
//!
//! The two encodings replace the legacy prefix bytes with one of their own,
//! and they carry in their bit fields what the legacy encodings carried in
//! separate bytes: which of the three opcode maps the instruction lives in,
//! which mandatory prefix selects the variant, a third source operand, the
//! vector length, and the register-number extension bits.
//!
//! Two properties make the decoding exact rather than approximate, and both
//! are what the rest of the tool depends on:
//!
//! * The length of a vector instruction is fully determined by its opcode
//!   map and opcode byte. Every instruction in the `0f3a` map carries an
//!   immediate byte; the `0f38` map carries none. A ModRM byte, and the
//!   displacement it implies, are consumed by the same code that consumes
//!   them for legacy instructions. An opcode the table does not name still
//!   has a known length, and a wrong length would shift every following
//!   instruction, so the table is a convenience and not a precondition for
//!   staying in step with the byte stream.
//!
//! * The `vvvv` field names the second source operand of the three-operand
//!   instructions. Reading it is what separates `vaddps ymm0, ymm0, ymm1` from
//!   `vaddps ymm0, ymm1, ymm2`: the same bytes in the same places mean
//!   different things depending on which instruction they are.

use super::{Decoder, ModRm, Operand, Prefix, Size};
use occ_core::{CallTarget, FlowKind};

/// The vector registers an instruction can name.
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum VecReg {
    /// `xmm0..xmm31`, sixteen bytes wide.
    Xmm(u8),
    /// `ymm0..ymm31`, thirty-two bytes wide.
    Ymm(u8),
    /// `zmm0..zmm31`, sixty-four bytes wide.
    Zmm(u8),
    /// A mask register, `k0..k7`.
    Mask(u8),
}

impl VecReg {
    /// The register's width in bytes.
    #[allow(dead_code)]
    pub const fn width(self) -> u8 {
        match self {
            VecReg::Xmm(_) => 16,
            VecReg::Ymm(_) => 32,
            VecReg::Zmm(_) => 64,
            VecReg::Mask(_) => 8,
        }
    }
}

/// A VEX or EVEX prefix, read into one shape.
///
/// The two encodings differ in their details, but the decoder wants the same
/// facts from both: the map, the mandatory prefix, the third operand, the
/// vector length and the extension bits.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct VecPrefix {
    /// The opcode map: 1 = `0f`, 2 = `0f38`, 3 = `0f3a`.
    pub map: u8,
    /// The mandatory prefix: 0 = none, 1 = `66`, 2 = `f3`, 3 = `f2`.
    pub pp: u8,
    /// The `vvvv` field, already inverted. Zero means the instruction has no
    /// third source operand.
    pub vvvv: u8,
    /// The vector length in bytes: 16, 32 or 64.
    pub length: u8,
    /// The operand-width bit, which says whether the operation is on 32- or
    /// 64-bit elements.
    pub w: bool,
    /// The extension bits in REX layout: bit 2 is R, bit 1 is X, bit 0 is B.
    /// Each is 0 or 8, the value added to a three-bit register number.
    pub rex: u8,
    /// The extension bit above bit 3 of the ModRM `reg` field, as 0 or 16.
    ///
    /// EVEX carries it in `P0` bit 4. VEX has no such bit, and for VEX this
    /// is zero. It extends the destination register only, which is why it
    /// is kept apart from the other one: the two were measured to move
    /// different registers.
    pub high_reg: u8,
    /// The extension bit above bit 3 of the `vvvv` field, as 0 or 16.
    ///
    /// EVEX carries it in `P2` bit 3. The same bit also extends the `r/m`
    /// register and, when the memory operand is indexed by a vector
    /// register, the index register. VEX has no such bit.
    pub high_v: u8,
    /// EVEX: the zeroing-mask bit. A set bit says the elements the mask
    /// selects are written as zero rather than as the computed result.
    pub zeroing: bool,
    /// EVEX: the broadcast bit, which says the memory operand is one element
    /// that is copied to every lane.
    pub broadcast: bool,
    /// EVEX: the mask register number. Zero means the instruction is not
    /// masked.
    pub mask: u8,
    /// True when the prefix came from an EVEX encoding.
    pub evex: bool,
}

impl VecPrefix {
    /// Read the prefix at `b[*i]`.
    ///
    /// The first byte has already been looked at; `*i` points at it. On
    /// success `*i` is left after the prefix.
    pub fn parse(first: u8, b: &[u8], i: &mut usize) -> Option<VecPrefix> {
        match first {
            0xc5 => {
                // Two-byte VEX: R, vvvv, L, pp.
                let byte = *b.get(*i + 1)?;
                *i += 2;
                let r = (byte >> 7) & 1 ^ 1;
                let vvvv = !(byte >> 3) & 0x0f;
                let l = (byte >> 2) & 1;
                Some(VecPrefix {
                    map: 1,
                    pp: byte & 3,
                    vvvv,
                    length: if l == 1 { 32 } else { 16 },
                    w: false,
                    rex: r << 2,
                    high_reg: 0,
                    high_v: 0,
                    zeroing: false,
                    broadcast: false,
                    mask: 0,
                    evex: false,
                })
            }
            0xc4 => {
                // Three-byte VEX: R, X, B and the map, then W, vvvv, L, pp.
                let b1 = *b.get(*i + 1)?;
                let b2 = *b.get(*i + 2)?;
                *i += 3;
                let r = (b1 >> 7) & 1 ^ 1;
                let x = (b1 >> 6) & 1 ^ 1;
                let bb = (b1 >> 5) & 1 ^ 1;
                let w = (b2 >> 7) & 1;
                let vvvv = !(b2 >> 3) & 0x0f;
                let l = (b2 >> 2) & 1;
                Some(VecPrefix {
                    map: b1 & 0x1f,
                    pp: b2 & 3,
                    vvvv,
                    length: if l == 1 { 32 } else { 16 },
                    w: w != 0,
                    rex: (r << 2) | (x << 1) | bb,
                    high_reg: 0,
                    high_v: 0,
                    zeroing: false,
                    broadcast: false,
                    mask: 0,
                    evex: false,
                })
            }
            0x62 => {
                // EVEX: three bytes of payload. The two bits below the map
                // must be zero, and a byte that does not have them is not an
                // EVEX prefix at all.
                let b1 = *b.get(*i + 1)?;
                let b2 = *b.get(*i + 2)?;
                let b3 = *b.get(*i + 3)?;
                if b1 & 0x0c != 0 {
                    return None;
                }
                *i += 4;
                // The extension bits are stored inverted. The measured
                // layout is:
                //
                //     P0 = R̄ X̄ B̄ R̄' 0 mmm
                //     P1 = W ~vvvv 1 pp
                //     P2 = z L'L b V̄' aaa
                //
                // and the register numbers they extend are:
                //
                //     reg  = modrm.reg | (R << 3) | (R' << 4)
                //     rm   = modrm.rm  | (B << 3) | (X << 4)
                //     vvvv = ~P1[6:3]  | (V' << 4)
                let r = (((b1 >> 7) & 1) ^ 1) << 3;
                let x = (((b1 >> 6) & 1) ^ 1) << 1;
                let bb = (((b1 >> 5) & 1) ^ 1) << 3;
                let r2 = (((b1 >> 4) & 1) ^ 1) << 4;
                let v2 = (((b3 >> 3) & 1) ^ 1) << 4;
                let w = (b2 >> 7) & 1;
                let vvvv = !(b2 >> 3) & 0x0f;
                let pp = b2 & 3;
                let ll = (b3 >> 5) & 3;
                let length = match ll {
                    0 => 16,
                    1 => 32,
                    _ => 64,
                };
                Some(VecPrefix {
                    map: b1 & 0x07,
                    pp,
                    vvvv,
                    length,
                    w: w != 0,
                    rex: r | x | bb,
                    high_reg: r2,
                    high_v: v2,
                    zeroing: b3 & 0x80 != 0,
                    broadcast: b3 & 0x10 != 0,
                    mask: b3 & 0x07,
                    evex: true,
                })
            }
            _ => None,
        }
    }
}

/// The shape of a vector instruction's operands.
///
/// The forms are the ones that actually occur. What they have in common is
/// that the destination is named by one of the ModRM fields and the last
/// source by another, so the form only says where the pieces go.
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Form {
    /// No operands.
    None,
    /// `dst = reg`, `src = rm`.
    Two,
    /// `dst = reg`, `src1 = vvvv`, `src2 = rm`.
    Three,
    /// `dst = reg`, `src1 = vvvv`, `src2 = rm`, and a fourth source the
    /// immediate byte selects.
    Four,
    /// `dst = reg`, `src1 = vvvv`, `src2 = rm`, and a fourth source named
    /// by the `is4` byte.
    ///
    /// The `is4` byte is the last byte of the instruction, after the
    /// displacement, and its top four bits name the fourth source register.
    /// The blend and fused-multiply-add families have this shape.
    Is4,
    /// `dst = reg`, `src1 = vvvv`, `src2 = rm`, a fourth source named by
    /// the `is4` byte, and an immediate.
    ///
    /// The permutation family has this shape, and its `is4` byte carries
    /// the immediate in its low four bits: one byte holds both.
    Is4Imm,
    /// `dst = rm`, `src = reg`.
    Store,
    /// `dst = rm`, `src1 = vvvv`, `src2 = reg`.
    StoreThree,
    /// `dst = rm`, `src1 = vvvv`, `src2 = reg`, and a fourth source.
    StoreFour,
    /// `dst = reg` (general purpose), `src = rm` (vector).
    GpDst,
    /// `dst = reg` (general purpose), `src1 = vvvv`, `src2 = rm` (vector).
    GpDstThree,
    /// `dst = rm` (general purpose or memory), `src = reg` (vector).
    GpStore,
    /// `dst = rm` (general purpose or memory), `src1 = vvvv`, `src2 = reg`.
    GpStoreThree,
    /// `dst = reg` (vector), `src = rm` (general purpose).
    GpSrc,
    /// `dst = reg` (vector), `src1 = vvvv`, `src2 = rm` (general purpose).
    ThreeGpSrc,
    /// EVEX: `dst = mask`, `src1 = vvvv`, `src2 = rm` (vector).
    ///
    /// The comparison instructions write their result into a mask register
    /// rather than into a vector one.
    MaskThree,
    /// EVEX: `dst = mask`, `src = rm` (vector).
    MaskTwo,
    /// EVEX: `dst = reg` (vector), `src = rm` (vector, memory only).
    VecMem,
    /// EVEX: `dst = reg` (vector), `src1 = vvvv`, `src2 = rm` (vector,
    /// memory only).
    VecMemThree,
    /// EVEX: `dst = reg` (vector), `src1 = rm` (vector), `src2 = rm`
    /// (general purpose) — the gather instructions, where the memory
    /// operand is indexed by a vector register.
    Gather,
    /// EVEX: `dst = rm` (memory), `src = reg` (vector) — the scatter
    /// instructions, where the memory operand is indexed by a vector
    /// register.
    Scatter,
    /// EVEX: `dst = reg` (vector), `src1 = rm` (vector), `src2 = rm`
    /// (general purpose).
    VecVecMem,
    /// The instruction has only a memory operand.
    MemOnly,
    /// The instruction has only a general-purpose register operand.
    GpOnly,
    /// The instruction has only a mask register operand.
    MaskOnly,
    /// EVEX: `dst = mask`, `src1 = vvvv`, `src2 = rm`, and a fourth
    /// source the immediate byte selects.
    MaskFour,
    /// `dst = rm` (mask), `src = reg` (mask) — the moves between two mask
    /// registers.
    MaskStore,
    /// `dst = reg` (mask), `src = rm` (general purpose) — the move that
    /// reads a general-purpose register into a mask register.
    MaskGpStore,
    /// `dst = reg` (general purpose), `src = rm` (mask) — the move that
    /// writes a mask register out to a general-purpose register.
    GpMaskSrc,
    /// `dst = reg` (general purpose), `src1 = vvvv` (general purpose),
    /// `src2 = rm` (general purpose or memory).
    ///
    /// The BMI and BMI2 instructions have three general-purpose operands:
    /// the destination is in the ModRM `reg` field, the first source in the
    /// `vvvv` field of the prefix, and the second source in the ModRM `rm`
    /// field.
    GpThree,
    /// `dst = vvvv` (general purpose), `src = rm` (general purpose or
    /// memory).
    ///
    /// The `blsr`, `blsmsk` and `blsi` instructions put their destination in
    /// the `vvvv` field of the prefix rather than in the ModRM `reg` field,
    /// and they use the `reg` field to tell the three of them apart. The
    /// shape is the one the manuals call "non-destructive destination".
    GpVvvvDst,
    /// `dst = reg` (mask), `src1 = vvvv` (mask), `src2 = rm` (mask).
    ///
    /// The mask logic instructions take three mask registers: the
    /// destination is in the ModRM `reg` field, the first source in the
    /// `vvvv` field of the prefix, and the second source in the ModRM `rm`
    /// field.
    MaskThreeReg,
    /// A scalar load: one operand fewer when the source is memory.
    ///
    /// `vmovss xmm1, xmm2, xmm3` names the lane it takes from its second
    /// source, and needs three operands to do it. `vmovss xmm1, [rax]`
    /// has only one lane to take, so the memory form drops the second
    /// source and the `vvvv` field goes unused. Which of the two an
    /// encoding means is decided by the ModRM mode, not by the table.
    ScalarLoad,
    /// A scalar store: one operand fewer when the destination is memory.
    ///
    /// `vmovss xmm1, xmm2, xmm3` writes a lane of a register and must name
    /// it. `vmovss [rax], xmm2` writes to memory, where there is only one
    /// lane to write, so the destination is the memory operand and the
    /// `vvvv` field names the source.
    ScalarStore,
}

/// The vector length an instruction is defined at.
///
/// Most instructions exist at every length, and the prefix's `L` bit
/// chooses between them. A few do not: the mask logic family exists only
/// at the 256-bit length, and the moves between two mask registers exist
/// only at the 128-bit one. The value of the `L` bit is then not a choice
/// but a condition, and an encoding that does not meet it names no
/// instruction at all.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Length {
    /// The instruction exists at every length.
    Any,
    /// The instruction exists only at the 128-bit length.
    Xmm,
    /// The instruction exists only at the 256-bit length.
    Ymm,
}

impl Length {
    /// Whether the given vector length, in bytes, is the one this
    /// instruction is defined at.
    pub fn allows(self, length: u8) -> bool {
        match self {
            Length::Any => true,
            Length::Xmm => length == 16,
            Length::Ymm => length == 32,
        }
    }
}

/// The size of the memory operand.
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum MemSize {
    /// The whole vector: `xmmword`, `ymmword` or `zmmword`.
    Vector,
    /// One element, of the given width in bytes.
    Element(u8),
    /// The instruction has no memory operand.
    None,
}

/// One vector opcode, in one mandatory prefix.
///
/// The table has one row for every combination of the opcode byte and the
/// mandatory prefix, because that is how the instruction is actually
/// named: the prefix picks the variant, and the variant has its own
/// operand shape and its own conditions on the rest of the prefix. The
/// `0f` map's `0x10` is four different instructions — `vmovups`,
/// `vmovupd`, `vmovss` and `vmovsd` — and each of them says something
/// different about the length and about the third operand.
pub struct VecOp {
    /// The opcode byte.
    pub op: u8,
    /// The mandatory prefix: 0 = none, 1 = `66`, 2 = `f3`, 3 = `f2`.
    pub pp: u8,
    /// The mnemonic for the `W` bit clear.
    pub name: &'static str,
    /// The mnemonic for the `W` bit set.
    ///
    /// Most instructions do not read the `W` bit, and for those the name
    /// is the same either way. The ones that do read it name a different
    /// instruction: `kmovw` and `kmovq` are the same opcode with a
    /// different width.
    pub name_w: &'static str,
    /// The operand shape.
    pub form: Form,
    /// The size of the memory operand.
    pub mem: MemSize,
    /// The size of the memory operand when the `W` bit is set.
    ///
    /// Most instructions read the same width either way. The ones that do
    /// not are the moves between a vector and a general-purpose register:
    /// `vmovd` moves four bytes and `vmovq` moves eight, and they are the
    /// same opcode with the `W` bit set or clear.
    pub mem_w: MemSize,
    /// A trailing immediate byte.
    pub imm8: bool,
    /// The mnemonic for each value of the ModRM `reg` field.
    ///
    /// A few instructions encode several instructions in one opcode byte
    /// and tell them apart by the `reg` field: `0f38 f3` is `blsr`,
    /// `blsmsk` or `blsi` depending on what that field holds. For those,
    /// this array holds the name for `reg` values zero to seven, and the
    /// other field holds the name that applies when no `reg` value selects
    /// a different instruction.
    pub names_reg: [&'static str; 8],
    /// True when the `reg` field selects the name.
    ///
    /// When this is false the `reg` field carries no meaning and the
    /// instruction is named the same way whatever it holds.
    pub by_reg: bool,
    /// The vector length the instruction is defined at.
    ///
    /// Most instructions exist at every length, and the prefix's `L` bit
    /// chooses between them. A few exist at only one, and the length is
    /// then a condition rather than a choice: an encoding that does not
    /// meet it names no instruction at all.
    pub length: Length,
    /// True when the `vvvv` field of the prefix must hold zero.
    ///
    /// The instructions that have no third source operand leave the field
    /// at its "unused" value, and an encoding that puts a register number
    /// there names no instruction.
    pub vvvv_zero: bool,
    /// True when the instruction reads or writes memory and has no
    /// register form.
    ///
    /// The MXCSR instructions are the case: `ldmxcsr` and `stmxcsr` name
    /// a memory location and nothing else, so an encoding that puts a
    /// register in the `r/m` field names no instruction.
    pub mem_only: bool,
    /// Whether the instruction exists only when the `W` bit is set.
    ///
    /// `vpermq` permutes qwords, and a qword is only a qword when the
    /// encoding says so: the same opcode with `W` clear names no
    /// instruction at all. The bit is not a choice between two names
    /// here, it is a condition on the encoding.
    pub w_only: bool,
}

impl VecOp {
    /// The mnemonic for the given `W` bit and ModRM `reg` field.
    ///
    /// The `reg` field only matters for the instructions that use it to
    /// name the instruction; for the others the value is the one the `W`
    /// bit selects.
    pub fn name_reg(&self, w: bool, reg: u8) -> &'static str {
        if self.w_only && !w {
            return "";
        }
        if self.by_reg {
            let n = self.names_reg[usize::from(reg & 7)];
            if !n.is_empty() {
                return n;
            }
            // A `reg` value that selects no instruction of its own means
            // the encoding is not an instruction.
            return "";
        }
        if w {
            self.name_w
        } else {
            self.name
        }
    }

    /// Whether the prefix's fields allow the instruction to exist.
    ///
    /// Two of the fields decide it: the vector length, for the few
    /// instructions that exist at only one, and the `vvvv` field, for the
    /// ones that have no third source operand.
    pub fn prefix_allows(&self, length: u8, vvvv: u8) -> bool {
        self.length.allows(length) && !(self.vvvv_zero && vvvv != 0)
    }
}

/// `const` constructor, so the tables stay readable.
#[allow(clippy::too_many_arguments)]
pub const fn vop(
    op: u8,
    pp: u8,
    name: &'static str,
    form: Form,
    mem: MemSize,
    imm8: bool,
) -> VecOp {
    VecOp {
        op,
        pp,
        name,
        name_w: name,
        form,
        mem,
        mem_w: mem,
        imm8,
        names_reg: ["", "", "", "", "", "", "", ""],
        by_reg: false,
        length: Length::Any,
        vvvv_zero: false,
        mem_only: false,
        w_only: false,
    }
}

/// `const` constructor for the instructions whose name the `W` bit changes.
#[allow(clippy::too_many_arguments)]
pub const fn vop_w(
    op: u8,
    pp: u8,
    name: &'static str,
    name_w: &'static str,
    form: Form,
    mem: MemSize,
    imm8: bool,
) -> VecOp {
    VecOp {
        op,
        pp,
        name,
        name_w,
        form,
        mem,
        mem_w: mem,
        imm8,
        names_reg: ["", "", "", "", "", "", "", ""],
        by_reg: false,
        length: Length::Any,
        vvvv_zero: false,
        mem_only: false,
        w_only: false,
    }
}

/// `const` constructor for an instruction whose name changes with the
/// `W` bit and which exists at only one vector length.
///
/// The mask-register instructions are the case: `knotw` and `knotq` are
/// one opcode, and both exist only at the 128-bit length, which is to say
/// with `L` clear. The length is a condition on the encoding, not a
/// choice of width.
#[allow(clippy::too_many_arguments)]
pub const fn vop_wl(
    op: u8,
    pp: u8,
    name: &'static str,
    name_w: &'static str,
    form: Form,
    mem: MemSize,
    imm8: bool,
    length: Length,
) -> VecOp {
    VecOp {
        op,
        pp,
        name,
        name_w,
        form,
        mem,
        mem_w: mem,
        imm8,
        names_reg: ["", "", "", "", "", "", "", ""],
        by_reg: false,
        length,
        vvvv_zero: false,
        mem_only: false,
        w_only: false,
    }
}

/// `const` constructor for the instructions whose name and whose memory
/// operand both change with the `W` bit.
///
/// The moves between a vector and a general-purpose register are the case:
/// `vmovd` moves four bytes and `vmovq` moves eight, and they are the
/// same opcode with the `W` bit clear or set.
#[allow(clippy::too_many_arguments)]
pub const fn vop_wm(
    op: u8,
    pp: u8,
    name: &'static str,
    name_w: &'static str,
    form: Form,
    mem: MemSize,
    mem_w: MemSize,
    imm8: bool,
) -> VecOp {
    VecOp {
        op,
        pp,
        name,
        name_w,
        form,
        mem,
        mem_w,
        imm8,
        names_reg: ["", "", "", "", "", "", "", ""],
        by_reg: false,
        length: Length::Any,
        vvvv_zero: false,
        mem_only: false,
        w_only: false,
    }
}

/// `const` constructor for the instructions whose name the ModRM `reg`
/// field selects.
///
/// The names are given for the `reg` values zero to seven; an empty name
/// means that value selects no instruction of its own.
#[allow(clippy::too_many_arguments)]
pub const fn vop_reg(
    op: u8,
    pp: u8,
    names_reg: [&'static str; 8],
    form: Form,
    mem: MemSize,
    imm8: bool,
) -> VecOp {
    VecOp {
        op,
        pp,
        name: "",
        name_w: "",
        form,
        mem,
        mem_w: mem,
        imm8,
        names_reg,
        by_reg: true,
        length: Length::Any,
        vvvv_zero: false,
        mem_only: false,
        w_only: false,
    }
}

/// `const` constructor for the instructions the prefix's fields constrain.
///
/// The mask logic family exists only at one vector length, and the
/// instructions that have no third source operand require the `vvvv` field
/// to hold its "unused" value. Both are measured rather than assumed, and
/// this constructor is how the measurement is written down.
#[allow(clippy::too_many_arguments)]
pub const fn vop_c(
    op: u8,
    pp: u8,
    name: &'static str,
    form: Form,
    mem: MemSize,
    imm8: bool,
    length: Length,
    vvvv_zero: bool,
) -> VecOp {
    VecOp {
        op,
        pp,
        name,
        name_w: name,
        form,
        mem,
        mem_w: mem,
        imm8,
        names_reg: ["", "", "", "", "", "", "", ""],
        by_reg: false,
        length,
        vvvv_zero,
        mem_only: false,
        w_only: false,
    }
}

/// `const` constructor for an instruction that exists only with `W` set.
///
/// The permutation of qwords is the case: the opcode says "permute", and
/// the `W` bit says the elements are qwords. Without it the encoding
/// names nothing, which is why the bit is a condition rather than a
/// choice of name.
#[allow(clippy::too_many_arguments)]
pub const fn vop_q(
    op: u8,
    pp: u8,
    name: &'static str,
    form: Form,
    mem: MemSize,
    imm8: bool,
    length: Length,
    vvvv_zero: bool,
) -> VecOp {
    VecOp {
        op,
        pp,
        name,
        name_w: name,
        form,
        mem,
        mem_w: mem,
        imm8,
        names_reg: ["", "", "", "", "", "", "", ""],
        by_reg: false,
        length,
        vvvv_zero,
        mem_only: false,
        w_only: true,
    }
}

/// `const` constructor for the instructions that name themselves by the
/// ModRM `reg` field and have no register form.
///
/// The MXCSR instructions are the case: `0f ae /2` reads the control word
/// from memory and `0f ae /3` writes it, and neither of them has a
/// register form at all.
#[allow(clippy::too_many_arguments)]
pub const fn vop_m(
    op: u8,
    pp: u8,
    names_reg: [&'static str; 8],
    mem: MemSize,
    imm8: bool,
    length: Length,
) -> VecOp {
    VecOp {
        op,
        pp,
        name: "",
        name_w: "",
        form: Form::MemOnly,
        mem,
        mem_w: mem,
        imm8,
        names_reg,
        by_reg: true,
        length,
        vvvv_zero: false,
        mem_only: true,
        w_only: false,
    }
}

/// The names of the comparison instruction.
///
/// The condition is part of the instruction's name: `vcmpeqps` and
/// `vcmpneqps` are different instructions, and a listing that calls them
/// both `vcmpps` has lost the only thing that tells them apart. There are
/// sixteen conditions and four precisions, and the machine has one
/// spelling for each pair.
const CMP_NAMES: [[&str; 4]; 16] = [
    [
        "vcmpeqps", "vcmpeqpd", "vcmpeqss", "vcmpeqsd",
    ],
    [
        "vcmpltps", "vcmpltpd", "vcmpltss", "vcmpltsd",
    ],
    [
        "vcmpleps", "vcmplepd", "vcmpless", "vcmplesd",
    ],
    [
        "vcmpunordps", "vcmpunordpd", "vcmpunordss", "vcmpunordsd",
    ],
    [
        "vcmpneqps", "vcmpneqpd", "vcmpneqss", "vcmpneqsd",
    ],
    [
        "vcmpnltps", "vcmpnltpd", "vcmpnltss", "vcmpnltsd",
    ],
    [
        "vcmpnleps", "vcmpnlepd", "vcmpnless", "vcmpnlesd",
    ],
    [
        "vcmpordps", "vcmpordpd", "vcmpordss", "vcmpordsd",
    ],
    [
        "vcmpeq_uqps", "vcmpeq_uqpd", "vcmpeq_uqss", "vcmpeq_uqsd",
    ],
    [
        "vcmpngeps", "vcmpngepd", "vcmpngess", "vcmpngesd",
    ],
    [
        "vcmpngtps", "vcmpngtpd", "vcmpngtss", "vcmpngtsd",
    ],
    [
        "vcmpfalseps", "vcmpfalsepd", "vcmpfalsess", "vcmpfalsesd",
    ],
    [
        "vcmpneq_oqps", "vcmpneq_oqpd", "vcmpneq_oqss", "vcmpneq_oqsd",
    ],
    [
        "vcmpgeps", "vcmpgepd", "vcmpgess", "vcmpgesd",
    ],
    [
        "vcmpgtps", "vcmpgtpd", "vcmpgtss", "vcmpgtsd",
    ],
    [
        "vcmptrueps", "vcmptruepd", "vcmptruess", "vcmptruesd",
    ],
];

impl Decoder {
    /// Decode a VEX or EVEX instruction.
    ///
    /// `b[*i]` is the prefix byte. On success the instruction's length has
    /// been consumed from `b` and the operands are built.
    pub(super) fn decode_vector(
        &mut self,
        first: u8,
        p: &Prefix,
        os: Size,
        asz: Size,
        b: &[u8],
        i: &mut usize,
    ) -> Option<(&'static str, FlowKind, Option<CallTarget>, Vec<Operand>)> {
        let vp = VecPrefix::parse(first, b, i)?;
        if vp.map == 0 {
            return None;
        }
        let opcode = *b.get(*i)?;
        *i += 1;
        // Where the instruction ends if it turns out to name nothing: the
        // opcode byte is the last one that belongs to it, because an
        // encoding that names no instruction cannot say what its operands
        // would be. `objdump` draws the line in the same place, which is
        // what keeps the two listings aligned across a byte sequence that
        // is data rather than code.
        let after_opcode = *i;

        // The extension bits are carried in the same layout as a REX prefix,
        // so the ModRM decoder needs no change to read them.
        let fake = Prefix {
            op66: false,
            op67: p.op67,
            lock: false,
            rep: None,
            seg: p.seg,
            rex: vp.rex & 0x0f,
            rex_present: true,
            count: p.count,
        };

        // The EVEX encoding has its own table: it names more instructions
        // and it names them with more fields, so the two cannot share one.
        if vp.evex {
            return self.decode_evex(opcode, &vp, &fake, os, asz, b, i);
        }
        if vp.map > 3 {
            return None;
        }

        // The instructions whose shape the table cannot express.
        if vp.map == 1 {
            match opcode {
                0x71..=0x73 => return self.decode_vector_shift(opcode, &vp, &fake, asz, b, i),
                0x77 => {
                    // Clearing the upper half of the vector registers takes
                    // no operands at all.
                    let name = if vp.length == 32 { "vzeroall" } else { "vzeroupper" };
                    return Some((name, FlowKind::Next, None, vec![]));
                }
                0xc2 => return self.decode_vector_compare(&vp, &fake, asz, b, i),
                _ => {}
            }
        }

        // The carry-less multiplication is named after the quadwords it
        // works on, and the immediate byte says which ones, so the name
        // is known only after that byte has been read.
        if vp.map == 3 && opcode == 0x44 {
            return self.decode_vector_pclmul(&vp, &fake, asz, b, i);
        }

        // The gather and scatter family is the only one whose memory
        // operand is addressed by a vector register, which the table
        // below cannot express, so it has a table of its own.
        if vp.map == 2 {
            if let Some(e) = super::vsib_tables::lookup(false, opcode, vp.pp) {
                return self.decode_vsib(e, &vp, &fake, os, asz, b, i);
            }
        }

        let table = super::vector_tables::table(vp.map);
        // The table has one row for every combination of the opcode byte
        // and the mandatory prefix, because that is the granularity at
        // which the instruction is named.
        let entry = table
            .iter()
            .find(|e| e.op == opcode && e.pp == vp.pp);

        // The ModRM byte is read before the name is settled, because a
        // few instructions name themselves by the `reg` field of that
        // byte: `0f38 f3` is `blsr`, `blsmsk` or `blsi` depending on
        // what it holds.
        //
        // The size of the memory operand is needed to read the byte, and
        // the size comes from the table, so the table entry is looked up
        // first and only the name waits for the byte.
        let Some(entry) = entry else {
            // The opcode names no instruction in this map at all.
            return Some(("(bad)", FlowKind::Next, None, vec![]));
        };
        let form = entry.form;
        let has_imm = entry.imm8;
        let mem = if vp.w { entry.mem_w } else { entry.mem };
        let size = mem_size(mem, vp.length);
        let m = self.modrm(&fake, size, asz, b, i)?;

        // Some instructions exist only for certain values of the `L` bit,
        // and some only when the `vvvv` field holds a particular value. A
        // name that does not hold for the encoding is not the name of the
        // instruction.
        //
        // The instructions that read or write memory and have no register
        // form are the other case: a register in the `r/m` field means the
        // encoding names no instruction.
        let n = entry.name_reg(vp.w, m.reg);
        let name = if n.is_empty()
            || !entry.prefix_allows(vp.length, vp.vvvv)
            || (entry.mem_only && m.mode == 3)
        {
            // The combination of opcode and prefix is not an instruction.
            // Nothing after the opcode byte belongs to it, so the byte
            // string ends there.
            *i = after_opcode;
            return Some(("(bad)", FlowKind::Next, None, vec![]));
        } else {
            n
        };

        let reg = self.vector_reg(&vp, m.reg);
        let rm_is_general = matches!(
            form,
            Form::GpSrc
                | Form::ThreeGpSrc
                | Form::GpStore
                | Form::GpStoreThree
                | Form::GpThree
                | Form::GpVvvvDst
        );
        let rm = if rm_is_general {
            // A general-purpose operand may also be a memory reference: the
            // extraction instructions write one, and the move instructions
            // read one.
            let gp_size = match form {
                Form::GpStore | Form::GpStoreThree => Size::B,
                _ => {
                    if vp.w {
                        Size::Q
                    } else {
                        Size::D
                    }
                }
            };
            if m.mem.is_some() {
                super::mem_operand(&m, gp_size, &fake)
            } else {
                Operand::Reg(self.gp_name(m.rm, os, &fake))
            }
        } else {
            self.vector_operand(&vp, &m, size, &fake)
        };

        // The instructions with four source operands carry one more byte
        // than the others: the `is4` byte, whose top four bits name the
        // fourth source register. It comes after the displacement, so it
        // is read here, once the ModRM byte and everything it implies have
        // been consumed.
        //
        // The permutation family keeps its immediate in the low four bits
        // of the same byte, so for that shape the byte holds both and
        // nothing else follows it.
        let mut is4_imm = 0i64;
        let is4 = match form {
            Form::Is4 => Some(self.imm(b, i, 1)? as u8 >> 4),
            Form::Is4Imm => {
                let byte = self.imm(b, i, 1)? as u8;
                is4_imm = i64::from(byte & 0x0f);
                Some(byte >> 4)
            }
            _ => None,
        };

        let mut ops = Vec::new();
        self.push_operands(&mut ops, form, &vp, &m, reg, rm, size, os, &fake, is4);
        if has_imm {
            let v = self.imm(b, i, 1)?;
            ops.push(Operand::Imm(v, Size::B));
        } else if matches!(form, Form::Is4Imm) {
            ops.push(Operand::Imm(is4_imm, Size::B));
        }
        Some((name, FlowKind::Next, None, ops))
    }

    /// Build the operand list for a vector instruction.
    ///
    /// The shape says where the destination is and where the sources are.
    /// The two registers that come from the ModRM byte are already built
    /// by the caller, because which of them is a general-purpose register
    /// depends on the shape as well.
    #[allow(clippy::too_many_arguments)]
    fn push_operands(
        &self,
        ops: &mut Vec<Operand>,
        form: Form,
        vp: &VecPrefix,
        m: &ModRm,
        reg: Operand,
        rm: Operand,
        size: Size,
        os: Size,
        fake: &Prefix,
        is4: Option<u8>,
    ) {
        match form {
            Form::None => {}
            Form::Two => {
                ops.push(reg);
                ops.push(rm);
            }
            Form::Three => {
                ops.push(reg);
                ops.push(self.vvvv_reg(vp));
                ops.push(rm);
            }
            Form::Four => {
                // The fused multiply-add family has a result and three
                // sources. The fourth is named by the immediate byte, which
                // selects one of the registers the encoding can address.
                ops.push(reg);
                ops.push(self.vvvv_reg(vp));
                ops.push(rm);
            }
            Form::Is4 => {
                // The fourth source is named by the `is4` byte, and it comes
                // last: after the result, the second source and the third.
                ops.push(reg);
                ops.push(self.vvvv_reg(vp));
                ops.push(rm);
                if let Some(n) = is4 {
                    ops.push(self.is4_reg(vp, n));
                }
            }
            Form::Is4Imm => {
                // The fourth source comes before the immediate byte, which
                // is why this shape is separate from `Is4`.
                ops.push(reg);
                ops.push(self.vvvv_reg(vp));
                ops.push(rm);
                if let Some(n) = is4 {
                    ops.push(self.is4_reg(vp, n));
                }
            }
            Form::Store => {
                ops.push(rm);
                ops.push(reg);
            }
            Form::ScalarLoad => {
                // The register form has to say which lane it reads from;
                // the memory form has only one lane to read.
                ops.push(reg);
                if m.mode == 3 {
                    ops.push(self.vvvv_reg(vp));
                }
                ops.push(rm);
            }
            Form::ScalarStore => {
                // The register form writes a lane of a register and names
                // it; the memory form writes memory, and the source is
                // then the register the `vvvv` field names.
                if m.mode == 3 {
                    ops.push(reg);
                    ops.push(self.vvvv_reg(vp));
                    ops.push(rm);
                } else {
                    ops.push(rm);
                    ops.push(self.vvvv_reg(vp));
                }
            }
            Form::StoreThree | Form::StoreFour => {
                ops.push(rm);
                ops.push(self.vvvv_reg(vp));
                ops.push(reg);
            }
            Form::GpDst => {
                ops.push(Operand::Reg(self.gp_name(m.reg, os, fake)));
                ops.push(self.vector_operand(vp, m, size, fake));
            }
            Form::GpDstThree => {
                ops.push(Operand::Reg(self.gp_name(m.reg, os, fake)));
                ops.push(self.vvvv_reg(vp));
                ops.push(self.vector_operand(vp, m, size, fake));
            }
            Form::GpStore => {
                ops.push(rm);
                ops.push(reg);
            }
            Form::GpStoreThree => {
                ops.push(rm);
                ops.push(self.vvvv_reg(vp));
                ops.push(reg);
            }
            Form::GpSrc => {
                ops.push(reg);
                ops.push(rm);
            }
            Form::ThreeGpSrc => {
                ops.push(reg);
                ops.push(self.vvvv_reg(vp));
                ops.push(rm);
            }
            Form::MaskThree => {
                // The result of the comparison goes into a mask register.
                ops.push(self.mask_reg(vp, m.reg));
                ops.push(self.vvvv_reg(vp));
                ops.push(rm);
            }
            Form::MaskTwo => {
                ops.push(self.mask_reg(vp, m.reg));
                ops.push(rm);
            }
            Form::MaskFour => {
                ops.push(self.mask_reg(vp, m.reg));
                ops.push(self.vvvv_reg(vp));
                ops.push(rm);
            }
            Form::VecMem => {
                ops.push(reg);
                ops.push(rm);
            }
            Form::VecMemThree => {
                ops.push(reg);
                ops.push(self.vvvv_reg(vp));
                ops.push(rm);
            }
            Form::VecVecMem => {
                ops.push(reg);
                ops.push(self.vvvv_reg(vp));
                ops.push(rm);
            }
            Form::Gather => {
                ops.push(reg);
                ops.push(rm);
                ops.push(self.vvvv_reg(vp));
            }
            Form::Scatter => {
                ops.push(rm);
                ops.push(reg);
            }
            Form::MemOnly => {
                ops.push(rm);
            }
            Form::GpOnly => {
                ops.push(Operand::Reg(self.gp_name(m.reg, os, fake)));
            }
            Form::MaskOnly => {
                // The instruction names one mask register and nothing
                // else.
                ops.push(Operand::vec(VecReg::Mask(m.reg & 7)));
            }
            Form::MaskStore => {
                // The move between two mask registers: the destination is
                // the reg field and the source is the r/m field.
                ops.push(Operand::vec(VecReg::Mask(m.reg & 7)));
                ops.push(Operand::vec(VecReg::Mask(m.rm & 7)));
            }
            Form::MaskGpStore => {
                // The move that reads a general-purpose register into a
                // mask register: the destination is the reg field and the
                // source is the r/m field.
                ops.push(Operand::vec(VecReg::Mask(m.reg & 7)));
                ops.push(Operand::Reg(self.gp_name(m.rm, os, fake)));
            }
            Form::GpMaskSrc => {
                // The move that writes a mask register out to a
                // general-purpose register: the destination is the reg
                // field and the source is the r/m field.
                ops.push(Operand::Reg(self.gp_name(m.reg, os, fake)));
                ops.push(Operand::vec(VecReg::Mask(m.rm & 7)));
            }
            Form::GpThree => {
                // The BMI family: the destination is the reg field, the
                // first source is the `vvvv` field of the prefix, and the
                // second source is the r/m field.
                ops.push(Operand::Reg(self.gp_name(m.reg, os, fake)));
                ops.push(Operand::Reg(self.gp_name(vp.vvvv, os, fake)));
                ops.push(rm);
            }
            Form::GpVvvvDst => {
                // The non-destructive destination form: the destination is
                // the `vvvv` field of the prefix and the source is the r/m
                // field.
                ops.push(Operand::Reg(self.gp_name(vp.vvvv, os, fake)));
                ops.push(rm);
            }
            Form::MaskThreeReg => {
                // The mask logic family: three mask registers.
                ops.push(Operand::vec(VecReg::Mask(m.reg & 7)));
                ops.push(Operand::vec(VecReg::Mask(vp.vvvv & 7)));
                ops.push(Operand::vec(VecReg::Mask(m.rm & 7)));
            }
        }
    }

    /// Decode a gather or a scatter.
    ///
    /// The family is the only one whose memory operand is addressed by a
    /// vector register, and the two widths it involves are not the same:
    /// the index holds one element per lane of the data register, so an
    /// index twice as wide as the data halves the number of lanes. The
    /// prefix's `L` field gives the wider of the two, which is why the
    /// data register is not always the one the length names.
    #[allow(clippy::too_many_arguments)]
    fn decode_vsib(
        &mut self,
        entry: &'static super::vsib_tables::VsibOp,
        vp: &VecPrefix,
        fake: &Prefix,
        os: Size,
        asz: Size,
        b: &[u8],
        i: &mut usize,
    ) -> Option<(&'static str, FlowKind, Option<CallTarget>, Vec<Operand>)> {
        // The narrower of the two registers is the wider one scaled by
        // the ratio of the element widths. Both are clamped to `xmm`:
        // there is no narrower vector register to name, and the
        // encoding at the shortest length uses one anyway.
        let elem = if vp.w { entry.elem_w } else { entry.elem };
        let wide = u16::from(elem.max(entry.index));
        let full = u16::from(vp.length);
        let data_bytes = (full * u16::from(elem) / wide).max(16) as u8;
        let index_bytes = (full * u16::from(entry.index) / wide).max(16) as u8;

        let mut shaped = *vp;
        shaped.length = data_bytes;
        let size = if elem == 8 { Size::Q } else { Size::D };
        let m = self.modrm(fake, size, asz, b, i)?;

        let dest = self.vector_reg(&shaped, m.reg);
        let mut mem = super::mem_operand(&m, size, fake);
        mem = with_vsib(mem, u16::from(index_bytes) * 8, shaped.high_v);
        if entry.form == Form::Scatter {
            // A scatter writes memory under a mask, and the mask is
            // written on the operand it guards rather than on a register.
            mem = with_mem_mask(mem, shaped.mask);
        }

        let mut ops = Vec::new();
        match entry.form {
            Form::Gather => {
                // The AVX2 form names its write mask as a vector
                // register, in the `vvvv` field.
                ops.push(dest);
                ops.push(mem);
                ops.push(self.vvvv_reg(&shaped));
            }
            Form::Scatter => {
                // The source is named by the `reg` field, at the data
                // width rather than at the prefix's length.
                let n = m.reg | shaped.high_reg;
                let src = Operand::vec(match data_bytes {
                    32 => VecReg::Ymm(n & 0x1f),
                    64 => VecReg::Zmm(n & 0x1f),
                    _ => VecReg::Xmm(n & 0x1f),
                });
                ops.push(mem);
                ops.push(src);
            }
            _ => {
                ops.push(dest);
                ops.push(mem);
            }
        }
        let _ = os;
        let name = if vp.w { entry.name_w } else { entry.name };
        Some((name, FlowKind::Next, None, ops))
    }

    /// Decode an EVEX instruction.
    ///
    /// The EVEX encoding has more fields than the VEX one, and they all
    /// take part in naming the instruction: the opcode map, the mandatory
    /// prefix, the `W` bit and the vector length. The table is keyed by
    /// the first three, and each row holds the name for each of the three
    /// vector lengths.
    #[allow(clippy::too_many_arguments)]
    fn decode_evex(
        &mut self,
        opcode: u8,
        vp: &VecPrefix,
        fake: &Prefix,
        os: Size,
        asz: Size,
        b: &[u8],
        i: &mut usize,
    ) -> Option<(&'static str, FlowKind, Option<CallTarget>, Vec<Operand>)> {
        // The vector length, as the table indexes it.
        let ll = match vp.length {
            16 => 0u8,
            32 => 1,
            _ => 2,
        };
        // The gather and scatter family is measured into a table of its
        // own, because its operand widths are not the prefix's.
        if vp.map == 2 {
            if let Some(e) = super::vsib_tables::lookup(true, opcode, vp.pp) {
                return self.decode_vsib(e, vp, fake, os, asz, b, i);
            }
        }
        let entry = super::evex_tables::lookup(opcode, vp.map, vp.pp, vp.w);
        // The comparison instructions are named after the condition they
        // test, and the condition is the immediate byte. The name is
        // therefore known only after the byte has been read.
        let cmp = super::cmp_tables::lookup(opcode, vp.map, vp.pp, vp.w);
        let Some(entry) = entry else {
            // The opcode names no instruction in this map at all, and the
            // encoding cannot say what its operands would be, so nothing
            // after the opcode byte belongs to it.
            return Some(("(bad)", FlowKind::Next, None, vec![]));
        };
        let n = entry.name(ll);
        if n.is_empty() {
            // The combination names no instruction at this length. The
            // byte string still ends at the opcode: a length that is not
            // an instruction has no operands to read.
            return Some(("(bad)", FlowKind::Next, None, vec![]));
        }
        let (name, form, has_imm, mem, elem) =
            (n, entry.form, entry.imm8, entry.mem, entry.elem);

        let mut size = mem_size(mem, vp.length);
        let mut m = self.modrm(fake, size, asz, b, i)?;
        // A one-byte displacement is not a number of bytes in an EVEX
        // encoding: it is a count of operand-sized units, and the size of
        // a unit depends on what the instruction reads. A full-vector
        // move counts in whole vectors, a scalar move counts in elements,
        // and a gather counts in elements even though the address it
        // forms is a vector of them.
        if m.mode == 1 {
            let unit = mem_bytes(mem, vp.length);
            if unit > 1 {
                if let Some(ref mut mem_ref) = m.mem {
                    mem_ref.disp *= i64::from(unit);
                }
            }
        }
        // The broadcast bit means two different things, and the ModRM mode
        // is what picks: with a memory operand it says the operand is one
        // element copied to every lane, and with a register operand it
        // selects a rounding mode. Only the first is a property of the
        // operand, so only the first changes what is printed.
        let broadcast = if vp.broadcast && elem != 0 && m.mem.is_some() {
            size = mem_size(MemSize::Element(elem), vp.length);
            Some(vp.length / elem)
        } else {
            None
        };
        // The source register is not always as wide as the prefix says.
        // An instruction that reads one element of memory reads one
        // register's worth of them too, and the smallest vector register
        // is `xmm`: `vpbroadcastd zmm1, xmm2` names a 512-bit
        // destination beside a 128-bit source, because there is no
        // smaller register to name.
        let mut src_shape = *vp;
        src_shape.length = mem_bytes(mem, vp.length).clamp(16, vp.length);

        // A store writes memory, and the mask on a store guards memory:
        // `vmovdqu64 [rax]{k1}, zmm1` leaves the bytes outside the mask
        // untouched. The destination register carries no mask, because
        // no register is written.
        let store_form = matches!(
            form,
            Form::Store | Form::StoreThree | Form::StoreFour | Form::ScalarStore
        );
        let mut dst_shape = *vp;
        if entry.xmm_dest && !store_form {
            // A scalar instruction names one lane whatever the prefix
            // says, and one lane is an `xmm`.
            //
            // A store is not one of those: the `xmm` its register form
            // names is the memory operand, and the register beside it is
            // the whole vector the lanes are taken out of.
            dst_shape.length = 16;
            src_shape.length = 16;
        }
        let masked_memory = store_form && m.mem.is_some();
        let reg = if masked_memory {
            Operand::vec(match dst_shape.length {
                32 => VecReg::Ymm(m.reg & 0x1f),
                64 => VecReg::Zmm(m.reg & 0x1f),
                _ => VecReg::Xmm(m.reg & 0x1f),
            })
        } else {
            self.vector_reg(&dst_shape, m.reg)
        };
        let rm_is_general = matches!(
            form,
            Form::GpSrc | Form::ThreeGpSrc | Form::GpStore | Form::GpStoreThree
        );
        let rm = if rm_is_general {
            let gp_size = match form {
                Form::GpStore | Form::GpStoreThree => Size::B,
                _ => {
                    if vp.w {
                        Size::Q
                    } else {
                        Size::D
                    }
                }
            };
            if m.mem.is_some() {
                super::mem_operand(&m, gp_size, fake)
            } else {
                Operand::Reg(self.gp_name(m.rm, os, fake))
            }
        } else {
            self.vector_operand(&src_shape, &m, size, fake)
        };
        let mut rm = match broadcast {
            Some(lanes) => with_broadcast(rm, size, lanes),
            None => rm,
        };
        if masked_memory {
            rm = with_mem_mask(rm, vp.mask);
        }

        let mut ops = Vec::new();
        // The `vvvv` field is a source register, and a scalar
        // instruction names it at the same width as the destination.
        self.push_operands(&mut ops, form, &dst_shape, &m, reg, rm, size, os, fake, None);
        let mut imm_value = 0i64;
        if has_imm {
            imm_value = self.imm(b, i, 1)?;
            ops.push(Operand::Imm(imm_value, Size::B));
        }
        // A comparison instruction is named after the condition it tests,
        // and the condition is the immediate byte. When the table has the
        // family, the name comes from it.
        // The condition is the low three bits of the immediate, and an
        // immediate with anything above them set names no condition: the
        // listing falls back to the family name, `vpcmpud` rather than
        // `vpcmpequd`.
        let mut name = match cmp {
            Some(c) if (0..8).contains(&imm_value) => c.name(imm_value as u8),
            _ => name,
        };
        // The carry-less multiplication is named after the quadwords it
        // works on, and the immediate byte says which ones.
        if vp.map == 3 && opcode == 0x44 {
            if let Some(n) = super::pclmul::name(true, imm_value as u8) {
                name = n;
            }
        }
        Some((name, FlowKind::Next, None, ops))
    }

    /// The shifts that take an immediate count.
    ///
    /// Their source is the r/m operand and the count is the immediate; the
    /// `vvvv` field is reserved and must be all ones.
    fn decode_vector_shift(
        &mut self,
        opcode: u8,
        vp: &VecPrefix,
        fake: &Prefix,
        asz: Size,
        b: &[u8],
        i: &mut usize,
    ) -> Option<(&'static str, FlowKind, Option<CallTarget>, Vec<Operand>)> {
        let m = self.modrm(fake, Size::X, asz, b, i)?;
        let name = match (opcode, m.reg & 7) {
            (0x71, 2) => "vpsrlw",
            (0x71, 4) => "vpsraw",
            (0x71, 6) => "vpsllw",
            (0x72, 2) => "vpsrld",
            (0x72, 4) => "vpsrad",
            (0x72, 6) => "vpslld",
            (0x73, 2) => "vpsrlq",
            (0x73, 3) => "vpsrldq",
            (0x73, 6) => "vpsllq",
            (0x73, 7) => "vpslldq",
            _ => return None,
        };
        let count = self.imm(b, i, 1)?;
        let ops = vec![
            self.vector_reg(vp, m.reg),
            self.vector_operand(vp, &m, Size::X, fake),
            Operand::Imm(count, Size::B),
        ];
        Some((name, FlowKind::Next, None, ops))
    }

    /// The comparison with a condition in the immediate byte.
    /// The carry-less multiplication, named after the quadwords it works on.
    ///
    /// The immediate byte selects the quadwords, and it is the last byte
    /// of the instruction, so the name is settled after it has been read.
    fn decode_vector_pclmul(
        &mut self,
        vp: &VecPrefix,
        fake: &Prefix,
        asz: Size,
        b: &[u8],
        i: &mut usize,
    ) -> Option<(&'static str, FlowKind, Option<CallTarget>, Vec<Operand>)> {
        let m = self.modrm(fake, Size::X, asz, b, i)?;
        let imm = self.imm(b, i, 1)? as u8;
        let name = super::pclmul::name(true, imm).unwrap_or("vpclmulqdq");
        let ops = vec![
            self.vector_reg(vp, m.reg),
            self.vvvv_reg(vp),
            self.vector_operand(vp, &m, Size::X, fake),
        ];
        Some((name, FlowKind::Next, None, ops))
    }

    /// The vector compare instructions, named after their condition.
    fn decode_vector_compare(
        &mut self,
        vp: &VecPrefix,
        fake: &Prefix,
        asz: Size,
        b: &[u8],
        i: &mut usize,
    ) -> Option<(&'static str, FlowKind, Option<CallTarget>, Vec<Operand>)> {
        let m = self.modrm(fake, Size::X, asz, b, i)?;
        let cond = self.imm(b, i, 1)? as u8 & 0x0f;
        let name = CMP_NAMES[usize::from(cond & 0x0f)][usize::from(vp.pp & 3)];
        let ops = vec![
            self.vector_reg(vp, m.reg),
            self.vvvv_reg(vp),
            self.vector_operand(vp, &m, Size::X, fake),
        ];
        Some((name, FlowKind::Next, None, ops))
    }

    /// The vector register the destination is named by.
    fn vector_reg(&self, vp: &VecPrefix, reg: u8) -> Operand {
        let n = reg | vp.high_reg;
        let reg = match vp.length {
            32 => VecReg::Ymm(n & 0x1f),
            64 => VecReg::Zmm(n & 0x1f),
            _ => VecReg::Xmm(n & 0x1f),
        };
        // The write mask is a property of the register the instruction
        // writes, and of no other operand: `vvvv` and `rm` name sources
        // that the mask does not touch.
        Operand::Vec {
            reg,
            mask: vp.mask,
            // Zeroing without a mask is not a thing the encoding can say;
            // the bit would be describing a mask that is not there.
            zeroing: vp.zeroing && vp.mask != 0,
        }
    }

    /// The vector register the `vvvv` field names.
    fn vvvv_reg(&self, vp: &VecPrefix) -> Operand {
        let n = vp.vvvv | vp.high_v;
        Operand::vec(match vp.length {
            32 => VecReg::Ymm(n & 0x1f),
            64 => VecReg::Zmm(n & 0x1f),
            _ => VecReg::Xmm(n & 0x1f),
        })
    }

    /// The register the `is4` byte names.
    ///
    /// The byte's top four bits hold the register number, and the register
    /// is a vector register of the instruction's own length.
    fn is4_reg(&self, vp: &VecPrefix, n: u8) -> Operand {
        Operand::vec(match vp.length {
            32 => VecReg::Ymm(n & 0x1f),
            64 => VecReg::Zmm(n & 0x1f),
            _ => VecReg::Xmm(n & 0x1f),
        })
    }

    /// The mask register the prefix names.
    ///
    /// The EVEX prefix carries the number of the mask register that guards
    /// the instruction. A zero means the instruction is not masked.
    /// The mask register the instruction writes, and the mask that
    /// guards it.
    ///
    /// A comparison writes its result to the register the ModRM `reg`
    /// field names, and the prefix's `aaa` field is a second mask over
    /// that write: `vpcmpud k0{k1}` writes the lanes `k1` selects and
    /// leaves the rest of `k0` alone. Reading the number from the wrong
    /// field turns the guard into the destination.
    fn mask_reg(&self, vp: &VecPrefix, reg: u8) -> Operand {
        Operand::Vec {
            reg: VecReg::Mask(reg & 0x07),
            mask: vp.mask,
            zeroing: false,
        }
    }

    /// The r/m operand, as a vector register or a memory reference.
    fn vector_operand(&self, vp: &VecPrefix, m: &ModRm, size: Size, fake: &Prefix) -> Operand {
        if m.mem.is_some() {
            return super::mem_operand(m, size, fake);
        }
        let n = m.rm | vp.high_v;
        Operand::vec(match vp.length {
            32 => VecReg::Ymm(n & 0x1f),
            64 => VecReg::Zmm(n & 0x1f),
            _ => VecReg::Xmm(n & 0x1f),
        })
    }

    /// The name of a general-purpose register.
    fn gp_name(&self, n: u8, os: Size, fake: &Prefix) -> &'static str {
        let regs = os.regs(fake.rex_present);
        regs[usize::from(n) & 15]
    }
}

/// The same memory operand, addressed by a vector register of the given
/// width in bits.
///
/// `high` is the extension bit above bit 3 of the index register. A
/// vector index has more registers to name than a SIB byte has bits, so
/// EVEX spends the bit that extends `vvvv` on the index instead: the two
/// cannot both be in use, because a gather has no third source operand.
fn with_vsib(op: super::Operand, bits: u16, high: u8) -> super::Operand {
    match op {
        super::Operand::Mem {
            base,
            index,
            scale,
            disp,
            rip_relative,
            size,
            broadcast,
            mask,
            ..
        } => super::Operand::Mem {
            base,
            index: index.map(|n| n | high),
            scale,
            disp,
            rip_relative,
            size,
            broadcast,
            vsib: Some(bits),
            mask,
        },
        other => other,
    }
}

/// The same memory operand, written under the given mask register.
fn with_mem_mask(op: super::Operand, mask: u8) -> super::Operand {
    match op {
        super::Operand::Mem {
            base,
            index,
            scale,
            disp,
            rip_relative,
            size,
            broadcast,
            vsib,
            ..
        } => super::Operand::Mem {
            base,
            index,
            scale,
            disp,
            rip_relative,
            size,
            broadcast,
            vsib,
            mask,
        },
        other => other,
    }
}

/// The same memory operand, read as one element copied to every lane.
fn with_broadcast(op: super::Operand, size: super::Size, lanes: u8) -> super::Operand {
    match op {
        super::Operand::Mem {
            base,
            index,
            scale,
            disp,
            rip_relative,
            vsib,
            mask,
            ..
        } => super::Operand::Mem {
            base,
            index,
            scale,
            disp,
            rip_relative,
            size: Some(size),
            broadcast: Some(lanes),
            vsib,
            mask,
        },
        other => other,
    }
}


/// The size of the memory operand.
/// The width of the memory operand in bytes, which is the unit a
/// one-byte displacement counts in.
fn mem_bytes(mem: MemSize, length: u8) -> u8 {
    match mem {
        MemSize::Vector => length,
        MemSize::Element(w) => w,
        MemSize::None => 1,
    }
}

fn mem_size(mem: MemSize, length: u8) -> Size {
    match mem {
        MemSize::Vector => match length {
            32 => Size::Y,
            64 => Size::Z,
            _ => Size::X,
        },
        MemSize::Element(w) => match w {
            1 => Size::B,
            2 => Size::W,
            8 => Size::Q,
            16 => Size::X,
            _ => Size::D,
        },
        MemSize::None => Size::X,
    }
}
