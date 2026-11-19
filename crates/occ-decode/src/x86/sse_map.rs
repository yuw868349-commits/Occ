//! The legacy `0f38` and `0f3a` opcode maps.
//!
//! These two maps hold the instructions that arrived with SSE3.1, SSSE3,
//! SSE4.1, SSE4.2 and the carry-less multiplication, and the ones the newer
//! extensions added afterwards. They have no prefix of their own: the
//! mandatory prefix in front of `0f` is what says which of them an opcode
//! byte means, and the same byte means four different instructions
//! depending on it.
//!
//! The tables were built by encoding every combination and asking an
//! independent decoder what it saw, so the names, the operand shapes and
//! the immediate bytes are facts rather than recollections. The script that
//! does this is `scripts/vector_tables.py`.

use super::Size;

/// The operand shape of an instruction in one of the legacy maps.
///
/// The shape says where the destination is and what the sources are. The
/// legacy maps need more shapes than the vector ones do, because some of
/// their instructions move between a vector register and a general-purpose
/// one, and some of them name a general-purpose register as their
/// destination.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SseForm {
    /// `dst = reg` (vector), `src = rm` (vector or memory).
    VecReg,
    /// `dst = reg` (vector), `src = rm` (memory only).
    VecMem,
    /// `dst = rm` (general purpose or memory), `src = reg` (vector).
    GpStore,
    /// `dst = reg` (general purpose), `src = rm` (general purpose or
    /// memory).
    GpReg,
    /// `dst = rm` (general purpose or memory), `src = reg` (general
    /// purpose).
    GpMemStore,
    /// `dst = reg` (general purpose), `src = rm` (memory only).
    GpRegMem,
    /// `dst = reg` (vector), `src = rm` (general purpose).
    VecGp,
    /// `dst = reg` (vector), `src = rm` (vector or memory), and the implicit
    /// `xmm0` that the instruction blends in.
    VecRegXmm0,
    /// A single memory operand.
    MemOnly,
}

/// One instruction of a legacy vector opcode map.
pub struct SseOp {
    /// The opcode byte.
    pub op: u8,
    /// The mandatory prefix: none, `66`, `f3`, `f2`.
    pub pp: u8,
    /// The name of the instruction.
    pub name: &'static str,
    /// The operand shape.
    pub form: SseForm,
    /// The width of the r/m operand.
    pub rm: Size,
    /// A trailing immediate byte.
    pub imm8: bool,
    /// The name under `rex.W`, when the bit changes which instruction
    /// this is.
    ///
    /// `pinsrd` and `pinsrq` are the same opcode: the bit says whether the
    /// source is a dword or a qword. An empty string means the bit changes
    /// nothing about the name — `pinsrb` reads the same byte either way.
    pub name_w: &'static str,
    /// Whether `66` sets the operand size of this instruction rather than
    /// naming it.
    ///
    /// The two readings of the byte are different questions. Most of the
    /// instructions in these maps use `66` to say which instruction they
    /// are, and a second mandatory prefix beside it would be a
    /// contradiction. The ones that carry `f2` as their own prefix and
    /// take a general-purpose operand can have both: `66 f2 0f 38 f1` is
    /// the carry-less checksum of a word, and the `66` is the width, which
    /// is why the byte is not a second mandatory prefix there.
    pub size66: bool,
}

/// The instruction at the given opcode and mandatory prefix.
///
/// The tables are sorted by the opcode byte and then by the prefix, so the
/// lookup is a binary search.
pub fn lookup(map: u8, op: u8, pp: u8) -> Option<&'static SseOp> {
    let table = if map == 0x38 {
        LEGACY_0F38
    } else {
        LEGACY_0F3A
    };
    table
        .binary_search_by_key(&(op, pp), |e| (e.op, e.pp))
        .ok()
        .map(|i| &table[i])
}

const fn sse_op(op: u8, pp: u8, name: &'static str, form: SseForm, rm: Size, imm8: bool) -> SseOp {
    SseOp {
        op,
        pp,
        name,
        form,
        rm,
        imm8,
        name_w: "",
        size66: false,
    }
}

/// An instruction whose name depends on the operand-width bit.
const fn sse_wq(
    op: u8,
    pp: u8,
    name: &'static str,
    name_w: &'static str,
    form: SseForm,
    rm: Size,
    imm8: bool,
) -> SseOp {
    SseOp {
        op,
        pp,
        name,
        form,
        rm,
        imm8,
        name_w,
        size66: false,
    }
}

/// An instruction for which `66` sets the operand size rather than naming
/// the instruction.
const fn sse_sz(op: u8, pp: u8, name: &'static str, form: SseForm, rm: Size, imm8: bool) -> SseOp {
    SseOp {
        op,
        pp,
        name,
        form,
        rm,
        imm8,
        name_w: "",
        size66: true,
    }
}

/// The legacy instructions in the `0f38` map.
const LEGACY_0F38: &[SseOp] = &[
    sse_op(0x00, 0, "pshufb", SseForm::VecReg, Size::X, false),
    sse_op(0x00, 1, "pshufb", SseForm::VecReg, Size::X, false),
    sse_op(0x01, 0, "phaddw", SseForm::VecReg, Size::X, false),
    sse_op(0x01, 1, "phaddw", SseForm::VecReg, Size::X, false),
    sse_op(0x02, 0, "phaddd", SseForm::VecReg, Size::X, false),
    sse_op(0x02, 1, "phaddd", SseForm::VecReg, Size::X, false),
    sse_op(0x03, 0, "phaddsw", SseForm::VecReg, Size::X, false),
    sse_op(0x03, 1, "phaddsw", SseForm::VecReg, Size::X, false),
    sse_op(0x04, 0, "pmaddubsw", SseForm::VecReg, Size::X, false),
    sse_op(0x04, 1, "pmaddubsw", SseForm::VecReg, Size::X, false),
    sse_op(0x05, 0, "phsubw", SseForm::VecReg, Size::X, false),
    sse_op(0x05, 1, "phsubw", SseForm::VecReg, Size::X, false),
    sse_op(0x06, 0, "phsubd", SseForm::VecReg, Size::X, false),
    sse_op(0x06, 1, "phsubd", SseForm::VecReg, Size::X, false),
    sse_op(0x07, 0, "phsubsw", SseForm::VecReg, Size::X, false),
    sse_op(0x07, 1, "phsubsw", SseForm::VecReg, Size::X, false),
    sse_op(0x08, 0, "psignb", SseForm::VecReg, Size::X, false),
    sse_op(0x08, 1, "psignb", SseForm::VecReg, Size::X, false),
    sse_op(0x09, 0, "psignw", SseForm::VecReg, Size::X, false),
    sse_op(0x09, 1, "psignw", SseForm::VecReg, Size::X, false),
    sse_op(0x0a, 0, "psignd", SseForm::VecReg, Size::X, false),
    sse_op(0x0a, 1, "psignd", SseForm::VecReg, Size::X, false),
    sse_op(0x0b, 0, "pmulhrsw", SseForm::VecReg, Size::X, false),
    sse_op(0x0b, 1, "pmulhrsw", SseForm::VecReg, Size::X, false),
    sse_op(0x10, 1, "pblendvb", SseForm::VecRegXmm0, Size::X, false),
    sse_op(0x14, 1, "blendvps", SseForm::VecRegXmm0, Size::X, false),
    sse_op(0x15, 1, "blendvpd", SseForm::VecRegXmm0, Size::X, false),
    sse_op(0x17, 1, "ptest", SseForm::VecReg, Size::X, false),
    sse_op(0x1c, 0, "pabsb", SseForm::VecReg, Size::X, false),
    sse_op(0x1c, 1, "pabsb", SseForm::VecReg, Size::X, false),
    sse_op(0x1d, 0, "pabsw", SseForm::VecReg, Size::X, false),
    sse_op(0x1d, 1, "pabsw", SseForm::VecReg, Size::X, false),
    sse_op(0x1e, 0, "pabsd", SseForm::VecReg, Size::X, false),
    sse_op(0x1e, 1, "pabsd", SseForm::VecReg, Size::X, false),
    sse_op(0x20, 1, "pmovsxbw", SseForm::VecReg, Size::X, false),
    sse_op(0x21, 1, "pmovsxbd", SseForm::VecReg, Size::X, false),
    sse_op(0x22, 1, "pmovsxbq", SseForm::VecReg, Size::X, false),
    sse_op(0x23, 1, "pmovsxwd", SseForm::VecReg, Size::X, false),
    sse_op(0x24, 1, "pmovsxwq", SseForm::VecReg, Size::X, false),
    sse_op(0x25, 1, "pmovsxdq", SseForm::VecReg, Size::X, false),
    sse_op(0x28, 1, "pmuldq", SseForm::VecReg, Size::X, false),
    sse_op(0x29, 1, "pcmpeqq", SseForm::VecReg, Size::X, false),
    sse_op(0x2a, 1, "movntdqa", SseForm::VecMem, Size::X, false),
    sse_op(0x2b, 1, "packusdw", SseForm::VecReg, Size::X, false),
    sse_op(0x30, 1, "pmovzxbw", SseForm::VecReg, Size::X, false),
    sse_op(0x31, 1, "pmovzxbd", SseForm::VecReg, Size::X, false),
    sse_op(0x32, 1, "pmovzxbq", SseForm::VecReg, Size::X, false),
    sse_op(0x33, 1, "pmovzxwd", SseForm::VecReg, Size::X, false),
    sse_op(0x34, 1, "pmovzxwq", SseForm::VecReg, Size::X, false),
    sse_op(0x35, 1, "pmovzxdq", SseForm::VecReg, Size::X, false),
    sse_op(0x37, 1, "pcmpgtq", SseForm::VecReg, Size::X, false),
    sse_op(0x38, 1, "pminsb", SseForm::VecReg, Size::X, false),
    sse_op(0x39, 1, "pminsd", SseForm::VecReg, Size::X, false),
    sse_op(0x3a, 1, "pminuw", SseForm::VecReg, Size::X, false),
    sse_op(0x3b, 1, "pminud", SseForm::VecReg, Size::X, false),
    sse_op(0x3c, 1, "pmaxsb", SseForm::VecReg, Size::X, false),
    sse_op(0x3d, 1, "pmaxsd", SseForm::VecReg, Size::X, false),
    sse_op(0x3e, 1, "pmaxuw", SseForm::VecReg, Size::X, false),
    sse_op(0x3f, 1, "pmaxud", SseForm::VecReg, Size::X, false),
    sse_op(0x40, 1, "pmulld", SseForm::VecReg, Size::X, false),
    sse_op(0x41, 1, "phminposuw", SseForm::VecReg, Size::X, false),
    sse_op(0x80, 1, "invept", SseForm::GpRegMem, Size::X, false),
    sse_op(0x81, 1, "invvpid", SseForm::GpRegMem, Size::X, false),
    sse_op(0x82, 1, "invpcid", SseForm::GpRegMem, Size::X, false),
    sse_op(0xc8, 0, "sha1nexte", SseForm::VecReg, Size::X, false),
    sse_op(0xc9, 0, "sha1msg1", SseForm::VecReg, Size::X, false),
    sse_op(0xca, 0, "sha1msg2", SseForm::VecReg, Size::X, false),
    sse_op(0xcb, 0, "sha256rnds2", SseForm::VecRegXmm0, Size::X, false),
    sse_op(0xcc, 0, "sha256msg1", SseForm::VecReg, Size::X, false),
    sse_op(0xcd, 0, "sha256msg2", SseForm::VecReg, Size::X, false),
    sse_op(0xcf, 1, "gf2p8mulb", SseForm::VecReg, Size::X, false),
    sse_op(0xd8, 2, "aesencwide128kl", SseForm::MemOnly, Size::X, false),
    sse_op(0xdb, 1, "aesimc", SseForm::VecReg, Size::X, false),
    sse_op(0xdc, 1, "aesenc", SseForm::VecReg, Size::X, false),
    sse_op(0xdc, 2, "loadiwkey", SseForm::VecReg, Size::X, false),
    sse_op(0xdd, 1, "aesenclast", SseForm::VecReg, Size::X, false),
    sse_op(0xdd, 2, "aesdec128kl", SseForm::VecMem, Size::X, false),
    sse_op(0xde, 1, "aesdec", SseForm::VecReg, Size::X, false),
    sse_op(0xde, 2, "aesenc256kl", SseForm::VecMem, Size::X, false),
    sse_op(0xdf, 1, "aesdeclast", SseForm::VecReg, Size::X, false),
    sse_op(0xdf, 2, "aesdec256kl", SseForm::VecMem, Size::X, false),
    sse_op(0xf0, 0, "movbe", SseForm::GpRegMem, Size::D, false),
    sse_op(0xf0, 1, "movbe", SseForm::GpRegMem, Size::D, false),
    sse_sz(0xf0, 3, "crc32", SseForm::GpReg, Size::B, false),
    sse_op(0xf1, 0, "movbe", SseForm::GpMemStore, Size::D, false),
    sse_op(0xf1, 1, "movbe", SseForm::GpMemStore, Size::D, false),
    sse_sz(0xf1, 3, "crc32", SseForm::GpReg, Size::D, false),
    sse_op(0xf5, 1, "wrussd", SseForm::GpMemStore, Size::X, false),
    sse_op(0xf6, 0, "wrssd", SseForm::GpMemStore, Size::X, false),
    sse_op(0xf6, 1, "adcx", SseForm::GpReg, Size::D, false),
    sse_op(0xf6, 2, "adox", SseForm::GpReg, Size::D, false),
    sse_op(0xf8, 2, "uwrmsr", SseForm::GpReg, Size::D, false),
    sse_op(0xf8, 3, "urdmsr", SseForm::GpReg, Size::D, false),
    sse_op(0xf9, 0, "movdiri", SseForm::GpMemStore, Size::Y, false),
    sse_op(0xfa, 2, "encodekey128", SseForm::GpReg, Size::D, false),
    sse_op(0xfb, 2, "encodekey256", SseForm::GpReg, Size::D, false),
    sse_op(0xfc, 0, "aadd", SseForm::GpMemStore, Size::Y, false),
    sse_op(0xfc, 1, "aand", SseForm::GpMemStore, Size::Y, false),
    sse_op(0xfc, 2, "axor", SseForm::GpMemStore, Size::Y, false),
    sse_op(0xfc, 3, "aor", SseForm::GpMemStore, Size::Y, false),
];

/// The legacy instructions in the `0f3a` map.
///
/// Every instruction in this map carries an immediate byte.
const LEGACY_0F3A: &[SseOp] = &[
    sse_op(0x08, 1, "roundps", SseForm::VecReg, Size::X, true),
    sse_op(0x09, 1, "roundpd", SseForm::VecReg, Size::X, true),
    sse_op(0x0a, 1, "roundss", SseForm::VecReg, Size::D, true),
    sse_op(0x0b, 1, "roundsd", SseForm::VecReg, Size::Q, true),
    sse_op(0x0c, 1, "blendps", SseForm::VecReg, Size::X, true),
    sse_op(0x0d, 1, "blendpd", SseForm::VecReg, Size::X, true),
    sse_op(0x0e, 1, "pblendw", SseForm::VecReg, Size::X, true),
    sse_op(0x0f, 0, "palignr", SseForm::VecReg, Size::X, true),
    sse_op(0x0f, 1, "palignr", SseForm::VecReg, Size::X, true),
    sse_op(0x14, 1, "pextrb", SseForm::GpStore, Size::B, true),
    sse_op(0x15, 1, "pextrw", SseForm::GpStore, Size::W, true),
    sse_wq(0x16, 1, "pextrd", "pextrq", SseForm::GpStore, Size::D, true),
    sse_op(0x17, 1, "extractps", SseForm::GpStore, Size::D, true),
    sse_op(0x20, 1, "pinsrb", SseForm::VecGp, Size::B, true),
    sse_op(0x21, 1, "insertps", SseForm::VecReg, Size::X, true),
    sse_wq(0x22, 1, "pinsrd", "pinsrq", SseForm::VecGp, Size::D, true),
    sse_op(0x40, 1, "dpps", SseForm::VecReg, Size::X, true),
    sse_op(0x41, 1, "dppd", SseForm::VecReg, Size::X, true),
    sse_op(0x42, 1, "mpsadbw", SseForm::VecReg, Size::X, true),
    // The carry-less multiplication is named after the quadwords it works
    // on, and the immediate byte says which ones, so the name in this
    // table is only the one it takes when the byte names no choice.
    sse_op(0x44, 1, "pclmulqdq", SseForm::VecReg, Size::X, true),
    sse_op(0x60, 1, "pcmpestrm", SseForm::VecReg, Size::X, true),
    sse_op(0x61, 1, "pcmpestri", SseForm::VecReg, Size::X, true),
    sse_op(0x62, 1, "pcmpistrm", SseForm::VecReg, Size::X, true),
    sse_op(0x63, 1, "pcmpistri", SseForm::VecReg, Size::X, true),
    sse_op(0xcc, 0, "sha1rnds4", SseForm::VecReg, Size::X, true),
    sse_op(0xce, 1, "gf2p8affineqb", SseForm::VecReg, Size::X, true),
    sse_op(0xcf, 1, "gf2p8affineinvqb", SseForm::VecReg, Size::X, true),
    sse_op(0xdf, 1, "aeskeygenassist", SseForm::VecReg, Size::X, true),
];
