//! x86 lifting: Intel-syntax text to [`Stmt`]s.
//!
//! The decoder renders every instruction to text; this module reads that text
//! back. That sounds like a detour, and it is one, but it keeps a single source
//! of truth for operand formatting and makes the lifter testable against the
//! same strings a human sees.
//!
//! Both widths are modelled. The instruction set is the same one, and the
//! differences that matter are a small, closed list: the registers are half as
//! wide and named differently, the stack slots are four bytes instead of
//! eight, there are no `r8`-`r15`, and every argument arrives on the stack
//! instead of in a register. [`Mode`] carries that and the target reads it.
//!
//! What is modelled: general-purpose registers at all four widths, the stack
//! frame (so `[rbp-0x8]` becomes `local_8`), flags as a pending comparison,
//! and the common integer instruction set. What is not: floating point and
//! vector registers, and system instructions. Those become [`Stmt::Comment`]s,
//! so the output shows exactly what was skipped instead of pretending.

use occ_core::{Addr, FlowKind, Instruction};

use super::{
    merge_write, slot_name,
    Core, Lifted, Operand, Parsed, Reg, Target, VecClass, parse_int, split_operands,
};
use crate::ir::{
    BinOp, BoolOp, CastKind, CondOp, Expr, FlagCond, FlagState, Intrinsic, Stmt, UnOp, mask,
    sign_extend,
};

/// The register width a target models.
///
/// This is not a property of an instruction but of the image: the same
/// `mov eax, [ebp+8]` means one thing to a 32-bit processor and does not
/// exist for a 64-bit one, which spells the same operation `mov rax,
/// [rbp+8]` and keeps its arguments in registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// 64-bit: sixteen general-purpose registers, `rax` through `r15`.
    Bits64,
    /// 32-bit: eight general-purpose registers, `eax` through `edi`.
    Bits32,
}

impl Mode {
    /// The full width of a general-purpose register, in bytes.
    pub const fn width(self) -> u8 {
        match self {
            Mode::Bits64 => 8,
            Mode::Bits32 => 4,
        }
    }

    /// True for the 32-bit mode.
    pub const fn is_32(self) -> bool {
        matches!(self, Mode::Bits32)
    }
}

/// Registers by name: name, index, size, bit offset.
const REG_TABLE: &[(&str, u8, u8, u8)] = &[
    ("rax", 0, 8, 0),
    ("eax", 0, 4, 0),
    ("ax", 0, 2, 0),
    ("al", 0, 1, 0),
    ("ah", 0, 1, 8),
    ("rcx", 1, 8, 0),
    ("ecx", 1, 4, 0),
    ("cx", 1, 2, 0),
    ("cl", 1, 1, 0),
    ("ch", 1, 1, 8),
    ("rdx", 2, 8, 0),
    ("edx", 2, 4, 0),
    ("dx", 2, 2, 0),
    ("dl", 2, 1, 0),
    ("dh", 2, 1, 8),
    ("rbx", 3, 8, 0),
    ("ebx", 3, 4, 0),
    ("bx", 3, 2, 0),
    ("bl", 3, 1, 0),
    ("bh", 3, 1, 8),
    ("rsp", 4, 8, 0),
    ("esp", 4, 4, 0),
    ("sp", 4, 2, 0),
    ("spl", 4, 1, 0),
    ("rbp", 5, 8, 0),
    ("ebp", 5, 4, 0),
    ("bp", 5, 2, 0),
    ("bpl", 5, 1, 0),
    ("rsi", 6, 8, 0),
    ("esi", 6, 4, 0),
    ("si", 6, 2, 0),
    ("sil", 6, 1, 0),
    ("rdi", 7, 8, 0),
    ("edi", 7, 4, 0),
    ("di", 7, 2, 0),
    ("dil", 7, 1, 0),
    ("r8", 8, 8, 0),
    ("r8d", 8, 4, 0),
    ("r8w", 8, 2, 0),
    ("r8b", 8, 1, 0),
    ("r9", 9, 8, 0),
    ("r9d", 9, 4, 0),
    ("r9w", 9, 2, 0),
    ("r9b", 9, 1, 0),
    ("r10", 10, 8, 0),
    ("r10d", 10, 4, 0),
    ("r10w", 10, 2, 0),
    ("r10b", 10, 1, 0),
    ("r11", 11, 8, 0),
    ("r11d", 11, 4, 0),
    ("r11w", 11, 2, 0),
    ("r11b", 11, 1, 0),
    ("r12", 12, 8, 0),
    ("r12d", 12, 4, 0),
    ("r12w", 12, 2, 0),
    ("r12b", 12, 1, 0),
    ("r13", 13, 8, 0),
    ("r13d", 13, 4, 0),
    ("r13w", 13, 2, 0),
    ("r13b", 13, 1, 0),
    ("r14", 14, 8, 0),
    ("r14d", 14, 4, 0),
    ("r14w", 14, 2, 0),
    ("r14b", 14, 1, 0),
    ("r15", 15, 8, 0),
    ("r15d", 15, 4, 0),
    ("r15w", 15, 2, 0),
    ("r15b", 15, 1, 0),
    ("rip", 16, 8, 0),
];

/// Canonical 64-bit names by index.
const REG64: [&str; 17] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12",
    "r13", "r14", "r15", "rip",
];

/// Canonical 32-bit names by index.
///
/// The eight names are the same registers, under the names a 32-bit
/// processor gives them. Keeping a second table rather than truncating the
/// first is what stops `mov eax, [ebp+8]` from being read as a write to
/// `rax`: they are the same register, and a reader of one width's listing
/// sees one width's names.
const REG32: [&str; 8] = ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"];

/// Registers of the 32-bit mode: name, index, size, bit offset.
///
/// The high-byte forms (`ah`, `ch`, …) exist here as they do in 64-bit mode.
/// The low-byte forms that need a REX prefix — `spl`, `bpl`, `sil`, `dil` —
/// do not, because a 32-bit encoding has no REX prefix to spend on them.
const REG32_TABLE: &[(&str, u8, u8, u8)] = &[
    ("eax", 0, 4, 0),
    ("ax", 0, 2, 0),
    ("al", 0, 1, 0),
    ("ah", 0, 1, 8),
    ("ecx", 1, 4, 0),
    ("cx", 1, 2, 0),
    ("cl", 1, 1, 0),
    ("ch", 1, 1, 8),
    ("edx", 2, 4, 0),
    ("dx", 2, 2, 0),
    ("dl", 2, 1, 0),
    ("dh", 2, 1, 8),
    ("ebx", 3, 4, 0),
    ("bx", 3, 2, 0),
    ("bl", 3, 1, 0),
    ("bh", 3, 1, 8),
    ("esp", 4, 4, 0),
    ("sp", 4, 2, 0),
    ("ebp", 5, 4, 0),
    ("bp", 5, 2, 0),
    ("esi", 6, 4, 0),
    ("si", 6, 2, 0),
    ("edi", 7, 4, 0),
    ("di", 7, 2, 0),
];

const RSP: u8 = 4;
const RBP: u8 = 5;
const RAX: u8 = 0;
const RCX: u8 = 1;
const RIP: u8 = 16;

/// Registers a call may overwrite, in 64-bit mode: every general-purpose
/// register that is neither callee-saved nor the stack or frame pointer.
const CALLER_SAVED: &[u8] = &[0, 1, 2, 6, 7, 8, 9, 10, 11];

/// Registers a call may overwrite in 32-bit mode.
///
/// `esi` and `edi` are in the 64-bit list and not in this one, because the
/// two conventions disagree about them: `cdecl` makes them callee-saved, and
/// marking them clobbered turned a value the callee had to preserve into
/// `<esi after a call>`.
const CALLER_SAVED_32: &[u8] = &[0, 1, 2];

/// Registers a callee has to preserve, by mode.
///
/// `push ebx` at the top of a function is the function saving a register it
/// is about to borrow, and the same instruction in the middle of a body is
/// an argument being pushed. The difference is the position, and this is the
/// half of it that the mnemonic can answer.
const fn is_callee_saved(mode: Mode, idx: u8) -> bool {
    match mode {
        Mode::Bits32 => matches!(idx, 3 | 5 | 6 | 7),
        Mode::Bits64 => matches!(idx, 3 | 5 | 12 | 13 | 14 | 15),
    }
}

/// Integer argument registers in System V order: `rdi`, `rsi`, `rdx`,
/// `rcx`, `r8`, `r9`.
pub const ARG_REGS: [u8; 6] = [7, 6, 2, 1, 8, 9];

/// Integer argument registers in Windows x64 order: `rcx`, `rdx`, `r8`,
/// `r9`.
///
/// The Windows convention shares the register file with System V but not the
/// order, so a program read with the wrong one has its arguments shuffled:
/// the second argument arrives as the first, the fourth as the second, and
/// the fifth and sixth are not passed in registers at all.
pub const WIN64_ARG_REGS: [u8; 4] = [1, 2, 8, 9];

/// Which calling convention a binary's functions follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Abi {
    /// System V, used by Linux, the BSDs and macOS on x86-64.
    SysV,
    /// The Windows x64 convention.
    Win64,
    /// 32-bit x86, arguments on the stack, caller removes them.
    Cdecl,
    /// 32-bit x86, arguments on the stack, callee removes them.
    Stdcall,
}

impl Abi {
    /// True when the callee removes the arguments from the stack.
    ///
    /// This shows up in the code as `ret 8` instead of `ret`, so the fact is
    /// in the instruction stream and does not have to be guessed from the
    /// container. It decides how a `ret imm` is read: a callee that removes
    /// eight bytes takes two four-byte arguments, which is a fact about the
    /// function's arity that the body may never mention.
    pub const fn pops_arguments(self) -> bool {
        matches!(self, Abi::Stdcall)
    }
}

/// The x86 target.
#[derive(Clone, Copy)]
pub struct X86Target {
    /// Which convention its argument registers follow.
    abi: Abi,
    /// Which register width it models.
    mode: Mode,
}

impl X86Target {
    /// The 64-bit System V target.
    pub const fn sysv() -> Self {
        X86Target {
            abi: Abi::SysV,
            mode: Mode::Bits64,
        }
    }

    /// The 64-bit Windows target.
    pub const fn win64() -> Self {
        X86Target {
            abi: Abi::Win64,
            mode: Mode::Bits64,
        }
    }

    /// The 32-bit System V target: every argument on the stack.
    pub const fn cdecl() -> Self {
        X86Target {
            abi: Abi::Cdecl,
            mode: Mode::Bits32,
        }
    }

    /// The 32-bit Windows target: every argument on the stack, removed by
    /// the callee.
    pub const fn stdcall() -> Self {
        X86Target {
            abi: Abi::Stdcall,
            mode: Mode::Bits32,
        }
    }

    /// Which width this target models.
    pub const fn mode(&self) -> Mode {
        self.mode
    }
}

/// The register a name refers to, independent of the calling convention.
fn lookup_reg(name: &str) -> Option<Reg> {
    REG_TABLE
        .iter()
        .find(|(n, _, _, _)| *n == name)
        .map(|(_, idx, size, offset)| Reg::view(*idx, *size, *offset))
}

/// The register a name refers to in a given mode.
///
/// The two modes share a register file but not a spelling, and the names are
/// not interchangeable: `rax` is not a register a 32-bit processor has, and
/// `eax` in a 64-bit listing is a view of `rax` at half width. Reading a
/// 32-bit instruction with the 64-bit table would turn `push eax` into a
/// four-byte push of a name the body never uses.
fn lookup_reg_in(mode: Mode, name: &str) -> Option<Reg> {
    match mode {
        Mode::Bits64 => lookup_reg(name),
        Mode::Bits32 => REG32_TABLE
            .iter()
            .find(|(n, _, _, _)| *n == name)
            .map(|(_, idx, size, offset)| Reg::view(*idx, *size, *offset)),
    }
}

impl Target for X86Target {
    fn lookup_reg(&self, name: &str) -> Option<Reg> {
        lookup_reg_in(self.mode, name)
    }

    fn reg_name(&self, idx: u8) -> String {
        let table = match self.mode {
            Mode::Bits64 => REG64.get(usize::from(idx)).copied(),
            Mode::Bits32 => REG32.get(usize::from(idx)).copied(),
        };
        table.unwrap_or("rax").to_string()
    }

    fn reg_width(&self, _idx: u8) -> u8 {
        self.mode.width()
    }

    fn arg_regs(&self) -> &[u8] {
        match self.abi {
            Abi::SysV => &ARG_REGS,
            Abi::Win64 => &WIN64_ARG_REGS,
            // Every argument is pushed; there is no register to name.
            Abi::Cdecl | Abi::Stdcall => &[],
        }
    }

    fn args_on_stack(&self) -> bool {
        self.mode.is_32()
    }

    fn pops_arguments(&self) -> bool {
        self.abi.pops_arguments()
    }

    fn ret_reg(&self) -> u8 {
        RAX
    }

    fn sp_reg(&self) -> u8 {
        RSP
    }

    fn fp_reg(&self) -> Option<u8> {
        Some(RBP)
    }

    fn is_pc(&self, idx: u8) -> bool {
        idx == RIP
    }

    fn sp_alias(&self) -> &'static str {
        "frame"
    }

    fn branch_cond(&self, mnemonic: &str) -> Option<FlagCond> {
        cond_of_jcc(mnemonic)
    }

    fn setcc_cond(&self, suffix: &str) -> Option<FlagCond> {
        cond_of_jcc(&format!("j{suffix}"))
    }

    fn cmov_cond(&self, suffix: &str) -> Option<FlagCond> {
        cond_of_jcc(&format!("j{suffix}"))
    }

    fn float_branch_cond(
        &self,
        mnemonic: &str,
        lhs: &Expr,
        rhs: &Expr,
        bits: u8,
    ) -> Option<Expr> {
        float_condition(mnemonic, lhs, rhs, bits)
    }

    fn parse(&self, text: &str) -> Option<Parsed> {
        parse_in(self.mode, text)
    }
}

/// The condition a branch tests, when the flags come from a
/// floating-point comparison.
///
/// The machine's compare sets three flags: below, equal, and unordered.
/// A branch asks about them in combinations, and the combinations do not
/// reduce to the integer conditions: "below" is *less than or unordered*,
/// and no integer comparison says that.
///
/// The unordered case is what makes the difference. A comparison involving
/// a not-a-number sets all three flags, so `ja` — "above", which needs
/// neither below nor equal — is false for a not-a-number, while `jbe` is
/// true.
fn float_condition(mnemonic: &str, lhs: &Expr, rhs: &Expr, bits: u8) -> Option<Expr> {
    let cmp = |op: BinOp| Expr::Bin {
        op,
        lhs: Box::new(lhs.clone()),
        rhs: Box::new(rhs.clone()),
        bits,
    };
    // Two conditions that are each zero or one; the bitwise or of two
    // booleans is their logical or.
    let either = |a: Expr, b: Expr| Expr::Bin {
        op: BinOp::Or,
        lhs: Box::new(a),
        rhs: Box::new(b),
        bits: 1,
    };
    Some(match mnemonic {
        // Equal is set for a not-a-number as well.
        "je" | "jz" => either(cmp(BinOp::FEq), cmp(BinOp::FUnordered)),
        "jne" | "jnz" => cmp(BinOp::FNe),
        // Above needs neither below nor equal, which a not-a-number is not.
        "ja" | "nbe" => cmp(BinOp::FGt),
        "jae" | "nb" | "nc" => cmp(BinOp::FGe),
        "jb" | "c" | "nae" => either(cmp(BinOp::FLt), cmp(BinOp::FUnordered)),
        "jbe" | "na" => either(cmp(BinOp::FLe), cmp(BinOp::FUnordered)),
        "jp" => cmp(BinOp::FUnordered),
        "jnp" => either(cmp(BinOp::FLe), cmp(BinOp::FGe)),
        _ => return None,
    })
}

/// Condition for a conditional branch mnemonic.
fn cond_of_jcc(mnemonic: &str) -> Option<FlagCond> {
    let suffix = mnemonic.strip_prefix('j')?;
    Some(match suffix {
        // The overflow flag on its own.
        "o" => FlagCond::O,
        "no" => FlagCond::No,
        // The carry flag on its own.
        "b" | "c" | "nae" => FlagCond::C,
        "ae" | "nb" | "nc" => FlagCond::Nc,
        // The zero flag on its own.
        "e" | "z" => FlagCond::Z,
        "ne" | "nz" => FlagCond::Nz,
        // The carry and the zero flag together.
        "be" | "na" => FlagCond::Be,
        "a" | "nbe" => FlagCond::A,
        // The sign flag on its own.
        "s" => FlagCond::S,
        "ns" => FlagCond::Ns,
        // The parity flag on its own.
        "p" | "pe" => FlagCond::Po,
        "np" | "pno" => FlagCond::Npo,
        // The sign and the overflow flag together.
        "l" | "nge" => FlagCond::L,
        "ge" | "nl" => FlagCond::Nl,
        "g" | "nle" => FlagCond::G,
        "le" | "ng" => FlagCond::Le,
        _ => return None,
    })
}

/// The zero flag of a result.
///
/// The flag is set when the result is zero, which is a question about the
/// result and not about the operands.
fn zero_flag(result: &Expr, bits: u8) -> Expr {
    Expr::Cond {
        op: CondOp::Eq,
        lhs: Box::new(result.clone()),
        rhs: Box::new(Expr::Const(0)),
        bits,
    }
}

/// Mask a shift or rotate count the way the processor does: five bits of it,
/// six when the operand is 64 bits wide. A count of 33 shifts by one, and a
/// count of 32 shifts by nothing.
fn mask_count(raw: Expr, bits: u64, width: u8) -> Expr {
    let limit = if bits >= 64 { 63 } else { 31 };
    match raw {
        Expr::Const(v) => Expr::Const(v & limit),
        raw => Expr::Bin {
            op: BinOp::And,
            lhs: Box::new(raw),
            rhs: Box::new(Expr::Const(limit)),
            bits: width,
        },
    }
}

/// Take a count modulo the number of positions the field has.
///
/// A rotation by a whole turn is a rotation by nothing, and the processor
/// changes no flag for it, so the modulo is what decides whether the flags
/// are set at all.
fn turn_mod(count: Expr, positions: u64, width: u8) -> Expr {
    match count {
        Expr::Const(v) => Expr::Const(v % positions),
        count => Expr::Bin {
            op: BinOp::UMod,
            lhs: Box::new(count),
            rhs: Box::new(Expr::Const(positions)),
            bits: width,
        },
    }
}

/// The sign flag of a result.
///
/// The flag is set when the result is negative, which is its top bit.
fn sign_flag(result: &Expr, bits: u8) -> Expr {    Expr::Cond {
        op: CondOp::Slt,
        lhs: Box::new(result.clone()),
        rhs: Box::new(Expr::Const(0)),
        bits,
    }
}

/// The parity flag of a result.
///
/// The flag is set when the low byte of the result has an even number of
/// set bits.
fn parity_flag(result: &Expr, bits: u8) -> Expr {
    Expr::Intrinsic {
        op: Intrinsic::ParityEven,
        args: vec![result.clone()],
        bits,
    }
}

/// Parse the decoder's textual form, reading register names as 64-bit ones.
pub fn parse(text: &str) -> Option<Parsed> {
    parse_in(Mode::Bits64, text)
}

/// Parse the decoder's textual form for one register width.
pub fn parse_in(mode: Mode, text: &str) -> Option<Parsed> {
    // The decoder appends resolved addresses as comments and branch deltas in
    // parentheses; neither is part of the instruction.
    let mut body = text;
    let mut pc_target = None;
    if let Some(idx) = body.find("  #") {
        pc_target = parse_int(body[idx + 3..].trim());
        body = &body[..idx];
    }
    if let Some(idx) = body.find(" (rel ") {
        body = &body[..idx];
    }
    let body = body.trim();
    if body.is_empty() {
        return None;
    }
    // A prefix changes what the instruction does: `lock` makes it atomic,
    // `rep` makes it a loop. They are not decoration, so they are kept
    // rather than dropped with the mnemonic.
    let mut prefixes: Vec<String> = Vec::new();
    let mut body = body;
    while let Some((head, rest)) = body.split_once(' ') {
        let head = head.to_ascii_lowercase();
        if !matches!(
            head.as_str(),
            "lock" | "rep" | "repe" | "repz" | "repne" | "repnz" | "bnd" | "notrack"
        ) {
            break;
        }
        prefixes.push(head);
        body = rest.trim();
    }
    let (mnemonic, rest) = match body.split_once(' ') {
        Some((m, r)) => (m, r.trim()),
        None => (body, ""),
    };
    let mut ops = Vec::new();
    if !rest.is_empty() {
        for part in split_operands(rest) {
            ops.push(parse_operand(mode, part.trim())?);
        }
    }
    Some(Parsed {
        mnemonic: mnemonic.to_ascii_lowercase(),
        ops,
        pc_target,
        prefixes,
    })
}

fn parse_operand(mode: Mode, s: &str) -> Option<Operand> {
    let (size_hint, rest) = split_size_prefix(s);
    let mut rest = rest.trim();
    let mut seg = None;
    // A segment prefix is not decoration: it says the operand is memory. The
    // decoder prints one on every absolute operand, including the ones it
    // renders without brackets, and `mov eax, ds:0xc0` is a load from 0xc0
    // rather than the number 0xc0.
    let mut addressed = false;
    // Four of the six segments are the flat model's own: their base is zero,
    // and `ds:0x40` is the address `0x40`. `fs` and `gs` really do point
    // somewhere else, which is how a thread-local access is spelled, so those
    // two are kept on the operand.
    for (prefix, name) in [
        ("fs:", Some("fs")),
        ("gs:", Some("gs")),
        ("ds:", None),
        ("es:", None),
        ("cs:", None),
        ("ss:", None),
    ] {
        if let Some(r) = rest.strip_prefix(prefix) {
            seg = name;
            rest = r.trim();
            addressed = true;
            break;
        }
    }
    if let Some(inner) = rest.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
        let mut op = parse_memory(mode, inner, size_hint.unwrap_or(0))?;
        if let Operand::Mem { seg: slot, .. } = &mut op {
            *slot = seg;
        }
        return Some(op);
    }
    if let Some(reg) = lookup_reg_in(mode, rest) {
        return Some(Operand::Reg(reg));
    }
    // A vector register is not a general-purpose register: it holds
    // elements, and the lifter reads and writes them one at a time.
    if let Some(n) = vector_index(rest, "xmm") {
        return Some(Operand::Vec(VecClass::Xmm, n));
    }
    if let Some(n) = vector_index(rest, "ymm") {
        return Some(Operand::Vec(VecClass::Ymm, n));
    }
    if let Some(n) = vector_index(rest, "mm") {
        return Some(Operand::Vec(VecClass::Mm, n));
    }
    // An x87 stack slot, `st(0)` through `st(7)`.
    if rest == "st" {
        return Some(Operand::Fpu(0));
    }
    if let Some(inner) = rest.strip_prefix("st(").and_then(|r| r.strip_suffix(')'))
        && let Some(n) = inner.parse::<u8>().ok()
        && n < 8
    {
        return Some(Operand::Fpu(n));
    }
    // A bare number is an immediate, unless the instruction said what size
    // the memory it addresses is or named a segment: `mov dword ptr ds:0xb4,
    // 1` has no brackets, and reading it as two immediates drops the store
    // entirely while still looking like an instruction that parsed.
    if let Some(v) = parse_int(rest) {
        if size_hint.is_some() || addressed {
            return Some(Operand::Mem {
                size: size_hint.unwrap_or(0),
                base: None,
                index: None,
                scale: 1,
                disp: v as i64,
                abs: Some(v),
                seg,
                base_update: None,
            });
        }
        return Some(Operand::Imm(v));
    }
    None
}

/// The register number in a vector register name, if the name is one.
fn vector_index(name: &str, prefix: &str) -> Option<u8> {
    let rest = name.strip_prefix(prefix)?;
    if rest.is_empty() {
        return None;
    }
    let n: u8 = rest.parse().ok()?;
    // `xmm` has sixteen registers, `mm` has eight.
    if prefix == "mm" { (n < 8).then_some(n) } else { (n < 16).then_some(n) }
}

/// Size keywords the decoder emits before `ptr`.
fn split_size_prefix(s: &str) -> (Option<u8>, &str) {
    for (word, size) in [
        ("byte ptr", 1u8),
        ("word ptr", 2),
        ("dword ptr", 4),
        ("qword ptr", 8),
        ("tbyte ptr", 10),
        ("xmmword ptr", 16),
    ] {
        if let Some(rest) = s.strip_prefix(word) {
            return (Some(size), rest);
        }
    }
    (None, s)
}

fn parse_memory(mode: Mode, inner: &str, size: u8) -> Option<Operand> {
    let inner = inner.trim();
    let mut base = None;
    let mut index = None;
    let mut scale = 1u64;
    let mut disp = 0i64;
    let mut absolute = false;

    let mut sign = 1i64;
    let mut token = String::new();
    let mut terms: Vec<(i64, String)> = Vec::new();
    for c in inner.chars() {
        match c {
            '+' => {
                if !token.trim().is_empty() {
                    terms.push((sign, token.trim().to_string()));
                }
                token.clear();
                sign = 1;
            }
            '-' => {
                if !token.trim().is_empty() {
                    terms.push((sign, token.trim().to_string()));
                }
                token.clear();
                sign = -1;
            }
            c => token.push(c),
        }
    }
    if !token.trim().is_empty() {
        terms.push((sign, token.trim().to_string()));
    }

    for (sign, term) in terms {
        // `eiz` and `riz` are how the assembler spells "no index register"
        // in an addressing form that has a place for one. They are not
        // registers and hold nothing; the term that names them is absent.
        if term == "eiz" || term == "riz" {
            continue;
        }
        if let Some((r, s)) = term.split_once('*') {
            let reg = lookup_reg_in(mode, r.trim())?;
            index = Some(reg);
            scale = parse_int(s).unwrap_or(1);
            continue;
        }
        if let Some(reg) = lookup_reg_in(mode, &term) {
            if base.is_none() {
                base = Some(reg);
            } else {
                index = Some(reg);
            }
            continue;
        }
        if let Some(v) = parse_int(&term) {
            disp = disp.wrapping_add(sign * v as i64);
            continue;
        }
        // A bare address with no base register.
        absolute = true;
    }

    Some(Operand::Mem {
        size,
        base,
        index,
        scale,
        disp,
        abs: if absolute { Some(disp as u64) } else { None },
        seg: None,
        base_update: None,
    })
}

/// The x86 lifter.
pub struct Lifter<'a> {
    /// Shared state.
    pub core: Core<'a>,
    /// True once the prologue has been consumed.
    pub prologue_seen: bool,
    /// The x87 stack's top, as the physical slot it lives in.
    ///
    /// Eight means the stack is empty, which is the state every function
    /// starts in: the ABI requires the x87 stack to be empty at a function
    /// boundary, so a function that uses it pushes what it needs and pops
    /// it again before it returns.
    pub fpu_top: u8,
    /// Counter for the temporary names the vector and x87 lifts need.
    temps: u32,
    /// True when the function wrote a vector register.
    wrote_vector: bool,
    /// True when the function wrote the x87 stack.
    wrote_x87: bool,
    /// The width of the last value written to `xmm0`.
    float_width: u8,
    /// The condition of a trailing loop instruction.
    ///
    /// The loop family is a conditional branch and a decrement in one
    /// instruction, so the condition is not the machine's condition codes
    /// but a question about the counter and, for two of them, about the
    /// zero flag as well.
    loop_condition: Option<Expr>,
    /// Which register width this lifter reads.
    ///
    /// Taken from the target rather than passed in: the register width *is*
    /// the mode, and two sources for one fact drift.
    mode: Mode,
    /// Values pushed since the call being lifted, oldest first.
    ///
    /// A 32-bit call is `push arg2; push arg1; call f`, so the values
    /// between one call and the next are that call's arguments, and the
    /// last one pushed is the first one passed. The list is emptied when the
    /// call is lifted, because the next call pushes its own.
    pushed: Vec<Expr>,
    /// True once the prologue is over.
    ///
    /// The prologue of a 32-bit function is a run of `push ebx`, `push esi`
    /// and so on that saves callee-saved registers, and those pushes are not
    /// arguments. The run ends at the first instruction that is not a push;
    /// after that a push of a callee-saved register is a value the program
    /// is passing.
    frame_ready: bool,
    /// Bytes of argument the epilogue removes, from `ret 8`.
    ///
    /// A callee that removes its own arguments states its arity in the
    /// instruction, whether or not the body reads all of them.
    pops: Option<u64>,
}

/// How many arguments a 32-bit function can be handed before the count stops
/// being worth naming.
///
/// Reading past the ones the body mentions would invent parameters; stopping
/// too early leaves a read of `arg_28_u32` unnamed. Eight is where the two
/// errors meet for code a compiler produces, and a reader can still see the
/// offset in the name beyond it.
const MAX_STACK_ARGS: usize = 8;

impl<'a> Lifter<'a> {
    /// Build a lifter.
    pub fn new(
        target: &'static dyn Target,
        name_of: &'a dyn Fn(Addr) -> String,
        symbols: &'a dyn Fn(&str) -> Option<u64>,
    ) -> Self {
        Lifter {
            core: Core::new(target, name_of).with_symbols(symbols),
            prologue_seen: false,
            fpu_top: 8,
            temps: 0,
            wrote_vector: false,
            wrote_x87: false,
            float_width: 0,
            loop_condition: None,
            mode: if target.reg_width(0) == 4 {
                Mode::Bits32
            } else {
                Mode::Bits64
            },
            pushed: Vec::new(),
            frame_ready: false,
            pops: None,
        }
    }

    /// The full width of a general-purpose register, in bytes.
    fn width(&self) -> u8 {
        self.mode.width()
    }

    /// The canonical name of a register at this target's width.
    fn reg(&self, idx: u8) -> String {
        self.core.name(idx)
    }

    /// Add a value to a register, in place.
    ///
    /// The string instructions move their pointers by the width of one
    /// element, and both the name and the width of the arithmetic come from
    /// the mode: `esi` and `edi` in 32-bit mode wrap at four bytes.
    fn bump(&mut self, idx: u8, value: Expr, stmts: &mut Vec<Stmt>) {
        let name = self.reg(idx);
        let sum = Expr::Bin {
            op: BinOp::Add,
            lhs: Box::new(Expr::var(name.clone())),
            rhs: Box::new(value),
            bits: self.width(),
        };
        self.core.assign(&name, sum, stmts);
    }

    /// The arguments a call at this point passes.
    ///
    /// A 32-bit call passes the values pushed since the last one, and the
    /// last value pushed is the first argument: the conventions that use the
    /// stack push the arguments right to left. The register architectures
    /// pass only the argument registers this function has already set, which
    /// is what a call in the middle of a body needs.
    fn args_for_call(&self) -> Vec<Expr> {
        if self.core.target.args_on_stack() {
            return self.pushed.iter().rev().cloned().collect();
        }
        self.core.call_args()
    }

    /// The arguments of the call being lifted, which consumes them.
    ///
    /// The next call in this block pushes its own arguments, and a list that
    /// kept these would give it a phantom argument per call already made.
    fn take_args_for_call(&mut self) -> Vec<Expr> {
        if self.core.target.args_on_stack() {
            return std::mem::take(&mut self.pushed).into_iter().rev().collect();
        }
        self.core.call_args()
    }

    /// The count register times the width of one element.
    fn count_times(&self, width: u8) -> Expr {
        let count = Expr::var(self.reg(RCX));
        if width == 1 {
            // A byte count is the count. Multiplying it by one says the same
            // thing more slowly.
            return count;
        }
        Expr::Bin {
            op: BinOp::Mul,
            lhs: Box::new(count),
            rhs: Box::new(Expr::Const(u64::from(width))),
            bits: self.width(),
        }
    }

    /// Give every memory operand that states no width the width the
    /// instruction works at.
    ///
    /// The decoder prints `dword ptr` when nothing else in the encoding says
    /// how wide the access is, and leaves it out when a register operand
    /// already says: `mov eax, ds:0xc0` loads four bytes because `eax` is
    /// four bytes wide. A memory operand with no stated width and no register
    /// to borrow one from is one machine word — that is what `push [esp+8]`
    /// moves, and in 32-bit mode a word is four bytes, not eight. Reading it
    /// at eight is how `return sink;` became `return *(u64*)0xc0;`.
    fn size_operands(&self, ops: &mut [Operand]) {
        let from_reg = ops.iter().find_map(|o| match o {
            Operand::Reg(r) => Some(r.size),
            Operand::Modified { reg, .. } => Some(reg.size),
            _ => None,
        });
        let width = from_reg.unwrap_or_else(|| self.width());
        for op in ops.iter_mut() {
            if let Operand::Mem { size, .. } = op
                && *size == 0
            {
                *size = width;
            }
        }
    }

    /// A name for a value that is computed once and used twice.
    ///
    /// A 64-bit vector element is two 32-bit elements, so writing it back
    /// needs the value twice: once for its low half and once for its high
    /// half. Naming it keeps the second use from reading the first use's
    /// result.
    fn new_temp(&mut self) -> String {
        self.temps += 1;
        format!("v{}", self.temps)
    }

    /// A name for a value that has to survive a write to where it came
    /// from.
    ///
    /// The IR reads variables by name, so a value captured before the slot
    /// it came from is written is read again afterwards would see the new
    /// value. A register exchange, and a read-modify-write that keeps the
    /// old value, both need the old value after the write, so they go
    /// through one of these.
    fn new_temp_named(&mut self, prefix: &str) -> String {
        self.temps += 1;
        format!("{prefix}{}", self.temps)
    }

    fn lift_insn(&mut self, insn: &Instruction, out: &mut Lifted) {
        let Some(mut p) = self.core.target.parse(&insn.text) else {
            out.stmts
                .push(Stmt::Comment(format!("{} ; unparsed", insn.text)));
            return;
        };
        self.core.pc_target = p
            .pc_target
            .or_else(|| insn.mem_reads.first().map(|a| a.0))
            .or_else(|| insn.mem_writes.first().map(|a| a.0));
        self.size_operands(&mut p.ops);
        let m = p.mnemonic.as_str();
        // A push at the very front of a function is the function saving a
        // register; once anything else has happened, a push is a value on
        // its way to a call. The boundary markers a compiler emits at an
        // entry point are not "anything else", so they keep it open.
        if !matches!(m, "push" | "nop" | "endbr64" | "endbr32" | "pause" | "fnop") {
            self.frame_ready = true;
        }
        // The instructions whose shape the general-purpose path cannot
        // describe: vector registers, the x87 stack, atomic
        // read-modify-writes, and the string instructions with their
        // implicit registers.
        let repeated = p
            .prefixes
            .iter()
            .any(|x| x == "rep" || x == "repe" || x == "repz");
        if self.lift_vec(m, &p.ops, &mut out.stmts)
            || self.lift_x87(m, &p.ops, &mut out.stmts)
            || self.lift_atomic(m, &p.ops, &mut out.stmts)
            || self.lift_string(m, &p.ops, repeated, insn.addr, &mut out.stmts)
        {
            return;
        }
        match m {
            "nop" | "endbr64" | "endbr32" | "pause" | "hlt" | "ud2" => {}
            "mov" | "movabs" => self.lift_mov(&p.ops, &mut out.stmts),
            "movzx" => self.lift_widening(&p.ops, false, &mut out.stmts),
            "movsx" | "movsxd" => self.lift_widening(&p.ops, true, &mut out.stmts),
            "lea" => self.lift_lea(&p.ops, &mut out.stmts),
            "add" => self.lift_alu(BinOp::Add, &p.ops, &mut out.stmts),
            "sub" => self.lift_alu(BinOp::Sub, &p.ops, &mut out.stmts),
            "and" => self.lift_alu(BinOp::And, &p.ops, &mut out.stmts),
            "or" => self.lift_alu(BinOp::Or, &p.ops, &mut out.stmts),
            "xor" => self.lift_alu(BinOp::Xor, &p.ops, &mut out.stmts),
            "adc" => self.lift_carry_arith(true, &p.ops, &mut out.stmts),
            "sbb" => self.lift_carry_arith(false, &p.ops, &mut out.stmts),
            "inc" => self.lift_inc_dec(BinOp::Add, &with_imm(&p.ops, 1), &mut out.stmts),
            "dec" => self.lift_inc_dec(BinOp::Sub, &with_imm(&p.ops, 1), &mut out.stmts),
            "neg" => self.lift_unary(UnOp::Neg, &p.ops, &mut out.stmts),
            "not" => self.lift_unary(UnOp::Not, &p.ops, &mut out.stmts),
            "shl" | "sal" => self.lift_shift(BinOp::Shl, &p.ops, &mut out.stmts),
            "shr" => self.lift_shift(BinOp::Shr, &p.ops, &mut out.stmts),
            "sar" => self.lift_shift(BinOp::Sar, &p.ops, &mut out.stmts),
            "imul" | "mul" => self.lift_imul(m, &p.ops, &mut out.stmts),
            "loop" | "loope" | "loopne" | "jrcxz" | "jecxz" => {
                self.lift_loop(m, &mut out.stmts)
            }
            "cmp" => self.lift_cmp(&p.ops, false),
            "test" => self.lift_cmp(&p.ops, true),
            "sete" | "setz" | "setne" | "setnz" | "setl" | "setle" | "setg" | "setge" | "setb"
            | "setbe" | "seta" | "setae" | "sets" | "setns" => {
                self.lift_setcc(m, &p.ops, &mut out.stmts)
            }
            "cmove" | "cmovz" | "cmovne" | "cmovnz" | "cmovl" | "cmovle" | "cmovg" | "cmovge"
            | "cmovb" | "cmovbe" | "cmova" | "cmovae" | "cmovs" | "cmovns" => {
                self.lift_cmov(m, &p.ops, &mut out.stmts)
            }
            "push" => self.lift_push(&p.ops, &mut out.stmts),
            "pop" => self.lift_pop(&p.ops, &mut out.stmts),
            "leave" => {
                // `leave` is `mov rsp, rbp` followed by `pop rbp`, so the
                // stack pointer ends one width above where the frame pointer
                // was anchored — back at the frame base when the prologue was
                // the usual `push rbp; mov rbp, rsp`.
                self.core.sp_off = self
                    .core
                    .fp_off
                    .map_or(0, |off| off + i64::from(self.width()));
                self.core.fp_off = None;
            }
            "cdqe" => self.lift_cast(RAX, 4, 8, &mut out.stmts),
            "cwde" => self.lift_cast(RAX, 2, 4, &mut out.stmts),
            "cdq" | "cqo" | "cwd" => {
                out.stmts
                    .push(Stmt::Comment(format!("{m} ; sign extension of rax")));
            }
            "xchg" => self.lift_xchg(&p.ops, &mut out.stmts),
            "rol" | "ror" => self.lift_rotate(m, &p.ops, &mut out.stmts),
            "rcl" | "rcr" => self.lift_rotate_carry(m, &p.ops, &mut out.stmts),
            "bswap" => self.lift_bswap(&p.ops, &mut out.stmts),
            "bsr" | "bsf" | "tzcnt" | "lzcnt" => self.lift_bit_scan(m, &p.ops, &mut out.stmts),
            "bt" | "bts" | "btr" | "btc" => self.lift_bit_test(m, &p.ops, &mut out.stmts),
            "div" | "idiv" => self.lift_div(m, &p.ops, &mut out.stmts),
            "shld" | "shrd" => self.lift_double_shift(m, &p.ops, &mut out.stmts),
            "syscall" | "sysenter" => self.lift_syscall(insn.addr, &mut out.stmts),
            "int" => self.lift_int(&p.ops, insn.addr, &mut out.stmts),
            "rdtsc" | "rdtscp" => self.lift_timestamp(insn.addr, m, &mut out.stmts),
            "clc" | "stc" | "cmc" | "cld" | "std" | "sti" | "cli" => {
                self.lift_flag_instruction(m, insn.addr, &mut out.stmts)
            }
            "call" => self.lift_call(insn, &p.ops, &mut out.stmts),
            "jmp" => self.lift_jmp(&p.ops, &mut out.stmts),
            "ret" | "retn" => self.lift_ret(&p.ops, insn.addr, &mut out.stmts),
            other => {
                if cond_of_jcc(other).is_some() {
                    // Handled by take_condition once the block ends.
                } else {
                    out.stmts
                        .push(Stmt::Comment(format!("{} ; not lifted", insn.text)));
                }
            }
        }
    }

    fn lift_mov(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        // `mov rbp, rsp` anchors the frame instead of assigning a value.
        if let (Operand::Reg(dst), Operand::Reg(src)) = (&ops[0], &ops[1])
            && dst.idx == RBP
            && src.idx == RSP
        {
            self.core.fp_off = Some(self.core.sp_off);
            self.core.fp_anchor = Some(self.core.sp_off);
            self.prologue_seen = true;
            return;
        }
        let value = self.snapshot(&ops[1], stmts);
        self.core.write(&ops[0], value, stmts);
    }

    /// `movzx`/`movsx`: read the source at its own width, then widen.
    fn lift_widening(&mut self, ops: &[Operand], signed: bool, stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let src = self.snapshot(&ops[1], stmts);
        let from = ops[1].size();
        let to = ops[0].size().max(from);
        let value = Expr::Cast {
            kind: if signed { CastKind::Sext } else { CastKind::Zext },
            arg: Box::new(src),
            from,
            to,
        };
        self.core.write(&ops[0], value, stmts);
    }

    fn lift_lea(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let Operand::Mem { base, index, disp, .. } = &ops[1] else {
            return;
        };
        // `lea rsp, [rbp-0x8]` restores the stack pointer; keep the frame in
        // sync instead of emitting an assignment.
        if let Operand::Reg(dst) = &ops[0]
            && dst.idx == RSP
            && index.is_none()
            && let Some(b) = base
            && b.idx == RBP
            && let Some(off) = self.core.fp_off
        {
            self.core.sp_off = off + disp;
            return;
        }
        let addr = self.core.addr_of(&ops[1], stmts);
        self.core.write(&ops[0], addr, stmts);
    }

    fn lift_alu(&mut self, op: BinOp, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        // Stack pointer arithmetic is frame bookkeeping, not a value.
        if let Operand::Reg(dst) = &ops[0]
            && dst.idx == RSP
            && let Operand::Imm(v) = &ops[1]
        {
            match op {
                BinOp::Sub => self.core.adjust_sp(-(*v as i64)),
                BinOp::Add => self.core.adjust_sp(*v as i64),
                _ => self.core.dynamic_sp = true,
            }
            return;
        }
        let bits = ops[0].size();
        // `xor eax, eax` and `sub eax, eax` clear a register without reading
        // it, and they are how a compiler writes a zero. Naming both operands
        // first turned the cancellation into a comparison between a value and
        // a copy of itself, which the simplifier cannot see, and a symbolic
        // executor then asks what the register held before a write that does
        // not depend on it.
        if matches!(op, BinOp::Xor | BinOp::Sub) && ops[0] == ops[1] {
            let zero = Expr::Const(0);
            let result = self.core.dest_value(&ops[0]);
            let flags = self.alu_flags(op, &zero, &zero, &result, bits);
            self.core.write(&ops[0], zero, stmts);
            self.core.set_flags(flags, None);
            return;
        }
        // An immediate is a bit pattern at the operand's width, and the
        // arithmetic is modular at that width. `add edi, 0xffffff9c` adds
        // minus a hundred, not plus four billion; reading the pattern as an
        // unsigned 64-bit number turns `index - 100` into `index + 4294967196`
        // and every case label derived from it is off by 2^32. Sign-extending
        // to 64 bits keeps the modular value and keeps the offset small enough
        // for a human (and a case label) to read. Widths above four bytes are
        // already sign-extended by the decoder.
        let operand = if let Operand::Imm(v) = &ops[1] {
            Expr::Const(sign_extend(*v, bits) as u64)
        } else {
            self.snapshot(&ops[1], stmts)
        };
        // The carry and the overflow say things about the value that was
        // there before the operation, so it is given a name of its own.
        // The logical operations say nothing about the old value, and they
        // are the common ones, so they do not pay for the name.
        let previous = if matches!(op, BinOp::Add | BinOp::Sub) {
            self.snapshot(&ops[0], stmts)
        } else {
            self.core.read(&ops[0], stmts)
        };
        let value = Expr::Bin {
            op,
            lhs: Box::new(previous.clone()),
            rhs: Box::new(operand.clone()),
            bits,
        };
        let result = self.core.dest_value(&ops[0]);
        let flags = self.alu_flags(op, &previous, &operand, &result, bits);
        self.core.write(&ops[0], value, stmts);
        self.core.set_flags(flags, None);
    }

    /// `adc` and `sbb`: the arithmetic that arrives with a carry.
    ///
    /// The carry is a value, not a mode. Modelling it as `add` with a note
    /// that the carry was ignored makes a loop that adds two multi-word
    /// numbers produce the wrong answer at every word after the first, and the
    /// note does not travel with the value it corrupts.
    fn lift_carry_arith(&mut self, is_adc: bool, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let bits = ops[0].size();
        let previous = self.snapshot(&ops[0], stmts);
        let operand = self.snapshot(&ops[1], stmts);
        let carry = self.carry_in();
        let value = Expr::Intrinsic {
            op: if is_adc {
                Intrinsic::AdcSum
            } else {
                Intrinsic::SbbDiff
            },
            args: vec![previous.clone(), operand.clone(), carry.clone()],
            bits,
        };
        self.core.write(&ops[0], value, stmts);
        let result = self.core.dest_value(&ops[0]);
        let cf = Expr::Intrinsic {
            op: if is_adc {
                Intrinsic::AdcCarry
            } else {
                Intrinsic::SbbBorrow
            },
            args: vec![previous.clone(), operand.clone(), carry],
            bits,
        };
        let of = Expr::Intrinsic {
            op: if is_adc {
                Intrinsic::OverflowAdd
            } else {
                Intrinsic::OverflowSub
            },
            args: vec![previous, operand, result.clone()],
            bits,
        };
        self.core.set_flags(
            FlagState {
                cf: Some(cf),
                zf: Some(zero_flag(&result, bits)),
                sf: Some(sign_flag(&result, bits)),
                of: Some(of),
                pf: Some(parity_flag(&result, bits)),
            },
            None,
        );
    }

    /// The carry flag as a value, for the instructions that add it in.
    ///
    /// An instruction that arrives before any operation set the carry has no
    /// value to add. Saying so is better than adding zero: a caller reading
    /// the output can see the gap, and the interpreter refuses to invent one.
    fn carry_in(&self) -> Expr {
        match &self.core.flags.cf {
            Some(e) => e.clone(),
            None => Expr::Unknown("carry-in flag is not defined on this path".into()),
        }
    }

    /// The flags a binary arithmetic operation leaves behind.
    ///
    /// The carry and the overflow are the two the machine computes from the
    /// operands rather than from the result. The logical operations clear
    /// both, the addition sets the carry when the sum did not fit, and the
    /// subtraction sets it when the difference did not.
    fn alu_flags(
        &self,
        op: BinOp,
        previous: &Expr,
        operand: &Expr,
        result: &Expr,
        bits: u8,
    ) -> FlagState {
        let (cf, of) = match op {
            BinOp::Add => (
                Some(Expr::Intrinsic {
                    op: Intrinsic::Carry,
                    args: vec![result.clone(), previous.clone()],
                    bits,
                }),
                Some(Expr::Intrinsic {
                    op: Intrinsic::OverflowAdd,
                    args: vec![previous.clone(), operand.clone(), result.clone()],
                    bits,
                }),
            ),
            BinOp::Sub => (
                Some(Expr::Intrinsic {
                    op: Intrinsic::Borrow,
                    args: vec![previous.clone(), operand.clone()],
                    bits,
                }),
                Some(Expr::Intrinsic {
                    op: Intrinsic::OverflowSub,
                    args: vec![previous.clone(), operand.clone(), result.clone()],
                    bits,
                }),
            ),
            // The logical operations clear the carry and the overflow.
            BinOp::And | BinOp::Or | BinOp::Xor => {
                (Some(Expr::Const(0)), Some(Expr::Const(0)))
            }
            // The division operations leave the flags alone, and the
            // multiplication ones set them from a product that does not
            // fit, which the caller works out.
            _ => (None, None),
        };
        FlagState {
            cf,
            zf: Some(zero_flag(result, bits)),
            sf: Some(sign_flag(result, bits)),
            of,
            pf: Some(parity_flag(result, bits)),
        }
    }

    /// A name for the value an operand holds before this instruction writes
    /// it.
    ///
    /// The IR reads variables by name, so a value captured before the
    /// register it came from is written is read again afterwards would see
    /// the new value. The carry of an addition is the sum that did not fit,
    /// and the overflow of a subtraction is the difference that did not,
    /// and both are about the value that was there before.
    fn snapshot(&mut self, op: &Operand, stmts: &mut Vec<Stmt>) -> Expr {
        let value = self.core.read(op, stmts);
        // Every operand is materialised, not just a register. `xchg [rax], cl`
        // writes memory and then reads the old value back out of the same
        // location; without a name for the first read, the second one sees the
        // store and the exchange turns into a copy.
        self.name(value, stmts)
    }

    /// Give a value a name, so a store that follows cannot change what it
    /// reads.
    fn name(&mut self, value: Expr, stmts: &mut Vec<Stmt>) -> Expr {
        let name = self.new_temp_named("old_");
        self.core.assign(&name, value, stmts);
        Expr::var(name)
    }

    /// `inc` and `dec`: the same arithmetic, with the carry left alone.
    ///
    /// The machine keeps the carry flag through an increment and a
    /// decrement, because a loop that counts does not want its carry
    /// disturbed. The overflow is the one flag that changes, and it can be
    /// read off the result: an increment overflows when the result is the
    /// smallest value the width can hold, and a decrement overflows when
    /// the result is the largest one.
    fn lift_inc_dec(&mut self, op: BinOp, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let bits = ops[0].size();
        let operand = self.snapshot(&ops[1], stmts);
        let previous = self.core.read(&ops[0], stmts);
        let value = Expr::Bin {
            op,
            lhs: Box::new(previous),
            rhs: Box::new(operand),
            bits,
        };
        let result = self.core.dest_value(&ops[0]);
        let limit = 1u64 << (u32::from(bits) * 8 - 1);
        let of = if op == BinOp::Add {
            // The smallest value of the width.
            Expr::Cond {
                op: CondOp::Eq,
                lhs: Box::new(result.clone()),
                rhs: Box::new(Expr::Const(limit)),
                bits,
            }
        } else {
            // The largest value of the width.
            Expr::Cond {
                op: CondOp::Eq,
                lhs: Box::new(result.clone()),
                rhs: Box::new(Expr::Const(limit - 1)),
                bits,
            }
        };
        let mut flags = self.alu_flags(op, &Expr::Const(0), &Expr::Const(1), &result, bits);
        flags.of = Some(of);
        // The carry is not the carry of the addition: the machine keeps it.
        flags.cf = None;
        self.core.write(&ops[0], value, stmts);
        self.core.set_flags(flags, None);
    }

    fn lift_unary(&mut self, op: UnOp, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let Some(op0) = ops.first() else { return };
        let value = self.core.read(op0, stmts);
        let size = op0.size();
        let value = Expr::Un {
            op,
            arg: Box::new(value),
            bits: size,
        };
        self.core.write(op0, value, stmts);
        // `not` leaves the flags alone. `neg` sets them, and both of the
        // flags it sets can be read off the result: the carry says the
        // value was not zero, and the overflow says the value was the
        // smallest one the width can hold, which is the only value whose
        // negation is itself.
        if op == UnOp::Neg {
            let result = self.core.dest_value(op0);
            let limit = 1u64 << (u32::from(size) * 8 - 1);
            let flags = FlagState {
                cf: Some(Expr::Cond {
                    op: CondOp::Ne,
                    lhs: Box::new(result.clone()),
                    rhs: Box::new(Expr::Const(0)),
                    bits: size,
                }),
                zf: Some(zero_flag(&result, size)),
                sf: Some(sign_flag(&result, size)),
                of: Some(Expr::Cond {
                    op: CondOp::Eq,
                    lhs: Box::new(result.clone()),
                    rhs: Box::new(Expr::Const(limit)),
                    bits: size,
                }),
                pf: Some(parity_flag(&result, size)),
            };
            self.core.set_flags(flags, None);
        }
    }

    fn lift_shift(&mut self, op: BinOp, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let bits = ops[0].size();
        // The carry is the bit that was shifted out, so the value before the
        // shift has to survive it.
        let previous = self.snapshot(&ops[0], stmts);
        // An immediate count is a number, and a number needs no protection
        // from the write this instruction is about to make. Copying it into a
        // temporary hid the value from the flag rules, which then took the
        // path for an unknown count and worked the carry out from the shifted
        // result instead of the value that was shifted.
        //
        // The count itself is masked before it is used: the processor reads
        // five bits of it, six for a 64-bit operand, so a count of 33 shifts
        // by one and a count of 32 does nothing.
        let limit = if bits >= 8 { 63 } else { 31 };
        let count = match &ops[1] {
            Operand::Imm(v) => Expr::Const(v & limit),
            _ => {
                let raw = self.snapshot(&ops[1], stmts);
                Expr::Bin {
                    op: BinOp::And,
                    lhs: Box::new(raw),
                    rhs: Box::new(Expr::Const(limit)),
                    bits,
                }
            }
        };
        let value = Expr::Bin {
            op,
            lhs: Box::new(previous.clone()),
            rhs: Box::new(count.clone()),
            bits,
        };
        let result = self.core.dest_value(&ops[0]);
        let flags = self.shift_flags(op, &previous, &count, &result, bits);
        self.core.write(&ops[0], value, stmts);
        if let Some(flags) = flags {
            self.core.set_flags(flags, None);
        }
    }

    /// The flags a shift leaves behind.
    ///
    /// The carry is the last bit the shift pushed out: bit `width - count` of
    /// the value before a left shift, bit `count - 1` of it before a right one.
    /// Both amounts are subtracted, so a count larger than the width wraps to a
    /// huge amount, and the interpreter reads a shift that large as "every bit
    /// is gone". A count past the width therefore carries out a zero without a
    /// case of its own, which is what the processor does.
    ///
    /// An arithmetic right shift is the exception: once the count passes the
    /// width it keeps producing the sign bit, so the carry is taken from an
    /// arithmetic shift rather than a logical one and lands on the sign by
    /// itself.
    ///
    /// The overflow of a left shift is the sign of the result against the
    /// carry, and measuring it on the processor showed the rule holds for every
    /// count, not only the documented one. A logical right shift defines the
    /// overflow for a count of one, where it is the sign of the value that was
    /// shifted, and clears it otherwise. An arithmetic right shift clears it
    /// always.
    ///
    /// A count of zero leaves every flag alone: the known-count path returns
    /// nothing, and the unknown-count path guards each flag so that a zero
    /// count stays unbound rather than inventing a value.
    fn shift_flags(
        &self,
        op: BinOp,
        previous: &Expr,
        count: &Expr,
        result: &Expr,
        bits: u8,
    ) -> Option<FlagState> {
        let binary = |op: BinOp, lhs: Expr, rhs: Expr| Expr::Bin {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            bits,
        };
        let carry = |count: &Expr| {
            let amount = match op {
                BinOp::Shl => binary(BinOp::Sub, Expr::Const(u64::from(bits) * 8), count.clone()),
                _ => binary(BinOp::Sub, count.clone(), Expr::Const(1)),
            };
            let shifted = match op {
                BinOp::Sar => binary(BinOp::Sar, previous.clone(), amount),
                _ => binary(BinOp::Shr, previous.clone(), amount),
            };
            binary(BinOp::And, shifted, Expr::Const(1))
        };
        let overflow = |known: Option<u64>, count: &Expr, carry: &Expr| match op {
            BinOp::Shl => Expr::Cond {
                op: CondOp::Ne,
                lhs: Box::new(sign_flag(result, bits)),
                rhs: Box::new(carry.clone()),
                bits,
            },
            BinOp::Shr => match known {
                Some(1) => sign_flag(previous, bits),
                Some(_) => Expr::Const(0),
                None => Expr::Select {
                    cond: Box::new(Expr::Cond {
                        op: CondOp::Eq,
                        lhs: Box::new(count.clone()),
                        rhs: Box::new(Expr::Const(1)),
                        bits,
                    }),
                    yes: Box::new(sign_flag(previous, bits)),
                    no: Box::new(Expr::Const(0)),
                },
            },
            _ => Expr::Const(0),
        };
        let state = |known: Option<u64>| {
            let carry = carry(count);
            let of = overflow(known, count, &carry);
            FlagState {
                cf: Some(carry),
                zf: Some(zero_flag(result, bits)),
                sf: Some(sign_flag(result, bits)),
                of: Some(of),
                pf: Some(parity_flag(result, bits)),
            }
        };
        // A count that is known makes every flag exact, and the common shifts
        // name their count.
        if let Expr::Const(c) = count {
            // The machine leaves every flag alone.
            if *c == 0 {
                return None;
            }
            return Some(state(Some(*c)));
        }
        // A count that is not known can be zero, and a shift of zero leaves
        // every flag alone, so the flags that would change are guarded.
        let is_zero = Expr::Cond {
            op: CondOp::Eq,
            lhs: Box::new(binary(
                BinOp::And,
                count.clone(),
                Expr::Const(u64::from(u32::from(bits) * 8 - 1)),
            )),
            rhs: Box::new(Expr::Const(0)),
            bits,
        };
        let guarded = |value: Expr| Expr::Select {
            cond: Box::new(is_zero.clone()),
            yes: Box::new(Expr::Unknown("flags".into())),
            no: Box::new(value),
        };
        let state = state(None);
        Some(FlagState {
            cf: state.cf.map(guarded),
            zf: state.zf.map(guarded),
            sf: state.sf.map(guarded),
            of: state.of.map(guarded),
            pf: state.pf.map(guarded),
        })
    }

    fn lift_imul(&mut self, m: &str, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        // `mul` and `imul` share an encoding, and the difference is entirely in
        // the top half of the product and in the flags: a signed product of two
        // negatives is positive, so the upper half is not the upper half of the
        // unsigned product.
        let signed = m == "imul";
        match ops.len() {
            1 => {
                // The one-operand form produces twice the operand width: `ax`
                // for a byte, `dx:ax` for a word, `edx:eax` for a dword and
                // `rdx:rax` for a qword. Keeping only the low half is exact
                // for the low half and wrong above it, which is where the
                // answer to "did the product fit" lives.
                let size = ops[0].size();
                let full = self.reg(0);
                let high_half = self.reg(2);
                let width = self.width();
                let accumulator = self.snapshot(&Operand::Reg(Reg::view(0, size, 0)), stmts);
                let operand = self.snapshot(&ops[0], stmts);
                let (lo, hi) = self.product(&accumulator, &operand, size, signed);
                let (cf, of) = self.mul_flags(&lo, &hi, size, signed);
                if size == 1 {
                    // `al * r/m8` lands in `ax`, both halves in one register.
                    let high = Expr::Bin {
                        op: BinOp::Shl,
                        lhs: Box::new(hi),
                        rhs: Box::new(Expr::Const(8)),
                        bits: 2,
                    };
                    let combined = Expr::Bin {
                        op: BinOp::Or,
                        lhs: Box::new(lo),
                        rhs: Box::new(high),
                        bits: 2,
                    };
                    let written = merge_write(&full, combined, 2, 0, width);
                    self.core.assign(&full, written, stmts);
                } else if size == self.width() || size == 4 {
                    // The product fills both registers, so each write replaces
                    // the whole register. A dword product lands in `eax` and
                    // `edx`; on a 64-bit target writing those clears the upper
                    // halves of `rax` and `rdx`, and a merged write would keep
                    // the old bits and report a product the instruction never
                    // produced. On a 32-bit target `eax` *is* the register, so
                    // there is nothing above it to clear.
                    let widen = |value: Expr| {
                        if size == 4 && width == 8 {
                            Expr::Cast {
                                kind: CastKind::Zext,
                                arg: Box::new(value),
                                from: 4,
                                to: 8,
                            }
                        } else {
                            value
                        }
                    };
                    self.core.assign(&full, widen(lo), stmts);
                    self.core.assign(&high_half, widen(hi), stmts);
                } else {
                    // A 16-bit product lands in `ax` and `dx`, which are views
                    // of two registers that keep their upper bits.
                    let low = merge_write(&full, lo, size, 0, width);
                    self.core.assign(&full, low, stmts);
                    let high = merge_write(&high_half, hi, size, 0, width);
                    self.core.assign(&high_half, high, stmts);
                }
                self.core.update_flags(FlagState {
                    cf,
                    zf: None,
                    sf: None,
                    of,
                    pf: None,
                });
            }
            2 => {
                let previous = self.snapshot(&ops[0], stmts);
                let operand = self.snapshot(&ops[1], stmts);
                let size = ops[0].size();
                let product = Expr::Bin {
                    op: BinOp::Mul,
                    lhs: Box::new(previous.clone()),
                    rhs: Box::new(operand.clone()),
                    bits: size,
                };
                let (lo, hi) = self.product(&previous, &operand, size, signed);
                let (cf, of) = self.mul_flags(&lo, &hi, size, signed);
                self.core.write(&ops[0], product, stmts);
                self.core.update_flags(FlagState {
                    cf,
                    zf: None,
                    sf: None,
                    of,
                    pf: None,
                });
            }
            3 => {
                let rhs = self.snapshot(&ops[2], stmts);
                let operand = self.snapshot(&ops[1], stmts);
                let size = ops[0].size();
                let product = Expr::Bin {
                    op: BinOp::Mul,
                    lhs: Box::new(operand.clone()),
                    rhs: Box::new(rhs.clone()),
                    bits: size,
                };
                let (lo, hi) = self.product(&operand, &rhs, size, signed);
                let (cf, of) = self.mul_flags(&lo, &hi, size, signed);
                self.core.write(&ops[0], product, stmts);
                self.core.update_flags(FlagState {
                    cf,
                    zf: None,
                    sf: None,
                    of,
                    pf: None,
                });
            }
            _ => {}
        }
    }

    /// The two halves of the product of two values at a given width.
    fn product(&self, a: &Expr, b: &Expr, size: u8, signed: bool) -> (Expr, Expr) {
        let (lo_op, hi_op) = if signed {
            (Intrinsic::IMulLo, Intrinsic::IMulHi)
        } else {
            (Intrinsic::MulLo, Intrinsic::MulHi)
        };
        let half = |op: Intrinsic| Expr::Intrinsic {
            op,
            args: vec![a.clone(), b.clone()],
            bits: size,
        };
        (half(lo_op), half(hi_op))
    }

    /// The carry and the overflow of a multiplication.
    ///
    /// Both flags say the same thing: the product did not fit in the width of
    /// the destination. For an unsigned multiply that is an upper half that is
    /// not zero. For a signed one it is an upper half that is not the sign of
    /// the lower half, which is why a negative product that fitted still has an
    /// upper half full of ones and must not be called an overflow.
    fn mul_flags(
        &self,
        lo: &Expr,
        hi: &Expr,
        bits: u8,
        signed: bool,
    ) -> (Option<Expr>, Option<Expr>) {
        let expected = if signed {
            // One when the low half is negative, zero when it is not: the
            // sign, spread over the whole width without a branch.
            Expr::Bin {
                op: BinOp::Sub,
                lhs: Box::new(Expr::Const(0)),
                rhs: Box::new(Expr::Bin {
                    op: BinOp::Shr,
                    lhs: Box::new(lo.clone()),
                    rhs: Box::new(Expr::Const(u64::from(bits) * 8 - 1)),
                    bits,
                }),
                bits,
            }
        } else {
            Expr::Const(0)
        };
        let overflow = Expr::Cond {
            op: CondOp::Ne,
            lhs: Box::new(hi.clone()),
            rhs: Box::new(expected),
            bits,
        };
        (Some(overflow.clone()), Some(overflow))
    }

    /// The loop family: count down and branch on the counter.
    ///
    /// `loop` branches while the counter is not zero, `loope` also wants
    /// the zero flag set and `loopne` wants it clear, and `jrcxz` branches
    /// when the counter is zero and leaves it alone. The counter is
    /// decremented before the test, and the flags are not touched at all.
    fn lift_loop(&mut self, m: &str, stmts: &mut Vec<Stmt>) {
        // The counter is the register one below the return value, and its
        // width decides how the decrement wraps: `ecx` from zero is
        // `0xffffffff` and `rcx` from zero is `0xffff_ffff_ffff_ffff`.
        let width = self.width();
        let counter = Operand::Reg(Reg::view(RCX, width, 0));
        let value = self.core.read(&counter, stmts);
        let decremented = Expr::Bin {
            op: BinOp::Sub,
            lhs: Box::new(value.clone()),
            rhs: Box::new(Expr::Const(1)),
            bits: width,
        };
        // The test is against the counter after the decrement. The
        // counter's name is enough: the expression that computes the
        // decrement mentions the value that was there before, and the
        // branch asks about the value that is there now.
        let not_zero = Expr::Cond {
            op: CondOp::Ne,
            lhs: Box::new(Expr::var(self.reg(RCX))),
            rhs: Box::new(Expr::Const(0)),
            bits: width,
        };
        let condition = match m {
            "loop" => not_zero,
            "loope" | "loopne" => {
                let Some(zero) = self.core.flags.zf.clone() else {
                    // The zero flag says nothing, and the machine's own
                    // condition cannot be answered without it.
                    self.core.write(&counter, decremented, stmts);
                    self.loop_condition = Some(Expr::Unknown("flags".into()));
                    return;
                };
                let zero = if m == "loope" {
                    zero
                } else {
                    Expr::Bool {
                        op: BoolOp::Not,
                        args: vec![zero],
                    }
                };
                Expr::Bool {
                    op: BoolOp::And,
                    args: vec![not_zero, zero],
                }
            }
            _ => {
                // `jrcxz` branches when the counter is zero, and leaves the
                // counter alone.
                Expr::Cond {
                    op: CondOp::Eq,
                    lhs: Box::new(value),
                    rhs: Box::new(Expr::Const(0)),
                    bits: 8,
                }
            }
        };
        if m != "jrcxz" && m != "jecxz" {
            self.core.write(&counter, decremented, stmts);
        }
        self.loop_condition = Some(condition);
    }

    /// `movmskps`/`movmskpd`: collect the sign bits of a vector into an
    /// integer.
    ///
    /// The mask has one bit per element, and the bit is the element's
    /// sign.
    fn lift_movmsk(&mut self, ops: &[Operand], count: u8, elem: u8, stmts: &mut Vec<Stmt>) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let mut mask = Expr::Const(0);
        for index in 0..count {
            let lane = self.vec_source(src, index, elem, stmts);
            // The sign of an element is its top bit.
            let sign = Expr::Bin {
                op: BinOp::Shr,
                lhs: Box::new(lane),
                rhs: Box::new(Expr::Const(u64::from(elem) * 8 - 1)),
                bits: elem,
            };
            let sign = Expr::Bin {
                op: BinOp::And,
                lhs: Box::new(sign),
                rhs: Box::new(Expr::Const(1)),
                bits: elem,
            };
            let bit = Expr::Bin {
                op: BinOp::Shl,
                lhs: Box::new(sign),
                rhs: Box::new(Expr::Const(u64::from(index))),
                bits: 4,
            };
            mask = Expr::Bin {
                op: BinOp::Or,
                lhs: Box::new(mask),
                rhs: Box::new(bit),
                bits: 4,
            };
        }
        self.core.write(dst, mask, stmts);
    }

    /// `pmovmskb`: collect the sign bits of every byte of a vector.
    ///
    /// The mask has one bit per byte, and the bit is the byte's top bit.
    fn lift_pmovmskb(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let mut mask = Expr::Const(0);
        for byte in 0..16u8 {
            // The vector is kept in 32-bit elements, so a byte is part of
            // one of them.
            let word = self.vec_source(src, byte / 4, 4, stmts);
            let sign = Expr::Bin {
                op: BinOp::Shr,
                lhs: Box::new(word),
                rhs: Box::new(Expr::Const(u64::from(byte % 4) * 8 + 7)),
                bits: 4,
            };
            let sign = Expr::Bin {
                op: BinOp::And,
                lhs: Box::new(sign),
                rhs: Box::new(Expr::Const(1)),
                bits: 4,
            };
            let bit = Expr::Bin {
                op: BinOp::Shl,
                lhs: Box::new(sign),
                rhs: Box::new(Expr::Const(u64::from(byte))),
                bits: 4,
            };
            mask = Expr::Bin {
                op: BinOp::Or,
                lhs: Box::new(mask),
                rhs: Box::new(bit),
                bits: 4,
            };
        }
        self.core.write(dst, mask, stmts);
    }

    fn lift_cmp(&mut self, ops: &[Operand], is_test: bool) {
        if ops.len() != 2 {
            return;
        }
        let mut sink = Vec::new();
        let lhs = self.core.read(&ops[0], &mut sink);
        let rhs = self.core.read(&ops[1], &mut sink);
        // `test a, b` sets flags from `a & b`, which is a comparison with zero.
        let (lhs, rhs) = if is_test {
            // `test x, x` is the zero test; writing it as `x & x` hides that.
            let value = if lhs == rhs {
                lhs
            } else {
                Expr::Bin {
                    op: BinOp::And,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    bits: ops[0].size(),
                }
            };
            (value, Expr::Const(0))
        } else {
            (lhs, rhs)
        };
        // A comparison is a subtraction that threw the difference away, and
        // the flags are the ones that subtraction leaves.
        let cmp = super::PendingCmp::int(lhs, rhs, ops[0].size());
        self.core.set_flags(cmp.derived_flags(), Some(cmp));
    }

    fn lift_setcc(&mut self, mnem: &str, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let Some(op) = ops.first() else { return };
        let Some(cond) = self.core.target.setcc_cond(&mnem[3..]) else {
            return;
        };
        // A `setcc` reads the same flags a `jcc` does, so the condition it
        // tests is the one the branch of the same name would test.
        let jcc = format!("j{}", &mnem[3..]);
        let value = self.core.condition_of(&jcc, cond);
        let value = Expr::Cast {
            kind: CastKind::Zext,
            arg: Box::new(value),
            from: 1,
            to: op.size(),
        };
        self.core.write(op, value, stmts);
    }

    fn lift_cmov(&mut self, mnem: &str, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let Some(cond) = self.core.target.cmov_cond(&mnem[4..]) else {
            return;
        };
        let other = self.snapshot(&ops[1], stmts);
        let current = self.core.read(&ops[0], stmts);
        let jcc = format!("j{}", &mnem[4..]);
        let cond = self.core.condition_of(&jcc, cond);
        self.core.write(
            &ops[0],
            Expr::Select {
                cond: Box::new(cond),
                yes: Box::new(other),
                no: Box::new(current),
            },
            stmts,
        );
    }

    fn lift_cast(&mut self, reg: u8, from: u8, to: u8, stmts: &mut Vec<Stmt>) {
        let name = self.reg(reg);
        let value = Expr::Cast {
            kind: CastKind::Sext,
            arg: Box::new(Expr::var(name.clone())),
            from,
            to,
        };
        self.core.assign(&name, value, stmts);
    }

    /// `push`: a value goes on the stack, or a register is saved.
    ///
    /// The two are one instruction. What tells them apart is where the push
    /// happens: the run of pushes at the front of a function saves the
    /// registers the function is about to borrow, and a push after that is a
    /// value on its way to a call. The stack pointer moves either way, which
    /// is what keeps the frame slots right for the rest of the block — a
    /// `push 1` that left the stack pointer alone put every later
    /// `[esp+N]` one slot out.
    fn lift_push(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let Some(op) = ops.first() else { return };
        self.core.adjust_sp(-i64::from(self.width()));
        if let Operand::Reg(r) = op {
            // `push rbp` anchors the frame: the value saved is the caller's
            // frame pointer, not a value the function computes.
            if Some(r.idx) == self.core.target.fp_reg() {
                return;
            }
            if !self.frame_ready && is_callee_saved(self.mode, r.idx) {
                let name = self.reg(r.idx);
                stmts.push(Stmt::Comment(format!(
                    "push {name} ; saved at frame{:+}",
                    self.core.sp_off
                )));
                return;
            }
        }
        if !self.core.target.args_on_stack() {
            // A 64-bit call passes its arguments in registers, so a push here
            // is a saved value or a spill and saying it is an argument would
            // invent one.
            if let Operand::Reg(r) = op {
                let name = self.reg(r.idx);
                stmts.push(Stmt::Comment(format!(
                    "push {name} ; saved at frame{:+}",
                    self.core.sp_off
                )));
            }
            return;
        }
        // The value is named before it is pushed. The register it came from
        // is about to be reused — `push eax; lea eax, [edx+edx*2]; push eax`
        // is how a two-argument call is set up — and a bare name in the list
        // would read whatever the register holds when the call is reached,
        // turning two different arguments into two copies of the second.
        let value = match op {
            Operand::Imm(v) => Expr::Const(*v),
            op => self.snapshot(op, stmts),
        };
        self.pushed.push(value);
    }

    fn lift_pop(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let Some(op) = ops.first() else { return };
        self.core.adjust_sp(i64::from(self.width()));
        if let Operand::Reg(r) = op {
            if Some(r.idx) == self.core.target.fp_reg() {
                self.core.fp_off = None;
                return;
            }
            // The value came from a matching push; keeping the register name is
            // honest and avoids a fake load from an invented stack slot.
            let name = self.reg(r.idx);
            stmts.push(Stmt::Comment(format!(
                "pop {name} ; restored from the frame"
            )));
        }
    }

    fn lift_xchg(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let a = self.snapshot(&ops[0], stmts);
        let b = self.snapshot(&ops[1], stmts);
        self.core.write(&ops[0], b, stmts);
        self.core.write(&ops[1], a, stmts);
    }

    /// `rol`/`ror`: rotate the destination by a count.
    ///
    /// The count is masked to the width of the operand before the
    /// rotation, so a count of 64 on a 32-bit operand leaves it alone.
    /// The flags a rotation leaves behind, with the rule that a rotation by
    /// nothing leaves every flag alone.
    fn install_rotate_flags(&mut self, count: &Expr, width: u8, state: FlagState) {
        let guard = match count {
            // Nothing turned, nothing changed.
            Expr::Const(0) => return,
            Expr::Const(_) => None,
            _ => Some(Expr::Cond {
                op: CondOp::Eq,
                lhs: Box::new(count.clone()),
                rhs: Box::new(Expr::Const(0)),
                bits: width,
            }),
        };
        let state = match guard {
            None => state,
            Some(cond) => {
                let wrap = |value: Option<Expr>| {
                    value.map(|value| Expr::Select {
                        cond: Box::new(cond.clone()),
                        yes: Box::new(Expr::Unknown("flags".into())),
                        no: Box::new(value),
                    })
                };
                FlagState {
                    cf: wrap(state.cf),
                    zf: wrap(state.zf),
                    sf: wrap(state.sf),
                    of: wrap(state.of),
                    pf: wrap(state.pf),
                }
            }
        };
        self.core.update_flags(state);
    }

    fn lift_rotate(&mut self, m: &str, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let (Some(dst), Some(count)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let width = dst.size();
        let bits = u64::from(width) * 8;
        let previous = self.snapshot(dst, stmts);
        // The count is named before the write for the same reason the value
        // is: `rol cl, cl` counts with the register it is about to overwrite.
        let raw = match count {
            Operand::Imm(v) => Expr::Const(*v),
            _ => self.snapshot(count, stmts),
        };
        // The count is masked to five bits (six for a 64-bit operand) and then
        // taken modulo the width, which the intrinsic does.
        let count = mask_count(raw, bits, width);
        let turn = turn_mod(count, bits, width);
        let turned = Expr::Intrinsic {
            op: if m == "rol" {
                Intrinsic::Rol
            } else {
                Intrinsic::Ror
            },
            args: vec![previous, turn.clone()],
            bits: width,
        };
        self.core.write(dst, turned.clone(), stmts);
        // A rotation carries the bit that came round: for a left rotation the
        // one that arrived at the bottom, for a right one the one that arrived
        // at the top.
        let carried = match m {
            "rol" => Expr::Bin {
                op: BinOp::And,
                lhs: Box::new(turned.clone()),
                rhs: Box::new(Expr::Const(1)),
                bits: width,
            },
            _ => sign_flag(&turned, width),
        };
        // The overflow compares the two bits that swapped places: the bottom
        // against the top for a left rotation, the top against the one below
        // it for a right one. Measuring the processor showed both rules hold
        // for every count, not only for the count of one the manual defines.
        let of = match m {
            "rol" => Expr::Cond {
                op: CondOp::Ne,
                lhs: Box::new(carried.clone()),
                rhs: Box::new(sign_flag(&turned, width)),
                bits: width,
            },
            _ => Expr::Cond {
                op: CondOp::Ne,
                lhs: Box::new(sign_flag(&turned, width)),
                rhs: Box::new(Expr::Bin {
                    op: BinOp::And,
                    lhs: Box::new(Expr::Bin {
                        op: BinOp::Shr,
                        lhs: Box::new(turned),
                        rhs: Box::new(Expr::Const(bits - 2)),
                        bits: width,
                    }),
                    rhs: Box::new(Expr::Const(1)),
                    bits: width,
                }),
                bits: width,
            },
        };
        // Nothing else changes: a rotation leaves the zero, sign and parity
        // flags as they were.
        self.install_rotate_flags(
            &turn,
            width,
            FlagState {
                cf: Some(carried),
                zf: None,
                sf: None,
                of: Some(of),
                pf: None,
            },
        );
    }

    /// `rcl`/`rcr`: rotate through the carry.
    ///
    /// The carry joins the value as a bit above the top, the field turns, and
    /// what comes back below the top is the carry again. The field is one bit
    /// wider than the operand, so a byte turns through nine positions and not
    /// eight, and a count of nine is a count of nothing.
    fn lift_rotate_carry(&mut self, m: &str, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let (Some(dst), Some(count)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let width = dst.size();
        let bits = u64::from(width) * 8;
        // The value before the rotation holds the bit that becomes the carry,
        // and the write is about to overwrite it.
        let previous = self.snapshot(dst, stmts);
        // The count is named before the write: `rcl cl, cl` counts with the
        // register it is about to overwrite.
        let raw = match count {
            Operand::Imm(v) => Expr::Const(*v),
            _ => self.snapshot(count, stmts),
        };
        let count = mask_count(raw, bits, width);
        // The count is taken modulo the width plus the carry.
        let turned_by = turn_mod(count.clone(), bits + 1, width);
        let carry_in = self.carry_in();
        let turned = Expr::Intrinsic {
            op: if m == "rcl" {
                Intrinsic::Rcl
            } else {
                Intrinsic::Rcr
            },
            args: vec![previous.clone(), turned_by.clone(), carry_in],
            bits: width,
        };
        self.core.write(dst, turned.clone(), stmts);
        // The carry comes back out of the top: a left rotation moved the bit
        // at `width - count` up to it, a right one the bit at `count - 1`.
        let carried = match m {
            "rcl" => Expr::Bin {
                op: BinOp::And,
                lhs: Box::new(Expr::Bin {
                    op: BinOp::Shr,
                    lhs: Box::new(previous),
                    rhs: Box::new(Expr::Bin {
                        op: BinOp::Sub,
                        lhs: Box::new(Expr::Const(bits)),
                        rhs: Box::new(turned_by.clone()),
                        bits: width,
                    }),
                    bits: width,
                }),
                rhs: Box::new(Expr::Const(1)),
                bits: width,
            },
            _ => Expr::Bin {
                op: BinOp::And,
                lhs: Box::new(Expr::Bin {
                    op: BinOp::Shr,
                    lhs: Box::new(previous),
                    rhs: Box::new(Expr::Bin {
                        op: BinOp::Sub,
                        lhs: Box::new(turned_by.clone()),
                        rhs: Box::new(Expr::Const(1)),
                        bits: width,
                    }),
                    bits: width,
                }),
                rhs: Box::new(Expr::Const(1)),
                bits: width,
            },
        };
        let of = match m {
            "rcl" => Expr::Cond {
                op: CondOp::Ne,
                lhs: Box::new(carried.clone()),
                rhs: Box::new(sign_flag(&turned, width)),
                bits: width,
            },
            _ => Expr::Cond {
                op: CondOp::Ne,
                lhs: Box::new(sign_flag(&turned, width)),
                rhs: Box::new(Expr::Bin {
                    op: BinOp::And,
                    lhs: Box::new(Expr::Bin {
                        op: BinOp::Shr,
                        lhs: Box::new(turned),
                        rhs: Box::new(Expr::Const(bits - 2)),
                        bits: width,
                    }),
                    rhs: Box::new(Expr::Const(1)),
                    bits: width,
                }),
                bits: width,
            },
        };
        self.install_rotate_flags(
            &turned_by,
            width,
            FlagState {
                cf: Some(carried),
                zf: None,
                sf: None,
                of: Some(of),
                pf: None,
            },
        );
    }

    /// `bswap`: reverse the bytes of the destination.
    fn lift_bswap(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let Some(dst) = ops.first() else { return };
        let width = dst.size();
        let value = self.core.read(dst, stmts);
        let value = Expr::Intrinsic {
            op: Intrinsic::Bswap,
            args: vec![value],
            bits: width,
        };
        self.core.write(dst, value, stmts);
        // A byte swap leaves the flags alone.
    }

    /// `bsr`/`bsf`/`tzcnt`/`lzcnt`: find a bit.
    ///
    /// `tzcnt` and `lzcnt` count the zeros and answer the width when the
    /// source is all zeros. `bsf` and `bsr` have no such answer: the result is
    /// undefined for a zero source, and the processor leaves the destination
    /// exactly as it was, which is what this says.
    ///
    /// Only the counting operations define flags, the zero flag to whether the
    /// source was zero and the carry to the same thing. The scans define
    /// neither, and claiming a zero flag for them made `bsf rax, rax` disagree
    /// with the processor on every zero it met.
    fn lift_bit_scan(&mut self, m: &str, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let width = dst.size();
        let value = self.snapshot(src, stmts);
        let op = match m {
            "bsr" => Intrinsic::Bsr,
            "bsf" => Intrinsic::Bsf,
            "tzcnt" => Intrinsic::Tzcnt,
            _ => Intrinsic::Lzcnt,
        };
        let zero = Expr::Cond {
            op: CondOp::Eq,
            lhs: Box::new(value.clone()),
            rhs: Box::new(Expr::Const(0)),
            bits: width,
        };
        let scan = Expr::Intrinsic {
            op,
            args: vec![value],
            bits: width,
        };
        let result = match m {
            "bsf" | "bsr" => {
                let previous = self.snapshot(dst, stmts);
                Expr::Select {
                    cond: Box::new(zero.clone()),
                    yes: Box::new(previous),
                    no: Box::new(scan),
                }
            }
            _ => scan,
        };
        self.core.write(dst, result, stmts);
        let (cf, zf) = match m {
            "tzcnt" | "lzcnt" => (Some(zero.clone()), Some(zero)),
            _ => (None, None),
        };
        self.core.update_flags(FlagState {
            cf,
            zf,
            sf: None,
            of: None,
            pf: None,
        });
    }

    /// `bt`/`bts`/`btr`/`btc`: read a bit, and for the three that write
    /// one, change it.
    ///
    /// The carry flag holds the bit as it was before the write, and the
    /// other flags are left alone: the machine does not define them, so
    /// nothing after this instruction can be said about them.
    fn lift_bit_test(&mut self, m: &str, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let (Some(dst), Some(index)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let width = dst.size();
        // The two forms count bits differently. A register is a bit field and
        // the index is taken modulo the width of the field, so bit 64 of a
        // 64-bit register is bit zero again. Memory is a bit string: the index
        // is a signed offset added to the address in bytes, and its low three
        // bits pick the bit inside the byte that lands there.
        //
        // `btc rax, rax` also reads the same register twice and then writes it,
        // so both reads are named before the write. Without that the carry
        // reports the bit of the value the instruction just produced.
        let (value, position, target) = if let Operand::Mem { .. } = dst {
            let base = self.core.addr_of(dst, stmts);
            let index = self.snapshot(index, stmts);
            let target = Expr::Bin {
                op: BinOp::Add,
                lhs: Box::new(base),
                rhs: Box::new(Expr::Bin {
                    op: BinOp::Sar,
                    lhs: Box::new(index.clone()),
                    rhs: Box::new(Expr::Const(3)),
                    bits: 8,
                }),
                bits: 8,
            };
            let value = self.name(
                Expr::Load {
                    addr: Box::new(target.clone()),
                    size: 1,
                    signed: false,
                },
                stmts,
            );
            let position = Expr::Bin {
                op: BinOp::And,
                lhs: Box::new(index),
                rhs: Box::new(Expr::Const(7)),
                bits: 8,
            };
            (value, position, Some(target))
        } else {
            let value = self.snapshot(dst, stmts);
            let index = self.snapshot(index, stmts);
            let position = Expr::Bin {
                op: BinOp::And,
                lhs: Box::new(index),
                rhs: Box::new(Expr::Const(u64::from(width) * 8 - 1)),
                bits: width,
            };
            (value, position, None)
        };
        let unit = if target.is_some() { 1 } else { width };
        // The bit is the one the position names, counted from the low end.
        let bit = Expr::Bin {
            op: BinOp::And,
            lhs: Box::new(Expr::Bin {
                op: BinOp::Shr,
                lhs: Box::new(value.clone()),
                rhs: Box::new(position.clone()),
                bits: unit,
            }),
            rhs: Box::new(Expr::Const(1)),
            bits: unit,
        };
        // The carry flag holds the bit as it was, so the branch that reads it
        // asks whether the bit was set. The other flags are left alone.
        self.core.update_flags(FlagState {
            cf: Some(Expr::Cond {
                op: CondOp::Ne,
                lhs: Box::new(bit),
                rhs: Box::new(Expr::Const(0)),
                bits: unit,
            }),
            zf: None,
            sf: None,
            of: None,
            pf: None,
        });
        if m == "bt" {
            return;
        }
        let one = Expr::Bin {
            op: BinOp::Shl,
            lhs: Box::new(Expr::Const(1)),
            rhs: Box::new(position),
            bits: unit,
        };
        let updated = match m {
            "bts" => Expr::Bin {
                op: BinOp::Or,
                lhs: Box::new(value),
                rhs: Box::new(one),
                bits: unit,
            },
            "btr" => Expr::Bin {
                op: BinOp::And,
                lhs: Box::new(value),
                rhs: Box::new(Expr::Un {
                    op: UnOp::Not,
                    arg: Box::new(one),
                    bits: unit,
                }),
                bits: unit,
            },
            _ => Expr::Bin {
                op: BinOp::Xor,
                lhs: Box::new(value),
                rhs: Box::new(one),
                bits: unit,
            },
        };
        match target {
            // The bit string form writes the byte the bit landed in.
            Some(addr) => stmts.push(Stmt::Store {
                addr,
                value: updated,
                size: 1,
            }),
            None => self.core.write(dst, updated, stmts),
        }
    }

    /// `div`/`idiv`: divide the two-word dividend by an operand.
    ///
    /// The dividend is the pair `rdx:rax`, the quotient comes back in
    /// `rax` and the remainder in `rdx`. Every flag is left undefined, so
    /// nothing can be said about the branch that follows.
    fn lift_div(&mut self, m: &str, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let Some(src) = ops.first() else { return };
        let width = src.size();
        // The dividend is the high word followed by the low one, each at the
        // width the instruction works on. All three reads are named before the
        // writes: the divisor can be the accumulator the quotient is about to
        // land in, and then the remainder would divide the quotient by itself.
        // A byte division takes its dividend from `ax`, so the high half is
        // `ah`, not `dl`: the pair lives inside the accumulator.
        let high = if width == 1 {
            self.snapshot(&Operand::Reg(Reg::view(0, 1, 8)), stmts)
        } else {
            self.snapshot(&Operand::Reg(Reg::view(2, width, 0)), stmts)
        };
        let low = self.snapshot(&Operand::Reg(Reg::view(0, width, 0)), stmts);
        let divisor = self.snapshot(src, stmts);
        let signed = m == "idiv";
        let quotient = Expr::Intrinsic {
            op: if signed {
                Intrinsic::IDiv
            } else {
                Intrinsic::Div
            },
            args: vec![high.clone(), low.clone(), divisor.clone()],
            bits: width,
        };
        let remainder = Expr::Intrinsic {
            op: if signed {
                Intrinsic::IDivRem
            } else {
                Intrinsic::DivRem
            },
            args: vec![high, low, divisor],
            bits: width,
        };
        self.core
            .write(&Operand::Reg(Reg::view(0, width, 0)), quotient, stmts);
        // A byte division is the one that does not use the high half of the
        // accumulator pair: the quotient goes to `al` and the remainder to
        // `ah`, both inside `rax`, and `rdx` is not involved at all.
        let rest = if width == 1 {
            Reg::view(0, 1, 8)
        } else {
            Reg::view(2, width, 0)
        };
        self.core
            .write(&Operand::Reg(rest), remainder, stmts);
        // A division leaves every flag undefined.
        self.core.pending_cmp = Some(super::PendingCmp::with_flags(
            Expr::Const(0),
            Expr::Const(0),
            width,
            0,
        ));
    }

    /// `shld`/`shrd`: shift a destination and fold a second value in.
    ///
    /// The result is the destination shifted one way with the bits of the
    /// source shifted the other way filling in. The count is masked to the
    /// width of the operand, and a count of zero leaves the destination
    /// alone, which the same formula gives.
    fn lift_double_shift(&mut self, m: &str, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let (Some(dst), Some(src), Some(count)) = (ops.first(), ops.get(1), ops.get(2)) else {
            return;
        };
        let width = dst.size();
        let bits = u64::from(width * 8);
        let value = self.core.read(dst, stmts);
        let source = self.core.read(src, stmts);
        let count = self.core.read(count, stmts);
        let count = Expr::Bin {
            op: BinOp::And,
            lhs: Box::new(count),
            rhs: Box::new(Expr::Const(bits - 1)),
            bits: width,
        };
        let (first, second) = if m == "shld" {
            (value.clone(), source)
        } else {
            (source, value.clone())
        };
        let up = Expr::Bin {
            op: BinOp::Shl,
            lhs: Box::new(first),
            rhs: Box::new(count.clone()),
            bits: width,
        };
        let down = Expr::Bin {
            op: BinOp::Shr,
            lhs: Box::new(second),
            rhs: Box::new(Expr::Bin {
                op: BinOp::Sub,
                lhs: Box::new(Expr::Const(bits)),
                rhs: Box::new(count),
                bits: width,
            }),
            bits: width,
        };
        let result = Expr::Bin {
            op: BinOp::Or,
            lhs: Box::new(up),
            rhs: Box::new(down),
            bits: width,
        };
        self.core.write(dst, result, stmts);
        // The carry flag holds a bit that came out of the shift, and the
        // overflow flag is only defined for a count of one. Neither is
        // describable here.
        self.core.pending_cmp = Some(super::PendingCmp::with_flags(
            Expr::Const(0),
            Expr::Const(0),
            width,
            0,
        ));
    }

    /// `syscall`/`sysenter`/`int 0x80`: a call into the kernel.
    ///
    /// The number is in the accumulator and the arguments are in the
    /// registers the kernel reads them from — the same six in a different
    /// order, and the order is the one thing the width changes:
    /// `rdi, rsi, rdx, r10, r8, r9` on a 64-bit kernel and
    /// `ebx, ecx, edx, esi, edi, ebp` on a 32-bit one. The result comes back
    /// in the accumulator, and the kernel chooses it, so there is nothing to
    /// say about its value.
    fn lift_syscall(&mut self, addr: Addr, stmts: &mut Vec<Stmt>) {
        let args = match self.mode {
            Mode::Bits64 => vec![
                Expr::var(self.reg(RAX)),
                Expr::var(self.reg(7)),
                Expr::var(self.reg(6)),
                Expr::var(self.reg(2)),
                Expr::var(self.reg(10)),
                Expr::var(self.reg(8)),
                Expr::var(self.reg(9)),
            ],
            Mode::Bits32 => vec![
                Expr::var(self.reg(RAX)),
                Expr::var(self.reg(3)),
                Expr::var(self.reg(RCX)),
                Expr::var(self.reg(2)),
                Expr::var(self.reg(6)),
                Expr::var(self.reg(7)),
                Expr::var(self.reg(RBP)),
            ],
        };
        stmts.push(Stmt::Call {
            name: "syscall".into(),
            args,
            ret: None,
            addr,
            target: None,
        });
        let dest = Operand::Reg(Reg::full(RAX, self.width()));
        let value = Expr::Unknown("syscall".into());
        self.core.write(&dest, value, stmts);
    }

    /// `int imm`: a software interrupt.
    ///
    /// The only vector a compiler emits is `0x80`, the 32-bit system-call
    /// gate, and `0x2e`/`0x2f` in the Windows world. Everything else is a
    /// call into something this tool does not model, and saying so is better
    /// than guessing which.
    fn lift_int(&mut self, ops: &[Operand], addr: Addr, stmts: &mut Vec<Stmt>) {
        match ops.first() {
            Some(Operand::Imm(0x80)) if self.mode.is_32() => self.lift_syscall(addr, stmts),
            Some(Operand::Imm(0x2e)) => {
                stmts.push(Stmt::Call {
                    name: "sysenter".into(),
                    args: vec![Expr::var(self.reg(RAX))],
                    ret: None,
                    addr,
                    target: None,
                });
                self.core.notes.push((
                    addr,
                    "int 0x2e: a system call through the Windows gate".into(),
                ));
            }
            Some(Operand::Imm(3)) => {
                stmts.push(Stmt::Comment("int 3 ; breakpoint".into()));
            }
            Some(Operand::Imm(n)) => {
                let n = *n;
                stmts.push(Stmt::Comment(format!(
                    "int {n:#x} ; a software interrupt this tool does not model"
                )));
            }
            _ => {}
        }
    }

    /// `ret`: return, optionally removing the arguments first.
    ///
    /// `ret 8` pops the return address and then eight bytes of arguments, so
    /// a callee that cleans up its own stack also states how many argument
    /// bytes it takes. That count is a fact about the function's signature
    /// the body may never mention: a function that ignores its second
    /// argument still has one, and the callers still push it.
    fn lift_ret(&mut self, ops: &[Operand], addr: Addr, stmts: &mut Vec<Stmt>) {
        let _ = stmts;
        let Some(Operand::Imm(bytes)) = ops.first() else {
            return;
        };
        self.pops = Some(*bytes);
        // A callee that removes its own arguments is stating its convention.
        // Reading a `ret 8` in an image that was opened as `cdecl` means the
        // image is mixed, which is worth saying once rather than per call.
        if self.mode.is_32() && !self.core.target.pops_arguments() {
            self.core.notes.push((
                addr,
                format!(
                    "`ret {bytes}` removes {bytes} bytes of arguments: this function cleans up its own stack"
                ),
            ));
        }
    }

    /// `rdtsc`: the time-stamp counter.
    ///
    /// The counter is the only thing that reads it, and it is read twice
    /// for the two halves of the result, so the value is one unknown.
    fn lift_timestamp(&mut self, addr: Addr, m: &str, stmts: &mut Vec<Stmt>) {
        let value = Expr::Unknown("timestamp".into());
        self.core
            .write(&Operand::Reg(Reg::full(RAX, 8)), value.clone(), stmts);
        if m == "rdtscp" {
            self.core
                .write(&Operand::Reg(Reg::full(2, 8)), value, stmts);
        }
        self.core.notes.push((
            addr,
            "the time-stamp counter reads the processor's clock".into(),
        ));
    }

    /// The flag instructions.
    ///
    /// `clc`, `stc` and `cmc` change the carry flag and nothing else, so
    /// only the carry is claimed. `cld` and `std` set the direction the
    /// string instructions run in, and `sti` and `cli` enable and disable
    /// interrupts; neither shows up in the data flow.
    fn lift_flag_instruction(&mut self, m: &str, addr: Addr, stmts: &mut Vec<Stmt>) {
        match m {
            "clc" | "stc" | "cmc" => {
                // The carry is a single bit, and the comparison model
                // carries a pair. A carry of one is a pair of zero and
                // one: the branch that reads it asks "is the carry
                // smaller than one", which is the carry.
                let (lhs, rhs) = match m {
                    "stc" => (Expr::Const(0), Expr::Const(1)),
                    "clc" => (Expr::Const(1), Expr::Const(0)),
                    // A complemented carry is neither of those.
                    _ => (Expr::Unknown("carry".into()), Expr::Unknown("carry".into())),
                };
                self.core.pending_cmp = Some(super::PendingCmp::with_flags(
                    lhs,
                    rhs,
                    8,
                    super::FLAG_CF,
                ));
            }
            "cld" | "std" => {
                self.core.notes.push((
                    addr,
                    if m == "std" {
                        "the direction flag is set: the string instructions run backwards"
                            .into()
                    } else {
                        "the direction flag is clear: the string instructions run forwards"
                            .into()
                    },
                ));
            }
            "sti" | "cli" => {
                self.core.notes.push((
                    addr,
                    if m == "sti" {
                        "interrupts are enabled".into()
                    } else {
                        "interrupts are disabled".into()
                    },
                ));
            }
            _ => {}
        }
        let _ = stmts;
    }

    fn lift_call(&mut self, insn: &Instruction, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let direct = match insn.target {
            Some(occ_core::CallTarget::Direct { to }) => Some(to),
            _ => None,
        };
        let name = match insn.target {
            Some(occ_core::CallTarget::Direct { to }) => (self.core.name_of)(to),
            _ => match ops.first() {
                Some(o) => {
                    let target = self.core.read(o, stmts);
                    format!("(*{target})")
                }
                None => "(*?)".into(),
            },
        };
        // Only pass the arguments this function actually set: a call in the
        // middle of a function has live values in earlier ones, and a 32-bit
        // call passes the values pushed since the last one.
        let args = self.take_args_for_call();
        // The callee is free to write any flag.
        self.core.pending_cmp = None;
        let ret_reg = self.reg(RAX);
        if !matches!(insn.flow, FlowKind::Return) {
            stmts.push(Stmt::Call {
                name,
                args,
                ret: Some(ret_reg),
                addr: insn.addr,
                target: direct,
            });
            // Caller-saved registers are dead after the call; carrying them
            // forward made every later call inherit the previous call's
            // arguments. A read of one of them before the function writes it
            // again is the callee's leftovers, not the value from before the
            // call.
            self.core.written.clear();
            self.core.written.insert(RAX);
            // The call defines the return register, and a function whose only
            // value is a call's result has to return it: without this the
            // body printed `return;` and the caller's argument disappeared.
            self.core.ever_written.insert(RAX);
            let caller_saved = match self.mode {
                Mode::Bits64 => CALLER_SAVED,
                Mode::Bits32 => CALLER_SAVED_32,
            };
            self.core.clobber(caller_saved, RAX);
        } else {
            stmts.push(Stmt::Call {
                name,
                args,
                ret: None,
                addr: insn.addr,
                target: direct,
            });
        }
    }

    /// `jmp`: a jump through a register is a computed target, and the value
    /// in that register is the target.
    fn lift_jmp(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let Some(op) = ops.first() else { return };
        if let Operand::Reg(r) = op {
            let value = self.core.read_reg(*r);
            let name = self.reg(r.idx);
            stmts.push(Stmt::Comment(format!(
                "jmp {name} ; computed jump target"
            )));
            let _ = value;
        }
    }

    // ------------------------------------------------------------------
    // Vector registers
    // ------------------------------------------------------------------

    /// Read one 32-bit element of a vector register.
    fn vec_read(&self, class: VecClass, reg: u8, index: u8) -> Expr {
        Expr::var(vec_element(class, reg, index))
    }

    /// Read one 64-bit element: the two 32-bit elements that hold it.
    ///
    /// The machine stores a double as eight bytes, little endian, so the
    /// low half of the value is the low element and the high half is the
    /// high one.
    fn vec_read_wide(&self, class: VecClass, reg: u8, index: u8) -> Expr {
        let low = Expr::var(vec_element(class, reg, index));
        let high = Expr::var(vec_element(class, reg, index + 1));
        Expr::Bin {
            op: BinOp::Or,
            lhs: Box::new(Expr::Bin {
                op: BinOp::Shl,
                lhs: Box::new(high),
                rhs: Box::new(Expr::Const(32)),
                bits: 8,
            }),
            rhs: Box::new(low),
            bits: 8,
        }
    }

    /// Write one 32-bit element of a vector register.
    fn vec_write(
        &mut self,
        class: VecClass,
        reg: u8,
        index: u8,
        value: Expr,
        stmts: &mut Vec<Stmt>,
    ) {
        self.note_vector_write(class, reg, 4);
        self.core.assign(&vec_element(class, reg, index), value, stmts);
    }

    /// Write one 64-bit element: split it into the two 32-bit elements.
    fn vec_write_wide(
        &mut self,
        class: VecClass,
        reg: u8,
        index: u8,
        value: Expr,
        stmts: &mut Vec<Stmt>,
    ) {
        self.note_vector_write(class, reg, 8);
        let tmp = self.new_temp();
        self.core.assign(&tmp, value, stmts);
        let low = Expr::Cast {
            kind: CastKind::Trunc,
            arg: Box::new(Expr::var(tmp.clone())),
            from: 8,
            to: 4,
        };
        let high = Expr::Cast {
            kind: CastKind::Trunc,
            arg: Box::new(Expr::Bin {
                op: BinOp::Shr,
                lhs: Box::new(Expr::var(tmp)),
                rhs: Box::new(Expr::Const(32)),
                bits: 8,
            }),
            from: 8,
            to: 4,
        };
        self.core.assign(&vec_element(class, reg, index), low, stmts);
        self.core.assign(&vec_element(class, reg, index + 1), high, stmts);
    }

    /// Record that the function wrote a vector register.
    ///
    /// Which register the return value is in depends on the function's
    /// type, which the code does not carry. What the code does say is
    /// where the function left its last value, and `xmm0` is where the
    /// ABI puts a floating-point result.
    fn note_vector_write(&mut self, class: VecClass, reg: u8, elem: u8) {
        self.wrote_vector = true;
        if class == VecClass::Xmm && reg == 0 {
            self.float_width = elem;
        }
    }

    /// The address of one element of a memory operand.
    fn elem_address(&mut self, op: &Operand, offset: u64, stmts: &mut Vec<Stmt>) -> Expr {
        let base = self.core.addr_of(op, stmts);
        if offset == 0 {
            return base;
        }
        Expr::Bin {
            op: BinOp::Add,
            lhs: Box::new(base),
            rhs: Box::new(Expr::Const(offset)),
            bits: 8,
        }
    }

    /// One element of the source operand.
    fn vec_source(&mut self, src: &Operand, index: u8, elem: u8, stmts: &mut Vec<Stmt>) -> Expr {
        match src {
            Operand::Vec(class, reg) => {
                if elem == 8 {
                    self.vec_read_wide(*class, *reg, index * 2)
                } else {
                    self.vec_read(*class, *reg, index)
                }
            }
            Operand::Mem { .. } => {
                if let Some(slot) = self.core.stack_slot(src) {
                    return Expr::var(super::slot_name(slot, elem));
                }
                let addr = self.elem_address(src, u64::from(index) * u64::from(elem), stmts);
                Expr::Load {
                    addr: Box::new(addr),
                    size: elem,
                    signed: false,
                }
            }
            // A general-purpose register is the source of a `movd` or a
            // `movq`, and of a conversion from an integer.
            other => self.core.read(other, stmts),
        }
    }

    /// One element of the destination operand, as it was before the write.
    fn vec_old(&mut self, dst: &Operand, index: u8, elem: u8, stmts: &mut Vec<Stmt>) -> Expr {
        match dst {
            Operand::Vec(class, reg) => {
                if elem == 8 {
                    self.vec_read_wide(*class, *reg, index * 2)
                } else {
                    self.vec_read(*class, *reg, index)
                }
            }
            Operand::Mem { .. } => {
                if let Some(slot) = self.core.stack_slot(dst) {
                    return Expr::var(super::slot_name(slot, elem));
                }
                let addr = self.elem_address(dst, u64::from(index) * u64::from(elem), stmts);
                Expr::Load {
                    addr: Box::new(addr),
                    size: elem,
                    signed: false,
                }
            }
            other => self.core.read(other, stmts),
        }
    }

    /// Write one element of the destination operand.
    fn vec_store(
        &mut self,
        dst: &Operand,
        index: u8,
        elem: u8,
        value: Expr,
        stmts: &mut Vec<Stmt>,
    ) {
        match dst {
            Operand::Vec(class, reg) => {
                if elem == 8 {
                    self.vec_write_wide(*class, *reg, index * 2, value, stmts);
                } else {
                    self.vec_write(*class, *reg, index, value, stmts);
                }
            }
            Operand::Mem { .. } => {
                if let Some(slot) = self.core.stack_slot(dst) {
                    self.core.assign(&super::slot_name(slot, elem), value, stmts);
                    return;
                }
                let addr = self.elem_address(dst, u64::from(index) * u64::from(elem), stmts);
                stmts.push(Stmt::Store {
                    addr,
                    value,
                    size: elem,
                });
            }
            // A general-purpose register is the destination of a `movd` or
            // a `movq`.
            other => self.core.write(other, value, stmts),
        }
    }

    /// Apply an operation to every element of the destination.
    ///
    /// `count` counts elements of `elem` bytes; an element of eight bytes
    /// spans two of the 32-bit elements the register is modelled as.
    ///
    /// Every value is read before any of them is written, because an
    /// instruction may use its own destination as its source: `addps xmm0,
    /// xmm0` doubles the register, and reading it after the first write
    /// would double half of it twice.
    fn vec_each(
        &mut self,
        ops: &[Operand],
        count: u8,
        elem: u8,
        op: Option<BinOp>,
        stmts: &mut Vec<Stmt>,
    ) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let mut values = Vec::new();
        for index in 0..count {
            let source = self.vec_source(src, index, elem, stmts);
            let value = match op {
                None => source,
                Some(op) => {
                    let old = self.vec_old(dst, index, elem, stmts);
                    Expr::Bin {
                        op,
                        lhs: Box::new(old),
                        rhs: Box::new(source),
                        bits: elem,
                    }
                }
            };
            values.push(value);
        }
        for (index, value) in values.into_iter().enumerate() {
            self.vec_store(dst, index as u8, elem, value, stmts);
        }
    }

    /// Apply an operation to every element of the destination, with one
    /// operand: the square root and the conversions that keep the width.
    fn vec_each_unary(
        &mut self,
        ops: &[Operand],
        count: u8,
        elem: u8,
        op: BinOp,
        stmts: &mut Vec<Stmt>,
    ) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let mut values = Vec::new();
        for index in 0..count {
            let source = self.vec_source(src, index, elem, stmts);
            values.push(Expr::Bin {
                op,
                lhs: Box::new(source),
                rhs: Box::new(Expr::Const(0)),
                bits: elem,
            });
        }
        for (index, value) in values.into_iter().enumerate() {
            self.vec_store(dst, index as u8, elem, value, stmts);
        }
    }

    /// Compare every element of the destination with the source.
    ///
    /// The machine writes a mask, not a flag: every bit of an element is
    /// set where the comparison holds. A comparison that holds is one, and
    /// multiplying by the mask turns it into the bits the machine writes.
    fn vec_each_compare(
        &mut self,
        ops: &[Operand],
        count: u8,
        elem: u8,
        op: BinOp,
        stmts: &mut Vec<Stmt>,
    ) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let mut values = Vec::new();
        for index in 0..count {
            let source = self.vec_source(src, index, elem, stmts);
            let old = self.vec_old(dst, index, elem, stmts);
            let cmp = Expr::Bin {
                op,
                lhs: Box::new(old),
                rhs: Box::new(source),
                bits: elem,
            };
            values.push(Expr::Bin {
                op: BinOp::Mul,
                lhs: Box::new(cmp),
                rhs: Box::new(Expr::Const(mask(elem))),
                bits: elem,
            });
        }
        for (index, value) in values.into_iter().enumerate() {
            self.vec_store(dst, index as u8, elem, value, stmts);
        }
    }

    /// Compare every element of the destination with the source, as an
    /// integer comparison.
    fn vec_each_compare_int(
        &mut self,
        ops: &[Operand],
        count: u8,
        elem: u8,
        cond: CondOp,
        stmts: &mut Vec<Stmt>,
    ) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let mut values = Vec::new();
        for index in 0..count {
            let source = self.vec_source(src, index, elem, stmts);
            let old = self.vec_old(dst, index, elem, stmts);
            let cmp = Expr::Cond {
                op: cond,
                lhs: Box::new(old),
                rhs: Box::new(source),
                bits: elem,
            };
            values.push(Expr::Bin {
                op: BinOp::Mul,
                lhs: Box::new(cmp),
                rhs: Box::new(Expr::Const(mask(elem))),
                bits: elem,
            });
        }
        for (index, value) in values.into_iter().enumerate() {
            self.vec_store(dst, index as u8, elem, value, stmts);
        }
    }

    /// Lift a vector instruction.
    ///
    /// The families are named by what they do, because the machine's own
    /// names say only the width: `addps` adds four single-precision
    /// elements, `addpd` two double-precision ones, and `addss` one.
    fn lift_vec(&mut self, m: &str, ops: &[Operand], stmts: &mut Vec<Stmt>) -> bool {
        match m {
            // Sixteen bytes: four 32-bit elements.
            "movups" | "movupd" | "movaps" | "movapd" | "movdqa" | "movdqu" => {
                self.vec_each(ops, 4, 4, None, stmts);
            }
            // The sign bits of the elements, collected into an integer.
            "movmskps" => self.lift_movmsk(ops, 4, 4, stmts),
            "movmskpd" => self.lift_movmsk(ops, 2, 8, stmts),
            "pmovmskb" => self.lift_pmovmskb(ops, stmts),
            // One 32-bit element.
            "movss" => self.vec_each(ops, 1, 4, None, stmts),
            // One 64-bit element: two 32-bit elements.
            "movsd" => self.vec_each(ops, 1, 8, None, stmts),
            // The low 64 bits of the source into the low element, or the
            // high 64 bits into the high element.
            "movlps" | "movlpd" => self.vec_each(ops, 1, 8, None, stmts),
            "movhps" | "movhpd" => self.vec_each_high(ops, stmts),
            // Packed single precision: four 32-bit elements.
            "addps" => self.vec_each(ops, 4, 4, Some(BinOp::FAdd), stmts),
            "subps" => self.vec_each(ops, 4, 4, Some(BinOp::FSub), stmts),
            "mulps" => self.vec_each(ops, 4, 4, Some(BinOp::FMul), stmts),
            "divps" => self.vec_each(ops, 4, 4, Some(BinOp::FDiv), stmts),
            "minps" => self.vec_each(ops, 4, 4, Some(BinOp::FMin), stmts),
            "maxps" => self.vec_each(ops, 4, 4, Some(BinOp::FMax), stmts),
            "sqrtps" => self.vec_each_unary(ops, 4, 4, BinOp::FSqrt, stmts),
            "andps" => self.vec_each(ops, 4, 4, Some(BinOp::And), stmts),
            "orps" => self.vec_each(ops, 4, 4, Some(BinOp::Or), stmts),
            "xorps" => self.vec_each(ops, 4, 4, Some(BinOp::Xor), stmts),
            // Packed double precision: two 64-bit elements.
            "addpd" => self.vec_each(ops, 2, 8, Some(BinOp::FAdd), stmts),
            "subpd" => self.vec_each(ops, 2, 8, Some(BinOp::FSub), stmts),
            "mulpd" => self.vec_each(ops, 2, 8, Some(BinOp::FMul), stmts),
            "divpd" => self.vec_each(ops, 2, 8, Some(BinOp::FDiv), stmts),
            "minpd" => self.vec_each(ops, 2, 8, Some(BinOp::FMin), stmts),
            "maxpd" => self.vec_each(ops, 2, 8, Some(BinOp::FMax), stmts),
            "sqrtpd" => self.vec_each_unary(ops, 2, 8, BinOp::FSqrt, stmts),
            "andpd" => self.vec_each(ops, 4, 4, Some(BinOp::And), stmts),
            "orpd" => self.vec_each(ops, 4, 4, Some(BinOp::Or), stmts),
            "xorpd" => self.vec_each(ops, 4, 4, Some(BinOp::Xor), stmts),
            // Scalar single precision: one 32-bit element.
            "addss" => self.vec_each(ops, 1, 4, Some(BinOp::FAdd), stmts),
            "subss" => self.vec_each(ops, 1, 4, Some(BinOp::FSub), stmts),
            "mulss" => self.vec_each(ops, 1, 4, Some(BinOp::FMul), stmts),
            "divss" => self.vec_each(ops, 1, 4, Some(BinOp::FDiv), stmts),
            "minss" => self.vec_each(ops, 1, 4, Some(BinOp::FMin), stmts),
            "maxss" => self.vec_each(ops, 1, 4, Some(BinOp::FMax), stmts),
            "sqrtss" => self.vec_each_unary(ops, 1, 4, BinOp::FSqrt, stmts),
            // Scalar double precision: one 64-bit element.
            "addsd" => self.vec_each(ops, 1, 8, Some(BinOp::FAdd), stmts),
            "subsd" => self.vec_each(ops, 1, 8, Some(BinOp::FSub), stmts),
            "mulsd" => self.vec_each(ops, 1, 8, Some(BinOp::FMul), stmts),
            "divsd" => self.vec_each(ops, 1, 8, Some(BinOp::FDiv), stmts),
            "minsd" => self.vec_each(ops, 1, 8, Some(BinOp::FMin), stmts),
            "maxsd" => self.vec_each(ops, 1, 8, Some(BinOp::FMax), stmts),
            "sqrtsd" => self.vec_each_unary(ops, 1, 8, BinOp::FSqrt, stmts),
            // The integer set: four 32-bit elements.
            "pand" => self.vec_each(ops, 4, 4, Some(BinOp::And), stmts),
            "por" => self.vec_each(ops, 4, 4, Some(BinOp::Or), stmts),
            "pxor" => self.vec_each(ops, 4, 4, Some(BinOp::Xor), stmts),
            "pandn" => self.vec_each_not_and(ops, stmts),
            "paddd" => self.vec_each(ops, 4, 4, Some(BinOp::Add), stmts),
            "psubd" => self.vec_each(ops, 4, 4, Some(BinOp::Sub), stmts),
            "paddq" => self.vec_each(ops, 2, 8, Some(BinOp::Add), stmts),
            "psubq" => self.vec_each(ops, 2, 8, Some(BinOp::Sub), stmts),
            "pcmpeqd" => self.vec_each_compare_int(ops, 4, 4, CondOp::Eq, stmts),
            "pcmpgtd" => self.vec_each_compare_int(ops, 4, 4, CondOp::Sgt, stmts),
            "pslld" => self.vec_each_shift(ops, 4, 4, BinOp::Shl, stmts),
            "psrld" => self.vec_each_shift(ops, 4, 4, BinOp::Shr, stmts),
            "psrad" => self.vec_each_shift(ops, 4, 4, BinOp::Sar, stmts),
            "psllq" => self.vec_each_shift(ops, 2, 8, BinOp::Shl, stmts),
            "psrlq" => self.vec_each_shift(ops, 2, 8, BinOp::Shr, stmts),
            // The byte and word sets work on narrower elements than the
            // model keeps. The ones that are element-wise are lifted at the
            // model's width, and the narrowing is said out loud; the ones
            // that move elements around are not modelled at all.
            "paddb" | "paddw" => {
                self.vec_each(ops, 4, 4, Some(BinOp::Add), stmts);
                self.vec_narrow_note();
            }
            "psubb" | "psubw" => {
                self.vec_each(ops, 4, 4, Some(BinOp::Sub), stmts);
                self.vec_narrow_note();
            }
            "pmullw" => {
                self.vec_each(ops, 4, 4, Some(BinOp::Mul), stmts);
                self.vec_narrow_note();
            }
            "pcmpeqb" | "pcmpeqw" => {
                self.vec_each_compare_int(ops, 4, 4, CondOp::Eq, stmts);
                self.vec_narrow_note();
            }
            "pcmpgtb" | "pcmpgtw" => {
                self.vec_each_compare_int(ops, 4, 4, CondOp::Sgt, stmts);
                self.vec_narrow_note();
            }
            "psllw" => {
                self.vec_each_shift(ops, 4, 4, BinOp::Shl, stmts);
                self.vec_narrow_note();
            }
            "psrlw" => {
                self.vec_each_shift(ops, 4, 4, BinOp::Shr, stmts);
                self.vec_narrow_note();
            }
            "psraw" => {
                self.vec_each_shift(ops, 4, 4, BinOp::Sar, stmts);
                self.vec_narrow_note();
            }
            "pminub" | "pmaxub" | "packsswb" | "packuswb" | "packssdw"
            | "punpcklbw" | "punpcklwd" | "punpckhbw" | "punpckhwd" | "pslldq"
            | "psrldq" => {
                stmts.push(Stmt::Comment(format!("{m} ; not modelled")));
                self.vec_narrow_note();
            }
            // The shuffles.
            "shufps" => self.vec_shufps(ops, stmts),
            "shufpd" => self.vec_shufpd(ops, stmts),
            "unpcklps" => self.vec_unpack(ops, false, stmts),
            "unpckhps" => self.vec_unpack(ops, true, stmts),
            "unpcklpd" => self.vec_unpack_wide(ops, false, stmts),
            "unpckhpd" => self.vec_unpack_wide(ops, true, stmts),
            "punpcklqdq" => self.vec_unpack_wide(ops, false, stmts),
            "punpckhqdq" => self.vec_unpack_wide(ops, true, stmts),
            "pshufd" => self.vec_pshufd(ops, stmts),
            "movsldup" => self.vec_dup(ops, 1, stmts),
            "movddup" => self.vec_dup(ops, 2, stmts),
            "movshdup" => self.vec_dup_high(ops, stmts),
            // The conversions.
            "cvtsi2ss" => self.vec_cvtsi(ops, 4, BinOp::I2F32, stmts),
            "cvtsi2sd" => self.vec_cvtsi(ops, 8, BinOp::I2F64, stmts),
            "cvttss2si" | "cvtss2si" => self.vec_cvt_to_int(ops, 4, BinOp::F2I32, stmts),
            "cvttsd2si" | "cvtsd2si" => self.vec_cvt_to_int(ops, 8, BinOp::F2I64, stmts),
            "cvtss2sd" => self.vec_cast(ops, 4, 8, BinOp::F2F64, stmts),
            "cvtsd2ss" => self.vec_cast(ops, 8, 4, BinOp::F2F32, stmts),
            "cvtps2pd" => self.vec_widen(ops, 4, 8, BinOp::F2F64, stmts),
            "cvtpd2ps" => self.vec_narrow_floats(ops, 8, 4, BinOp::F2F32, stmts),
            "cvtdq2ps" => self.vec_cast(ops, 4, 4, BinOp::F2F32, stmts),
            "cvtps2dq" | "cvttps2dq" => self.vec_cast(ops, 4, 4, BinOp::F2I32, stmts),
            // A move between a vector register and a general-purpose one.
            "movd" => self.vec_mov_gp(ops, 4, stmts),
            "movq" => self.vec_mov_gp(ops, 8, stmts),
            // The comparisons that set flags instead of writing a mask.
            "ucomiss" | "comiss" => self.vec_set_flags(ops, 4, stmts),
            "ucomisd" | "comisd" => self.vec_set_flags(ops, 8, stmts),
            "cmpss" => self.vec_each_compare(ops, 1, 4, BinOp::FEq, stmts),
            "cmpsd" => self.vec_each_compare(ops, 1, 8, BinOp::FEq, stmts),
            "cmpps" => self.vec_each_compare(ops, 4, 4, BinOp::FEq, stmts),
            "cmppd" => self.vec_each_compare(ops, 2, 8, BinOp::FEq, stmts),
            "emms" => {}
            _ => return false,
        }
        true
    }

    /// `movhps`/`movhpd`: the low 64 bits of the source into the high
    /// element of the destination.
    fn vec_each_high(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let Operand::Vec(class, reg) = dst else {
            return;
        };
        let source = self.vec_source(src, 0, 8, stmts);
        self.vec_write_wide(*class, *reg, 2, source, stmts);
    }

    /// `pandn`: the destination's complement and the source.
    fn vec_each_not_and(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let mut values = Vec::new();
        for index in 0..4 {
            let old = self.vec_old(dst, index, 4, stmts);
            let source = self.vec_source(src, index, 4, stmts);
            values.push(Expr::Bin {
                op: BinOp::And,
                lhs: Box::new(Expr::Un {
                    op: UnOp::Not,
                    arg: Box::new(old),
                    bits: 4,
                }),
                rhs: Box::new(source),
                bits: 4,
            });
        }
        for (index, value) in values.into_iter().enumerate() {
            self.vec_store(dst, index as u8, 4, value, stmts);
        }
    }

    /// A shift of every element by the same count.
    fn vec_each_shift(
        &mut self,
        ops: &[Operand],
        count: u8,
        elem: u8,
        op: BinOp,
        stmts: &mut Vec<Stmt>,
    ) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        // The count is either an immediate or the low element of a register.
        let amount = match src {
            Operand::Imm(v) => Expr::Const(*v),
            other => self.vec_source(other, 0, 4, stmts),
        };
        let mut values = Vec::new();
        for index in 0..count {
            let old = self.vec_old(dst, index, elem, stmts);
            values.push(Expr::Bin {
                op,
                lhs: Box::new(old),
                rhs: Box::new(amount.clone()),
                bits: elem,
            });
        }
        for (index, value) in values.into_iter().enumerate() {
            self.vec_store(dst, index as u8, elem, value, stmts);
        }
    }

    /// `shufps`: the destination's elements are picked from the source and
    /// from the destination by an immediate.
    fn vec_shufps(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let Some(imm) = ops.get(2).and_then(|o| match o {
            Operand::Imm(v) => Some(*v as u8),
            _ => None,
        }) else {
            return;
        };
        let source: Vec<Expr> = (0..4).map(|i| self.vec_source(src, i, 4, stmts)).collect();
        let old: Vec<Expr> = (0..4).map(|i| self.vec_old(dst, i, 4, stmts)).collect();
        let picks = [
            source[(imm & 3) as usize].clone(),
            source[((imm >> 2) & 3) as usize].clone(),
            old[((imm >> 4) & 3) as usize].clone(),
            old[((imm >> 6) & 3) as usize].clone(),
        ];
        for (index, value) in picks.into_iter().enumerate() {
            self.vec_store(dst, index as u8, 4, value, stmts);
        }
    }

    /// `shufpd`: the destination's 64-bit elements are picked from the
    /// source's by an immediate.
    fn vec_shufpd(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let Some(imm) = ops.get(2).and_then(|o| match o {
            Operand::Imm(v) => Some(*v as u8),
            _ => None,
        }) else {
            return;
        };
        let first = self.vec_source(src, imm & 1, 8, stmts);
        let second = self.vec_source(src, (imm >> 1) & 1, 8, stmts);
        self.vec_store(dst, 0, 8, first, stmts);
        self.vec_store(dst, 1, 8, second, stmts);
    }

    /// `unpcklps`/`unpckhps`: the source's elements are interleaved with
    /// the destination's.
    fn vec_unpack(&mut self, ops: &[Operand], high: bool, stmts: &mut Vec<Stmt>) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let base = if high { 2 } else { 0 };
        let source: Vec<Expr> = (0..2).map(|i| self.vec_source(src, base + i, 4, stmts)).collect();
        let old: Vec<Expr> = (0..2).map(|i| self.vec_old(dst, base + i, 4, stmts)).collect();
        let picks = [
            source[0].clone(),
            old[0].clone(),
            source[1].clone(),
            old[1].clone(),
        ];
        for (index, value) in picks.into_iter().enumerate() {
            self.vec_store(dst, index as u8, 4, value, stmts);
        }
    }

    /// `unpcklpd`/`unpckhpd`/`punpcklqdq`/`punpckhqdq`: the 64-bit
    /// elements are interleaved.
    fn vec_unpack_wide(&mut self, ops: &[Operand], high: bool, stmts: &mut Vec<Stmt>) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let base = if high { 1 } else { 0 };
        let source = self.vec_source(src, base, 8, stmts);
        let old = self.vec_old(dst, base, 8, stmts);
        self.vec_store(dst, base, 8, source, stmts);
        self.vec_store(dst, base + 1, 8, old, stmts);
    }

    /// `pshufd`: the destination's elements are picked from the source's
    /// by an immediate.
    fn vec_pshufd(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let Some(imm) = ops.get(2).and_then(|o| match o {
            Operand::Imm(v) => Some(*v as u8),
            _ => None,
        }) else {
            return;
        };
        let source: Vec<Expr> = (0..4).map(|i| self.vec_source(src, i, 4, stmts)).collect();
        for index in 0..4 {
            let pick = ((imm >> (2 * index)) & 3) as usize;
            self.vec_store(dst, index, 4, source[pick].clone(), stmts);
        }
    }

    /// `movsldup`/`movddup`: every element is a copy of another one.
    fn vec_dup(&mut self, ops: &[Operand], step: u8, stmts: &mut Vec<Stmt>) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let mut values = Vec::new();
        for index in 0..4 {
            values.push(self.vec_source(src, index / step, 4, stmts));
        }
        for (index, value) in values.into_iter().enumerate() {
            self.vec_store(dst, index as u8, 4, value, stmts);
        }
    }

    /// `movshdup`: every element is a copy of the one above it.
    fn vec_dup_high(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let mut values = Vec::new();
        for index in 0..4 {
            values.push(self.vec_source(src, index / 2 * 2 + 1, 4, stmts));
        }
        for (index, value) in values.into_iter().enumerate() {
            self.vec_store(dst, index as u8, 4, value, stmts);
        }
    }

    /// A conversion from an integer register to a float element.
    fn vec_cvtsi(&mut self, ops: &[Operand], elem: u8, op: BinOp, stmts: &mut Vec<Stmt>) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let value = self.core.read(src, stmts);
        let value = Expr::Bin {
            op,
            lhs: Box::new(value),
            rhs: Box::new(Expr::Const(0)),
            bits: src.size(),
        };
        self.vec_store(dst, 0, elem, value, stmts);
    }

    /// A conversion from a float element to an integer register.
    fn vec_cvt_to_int(
        &mut self,
        ops: &[Operand],
        elem: u8,
        op: BinOp,
        stmts: &mut Vec<Stmt>,
    ) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let value = self.vec_source(src, 0, elem, stmts);
        let value = Expr::Bin {
            op,
            lhs: Box::new(value),
            rhs: Box::new(Expr::Const(0)),
            bits: elem,
        };
        self.core.write(dst, value, stmts);
    }

    /// A conversion that keeps the number of elements and changes their
    /// width.
    fn vec_cast(
        &mut self,
        ops: &[Operand],
        from: u8,
        to: u8,
        op: BinOp,
        stmts: &mut Vec<Stmt>,
    ) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let count = if from == 8 { 2 } else { 4 };
        let mut values = Vec::new();
        for index in 0..count {
            let source = self.vec_source(src, index, from, stmts);
            values.push(Expr::Bin {
                op,
                lhs: Box::new(source),
                rhs: Box::new(Expr::Const(0)),
                bits: from,
            });
        }
        for (index, value) in values.into_iter().enumerate() {
            self.vec_store(dst, index as u8, to, value, stmts);
        }
    }

    /// `cvtps2pd`: two single-precision elements become one double.
    fn vec_widen(&mut self, ops: &[Operand], from: u8, to: u8, op: BinOp, stmts: &mut Vec<Stmt>) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let mut values = Vec::new();
        for index in 0..2 {
            let source = self.vec_source(src, index, from, stmts);
            values.push(Expr::Bin {
                op,
                lhs: Box::new(source),
                rhs: Box::new(Expr::Const(0)),
                bits: from,
            });
        }
        for (index, value) in values.into_iter().enumerate() {
            self.vec_store(dst, index as u8, to, value, stmts);
        }
    }

    /// `cvtpd2ps`: one double becomes two single-precision elements.
    fn vec_narrow_floats(
        &mut self,
        ops: &[Operand],
        from: u8,
        to: u8,
        op: BinOp,
        stmts: &mut Vec<Stmt>,
    ) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let source = self.vec_source(src, 0, from, stmts);
        let value = Expr::Bin {
            op,
            lhs: Box::new(source),
            rhs: Box::new(Expr::Const(0)),
            bits: from,
        };
        self.vec_store(dst, 0, to, value.clone(), stmts);
        self.vec_store(dst, 1, to, value, stmts);
    }

    /// `movd`/`movq`: a move between a vector register and a
    /// general-purpose one.
    fn vec_mov_gp(&mut self, ops: &[Operand], elem: u8, stmts: &mut Vec<Stmt>) {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        match (dst, src) {
            (Operand::Vec(..), Operand::Reg(..)) => {
                let value = self.core.read(src, stmts);
                self.vec_store(dst, 0, elem, value, stmts);
            }
            (Operand::Reg(..), Operand::Vec(..)) => {
                let value = self.vec_source(src, 0, elem, stmts);
                self.core.write(dst, value, stmts);
            }
            _ => {}
        }
    }

    /// `ucomiss`/`ucomisd`: the comparison sets flags instead of writing
    /// a mask.
    fn vec_set_flags(&mut self, ops: &[Operand], elem: u8, stmts: &mut Vec<Stmt>) {
        let (Some(lhs), Some(rhs)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let left = self.vec_source(lhs, 0, elem, stmts);
        let right = self.vec_source(rhs, 0, elem, stmts);
        self.core.pending_cmp = Some(super::PendingCmp::float(left, right, elem));
    }

    /// A vector instruction whose elements are narrower than the model
    /// keeps.
    ///
    /// The operation is still lifted, at the model's width, and the
    /// narrowing is recorded as a note: the output then says what was
    /// approximated instead of pretending the width was right.
    fn vec_narrow_note(&mut self) {
        self.core.notes.push((
            Addr::NULL,
            "a vector instruction on 8- or 16-bit elements is modelled at 32-bit width".into(),
        ));
    }

    // ------------------------------------------------------------------
    // The x87 stack
    // ------------------------------------------------------------------

    /// The physical slot a logical stack slot lives in.
    fn fpu_physical(&self, slot: u8) -> u8 {
        (self.fpu_top + slot) & 7
    }

    /// The value in a logical stack slot.
    fn fpu_read(&self, slot: u8) -> Expr {
        Expr::var(fpu_slot_name(self.fpu_physical(slot)))
    }

    /// Push a value onto the x87 stack.
    fn fpu_push(&mut self, value: Expr, stmts: &mut Vec<Stmt>) {
        if self.fpu_top == 0 {
            self.core.notes.push((
                Addr::NULL,
                "the x87 stack overflowed: the pushed value is lost".into(),
            ));
            return;
        }
        self.fpu_top -= 1;
        self.fpu_assign(self.fpu_top, value, stmts);
    }

    /// Write a value into a slot of the x87 stack.
    ///
    /// The write is recorded, because a function that leaves a value on
    /// the x87 stack returns it from there.
    fn fpu_assign(&mut self, slot: u8, value: Expr, stmts: &mut Vec<Stmt>) {
        self.wrote_x87 = true;
        self.core.assign(&fpu_slot_name(slot), value, stmts);
    }

    /// Pop a value off the x87 stack.
    fn fpu_pop(&mut self) {
        if self.fpu_top >= 7 {
            self.core.notes.push((
                Addr::NULL,
                "the x87 stack underflowed: there was nothing to pop".into(),
            ));
            return;
        }
        self.fpu_top += 1;
    }

    /// Lift an x87 instruction.
    ///
    /// The three arithmetic groups differ in where the result lands and
    /// not in what it computes: `D8` writes the top of the stack, `DC`
    /// writes the slot it names, and `DE` writes that slot and pops. The
    /// arithmetic is the same in all three, so the mnemonic says which
    /// operation it is and the operands say where the result goes.
    fn lift_x87(&mut self, m: &str, ops: &[Operand], stmts: &mut Vec<Stmt>) -> bool {
        match m {
            "fld" => {
                let Some(src) = ops.first() else { return true };
                match src {
                    Operand::Fpu(slot) => {
                        let value = self.fpu_read(*slot);
                        self.fpu_push(value, stmts);
                    }
                    other => {
                        // The width is the operand's own: `fld tbyte` reads
                        // ten bytes of extended precision, and reading
                        // eight of them would name the wrong slot.
                        let width = other.size();
                        let value = self.core.read_load(
                            other,
                            super::Access {
                                width,
                                signed: false,
                            },
                            stmts,
                        );
                        self.fpu_push(value, stmts);
                    }
                }
            }
            "fst" | "fstp" => {
                let Some(dst) = ops.first() else { return true };
                let value = self.fpu_read(0);
                match dst {
                    Operand::Fpu(slot) => {
                        let physical = self.fpu_physical(*slot);
                        self.fpu_assign(physical, value, stmts);
                    }
                    other => {
                        let width = other.size();
                        self.core.store(
                            other,
                            value,
                            super::Access {
                                width,
                                signed: false,
                            },
                            stmts,
                        );
                    }
                }
                if m == "fstp" {
                    self.fpu_pop();
                }
            }
            "fxch" => {
                let Some(Operand::Fpu(slot)) = ops.first() else {
                    return true;
                };
                // The two slots trade places. The value that leaves the
                // top has to be held somewhere first: reading it by name
                // after the other slot has been written would read the
                // new value, not the old one.
                let top = self.fpu_read(0);
                let other = self.fpu_read(*slot);
                let held = self.new_temp_named("x87_");
                self.core.assign(&held, top, stmts);
                self.fpu_assign(self.fpu_top, other, stmts);
                self.fpu_assign(self.fpu_physical(*slot), Expr::var(held), stmts);
            }
            "fadd" | "fmul" | "fsub" | "fsubr" | "fdiv" | "fdivr" | "faddp" | "fmulp"
            | "fsubp" | "fsubrp" | "fdivp" | "fdivrp" => {
                let (Some(Operand::Fpu(first)), Some(Operand::Fpu(second))) =
                    (ops.first(), ops.get(1))
                else {
                    return true;
                };
                // All three groups compute the same thing: the top of the
                // stack is the left operand and the slot the instruction
                // names is the right one. The groups differ in where the
                // result lands, not in what it computes. That is what the
                // hardware says, and it is not what the manual says: the
                // manual writes the subtraction the other way round.
                //
                // The decoder prints the destination first, so the two
                // operands arrive in the order the group puts them: `D8`
                // names `st(0)` first, `DC` and `DE` name the other slot
                // first. Whichever operand names `st(0)` is therefore the
                // left one.
                let (top_slot, other_slot) = if *first == 0 {
                    (*first, *second)
                } else {
                    (*second, *first)
                };
                let top = self.fpu_read(top_slot);
                let other = self.fpu_read(other_slot);
                // The `r` forms compute the other way round: `fsubr` is
                // `b - a`, not `a - b`. The `p` suffix belongs to the
                // operation, so the name is stripped before it is read.
                let base = m.strip_suffix('p').unwrap_or(m);
                let reversed = base.ends_with('r');
                let (lhs, rhs) = if reversed { (other, top) } else { (top, other) };
                let op = match base {
                    "fadd" => BinOp::FAdd,
                    "fmul" => BinOp::FMul,
                    "fsub" | "fsubr" => BinOp::FSub,
                    "fdiv" | "fdivr" => BinOp::FDiv,
                    _ => BinOp::FAdd,
                };
                let value = Expr::Bin {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    bits: 10,
                };
                // The destination is the slot the instruction names: the
                // top when it names the top, the other slot otherwise.
                let dest = if *first == 0 {
                    self.fpu_top
                } else {
                    self.fpu_physical(*first)
                };
                self.fpu_assign(dest, value, stmts);
                if m.ends_with('p') {
                    self.fpu_pop();
                }
            }
            "fcom" | "fcomi" | "fucomi" | "fcomip" | "fucomip" => {
                let other = match ops.first() {
                    Some(Operand::Fpu(slot)) => *slot,
                    _ => 1,
                };
                let left = self.fpu_read(0);
                let right = self.fpu_read(other);
                self.core.pending_cmp = Some(super::PendingCmp::float(left, right, 10));
                if matches!(m, "fcomip" | "fucomip") {
                    self.fpu_pop();
                }
            }
            "fcomp" | "fcompp" => {
                let other = match ops.first() {
                    Some(Operand::Fpu(slot)) => *slot,
                    _ => 1,
                };
                let left = self.fpu_read(0);
                let right = self.fpu_read(other);
                self.core.pending_cmp = Some(super::PendingCmp::float(left, right, 10));
                self.fpu_pop();
                if m == "fcompp" {
                    self.fpu_pop();
                }
            }
            "fchs" => {
                let value = self.fpu_read(0);
                let value = Expr::Bin {
                    op: BinOp::FNeg,
                    lhs: Box::new(value),
                    rhs: Box::new(Expr::Const(0)),
                    bits: 10,
                };
                self.fpu_assign(self.fpu_top, value, stmts);
            }
            "fabs" => {
                let value = self.fpu_read(0);
                let value = Expr::Bin {
                    op: BinOp::FAbs,
                    lhs: Box::new(value),
                    rhs: Box::new(Expr::Const(0)),
                    bits: 10,
                };
                self.fpu_assign(self.fpu_top, value, stmts);
            }
            "fsqrt" => {
                let value = self.fpu_read(0);
                let value = Expr::Bin {
                    op: BinOp::FSqrt,
                    lhs: Box::new(value),
                    rhs: Box::new(Expr::Const(0)),
                    bits: 10,
                };
                self.fpu_assign(self.fpu_top, value, stmts);
            }
            "fldz" => self.fpu_push(Expr::Const(0), stmts),
            "fld1" => self.fpu_push(Expr::Unknown("1.0L".into()), stmts),
            "fldpi" => self.fpu_push(Expr::Unknown("3.14159265358979323846L".into()), stmts),
            "fldl2e" => self.fpu_push(Expr::Unknown("1.44269504088896340736L".into()), stmts),
            "fldlg2" => self.fpu_push(Expr::Unknown("0.30102999566398119521L".into()), stmts),
            "fldln2" => self.fpu_push(Expr::Unknown("0.69314718055994530942L".into()), stmts),
            "fild" => {
                let Some(src) = ops.first() else { return true };
                let width = src.size();
                let value = self.core.read_load(
                    src,
                    super::Access {
                        width,
                        signed: false,
                    },
                    stmts,
                );
                let value = Expr::Bin {
                    op: BinOp::I2F80,
                    lhs: Box::new(value),
                    rhs: Box::new(Expr::Const(0)),
                    bits: width,
                };
                self.fpu_push(value, stmts);
            }
            "fistp" | "fisttp" | "fist" => {
                let Some(dst) = ops.first() else { return true };
                let width = dst.size();
                let value = self.fpu_read(0);
                let value = Expr::Bin {
                    op: if width == 8 {
                        BinOp::F2I64
                    } else {
                        BinOp::F2I32
                    },
                    lhs: Box::new(value),
                    rhs: Box::new(Expr::Const(0)),
                    bits: 10,
                };
                self.core.store(
                    dst,
                    value,
                    super::Access {
                        width,
                        signed: false,
                    },
                    stmts,
                );
                if m != "fist" {
                    self.fpu_pop();
                }
            }
            "fnstsw" | "fnstcw" | "fldcw" | "fnstenv" | "fldenv" | "frstor"
            | "fnsave" | "finit" | "fnclex" => {
                // The control and status words have no place in the
                // decompiler's world; saying so is better than dropping
                // them.
                stmts.push(Stmt::Comment(format!("{m} ; not modelled")));
            }
            _ => return false,
        }
        true
    }

    // ------------------------------------------------------------------
    // Atomic instructions
    // ------------------------------------------------------------------

    /// Lift an atomic read-modify-write.
    fn lift_atomic(&mut self, m: &str, ops: &[Operand], stmts: &mut Vec<Stmt>) -> bool {
        // The mnemonic decides first. An instruction with one operand is
        // not an atomic read-modify-write, and claiming it was would
        // swallow the call, the jump and the push.
        if m != "xadd" && m != "cmpxchg" {
            return false;
        }
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return true;
        };
        match m {
            "xadd" => {
                let width = dst.size();
                // Both values are named before either write. The old value is
                // needed twice -- in the sum and in the source register -- and
                // the addend is needed once, but `xadd al, cl` writes `cl` with
                // the old `al` before it computes the sum, so reading the
                // addend after the first write adds the wrong register.
                let held = self.snapshot(dst, stmts);
                let addend = self.snapshot(src, stmts);
                let sum = Expr::Bin {
                    op: BinOp::Add,
                    lhs: Box::new(held.clone()),
                    rhs: Box::new(addend),
                    bits: width,
                };
                // The old value goes back to the source register, and that
                // write comes first: `xadd al, al` names one register twice,
                // the processor ends with the sum in it, and writing the
                // source last would leave the value it started with.
                self.core.write(src, held, stmts);
                self.core.write(dst, sum, stmts);
                self.core.notes.push((
                    Addr::NULL,
                    "xadd: an atomic read-modify-write; no other thread sees the half state"
                        .into(),
                ));
            }
            "cmpxchg" => {
                let width = dst.size();
                let value = self.core.read(dst, stmts);
                // The memory's old value is read once and used twice: for
                // the comparison and for the value the accumulator keeps
                // when the comparison fails. Reading it twice would read it
                // after the store, which is the new value.
                let held_memory = self.new_temp_named("old_");
                self.core.assign(&held_memory, value, stmts);
                // The comparison is against the accumulator, which is the
                // register the instruction's width selects.
                let accumulator = Operand::Reg(Reg::view(0, width, 0));
                // Named like the destination: the accumulator is written
                // before the exchange, and a condition that read it live would
                // compare the destination against the value the exchange just
                // put there -- equal every time, and the exchange always taken.
                let held = self.snapshot(&accumulator, stmts);
                let same = Expr::Cond {
                    op: CondOp::Eq,
                    lhs: Box::new(Expr::var(held_memory.clone())),
                    rhs: Box::new(held.clone()),
                    bits: width,
                };
                // The flags are the ones `cmp` would leave, and `cmp`
                // subtracts the destination from the accumulator.
                self.core.pending_cmp = Some(super::PendingCmp::int(
                    held.clone(),
                    Expr::var(held_memory.clone()),
                    width,
                ));
                let wanted = self.snapshot(src, stmts);
                // Equal: the destination takes the source and the accumulator
                // keeps what it had. Not equal: the accumulator takes the
                // destination's old value and the destination is untouched.
                //
                // The accumulator is written first. `cmpxchg al, cl` names the
                // accumulator as the destination too, and writing it last left
                // the old value there instead of the source the exchange had
                // just stored.
                self.core.write(
                    &accumulator,
                    Expr::Select {
                        cond: Box::new(same.clone()),
                        yes: Box::new(held),
                        no: Box::new(Expr::var(held_memory.clone())),
                    },
                    stmts,
                );
                self.core.write(
                    dst,
                    Expr::Select {
                        cond: Box::new(same),
                        yes: Box::new(wanted),
                        no: Box::new(Expr::var(held_memory)),
                    },
                    stmts,
                );
                self.core.notes.push((
                    Addr::NULL,
                    "cmpxchg: an atomic compare-and-exchange".into(),
                ));
            }
            _ => return false,
        }
        true
    }

    // ------------------------------------------------------------------
    // String instructions
    // ------------------------------------------------------------------

    /// Lift a string instruction.
    ///
    /// The source, the destination and the count are the registers the
    /// instruction names implicitly: `rsi`, `rdi` and `rcx`. With `rep` the
    /// instruction is a loop, and a loop is not unrolled: the output says
    /// how many elements move and leaves the copying itself alone.
    fn lift_string(
        &mut self,
        m: &str,
        ops: &[Operand],
        repeated: bool,
        addr: Addr,
        stmts: &mut Vec<Stmt>,
    ) -> bool {
        let Some(Operand::Mem { size, .. }) = ops.first() else {
            return false;
        };
        let width = if *size == 0 { 1 } else { *size };
        match m {
            "movsb" | "movs" => {
                let Some(Operand::Mem { .. }) = ops.get(1) else {
                    return true;
                };
                if repeated {
                    // The instruction is a loop, and the IR has no loop and
                    // no bulk copy. The copy is therefore written as the one
                    // operation it is: the whole count, moved at once. The
                    // pointers and the count are then what the machine
                    // leaves behind.
                    let source = self.elem_address(&ops[0], 0, stmts);
                    let destination = self.elem_address(&ops[1], 0, stmts);
                    let count = self.count_times(width);
                    stmts.push(Stmt::Call {
                        name: "memcpy".into(),
                        args: vec![destination, source, count.clone()],
                        ret: None,
                        addr,
                        target: None,
                    });
                    self.bump(6, count.clone(), stmts);
                    self.bump(7, count, stmts);
                    let count_reg = self.reg(RCX);
                    self.core.assign(&count_reg, Expr::Const(0), stmts);
                } else {
                    let source = self.elem_address(&ops[0], 0, stmts);
                    let destination = self.elem_address(&ops[1], 0, stmts);
                    stmts.push(Stmt::Store {
                        addr: destination,
                        value: Expr::Load {
                            addr: Box::new(source),
                            size: width,
                            signed: false,
                        },
                        size: width,
                    });
                    let step = Expr::Const(u64::from(width));
                    self.bump(6, step.clone(), stmts);
                    self.bump(7, step, stmts);
                }
                self.core.notes.push((
                    addr,
                    if repeated {
                        format!("{m}: a loop of {width}-byte copies, written as one memcpy")
                    } else {
                        format!("{m}: one {width}-byte copy")
                    },
                ));
            }
            "stosb" | "stos" => {
                let destination = self.elem_address(&ops[0], 0, stmts);
                // The value stored is the accumulator, whose width the
                // instruction's operand size selects.
                // The value stored is the accumulator, read at the width the
                // instruction works on: `al` is the low byte of `rax`, not a
                // variable of its own.
                let accumulator =
                    self.core
                        .read(&Operand::Reg(Reg::view(0, width, 0)), stmts);
                if repeated {
                    let count = self.count_times(width);
                    stmts.push(Stmt::Call {
                        name: "memset".into(),
                        args: vec![destination, accumulator, count.clone()],
                        ret: None,
                        target: None,
                        addr,
                    });
                    self.bump(7, count, stmts);
                    let count_reg = self.reg(RCX);
                    self.core.assign(&count_reg, Expr::Const(0), stmts);
                } else {
                    stmts.push(Stmt::Store {
                        addr: destination,
                        value: accumulator,
                        size: width,
                    });
                    self.bump(7, Expr::Const(u64::from(width)), stmts);
                }
                self.core.notes.push((
                    addr,
                    if repeated {
                        format!("{m}: a loop of {width}-byte stores, written as one memset")
                    } else {
                        format!("{m}: one {width}-byte store")
                    },
                ));
            }
            _ => return false,
        }
        true
    }
}



/// The name of one element of a vector register.
///
/// A vector register is modelled as the 32-bit elements it holds. That is
/// the width most of the machine's vector instructions work on — the
/// packed single-precision and the integer sets both use it — and the
/// 64-bit float operations read two adjacent elements as one value.
fn vec_element(class: VecClass, reg: u8, index: u8) -> String {
    let prefix = match class {
        VecClass::Xmm => "xmm",
        VecClass::Ymm => "ymm",
        VecClass::Mm => "mm",
    };
    format!("{prefix}{reg}_{index}")
}

/// The name of one physical slot of the x87 stack.
///
/// The slots are numbered by the hardware, and the hardware's numbering
/// runs downwards: the first value a function pushes is in the highest
/// numbered slot. Naming them the way a reader counts them — the top is
/// the first — keeps the output's names in the order the stack holds.
fn fpu_slot_name(physical: u8) -> String {
    format!("st_{}", 7u8.saturating_sub(physical))
}

impl super::Lifter for Lifter<'_> {
    fn lift_block(&mut self, insns: &[Instruction]) -> Lifted {
        let mut out = Lifted::default();
        // The register state is per block, not per function. A block that
        // prepares a call's arguments is a different block from the one that
        // dispatched to it, and the second must not inherit the first's
        // registers: that made `puts(leaked, spilled)` where the callee
        // takes one argument.
        self.core.written.clear();
        // The values pushed for a call are per block for the same reason: a
        // push and the call it feeds are always adjacent, and carrying the
        // list across a block boundary would give the next call arguments
        // that belong to a path this block cannot see.
        self.pushed.clear();
        for insn in insns {
            let before = out.stmts.len();
            self.lift_insn(insn, &mut out);
            for _ in before..out.stmts.len() {
                out.stmt_at.push(insn.addr);
            }
        }
        let last = insns.last();
        if let Some(insn) = last {
            // The loop family carries its own condition: it is a question
            // about the counter and not one of the machine's condition
            // codes.
            if let Some(condition) = self.loop_condition.take() {
                out.branch = Some(condition);
            } else if matches!(insn.flow, FlowKind::ConditionalBranch) {
                let parsed = self.core.target.parse(&insn.text);
                out.branch = Some(match parsed {
                    Some(p) => self.core.take_condition(&p),
                    None => Expr::Unknown("condition".into()),
                });
            } else if matches!(insn.flow, FlowKind::IndirectBranch)
                || (matches!(insn.flow, FlowKind::UnconditionalBranch)
                    && insn.target.is_none())
            {
                // The operand is the whole point: `jmp *rax` and
                // `jmp *(%rdx,%rax,8)` both leave through a value the block
                // computed, and reading that value is what shows the table
                // behind a switch the recovery did not name.
                let target = match self.core.target.parse(&insn.text) {
                    Some(parsed) => match parsed.ops.first() {
                        Some(op) => self.core.read(op, &mut out.stmts),
                        None => Expr::Unknown("indirect target".into()),
                    },
                    None => Expr::Unknown("indirect target".into()),
                };
                out.indirect = Some(target);
            }
        }
        out
    }

    fn call_args(&self) -> Vec<Expr> {
        self.args_for_call()
    }

    fn take_call_args(&mut self) -> Vec<Expr> {
        self.take_args_for_call()
    }

    fn memory_args(&self) -> Vec<String> {
        if !self.mode.is_32() {
            return Vec::new();
        }
        // A 32-bit function reads its arguments at fixed offsets above the
        // return address, so they are the slots at 4, 8, 12 and so on from
        // the frame base. The frame slot naming already calls those `arg_N`,
        // which is what makes them findable here without another table.
        (0..MAX_STACK_ARGS)
            .map(|i| slot_name(4 + 4 * i as i64, self.width()))
            .collect()
    }

    fn declared_args(&self) -> Option<usize> {
        // `ret imm` states the argument bytes a callee removes. Only the
        // 32-bit mode has a fixed argument width for that to be a count
        // rather than a lower bound.
        let bytes = self.pops?;
        if !self.mode.is_32() || bytes % u64::from(self.width()) != 0 {
            return None;
        }
        Some((bytes / u64::from(self.width())) as usize)
    }

    fn ret_reg(&self) -> String {
        // The x86-64 ABI returns an integer or a pointer in `rax`, a
        // floating-point value in `xmm0`, and an extended-precision one on
        // the x87 stack. Which of the three a function returns is a fact
        // about its type, and the type is not in the code. What the code
        // does say is where the function left its last value, so that is
        // what decides.
        if self.core.ever_written.contains(&RAX) {
            return self.reg(RAX);
        }
        if self.wrote_vector {
            return "xmm0_0".to_string();
        }
        // A value left on the x87 stack is in the top slot, and the top
        // slot is where the ABI says the return value is. An empty stack
        // means the function returned something else.
        if self.wrote_x87 && self.fpu_top < 8 {
            return fpu_slot_name(self.fpu_top);
        }
        self.reg(RAX)
    }

    fn ret_type(&self) -> Option<&'static str> {
        if self.core.ever_written.contains(&RAX) {
            // A 32-bit function returns what fits in `eax`, and calling that
            // `u64` would tell a caller the value has an upper half it has
            // never held.
            return Some(if self.mode.is_32() { "u32" } else { "u64" });
        }
        if self.wrote_vector {
            return Some(match self.float_width {
                4 => "float",
                _ => "double",
            });
        }
        if self.wrote_x87 && self.fpu_top < 8 {
            return Some("long double");
        }
        None
    }

    fn ret_defined(&self) -> bool {
        // The function returned a value only if it wrote the place the
        // value comes from.
        self.core.ever_written.contains(&RAX)
            || self.wrote_vector
            || (self.wrote_x87 && self.fpu_top < 8)
    }

    fn arg_reg_names(&self) -> Vec<String> {
        // The order comes from the target, not from the table: a Windows
        // binary names its arguments in `rcx`, `rdx`, `r8`, `r9` order, and
        // taking the System V order here put every argument in the wrong
        // place while the calls themselves looked right.
        self.core
            .target
            .arg_regs()
            .iter()
            .map(|r| self.reg(*r))
            .collect()
    }
    fn sp_alias(&self) -> &'static str {
        self.core.target.sp_alias()
    }

    fn sp_reg_name(&self) -> String {
        self.reg(RSP)
    }

    fn notes(&mut self) -> Vec<(Addr, String)> {
        std::mem::take(&mut self.core.notes)
    }

    fn frame_base_slot(&self, reg: &str) -> Option<i64> {
        let want = self.core.target.fp_reg()?;
        let got = self.core.target.lookup_reg(reg)?.idx;
        if got == want {
            self.core.fp_anchor.or(self.core.fp_off)
        } else {
            None
        }
    }
}

/// Append an immediate operand, for `inc`/`dec`.
fn with_imm(ops: &[Operand], value: u64) -> Vec<Operand> {
    let mut v = ops.to_vec();
    if v.len() == 1 {
        v.push(Operand::Imm(value));
    }
    v
}

/// Register name by index, at 64-bit width.
pub fn reg_name(idx: u8) -> &'static str {
    REG64.get(usize::from(idx)).copied().unwrap_or("rax")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::type_name;
    use crate::lift::Lifter as LifterTrait;
    use occ_core::MAX_INSN_LEN;

    /// Lift a sequence of instructions and render the statements.
    ///
    /// The lifter reads the same text a reader sees, so a test can name the
    /// instructions it means without encoding them by hand.
    fn lift_branch(texts: &[&str]) -> (Vec<String>, Option<String>) {
        let name_of = |_: Addr| "f".to_string();
        let symbols = |_: &str| None;
        static SYSV: X86Target = X86Target::sysv();
        let mut lifter = Lifter::new(&SYSV, &name_of, &symbols);
        let last = texts.len().saturating_sub(1);
        let insns: Vec<Instruction> = texts
            .iter()
            .enumerate()
            .map(|(i, text)| Instruction {
                addr: Addr((i as u64) * 8),
                len: 8,
                bytes: [0; MAX_INSN_LEN],
                text: (*text).to_string(),
                mnemonic: text.split_whitespace().next().unwrap_or("").into(),
                flow: if i == last {
                    FlowKind::ConditionalBranch
                } else {
                    FlowKind::Next
                },
                target: None,
                mem_reads: Vec::new(),
                mem_writes: Vec::new(),
                mem_addr: Vec::new(),
            })
            .collect();
        let lifted = LifterTrait::lift_block(&mut lifter, &insns);
        let stmts = lifted
            .stmts
            .iter()
            .map(|stmt| match stmt {
                Stmt::Assign { dest, value } => format!("{dest} = {}", crate::render::expr(value)),
                Stmt::Store {
                    addr,
                    value,
                    size,
                } => format!(
                    "*({}*) {} = {}",
                    type_name(*size, false),
                    crate::render::expr(addr),
                    crate::render::expr(value)
                ),
                Stmt::Call { name, args, .. } => {
                    let args: Vec<String> =
                        args.iter().map(crate::render::expr).collect();
                    format!("{name}({})", args.join(", "))
                }
                Stmt::If { cond, .. } => {
                    format!("if ({}) {{ ... }}", crate::render::expr(cond))
                }
                Stmt::Comment(c) => format!("// {c}"),
            })
            .collect();
        let branch = lifted.branch.as_ref().map(crate::render::expr);
        (stmts, branch)
    }

    fn lift_texts(texts: &[&str]) -> Vec<String> {
        let name_of = |_: Addr| "f".to_string();
        let symbols = |_: &str| None;
        static SYSV: X86Target = X86Target::sysv();
        let mut lifter = Lifter::new(&SYSV, &name_of, &symbols);
        let insns: Vec<Instruction> = texts
            .iter()
            .enumerate()
            .map(|(i, text)| Instruction {
                addr: Addr((i as u64) * 8),
                len: 8,
                bytes: [0; MAX_INSN_LEN],
                text: (*text).to_string(),
                mnemonic: text.split_whitespace().next().unwrap_or("").into(),
                flow: FlowKind::Next,
                target: None,
                mem_reads: Vec::new(),
                mem_writes: Vec::new(),
                mem_addr: Vec::new(),
            })
            .collect();
        LifterTrait::lift_block(&mut lifter, &insns)
            .stmts
            .iter()
            .map(|stmt| match stmt {
                Stmt::Assign { dest, value } => format!("{dest} = {}", crate::render::expr(value)),
                Stmt::Store {
                    addr,
                    value,
                    size,
                } => format!(
                    "*({}*) {} = {}",
                    type_name(*size, false),
                    crate::render::expr(addr),
                    crate::render::expr(value)
                ),
                Stmt::Call { name, args, .. } => {
                    let args: Vec<String> =
                        args.iter().map(crate::render::expr).collect();
                    format!("{name}({})", args.join(", "))
                }
                Stmt::If { cond, .. } => {
                    format!("if ({}) {{ ... }}", crate::render::expr(cond))
                }
                Stmt::Comment(c) => format!("// {c}"),
            })
            .collect()
    }

    #[test]
    fn parses_the_decoders_operand_forms() {
        let p = parse("mov rax, qword ptr [rip+0x1efb9]  # 0x22fc8").unwrap();
        assert_eq!(p.mnemonic, "mov");
        assert_eq!(p.ops.len(), 2);
        // The resolved address travels beside the operand, not inside it: the
        // operand itself is still `[rip+disp]`, which is what the instruction
        // encodes.
        assert_eq!(p.pc_target, Some(0x22fc8));
        match &p.ops[1] {
            Operand::Mem { base, .. } => assert_eq!(base.unwrap().idx, RIP),
            other => panic!("{other:?}"),
        }

        let p = parse("mov dword ptr [rbp - 0x18], edi").unwrap();
        match &p.ops[0] {
            Operand::Mem {
                size,
                base,
                disp,
                ..
            } => {
                assert_eq!(*size, 4);
                assert_eq!(base.unwrap().idx, RBP);
                assert_eq!(*disp, -0x18);
            }
            other => panic!("{other:?}"),
        }

        let p = parse("lea rdx, [rax + rax*4]").unwrap();
        match &p.ops[1] {
            Operand::Mem { index, scale, .. } => {
                assert_eq!(index.unwrap().idx, RAX);
                assert_eq!(*scale, 4);
            }
            other => panic!("{other:?}"),
        }

        let p = parse("je 0x4014 (rel +0x2)").unwrap();
        assert_eq!(p.ops, vec![Operand::Imm(0x4014)]);
    }

    #[test]
    fn high_byte_registers_carry_their_offset() {
        let r = X86Target::sysv().lookup_reg("ah").unwrap();
        assert_eq!(r.idx, RAX);
        assert_eq!(r.size, 1);
        assert_eq!(r.offset, 8);
    }

    #[test]
    fn the_x87_stack_computes_the_hardware_arithmetic() {
        // The sequence gcc emits for `a * b + a / b`, with the long
        // doubles passed on the stack.
        let stmts = lift_texts(&[
            "fld tbyte ptr [rsp + 0x8]",
            "fld tbyte ptr [rsp + 0x18]",
            "fld st(1)",
            "fmul st(0), st(1)",
            "fxch st(2)",
            "fdivp st(1), st(0)",
            "faddp st(1), st(0)",
        ]);
        for s in &stmts {
            println!("{s}");
        }
        // The two arguments are read as ten-byte extended-precision slots.
        assert!(
            stmts.iter().any(|s| s.contains("arg_8_f80")),
            "{stmts:#?}"
        );
        assert!(
            stmts.iter().any(|s| s.contains("arg_18_f80")),
            "{stmts:#?}"
        );
        // The multiplication is `a * b` and the division is `a / b`, in that
        // order, and the sum is the two of them.
        assert!(
            stmts.iter().any(|s| s.contains("f80_mul(")),
            "{stmts:#?}"
        );
        assert!(
            stmts.iter().any(|s| s.contains("f80_div(")),
            "{stmts:#?}"
        );
        assert!(
            stmts.iter().any(|s| s.contains("f80_add(")),
            "{stmts:#?}"
        );
    }

    /// The condition a branch tests, as it is printed.
    fn branch_of(texts: &[&str]) -> String {
        let (_, branch) = lift_branch(texts);
        branch.unwrap_or_else(|| "none".to_string())
    }

    #[test]
    fn a_comparison_says_how_its_operands_are_read() {
        // The same two registers are two different numbers, and the
        // branch that consumes the comparison has to say which one it
        // asked about.
        assert_eq!(
            branch_of(&["cmp edi, esi", "jl 0x40"]),
            "(i32)rdi < (i32)rsi"
        );
        assert_eq!(
            branch_of(&["cmp edi, esi", "jb 0x40"]),
            "(u32)rdi < (u32)rsi"
        );
        assert_eq!(
            branch_of(&["cmp edi, esi", "je 0x40"]),
            "(u32)rdi == (u32)rsi"
        );
    }

    #[test]
    fn the_sign_flag_is_the_sign_of_the_difference() {
        // `cmp 0, 0x80000000` leaves the sign flag set, because the
        // difference is negative, even though the first operand is not.
        // Reading the sign of the operands would answer a different
        // question.
        assert_eq!(
            branch_of(&["cmp edi, esi", "js 0x40"]),
            "(i32)((u32)rdi - (u32)rsi) < 0"
        );
    }

    #[test]
    fn the_carry_of_an_addition_is_the_sum_that_did_not_fit() {
        // The carry says the sum did not fit, and the sum did not fit
        // when the result is smaller than the value that was there
        // before.
        // The source is named first, so the destination's old value is the
        // second temporary: `add al, al` needs both to survive the write.
        assert_eq!(
            branch_of(&["add edi, esi", "jb 0x40"]),
            "carry((u32)rdi, old_2)"
        );
    }

    #[test]
    fn the_borrow_of_a_subtraction_is_the_difference_that_did_not_fit() {
        // The borrow says the minuend was smaller than the subtrahend.
        assert_eq!(
            branch_of(&["sub edi, esi", "jb 0x40"]),
            "borrow(old_2, old_1)"
        );
    }

    #[test]
    fn the_overflow_of_a_negation_is_the_smallest_value_of_the_width() {
        // The negation of the smallest value is itself, and that is the
        // only value whose negation overflows.
        assert_eq!(
            branch_of(&["neg edi", "jo 0x40"]),
            "(u32)rdi == 0x80000000"
        );
    }

    #[test]
    fn the_zero_flag_of_a_logical_operation_is_the_result() {
        // The logical operations clear the carry and the overflow, and
        // their zero flag is the result.
        assert_eq!(
            branch_of(&["and edi, esi", "je 0x40"]),
            "(u32)rdi == 0"
        );
        assert_eq!(
            branch_of(&["test edi, esi", "jz 0x40"]),
            "(u32)((u32)rdi & (u32)rsi) == 0"
        );
    }

    #[test]
    fn the_loop_family_counts_down_and_branches_on_the_counter() {
        // `loop` decrements the counter and branches while it is not
        // zero. The branch asks about the counter after the decrement.
        let (stmts, branch) = lift_branch(&["loop 0x40"]);
        assert_eq!(branch.as_deref(), Some("rcx != 0"));
        assert!(stmts.iter().any(|s| s.contains("rcx = rcx - 1")));
        // `jrcxz` branches when the counter is zero, and leaves it alone.
        let (stmts, branch) = lift_branch(&["jrcxz 0x40"]);
        assert_eq!(branch.as_deref(), Some("rcx == 0"));
        assert!(!stmts.iter().any(|s| s.contains("rcx = ")));
    }

    #[test]
    fn a_loop_that_wants_the_zero_flag_says_so_when_there_is_none() {
        // `loope` wants the zero flag, and without a comparison before it
        // there is no zero flag to read.
        let (_, branch) = lift_branch(&["loope 0x40"]);
        assert_eq!(branch.as_deref(), Some("<flags>"));
    }

    #[test]
    fn the_mask_of_a_vector_is_the_sign_of_its_elements() {
        let (stmts, _) = lift_branch(&["movmskps eax, xmm0"]);
        let line = stmts.first().expect("a statement");
        // Four elements, four bits, and each bit is the element's sign.
        assert!(line.contains("xmm0_0 >> 0x1f & 1"), "{line}");
        assert!(line.contains("(xmm0_3 >> 0x1f & 1) << 3"), "{line}");
    }
}
