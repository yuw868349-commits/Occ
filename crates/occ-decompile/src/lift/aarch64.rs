//! AArch64 lifting: text to statements.
//!
//! AArch64 differs from x86 in three ways that shape this module.
//!
//! * **One condition.** AArch64 has a single set of condition flags, set by
//!   the instruction that computes them and consumed by the branch. `cmp`
//!   therefore behaves like x86's `cmp`, and `b.eq` like x86's `je`.
//! * **A 32-bit view.** `w0` is the low half of `x0`. Writing `w0` zeroes the
//!   upper half of `x0`, which is what the shared [`Core`] does.
//! * **Pre-indexed addressing.** `ldr x0, [x1, #8]!` writes back to `x1`,
//!   and `stp x29, x30, [sp, #-16]!` is how a frame is set up. The parser
//!   keeps the write-back so the frame bookkeeping sees it.

use occ_core::{Addr, FlowKind, Instruction};

use super::{
    Access, Core, Lifted, ModKind, Operand, Parsed, Reg, Target, parse_int, split_operands,
};
use crate::ir::{BinOp, CastKind, CondOp, Expr, FlagCond, Stmt, UnOp};

/// Register number of the stack pointer.
///
/// AArch64 encodes the zero register and the stack pointer as the same number,
/// 31, and which one it means depends on the instruction. The decoder prints
/// `xzr`/`wzr` for the zero register and `sp`/`wsp` for the stack pointer, so
/// this module keeps them apart: the stack pointer moves the frame, and the
/// zero register reads as a constant.
const SP: u8 = 31;

/// The zero register's pseudo-index.
///
/// It is not a register the architecture numbers — 31 already belongs to the
/// stack pointer — so it gets an index past the end of the real file. Writing
/// there is discarded and reading it yields zero, which is what the hardware
/// does; giving it the stack pointer's number instead made `csel ..., wzr`
/// print as `... : sp`, and turned every comparison against `xzr` into a
/// comparison against an unknown.
const XZR: u8 = 32;

/// Full-width names, by register index.
const X_NAMES: [&str; 32] = [
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13",
    "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26",
    "x27", "x28", "x29", "x30", "sp",
];

/// The AArch64 target.
pub struct Aarch64Target;

impl Target for Aarch64Target {
    fn lookup_reg(&self, name: &str) -> Option<Reg> {
        if let Some(rest) = name.strip_prefix('x') {
            if let Some(n) = parse_int(rest)
                && n < 31
            {
                return Some(Reg::full(n as u8, 8));
            }
        }
        if let Some(rest) = name.strip_prefix('w') {
            if let Some(n) = parse_int(rest)
                && n < 31
            {
                // `w0` is the low half of `x0`.
                return Some(Reg::view(n as u8, 4, 0));
            }
        }
        match name {
            "sp" => Some(Reg::full(SP, 8)),
            "wsp" => Some(Reg::view(SP, 4, 0)),
            // The zero register shares the encoding, not the meaning.
            "xzr" => Some(Reg::full(XZR, 8)),
            "wzr" => Some(Reg::view(XZR, 4, 0)),
            _ => None,
        }
    }

    fn reg_name(&self, idx: u8) -> String {
        if idx == XZR {
            return "xzr".to_string();
        }
        X_NAMES.get(usize::from(idx)).copied().unwrap_or("x0").to_string()
    }

    fn reg_width(&self, _idx: u8) -> u8 {
        8
    }

    fn zero_reg(&self) -> Option<u8> {
        Some(XZR)
    }

    fn arg_regs(&self) -> &[u8] {
        // AAPCS64: x0 through x7.
        &[0, 1, 2, 3, 4, 5, 6, 7]
    }

    fn ret_reg(&self) -> u8 {
        0
    }

    fn sp_reg(&self) -> u8 {
        SP
    }

    fn fp_reg(&self) -> Option<u8> {
        // x29 is the frame pointer in the standard prologue.
        Some(29)
    }

    fn is_pc(&self, _idx: u8) -> bool {
        false
    }

    fn sp_alias(&self) -> &'static str {
        "frame"
    }

    fn branch_cond(&self, mnemonic: &str) -> Option<FlagCond> {
        // `b.eq`, `b.ne`, and the rest of the condition suffix set.
        let suffix = mnemonic.strip_prefix("b.")?;
        cond_of_suffix(suffix)
    }

    fn setcc_cond(&self, suffix: &str) -> Option<FlagCond> {
        cond_of_suffix(suffix)
    }

    fn cmov_cond(&self, suffix: &str) -> Option<FlagCond> {
        cond_of_suffix(suffix)
    }

    fn mem_access(&self, mnemonic: &str, data_size: u8) -> Access {
        // AArch64 puts the width of a plain access in the register, not in
        // the mnemonic: `str w2` stores four bytes and `str x2` stores eight.
        // The byte and half forms name their width in the mnemonic, and the
        // `s` forms sign-extend.
        let (width, signed) = match mnemonic {
            "ldrb" | "ldurb" | "strb" | "sturb" => (1, false),
            "ldrh" | "ldurh" | "strh" | "sturh" => (2, false),
            "ldrsb" | "ldursb" => (1, true),
            "ldrsh" | "ldursh" => (2, true),
            // `ldrsw` always loads 32 bits and sign-extends them, whatever
            // the destination register is.
            "ldrsw" | "ldursw" => (4, true),
            _ => (data_size, false),
        };
        Access { width, signed }
    }

    fn parse(&self, text: &str) -> Option<Parsed> {
        parse_aarch64(text)
    }
}

/// Condition for an AArch64 condition suffix.
///
/// The suffix set is the inverse of x86's: `eq`, `ne`, `cs`/`hs`, `cc`/`lo`,
/// `mi`, `pl`, `vs`, `vc`, `hi`, `ls`, `ge`, `lt`, `gt`, `le`. `cs` and `cc`
/// are the carry versions, which are the unsigned comparisons.
fn cond_of_suffix(suffix: &str) -> Option<FlagCond> {
    Some(match suffix {
        "eq" => FlagCond::Z,
        "ne" => FlagCond::Nz,
        "cs" | "hs" => FlagCond::C,
        "cc" | "lo" => FlagCond::Nc,
        "mi" => FlagCond::S,
        "pl" => FlagCond::Ns,
        "hi" => FlagCond::A,
        "ls" => FlagCond::Be,
        "ge" => FlagCond::Nl,
        "lt" => FlagCond::L,
        "gt" => FlagCond::G,
        "le" => FlagCond::Le,
        // `vs` and `vc` test for overflow.
        "vs" => FlagCond::O,
        "vc" => FlagCond::No,
        _ => return None,
    })
}

/// The condition names AArch64 uses, for recognising an operand that names
/// one.
const COND_NAMES: [&str; 16] = [
    "eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al",
    "nv",
];

/// Parse an AArch64 instruction's text.
fn parse_aarch64(text: &str) -> Option<Parsed> {
    let body = text.trim();
    if body.is_empty() {
        return None;
    }
    let (mnemonic, rest) = match body.split_once(' ') {
        Some((m, r)) => (m, r),
        None => (body, ""),
    };
    let mut ops: Vec<Operand> = Vec::new();
    for part in split_operands(rest.trim()) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        // A shift or an extension modifies the operand before it rather
        // than standing on its own, so it never becomes a fourth operand.
        if let Some((kind, amount)) = parse_modifier(part) {
            match ops.pop() {
                Some(Operand::Reg(reg)) => ops.push(Operand::Modified { reg, kind, amount }),
                Some(other) => {
                    ops.push(other);
                    return None;
                }
                None => return None,
            }
            continue;
        }
        ops.push(parse_aarch64_operand(part)?);
    }
    // The post-indexed form puts the write-back outside the brackets —
    // `ldr w1, [x2], #4` — so the comma splitter saw it as an operand of its
    // own. It belongs to the memory operand: leaving it out dropped the load
    // entirely, because `ldr` then had three operands where it takes two.
    let delta = match ops.last() {
        Some(Operand::Imm(v)) => Some(*v),
        _ => None,
    };
    if let Some(delta) = delta
        && ops.len() >= 2
    {
        let at = ops.len() - 2;
        if let Some(Operand::Mem { base_update, .. }) = ops.get_mut(at)
            && base_update.is_none()
        {
            *base_update = Some(delta as i64);
            ops.pop();
        }
    }
    Some(Parsed {
        mnemonic: mnemonic.to_ascii_lowercase(),
        ops,
        pc_target: None,
        prefixes: Vec::new(),
    })
}

/// Parse the modifier AArch64 writes after a register: `lsl #2`, `sxtw #2`,
/// or a bare `uxtb`.
fn parse_modifier(text: &str) -> Option<(ModKind, u8)> {
    let mut parts = text.split_whitespace();
    let kind = ModKind::of_name(parts.next()?)?;
    let amount = match parts.next() {
        Some(t) => parse_int(t.strip_prefix('#')?)? as u8,
        None => 0,
    };
    if parts.next().is_some() {
        return None;
    }
    Some((kind, amount))
}

/// One AArch64 operand.
fn parse_aarch64_operand(s: &str) -> Option<Operand> {
    let s = s.trim();
    // An immediate: `#0x10`, `#-16`.
    if let Some(rest) = s.strip_prefix('#') {
        return Some(Operand::Imm(parse_int(rest)?));
    }
    // A memory reference: `[base]`, `[base, #off]`, `[base, #off]!`,
    // `[base, xN, lsl #3]`, `[base, xN]`.
    if let Some(inner) = s.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
        return parse_aarch64_memory(inner);
    }
    // A branch target, printed as a bare address.
    if let Some(v) = parse_int(s) {
        return Some(Operand::Imm(v));
    }
    if COND_NAMES.contains(&s) {
        return Some(Operand::Cond(s.to_string()));
    }
    let reg = Aarch64Target.lookup_reg(s)?;
    Some(Operand::Reg(reg))
}

/// The inside of a bracketed memory operand.
///
/// AArch64 has four addressing modes, and the two that move the base register
/// are the ones the frame bookkeeping depends on:
///
/// * `[base]` — no offset, no write-back.
/// * `[base, #off]` — an offset.
/// * `[base, #off]!` — pre-indexed: the access happens at `base + off`, and
///   `base` becomes `base + off`.
/// * `[base], #off` — post-indexed: the access happens at `base`, and `base`
///   becomes `base + off`.
fn parse_aarch64_memory(inner: &str) -> Option<Operand> {
    let inner = inner.trim();
    // The post-indexed form puts the offset outside the brackets.
    let (inner, post_index) = match inner.split_once("], ") {
        Some((head, tail)) => (head, Some(tail.trim())),
        None => (inner, None),
    };
    let mut inner = inner.trim();
    let writeback = inner.ends_with('!');
    if writeback {
        inner = inner.trim_end_matches('!').trim();
    }
    let inner = inner
        .strip_prefix('[')
        .and_then(|r| r.strip_suffix(']'))
        .unwrap_or(inner)
        .trim();
    let mut parts = split_operands(inner);
    if parts.is_empty() {
        return None;
    }
    let base_text = parts.remove(0).trim();
    let base = Aarch64Target.lookup_reg(base_text);
    if base_text.is_empty() {
        return None;
    }
    let mut disp = 0i64;
    let mut index = None;
    let mut scale = 1u64;
    for part in parts {
        let part = part.trim();
        if let Some(rest) = part.strip_prefix('#') {
            disp = disp.wrapping_add(parse_int(rest)? as i64);
            continue;
        }
        // A shift or an extension on the index: `[x1, w2, sxtw #2]`.
        if let Some((kind, amount)) = parse_modifier(part) {
            index?;
            let _ = kind;
            scale = 1u64 << amount;
            continue;
        }
        // A shifted register index: `x2, lsl #3`.
        if let Some((reg_text, shift)) = part.split_once(',') {
            let reg_text = reg_text.trim();
            let shift = shift.trim();
            if let Some(sh) = shift.strip_prefix("lsl #") {
                index = Aarch64Target.lookup_reg(reg_text);
                scale = 1u64 << parse_int(sh)?;
                continue;
            }
        }
        // A register index without a shift.
        if let Some(reg) = Aarch64Target.lookup_reg(part) {
            index = Some(reg);
            continue;
        }
        return None;
    }
    // The write-back amount: pre-indexed uses the offset, post-indexed uses
    // the separate value.
    let base_update = if writeback {
        Some(disp)
    } else {
        post_index.and_then(parse_int).map(|v| v as i64)
    };
    Some(Operand::Mem {
        size: 0,
        base,
        index,
        scale,
        disp,
        abs: None,
        seg: None,
        base_update,
    })
}

/// The AArch64 lifter.
pub struct Lifter<'a> {
    /// Shared state.
    pub core: Core<'a>,
}

impl<'a> Lifter<'a> {
    /// Build a lifter.
    pub fn new(
        target: &'static dyn Target,
        name_of: &'a dyn Fn(Addr) -> String,
        symbols: &'a dyn Fn(&str) -> Option<u64>,
    ) -> Self {
        Lifter {
            core: Core::new(target, name_of).with_symbols(symbols),
        }
    }

    fn lift_insn(&mut self, insn: &Instruction, out: &mut Lifted) {
        let Some(p) = self.core.target.parse(&insn.text) else {
            out.stmts
                .push(Stmt::Comment(format!("{} ; unparsed", insn.text)));
            return;
        };
        self.core.pc_target = p
            .pc_target
            .or_else(|| insn.mem_reads.first().map(|a| a.0))
            .or_else(|| insn.mem_writes.first().map(|a| a.0));
        let m = p.mnemonic.as_str();
        match m {
            "nop" | "hint" | "yield" | "wfe" | "wfi" | "sev" | "isb" | "dmb" | "dsb" | "ret"
            | "eret" | "drps" | "brk" | "hlt" | "sevl" => {}
            "mov" => self.lift_mov(&p.ops, &mut out.stmts),
            "movz" => self.lift_mov_imm(&p.ops, &mut out.stmts, false),
            "movn" => self.lift_mov_imm(&p.ops, &mut out.stmts, true),
            "movk" => self.lift_movk(&p.ops, &mut out.stmts),
            "adr" | "adrp" => self.lift_adr(&p.ops, &mut out.stmts),
            "add" | "adds" => self.lift_alu(BinOp::Add, &p.ops, &mut out.stmts),
            "sub" | "subs" => self.lift_alu(BinOp::Sub, &p.ops, &mut out.stmts),
            "and" | "ands" => self.lift_alu(BinOp::And, &p.ops, &mut out.stmts),
            "orr" => self.lift_alu(BinOp::Or, &p.ops, &mut out.stmts),
            "eor" => self.lift_alu(BinOp::Xor, &p.ops, &mut out.stmts),
            "bic" => self.lift_bic(&p.ops, &mut out.stmts),
            "orn" => self.lift_orn(&p.ops, &mut out.stmts),
            "neg" | "negs" => self.lift_unary(UnOp::Neg, &p.ops, &mut out.stmts),
            "mvn" => self.lift_mvn(&p.ops, &mut out.stmts),
            "lsl" | "lslv" => self.lift_shift(BinOp::Shl, &p.ops, &mut out.stmts),
            "lsr" | "lsrv" => self.lift_shift(BinOp::Shr, &p.ops, &mut out.stmts),
            "asr" | "asrv" => self.lift_shift(BinOp::Sar, &p.ops, &mut out.stmts),
            "ror" => self.lift_ror(&p.ops, &mut out.stmts),
            "mul" => self.lift_mul(&p.ops, &mut out.stmts),
            "madd" => self.lift_madd(&p.ops, &mut out.stmts, false),
            "msub" => self.lift_madd(&p.ops, &mut out.stmts, true),
            "sdiv" => self.lift_div(BinOp::Div, &p.ops, &mut out.stmts),
            "udiv" => self.lift_div(BinOp::UDiv, &p.ops, &mut out.stmts),
            "cmp" => self.lift_cmp(&p.ops, &mut out.stmts),
            "cmn" => self.lift_cmn(&p.ops, &mut out.stmts),
            "tst" => self.lift_tst(&p.ops, &mut out.stmts),
            "csel" | "csinc" | "csinv" | "csneg" => self.lift_csel(m, &p.ops, &mut out.stmts),
            "cset" | "csetm" => self.lift_cset(&p.ops, &mut out.stmts),
            "sxtb" => self.lift_extend(true, 1, &p.ops, &mut out.stmts),
            "sxth" => self.lift_extend(true, 2, &p.ops, &mut out.stmts),
            "sxtw" => self.lift_extend(true, 4, &p.ops, &mut out.stmts),
            "uxtb" => self.lift_extend(false, 1, &p.ops, &mut out.stmts),
            "uxth" => self.lift_extend(false, 2, &p.ops, &mut out.stmts),
            "ldp" => self.lift_pair(&p.ops, &mut out.stmts, false),
            "stp" => self.lift_pair(&p.ops, &mut out.stmts, true),
            "ldr" | "ldur" | "ldrb" | "ldrh" | "ldrsb" | "ldrsh" | "ldrsw" | "ldurh"
            | "ldurb" | "ldursb" | "ldursh" | "ldursw" => {
                self.lift_load(m, &p.ops, &mut out.stmts)
            }
            "str" | "stur" | "strb" | "strh" | "sturb" | "sturh" => {
                self.lift_store(m, &p.ops, &mut out.stmts)
            }
            "bl" | "blr" => self.lift_call(insn, &p.ops, &mut out.stmts),
            "b" | "br" => {}
            // The compare-and-branch forms test a register, not the flags.
            // The condition is built where the block's terminator is, so
            // nothing is lifted here.
            "cbz" | "cbnz" | "tbz" | "tbnz" => {}
            other => {
                if self.core.target.branch_cond(other).is_some() {
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
        let Operand::Reg(dst) = &ops[0] else {
            return;
        };
        // `mov x29, sp` anchors the frame instead of assigning a value.
        if dst.idx == 29
            && matches!(&ops[1], Operand::Reg(r) if r.idx == SP && r.size == 8)
        {
            self.core.fp_off = Some(self.core.sp_off);
            return;
        }
        let value = self.core.read(&ops[1], stmts);
        self.core.write(&ops[0], value, stmts);
    }

    /// `movz`/`movn`: move an immediate, optionally shifted.
    ///
    /// `movz x0, #1, lsl #16` places the value at bit 16, and the zero is
    /// everything else. `movn` moves the complement, so the whole register is
    /// defined.
    fn lift_mov_imm(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>, invert: bool) {
        if ops.len() < 2 {
            return;
        }
        let Operand::Imm(value) = &ops[1] else {
            return;
        };
        let mut value = *value;
        if invert {
            value = !value;
        }
        // The shift is the fourth operand, when present.
        if ops.len() >= 4
            && let Operand::Imm(shift) = &ops[3]
        {
            value = value.wrapping_shl(*shift as u32);
        }
        self.core.write(&ops[0], Expr::Const(value), stmts);
    }

    /// `movk`: keep the other bits and replace one 16-bit field.
    fn lift_movk(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() < 2 {
            return;
        }
        let Operand::Imm(value) = &ops[1] else {
            return;
        };
        let mut shift = 0u32;
        if ops.len() >= 4
            && let Operand::Imm(sh) = &ops[3]
        {
            shift = *sh as u32;
        }
        let dst = match &ops[0] {
            Operand::Reg(r) => self.core.name(r.idx),
            _ => return,
        };
        let field = 0xffffu64 << shift;
        let cleared = Expr::Bin {
            op: BinOp::And,
            lhs: Box::new(Expr::var(dst.clone())),
            rhs: Box::new(Expr::Const(!field)),
            bits: 8,
        };
        let inserted = Expr::Bin {
            op: BinOp::Or,
            lhs: Box::new(cleared),
            rhs: Box::new(Expr::Const(value.wrapping_shl(shift))),
            bits: 8,
        };
        self.core.assign(&dst, inserted, stmts);
    }

    /// `adr`/`adrp`: form an address.
    ///
    /// The decoder resolves the target to an absolute address and prints it
    /// as the second operand, so the value is a constant rather than an
    /// arithmetic expression. `adrp` already carries the page-aligned
    /// address; the low twelve bits come from the instruction that adds to
    /// it, or from the offset in the access that follows.
    fn lift_adr(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let Operand::Imm(target) = &ops[1] else {
            return;
        };
        self.core.write(&ops[0], Expr::Const(*target), stmts);
    }

    fn lift_alu(&mut self, op: BinOp, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 3 {
            return;
        }
        // `add sp, sp, #16` is frame bookkeeping, not a value.
        if let Operand::Reg(dst) = &ops[0]
            && dst.idx == SP
            && let Operand::Reg(src) = &ops[1]
            && src.idx == SP
        {
            if let Operand::Imm(delta) = &ops[2] {
                match op {
                    BinOp::Add => self.core.adjust_sp(*delta as i64),
                    BinOp::Sub => self.core.adjust_sp(-(*delta as i64)),
                    _ => self.core.dynamic_sp = true,
                }
                return;
            }
            self.core.dynamic_sp = true;
            return;
        }
        let lhs = self.core.read(&ops[1], stmts);
        let rhs = self.core.read(&ops[2], stmts);
        let size = ops[0].size();
        let value = Expr::Bin {
            op,
            lhs: Box::new(lhs.clone()),
            rhs: Box::new(rhs.clone()),
            bits: size,
        };
        self.core.write(&ops[0], value, stmts);
    }

    /// `bic`: bitwise clear, `dst = dst & ~src`.
    fn lift_bic(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 3 {
            return;
        }
        let lhs = self.core.read(&ops[1], stmts);
        let rhs = self.core.read(&ops[2], stmts);
        let size = ops[0].size();
        let inverted = Expr::Un {
            op: UnOp::Not,
            arg: Box::new(rhs),
            bits: size,
        };
        let value = Expr::Bin {
            op: BinOp::And,
            lhs: Box::new(lhs),
            rhs: Box::new(inverted),
            bits: size,
        };
        self.core.write(&ops[0], value, stmts);
    }

    /// `orn`: bitwise or with negation, `dst = dst | ~src`.
    fn lift_orn(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 3 {
            return;
        }
        let lhs = self.core.read(&ops[1], stmts);
        let rhs = self.core.read(&ops[2], stmts);
        let size = ops[0].size();
        let inverted = Expr::Un {
            op: UnOp::Not,
            arg: Box::new(rhs),
            bits: size,
        };
        let value = Expr::Bin {
            op: BinOp::Or,
            lhs: Box::new(lhs),
            rhs: Box::new(inverted),
            bits: size,
        };
        self.core.write(&ops[0], value, stmts);
    }

    fn lift_unary(&mut self, op: UnOp, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let value = self.core.read(&ops[1], stmts);
        let size = ops[0].size();
        let value = Expr::Un {
            op,
            arg: Box::new(value),
            bits: size,
        };
        self.core.write(&ops[0], value, stmts);
    }

    /// `mvn`: move the complement of a value.
    fn lift_mvn(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let value = self.core.read(&ops[1], stmts);
        let size = ops[0].size();
        let value = Expr::Un {
            op: UnOp::Not,
            arg: Box::new(value),
            bits: size,
        };
        self.core.write(&ops[0], value, stmts);
    }

    fn lift_shift(&mut self, op: BinOp, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() < 3 {
            return;
        }
        let lhs = self.core.read(&ops[1], stmts);
        let rhs = self.core.read(&ops[2], stmts);
        let size = ops[0].size();
        let value = Expr::Bin {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            bits: size,
        };
        self.core.write(&ops[0], value, stmts);
    }

    /// `ror`: rotate right.
    ///
    /// The IR has no rotate, so it is expressed as a shift and an or.
    fn lift_ror(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() < 3 {
            return;
        }
        let lhs = self.core.read(&ops[1], stmts);
        let rhs = self.core.read(&ops[2], stmts);
        let size = ops[0].size();
        let bits = u32::from(size) * 8;
        let complement = Expr::Bin {
            op: BinOp::Sub,
            lhs: Box::new(Expr::Const(u64::from(bits))),
            rhs: Box::new(rhs.clone()),
            bits: size,
        };
        let high = Expr::Bin {
            op: BinOp::Shl,
            lhs: Box::new(lhs.clone()),
            rhs: Box::new(complement),
            bits: size,
        };
        let low = Expr::Bin {
            op: BinOp::Shr,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            bits: size,
        };
        let value = Expr::Bin {
            op: BinOp::Or,
            lhs: Box::new(high),
            rhs: Box::new(low),
            bits: size,
        };
        self.core.write(&ops[0], value, stmts);
    }

    /// `mul`.
    fn lift_mul(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 3 {
            return;
        }
        let lhs = self.core.read(&ops[1], stmts);
        let rhs = self.core.read(&ops[2], stmts);
        let size = ops[0].size();
        let value = Expr::Bin {
            op: BinOp::Mul,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            bits: size,
        };
        self.core.write(&ops[0], value, stmts);
    }

    /// `madd`/`msub`: multiply and add or subtract an accumulator.
    ///
    /// `madd x0, x1, x2, x3` is `x0 = x3 + x1 * x2`, and `msub` subtracts.
    fn lift_madd(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>, subtract: bool) {
        if ops.len() != 4 {
            return;
        }
        let acc = self.core.read(&ops[1], stmts);
        let lhs = self.core.read(&ops[2], stmts);
        let rhs = self.core.read(&ops[3], stmts);
        let size = ops[0].size();
        let product = Expr::Bin {
            op: BinOp::Mul,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            bits: size,
        };
        let value = Expr::Bin {
            op: if subtract { BinOp::Sub } else { BinOp::Add },
            lhs: Box::new(acc),
            rhs: Box::new(product),
            bits: size,
        };
        self.core.write(&ops[0], value, stmts);
    }

    fn lift_div(&mut self, op: BinOp, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 3 {
            return;
        }
        let lhs = self.core.read(&ops[1], stmts);
        let rhs = self.core.read(&ops[2], stmts);
        let size = ops[0].size();
        let value = Expr::Bin {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            bits: size,
        };
        self.core.write(&ops[0], value, stmts);
    }

    /// `cmp`: compare without keeping the result.
    ///
    /// `cmp` is `subs` with the destination omitted, so it takes two
    /// operands.
    fn lift_cmp(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let lhs = self.core.read(&ops[0], stmts);
        let rhs = self.core.read(&ops[1], stmts);
        self.set_flags(lhs, rhs, ops[0].size());
    }

    /// Record the flags a comparison left behind.
    ///
    /// The width travels with them: `cmp w0, #0` compares 32-bit values, and
    /// `b.lt` reads them as signed.
    fn set_flags(&mut self, lhs: Expr, rhs: Expr, width: u8) {
        // The flags are the ones a subtraction leaves, and the branch that
        // follows can be written either way.
        let cmp = super::PendingCmp::int(lhs, rhs, width);
        self.core.set_flags(cmp.derived_flags(), Some(cmp));
    }

    /// `cmn`: compare with the negation of the second operand.
    ///
    /// `cmn a, b` is `a + ~b`, so the comparison is against the complement.
    /// Writing it as `a != b` would be wrong for the carry conditions, so
    /// the complement is kept.
    fn lift_cmn(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let lhs = self.core.read(&ops[0], stmts);
        let rhs = self.core.read(&ops[1], stmts);
        let size = ops[0].size();
        let inverted = Expr::Un {
            op: UnOp::Not,
            arg: Box::new(rhs),
            bits: size,
        };
        self.set_flags(lhs, inverted, size);
    }

    /// `tst`: test bits without keeping the result.
    fn lift_tst(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let lhs = self.core.read(&ops[0], stmts);
        let rhs = self.core.read(&ops[1], stmts);
        let size = ops[0].size();
        let value = Expr::Bin {
            op: BinOp::And,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            bits: size,
        };
        self.set_flags(value, Expr::Const(0), size);
    }

    /// `csel` and its variants: choose between two values on a condition.
    ///
    /// The three variants differ in what they do to the value they choose
    /// when the condition is false, and the difference is the whole point of
    /// having them: `csinc` adds one, `csinv` complements, and `csneg`
    /// negates. Lifting all four as `csel` printed `wzr` as a plain zero
    /// where the machine produced all ones.
    fn lift_csel(&mut self, mnemonic: &str, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 4 {
            return;
        }
        let Some(cond) = ops[3].cond_suffix().and_then(|s| self.core.target.cmov_cond(s))
        else {
            return;
        };
        let condition = self.core.condition_of("", cond);
        let yes = self.core.read(&ops[1], stmts);
        let mut no = self.core.read(&ops[2], stmts);
        let size = ops[0].size();
        match mnemonic {
            "csinc" => {
                no = Expr::Bin {
                    op: BinOp::Add,
                    lhs: Box::new(no),
                    rhs: Box::new(Expr::Const(1)),
                    bits: size,
                }
            }
            "csinv" => {
                no = Expr::Un {
                    op: UnOp::Not,
                    arg: Box::new(no),
                    bits: size,
                }
            }
            "csneg" => {
                no = Expr::Un {
                    op: UnOp::Neg,
                    arg: Box::new(no),
                    bits: size,
                }
            }
            _ => {}
        }
        self.core.write(
            &ops[0],
            Expr::Select {
                cond: Box::new(condition),
                yes: Box::new(yes),
                no: Box::new(no),
            },
            stmts,
        );
    }

    /// `cset`: produce 1 when the condition holds.
    fn lift_cset(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let Some(cond) = ops[1].cond_suffix().and_then(|s| self.core.target.setcc_cond(s))
        else {
            return;
        };
        let value = self.core.condition_of("", cond);
        let value = Expr::Cast {
            kind: CastKind::Zext,
            arg: Box::new(value),
            from: 1,
            to: ops[0].size(),
        };
        self.core.write(&ops[0], value, stmts);
    }

    /// Sign or zero extension to a wider register.
    fn lift_extend(&mut self, signed: bool, from: u8, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let src = self.core.read(&ops[1], stmts);
        let to = ops[0].size();
        let value = Expr::Cast {
            kind: if signed { CastKind::Sext } else { CastKind::Zext },
            arg: Box::new(src),
            from,
            to,
        };
        self.core.write(&ops[0], value, stmts);
    }

    /// `ldp`/`stp`: load or store a pair of registers.
    ///
    /// The pair is at `[base, #off]`; the write-back form also moves the base
    /// register, which the frame bookkeeping needs.
    fn lift_pair(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>, store: bool) {
        if ops.len() < 3 {
            return;
        }
        let first = &ops[0];
        let second = &ops[1];
        let Operand::Mem {
            base,
            index,
            scale,
            disp,
            base_update,
            ..
        } = &ops[2]
        else {
            return;
        };
        let width = first.size();
        let addr = Expr::Bin {
            op: BinOp::Add,
            lhs: Box::new(match base {
                Some(b) => self.core.read_reg(*b),
                None => Expr::Const(0),
            }),
            rhs: Box::new(Expr::Const(*disp as u64)),
            bits: 8,
        };
        let _ = (index, scale);
        if store {
            let v1 = self.core.read(first, stmts);
            let v2 = self.core.read(second, stmts);
            stmts.push(Stmt::Store {
                addr: addr.clone(),
                value: v1,
                size: width,
            });
            stmts.push(Stmt::Store {
                addr: Expr::Bin {
                    op: BinOp::Add,
                    lhs: Box::new(addr.clone()),
                    rhs: Box::new(Expr::Const(u64::from(width))),
                    bits: 8,
                },
                value: v2,
                size: width,
            });
        } else {
            let v1 = Expr::Load {
                addr: Box::new(addr.clone()),
                size: width,
                signed: false,
            };
            let v2 = Expr::Load {
                addr: Box::new(Expr::Bin {
                    op: BinOp::Add,
                    lhs: Box::new(addr),
                    rhs: Box::new(Expr::Const(u64::from(width))),
                    bits: 8,
                }),
                size: width,
                signed: false,
            };
            self.core.write(first, v1, stmts);
            self.core.write(second, v2, stmts);
        }
        // The write-back moves the base register, and with it the frame.
        let _ = base_update;
        self.apply_writeback(&ops[2], stmts);
    }

    /// A load: the destination takes the value at the address.
    ///
    /// The width comes from the instruction: AArch64 names it in the
    /// register for a plain access and in the mnemonic for the byte, half
    /// and sign-extending forms.
    fn lift_load(&mut self, mnemonic: &str, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let access = self.core.access_of(mnemonic, &ops[1], ops[0].size());
        let value = self.core.read_load(&ops[1], access, stmts);
        self.core.write(&ops[0], value, stmts);
        self.apply_writeback(&ops[1], stmts);
    }

    /// A store: the address receives the value.
    fn lift_store(&mut self, mnemonic: &str, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let access = self.core.access_of(mnemonic, &ops[1], ops[0].size());
        let value = self.core.read(&ops[0], stmts);
        self.core.store(&ops[1], value, access, stmts);
        self.apply_writeback(&ops[1], stmts);
    }

    /// Apply the write-back of a pre-indexed or post-indexed access.
    ///
    /// The base register is part of what the instruction does: `ldr w1,
    /// [x2], #4` walks a pointer, and dropping the walk turns a loop that
    /// advances into one that spins forever on the same address. A stack
    /// pointer base moves the frame instead, because the frame is what the
    /// tool names rather than a value.
    fn apply_writeback(&mut self, op: &Operand, stmts: &mut Vec<Stmt>) {
        let Operand::Mem {
            base,
            base_update: Some(update),
            ..
        } = op
        else {
            return;
        };
        let Some(base) = base else {
            return;
        };
        if base.idx == SP {
            self.core.adjust_sp(*update);
            return;
        }
        let dest = self.core.name(base.idx);
        let old = self.core.read_reg(*base);
        let value = if *update < 0 {
            Expr::Bin {
                op: BinOp::Sub,
                lhs: Box::new(old),
                rhs: Box::new(Expr::Const(update.unsigned_abs())),
                bits: 8,
            }
        } else {
            Expr::Bin {
                op: BinOp::Add,
                lhs: Box::new(old),
                rhs: Box::new(Expr::Const(*update as u64)),
                bits: 8,
            }
        };
        self.core.assign(&dest, value, stmts);
    }

    fn lift_call(&mut self, insn: &Instruction, ops: &[Operand], stmts: &mut Vec<Stmt>) {
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
        let args = self.core.call_args();
        // The callee may write any flag.
        self.core.pending_cmp = None;
        stmts.push(Stmt::Call {
            name,
            args,
            ret: Some(self.core.name(self.core.target.ret_reg())),
            addr: insn.addr,
        });
        self.core.written.clear();
        self.core.written.insert(self.core.target.ret_reg());
        self.core.ever_written.insert(self.core.target.ret_reg());
        // AAPCS64 keeps x19..x29; the rest are the callee's to overwrite.
        let caller_saved: Vec<u8> = (0..=18u8).collect();
        self.core.clobber(&caller_saved, self.core.target.ret_reg());
    }

    /// The condition a block-ending branch tests.
    ///
    /// Three of AArch64's branches do not read the condition flags at all.
    /// `cbz`/`cbnz` test one register against zero and `tbz`/`tbnz` test one
    /// bit of it, and the flags may be arbitrarily stale when they run. They
    /// are also the branches a compiler emits far more often than `b.eq`,
    /// because they need no separate `cmp`: leaving them out dropped whole
    /// arms of a switch into `if (<condition>)`.
    fn branch_condition(&mut self, insn: &Instruction) -> Expr {
        let Some(p) = self.core.target.parse(&insn.text) else {
            return Expr::Unknown("condition".into());
        };
        match p.mnemonic.as_str() {
            "cbz" | "cbnz" => {
                let Some(Operand::Reg(r)) = p.ops.first().cloned() else {
                    return Expr::Unknown("condition".into());
                };
                let lhs = self.core.read_reg(r);
                Expr::Cond {
                    op: if p.mnemonic == "cbz" {
                        CondOp::Eq
                    } else {
                        CondOp::Ne
                    },
                    lhs: Box::new(lhs),
                    rhs: Box::new(Expr::Const(0)),
                    bits: r.size,
                }
            }
            "tbz" | "tbnz" => {
                // `tbz x0, #5, target` branches when bit 5 of x0 is clear.
                let (Some(Operand::Reg(r)), Some(Operand::Imm(bit))) =
                    (p.ops.first().cloned(), p.ops.get(1))
                else {
                    return Expr::Unknown("condition".into());
                };
                let shifted = Expr::Bin {
                    op: BinOp::Shr,
                    lhs: Box::new(self.core.read_reg(r)),
                    rhs: Box::new(Expr::Const(*bit)),
                    bits: r.size,
                };
                let masked = Expr::Bin {
                    op: BinOp::And,
                    lhs: Box::new(shifted),
                    rhs: Box::new(Expr::Const(1)),
                    bits: r.size,
                };
                Expr::Cond {
                    op: if p.mnemonic == "tbz" {
                        CondOp::Eq
                    } else {
                        CondOp::Ne
                    },
                    lhs: Box::new(masked),
                    rhs: Box::new(Expr::Const(0)),
                    bits: r.size,
                }
            }
            _ => self.core.take_condition(&p),
        }
    }

    /// The register a `br` jumps through.
    fn indirect_target(&mut self, insn: &Instruction) -> Expr {
        let Some(p) = self.core.target.parse(&insn.text) else {
            return Expr::Unknown("indirect target".into());
        };
        let mut scratch: Vec<Stmt> = Vec::new();
        match p.ops.first() {
            Some(o) => self.core.read(o, &mut scratch),
            None => Expr::Unknown("indirect target".into()),
        }
    }
}

impl super::Lifter for Lifter<'_> {
    fn lift_block(&mut self, insns: &[Instruction]) -> Lifted {
        let mut out = Lifted::default();
        self.core.written.clear();
        for insn in insns {
            self.lift_insn(insn, &mut out);
        }
        let last = insns.last();
        if let Some(insn) = last {
            match insn.flow {
                FlowKind::ConditionalBranch => out.branch = Some(self.branch_condition(insn)),
                // `br x8` leaves the block through a register. The register
                // is the one thing a reader needs to see: without it the
                // jump reads as "somewhere".
                FlowKind::UnconditionalBranch | FlowKind::IndirectBranch
                    if insn.target.is_none() =>
                {
                    out.indirect = Some(self.indirect_target(insn));
                }
                _ => {}
            }
        }
        out
    }

    fn call_args(&self) -> Vec<Expr> {
        self.core.call_args()
    }

    fn ret_reg(&self) -> String {
        self.core.name(self.core.target.ret_reg())
    }

    fn arg_reg_names(&self) -> Vec<String> {
        self.core
            .target
            .arg_regs()
            .iter()
            .map(|r| self.core.name(*r))
            .collect()
    }
    fn sp_alias(&self) -> &'static str {
        self.core.target.sp_alias()
    }

    fn sp_reg_name(&self) -> String {
        self.core.name(self.core.target.sp_reg())
    }

    fn notes(&mut self) -> Vec<(Addr, String)> {
        std::mem::take(&mut self.core.notes)
    }
}
