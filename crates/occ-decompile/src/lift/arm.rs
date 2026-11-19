//! ARM32 lifting: A32 text to statements.
//!
//! Three things shape this module, and all three are differences from every
//! other architecture the decompiler lifts.
//!
//! * **Every instruction is conditional.** `addeq r0, r1, r2` computes a sum
//!   on one path and does nothing on the other. A value it writes is a
//!   select between the old and the new value; a store it performs is a
//!   statement that only runs sometimes.
//! * **The program counter is a register.** `ldr r0, [pc, #16]` reads a
//!   literal from the code, `mov pc, lr` returns, and `add r0, pc, #8`
//!   computes an address from where the instruction is. The decoder resolves
//!   the address; this module treats the value as the constant it is.
//! * **The flags are a first-class result.** A comparison can be written as
//!   a relation between its operands, and most of the condition codes are
//!   exactly that relation. `ge` is "signed greater or equal", which is
//!   `>=` on the two values and not a question about two separate flags.
//!
//! What is not modelled: VFP and NEON, the DSP extensions, and coprocessor
//! moves. Those arrive as `.word` from the decoder, and the body says so
//! instead of pretending.

use occ_core::{Addr, FlowKind, Instruction};

use super::{
    Core, Lifted, ModKind, Operand, Parsed, PendingCmp, Reg, Target, parse_int, split_operands,
};
use crate::ir::{BinOp, CondOp, Expr, FlagCond, FlagState, Intrinsic, Stmt, UnOp};

/// The stack pointer.
const SP: u8 = 13;
/// The link register.
const LR: u8 = 14;
/// The program counter.
const PC: u8 = 15;

/// Register names by index.
const NAMES: [&str; 16] = [
    "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "sp", "lr",
    "pc",
];

/// A shift amount of 255 in an operand means "the amount is the next
/// register". ARM can shift by a register, and the operand type carries an
/// immediate; the sentinel keeps the register in the operand list where the
/// lifter can see it.
const SHIFT_BY_REGISTER: u8 = 255;

/// The ARM32 target.
pub struct ArmTarget;

impl ArmTarget {
    /// The one target there is.
    pub const fn new() -> Self {
        ArmTarget
    }
}

impl Default for ArmTarget {
    fn default() -> Self {
        ArmTarget
    }
}

impl Target for ArmTarget {
    fn lookup_reg(&self, name: &str) -> Option<Reg> {
        let idx = match name {
            "sp" => SP,
            "lr" => LR,
            "pc" => PC,
            _ => {
                let n: u8 = name.strip_prefix('r')?.parse().ok()?;
                if n > 12 {
                    return None;
                }
                n
            }
        };
        Some(Reg::full(idx, 4))
    }

    fn reg_name(&self, idx: u8) -> String {
        NAMES
            .get(usize::from(idx))
            .copied()
            .unwrap_or("r0")
            .to_string()
    }

    fn reg_width(&self, _idx: u8) -> u8 {
        4
    }

    fn arg_regs(&self) -> &[u8] {
        // AAPCS: r0 through r3, and the return value in r0.
        &[0, 1, 2, 3]
    }

    fn ret_reg(&self) -> u8 {
        0
    }

    fn sp_reg(&self) -> u8 {
        SP
    }

    fn fp_reg(&self) -> Option<u8> {
        // r11 is the frame pointer in the AAPCS frame.
        Some(11)
    }

    fn is_pc(&self, idx: u8) -> bool {
        idx == PC
    }

    fn sp_alias(&self) -> &'static str {
        "frame"
    }

    fn branch_cond(&self, mnemonic: &str) -> Option<FlagCond> {
        let (base, _, cond) = split_mnemonic(mnemonic);
        (base == "b" || base == "bl").then_some(cond).flatten()
    }

    fn setcc_cond(&self, suffix: &str) -> Option<FlagCond> {
        cond_of_suffix(suffix)
    }

    fn cmov_cond(&self, suffix: &str) -> Option<FlagCond> {
        cond_of_suffix(suffix)
    }

    fn parse(&self, text: &str) -> Option<Parsed> {
        parse_arm(text)
    }
}

/// The condition a two-letter suffix names.
///
/// The carry flag is named the other way round from x86, and deliberately so.
/// A32 calls the flag "set" when the comparison did not borrow, and x86
/// calls it "set" when it did; the flag this module records is the borrow,
/// so `cs` — carry set, no borrow, above or equal — is the *negation* of
/// the borrow. Mapping the two names this way makes every path that reads
/// the flag agree: the one that compares two values and the one that reads
/// the flag itself.
fn cond_of_suffix(suffix: &str) -> Option<FlagCond> {
    Some(match suffix {
        "eq" => FlagCond::Z,
        "ne" => FlagCond::Nz,
        "cs" | "hs" => FlagCond::Nc,
        "cc" | "lo" => FlagCond::C,
        "mi" => FlagCond::S,
        "pl" => FlagCond::Ns,
        "vs" => FlagCond::O,
        "vc" => FlagCond::No,
        "hi" => FlagCond::A,
        "ls" => FlagCond::Be,
        "ge" => FlagCond::Nl,
        "lt" => FlagCond::L,
        "gt" => FlagCond::G,
        "le" => FlagCond::Le,
        _ => return None,
    })
}

/// The instructions this lifter understands, without condition or `s`.
///
/// The instruction set the lifter knows.
///
/// The list holds both instruction sets of the architecture. A32 and T32
/// are two encodings of the same machine, and a compiler picks between
/// them per function, so a binary holds both and the lifter has to read
/// both. The names that are Thumb's own — `cbz`, the extends, the
/// conditional block — are in the list too.
const BASES: &[&str] = &[
    "adc", "add", "adr", "and", "asr", "b", "bfc", "bfi", "bic", "bkpt", "bl", "blx", "blxns",
    "bx", "bxns", "cbz", "cbnz", "clz", "cmn", "cmp", "crc32b", "crc32cb", "crc32ch", "crc32cw",
    "crc32h", "crc32w", "eor", "ldm", "ldmdb", "ldmia", "ldr", "ldrb", "ldrd", "ldrex", "ldrexb",
    "ldrexd", "ldrexh", "ldrh", "ldrsb", "ldrsh", "lsl", "lsr", "mla", "mls", "mov", "movw",
    "movt", "mul", "mvn", "nop", "orr", "pop", "push", "qadd", "qdadd", "qdsub", "qsub", "rbit",
    "rev", "rev16", "revsh", "ror", "rrx", "rsb", "sbc", "sbfx", "sdiv", "sel", "serv", "smull",
    "smlal", "stm", "stmdb", "stmia", "str", "strb", "strd", "strex", "strexb", "strexd", "strexh",
    "strh", "sub", "svc", "swi", "sxtab", "sxtab16", "sxtah", "sxtb", "sxth", "teq", "tst", "ubfx",
    "udiv", "umaal", "umlal", "umull", "uxtab", "uxtab16", "uxtah", "uxtb", "uxth", "wfe", "wfi",
    "yield",
];

/// The parallel and the exchanged operations, and the intrinsic each one
/// is.
///
/// The name says how the halves are read and how the result is treated:
/// the `s` prefix reads them as signed numbers and the `u` prefix as
/// unsigned ones, the `q` prefix holds the result at the ends of the
/// range and the `sh` and `uh` prefixes halve it. The family says what
/// is done to them, and the ending says how wide they are.
///
/// The exchanged ones take the halves of the two operands the other way
/// round, which is what the `asx` and `sax` families say.
fn parallel_intrinsic(base: &str) -> Option<(Intrinsic, u8)> {
    use Intrinsic::{
        ExchangeAdd, ExchangeAddHalf, ExchangeAddHalfUns, ExchangeAddSat, ExchangeAddSatUns,
        ExchangeAddUns, ExchangeSub, ExchangeSubHalf, ExchangeSubHalfUns, ExchangeSubSat,
        ExchangeSubSatUns, ExchangeSubUns, ParallelAdd, ParallelAddHalf, ParallelAddHalfUns,
        ParallelAddSat, ParallelAddSatUns, ParallelAddUns, ParallelSub, ParallelSubHalf,
        ParallelSubHalfUns, ParallelSubSat, ParallelSubSatUns, ParallelSubUns,
    };
    // The exchanged families come first, because their names end in the
    // same letters as the parallel ones.
    if let Some(rest) = base.strip_suffix("asx") {
        let lane = 2;
        return Some(match rest {
            "s" => (ExchangeAdd, lane),
            "q" => (ExchangeAddSat, lane),
            "sh" => (ExchangeAddHalf, lane),
            "u" => (ExchangeAddUns, lane),
            "uq" => (ExchangeAddSatUns, lane),
            "uh" => (ExchangeAddHalfUns, lane),
            _ => return None,
        });
    }
    if let Some(rest) = base.strip_suffix("sax") {
        let lane = 2;
        return Some(match rest {
            "s" => (ExchangeSub, lane),
            "q" => (ExchangeSubSat, lane),
            "sh" => (ExchangeSubHalf, lane),
            "u" => (ExchangeSubUns, lane),
            "uq" => (ExchangeSubSatUns, lane),
            "uh" => (ExchangeSubHalfUns, lane),
            _ => return None,
        });
    }
    // The parallel ones say how wide the lane is at the end of the name.
    let (body, lane) = match base.as_bytes().last() {
        Some(b'8') => (&base[..base.len() - 1], 1),
        Some(b'6') if base.ends_with("16") => (&base[..base.len() - 2], 2),
        _ => return None,
    };
    Some(match body {
        "sadd" => (ParallelAdd, lane),
        "qadd" => (ParallelAddSat, lane),
        "shadd" => (ParallelAddHalf, lane),
        "uadd" => (ParallelAddUns, lane),
        "uqadd" => (ParallelAddSatUns, lane),
        "uhadd" => (ParallelAddHalfUns, lane),
        "ssub" => (ParallelSub, lane),
        "qsub" => (ParallelSubSat, lane),
        "shsub" => (ParallelSubHalf, lane),
        "usub" => (ParallelSubUns, lane),
        "uqsub" => (ParallelSubSatUns, lane),
        "uhsub" => (ParallelSubHalfUns, lane),
        _ => return None,
    })
}

/// Split a mnemonic into its base, whether it sets the flags, and its
/// condition.
///
/// The pieces are not separable by string surgery alone: `bics` ends in `cs`,
/// which is a condition, and `bls` is `b` with `ls` and not `bl` with `s`.
/// Every candidate is therefore checked against the instruction set before it
/// is accepted.
///
/// The wide forms carry a marker that says how the instruction is encoded
/// and not what it does, and it goes at the end: `add.w` is an add, and
/// `beq.w` is a conditional branch whose condition comes before the
/// marker. The marker is taken off first, so the condition is found where
/// it is.
fn split_mnemonic(m: &str) -> (&str, bool, Option<FlagCond>) {
    let m = m.strip_suffix(".w").unwrap_or(m);
    let known = |s: &str| BASES.contains(&s);
    if let Some((base, sfx)) = m.split_at_checked(m.len().saturating_sub(2))
        && !base.is_empty()
    {
        if let Some(fc) = cond_of_suffix(sfx) {
            if known(base) {
                return (base, false, Some(fc));
            }
            if let Some(stem) = base.strip_suffix('s')
                && known(stem)
            {
                return (stem, true, Some(fc));
            }
        }
    }
    if known(m) {
        return (m, false, None);
    }
    if let Some(stem) = m.strip_suffix('s')
        && known(stem)
    {
        return (stem, true, None);
    }
    (m, false, None)
}

/// Parse an ARM instruction's text.
fn parse_arm(text: &str) -> Option<Parsed> {
    let text = text.trim();
    let (mnemonic, rest) = match text.split_once(char::is_whitespace) {
        Some((m, r)) => (m, r.trim()),
        None => (text, ""),
    };
    let mut ops: Vec<Operand> = Vec::new();
    if !rest.is_empty() {
        for part in split_operands(rest) {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if part == "rrx" {
                // `rrx` is a one-bit rotate through the carry, and the
                // amount is not in the instruction: the encoding spends
                // the zero that `ror #0` would have used on it.
                let prev = ops.pop()?;
                let Operand::Reg(reg) = prev else {
                    return None;
                };
                ops.push(Operand::Modified {
                    reg,
                    kind: ModKind::Ror,
                    amount: 0,
                });
                continue;
            }
            if let Some((kind, tail)) = shift_prefix(part) {
                // `r1, lsl #2` — the shift belongs to the operand before it.
                let prev = ops.pop()?;
                let Operand::Reg(reg) = prev else {
                    return None;
                };
                let tail = tail.trim();
                if let Some(amount) = tail.strip_prefix('#') {
                    ops.push(Operand::Modified {
                        reg,
                        kind,
                        amount: parse_int(amount)? as u8,
                    });
                } else {
                    ops.push(Operand::Modified {
                        reg,
                        kind,
                        amount: SHIFT_BY_REGISTER,
                    });
                    ops.push(Operand::Reg(register(tail)?));
                }
                continue;
            }
            ops.push(parse_operand(part)?);
        }
    }
    Some(Parsed {
        mnemonic: mnemonic.to_ascii_lowercase(),
        ops,
        pc_target: None,
        prefixes: Vec::new(),
    })
}

/// A shift keyword and what follows it.
fn shift_prefix(s: &str) -> Option<(ModKind, &str)> {
    for (word, kind) in [
        ("lsl", ModKind::Lsl),
        ("lsr", ModKind::Lsr),
        ("asr", ModKind::Asr),
        ("ror", ModKind::Ror),
        ("rrx", ModKind::Ror),
    ] {
        if let Some(rest) = s.strip_prefix(word) {
            if rest.is_empty() || rest.starts_with(char::is_whitespace) {
                return Some((kind, rest));
            }
        }
    }
    None
}

fn register(name: &str) -> Option<Reg> {
    let idx = match name.trim() {
        "sp" => SP,
        "lr" => LR,
        "pc" => PC,
        _ => {
            let n: u8 = name.trim().strip_prefix('r')?.parse().ok()?;
            if n > 12 {
                return None;
            }
            n
        }
    };
    Some(Reg::full(idx, 4))
}

fn parse_operand(s: &str) -> Option<Operand> {
    if let Some(reg) = register(s) {
        return Some(Operand::Reg(reg));
    }
    if let Some(rest) = s.strip_prefix('#') {
        return parse_int(rest).map(Operand::Imm);
    }
    if s.starts_with('[') {
        return parse_mem(s, 4);
    }
    // A register list belongs to the block-transfer path, not to a single
    // operand; the list is read from the text there.
    if s.starts_with('{') {
        return None;
    }
    parse_int(s).map(Operand::Imm)
}

/// Parse `[rn]`, `[rn, #off]`, `[rn, rm]`, `[rn, rm, lsl #2]`, with an
/// optional trailing `!`.
fn parse_mem(s: &str, size: u8) -> Option<Operand> {
    let (inner, write_back) = match s.strip_suffix('!') {
        Some(r) => (r, true),
        None => (s, false),
    };
    let inner = inner.strip_prefix('[')?.strip_suffix(']')?;
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    if parts.is_empty() {
        return None;
    }
    let base = register(parts[0])?;
    let mut index = None;
    let mut scale = 1;
    let mut disp = 0i64;
    if let Some(second) = parts.get(1) {
        if let Some(off) = second.strip_prefix('#') {
            disp = parse_int(off)? as i64;
        } else {
            index = Some(register(second)?);
        }
    }
    if let Some(third) = parts.get(2) {
        let third = third.trim();
        let (kind, tail) = shift_prefix(third)?;
        if kind != ModKind::Lsl {
            // A scaled index is an array access; only the left shift scales.
            return None;
        }
        let amount = parse_int(tail.trim().strip_prefix('#')?)?;
        scale = 1u64 << amount;
    }
    Some(Operand::Mem {
        size,
        base: Some(base),
        index,
        scale,
        disp,
        abs: None,
        seg: None,
        base_update: write_back.then_some(disp),
    })
}

/// The ARM32 lifter.
pub struct Lifter<'a> {
    /// Shared state: registers, flags, the frame.
    core: Core<'a>,
    /// Names handed out to comparison operands, so a condition that outlives
    /// the register it was built from still names the value it compared.
    cmp_seq: usize,
    /// Whether the function has saved the link register.
    ///
    /// A `push` that saves it is the prologue, and the `pop` that undoes it
    /// is the epilogue; between them the frame is the function's own.
    saved_link: bool,
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
            cmp_seq: 0,
            saved_link: false,
        }
    }

    /// Write a register, honouring a condition.
    ///
    /// A conditional write leaves the register alone when the condition does
    /// not hold, which is a select between the new value and the old one.
    /// Writing the new value unconditionally would make `moveq r0, #1`
    /// assign on both paths.
    fn write_reg(&mut self, idx: u8, value: Expr, guard: Option<&Expr>, stmts: &mut Vec<Stmt>) {
        // The block-level set is what a call site reads to decide which
        // argument registers this block has defined, and the function-level
        // set is what the return type is inferred from. Neither is updated
        // by `assign`, which only appends the statement.
        self.core.written.insert(idx);
        self.core.ever_written.insert(idx);
        // A register the code writes again holds what was written, not
        // what the call before it left in it.
        self.core.clobbered.remove(&idx);
        let name = self.core.name(idx);
        let value = match guard {
            Some(cond) => Expr::Select {
                cond: Box::new(cond.clone()),
                yes: Box::new(value),
                no: Box::new(self.core.read_reg(Reg::full(idx, 4))),
            },
            None => value,
        };
        self.core.assign(&name, value, stmts);
    }

    /// Emit a store, honouring a condition.
    fn store(
        &mut self,
        addr: Expr,
        value: Expr,
        size: u8,
        guard: Option<&Expr>,
        stmts: &mut Vec<Stmt>,
    ) {
        let st = Stmt::Store { addr, value, size };
        match guard {
            Some(cond) => stmts.push(Stmt::If {
                cond: cond.clone(),
                then_body: vec![st],
                else_body: Vec::new(),
            }),
            None => stmts.push(st),
        }
    }

    /// Read an operand as a value.
    fn read(&mut self, op: &Operand) -> Expr {
        match op {
            Operand::Modified {
                reg,
                kind: _,
                amount: SHIFT_BY_REGISTER,
            } => {
                // The amount is the register that follows; the caller put it
                // in the operand list. Without it there is nothing to shift
                // by, so the value is the register itself.
                self.core.read_reg(*reg)
            }
            Operand::Modified { reg, kind, amount } if kind.is_shift() => {
                self.shifted(self.core.read_reg(*reg), *kind, u32::from(*amount))
            }
            Operand::Modified { reg, kind, amount } => {
                self.core.read_modified(*reg, *kind, *amount)
            }
            Operand::Reg(r) => self.core.read_reg(*r),
            Operand::Imm(v) => Expr::Const(*v),
            other => self.core.read(other, &mut Vec::new()),
        }
    }

    /// Apply a shift to a value, the way A32 defines it.
    ///
    /// Two of the encodings do not mean what they look like. `ror #0` is not
    /// a rotate by zero: the encoding has no room for a thirty-second
    /// amount, and it spends the zero on the one-bit rotate through the
    /// carry that the assembler spells `rrx`. And a shift by the width of
    /// the value is not a shift at all in C — `lsr #32` means the value is
    /// gone, and `asr #32` means it is all sign.
    fn shifted(&mut self, value: Expr, kind: ModKind, amount: u32) -> Expr {
        if kind == ModKind::Ror && amount == 0 {
            // The carry comes in at the top and the value leaves by the
            // bottom, one position at a time.
            return Expr::Intrinsic {
                op: Intrinsic::Rcr,
                args: vec![value, Expr::Const(1), self.carry_in()],
                bits: 4,
            };
        }
        match kind {
            ModKind::Lsl | ModKind::Lsr if amount >= 32 => Expr::Const(0),
            ModKind::Asr if amount >= 32 => Expr::Bin {
                op: BinOp::Sar,
                lhs: Box::new(value),
                rhs: Box::new(Expr::Const(31)),
                bits: 4,
            },
            ModKind::Ror if amount >= 32 => value,
            ModKind::Ror => Expr::Intrinsic {
                op: Intrinsic::Ror,
                args: vec![value, Expr::Const(u64::from(amount))],
                bits: 4,
            },
            _ => Expr::Bin {
                op: match kind {
                    ModKind::Lsl => BinOp::Shl,
                    ModKind::Lsr => BinOp::Shr,
                    ModKind::Asr => BinOp::Sar,
                    _ => BinOp::Shr,
                },
                lhs: Box::new(value),
                rhs: Box::new(Expr::Const(u64::from(amount))),
                bits: 4,
            },
        }
    }

    /// A shift whose amount is a register.
    fn read_shifted(&mut self, value: Expr, kind: ModKind, amount: Expr) -> Expr {
        if kind == ModKind::Ror {
            // The IR has a rotate, so a rotate by a register is one.
            return Expr::Intrinsic {
                op: Intrinsic::Ror,
                args: vec![value, amount],
                bits: 4,
            };
        }
        Expr::Bin {
            op: match kind {
                ModKind::Lsl => BinOp::Shl,
                ModKind::Lsr => BinOp::Shr,
                ModKind::Asr => BinOp::Sar,
                _ => BinOp::Shr,
            },
            lhs: Box::new(value),
            rhs: Box::new(amount),
            bits: 4,
        }
    }

    /// The carry flag as a value, for the instructions that add it in.
    ///
    /// An instruction that arrives before anything set the carry has no
    /// value to add. Saying so is better than adding zero: the reader can
    /// see the gap, and the interpreter refuses to invent one.
    fn carry_in(&self) -> Expr {
        match &self.core.flags.cf {
            Some(e) => e.clone(),
            None => Expr::Unknown("the carry is not defined on this path".into()),
        }
    }

    fn lift_insn(&mut self, insn: &Instruction, out: &mut Lifted) {
        // The head of an `it` block governs the instructions after it, and
        // the condition is already written into their own names. The block
        // itself is not an operation on any value: it is the way Thumb
        // says that the next few instructions run on one path.
        if is_it_block(&insn.text) {
            return;
        }
        // The block transfers carry a register list, which is not an operand
        // and would make the general parser reject the instruction.
        if self.block_transfer(insn.text.as_str(), &mut out.stmts) {
            return;
        }
        let Some(p) = parse_arm(&insn.text) else {
            out.stmts
                .push(Stmt::Comment(format!("{} ; unparsed", insn.text)));
            return;
        };
        let m = p.mnemonic.as_str();
        let (base, sets_flags, cond) = split_mnemonic(m);
        let guard = cond.map(|c| self.core.condition_of(m, c));
        let guard = guard.as_ref();
        let ops = &p.ops;
        match base {
            "movt" => {
                // `movt` writes the top half and leaves the bottom alone.
                let Some(rd) = reg_at(ops, 0) else { return };
                let Some(upper) = ops.get(1).and_then(|o| match o {
                    Operand::Imm(v) => Some(*v),
                    _ => None,
                }) else {
                    return;
                };
                let low = Expr::Bin {
                    op: BinOp::And,
                    lhs: Box::new(self.core.read_reg(Reg::full(rd, 4))),
                    rhs: Box::new(Expr::Const(0xffff)),
                    bits: 4,
                };
                let value = Expr::Bin {
                    op: BinOp::Or,
                    lhs: Box::new(low),
                    rhs: Box::new(Expr::Const((upper & 0xffff) << 16)),
                    bits: 4,
                };
                self.write_reg(rd, value, guard, &mut out.stmts);
            }
            "mov" | "movw" => {
                let Some(rd) = reg_at(ops, 0) else { return };
                let Some(op) = ops.get(1).cloned() else {
                    return;
                };
                // A move that carries a shift says which bit fell off the
                // end, and that bit is read from the value before the
                // shift — the write is about to overwrite it.
                let shift = match &op {
                    Operand::Modified { reg, kind, amount }
                        if kind.is_shift() && *amount != SHIFT_BY_REGISTER =>
                    {
                        Some((reg.idx, *kind, u32::from(*amount)))
                    }
                    _ => None,
                };
                let previous = shift.map(|(reg, _, _)| {
                    // The carry of a shift is the bit that fell off the end,
                    // and it is read from the value *before* the shift. The
                    // name is bound to a value of its own so that the flag
                    // does not end up reading the shifted value: the write
                    // below is about to replace it.
                    let value = self.core.read_reg(Reg::full(reg, 4));
                    self.bind(value, &mut out.stmts)
                });
                let value = self.operand_value(&op, ops, 1);
                self.write_reg(rd, value.clone(), guard, &mut out.stmts);
                if sets_flags {
                    match (shift, previous) {
                        (Some((_, kind, amount)), Some(previous)) => {
                            self.set_shift_flags(kind, amount, &previous, &value)
                        }
                        _ => self.set_result_flags(rd, &mut out.stmts),
                    }
                }
            }
            "mvn" => {
                let Some(rd) = reg_at(ops, 0) else { return };
                let Some(value) = ops.get(1).map(|o| self.operand_value(o, ops, 1)) else {
                    return;
                };
                let value = Expr::Un {
                    op: UnOp::Not,
                    arg: Box::new(value),
                    bits: 4,
                };
                self.write_reg(rd, value, guard, &mut out.stmts);
            }
            "add" | "adc" | "sub" | "sbc" | "rsb" | "and" | "orr" | "eor" | "bic" => {
                let op = match base {
                    "add" | "adc" => BinOp::Add,
                    "sub" | "sbc" => BinOp::Sub,
                    // `rsb rd, rn, #x` is `x - rn`: the reverse subtraction
                    // has its operands the other way round.
                    "rsb" => BinOp::Sub,
                    "and" => BinOp::And,
                    "orr" => BinOp::Or,
                    "eor" => BinOp::Xor,
                    // `bic rd, rn, rm` is `rn & ~rm`. The IR has no fused
                    // and-not, and writing it out is what the instruction
                    // means.
                    _ => BinOp::And,
                };
                let Some(rd) = reg_at(ops, 0) else { return };
                let (lhs, rhs) = if ops.len() >= 3 {
                    (
                        self.operand_value(&ops[1], ops, 1),
                        self.operand_value(&ops[2], ops, 2),
                    )
                } else if ops.len() == 2 {
                    // Two-operand form: the destination is also the source.
                    (
                        self.core.read_reg(Reg::full(rd, 4)),
                        self.operand_value(&ops[1], ops, 1),
                    )
                } else {
                    return;
                };
                let (lhs, rhs) = if base == "rsb" {
                    (rhs, lhs)
                } else {
                    (lhs, rhs)
                };
                let rhs = if base == "bic" {
                    Expr::Un {
                        op: UnOp::Not,
                        arg: Box::new(rhs),
                        bits: 4,
                    }
                } else {
                    rhs
                };
                // `adc` and `sbc` arrive with a carry, and the carry is a
                // value rather than a mode: a loop that adds two multi-word
                // numbers is wrong at every word after the first if the
                // carry is dropped.
                let carry = if base == "adc" || base == "sbc" {
                    Some(self.carry_in())
                } else {
                    None
                };
                let value = match &carry {
                    Some(c) => Expr::Intrinsic {
                        op: if base == "adc" {
                            Intrinsic::AdcSum
                        } else {
                            Intrinsic::SbbDiff
                        },
                        args: vec![lhs.clone(), rhs.clone(), c.clone()],
                        bits: 4,
                    },
                    None => Expr::Bin {
                        op,
                        lhs: Box::new(lhs.clone()),
                        rhs: Box::new(rhs.clone()),
                        bits: 4,
                    },
                };
                self.write_reg(rd, value.clone(), guard, &mut out.stmts);
                if sets_flags {
                    self.set_arith_flags(base, &lhs, &rhs, &value, carry.as_ref());
                }
            }
            "lsl" | "lsr" | "asr" | "ror" | "rrx" => {
                // A shift on its own: the value moves and, with the `s`,
                // the flags say which bit fell off the end.
                let Some(rd) = reg_at(ops, 0) else { return };
                let Some(rm) = ops.get(1).and_then(reg_of) else {
                    return;
                };
                let kind = match base {
                    "lsl" => ModKind::Lsl,
                    "lsr" => ModKind::Lsr,
                    "asr" => ModKind::Asr,
                    _ => ModKind::Ror,
                };
                // The value before the shift is what the carry is read
                // from, and the write is about to overwrite it.
                let previous = self.bind(self.core.read_reg(rm), &mut out.stmts);
                let value = match ops.get(2) {
                    Some(Operand::Imm(v)) => {
                        self.shifted(previous.clone(), kind, u32::try_from(*v).unwrap_or(0))
                    }
                    Some(other) => {
                        // The amount is in a register, so the carry out of
                        // the shift is not a fixed bit and is not modelled.
                        let amount = self.core.read(other, &mut out.stmts);
                        self.read_shifted(previous.clone(), kind, amount)
                    }
                    None => self.shifted(previous.clone(), kind, 0),
                };
                self.write_reg(rd, value.clone(), guard, &mut out.stmts);
                if sets_flags {
                    if let Some(Operand::Imm(v)) = ops.get(2) {
                        self.set_shift_flags(
                            kind,
                            u32::try_from(*v).unwrap_or(0),
                            &previous,
                            &value,
                        );
                    }
                }
            }
            "cmp" | "cmn" | "tst" | "teq" => {
                let Some(lhs) = ops.first().map(|o| self.operand_value(o, ops, 0)) else {
                    return;
                };
                let Some(rhs) = ops.get(1).map(|o| self.operand_value(o, ops, 1)) else {
                    return;
                };
                // The flags belong to the comparison, not to the registers
                // it read. A later instruction may overwrite one of them —
                // `cmp r1, #10; mov r1, #4; movlt r1, #3` does exactly
                // that — and a condition written against the register name
                // would then compare the new value. Naming the operands here
                // pins them to what the comparison saw.
                let lhs = self.snapshot(lhs, &mut out.stmts);
                let rhs = self.snapshot(rhs, &mut out.stmts);
                let (lhs, rhs) = match base {
                    "tst" | "teq" => (
                        Expr::Bin {
                            op: if base == "tst" {
                                BinOp::And
                            } else {
                                BinOp::Xor
                            },
                            lhs: Box::new(lhs),
                            rhs: Box::new(rhs),
                            bits: 4,
                        },
                        Expr::Const(0),
                    ),
                    // `cmn` is a comparison against the negation.
                    "cmn" => (
                        lhs,
                        Expr::Un {
                            op: UnOp::Neg,
                            arg: Box::new(rhs),
                            bits: 4,
                        },
                    ),
                    _ => (lhs, rhs),
                };
                let cmp = PendingCmp::int(lhs, rhs, 4);
                self.core.set_flags(cmp.derived_flags(), Some(cmp));
            }
            "mul" | "mla" => {
                let Some(rd) = reg_at(ops, 0) else { return };
                let Some(rm) = ops.get(1).and_then(reg_of) else {
                    return;
                };
                let Some(rs) = ops.get(2).and_then(reg_of) else {
                    return;
                };
                let product = Expr::Bin {
                    op: BinOp::Mul,
                    lhs: Box::new(self.core.read_reg(rm)),
                    rhs: Box::new(self.core.read_reg(rs)),
                    bits: 4,
                };
                let value = if base == "mla" {
                    let Some(ra) = ops.get(3).and_then(reg_of) else {
                        return;
                    };
                    Expr::Bin {
                        op: BinOp::Add,
                        lhs: Box::new(product),
                        rhs: Box::new(self.core.read_reg(ra)),
                        bits: 4,
                    }
                } else {
                    product
                };
                self.write_reg(rd, value, guard, &mut out.stmts);
                if sets_flags {
                    self.set_result_flags(rd, &mut out.stmts);
                }
            }
            "umaal" => {
                // An unsigned product added to a third register: the
                // accumulator the encoding keeps in the second field.
                let Some(rd) = reg_at(ops, 0) else { return };
                let (Some(rm), Some(rs), Some(rn)) = (
                    ops.get(1).and_then(reg_of),
                    ops.get(2).and_then(reg_of),
                    ops.get(3).and_then(reg_of),
                ) else {
                    return;
                };
                let value = Expr::Bin {
                    op: BinOp::Add,
                    lhs: Box::new(Expr::Bin {
                        op: BinOp::Mul,
                        lhs: Box::new(self.core.read_reg(rm)),
                        rhs: Box::new(self.core.read_reg(rs)),
                        bits: 4,
                    }),
                    rhs: Box::new(self.core.read_reg(rn)),
                    bits: 4,
                };
                self.write_reg(rd, value, guard, &mut out.stmts);
            }
            "smulbb" | "smultb" | "smulwb" | "smulwt" | "smulbt" | "smultt" => {
                // A multiply of one half of each operand. Which half is in
                // the name: `bb` is the bottom half of both.
                let Some(rd) = reg_at(ops, 0) else { return };
                let (Some(rm), Some(rs)) =
                    (ops.get(1).and_then(reg_of), ops.get(2).and_then(reg_of))
                else {
                    return;
                };
                let half = |value: Expr, top: bool| {
                    let narrow = if top {
                        Expr::Bin {
                            op: BinOp::Shr,
                            lhs: Box::new(value),
                            rhs: Box::new(Expr::Const(16)),
                            bits: 4,
                        }
                    } else {
                        value
                    };
                    Expr::Cast {
                        kind: crate::ir::CastKind::Trunc,
                        arg: Box::new(narrow),
                        from: 4,
                        to: 2,
                    }
                };
                // The two letters say which half of which operand: the
                // first is the first operand's and the second is the
                // second's, and `w` means the top word just as `t` does.
                let kind = &base[4..];
                let first_top = kind.starts_with('t');
                let second_top = kind.ends_with('t') || kind.ends_with('w');
                let value = Expr::Bin {
                    op: BinOp::Mul,
                    lhs: Box::new(half(self.core.read_reg(rm), first_top)),
                    rhs: Box::new(half(self.core.read_reg(rs), second_top)),
                    bits: 2,
                };
                self.write_reg(rd, value, guard, &mut out.stmts);
            }
            "umull" | "smull" | "umlal" | "smlal" => {
                // A multiply that produces twice as many bits as it was
                // given. The two registers hold the two halves of one
                // number, and the low half on its own is not the product:
                // `1 << 31` has a low half of zero.

                let (Some(lo_reg), Some(hi_reg)) =
                    (ops.first().and_then(reg_of), ops.get(1).and_then(reg_of))
                else {
                    return;
                };
                let (Some(rm), Some(rs)) =
                    (ops.get(2).and_then(reg_of), ops.get(3).and_then(reg_of))
                else {
                    return;
                };
                let signed = base.starts_with('s');
                // The product is one number twice as wide as its factors,
                // and the two registers are the two halves of it. It gets a
                // name of its own: the halves are the same product, and
                // without the name the high half would print the whole
                // product a second time.
                let first = self.bind(self.core.read_reg(rm), &mut out.stmts);
                let second = self.bind(self.core.read_reg(rs), &mut out.stmts);
                let widen = |value: Expr| Expr::Cast {
                    kind: if signed {
                        crate::ir::CastKind::Sext
                    } else {
                        crate::ir::CastKind::Zext
                    },
                    arg: Box::new(value),
                    from: 4,
                    to: 8,
                };
                let product = Expr::Bin {
                    op: BinOp::Mul,
                    lhs: Box::new(widen(first)),
                    rhs: Box::new(widen(second)),
                    bits: 8,
                };
                self.cmp_seq += 1;
                let temp = format!("p{}", self.cmp_seq);
                out.stmts.push(Stmt::Assign {
                    dest: temp.clone(),
                    value: product,
                });
                let whole = Expr::Var(temp);
                let product_lo = Expr::Cast {
                    kind: crate::ir::CastKind::Trunc,
                    arg: Box::new(whole.clone()),
                    from: 8,
                    to: 4,
                };
                let product_hi = Expr::Cast {
                    kind: crate::ir::CastKind::Trunc,
                    arg: Box::new(Expr::Bin {
                        op: BinOp::Shr,
                        lhs: Box::new(whole),
                        rhs: Box::new(Expr::Const(32)),
                        bits: 8,
                    }),
                    from: 8,
                    to: 4,
                };
                let (lo, hi) = if base.ends_with("lal") {
                    // The accumulate forms add the product to a pair that
                    // already holds a number, which is a sixty-four bit
                    // addition split across two registers.
                    let old_lo = self.core.read_reg(lo_reg);
                    let old_hi = self.core.read_reg(hi_reg);
                    let sum = Expr::Bin {
                        op: BinOp::Add,
                        lhs: Box::new(product_lo),
                        rhs: Box::new(old_lo.clone()),
                        bits: 4,
                    };
                    // The low half carries into the high one when the sum
                    // did not fit.
                    let carry = Expr::Cond {
                        op: CondOp::Ult,
                        lhs: Box::new(sum.clone()),
                        rhs: Box::new(old_lo),
                        bits: 4,
                    };
                    let high = Expr::Bin {
                        op: BinOp::Add,
                        lhs: Box::new(Expr::Bin {
                            op: BinOp::Add,
                            lhs: Box::new(product_hi),
                            rhs: Box::new(old_hi),
                            bits: 4,
                        }),
                        rhs: Box::new(carry),
                        bits: 4,
                    };
                    (sum, high)
                } else {
                    (product_lo, product_hi)
                };
                self.write_reg(lo_reg.idx, lo, guard, &mut out.stmts);
                self.write_reg(hi_reg.idx, hi, guard, &mut out.stmts);
            }
            "clz" => {
                let Some(rd) = reg_at(ops, 0) else { return };
                let Some(rm) = ops.get(1).and_then(reg_of) else {
                    return;
                };
                let value = Expr::Intrinsic {
                    op: crate::ir::Intrinsic::Lzcnt,
                    args: vec![self.core.read_reg(rm)],
                    bits: 4,
                };
                self.write_reg(rd, value, guard, &mut out.stmts);
            }
            "uxtb" | "uxth" | "sxtb" | "sxth" => {
                // The extension is part of the instruction: a byte that is
                // loaded and zero-extended is one operation, and reading it
                // as a plain move loses the half of the value the
                // instruction threw away.
                let Some(rd) = reg_at(ops, 0) else { return };
                let Some(rm) = ops.get(1).and_then(reg_of) else {
                    return;
                };
                let signed = base == "sxtb" || base == "sxth";
                let from = if base == "uxtb" || base == "sxtb" {
                    1
                } else {
                    2
                };
                let value = Expr::Cast {
                    kind: if signed {
                        crate::ir::CastKind::Sext
                    } else {
                        crate::ir::CastKind::Zext
                    },
                    arg: Box::new(self.core.read_reg(rm)),
                    from,
                    to: 4,
                };
                self.write_reg(rd, value, guard, &mut out.stmts);
            }
            "sdiv" | "udiv" => {
                let Some(rd) = reg_at(ops, 0) else { return };
                let (Some(rn), Some(rm)) =
                    (ops.get(1).and_then(reg_of), ops.get(2).and_then(reg_of))
                else {
                    return;
                };
                let value = Expr::Bin {
                    op: if base == "sdiv" {
                        BinOp::Div
                    } else {
                        BinOp::UDiv
                    },
                    lhs: Box::new(self.core.read_reg(rn)),
                    rhs: Box::new(self.core.read_reg(rm)),
                    bits: 4,
                };
                self.write_reg(rd, value, guard, &mut out.stmts);
            }
            "ldrd" | "strd" => {
                self.double_transfer(base, ops, guard, &mut out.stmts);
            }
            "ldr" | "ldrb" | "ldrh" | "ldrsb" | "ldrsh" | "str" | "strb" | "strh" => {
                self.transfer(base, ops, insn.addr, guard, &mut out.stmts);
            }
            "bx" | "bxns" => {
                // `bx lr` is a return, and the decoder already said so in the
                // flow. `bx r3` is a computed jump the block terminator
                // carries; there is no statement to emit.  `bxns` is the
                // same exchange with the security state changed as well,
                // and it is a jump in the same way.
            }
            "cbz" | "cbnz" => {
                // Thumb's compare-and-branch. The condition is the value of
                // the register against zero rather than a condition code,
                // and the block terminator carries it.
            }
            "blx" | "blxns" => {
                let Some(target) = ops.first().and_then(reg_of) else {
                    return;
                };
                // The callee is a value the function computed, and the only
                // name it has is the register it is in.
                let name = self.core.name(target.idx);
                // The register the target is in is not one of the arguments.
                let args = self.core.call_args_excluding(target.idx);
                let ret = self.core.name(0);
                let st = Stmt::Call {
                    name,
                    args,
                    ret: Some(ret),
                    addr: insn.addr,
                    target: None,
                };
                match guard {
                    Some(cond) => out.stmts.push(Stmt::If {
                        cond: cond.clone(),
                        then_body: vec![st],
                        else_body: Vec::new(),
                    }),
                    None => out.stmts.push(st),
                }
                // The call may have written any of the caller-saved
                // registers, and the return value is in r0.
                self.core.written.clear();
                self.core.written.insert(0);
                self.core.ever_written.insert(0);
                self.core.clobber(&[0, 1, 2, 3, 12], 0);
            }
            "b" | "bl" => {
                // A `bl` is a call; the block terminator carries the target
                // and the branch itself emits the call statement.
                if base == "bl" {
                    let target = insn.target.as_ref().and_then(|t| match t {
                        occ_core::CallTarget::Direct { to } => Some(*to),
                        _ => None,
                    });
                    let name = match target {
                        Some(a) => (self.core.name_of)(a),
                        None => Expr::Unknown("call target".into()).to_string(),
                    }
                    .to_string();
                    let args = self.core.call_args();
                    let ret = self.core.name(0);
                    let st = Stmt::Call {
                        name,
                        args,
                        ret: Some(ret),
                        addr: insn.addr,
                        target,
                    };
                    self.core.written.clear();
                    self.core.written.insert(0);
                    self.core.ever_written.insert(0);
                    self.core.clobber(&[0, 1, 2, 3, 12], 0);
                    out.stmts.push(st);
                }
            }
            "svc" | "swi" => {
                out.stmts.push(Stmt::Call {
                    name: "svc".into(),
                    args: Vec::new(),
                    ret: None,
                    addr: insn.addr,
                    target: None,
                });
                self.core.written.insert(0);
                self.core.ever_written.insert(0);
                self.core
                    .assign("r0", Expr::Unknown("svc".into()), &mut out.stmts);
            }
            "nop" => {}
            "sxtah" | "uxtah" | "sxtab" | "uxtab" | "sxtab16" | "uxtab16" => {
                // The extended additions take three registers and add
                // the extended one to the other, and the extension is
                // part of the instruction: a half word that is added
                // after being extended is one operation, and reading it
                // as a plain addition loses the half of the value the
                // instruction threw away.
                let Some(rd) = reg_at(ops, 0) else { return };
                let (Some(rn), Some(rm)) =
                    (ops.get(1).and_then(reg_of), ops.get(2).and_then(reg_of))
                else {
                    return;
                };
                // The third operand may carry a rotation, which the
                // encoder spells out as `ror #8` and the like.
                let rotated = match ops.get(3) {
                    Some(Operand::Imm(v)) => self.rotate_right(self.core.read_reg(rm), *v as u32),
                    _ => self.core.read_reg(rm),
                };
                // The `ab` forms add a byte and the `ah` forms add a half
                // word.  The `16` forms are a different thing again: they
                // work on both halves of the word at once, each half of
                // the second operand is extended and added to the
                // matching half of the first, and the carry of one half
                // never reaches the other.
                if base.ends_with("ab16") {
                    let value = Expr::Intrinsic {
                        op: if base.starts_with('s') {
                            Intrinsic::ExtendAddSigned
                        } else {
                            Intrinsic::ExtendAdd
                        },
                        args: vec![self.core.read_reg(rn), rotated],
                        bits: 4,
                    };
                    self.write_reg(rd, value, guard, &mut out.stmts);
                    return;
                }
                let (from, signed) = match base {
                    "sxtah" => (2, true),
                    "uxtah" => (2, false),
                    "sxtab" => (1, true),
                    _ => (1, false),
                };
                let extended = Expr::Cast {
                    kind: if signed {
                        crate::ir::CastKind::Sext
                    } else {
                        crate::ir::CastKind::Zext
                    },
                    arg: Box::new(rotated),
                    from,
                    to: 4,
                };
                let value = Expr::Bin {
                    op: BinOp::Add,
                    lhs: Box::new(self.core.read_reg(rn)),
                    rhs: Box::new(extended),
                    bits: 4,
                };
                self.write_reg(rd, value, guard, &mut out.stmts);
            }
            "rev" | "rev16" | "revsh" | "rbit" => {
                // The reversals are one operation each: reading them as
                // a chain of shifts would say the same thing, but the
                // machine does them in one step and the value they
                // throw away is part of what they mean.
                let Some(rd) = reg_at(ops, 0) else { return };
                let Some(rm) = ops.get(1).and_then(reg_of) else {
                    return;
                };
                let op = match base {
                    "rev" => Intrinsic::RevBytes,
                    "rev16" => Intrinsic::RevHalves,
                    "revsh" => Intrinsic::RevHalvesSigned,
                    _ => Intrinsic::RevBits,
                };
                let value = Expr::Intrinsic {
                    op,
                    args: vec![self.core.read_reg(rm)],
                    bits: 4,
                };
                self.write_reg(rd, value, guard, &mut out.stmts);
            }
            "sel" => {
                // The select picks each bit from one of two values, and
                // the mask says which one.
                let Some(rd) = reg_at(ops, 0) else { return };
                let (Some(rn), Some(rm)) =
                    (ops.get(1).and_then(reg_of), ops.get(2).and_then(reg_of))
                else {
                    return;
                };
                let value = Expr::Intrinsic {
                    op: Intrinsic::Select,
                    args: vec![
                        self.core.read_reg(rn),
                        self.core.read_reg(rm),
                        self.core.read_reg(rn),
                    ],
                    bits: 4,
                };
                self.write_reg(rd, value, guard, &mut out.stmts);
            }
            "qadd" | "qsub" | "qdadd" | "qdsub" => {
                // The saturating additions and subtractions, and their
                // doubled forms, which add or subtract twice the
                // second operand.
                let Some(rd) = reg_at(ops, 0) else { return };
                let (Some(rm), Some(rn)) =
                    (ops.get(1).and_then(reg_of), ops.get(2).and_then(reg_of))
                else {
                    return;
                };
                let doubled = matches!(base, "qdadd" | "qdsub");
                let sub = matches!(base, "qsub" | "qdsub");
                let second = if doubled {
                    Expr::Bin {
                        op: BinOp::Add,
                        lhs: Box::new(self.core.read_reg(rn)),
                        rhs: Box::new(self.core.read_reg(rn)),
                        bits: 4,
                    }
                } else {
                    self.core.read_reg(rn)
                };
                let value = Expr::Intrinsic {
                    op: if sub {
                        Intrinsic::SatSub
                    } else {
                        Intrinsic::SatAdd
                    },
                    args: vec![self.core.read_reg(rm), second],
                    bits: 4,
                };
                self.write_reg(rd, value, guard, &mut out.stmts);
            }
            _ if base.starts_with("crc32") => {
                // The checksums: a byte, a half word or a word of the
                // second operand is folded into the running value in
                // the first.  The two families differ only in the
                // polynomial they use.
                let Some(rd) = reg_at(ops, 0) else { return };
                let (Some(rn), Some(rm)) =
                    (ops.get(1).and_then(reg_of), ops.get(2).and_then(reg_of))
                else {
                    return;
                };
                let width: u32 = match base {
                    "crc32b" | "crc32cb" => 8,
                    "crc32h" | "crc32ch" => 16,
                    _ => 32,
                };
                let folded = Expr::Cast {
                    kind: crate::ir::CastKind::Zext,
                    arg: Box::new(self.core.read_reg(rm)),
                    from: (width / 8) as u8,
                    to: 4,
                };
                let value = Expr::Intrinsic {
                    op: if base.contains('c') && base.len() > 5 {
                        Intrinsic::Crc32c
                    } else {
                        Intrinsic::Crc32
                    },
                    args: vec![
                        self.core.read_reg(rn),
                        folded,
                        Expr::Const(u64::from(width)),
                    ],
                    bits: 4,
                };
                self.write_reg(rd, value, guard, &mut out.stmts);
            }
            _ if let Some(op) = parallel_intrinsic(base) => {
                // The parallel additions and subtractions, and the
                // exchanged ones that take the halves of the operands
                // the other way round.
                let Some(rd) = reg_at(ops, 0) else { return };
                let (Some(rn), Some(rm)) =
                    (ops.get(1).and_then(reg_of), ops.get(2).and_then(reg_of))
                else {
                    return;
                };
                let (op, lane) = op;
                let value = Expr::Intrinsic {
                    op,
                    args: vec![self.core.read_reg(rn), self.core.read_reg(rm)],
                    bits: lane,
                };
                self.write_reg(rd, value, guard, &mut out.stmts);
            }
            // A helper the decoder emitted for an encoding it does not
            // model: keep the text visible.
            _ => out.stmts.push(Stmt::Comment(insn.text.clone())),
        }
    }

    /// Rotate a value to the right by a whole number of places.
    ///
    /// The extended additions take a rotation of the value they extend,
    /// and the encoder writes it out as `ror #8` and the like.
    fn rotate_right(&mut self, value: Expr, amount: u32) -> Expr {
        if amount == 0 {
            return value;
        }
        Expr::Intrinsic {
            op: Intrinsic::Ror,
            args: vec![value, Expr::Const(u64::from(amount))],
            bits: 4,
        }
    }

    /// The value of an operand, including a register-shifted one, which is
    /// two operands in the list.
    fn operand_value(&mut self, op: &Operand, ops: &[Operand], at: usize) -> Expr {
        if let Operand::Modified {
            reg,
            kind,
            amount: SHIFT_BY_REGISTER,
        } = op
        {
            let base = self.core.read_reg(*reg);
            return match ops.get(at + 1).and_then(reg_of) {
                Some(amount) => self.read_shifted(base, *kind, self.core.read_reg(amount)),
                None => base,
            };
        }
        self.read(op)
    }

    /// Flags after an arithmetic instruction that sets them.
    ///
    /// `subs r1, r1, #1; bne` is how a counting loop ends, and the branch
    /// asks whether the result was zero. Only the zero and sign flags are
    /// modelled: the carry and overflow flags of an arithmetic result are
    /// not a relation between two values, and a branch that asks for them
    /// gets an honest unknown instead of a guessed relation.
    fn set_result_flags(&mut self, rd: u8, _stmts: &mut Vec<Stmt>) {
        let value = self.core.read_reg(Reg::full(rd, 4));
        let zero = Expr::Cond {
            op: CondOp::Eq,
            lhs: Box::new(value.clone()),
            rhs: Box::new(Expr::Const(0)),
            bits: 4,
        };
        let negative = Expr::Cond {
            op: CondOp::Slt,
            lhs: Box::new(value),
            rhs: Box::new(Expr::Const(0)),
            bits: 4,
        };
        self.core.set_flags(
            FlagState {
                zf: Some(zero),
                sf: Some(negative),
                ..FlagState::default()
            },
            None,
        );
    }

    /// The flags an arithmetic instruction leaves behind.
    ///
    /// The carry and the overflow are the two the machine computes from the
    /// operands rather than from the result, and they are what a following
    /// `adc`, `sbc` or `rrx` reads. The zero and the sign flags say what the
    /// result looks like.
    ///
    /// The carry is recorded the way x86 records it — the borrow, not the
    /// "no borrow" A32 calls the carry flag. The condition codes are mapped
    /// to match: `hs` and `lo` are the two names for the same question, and
    /// only one of them can be the one the comparison answers.
    fn set_arith_flags(
        &mut self,
        base: &str,
        lhs: &Expr,
        rhs: &Expr,
        result: &Expr,
        carry: Option<&Expr>,
    ) {
        let (cf, of) = match base {
            "add" => (
                // The sum did not fit: it came back smaller than what it
                // was added to.
                Expr::Cond {
                    op: CondOp::Ult,
                    lhs: Box::new(result.clone()),
                    rhs: Box::new(lhs.clone()),
                    bits: 4,
                },
                Expr::Intrinsic {
                    op: Intrinsic::OverflowAdd,
                    args: vec![lhs.clone(), rhs.clone(), result.clone()],
                    bits: 4,
                },
            ),
            "adc" => (
                Expr::Intrinsic {
                    op: Intrinsic::AdcCarry,
                    args: vec![
                        lhs.clone(),
                        rhs.clone(),
                        carry.cloned().unwrap_or(Expr::Const(0)),
                    ],
                    bits: 4,
                },
                Expr::Intrinsic {
                    op: Intrinsic::OverflowAdd,
                    args: vec![lhs.clone(), rhs.clone(), result.clone()],
                    bits: 4,
                },
            ),
            "sub" | "rsb" => (
                // The difference did not fit: what was subtracted from was
                // smaller than what was subtracted.
                Expr::Cond {
                    op: CondOp::Ult,
                    lhs: Box::new(lhs.clone()),
                    rhs: Box::new(rhs.clone()),
                    bits: 4,
                },
                Expr::Intrinsic {
                    op: Intrinsic::OverflowSub,
                    args: vec![lhs.clone(), rhs.clone(), result.clone()],
                    bits: 4,
                },
            ),
            "sbc" => (
                Expr::Intrinsic {
                    op: Intrinsic::SbbBorrow,
                    args: vec![
                        lhs.clone(),
                        rhs.clone(),
                        carry.cloned().unwrap_or(Expr::Const(0)),
                    ],
                    bits: 4,
                },
                Expr::Intrinsic {
                    op: Intrinsic::OverflowSub,
                    args: vec![lhs.clone(), rhs.clone(), result.clone()],
                    bits: 4,
                },
            ),
            // A logical operation clears both: no carry went out of it and
            // nothing overflowed.
            _ => (Expr::Const(0), Expr::Const(0)),
        };
        let zero = Expr::Cond {
            op: CondOp::Eq,
            lhs: Box::new(result.clone()),
            rhs: Box::new(Expr::Const(0)),
            bits: 4,
        };
        let negative = Expr::Cond {
            op: CondOp::Slt,
            lhs: Box::new(result.clone()),
            rhs: Box::new(Expr::Const(0)),
            bits: 4,
        };
        self.core.set_flags(
            FlagState {
                cf: Some(cf),
                zf: Some(zero),
                sf: Some(negative),
                of: Some(of),
                pf: None,
            },
            None,
        );
    }

    /// The flags a shift leaves behind.
    ///
    /// A shift's carry is the bit that fell off the end: the last one out
    /// on the right, the top one on the left. It is the bit the next
    /// `rrx` or `adc` puts back in, so it is not decoration.
    fn set_shift_flags(&mut self, kind: ModKind, amount: u32, previous: &Expr, result: &Expr) {
        let bit = |position: u32| {
            let shifted = Expr::Bin {
                op: BinOp::Shr,
                lhs: Box::new(previous.clone()),
                rhs: Box::new(Expr::Const(u64::from(position))),
                bits: 4,
            };
            Expr::Cond {
                op: CondOp::Eq,
                lhs: Box::new(Expr::Bin {
                    op: BinOp::And,
                    lhs: Box::new(shifted),
                    rhs: Box::new(Expr::Const(1)),
                    bits: 4,
                }),
                rhs: Box::new(Expr::Const(1)),
                bits: 4,
            }
        };
        let cf = match kind {
            // A shift to the left brings the top bit down to the bottom.
            ModKind::Lsl => bit(32 - amount),
            // A shift to the right takes the bottom bit up to the top.
            _ => bit(amount.saturating_sub(1)),
        };
        let zero = Expr::Cond {
            op: CondOp::Eq,
            lhs: Box::new(result.clone()),
            rhs: Box::new(Expr::Const(0)),
            bits: 4,
        };
        let negative = Expr::Cond {
            op: CondOp::Slt,
            lhs: Box::new(result.clone()),
            rhs: Box::new(Expr::Const(0)),
            bits: 4,
        };
        self.core.set_flags(
            FlagState {
                cf: Some(cf),
                zf: Some(zero),
                sf: Some(negative),
                of: None,
                pf: None,
            },
            None,
        );
    }

    /// A name for a value a comparison read, so the condition it leaves
    /// behind keeps meaning what it meant.
    fn snapshot(&mut self, value: Expr, stmts: &mut Vec<Stmt>) -> Expr {
        match value {
            Expr::Var(name) => {
                self.cmp_seq += 1;
                let temp = format!("cmp{}", self.cmp_seq);
                stmts.push(Stmt::Assign {
                    dest: temp.clone(),
                    value: Expr::Var(name),
                });
                Expr::Var(temp)
            }
            other => other,
        }
    }

    /// A name for a value, so that a later instruction cannot change what
    /// it means.
    ///
    /// `ldr r0, [r0], #4` reads through r0 and then moves it. Without a
    /// name of its own the read would see the moved pointer, and the two
    /// halves of one instruction would disagree about what was read.
    fn bind(&mut self, value: Expr, stmts: &mut Vec<Stmt>) -> Expr {
        match value {
            Expr::Var(name) => {
                self.cmp_seq += 1;
                let temp = format!("t{}", self.cmp_seq);
                stmts.push(Stmt::Assign {
                    dest: temp.clone(),
                    value: Expr::Var(name),
                });
                Expr::Var(temp)
            }
            other => other,
        }
    }

    /// A name for any value, so that a later instruction cannot change what
    /// it means.
    fn bind_value(&mut self, value: Expr, stmts: &mut Vec<Stmt>) -> Expr {
        self.cmp_seq += 1;
        let temp = format!("t{}", self.cmp_seq);
        stmts.push(Stmt::Assign {
            dest: temp.clone(),
            value,
        });
        Expr::Var(temp)
    }

    /// A load or store, with its addressing mode.
    fn transfer(
        &mut self,
        base: &str,
        ops: &[Operand],
        addr: Addr,
        guard: Option<&Expr>,
        stmts: &mut Vec<Stmt>,
    ) {
        let load = base.starts_with("ldr");
        let size = match base {
            "ldrb" | "strb" | "ldrsb" => 1,
            "ldrh" | "strh" | "ldrsh" => 2,
            _ => 4,
        };
        let Some(rt) = reg_at(ops, 0) else { return };
        let Some(mem) = ops.get(1) else { return };
        let Operand::Mem {
            base: Some(base_reg),
            disp,
            index,
            base_update,
            ..
        } = mem
        else {
            // A literal the decoder did not resolve, or an unparsed operand.
            return;
        };
        let base_reg = *base_reg;
        let disp = *disp;
        let index = *index;
        let write_back = *base_update;
        // `[pc, #n]` names a literal in the same section. The register holds
        // the instruction's address plus eight, which the decoder turned
        // into an absolute address; reading it as arithmetic on a variable
        // called `pc` turns a global into a parameter, which is what made a
        // store to a known address read as a store through `a2`.
        let literal = (base_reg.idx == PC)
            .then(|| Addr((i128::from(addr.get()) + 8 + i128::from(disp)) as u64));
        // A post-indexed access writes back after the access, and its offset
        // is the third operand rather than part of the brackets.
        let post = if ops.len() >= 3 {
            match &ops[2] {
                Operand::Imm(v) => Some(*v as i64),
                _ => None,
            }
        } else {
            None
        };
        let op = Operand::Mem {
            size,
            base: if literal.is_some() {
                None
            } else {
                Some(base_reg)
            },
            index,
            scale: 1,
            disp: if literal.is_some() { 0 } else { disp },
            abs: literal.map(|a| a.get()),
            seg: None,
            base_update: None,
        };
        let slot = self.core.stack_slot(&op);
        if load {
            let value = match slot {
                Some(slot) => Expr::var(super::slot_name(slot, size)),
                None => Expr::Load {
                    addr: Box::new(self.core.addr_of(&op, stmts)),
                    size,
                    signed: base.ends_with('s') && base != "ldrsb" && base != "ldrsh"
                        || base == "ldrsb"
                        || base == "ldrsh",
                },
            };
            // A load into the program counter is a jump, and the decoder
            // marks it; the value is not a statement.
            if rt == PC {
                return;
            }
            self.write_reg(rt, value, guard, stmts);
        } else {
            let value = self.core.read_reg(Reg::full(rt, 4));
            match (slot, base_reg.idx == SP) {
                (Some(slot), true) => {
                    let name = super::slot_name(slot, size);
                    let st = Stmt::Assign { dest: name, value };
                    match guard {
                        Some(cond) => stmts.push(Stmt::If {
                            cond: cond.clone(),
                            then_body: vec![st],
                            else_body: Vec::new(),
                        }),
                        None => stmts.push(st),
                    }
                }
                _ => {
                    let addr = self.core.addr_of(&op, stmts);
                    self.store(addr, value, size, guard, stmts);
                }
            }
        }
        // Write-back of the base register.
        let delta = post.or_else(|| {
            if write_back.is_some() {
                Some(disp)
            } else {
                None
            }
        });
        if let Some(delta) = delta
            && delta != 0
        {
            let name = self.core.name(base_reg.idx);
            // The pointer that moves is the one the instruction was given.
            // `ldr r0, [r0], #4` reads through r0 and writes r0, and the
            // write-back has to start from the pointer, not from the word
            // that was just read.
            let previous = if rt == base_reg.idx {
                self.bind(self.core.read_reg(base_reg), stmts)
            } else {
                self.core.read_reg(base_reg)
            };
            let value = Expr::Bin {
                op: BinOp::Add,
                lhs: Box::new(previous),
                rhs: Box::new(Expr::Const(delta as u64)),
                bits: 4,
            };
            // The stack pointer moving is frame bookkeeping, not a value.
            if base_reg.idx == SP {
                self.core.adjust_sp(delta);
            } else {
                self.core.written.insert(base_reg.idx);
                self.core.ever_written.insert(base_reg.idx);
                self.core.assign(&name, value, stmts);
            }
        }
    }

    /// `ldrd` / `strd`: two consecutive words, in two registers.
    ///
    /// The pair is one instruction and two accesses. The architecture
    /// requires the first register to be even and the address to be
    /// word-aligned, so the second word is always at the next address.
    fn double_transfer(
        &mut self,
        base: &str,
        ops: &[Operand],
        guard: Option<&Expr>,
        stmts: &mut Vec<Stmt>,
    ) {
        let (Some(lo), Some(hi), Some(mem)) = (
            ops.first().and_then(reg_of),
            ops.get(1).and_then(reg_of),
            ops.get(2),
        ) else {
            return;
        };
        let Some(addr_expr) = self.memory_address(mem) else {
            return;
        };
        // The base register may also be a destination: `ldrd r0, r1, [r0]`
        // reads through r0 and writes it. The read has to see the pointer
        // the instruction was given, not the word it is about to put
        // there, so the address is given a name of its own first.
        let addr_expr = if matches!(mem, Operand::Mem { base: Some(b), .. }
            if b.idx == lo.idx || b.idx == hi.idx)
        {
            self.bind_value(addr_expr, stmts)
        } else {
            addr_expr
        };
        let hi_addr = Expr::Bin {
            op: BinOp::Add,
            lhs: Box::new(addr_expr.clone()),
            rhs: Box::new(Expr::Const(4)),
            bits: 4,
        };
        if base == "ldrd" {
            let first = Expr::Load {
                addr: Box::new(addr_expr),
                size: 4,
                signed: false,
            };
            let second = Expr::Load {
                addr: Box::new(hi_addr),
                size: 4,
                signed: false,
            };
            self.write_reg(lo.idx, first, guard, stmts);
            self.write_reg(hi.idx, second, guard, stmts);
        } else {
            let first = self.core.read_reg(lo);
            let second = self.core.read_reg(hi);
            self.store(addr_expr, first, 4, guard, stmts);
            self.store(hi_addr, second, 4, guard, stmts);
        }
    }

    /// The address expression of a simple memory operand.
    fn memory_address(&mut self, mem: &Operand) -> Option<Expr> {
        if !matches!(mem, Operand::Mem { .. }) {
            return None;
        }
        Some(self.core.addr_of(mem, &mut Vec::new()))
    }

    /// A block transfer: `push`, `pop`, `ldm`, `stm`.
    ///
    /// The register list is not an operand the general parser can read, and
    /// the stack forms are the frame construction itself: `push {r4, lr}`
    /// saves two registers and moves the stack pointer by eight, and both
    /// halves of that have to happen.
    fn block_transfer(&mut self, text: &str, stmts: &mut Vec<Stmt>) -> bool {
        let Some((open, rest)) = text.split_once('{') else {
            return false;
        };
        let Some((list, tail)) = rest.split_once('}') else {
            return false;
        };
        let head = open.trim();
        let (mnemonic, mut base_text) = match head.split_once(char::is_whitespace) {
            Some((m, r)) => (m, r.trim().to_string()),
            None => (head, String::new()),
        };
        let lower = mnemonic.to_ascii_lowercase();
        let (base, _s, cond) = split_mnemonic(&lower);
        if !matches!(base, "push" | "pop" | "ldm" | "stm") {
            return false;
        }
        let guard = cond.map(|c| self.core.condition_of(&lower, c));
        let guard = guard.as_ref();
        let registers: Vec<u8> = list
            .split(',')
            .filter_map(|r| register(r.trim()).map(|r| r.idx))
            .collect();
        if registers.is_empty() {
            return false;
        }
        let count = registers.len() as i64 * 4;
        // `push {..}` is `stmdb sp!, {..}` and `pop {..}` is `ldmia sp!,
        // {..}`; the decoder prints the alias, and the frame moves by four
        // bytes per register either way.
        let mut offset = if base == "push" {
            self.core.adjust_sp(-count);
            count
        } else {
            0
        };
        let after = base_text.ends_with('!');
        base_text = base_text.trim_end_matches('!').trim().to_string();
        let pointer = if base_text.is_empty() {
            Reg::full(SP, 4)
        } else {
            match register(&base_text) {
                Some(r) => r,
                None => return false,
            }
        };
        // The frame a function builds for itself is not part of what it
        // computes. `push {r4, r5, r11, lr}` saves the registers the
        // caller owns so that the body can use them, and the matching
        // `pop` puts them back; nothing inside the function reads the
        // slots, and printing the traffic says the function stores four
        // values it never uses.
        //
        // The pair is recognised by the link register: a push that saves
        // it is a prologue, and a pop that restores it — or a pop that
        // follows one — is the matching epilogue.
        let saves_link = registers.contains(&LR);
        let restores_link = registers.contains(&PC);
        if base == "push" && saves_link {
            self.saved_link = true;
            return true;
        }
        if base == "pop" && (restores_link || self.saved_link) {
            self.core.adjust_sp(count);
            return true;
        }
        for &r in &registers {
            // The lowest register is at the lowest address for `push`, and
            // the list is written that way in the instruction.
            let slot = if base == "pop" { offset } else { offset - 4 };
            if base == "pop" {
                offset += 4;
            } else {
                offset -= 4;
            }
            let addr = Expr::Bin {
                op: BinOp::Add,
                lhs: Box::new(Expr::var(self.core.name(pointer.idx))),
                rhs: Box::new(Expr::Const(slot as u64)),
                bits: 4,
            };
            if base == "pop" || base == "ldm" {
                let value = Expr::Load {
                    addr: Box::new(addr),
                    size: 4,
                    signed: false,
                };
                if r == PC {
                    continue;
                }
                self.write_reg(r, value, guard, stmts);
            } else {
                let value = self.core.read_reg(Reg::full(r, 4));
                self.store(addr, value, 4, guard, stmts);
            }
        }
        if base == "pop" {
            self.core.adjust_sp(count);
        } else if after || base == "push" || base == "pop" {
            // `ldm r0!, {..}` and `stm r0!, {..}` move the base by the list
            // size; the stack forms already moved it.
            if pointer.idx != SP {
                let value = Expr::Bin {
                    op: BinOp::Add,
                    lhs: Box::new(self.core.read_reg(pointer)),
                    rhs: Box::new(Expr::Const(count as u64)),
                    bits: 4,
                };
                let name = self.core.name(pointer.idx);
                self.core.written.insert(pointer.idx);
                self.core.ever_written.insert(pointer.idx);
                self.core.assign(&name, value, stmts);
            }
        }
        let _ = tail;
        true
    }

    /// The condition on a conditional branch.
    fn branch_condition(&mut self, insn: &Instruction) -> Expr {
        let mnemonic = insn.text.split_whitespace().next().unwrap_or("");
        let (base, _, cond) = split_mnemonic(mnemonic);
        // `bxne lr` is the conditional return compilers use for one arm of a
        // guard. It is a branch for the block, and the condition it tests is
        // the same one `bne` would.
        if base != "b" && base != "bx" && base != "bxns" {
            // `cbz` and `cbnz` are Thumb's own compare-and-branch. The
            // condition is not a condition code: it is the value of one
            // register against zero.
            if base == "cbz" || base == "cbnz" {
                let rest = insn
                    .text
                    .split_once(char::is_whitespace)
                    .map(|(_, r)| r.trim())
                    .unwrap_or("");
                let Some(reg) = rest.split(',').next().and_then(|r| register(r.trim())) else {
                    return Expr::Unknown("condition".into());
                };
                let value = self.core.read_reg(reg);
                return Expr::Cond {
                    op: if base == "cbz" {
                        CondOp::Eq
                    } else {
                        CondOp::Ne
                    },
                    lhs: Box::new(value),
                    rhs: Box::new(Expr::Const(0)),
                    bits: reg.size,
                };
            }
            return Expr::Unknown("condition".into());
        }
        match cond {
            Some(c) => self.core.condition_of(mnemonic, c),
            None => Expr::Unknown("condition".into()),
        }
    }

    /// The register a computed jump goes through.
    fn indirect_target(&mut self, insn: &Instruction) -> Expr {
        let text = insn.text.trim();
        let (_, _, _) = split_mnemonic(text.split_whitespace().next().unwrap_or(""));
        let rest = text
            .split_once(char::is_whitespace)
            .map(|(_, r)| r)
            .unwrap_or("");
        match register(rest.trim()) {
            Some(r) => self.core.read_reg(r),
            None => Expr::Unknown("jump target".into()),
        }
    }
}

/// Whether an instruction is the head of an `it` block.
///
/// The name is `it` followed by one letter for every instruction after the
/// first: `t` for the arm that runs when the condition holds and `e` for
/// the arm that runs when it does not. The block itself does nothing, so
/// the lifter skips it and lets the conditions it carries speak for the
/// instructions that follow.
fn is_it_block(text: &str) -> bool {
    let name = text.split_whitespace().next().unwrap_or("");
    let Some(letters) = name.strip_prefix("it") else {
        return false;
    };
    letters.chars().all(|c| c == 't' || c == 'e')
}

fn reg_at(ops: &[Operand], i: usize) -> Option<u8> {
    ops.get(i).and_then(reg_of).map(|r| r.idx)
}

/// Whether an instruction is a return that happens only sometimes.
///
/// `bx lr` is the return an ARM function leaves through, and `bx<cond> lr`
/// is the same return on one path only. The text is the only place the
/// difference shows: the link register is named, and a condition stands
/// between it and the instruction.
///
/// `pop {..., pc}` is the other return, and it carries its condition in the
/// same place.
fn is_conditional_return(text: &str) -> bool {
    let text = text.trim();
    if let Some(rest) = text
        .strip_prefix("bx")
        .or_else(|| text.strip_prefix("bxns"))
    {
        let rest = rest.trim();
        let Some(condition) = rest.strip_suffix(" lr") else {
            return false;
        };
        // `bx lr` returns on every path; `bx<cond> lr` returns on one.
        return !condition.trim().is_empty();
    }
    // `pop {..., pc}` loads the program counter, which is a return.
    let Some((name, list)) = text.split_once('{') else {
        return false;
    };
    if !list.contains("pc") {
        return false;
    }
    let Some(condition) = name.trim().strip_prefix("pop") else {
        return false;
    };
    !condition.trim().is_empty()
}

fn reg_of(op: &Operand) -> Option<Reg> {
    match op {
        Operand::Reg(r) => Some(*r),
        Operand::Modified { reg, .. } => Some(*reg),
        _ => None,
    }
}

impl<'a> super::Lifter for Lifter<'a> {
    fn lift_block(&mut self, insns: &[Instruction]) -> Lifted {
        let mut out = Lifted::default();
        self.core.written.clear();
        for insn in insns {
            let before = out.stmts.len();
            self.lift_insn(insn, &mut out);
            for _ in before..out.stmts.len() {
                out.stmt_at.push(insn.addr);
            }
        }
        if let Some(insn) = insns.last() {
            match insn.flow {
                FlowKind::ConditionalBranch => {
                    out.branch = Some(self.branch_condition(insn));
                    // `bx<cond> lr` is a return on one path. The target is
                    // the link register rather than an address, so the
                    // block terminator has to be told that the path leaves
                    // the function by returning.
                    out.conditional_return = is_conditional_return(&insn.text);
                }
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

    fn ret_pair(&self) -> Option<(String, String)> {
        // A sixty-four bit value comes back in r0 and r1, the low half
        // first.
        Some((
            self.core.name(self.core.target.ret_reg()),
            self.core.name(self.core.target.ret_reg() + 1),
        ))
    }

    fn ret_type(&self) -> Option<&'static str> {
        if self.core.ever_written.contains(&0) {
            Some("u32")
        } else {
            None
        }
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
