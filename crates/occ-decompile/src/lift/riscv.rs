//! RISC-V lifting: text to statements.
//!
//! RISC-V has no condition flags. A branch names the two registers it
//! compares, so the condition is in the branch instruction itself rather than
//! in whatever the previous instruction left behind. That is the one place
//! this architecture differs from x86 and AArch64 in kind rather than in
//! spelling, and it is why the branch condition is built here from the
//! branch's own operands.
//!
//! The ABI names are the ones the decoder prints: `zero`, `ra`, `sp`, `a0`
//! and so on. They are easier to read than `x10`, and the decoder already
//! prints them everywhere.

use occ_core::{Addr, FlowKind, Instruction};

use super::{Access, Core, Lifted, Operand, Parsed, Reg, Target, parse_int, split_operands};
use crate::ir::{BinOp, CondOp, Expr, FlagCond, Stmt, UnOp};

/// Register names by index, in the ABI spelling.
const REGS: [&str; 32] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4",
    "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4",
    "t5", "t6",
];

const ZERO: u8 = 0;
const RA: u8 = 1;
const SP: u8 = 2;
const GP: u8 = 3;
const S0: u8 = 8;

/// Integer argument registers, in ABI order: `a0` through `a7`.
const ARG_REGS: [u8; 8] = [10, 11, 12, 13, 14, 15, 16, 17];

/// The RISC-V target.
pub struct RiscvTarget;

impl Target for RiscvTarget {
    fn lookup_reg(&self, name: &str) -> Option<Reg> {
        REGS
            .iter()
            .position(|n| *n == name)
            .map(|n| Reg::full(n as u8, 8))
    }

    fn reg_name(&self, idx: u8) -> String {
        REGS.get(usize::from(idx)).copied().unwrap_or("zero").to_string()
    }

    fn reg_width(&self, _idx: u8) -> u8 {
        8
    }

    fn arg_regs(&self) -> &[u8] {
        &ARG_REGS
    }

    fn ret_reg(&self) -> u8 {
        10
    }

    fn sp_reg(&self) -> u8 {
        SP
    }

    fn fp_reg(&self) -> Option<u8> {
        // `s0` is the frame pointer when the compiler emits a frame; it is
        // also `fp`, and the decoder prints the ABI spelling.
        Some(S0)
    }

    fn is_pc(&self, _idx: u8) -> bool {
        false
    }

    fn sp_alias(&self) -> &'static str {
        "frame"
    }

    fn branch_cond(&self, mnemonic: &str) -> Option<FlagCond> {
        // The condition is built from the branch's operands, not from a
        // pending comparison, so this is only used to recognise the mnemonic.
        Some(match mnemonic {
            "beq" | "beqz" => FlagCond::Z,
            "bne" | "bnez" => FlagCond::Nz,
            "blt" | "bltz" => FlagCond::L,
            "bge" | "bgez" => FlagCond::Nl,
            "bltu" | "bltuz" => FlagCond::C,
            "bgeu" | "bgeuz" => FlagCond::Nc,
            "blez" => FlagCond::Le,
            "bgtz" => FlagCond::G,
            _ => return None,
        })
    }

    fn setcc_cond(&self, _suffix: &str) -> Option<FlagCond> {
        // RISC-V has no conditional set: a comparison is a branch.
        None
    }

    fn cmov_cond(&self, _suffix: &str) -> Option<FlagCond> {
        None
    }

    fn mem_access(&self, mnemonic: &str, _data_size: u8) -> Access {
        // The width is in the mnemonic, not in the operand: `sb` is one byte,
        // `sw` is four, `sd` is eight.
        let width = match mnemonic {
            "lb" | "lbu" | "sb" => 1,
            "lh" | "lhu" | "sh" => 2,
            "lw" | "lwu" | "sw" => 4,
            _ => 8,
        };
        // On RV64 a 32-bit load sign-extends into the 64-bit register, and
        // `lwu` is the form that does not. The byte and half loads follow
        // the same rule: `lb` extends, `lbu` does not.
        let signed = matches!(mnemonic, "lb" | "lh" | "lw");
        Access { width, signed }
    }

    fn fixed_register(&self, idx: u8) -> Option<&'static str> {
        // `gp` points at `__global_pointer$`, which the linker fixes and the
        // startup code installs. Without it every global access through
        // `gp` is an address nobody can name.
        if idx == GP {
            Some("__global_pointer$")
        } else {
            None
        }
    }

    fn zero_reg(&self) -> Option<u8> {
        // `x0` is hardwired to zero: reads return zero and writes are
        // discarded. Lifting it as an ordinary register turned
        // `beq a0, zero` into a comparison against an unknown value.
        Some(ZERO)
    }

    fn parse(&self, text: &str) -> Option<Parsed> {
        parse_riscv(text)
    }
}

/// True when a jump instruction is the return: `jr ra` or `jr x1`.
fn is_return_jump(ops: &[Operand]) -> bool {
    matches!(ops, [Operand::Reg(r)] if r.idx == RA)
}

/// The value an upper-immediate instruction puts in its register.
///
/// The decoder prints the twenty-bit field the instruction carries, already
/// divided by 4096. The register receives that field shifted back into place
/// and sign-extended from 32 bits, so `lui a0, 0x1` yields `0x1000` and
/// `lui a0, -0x1` yields `0xfffffffffffff000`.
fn upper_immediate(field: u64) -> i64 {
    let value = ((field as u32) & 0xfffff) as i32;
    i64::from(value << 12)
}

/// Parse a RISC-V instruction's text.
fn parse_riscv(text: &str) -> Option<Parsed> {
    let body = text.trim();
    if body.is_empty() {
        return None;
    }
    let (mnemonic, rest) = match body.split_once(' ') {
        Some((m, r)) => (m, r),
        None => (body, ""),
    };
    let mut ops = Vec::new();
    for part in split_operands(rest.trim()) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        ops.push(parse_riscv_operand(part)?);
    }
    Some(Parsed {
        mnemonic: mnemonic.to_ascii_lowercase(),
        ops,
        pc_target: None,
        prefixes: Vec::new(),
    })
}

/// One RISC-V operand.
fn parse_riscv_operand(s: &str) -> Option<Operand> {
    let s = s.trim();
    // A memory reference: `8(a0)` or `-16(sp)`.
    if let Some(open) = s.find('(') {
        let close = s.rfind(')')?;
        let disp = parse_int(&s[..open])? as i64;
        let base = RiscvTarget.lookup_reg(s[open + 1..close].trim())?;
        return Some(Operand::Mem {
            size: 0,
            base: Some(base),
            index: None,
            scale: 1,
            disp,
            abs: None,
            seg: None,
            base_update: None,
        });
    }
    if let Some(v) = parse_int(s) {
        return Some(Operand::Imm(v));
    }
    let reg = RiscvTarget.lookup_reg(s)?;
    Some(Operand::Reg(reg))
}

/// The RISC-V lifter.
pub struct Lifter<'a> {
    /// Shared state.
    pub core: Core<'a>,
    /// The mnemonic of the branch being lifted, so the condition can be built
    /// from the branch's own operands.
    current_mnemonic: String,
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
            current_mnemonic: String::new(),
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
            "nop" | "fence" | "fence.i" | "ecall" | "ebreak" | "ret" | "unimp" => {}
            // `jr ra` is `ret`: the decoder prints the jump form because
            // that is what the instruction encodes.
            "jr" if is_return_jump(&p.ops) => {}
            "mv" => self.lift_mv(&p.ops, &mut out.stmts),
            "li" => self.lift_li(&p.ops, &mut out.stmts),
            "lui" => self.lift_lui(&p.ops, &mut out.stmts),
            "auipc" => self.lift_auipc(insn, &p.ops, &mut out.stmts),
            "add" | "addi" | "addw" | "addiw" => self.lift_alu(BinOp::Add, &p.ops, &mut out.stmts),
            "sub" | "subw" => self.lift_alu(BinOp::Sub, &p.ops, &mut out.stmts),
            "and" | "andi" => self.lift_alu(BinOp::And, &p.ops, &mut out.stmts),
            "or" | "ori" => self.lift_alu(BinOp::Or, &p.ops, &mut out.stmts),
            "xor" | "xori" => self.lift_alu(BinOp::Xor, &p.ops, &mut out.stmts),
            "sll" | "slli" | "sllw" | "slliw" => {
                self.lift_shift(BinOp::Shl, &p.ops, &mut out.stmts)
            }
            "srl" | "srli" | "srlw" | "srliw" => {
                self.lift_shift(BinOp::Shr, &p.ops, &mut out.stmts)
            }
            "sra" | "srai" | "sraw" | "sraiw" => {
                self.lift_shift(BinOp::Sar, &p.ops, &mut out.stmts)
            }
            "mul" | "mulw" => self.lift_mul(&p.ops, &mut out.stmts),
            "div" | "divw" => self.lift_div(BinOp::Div, &p.ops, &mut out.stmts),
            "divu" | "divuw" => self.lift_div(BinOp::UDiv, &p.ops, &mut out.stmts),
            "rem" | "remw" => self.lift_div(BinOp::Mod, &p.ops, &mut out.stmts),
            "remu" | "remuw" => self.lift_div(BinOp::UMod, &p.ops, &mut out.stmts),
            "slt" | "slti" => self.lift_slt(&p.ops, &mut out.stmts, true),
            "sltu" | "sltiu" => self.lift_slt(&p.ops, &mut out.stmts, false),
            "neg" => self.lift_neg(&p.ops, &mut out.stmts),
            "not" => self.lift_not(&p.ops, &mut out.stmts),
            "seqz" => self.lift_seqz(&p.ops, &mut out.stmts),
            "snez" => self.lift_snez(&p.ops, &mut out.stmts),
            "sltz" => self.lift_sltz(&p.ops, &mut out.stmts),
            "sgtz" => self.lift_sgtz(&p.ops, &mut out.stmts),
            "ld" | "lw" | "lwu" | "lh" | "lhu" | "lb" | "lbu" => {
                self.lift_load(m, &p.ops, &mut out.stmts)
            }
            "sd" | "sw" | "sh" | "sb" => self.lift_store(m, &p.ops, &mut out.stmts),
            "jal" | "jalr" => self.lift_call(insn, &p.ops, &mut out.stmts),
            "j" => {}
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

    /// `mv rd, rs`: copy.
    fn lift_mv(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let value = self.core.read(&ops[1], stmts);
        self.core.write(&ops[0], value, stmts);
    }

    /// `li rd, imm`: load an immediate.
    fn lift_li(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let Operand::Imm(value) = &ops[1] else {
            return;
        };
        self.core.write(&ops[0], Expr::Const(*value), stmts);
    }

    /// `lui rd, imm`: load the upper half of an immediate.
    ///
    /// The decoder prints the immediate already divided by 4096, because
    /// that is the field the instruction carries. The value the register
    /// ends up with is the field shifted back into place, sign-extended
    /// from 32 bits.
    fn lift_lui(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let Operand::Imm(value) = &ops[1] else {
            return;
        };
        self.core
            .write(&ops[0], Expr::Const(upper_immediate(*value) as u64), stmts);
    }

    /// `auipc rd, imm`: the program counter plus an upper-half immediate.
    ///
    /// The result is relative to the address of the instruction itself, not
    /// to the start of the function and not to the next instruction.
    fn lift_auipc(&mut self, insn: &Instruction, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let Operand::Imm(value) = &ops[1] else {
            return;
        };
        let target = insn.addr.get().wrapping_add(upper_immediate(*value) as u64);
        self.core.write(&ops[0], Expr::Const(target), stmts);
    }

    fn lift_alu(&mut self, op: BinOp, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 3 {
            return;
        }
        // `addi sp, sp, -16` is frame bookkeeping, not a value.
        if let Operand::Reg(dst) = &ops[0]
            && dst.idx == SP
            && let Operand::Reg(src) = &ops[1]
            && src.idx == SP
            && let Operand::Imm(delta) = &ops[2]
        {
            match op {
                BinOp::Add => self.core.adjust_sp(*delta as i64),
                BinOp::Sub => self.core.adjust_sp(-(*delta as i64)),
                _ => self.core.dynamic_sp = true,
            }
            return;
        }
        let lhs = self.core.read(&ops[1], stmts);
        let rhs = self.core.read(&ops[2], stmts);
        let value = Expr::Bin {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            bits: 8,
        };
        self.core.write(&ops[0], value, stmts);
    }

    fn lift_shift(&mut self, op: BinOp, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 3 {
            return;
        }
        let lhs = self.core.read(&ops[1], stmts);
        let rhs = self.core.read(&ops[2], stmts);
        let value = Expr::Bin {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            bits: 8,
        };
        self.core.write(&ops[0], value, stmts);
    }

    fn lift_mul(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 3 {
            return;
        }
        let lhs = self.core.read(&ops[1], stmts);
        let rhs = self.core.read(&ops[2], stmts);
        let value = Expr::Bin {
            op: BinOp::Mul,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            bits: 8,
        };
        self.core.write(&ops[0], value, stmts);
    }

    fn lift_div(&mut self, op: BinOp, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 3 {
            return;
        }
        let lhs = self.core.read(&ops[1], stmts);
        let rhs = self.core.read(&ops[2], stmts);
        let value = Expr::Bin {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            bits: 8,
        };
        self.core.write(&ops[0], value, stmts);
    }

    /// `slt`/`slti`: set a register to the result of a comparison.
    fn lift_slt(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>, signed: bool) {
        if ops.len() != 3 {
            return;
        }
        let lhs = self.core.read(&ops[1], stmts);
        let rhs = self.core.read(&ops[2], stmts);
        let value = Expr::Cond {
            op: if signed { CondOp::Slt } else { CondOp::Ult },
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            bits: ops[0].size(),
        };
        self.core.write(&ops[0], value, stmts);
    }

    fn lift_neg(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let value = self.core.read(&ops[1], stmts);
        let value = Expr::Un {
            op: UnOp::Neg,
            arg: Box::new(value),
            bits: 8,
        };
        self.core.write(&ops[0], value, stmts);
    }

    fn lift_not(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let value = self.core.read(&ops[1], stmts);
        let value = Expr::Un {
            op: UnOp::Not,
            arg: Box::new(value),
            bits: 8,
        };
        self.core.write(&ops[0], value, stmts);
    }

    /// `seqz rd, rs`: 1 when the register is zero.
    fn lift_seqz(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let value = self.core.read(&ops[1], stmts);
        let value = Expr::Cond {
            op: CondOp::Eq,
            lhs: Box::new(value),
            rhs: Box::new(Expr::Const(0)),
            bits: ops[1].size(),
        };
        self.core.write(&ops[0], value, stmts);
    }

    /// `snez rd, rs`: 1 when the register is not zero.
    fn lift_snez(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let value = self.core.read(&ops[1], stmts);
        let value = Expr::Cond {
            op: CondOp::Ne,
            lhs: Box::new(value),
            rhs: Box::new(Expr::Const(0)),
            bits: ops[1].size(),
        };
        self.core.write(&ops[0], value, stmts);
    }

    /// `sltz rd, rs`: 1 when the register is negative.
    fn lift_sltz(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let value = self.core.read(&ops[1], stmts);
        let value = Expr::Cond {
            op: CondOp::Slt,
            lhs: Box::new(value),
            rhs: Box::new(Expr::Const(0)),
            bits: ops[1].size(),
        };
        self.core.write(&ops[0], value, stmts);
    }

    /// `sgtz rd, rs`: 1 when the register is positive.
    fn lift_sgtz(&mut self, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let value = self.core.read(&ops[1], stmts);
        let value = Expr::Cond {
            op: CondOp::Sgt,
            lhs: Box::new(value),
            rhs: Box::new(Expr::Const(0)),
            bits: ops[1].size(),
        };
        self.core.write(&ops[0], value, stmts);
    }

    /// A load: the destination takes the value at the address.
    ///
    /// The address is not the value. A lifter that assigns the address to
    /// the destination turns every load into an address computation, and the
    /// analysis that follows then reasons about pointers where the program
    /// reasons about data.
    fn lift_load(&mut self, mnemonic: &str, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let access = self.core.access_of(mnemonic, &ops[1], ops[0].size());
        let value = self.core.read_load(&ops[1], access, stmts);
        self.core.write(&ops[0], value, stmts);
    }

    /// A store: the address receives the value.
    fn lift_store(&mut self, mnemonic: &str, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        if ops.len() != 2 {
            return;
        }
        let access = self.core.access_of(mnemonic, &ops[1], ops[0].size());
        let value = self.core.read(&ops[0], stmts);
        self.core.store(&ops[1], value, access, stmts);
    }

    fn lift_call(&mut self, insn: &Instruction, ops: &[Operand], stmts: &mut Vec<Stmt>) {
        let name = match insn.target {
            Some(occ_core::CallTarget::Direct { to }) => (self.core.name_of)(to),
            _ => match ops.last() {
                Some(o) => {
                    let target = self.core.read(o, stmts);
                    format!("(*{target})")
                }
                None => "(*?)".into(),
            },
        };
        let args = self.core.call_args();
        stmts.push(Stmt::Call {
            name,
            args,
            ret: Some(self.core.name(self.core.target.ret_reg())),
            addr: insn.addr,
        });
        self.core.written.clear();
        self.core.written.insert(self.core.target.ret_reg());
        self.core.ever_written.insert(self.core.target.ret_reg());
        // The temporaries and the argument registers are the callee's to
        // overwrite; `s0`..`s11` are the ones it has to restore.
        let caller_saved: Vec<u8> = (5..=7).chain(10..=17).chain(28..=31).collect();
        self.core.clobber(&caller_saved, self.core.target.ret_reg());
        let _ = (RA, ZERO, FlowKind::Call);
    }

    /// The condition a branch tests, built from the branch's own operands.
    ///
    /// RISC-V has no flags, so there is nothing pending to compare: the two
    /// registers the branch names are the comparison.
    fn branch_condition(&mut self, ops: &[Operand]) -> Expr {
        let cond = match self.core.target.branch_cond(&self.current_mnemonic) {
            Some(c) => match c.to_cond_op() {
                Some(op) => op,
                None => return Expr::Unknown("condition".into()),
            },
            None => return Expr::Unknown("condition".into()),
        };
        // The one-register forms compare against zero.
        let (lhs, rhs) = match ops {
            [Operand::Reg(a), Operand::Reg(b), ..] => {
                (self.core.read_reg(*a), self.core.read_reg(*b))
            }
            [Operand::Reg(a), ..] => (self.core.read_reg(*a), Expr::Const(0)),
            _ => return Expr::Unknown("condition".into()),
        };
        Expr::Cond {
            op: cond,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            bits: 8,
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
            if matches!(insn.flow, FlowKind::ConditionalBranch) {
                let parsed = self.core.target.parse(&insn.text);
                out.branch = Some(match parsed {
                    Some(p) => {
                        self.current_mnemonic = p.mnemonic.clone();
                        let cond = self.branch_condition(&p.ops);
                        self.current_mnemonic.clear();
                        cond
                    }
                    None => Expr::Unknown("condition".into()),
                });
            } else if matches!(insn.flow, FlowKind::UnconditionalBranch)
                && insn.target.is_none()
            {
                out.indirect = Some(Expr::Unknown("indirect target".into()));
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
