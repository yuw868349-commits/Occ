//! A concrete interpreter for the lifted IR.
//!
//! The decompiler renders the IR; the symbolic executor reasons about it; this
//! module *runs* it. That third use is what makes the other two checkable: the
//! same statements, given the same registers and memory, must produce the same
//! result the processor produces. When they do not, one of the two is wrong and
//! the difference says which operand the lifter misread.
//!
//! # What is modelled
//!
//! Integers, conditions, casts, memory, and the intrinsics the IR keeps for
//! operations the machine word has no spelling for. Every width is explicit, so
//! a 32-bit `add` that the lifter marked with `bits: 4` wraps at 32 bits here
//! exactly as it does on the processor.
//!
//! # What is not
//!
//! Flags are not a register. The IR keeps them as a [`FlagState`] — the
//! expression that holds when each flag is set — so a flag can be evaluated
//! when a condition reads it, and nothing is stored otherwise. A caller that
//! wants the flags after an instruction evaluates
//! [`Machine::flag`] for each one.
//!
//! x87's 80-bit format is evaluated through `f64`. That loses the last eleven
//! bits of the mantissa, so a differential test against a real `fsin` will
//! disagree in the low bits. The disagreement is the model's, not the lifter's,
//! and it is reported rather than hidden.
//!
//! Division by zero and an unbound variable are errors, not zero. A machine
//! that returned zero for a missing value would make every differential test
//! pass for the wrong reason.

use std::collections::BTreeMap;

use crate::ir::{
    BinOp, BoolOp, CastKind, CondOp, Expr, FlagCond, FlagState, Intrinsic, Stmt, Term, UnOp, mask,
};

/// Why evaluation stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fault {
    /// A value the program never defined was read.
    Unbound(String),
    /// Memory outside the modelled range was touched.
    Unmapped(u64),
    /// A division by zero, which the processor would fault on.
    DivideByZero,
    /// An operation the interpreter does not model.
    Unsupported(String),
    /// The step budget ran out. Not a bug: a caller's bound was reached.
    Budget,
}

impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fault::Unbound(name) => write!(f, "unbound value {name}"),
            Fault::Unmapped(a) => write!(f, "unmapped memory at {a:#x}"),
            Fault::DivideByZero => write!(f, "division by zero"),
            Fault::Unsupported(s) => write!(f, "unsupported: {s}"),
            Fault::Budget => write!(f, "step budget exhausted"),
        }
    }
}

impl std::error::Error for Fault {}

/// Registers, memory and the flags of one execution.
#[derive(Clone, Debug, Default)]
pub struct Machine {
    /// Named values: architectural registers and SSA temporaries.
    pub vars: BTreeMap<String, u64>,
    /// Modelled memory, byte addressed. Sparse: a page that was never written
    /// has no entry, and reading it is a fault rather than a zero.
    pub mem: BTreeMap<u64, u8>,
    /// Flags as the lifter computed them.
    pub flags: FlagState,
    /// Instructions of IR executed, for a budget.
    pub steps: u64,
}

impl Machine {
    /// An empty machine with no registers and no memory.
    pub fn new() -> Machine {
        Machine::default()
    }

    /// Set a named value.
    pub fn set(&mut self, name: &str, value: u64) {
        self.vars.insert(name.to_string(), value);
    }

    /// Read a named value, or `None`.
    pub fn get(&self, name: &str) -> Option<u64> {
        self.vars.get(name).copied()
    }

    /// Fill a memory range with bytes.
    pub fn map(&mut self, addr: u64, bytes: &[u8]) {
        for (i, b) in bytes.iter().enumerate() {
            self.mem.insert(addr.wrapping_add(i as u64), *b);
        }
    }

    /// Read a memory range back.
    pub fn peek(&self, addr: u64, len: usize) -> Option<Vec<u8>> {
        let mut out = Vec::with_capacity(len);
        for i in 0..len as u64 {
            out.push(*self.mem.get(&addr.wrapping_add(i))?);
        }
        Some(out)
    }

    /// Read `size` bytes little-endian at `addr`.
    fn load(&self, addr: u64, size: u8, signed: bool) -> Result<u64, Fault> {
        let n = u32::from(size);
        debug_assert!((1..=8).contains(&n));
        let mut v: u64 = 0;
        for i in 0..n {
            let a = addr.wrapping_add(u64::from(i));
            let b = *self.mem.get(&a).ok_or(Fault::Unmapped(a))?;
            v |= u64::from(b) << (8 * i);
        }
        Ok(if signed { sign_extend(v, size) } else { v })
    }

    /// Write `size` bytes little-endian at `addr`.
    fn store(&mut self, addr: u64, size: u8, value: u64) -> Result<(), Fault> {
        // Every byte of the range is checked before any is written: a partial
        // store would leave the machine in a state no processor can reach, and
        // a differential test would then disagree for a reason that is ours.
        for i in 0..u32::from(size) {
            let a = addr.wrapping_add(u64::from(i));
            if !self.mem.contains_key(&a) {
                return Err(Fault::Unmapped(a));
            }
        }
        for i in 0..u32::from(size) {
            let a = addr.wrapping_add(u64::from(i));
            self.mem.insert(a, ((value >> (8 * i)) & 0xff) as u8);
        }
        Ok(())
    }

    /// Evaluate one flag from the lifter's expressions, or `None` when the
    /// instruction did not define it.
    pub fn flag(&self, which: Flag) -> Option<bool> {
        let expr = match which {
            Flag::Cf => self.flags.cf.as_ref(),
            Flag::Zf => self.flags.zf.as_ref(),
            Flag::Sf => self.flags.sf.as_ref(),
            Flag::Of => self.flags.of.as_ref(),
            Flag::Pf => self.flags.pf.as_ref(),
        }?;
        self.eval(expr).ok().map(|v| v != 0)
    }

    /// Evaluate an expression.
    pub fn eval(&self, e: &Expr) -> Result<u64, Fault> {
        match e {
            Expr::Const(v) => Ok(*v),
            Expr::Var(name) => self
                .vars
                .get(name)
                .copied()
                .ok_or_else(|| Fault::Unbound(name.clone())),
            Expr::Load { addr, size, signed } => {
                let a = self.eval(addr)?;
                self.load(a, *size, *signed)
            }
            Expr::Bin { op, lhs, rhs, bits } => {
                let a = self.eval(lhs)?;
                if *op == BinOp::Div || *op == BinOp::Mod {
                    let b = self.eval(rhs)?;
                    return div(op, a, b, *bits, true);
                }
                if *op == BinOp::UDiv || *op == BinOp::UMod {
                    let b = self.eval(rhs)?;
                    return div(op, a, b, *bits, false);
                }
                let b = self.eval(rhs)?;
                bin(*op, a, b, *bits)
            }
            Expr::Un { op, arg, bits } => {
                let v = self.eval(arg)?;
                let m = mask(*bits);
                Ok(match op {
                    UnOp::Neg => (!v).wrapping_add(1) & m,
                    UnOp::Not => !v & m,
                    UnOp::LNot => u64::from(v & m == 0),
                })
            }
            Expr::Cond { op, lhs, rhs, bits } => {
                // Both operands are evaluated before the comparison: a
                // comparison of two loads must fault the same way whichever
                // side is short-circuited.
                let a = self.eval(lhs)?;
                let b = self.eval(rhs)?;
                Ok(u64::from(cond(*op, a, b, *bits)))
            }
            Expr::Bool { op, args } => {
                let mut vals = Vec::with_capacity(args.len());
                for a in args {
                    vals.push(self.eval(a)? != 0);
                }
                Ok(u64::from(match op {
                    BoolOp::And => vals.iter().all(|v| *v),
                    BoolOp::Or => vals.iter().any(|v| *v),
                    BoolOp::Not => !vals.first().copied().unwrap_or(false),
                    BoolOp::Eq => {
                        let first = vals.first().copied().unwrap_or(false);
                        vals.iter().all(|v| *v == first)
                    }
                    BoolOp::Ne => {
                        let first = vals.first().copied().unwrap_or(false);
                        vals.iter().any(|v| *v != first)
                    }
                }))
            }
            Expr::FlagCond { cond, state, bits } => {
                // The condition is answered from the flag expressions the
                // lifter attached, evaluated after the instruction that set
                // them. A flag the instruction did not define is whatever it
                // was before, which this machine cannot know and refuses to
                // invent.
                let mut m = self.clone();
                m.flags = (**state).clone();
                let get = |which: Flag| -> Result<bool, Fault> {
                    let expr = match which {
                        Flag::Cf => m.flags.cf.as_ref(),
                        Flag::Zf => m.flags.zf.as_ref(),
                        Flag::Sf => m.flags.sf.as_ref(),
                        Flag::Of => m.flags.of.as_ref(),
                        Flag::Pf => m.flags.pf.as_ref(),
                    };
                    match expr {
                        Some(e) => m.eval(e).map(|v| v != 0),
                        None => Err(Fault::Unsupported(format!(
                            "{which:?} is read but was not defined by the preceding operation"
                        ))),
                    }
                };
                let cf = get(Flag::Cf);
                let zf = get(Flag::Zf);
                let sf = get(Flag::Sf);
                let of = get(Flag::Of);
                let pf = get(Flag::Pf);
                let need = |r: Result<bool, Fault>| r;
                let b = match cond {
                    FlagCond::Z => need(zf),
                    FlagCond::Nz => need(zf).map(|v| !v),
                    FlagCond::C => need(cf),
                    FlagCond::Nc => need(cf).map(|v| !v),
                    FlagCond::Po => need(pf),
                    FlagCond::Npo => need(pf).map(|v| !v),
                    FlagCond::S => need(sf),
                    FlagCond::Ns => need(sf).map(|v| !v),
                    FlagCond::O => need(of),
                    FlagCond::No => need(of).map(|v| !v),
                    FlagCond::L => Ok(need(sf)? != need(of)?),
                    FlagCond::Nl => Ok(need(sf)? == need(of)?),
                    FlagCond::G => Ok(!need(zf)? && need(sf)? == need(of)?),
                    FlagCond::Le => Ok(need(zf)? || need(sf)? != need(of)?),
                    FlagCond::A => Ok(!need(cf)? && !need(zf)?),
                    FlagCond::Be => Ok(need(cf)? || need(zf)?),
                };
                let _ = bits;
                Ok(u64::from(b?))
            }
            Expr::Cast {
                kind,
                arg,
                from,
                to,
            } => {
                let v = self.eval(arg)?;
                Ok(match kind {
                    CastKind::Trunc => v & mask(*to),
                    CastKind::Zext => v & mask(*from) & mask(*to),
                    CastKind::Sext => sign_extend(v & mask(*from), *from) & mask(*to),
                })
            }
            Expr::Select { cond, yes, no } => {
                if self.eval(cond)? != 0 {
                    self.eval(yes)
                } else {
                    self.eval(no)
                }
            }
            Expr::Phi(_) => Err(Fault::Unsupported(
                "phi survived into execution; the program was not in SSA-complete form".into(),
            )),
            Expr::Intrinsic { op, args, bits } => self.intrinsic(*op, args, *bits),
            Expr::Unknown(text) => Err(Fault::Unsupported(format!("unlifted: {text}"))),
        }
    }

    /// Evaluate an intrinsic.
    fn intrinsic(&self, op: Intrinsic, args: &[Expr], bits: u8) -> Result<u64, Fault> {
        let mut vals = Vec::with_capacity(args.len());
        for a in args {
            vals.push(self.eval(a)?);
        }
        let m = mask(bits);
        let width = u32::from(bits) * 8;
        let r = match op {
            Intrinsic::Rol | Intrinsic::Ror => {
                let v = vals.first().copied().unwrap_or(0) & m;
                let n = vals.get(1).copied().unwrap_or(0) as u32 % width.max(1);
                if n == 0 {
                    v
                } else if op == Intrinsic::Rol {
                    ((v << n) | (v >> (width - n))) & m
                } else {
                    ((v >> n) | (v << (width - n))) & m
                }
            }
            Intrinsic::Bswap => {
                let v = vals.first().copied().unwrap_or(0);
                let out = match bits {
                    2 => u64::from((v as u16).swap_bytes()),
                    4 => u64::from((v as u32).swap_bytes()),
                    8 => v.swap_bytes(),
                    _ => v,
                };
                out & m
            }
            Intrinsic::Bsf | Intrinsic::Tzcnt => {
                let v = vals.first().copied().unwrap_or(0) & m;
                if op == Intrinsic::Bsf && v == 0 {
                    // Undefined on the processor; keeping the destination
                    // unchanged is the documented behaviour of most cores.
                    vals.first().copied().unwrap_or(0)
                } else {
                    u64::from(v.trailing_zeros().min(width))
                }
            }
            Intrinsic::Bsr | Intrinsic::Lzcnt => {
                let v = vals.first().copied().unwrap_or(0) & m;
                if op == Intrinsic::Bsr && v == 0 {
                    vals.first().copied().unwrap_or(0)
                } else {
                    u64::from((width - 1).saturating_sub(v.leading_zeros().min(width)))
                }
            }
            Intrinsic::ParityEven => {
                let v = (vals.first().copied().unwrap_or(0) & 0xff).count_ones() as u8;
                u64::from(v % 2 == 0)
            }
            Intrinsic::Carry => {
                // The lifter passes the operands in the order that makes the
                // carry visible in the result: `Carry(result, previous)` is the
                // carry of `previous + x`, and it is set exactly when the sum
                // wrapped below the value that was there before.
                let result = vals.first().copied().unwrap_or(0) & m;
                let previous = vals.get(1).copied().unwrap_or(0) & m;
                u64::from(result < previous)
            }
            Intrinsic::Borrow => {
                // `Borrow(previous, operand)`.
                let previous = vals.first().copied().unwrap_or(0) & m;
                let operand = vals.get(1).copied().unwrap_or(0) & m;
                u64::from(previous < operand)
            }
            Intrinsic::OverflowAdd => {
                // The overflow is a fact about the sign bits, not about the
                // arithmetic: the result has the wrong sign when both inputs
                // had the other one. Computing it from the exact sum instead
                // would call `adc al, al` an overflow whenever the carry came
                // in, because the exact sum of the two inputs is one less than
                // the result the instruction produced.
                let a = vals.first().copied().unwrap_or(0) & m;
                let b = vals.get(1).copied().unwrap_or(0) & m;
                let s = vals.get(2).copied().unwrap_or_else(|| a.wrapping_add(b)) & m;
                let sign = 1u64 << (width - 1);
                u64::from((a ^ s) & (b ^ s) & sign != 0)
            }
            Intrinsic::OverflowSub => {
                let a = vals.first().copied().unwrap_or(0) & m;
                let b = vals.get(1).copied().unwrap_or(0) & m;
                let d = vals.get(2).copied().unwrap_or_else(|| a.wrapping_sub(b)) & m;
                let sign = 1u64 << (width - 1);
                u64::from((a ^ b) & (a ^ d) & sign != 0)
            }
            Intrinsic::MovMask => {
                let v = vals.first().copied().unwrap_or(0) & m;
                u64::from(signed(v, bits) < 0)
            }
            Intrinsic::MulLo | Intrinsic::MulHi => {
                let a = vals.first().copied().unwrap_or(0) & m;
                let b = vals.get(1).copied().unwrap_or(0) & m;
                let product = u128::from(a) * u128::from(b);
                if op == Intrinsic::MulLo {
                    (product as u64) & m
                } else {
                    ((product >> width) as u64) & m
                }
            }
            Intrinsic::AdcSum | Intrinsic::AdcCarry | Intrinsic::SbbDiff | Intrinsic::SbbBorrow => {
                let a = vals.first().copied().unwrap_or(0) & m;
                let b = vals.get(1).copied().unwrap_or(0) & m;
                let c = vals.get(2).copied().unwrap_or(0) & 1;
                match op {
                    Intrinsic::AdcSum => a.wrapping_add(b).wrapping_add(c) & m,
                    Intrinsic::AdcCarry => {
                        u64::from(u128::from(a) + u128::from(b) + u128::from(c) > u128::from(m))
                    }
                    Intrinsic::SbbDiff => a.wrapping_sub(b).wrapping_sub(c) & m,
                    _ => u64::from(u128::from(a) < u128::from(b) + u128::from(c)),
                }
            }
            Intrinsic::IMulLo | Intrinsic::IMulHi => {
                let a = i128::from(signed(vals.first().copied().unwrap_or(0) & m, bits));
                let b = i128::from(signed(vals.get(1).copied().unwrap_or(0) & m, bits));
                let product = a.wrapping_mul(b);
                if op == Intrinsic::IMulLo {
                    (product as u64) & m
                } else {
                    ((product >> width) as u64) & m
                }
            }
            Intrinsic::Rcl | Intrinsic::Rcr => {
                // The carry is a bit above the top of the value, so the
                // rotation happens in a field one bit wider than the operand
                // and the result is what is left below that bit.
                let value = vals.first().copied().unwrap_or(0) & m;
                let count = vals.get(1).copied().unwrap_or(0);
                let carry = vals.get(2).copied().unwrap_or(0) & 1;
                let span = width + 1;
                let n = (count as u32) % span;
                if n == 0 {
                    value
                } else {
                    let all = if span >= 128 {
                        u128::MAX
                    } else {
                        (1u128 << span) - 1
                    };
                    let wide = (u128::from(value) | (u128::from(carry) << width)) & all;
                    let turned = if op == Intrinsic::Rcl {
                        ((wide << n) | (wide >> (span - n))) & all
                    } else {
                        ((wide >> n) | (wide << (span - n))) & all
                    };
                    (turned as u64) & m
                }
            }
            Intrinsic::Div | Intrinsic::IDiv | Intrinsic::DivRem | Intrinsic::IDivRem => {
                // The dividend is a pair -- high word, low word -- followed by
                // the divisor. Dividing the first two arguments instead
                // computed `high / low`, which gave a wrong quotient and called
                // the low word a zero divisor whenever it happened to be zero.
                let (high, low, divisor) = if vals.len() >= 3 {
                    (vals[0], vals[1], vals[2])
                } else {
                    (
                        0,
                        vals.first().copied().unwrap_or(0),
                        vals.get(1).copied().unwrap_or(0),
                    )
                };
                if divisor == 0 {
                    return Err(Fault::DivideByZero);
                }
                let signed_op = matches!(op, Intrinsic::IDiv | Intrinsic::IDivRem);
                let span = width * 2;
                let raw = (u128::from(high & m) << width) | u128::from(low & m);
                let (q, r): (i128, i128) = if signed_op {
                    // The pair is one number of twice the operand width, and
                    // its top bit is the sign of that wider number, not of
                    // either half.
                    let dividend = if span >= 128 {
                        raw as i128
                    } else if (raw >> (span - 1)) & 1 == 1 {
                        (raw | (!0u128 << span)) as i128
                    } else {
                        raw as i128
                    };
                    let d = i128::from(signed(divisor, bits));
                    (dividend.wrapping_div(d), dividend.wrapping_rem(d))
                } else {
                    let d = u128::from(divisor & m);
                    ((raw / d) as u64 as i128, (raw % d) as u64 as i128)
                };
                match op {
                    // The remainder is its own result, not the high half of a
                    // packed pair. The lifter writes it into the second
                    // register, and packing it packed the quotient there
                    // instead.
                    Intrinsic::DivRem | Intrinsic::IDivRem => (r as u64) & m,
                    _ => (q as u64) & m,
                }
            }
        };
        Ok(r)
    }

    /// Execute one statement.
    pub fn step(&mut self, stmt: &Stmt) -> Result<(), Fault> {
        self.steps = self.steps.saturating_add(1);
        match stmt {
            Stmt::Assign { dest, value } => {
                let v = self.eval(value)?;
                self.vars.insert(dest.clone(), v);
            }
            Stmt::Store { addr, value, size } => {
                let a = self.eval(addr)?;
                let v = self.eval(value)?;
                self.store(a, *size, v)?;
            }
            Stmt::Call { name, ret, .. } => {
                // A call is where the model stops. The register that would hold
                // the result is left unbound so a later read faults instead of
                // producing a number the interpreter made up.
                if let Some(dest) = ret {
                    return Err(Fault::Unsupported(format!(
                        "call to {name} returns into {dest}"
                    )));
                }
            }
            Stmt::Comment(_) => {}
        }
        Ok(())
    }

    /// Execute a sequence of statements.
    pub fn run_stmts(&mut self, stmts: &[Stmt]) -> Result<(), Fault> {
        for s in stmts {
            self.step(s)?;
        }
        Ok(())
    }

    /// Execute a function from its entry block.
    ///
    /// The budget counts blocks, not statements: a loop of three instructions
    /// is one block and would otherwise be unbounded.
    pub fn run_program(&mut self, program: &crate::Program, budget: u64) -> Result<Outcome, Fault> {
        let mut at = program.entry;
        let mut taken = 0u64;
        loop {
            if taken >= budget {
                return Err(Fault::Budget);
            }
            taken += 1;
            let block = program
                .blocks
                .get(at)
                .ok_or_else(|| Fault::Unsupported(format!("block {at} does not exist")))?;
            self.run_stmts(&block.stmts)?;
            match &block.term {
                Term::Jump(next) => at = *next,
                Term::Branch {
                    cond,
                    then_bb,
                    else_bb,
                } => {
                    let next = if self.eval(cond)? != 0 {
                        then_bb
                    } else {
                        else_bb
                    };
                    match next {
                        Some(block) => at = *block,
                        // The other side of the branch left the function:
                        // there is no block here to continue in, and
                        // inventing one would run code the function does
                        // not hold.
                        None => {
                            return Ok(Outcome {
                                value: None,
                                blocks: taken,
                            });
                        }
                    }
                }
                Term::Switch {
                    index, targets, ..
                } => {
                    let i = self.eval(index)? as usize;
                    at = *targets
                        .get(i)
                        .ok_or_else(|| Fault::Unsupported(format!("switch index {i} out of range")))?;
                }
                Term::Ret(v) => {
                    let value = match v {
                        Some(e) => Some(self.eval(e)?),
                        None => None,
                    };
                    return Ok(Outcome {
                        value,
                        blocks: taken,
                    });
                }
                // A computed jump leaves the function: the interpreter has
                // no image to resume in, and guessing a target would be a
                // value the program never computed.
                Term::Unreachable | Term::Indirect { .. } => {
                    return Ok(Outcome {
                        value: None,
                        blocks: taken,
                    });
                }
            }
        }
    }
}

/// What a completed run produced.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
    /// The value in the return register, when the function returned one.
    pub value: Option<u64>,
    /// Blocks executed.
    pub blocks: u64,
}

/// One condition-code flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flag {
    /// Carry.
    Cf,
    /// Zero.
    Zf,
    /// Sign.
    Sf,
    /// Overflow.
    Of,
    /// Parity.
    Pf,
}

/// Sign-extend the low `size` bytes of `v`.
pub fn sign_extend(v: u64, size: u8) -> u64 {
    if size >= 8 {
        return v;
    }
    let bits = u32::from(size) * 8;
    let shift = 64 - bits;
    (((v << shift) as i64) >> shift) as u64
}

/// Read the low `size` bytes of `v` as a signed number.
fn signed(v: u64, size: u8) -> i64 {
    sign_extend(v, size) as i64
}

/// Arithmetic on masked widths.
fn bin(op: BinOp, a: u64, b: u64, bits: u8) -> Result<u64, Fault> {
    let m = mask(bits);
    let r = match op {
        BinOp::Add => a.wrapping_add(b) & m,
        BinOp::Sub => a.wrapping_sub(b) & m,
        BinOp::Mul => a.wrapping_mul(b) & m,
        BinOp::And => a & b & m,
        BinOp::Or => a | b & m,
        BinOp::Xor => (a ^ b) & m,
        BinOp::Shl => {
            // A shift is not a rotation: a count at or past the width pushes
            // every bit out and leaves zero. Taking the count modulo the width
            // instead turned `shl al, 9` into `shl al, 1`, which is the same
            // value only by coincidence and never the same carry. The caller
            // is responsible for the processor's own masking of the count,
            // because that rule differs between architectures.
            let n = b as u32;
            if n >= u32::from(bits) * 8 { 0 } else { (a << n) & m }
        }
        BinOp::Shr => {
            let n = b as u32;
            if n >= u32::from(bits) * 8 { 0 } else { (a & m) >> n }
        }
        BinOp::Sar => {
            let width = u32::from(bits) * 8;
            let n = b as u32;
            if n >= width {
                if signed(a, bits) < 0 { m } else { 0 }
            } else {
                ((signed(a, bits) >> n) as u64) & m
            }
        }
        BinOp::Div | BinOp::UDiv | BinOp::Mod | BinOp::UMod => unreachable!("handled by caller"),
        BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FMin | BinOp::FMax
        | BinOp::FEq | BinOp::FNe | BinOp::FLt | BinOp::FLe | BinOp::FGt | BinOp::FGe
        | BinOp::FUnordered | BinOp::I2F32 | BinOp::I2F64 | BinOp::F2I32 | BinOp::F2I64
        | BinOp::F2F32 | BinOp::F2F64 | BinOp::FSqrt | BinOp::I2F80 | BinOp::FNeg | BinOp::FAbs => {
            return float(op, a, b, bits);
        }
    };
    Ok(r)
}

/// Division, signed or not, with the processor's fault on a zero divisor.
fn div(op: &BinOp, a: u64, b: u64, bits: u8, is_signed: bool) -> Result<u64, Fault> {
    if b == 0 {
        return Err(Fault::DivideByZero);
    }
    let m = mask(bits);
    let r = if is_signed {
        let x = signed(a, bits);
        let y = signed(b, bits);
        let q = x.wrapping_div(y);
        let rem = x.wrapping_rem(y);
        match op {
            BinOp::Div => q as u64,
            _ => rem as u64,
        }
    } else {
        let x = a & m;
        let y = b & m;
        match op {
            BinOp::UDiv => x / y,
            _ => x % y,
        }
    };
    Ok(r & m)
}

/// Comparison, at the width and signedness the IR states.
fn cond(op: CondOp, a: u64, b: u64, bits: u8) -> bool {
    let m = mask(bits);
    let (x, y) = (a & m, b & m);
    match op {
        CondOp::Eq => x == y,
        CondOp::Ne => x != y,
        CondOp::Ult => x < y,
        CondOp::Ule => x <= y,
        CondOp::Ugt => x > y,
        CondOp::Uge => x >= y,
        CondOp::Slt => signed(x, bits) < signed(y, bits),
        CondOp::Sle => signed(x, bits) <= signed(y, bits),
        CondOp::Sgt => signed(x, bits) > signed(y, bits),
        CondOp::Sge => signed(x, bits) >= signed(y, bits),
    }
}

/// Floating-point operations, evaluated on the bit patterns.
fn float(op: BinOp, a: u64, b: u64, bits: u8) -> Result<u64, Fault> {
    let single = bits == 4;
    // x87's 80-bit format has no Rust type; f64 is the closest and the comment
    // at the top of the module says what that costs.
    let fa = if single {
        f64::from(f32::from_bits(a as u32))
    } else {
        f64::from_bits(a)
    };
    let fb = if single {
        f64::from(f32::from_bits(b as u32))
    } else {
        f64::from_bits(b)
    };
    let pack = |v: f64| -> u64 {
        if single {
            u64::from((v as f32).to_bits())
        } else {
            v.to_bits()
        }
    };
    let r = match op {
        BinOp::FAdd => pack(fa + fb),
        BinOp::FSub => pack(fa - fb),
        BinOp::FMul => pack(fa * fb),
        BinOp::FDiv => pack(fa / fb),
        BinOp::FMin => pack(fa.min(fb)),
        BinOp::FMax => pack(fa.max(fb)),
        BinOp::FSqrt => pack(fa.sqrt()),
        BinOp::FNeg => pack(-fa),
        BinOp::FAbs => pack(fa.abs()),
        BinOp::FEq => u64::from(fa == fb),
        BinOp::FNe => u64::from(fa != fb),
        BinOp::FLt => u64::from(fa < fb),
        BinOp::FLe => u64::from(fa <= fb),
        BinOp::FGt => u64::from(fa > fb),
        BinOp::FGe => u64::from(fa >= fb),
        BinOp::FUnordered => u64::from(fa.is_nan() || fb.is_nan()),
        BinOp::I2F32 => pack(signed(a, bits) as f64),
        BinOp::I2F64 => (signed(a, bits) as f64).to_bits(),
        BinOp::I2F80 => (signed(a, bits) as f64).to_bits(),
        BinOp::F2I32 => {
            let v = fa as i64;
            (v as u64) & 0xffff_ffff
        }
        BinOp::F2I64 => fa as i64 as u64,
        BinOp::F2F32 => u64::from((fa as f32).to_bits()),
        BinOp::F2F64 => fa.to_bits(),
        _ => return Err(Fault::Unsupported(format!("float op {op:?}"))),
    };
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m() -> Machine {
        let mut m = Machine::new();
        m.map(0x1000, &[0u8; 256]);
        m
    }

    #[test]
    fn addition_wraps_at_the_width_the_ir_states() {
        let mut m = m();
        m.set("a", 0xffff_ffff);
        m.set("b", 1);
        let e = Expr::Bin {
            op: BinOp::Add,
            lhs: Box::new(Expr::var("a")),
            rhs: Box::new(Expr::var("b")),
            bits: 4,
        };
        assert_eq!(m.eval(&e).unwrap(), 0);
    }

    #[test]
    fn a_signed_shift_fills_with_the_sign_bit() {
        let mut m = m();
        m.set("a", 0xffff_ffff_ffff_fff8);
        let e = Expr::Bin {
            op: BinOp::Sar,
            lhs: Box::new(Expr::var("a")),
            rhs: Box::new(Expr::Const(2)),
            bits: 8,
        };
        assert_eq!(m.eval(&e).unwrap(), 0xffff_ffff_ffff_fffe);
    }

    #[test]
    fn a_shift_past_the_width_answers_zero() {
        let mut m = m();
        m.set("a", 1);
        let e = Expr::Bin {
            op: BinOp::Shl,
            lhs: Box::new(Expr::var("a")),
            rhs: Box::new(Expr::Const(32)),
            bits: 4,
        };
        // Taking the count modulo the width would answer one, which is the
        // value unchanged: a rotation. A shift moves every bit out and leaves
        // zero, and the processor does the same once its own masking of the
        // count has left a count at or past the width.
        assert_eq!(m.eval(&e).unwrap(), 0);
    }

    #[test]
    fn a_sign_extending_load_keeps_the_sign() {
        let mut m = m();
        m.map(0x2000, &[0xff]);
        let e = Expr::Load {
            addr: Box::new(Expr::Const(0x2000)),
            size: 1,
            signed: true,
        };
        assert_eq!(m.eval(&e).unwrap(), u64::MAX);
    }

    #[test]
    fn reading_unmapped_memory_is_a_fault_not_a_zero() {
        let m = m();
        let e = Expr::Load {
            addr: Box::new(Expr::Const(0x9999)),
            size: 4,
            signed: false,
        };
        assert_eq!(m.eval(&e), Err(Fault::Unmapped(0x9999)));
    }

    #[test]
    fn a_partial_store_leaves_memory_untouched() {
        let mut m = m();
        m.map(0x3000, &[1, 2]);
        let e = Stmt::Store {
            addr: Expr::Const(0x3000),
            value: Expr::Const(0xdead_beef),
            size: 4,
        };
        assert!(m.step(&e).is_err());
        assert_eq!(m.peek(0x3000, 2).unwrap(), vec![1, 2]);
    }

    #[test]
    fn division_by_zero_is_an_error() {
        let mut m = m();
        m.set("a", 10);
        let e = Expr::Bin {
            op: BinOp::UDiv,
            lhs: Box::new(Expr::var("a")),
            rhs: Box::new(Expr::Const(0)),
            bits: 8,
        };
        assert_eq!(m.eval(&e), Err(Fault::DivideByZero));
    }

    #[test]
    fn a_signed_division_rounds_towards_zero() {
        let mut m = m();
        m.set("a", (-7i64) as u64);
        m.set("b", 2);
        let e = Expr::Bin {
            op: BinOp::Div,
            lhs: Box::new(Expr::var("a")),
            rhs: Box::new(Expr::var("b")),
            bits: 8,
        };
        assert_eq!(m.eval(&e).unwrap() as i64, -3);
    }

    #[test]
    fn comparisons_read_the_width_they_were_given() {
        let mut m = m();
        m.set("a", 0xffff_ffff);
        let e = Expr::Cond {
            op: CondOp::Slt,
            lhs: Box::new(Expr::var("a")),
            rhs: Box::new(Expr::Const(0)),
            bits: 4,
        };
        assert_eq!(m.eval(&e).unwrap(), 1);
        let e = Expr::Cond {
            op: CondOp::Slt,
            lhs: Box::new(Expr::var("a")),
            rhs: Box::new(Expr::Const(0)),
            bits: 8,
        };
        assert_eq!(m.eval(&e).unwrap(), 0);
    }

    #[test]
    fn a_rotate_moves_bits_across_the_boundary() {
        let mut m = m();
        m.set("a", 0x8000_0001);
        let e = Expr::Intrinsic {
            op: Intrinsic::Rol,
            args: vec![Expr::var("a"), Expr::Const(1)],
            bits: 4,
        };
        assert_eq!(m.eval(&e).unwrap(), 3);
    }

    #[test]
    fn parity_counts_the_low_byte() {
        let mut m = m();
        m.set("a", 0b1011);
        let e = Expr::Intrinsic {
            op: Intrinsic::ParityEven,
            args: vec![Expr::var("a")],
            bits: 1,
        };
        // Three bits set is odd parity, so PF is clear.
        assert_eq!(m.eval(&e).unwrap(), 0);
        // The high nibble does not count: parity looks at the low byte only.
        m.set("a", 0b1011_0000);
        assert_eq!(m.eval(&e).unwrap(), 0);
        m.set("a", 0b11);
        assert_eq!(m.eval(&e).unwrap(), 1);
    }

    #[test]
    fn an_overflow_flag_reads_the_signed_result() {
        let mut m = m();
        m.set("a", i64::MAX as u64);
        m.set("b", 1);
        let e = Expr::Intrinsic {
            op: Intrinsic::OverflowAdd,
            args: vec![Expr::var("a"), Expr::var("b")],
            bits: 8,
        };
        assert_eq!(m.eval(&e).unwrap(), 1);
    }

    #[test]
    fn a_flag_condition_reads_the_state_it_carries() {
        let mut m = m();
        m.set("z", 0);
        let state = FlagState {
            cf: Some(Expr::Const(0)),
            zf: Some(Expr::var("z")),
            sf: Some(Expr::Const(0)),
            of: Some(Expr::Const(0)),
            pf: Some(Expr::Const(1)),
        };
        let e = Expr::FlagCond {
            cond: FlagCond::Nz,
            state: Box::new(state),
            bits: 4,
        };
        assert_eq!(m.eval(&e).unwrap(), 1);
    }

    #[test]
    fn a_condition_on_an_undefined_flag_is_refused() {
        let m = m();
        let state = FlagState {
            cf: None,
            zf: None,
            sf: None,
            of: None,
            pf: None,
        };
        let e = Expr::FlagCond {
            cond: FlagCond::Z,
            state: Box::new(state),
            bits: 4,
        };
        assert!(matches!(m.eval(&e), Err(Fault::Unsupported(_))));
    }
}
