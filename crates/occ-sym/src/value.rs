//! Values during execution: a machine word that is either known or an
//! expression over the function's inputs.
//!
//! There is no third state. "Unknown" is not a value here, because the IR it
//! comes from has no such thing: a read the lifter could not model is already
//! an [`Expr::Unknown`] or a bare parameter name, and treating either as a
//! symbolic input keeps the analysis honest — the result then says "this
//! depends on `rdi`", which is exactly what the bytes say.

use occ_decompile::ir::{
    BinOp, BoolOp, CastKind, CondOp, Expr, Intrinsic, UnOp, holds_at, sign_extend, truncate,
};

/// One word.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// Fully determined.
    Const(u64),
    /// An expression over symbolic inputs.
    Sym(Expr),
}

impl Value {
    /// A known word.
    pub const fn num(v: u64) -> Value {
        Value::Const(v)
    }

    /// Wrap an expression, folding it to a constant when it is one.
    pub fn expr(e: Expr) -> Value {
        match e {
            Expr::Const(v) => Value::Const(v),
            other => Value::Sym(other),
        }
    }

    /// The constant, when the value is known.
    pub const fn as_const(&self) -> Option<u64> {
        match self {
            Value::Const(v) => Some(*v),
            Value::Sym(_) => None,
        }
    }

    /// True when this is a constant.
    pub const fn is_const(&self) -> bool {
        matches!(self, Value::Const(_))
    }

    /// The value as an expression, for building bigger ones.
    pub fn to_expr(&self) -> Expr {
        match self {
            Value::Const(v) => Expr::Const(*v),
            Value::Sym(e) => e.clone(),
        }
    }

    /// Binary operation with the machine's width. Constant operands are folded
    /// through the same code the simplifier uses, so an executed expression and
    /// a simplified one can never disagree.
    pub fn bin(op: BinOp, lhs: &Value, rhs: &Value, bytes: u8) -> Value {
        let e = Expr::Bin {
            op,
            lhs: Box::new(lhs.to_expr()),
            rhs: Box::new(rhs.to_expr()),
            bits: bytes,
        };
        Value::expr(e.simplify())
    }

    /// Unary operation.
    pub fn un(op: UnOp, arg: &Value, bytes: u8) -> Value {
        let e = Expr::Un {
            op,
            arg: Box::new(arg.to_expr()),
            bits: bytes,
        };
        Value::expr(e.simplify())
    }

    /// Width conversion, applied to the constant when there is one.
    pub fn cast(kind: CastKind, arg: &Value, from: u8, to: u8) -> Value {
        if let Value::Const(v) = arg {
            let narrowed = truncate(*v, usize::from(from.min(to)));
            let out = match kind {
                CastKind::Sext => sign_extend(narrowed, from.min(to)) as u64,
                CastKind::Zext | CastKind::Trunc => narrowed,
            };
            return Value::Const(truncate(out, usize::from(to)));
        }
        let e = Expr::Cast {
            kind,
            arg: Box::new(arg.to_expr()),
            from,
            to,
        };
        Value::expr(e.simplify())
    }

    /// Comparison, producing 0 or 1 when both sides are known.
    ///
    /// The width decides how the operands are read. The same bits read as a
    /// signed number and as an unsigned number are two different numbers,
    /// and the branch that consumes the comparison has to know which one it
    /// asked about.
    pub fn compare(op: CondOp, lhs: &Value, rhs: &Value, bits: u8) -> Value {
        if let (Some(a), Some(b)) = (lhs.as_const(), rhs.as_const()) {
            return Value::Const(u64::from(holds_at(op, a, b, bits)));
        }
        Value::expr(
            Expr::Cond {
                op,
                lhs: Box::new(lhs.to_expr()),
                rhs: Box::new(rhs.to_expr()),
                bits,
            }
            .simplify(),
        )
    }

    /// A boolean combination of values.
    ///
    /// The combination is a value like any other: zero or one when every
    /// operand is known, and an expression over the inputs otherwise.
    pub fn bool_combination(op: BoolOp, args: &[Value]) -> Value {
        let known: Option<Vec<bool>> = args.iter().map(|a| a.truth()).collect();
        if let Some(values) = known {
            return Value::Const(u64::from(match op {
                BoolOp::And => values.iter().all(|v| *v),
                BoolOp::Or => values.iter().any(|v| *v),
                BoolOp::Not => !values[0],
                BoolOp::Eq => values[0] == values[1],
                BoolOp::Ne => values[0] != values[1],
            }));
        }
        Value::expr(
            Expr::Bool {
                op,
                args: args.iter().map(|a| a.to_expr()).collect(),
            }
            .simplify(),
        )
    }

    /// Truth of a value used as a branch condition: nonzero means true.
    ///
    /// `None` when the answer depends on an input.
    pub fn truth(&self) -> Option<bool> {
        match self {
            Value::Const(v) => Some(*v != 0),
            Value::Sym(Expr::Cond {
                op,
                lhs,
                rhs,
                bits,
            }) => {
                let a = const_of(lhs)?;
                let b = const_of(rhs)?;
                Some(holds_at(*op, a, b, *bits))
            }
            Value::Sym(_) => None,
        }
    }
}

fn const_of(e: &Expr) -> Option<u64> {
    match e {
        Expr::Const(v) => Some(*v),
        _ => None,
    }
}

/// Evaluate a comparison on two machine words, honouring signedness.
///
/// The width is the machine word, so the operands are read as 64-bit
/// numbers. A comparison of narrower values has to say so.
pub fn holds(op: CondOp, a: u64, b: u64) -> bool {
    holds_at(op, a, b, 8)
}

/// Evaluate an expression that must produce a boolean.
///
/// `env` maps names to words; a name missing from `env` evaluates to itself,
/// which is how a parameter that was never constrained shows up. Returning
/// `None` means the expression depends on something this evaluator cannot see
/// — a memory read, or an instruction the lifter could not model.
pub fn eval_bool(e: &Expr, env: &std::collections::HashMap<String, u64>) -> Option<bool> {
    match e {
        Expr::Const(v) => Some(*v != 0),
        Expr::Var(name) => Some(*env.get(name).unwrap_or(&0) != 0),
        Expr::Cond {
            op,
            lhs,
            rhs,
            bits,
        } => {
            let a = eval_word(lhs, env)?;
            let b = eval_word(rhs, env)?;
            Some(holds_at(*op, a, b, *bits))
        }
        Expr::Bool { op, args } => {
            let values: Option<Vec<bool>> = args
                .iter()
                .map(|a| eval_bool(a, env))
                .collect();
            let values = values?;
            Some(match op {
                BoolOp::And => values.iter().all(|v| *v),
                BoolOp::Or => values.iter().any(|v| *v),
                BoolOp::Not => !values[0],
                BoolOp::Eq => values[0] == values[1],
                BoolOp::Ne => values[0] != values[1],
            })
        }
        Expr::FlagCond { cond, state, .. } => {
            let e = state.condition(*cond)?;
            eval_bool(&e, env)
        }
        Expr::Un {
            op: UnOp::LNot,
            arg,
            ..
        } => Some(!eval_bool(arg, env)?),
        Expr::Un { .. } => Some(eval_word(e, env)? != 0),
        Expr::Bin { .. } => Some(eval_word(e, env)? != 0),
        Expr::Cast { .. } => Some(eval_word(e, env)? != 0),
        Expr::Select { cond, yes, no } => {
            if eval_bool(cond, env)? {
                eval_bool(yes, env)
            } else {
                eval_bool(no, env)
            }
        }
        Expr::Load { .. } | Expr::Member { .. } | Expr::Phi(_) | Expr::Unknown(_) => None,
        Expr::Intrinsic { .. } => None,
    }
}

/// Evaluate an expression to a word under a concrete assignment.
///
/// Only used to check a candidate model, so an unbound name defaults to zero:
/// the model is the whole assignment, and anything it does not mention is a
/// choice the caller did not care about.
pub fn eval_word(e: &Expr, env: &std::collections::HashMap<String, u64>) -> Option<u64> {
    match e {
        Expr::Const(v) => Some(*v),
        Expr::Var(name) => Some(*env.get(name).unwrap_or(&0)),
        Expr::Bin { op, lhs, rhs, bits } => {
            let a = eval_word(lhs, env)?;
            let b = eval_word(rhs, env)?;
            Some(fold_bin(*op, a, b, *bits)?)
        }
        Expr::Un { op, arg, bits } => {
            let a = eval_word(arg, env)?;
            let v = match op {
                UnOp::Neg => a.wrapping_neg(),
                UnOp::Not => !a,
                UnOp::LNot => u64::from(a == 0),
            };
            Some(truncate(v, usize::from(*bits)))
        }
        Expr::Cond {
            op,
            lhs,
            rhs,
            bits,
        } => {
            let a = eval_word(lhs, env)?;
            let b = eval_word(rhs, env)?;
            Some(u64::from(holds_at(*op, a, b, *bits)))
        }
        Expr::Bool { op, args } => {
            let values: Option<Vec<bool>> = args
                .iter()
                .map(|a| eval_bool(a, env))
                .collect();
            let values = values?;
            Some(u64::from(match op {
                BoolOp::And => values.iter().all(|v| *v),
                BoolOp::Or => values.iter().any(|v| *v),
                BoolOp::Not => !values[0],
                BoolOp::Eq => values[0] == values[1],
                BoolOp::Ne => values[0] != values[1],
            }))
        }
        Expr::FlagCond { cond, state, .. } => {
            let e = state.condition(*cond)?;
            Some(u64::from(eval_bool(&e, env)?))
        }
        Expr::Cast {
            kind,
            arg,
            from,
            to,
        } => {
            let v = eval_word(arg, env)?;
            let narrowed = truncate(v, usize::from((*from).min(*to)));
            Some(match kind {
                CastKind::Sext => truncate(sign_extend(narrowed, *from) as u64, usize::from(*to)),
                CastKind::Zext | CastKind::Trunc => {
                    truncate(narrowed, usize::from(*to))
                }
            })
        }
        Expr::Select { cond, yes, no } => {
            if eval_word(cond, env)? != 0 {
                eval_word(yes, env)
            } else {
                eval_word(no, env)
            }
        }
        Expr::Load { .. } | Expr::Member { .. } | Expr::Phi(_) | Expr::Unknown(_) => None,
        Expr::Intrinsic { op, args, bits } => {
            let values: Option<Vec<u64>> = args.iter().map(|a| eval_word(a, env)).collect();
            eval_intrinsic(*op, &values?, *bits)
        }
    }
}

/// Concrete semantics of an intrinsic.
///
/// The operations the machine defines on flags have a closed form, and the
/// others have none the evaluator can use.
fn eval_intrinsic(op: Intrinsic, args: &[u64], bits: u8) -> Option<u64> {
    let bytes = bits.clamp(1, 8);
    let width = u32::from(bytes) * 8;
    let mask = mask_of(bytes);
    let value = args.first().copied().unwrap_or(0) & mask;
    Some(match op {
        Intrinsic::ParityEven => u64::from((value as u8).count_ones() % 2 == 0),
        // The sum did not fit when the result is smaller than the value
        // that was there before.
        Intrinsic::Carry => {
            let previous = args.get(1).copied().unwrap_or(0) & mask;
            u64::from(value < previous)
        }
        // The difference did not fit when the minuend was smaller than the
        // value subtracted from it.
        Intrinsic::Borrow => {
            let previous = args.get(1).copied().unwrap_or(0) & mask;
            u64::from(value < previous)
        }
        Intrinsic::OverflowAdd => {
            let a = args.first().copied().unwrap_or(0) & mask;
            let b = args.get(1).copied().unwrap_or(0) & mask;
            let r = args.get(2).copied().unwrap_or(0) & mask;
            u64::from((a ^ r) & (b ^ r) & (1u64 << (width - 1)) != 0)
        }
        Intrinsic::OverflowSub => {
            let a = args.first().copied().unwrap_or(0) & mask;
            let b = args.get(1).copied().unwrap_or(0) & mask;
            let r = args.get(2).copied().unwrap_or(0) & mask;
            u64::from((a ^ b) & (a ^ r) & (1u64 << (width - 1)) != 0)
        }
        _ => return None,
    })
}

/// Concrete semantics of one binary operation, mirroring `ir::fold`.
fn fold_bin(op: BinOp, a: u64, b: u64, bytes: u8) -> Option<u64> {
    if op.is_float() {
        return fold_float(op, a, b, bytes);
    }
    let width = u32::from(bytes.clamp(1, 8)) * 8;
    let v = match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::Div => (a as i64).checked_div(b as i64)? as u64,
        BinOp::UDiv => a.checked_div(b)?,
        BinOp::Mod => (a as i64).checked_rem(b as i64)? as u64,
        BinOp::UMod => a.checked_rem(b)?,
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        BinOp::Shl => {
            if b >= u64::from(width) {
                0
            } else {
                a << b
            }
        }
        BinOp::Shr => {
            if b >= u64::from(width) {
                0
            } else {
                (truncate(a, usize::from(bytes)) & mask_of(bytes)) >> b
            }
        }
        BinOp::Sar => {
            if b >= u64::from(width) {
                if (a as i64) < 0 { u64::MAX } else { 0 }
            } else {
                let signed = sign_extend(truncate(a, usize::from(bytes)), bytes);
                ((signed >> b) as u64) & mask_of(bytes)
            }
        }
        // Handled before the match: the floating-point operations have their
        // own semantics and their own width.
        _ => return None,
    };
    Some(truncate(v, usize::from(bytes)))
}

const fn mask_of(bytes: u8) -> u64 {
    if bytes >= 8 {
        u64::MAX
    } else {
        (1u64 << (bytes as u32 * 8)) - 1
    }
}

/// Concrete semantics of one floating-point operation.
///
/// The operands are bit patterns: four bytes is a `float`, eight a `double`.
/// The comparisons are the ordered ones, so a not-a-number makes them false
/// — which is what the machine's compare instruction reports.
fn fold_float(op: BinOp, a: u64, b: u64, bytes: u8) -> Option<u64> {
    match bytes {
        4 => {
            let x = f32::from_bits(a as u32);
            let y = f32::from_bits(b as u32);
            Some(match op {
                BinOp::FAdd => u64::from((x + y).to_bits()),
                BinOp::FSub => u64::from((x - y).to_bits()),
                BinOp::FMul => u64::from((x * y).to_bits()),
                BinOp::FDiv => u64::from((x / y).to_bits()),
                BinOp::FEq => u64::from(x.partial_cmp(&y) == Some(std::cmp::Ordering::Equal)),
                BinOp::FNe => u64::from(matches!(
                    x.partial_cmp(&y),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Greater)
                )),
                BinOp::FLt => u64::from(x.partial_cmp(&y) == Some(std::cmp::Ordering::Less)),
                BinOp::FLe => u64::from(matches!(
                    x.partial_cmp(&y),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                )),
                BinOp::FGt => u64::from(x.partial_cmp(&y) == Some(std::cmp::Ordering::Greater)),
                BinOp::FGe => u64::from(matches!(
                    x.partial_cmp(&y),
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                )),
                BinOp::FUnordered => u64::from(x.is_nan() || y.is_nan()),
                BinOp::FSqrt => u64::from(x.sqrt().to_bits()),
                BinOp::I2F32 => u64::from((sign_extend(a, bytes) as f32).to_bits()),
                BinOp::I2F64 => (sign_extend(a, bytes) as f64).to_bits(),
                BinOp::F2I32 => u64::from((x as i32) as u32),
                BinOp::F2I64 => (x as i64) as u64,
                BinOp::F2U32 => u64::from(x as u32),
                BinOp::F2U64 => x as u64,
                BinOp::F2F32 => u64::from(x.to_bits()),
                BinOp::F2F64 => (x as f64).to_bits(),
                BinOp::FNeg => u64::from((-x).to_bits()),
                BinOp::FAbs => u64::from(x.abs().to_bits()),
                BinOp::FCeil => u64::from(x.ceil().to_bits()),
                BinOp::FFloor => u64::from(x.floor().to_bits()),
                BinOp::FRound => u64::from(x.round_ties_even().to_bits()),
                // An 80-bit value does not fit a machine word.
                _ => return None,
            })
        }
        8 => {
            let x = f64::from_bits(a);
            let y = f64::from_bits(b);
            Some(match op {
                BinOp::FAdd => (x + y).to_bits(),
                BinOp::FSub => (x - y).to_bits(),
                BinOp::FMul => (x * y).to_bits(),
                BinOp::FDiv => (x / y).to_bits(),
                BinOp::FEq => u64::from(x.partial_cmp(&y) == Some(std::cmp::Ordering::Equal)),
                BinOp::FNe => u64::from(matches!(
                    x.partial_cmp(&y),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Greater)
                )),
                BinOp::FLt => u64::from(x.partial_cmp(&y) == Some(std::cmp::Ordering::Less)),
                BinOp::FLe => u64::from(matches!(
                    x.partial_cmp(&y),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                )),
                BinOp::FGt => u64::from(x.partial_cmp(&y) == Some(std::cmp::Ordering::Greater)),
                BinOp::FGe => u64::from(matches!(
                    x.partial_cmp(&y),
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                )),
                BinOp::FUnordered => u64::from(x.is_nan() || y.is_nan()),
                BinOp::FSqrt => x.sqrt().to_bits(),
                BinOp::I2F32 => u64::from((sign_extend(a, bytes) as f32).to_bits()),
                BinOp::I2F64 => (sign_extend(a, bytes) as f64).to_bits(),
                BinOp::F2I32 => u64::from((x as i32) as u32),
                BinOp::F2I64 => (x as i64) as u64,
                BinOp::F2U32 => u64::from(x as u32),
                BinOp::F2U64 => x as u64,
                BinOp::F2F32 => u64::from((x as f32).to_bits()),
                BinOp::F2F64 => x.to_bits(),
                BinOp::FNeg => (-x).to_bits(),
                BinOp::FAbs => x.abs().to_bits(),
                BinOp::FCeil => x.ceil().to_bits(),
                BinOp::FFloor => x.floor().to_bits(),
                BinOp::FRound => x.round_ties_even().to_bits(),
                // An 80-bit value does not fit a machine word.
                _ => return None,
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_fold_through_the_simplifier() {
        let a = Value::num(3);
        let b = Value::num(4);
        assert_eq!(Value::bin(BinOp::Mul, &a, &b, 8), Value::num(12));
        let big = Value::num(0x0102_0304_0506_0708);
        let shift = Value::num(8);
        assert_eq!(
            Value::bin(BinOp::Shr, &big, &shift, 8),
            Value::num(0x0001_0203_0405_0607)
        );
    }

    #[test]
    fn casts_narrow_then_extend() {
        let v = Value::num(0xffff_ffff_ffff_fffe);
        assert_eq!(Value::cast(CastKind::Trunc, &v, 8, 4), Value::num(0xffff_fffe));
        assert_eq!(
            Value::cast(CastKind::Sext, &v, 1, 8),
            Value::num(0xffff_ffff_ffff_fffe)
        );
        assert_eq!(Value::cast(CastKind::Zext, &v, 1, 8), Value::num(0xfe));
    }

    #[test]
    fn comparisons_respect_signedness() {
        let a = Value::num(0xffff_ffff_ffff_ffff);
        let b = Value::num(1);
        assert_eq!(Value::compare(CondOp::Slt, &a, &b, 8), Value::num(1));
        assert_eq!(Value::compare(CondOp::Ult, &a, &b, 8), Value::num(0));
    }

    #[test]
    fn symbolic_values_stay_symbolic_until_forced() {
        let a = Value::expr(Expr::var("rdi"));
        let v = Value::bin(BinOp::Add, &a, &Value::num(1), 8);
        assert_eq!(v.truth(), None);
        assert_eq!(v.to_expr().to_string(), "rdi + 0x1");
    }
}

#[cfg(test)]
mod flag_tests {
    use super::*;
    use occ_decompile::ir::Intrinsic;
    use std::collections::HashMap;

    fn empty() -> HashMap<String, u64> {
        HashMap::new()
    }

    #[test]
    fn a_narrow_comparison_reads_narrow_values() {
        // 0x80000000 is a negative number as a 32-bit value and a large
        // positive one as a 64-bit value, and the comparison has to know
        // which it is asking about.
        assert!(holds_at(CondOp::Slt, 0x8000_0000, 0, 4));
        assert!(!holds_at(CondOp::Slt, 0x8000_0000, 0, 8));
        // The unsigned reading of the same bits is a large positive number,
        // and the width decides whether the bits above the low half count.
        assert!(holds_at(CondOp::Uge, 0x1_0000_0000, 1, 8));
        assert!(!holds_at(CondOp::Uge, 0x1_0000_0000, 1, 4));
    }

    #[test]
    fn the_carry_is_set_when_the_sum_did_not_fit() {
        let carry = |result: u64, previous: u64| Expr::Intrinsic {
            op: Intrinsic::Carry,
            args: vec![Expr::Const(result), Expr::Const(previous)],
            bits: 4,
        };
        // 0xffffffff + 1 does not fit in 32 bits, and the result is
        // smaller than the value that was there before.
        assert_eq!(eval_word(&carry(0, 0xffff_ffff), &empty()), Some(1));
        // 5 + 3 fits.
        assert_eq!(eval_word(&carry(8, 5), &empty()), Some(0));
    }

    #[test]
    fn the_borrow_is_set_when_the_minuend_was_smaller() {
        let borrow = |minuend: u64, subtrahend: u64| Expr::Intrinsic {
            op: Intrinsic::Borrow,
            args: vec![Expr::Const(minuend), Expr::Const(subtrahend)],
            bits: 4,
        };
        assert_eq!(eval_word(&borrow(3, 5), &empty()), Some(1));
        assert_eq!(eval_word(&borrow(5, 3), &empty()), Some(0));
    }

    #[test]
    fn the_overflow_is_set_when_the_sign_changed() {
        let overflow_add = |a: u64, b: u64, r: u64| Expr::Intrinsic {
            op: Intrinsic::OverflowAdd,
            args: vec![Expr::Const(a), Expr::Const(b), Expr::Const(r)],
            bits: 4,
        };
        let overflow_sub = |a: u64, b: u64, r: u64| Expr::Intrinsic {
            op: Intrinsic::OverflowSub,
            args: vec![Expr::Const(a), Expr::Const(b), Expr::Const(r)],
            bits: 4,
        };
        // 0x7fffffff + 1 does not fit in a signed 32-bit number.
        assert_eq!(
            eval_word(&overflow_add(0x7fff_ffff, 1, 0x8000_0000), &empty()),
            Some(1)
        );
        // 5 + 3 fits.
        assert_eq!(eval_word(&overflow_add(5, 3, 8), &empty()), Some(0));
        // 0x80000000 - 1 does not fit either.
        assert_eq!(
            eval_word(&overflow_sub(0x8000_0000, 1, 0x7fff_ffff), &empty()),
            Some(1)
        );
        assert_eq!(eval_word(&overflow_sub(5, 3, 2), &empty()), Some(0));
    }

    #[test]
    fn the_parity_flag_counts_the_set_bits_of_the_low_byte() {
        let parity = |v: u64| Expr::Intrinsic {
            op: Intrinsic::ParityEven,
            args: vec![Expr::Const(v)],
            bits: 4,
        };
        // Three set bits is odd parity.
        assert_eq!(eval_word(&parity(0b111), &empty()), Some(0));
        // Two set bits is even parity.
        assert_eq!(eval_word(&parity(0b110), &empty()), Some(1));
    }
}
