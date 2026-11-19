//! The intermediate representation the decompiler works on.
//!
//! Three layers, each a value that can be inspected on its own:
//!
//! * [`Expr`] and [`Stmt`] — a per-instruction lift with explicit data flow.
//! * [`Block`]/[`Term`] — statements grouped by basic block, with real
//!   successors instead of fall-through.
//! * [`Node`] — structured control flow (`if`, `while`, `switch`) plus plain
//!   assignments, which is what the renderer prints.
//!
//! The IR is deliberately untyped. Every value is a machine word of a known
//! *width*, and the width survives into the output (`(u32)`, `u64` casts), but
//! there is no struct or pointer type: those come from DWARF when available and
//! from nothing otherwise. Guessing them would produce output that looks more
//! confident than the evidence allows.

use std::fmt;

use occ_core::Addr;

/// Index of a basic block within one function.
pub type BlockId = usize;

/// Binary operators, with signedness where it changes the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Signed division.
    Div,
    /// Unsigned division.
    UDiv,
    /// Signed remainder.
    Mod,
    /// Unsigned remainder.
    UMod,
    /// Bitwise and.
    And,
    /// Bitwise or.
    Or,
    /// Bitwise exclusive or.
    Xor,
    /// Shift left.
    Shl,
    /// Logical shift right.
    Shr,
    /// Arithmetic shift right.
    Sar,
    /// Floating-point addition.
    ///
    /// The floating-point operations are separate from the integer ones
    /// because the same symbol means two different things: `+` on two words
    /// is integer addition, and `+` on two bit patterns that happen to be
    /// floating-point values is not. Keeping them apart is what lets the
    /// evaluator do the right thing with each.
    FAdd,
    /// Floating-point subtraction.
    FSub,
    /// Floating-point multiplication.
    FMul,
    /// Floating-point division.
    FDiv,
    /// Floating-point equality, false when either operand is not a number.
    ///
    /// This is the *ordered* equality: a comparison against a not-a-number
    /// is false, which is what the machine's compare instruction reports.
    FEq,
    /// Floating-point inequality, false when either operand is not a number.
    FNe,
    /// Floating-point less-than, false when either operand is not a number.
    FLt,
    /// Floating-point less-or-equal, false when either operand is not a number.
    FLe,
    /// Floating-point greater-than, false when either operand is not a number.
    FGt,
    /// Floating-point greater-or-equal, false when either operand is not a number.
    FGe,
    /// The smaller of two floating-point values.
    ///
    /// The machine's minimum is not a comparison: it returns the second
    /// operand when either one is not a number, and when both are zero.
    /// That is what `fold` reproduces.
    FMin,
    /// The larger of two floating-point values.
    ///
    /// The machine's maximum has the same rule as the minimum: the second
    /// operand wins when either one is not a number, and when both are
    /// zero.
    FMax,
    /// True when either operand is not a number.
    ///
    /// The machine's compare sets three flags, and a branch can ask about
    /// any of them: "below" is *less than or unordered*, "not below" is
    /// *greater or equal and ordered*. Saying "unordered" on its own is
    /// what lets both be written down.
    FUnordered,
    /// Convert an integer to a single-precision float.
    ///
    /// The width of the source is the operation's width; the result is
    /// four bytes.
    I2F32,
    /// Convert an integer to a double-precision float.
    ///
    /// The width of the source is the operation's width; the result is
    /// eight bytes.
    I2F64,
    /// Convert a float to a 32-bit integer, truncating towards zero.
    ///
    /// The width of the source is the operation's width; the result is
    /// four bytes.
    F2I32,
    /// Convert a float to a 64-bit integer, truncating towards zero.
    ///
    /// The width of the source is the operation's width; the result is
    /// eight bytes.
    F2I64,
    /// Convert a float to a 32-bit integer read as unsigned.
    ///
    /// The same truncation as [`BinOp::F2I32`], but the float is read as a
    /// number rather than as a signed one: a value of two billion is a
    /// number, not a negative one. The two are different operations and
    /// the machine has both.
    F2U32,
    /// Convert a float to a 64-bit integer read as unsigned.
    ///
    /// The same truncation as [`BinOp::F2I64`], with the float read as an
    /// unsigned number.
    F2U64,
    /// Convert a float to a single-precision float.
    ///
    /// The width of the source is the operation's width; the result is
    /// four bytes.
    F2F32,
    /// Convert a float to a double-precision float.
    ///
    /// The width of the source is the operation's width; the result is
    /// eight bytes.
    F2F64,
    /// Square root.
    ///
    /// The width of the source is the operation's width, and the result has
    /// the same width.
    FSqrt,
    /// Convert an integer to an 80-bit extended-precision float.
    ///
    /// The x87 stack holds extended-precision values, which are ten bytes
    /// wide and do not fit the machine word this IR is built on. The
    /// operation is therefore kept as a call: the value it produces is
    /// opaque, and pretending otherwise would put a wrong number in the
    /// output.
    I2F80,
    /// Negate a floating-point value.
    ///
    /// The x87 `fchs` negates an extended-precision value, whose sign bit
    /// is the top bit of a ten-byte number. Flipping a bit of a machine
    /// word would be a different operation, so this is a call.
    FNeg,
    /// Take the absolute value of a floating-point value.
    ///
    /// The x87 `fabs` clears the sign of an extended-precision value.
    /// Masking a machine word would clear a bit that is not the sign, so
    /// this is a call.
    FAbs,
    /// The smallest whole number not below a floating-point value.
    ///
    /// Rounding towards positive infinity is not an operation the
    /// arithmetic has, and the machine's own instruction is a call rather
    /// than an operator for the same reason.
    FCeil,
    /// The largest whole number not above a floating-point value.
    ///
    /// Rounding towards negative infinity.
    FFloor,
    /// The whole number nearest a floating-point value.
    ///
    /// A tie goes to the even one, which is what the machine's rounding
    /// instruction does.
    FRound,
}

impl BinOp {
    /// Symbol as printed.
    pub const fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::UDiv => "/",
            BinOp::Mod => "%",
            BinOp::UMod => "%",
            BinOp::And => "&",
            BinOp::Or => "|",
            BinOp::Xor => "^",
            BinOp::Shl => "<<",
            BinOp::Shr | BinOp::Sar => ">>",
            BinOp::FAdd => "+",
            BinOp::FSub => "-",
            BinOp::FMul => "*",
            BinOp::FDiv => "/",
            BinOp::FEq => "==",
            BinOp::FNe => "!=",
            BinOp::FLt => "<",
            BinOp::FLe => "<=",
            BinOp::FGt => ">",
            BinOp::FGe => ">=",
            BinOp::FMin => "min",
            BinOp::FMax => "max",
            BinOp::FUnordered => "unordered",
            BinOp::I2F32 => "f32",
            BinOp::I2F64 => "f64",
            BinOp::F2I32 => "i32",
            BinOp::F2I64 => "i64",
            BinOp::F2U32 => "u32",
            BinOp::F2U64 => "u64",
            BinOp::F2F32 => "f32",
            BinOp::F2F64 => "f64",
            BinOp::FSqrt => "sqrt",
            BinOp::I2F80 => "f80",
            BinOp::FNeg => "-",
            BinOp::FAbs => "abs",
            BinOp::FCeil => "ceil",
            BinOp::FFloor => "floor",
            BinOp::FRound => "round",
        }
    }

    /// True when this operation is a floating-point one.
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            BinOp::FAdd
                | BinOp::FSub
                | BinOp::FMul
                | BinOp::FDiv
                | BinOp::FEq
                | BinOp::FNe
                | BinOp::FLt
                | BinOp::FLe
                | BinOp::FGt
                | BinOp::FGe
                | BinOp::FMin
                | BinOp::FMax
                | BinOp::FUnordered
                | BinOp::I2F32
                | BinOp::I2F64
                | BinOp::F2I32
                | BinOp::F2I64
                | BinOp::F2U32
                | BinOp::F2U64
                | BinOp::F2F32
                | BinOp::F2F64
                | BinOp::FSqrt
                | BinOp::I2F80
                | BinOp::FNeg
                | BinOp::FAbs
                | BinOp::FCeil
                | BinOp::FFloor
                | BinOp::FRound
        )
    }

    /// The number of operands a floating-point operation takes.
    ///
    /// Most take two. The conversions and the square root take one, and
    /// the IR keeps them in the same shape as the binary ones so that a
    /// single printer can handle both.
    pub const fn float_arity(self) -> u8 {
        match self {
            BinOp::I2F32
            | BinOp::I2F64
            | BinOp::I2F80
            | BinOp::F2I32
            | BinOp::F2I64
            | BinOp::F2U32
            | BinOp::F2U64
            | BinOp::F2F32
            | BinOp::F2F64
            | BinOp::FSqrt
            | BinOp::FNeg
            | BinOp::FAbs
            | BinOp::FCeil
            | BinOp::FFloor
            | BinOp::FRound => 1,
            _ => 2,
        }
    }

    /// The name a floating-point operation is printed under.
    ///
    /// The rendered form is a call, not an operator: a bit pattern that
    /// holds a `double` is not a `double`, and spelling the operation as
    /// `a + b` would claim it is one.
    ///
    /// `bits` is the width the operation works on: four bytes for a
    /// `float`, eight for a `double`, ten for the x87 stack's extended
    /// precision.
    pub fn float_name(self, bits: u8) -> Option<String> {
        if !self.is_float() {
            return None;
        }
        let t = match bits {
            4 => "f32",
            8 => "f64",
            10 => "f80",
            _ => return None,
        };
        Some(match self {
            BinOp::FAdd => format!("{t}_add"),
            BinOp::FSub => format!("{t}_sub"),
            BinOp::FMul => format!("{t}_mul"),
            BinOp::FDiv => format!("{t}_div"),
            BinOp::FEq => format!("{t}_eq"),
            BinOp::FNe => format!("{t}_ne"),
            BinOp::FLt => format!("{t}_lt"),
            BinOp::FLe => format!("{t}_le"),
            BinOp::FGt => format!("{t}_gt"),
            BinOp::FGe => format!("{t}_ge"),
            BinOp::FMin => format!("{t}_min"),
            BinOp::FMax => format!("{t}_max"),
            BinOp::FUnordered => format!("{t}_unordered"),
            BinOp::FSqrt => format!("{t}_sqrt"),
            BinOp::FCeil => format!("{t}_ceil"),
            BinOp::FFloor => format!("{t}_floor"),
            BinOp::FRound => format!("{t}_round"),
            // The conversions name their target, not their source.
            BinOp::I2F32 => "f32_from_int".to_string(),
            BinOp::I2F64 => "f64_from_int".to_string(),
            BinOp::I2F80 => "f80_from_int".to_string(),
            BinOp::F2I32 => "i32_trunc".to_string(),
            BinOp::F2I64 => "i64_trunc".to_string(),
            BinOp::F2U32 => "u32_trunc".to_string(),
            BinOp::F2U64 => "u64_trunc".to_string(),
            BinOp::F2F32 => "f32_cast".to_string(),
            BinOp::F2F64 => "f64_cast".to_string(),
            BinOp::FNeg => format!("{t}_neg"),
            BinOp::FAbs => format!("{t}_abs"),
            _ => return None,
        })
    }

    /// Binding power, higher binds tighter. Used only for parentheses.
    pub const fn precedence(self) -> u8 {
        match self {
            // A floating-point operation is printed as a call, and a call
            // binds tighter than any operator.
            BinOp::FAdd
            | BinOp::FSub
            | BinOp::FMul
            | BinOp::FDiv
            | BinOp::FEq
            | BinOp::FNe
            | BinOp::FLt
            | BinOp::FLe
            | BinOp::FGt
            | BinOp::FGe
            | BinOp::FMin
            | BinOp::FMax
            | BinOp::FUnordered
            | BinOp::I2F32
            | BinOp::I2F64
            | BinOp::F2I32
            | BinOp::F2I64
            | BinOp::F2U32
            | BinOp::F2U64
            | BinOp::F2F32
            | BinOp::F2F64
            | BinOp::FSqrt
            | BinOp::I2F80
            | BinOp::FNeg
            | BinOp::FAbs
            | BinOp::FCeil
            | BinOp::FFloor
            | BinOp::FRound => 9,
            BinOp::Mul | BinOp::Div | BinOp::UDiv | BinOp::Mod | BinOp::UMod => 5,
            BinOp::Add | BinOp::Sub => 4,
            BinOp::Shl | BinOp::Shr | BinOp::Sar => 3,
            BinOp::And => 2,
            BinOp::Xor => 1,
            BinOp::Or => 0,
        }
    }
}

/// Comparison operators, as used in branch conditions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CondOp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Signed less than.
    Slt,
    /// Signed less or equal.
    Sle,
    /// Signed greater than.
    Sgt,
    /// Signed greater or equal.
    Sge,
    /// Unsigned less than.
    Ult,
    /// Unsigned less or equal.
    Ule,
    /// Unsigned greater than.
    Ugt,
    /// Unsigned greater or equal.
    Uge,
}

impl CondOp {
    /// Symbol as printed.
    pub const fn symbol(self) -> &'static str {
        match self {
            CondOp::Eq => "==",
            CondOp::Ne => "!=",
            CondOp::Slt => "<",
            CondOp::Sle => "<=",
            CondOp::Sgt => ">",
            CondOp::Sge => ">=",
            CondOp::Ult => "<",
            CondOp::Ule => "<=",
            CondOp::Ugt => ">",
            CondOp::Uge => ">=",
        }
    }

    /// True when the comparison treats its operands as signed.
    pub const fn signed(self) -> bool {
        matches!(self, CondOp::Slt | CondOp::Sle | CondOp::Sgt | CondOp::Sge)
    }

    /// True when the comparison treats its operands as unsigned.
    ///
    /// The two readings are different questions, and the output has to
    /// tell them apart: the same 32 bits are a negative number and a large
    /// positive one, and a branch that asks "below" asks one of them.
    pub const fn unsigned(self) -> bool {
        matches!(self, CondOp::Ult | CondOp::Ule | CondOp::Ugt | CondOp::Uge)
    }

    /// The flags the condition reads.
    ///
    /// A condition is a question about the flags, and the flags it asks
    /// about are not always the same ones. "Below" asks about the carry
    /// flag alone; "above" asks about the carry and the zero flags
    /// together, because the machine's "above" is the absence of both.
    pub const fn flags(self) -> u8 {
        use crate::lift::{FLAG_CF, FLAG_OF, FLAG_SF, FLAG_ZF};
        match self {
            CondOp::Eq | CondOp::Ne => FLAG_ZF,
            CondOp::Ult | CondOp::Uge => FLAG_CF,
            CondOp::Ule | CondOp::Ugt => FLAG_CF | FLAG_ZF,
            CondOp::Slt | CondOp::Sgt => FLAG_SF | FLAG_OF,
            CondOp::Sle | CondOp::Sge => FLAG_SF | FLAG_OF | FLAG_ZF,
        }
    }

    /// The condition on the flags this comparison asks about.
    ///
    /// Every comparison of the machine is a question about the flags, and
    /// the flags it reads differ: "below" reads the carry flag alone,
    /// "above" reads the carry and the zero flags together.
    pub const fn flag_cond(self) -> FlagCond {
        match self {
            CondOp::Eq => FlagCond::Z,
            CondOp::Ne => FlagCond::Nz,
            CondOp::Ult => FlagCond::C,
            CondOp::Uge => FlagCond::Nc,
            CondOp::Ule => FlagCond::Be,
            CondOp::Ugt => FlagCond::A,
            CondOp::Slt => FlagCond::L,
            CondOp::Sgt => FlagCond::G,
            CondOp::Sle => FlagCond::Le,
            CondOp::Sge => FlagCond::Nl,
        }
    }

    /// Mirror the operands, keeping the meaning.
    pub const fn swapped(self) -> CondOp {
        match self {
            CondOp::Eq => CondOp::Eq,
            CondOp::Ne => CondOp::Ne,
            CondOp::Slt => CondOp::Sgt,
            CondOp::Sle => CondOp::Sge,
            CondOp::Sgt => CondOp::Slt,
            CondOp::Sge => CondOp::Sle,
            CondOp::Ult => CondOp::Ugt,
            CondOp::Ule => CondOp::Uge,
            CondOp::Ugt => CondOp::Ult,
            CondOp::Uge => CondOp::Ule,
        }
    }

    /// Logical negation.
    pub const fn negated(self) -> CondOp {
        match self {
            CondOp::Eq => CondOp::Ne,
            CondOp::Ne => CondOp::Eq,
            CondOp::Slt => CondOp::Sge,
            CondOp::Sle => CondOp::Sgt,
            CondOp::Sgt => CondOp::Sle,
            CondOp::Sge => CondOp::Slt,
            CondOp::Ult => CondOp::Uge,
            CondOp::Ule => CondOp::Ugt,
            CondOp::Ugt => CondOp::Ule,
            CondOp::Uge => CondOp::Ult,
        }
    }
}

/// Unary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    /// Two's complement negation.
    Neg,
    /// Bitwise complement.
    Not,
    /// Logical negation, producing 0 or 1.
    LNot,
}

/// Width-changing conversions the machine performs implicitly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CastKind {
    /// Fill the high bits with zero.
    Zext,
    /// Replicate the sign bit.
    Sext,
    /// Drop the high bits.
    Trunc,
}

/// One value.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// Immediate.
    Const(u64),
    /// A named value: an SSA version, a parameter, or a promoted stack slot.
    Var(String),
    /// Memory read.
    Load {
        /// Address expression.
        addr: Box<Expr>,
        /// Access width in bytes.
        size: u8,
        /// True when the load sign-extends.
        signed: bool,
    },
    /// Arithmetic.
    Bin {
        /// Operator.
        op: BinOp,
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
        /// Width of the operation in bytes.
        bits: u8,
    },
    /// Unary operation.
    Un {
        /// Operator.
        op: UnOp,
        /// Operand.
        arg: Box<Expr>,
        /// Width in bytes.
        bits: u8,
    },
    /// Comparison producing a boolean.
    Cond {
        /// Operator.
        op: CondOp,
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
        /// Width of the compared operands in bytes.
        ///
        /// The width is not decoration. The same bits read as a signed
        /// number and as an unsigned number are two different numbers, and
        /// the branch that consumes the comparison has to know which one it
        /// asked about. A comparison of two 32-bit values that the branch
        /// reads as signed is a question about the sign of the difference,
        /// not about the sign of either operand.
        bits: u8,
    },
    /// A boolean combination of conditions.
    ///
    /// The condition codes ask about flags, and the answer is a combination
    /// of them: "less" asks whether the sign flag differs from the overflow
    /// flag, and "above" asks whether the carry flag is clear and the zero
    /// flag is clear. The bitwise operators of the arithmetic cannot say
    /// that, because their operands are words and these are answers.
    Bool {
        /// The combination.
        op: BoolOp,
        /// The operands, in the order the operation takes them.
        args: Vec<Expr>,
    },
    /// A condition on the condition-code flags.
    ///
    /// The flags are not a pair of values, so the condition is kept as a
    /// question about the flags themselves and answered from the values
    /// they hold.
    FlagCond {
        /// The condition.
        cond: FlagCond,
        /// The flags it reads.
        state: Box<FlagState>,
        /// Width of the operation that set the flags, in bytes.
        bits: u8,
    },
    /// Width conversion.
    Cast {
        /// Direction.
        kind: CastKind,
        /// Operand.
        arg: Box<Expr>,
        /// Source width in bytes.
        from: u8,
        /// Destination width in bytes.
        to: u8,
    },
    /// A conditional value: `cond ? yes : no`, which is what `cmov` and
    /// `setcc` compute without branching.
    Select {
        /// Predicate.
        cond: Box<Expr>,
        /// Value when the predicate holds.
        yes: Box<Expr>,
        /// Value otherwise.
        no: Box<Expr>,
    },
    /// A member of an aggregate, named by debug information.
    ///
    /// `*(u32 *)(p + 4)` where `p` points at `struct point` is `p->y` once the
    /// debug info says what `p` points at and where `y` lives. The width is
    /// kept because the load still has one, and the name is kept separately
    /// from the offset because that is what the source said.
    Member {
        /// Pointer to the aggregate.
        base: Box<Expr>,
        /// Byte offset of the member within the aggregate.
        offset: u64,
        /// Access width in bytes.
        size: u8,
        /// Member name as declared.
        name: String,
    },
    /// A phi node, present between SSA construction and structuring.
    Phi(Vec<(BlockId, Expr)>),
    /// An operation the machine word has no arithmetic for.
    ///
    /// A rotate, a byte swap, a bit scan and a two-word division have no
    /// spelling in the arithmetic the rest of the IR has. They are kept as
    /// calls with their operands, so the output says what the instruction
    /// did instead of dropping it or, worse, spelling it as something else.
    Intrinsic {
        /// The operation.
        op: Intrinsic,
        /// Its operands, in the order the machine takes them.
        args: Vec<Expr>,
        /// Width of the operation in bytes.
        bits: u8,
    },
    /// Something the lifter could not model, kept verbatim so the output never
    /// silently invents semantics.
    Unknown(String),
}

/// A boolean combination of conditions.
///
/// The arithmetic operators work on words. These work on answers, and they
/// produce an answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoolOp {
    /// Both operands hold.
    And,
    /// Either operand holds.
    Or,
    /// The operand does not hold.
    Not,
    /// The operands agree.
    Eq,
    /// The operands differ.
    Ne,
}

impl BoolOp {
    /// Symbol as printed.
    pub const fn symbol(self) -> &'static str {
        match self {
            BoolOp::And => "&&",
            BoolOp::Or => "||",
            BoolOp::Not => "!",
            BoolOp::Eq => "==",
            BoolOp::Ne => "!=",
        }
    }

    /// The number of operands the operation takes.
    pub const fn arity(self) -> usize {
        match self {
            BoolOp::Not => 1,
            _ => 2,
        }
    }

    /// How tightly the operation binds.
    ///
    /// The numbers follow the arithmetic's: a larger number binds tighter.
    /// The negation binds tighter than the conjunction, the conjunction
    /// tighter than the disjunction, and the disjunction tighter than the
    /// comparisons, which is the order a reader expects.
    pub const fn precedence(self) -> u8 {
        match self {
            BoolOp::Eq | BoolOp::Ne => 0,
            BoolOp::Or => 1,
            BoolOp::And => 2,
            BoolOp::Not => 6,
        }
    }
}

/// A condition on the condition-code flags.
///
/// The condition codes of the machine are not relations between two
/// values. "Less" asks whether the sign flag differs from the overflow
/// flag, and "above" asks whether the carry flag is clear and the zero
/// flag is clear. A comparison of two values says that only by accident,
/// and only when the flags came from a comparison, so the question is
/// named here and answered from the flags themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlagCond {
    /// The zero flag is set.
    Z,
    /// The zero flag is clear.
    Nz,
    /// The carry flag is set.
    C,
    /// The carry flag is clear.
    Nc,
    /// The parity flag is set.
    Po,
    /// The parity flag is clear.
    Npo,
    /// The sign flag is set.
    S,
    /// The sign flag is clear.
    Ns,
    /// The overflow flag is set.
    O,
    /// The overflow flag is clear.
    No,
    /// The sign flag differs from the overflow flag.
    L,
    /// The sign flag equals the overflow flag.
    Nl,
    /// The sign flag equals the overflow flag and the zero flag is clear.
    G,
    /// The sign flag differs from the overflow flag or the zero flag is set.
    Le,
    /// The carry flag is clear and the zero flag is clear.
    A,
    /// The carry flag is set or the zero flag is set.
    Be,
}

impl FlagCond {
    /// The condition that is the opposite of this one.
    pub const fn negated(self) -> FlagCond {
        match self {
            FlagCond::Z => FlagCond::Nz,
            FlagCond::Nz => FlagCond::Z,
            FlagCond::C => FlagCond::Nc,
            FlagCond::Nc => FlagCond::C,
            FlagCond::Po => FlagCond::Npo,
            FlagCond::Npo => FlagCond::Po,
            FlagCond::S => FlagCond::Ns,
            FlagCond::Ns => FlagCond::S,
            FlagCond::O => FlagCond::No,
            FlagCond::No => FlagCond::O,
            FlagCond::L => FlagCond::Nl,
            FlagCond::Nl => FlagCond::L,
            FlagCond::G => FlagCond::Le,
            FlagCond::Le => FlagCond::G,
            FlagCond::A => FlagCond::Be,
            FlagCond::Be => FlagCond::A,
        }
    }

    /// The comparison the condition is, when it is one.
    ///
    /// A condition that is a relation between two values has a comparison
    /// behind it. The conditions that ask about one flag, or about the
    /// difference between two flags, do not: no comparison of two values
    /// says whether the sign flag differs from the overflow flag.
    pub const fn to_cond_op(self) -> Option<CondOp> {
        Some(match self {
            FlagCond::Z => CondOp::Eq,
            FlagCond::Nz => CondOp::Ne,
            FlagCond::C => CondOp::Ult,
            FlagCond::Nc => CondOp::Uge,
            FlagCond::A => CondOp::Ugt,
            FlagCond::Be => CondOp::Ule,
            FlagCond::L => CondOp::Slt,
            FlagCond::Nl => CondOp::Sge,
            FlagCond::G => CondOp::Sgt,
            FlagCond::Le => CondOp::Sle,
            _ => return None,
        })
    }

    /// The flags the condition reads.
    pub const fn flags(self) -> u8 {
        use crate::lift::{FLAG_CF, FLAG_OF, FLAG_PF, FLAG_SF, FLAG_ZF};
        match self {
            FlagCond::Z | FlagCond::Nz => FLAG_ZF,
            FlagCond::C | FlagCond::Nc => FLAG_CF,
            FlagCond::Po | FlagCond::Npo => FLAG_PF,
            FlagCond::S | FlagCond::Ns => FLAG_SF,
            FlagCond::O | FlagCond::No => FLAG_OF,
            FlagCond::L | FlagCond::Nl => FLAG_SF | FLAG_OF,
            FlagCond::G => FLAG_SF | FLAG_OF | FLAG_ZF,
            FlagCond::Le => FLAG_SF | FLAG_OF | FLAG_ZF,
            FlagCond::A | FlagCond::Be => FLAG_CF | FLAG_ZF,
        }
    }
}

/// The condition-code flags an operation left behind.
///
/// Every flag holds the expression that holds when the flag is set, and
/// nothing at all when the operation did not define it. An operation that
/// leaves a flag alone leaves its entry empty: the machine keeps whatever
/// the flag held, and no expression says what that was.
///
/// The values are booleans. The zero flag of an addition is "the result is
/// zero", which is an answer and not a number, and the sign flag is "the
/// result is negative", which is the same kind of answer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FlagState {
    /// The carry flag: the addition carried, or the subtraction borrowed.
    pub cf: Option<Expr>,
    /// The zero flag: the result is zero.
    pub zf: Option<Expr>,
    /// The sign flag: the result is negative.
    pub sf: Option<Expr>,
    /// The overflow flag: the result did not fit.
    pub of: Option<Expr>,
    /// The parity flag: the low byte of the result has even parity.
    pub pf: Option<Expr>,
}

impl FlagState {
    /// Rewrite every variable name the flags read.
    ///
    /// The flag expressions hold the operands of the comparison that set
    /// them, so a rename that stopped at an `if`'s condition would leave
    /// `local_8_u64_1` inside the branch test while the assignment above it
    /// read `count_1`.
    pub fn rename_vars(&self, f: &dyn Fn(&str) -> Option<String>) -> FlagState {
        FlagState {
            cf: self.cf.as_ref().map(|e| e.rename_vars(f)),
            zf: self.zf.as_ref().map(|e| e.rename_vars(f)),
            sf: self.sf.as_ref().map(|e| e.rename_vars(f)),
            of: self.of.as_ref().map(|e| e.rename_vars(f)),
            pf: self.pf.as_ref().map(|e| e.rename_vars(f)),
        }
    }

    /// The value of one flag, named by its bit in the flag mask.
    pub fn get(&self, flag: u8) -> Option<&Expr> {
        use crate::lift::{FLAG_CF, FLAG_OF, FLAG_PF, FLAG_SF, FLAG_ZF};
        match flag {
            FLAG_CF => self.cf.as_ref(),
            FLAG_ZF => self.zf.as_ref(),
            FLAG_SF => self.sf.as_ref(),
            FLAG_OF => self.of.as_ref(),
            FLAG_PF => self.pf.as_ref(),
            _ => None,
        }
    }

    /// The condition the flags answer, as a boolean expression.
    ///
    /// Nothing when the operation did not define a flag the condition reads.
    /// The machine leaves that flag alone, so what it holds is whatever
    /// came before, and no expression says what that was.
    pub fn condition(&self, cond: FlagCond) -> Option<Expr> {
        use crate::lift::{FLAG_CF, FLAG_OF, FLAG_SF, FLAG_ZF};
        let flag = |mask: u8| -> Option<Expr> {
            self.get(mask).cloned()
        };
        // The flags answer the question in the form the branch asked it,
        // and the answer is simplified: a loop that ends on "not equal"
        // is a negation of a comparison, and the reader wants to see the
        // comparison rather than the negation.
        let answer = match cond {
            FlagCond::Z => flag(FLAG_ZF),
            FlagCond::Nz => flag(FLAG_ZF).map(not),
            FlagCond::C => flag(FLAG_CF),
            FlagCond::Nc => flag(FLAG_CF).map(not),
            FlagCond::Po => self.pf.clone(),
            FlagCond::Npo => self.pf.clone().map(not),
            FlagCond::S => self.sf.clone(),
            FlagCond::Ns => self.sf.clone().map(not),
            FlagCond::O => self.of.clone(),
            FlagCond::No => self.of.clone().map(not),
            FlagCond::L => self.differ(FLAG_SF, FLAG_OF),
            FlagCond::Nl => self.agree(FLAG_SF, FLAG_OF),
            FlagCond::G => match (self.get(FLAG_SF), self.get(FLAG_OF), self.get(FLAG_ZF)) {
                (Some(sf), Some(of), Some(zf)) => Some(Expr::Bool {
                    op: BoolOp::And,
                    args: vec![agree(sf, of), not(zf.clone())],
                }),
                _ => None,
            },
            FlagCond::Le => match (self.get(FLAG_SF), self.get(FLAG_OF), self.get(FLAG_ZF)) {
                (Some(sf), Some(of), Some(zf)) => Some(Expr::Bool {
                    op: BoolOp::Or,
                    args: vec![differ(sf, of), zf.clone()],
                }),
                _ => None,
            },
            FlagCond::A => match (self.get(FLAG_CF), self.get(FLAG_ZF)) {
                (Some(cf), Some(zf)) => Some(Expr::Bool {
                    op: BoolOp::And,
                    args: vec![not(cf.clone()), not(zf.clone())],
                }),
                _ => None,
            },
            FlagCond::Be => match (self.get(FLAG_CF), self.get(FLAG_ZF)) {
                (Some(cf), Some(zf)) => Some(Expr::Bool {
                    op: BoolOp::Or,
                    args: vec![cf.clone(), zf.clone()],
                }),
                _ => None,
            },
        };
        answer.map(|e| e.simplify())
    }

    /// Whether two flags differ, when both are known.
    fn differ(&self, a: u8, b: u8) -> Option<Expr> {
        match (self.get(a), self.get(b)) {
            (Some(a), Some(b)) => Some(differ(a, b)),
            _ => None,
        }
    }

    /// Whether two flags agree, when both are known.
    fn agree(&self, a: u8, b: u8) -> Option<Expr> {
        match (self.get(a), self.get(b)) {
            (Some(a), Some(b)) => Some(agree(a, b)),
            _ => None,
        }
    }

    /// All variables the flag values mention.
    pub fn vars(&self, out: &mut Vec<String>) {
        for value in [
            self.cf.as_ref(),
            self.zf.as_ref(),
            self.sf.as_ref(),
            self.of.as_ref(),
            self.pf.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            value.vars(out);
        }
    }

    /// Substitute variables by name in the flag values.
    pub fn substitute(&self, map: &std::collections::HashMap<String, Expr>) -> FlagState {
        let sub = |e: &Option<Expr>| e.as_ref().map(|e| e.substitute(map));
        FlagState {
            cf: sub(&self.cf),
            zf: sub(&self.zf),
            sf: sub(&self.sf),
            of: sub(&self.of),
            pf: sub(&self.pf),
        }
    }

    /// Fold the constant parts of the flag values.
    pub fn simplify(&self) -> FlagState {
        let simp = |e: &Option<Expr>| e.as_ref().map(|e| e.simplify());
        FlagState {
            cf: simp(&self.cf),
            zf: simp(&self.zf),
            sf: simp(&self.sf),
            of: simp(&self.of),
            pf: simp(&self.pf),
        }
    }
}

/// The overflow a sum of two values produced.
///
/// The sum overflowed when the two operands had the same sign and the
/// result has the other one. That is the sign bit of
/// `(a ^ result) & (b ^ result)`, and the branch that asks about the
/// overflow flag asks exactly that.
pub fn add_overflow(a: &Expr, b: &Expr, result: &Expr, bits: u8) -> Expr {
    Expr::Bin {
        op: BinOp::And,
        lhs: Box::new(Expr::Bin {
            op: BinOp::Xor,
            lhs: Box::new(a.clone()),
            rhs: Box::new(result.clone()),
            bits,
        }),
        rhs: Box::new(Expr::Bin {
            op: BinOp::Xor,
            lhs: Box::new(b.clone()),
            rhs: Box::new(result.clone()),
            bits,
        }),
        bits,
    }
}

/// The overflow a difference of two values produced.
///
/// The difference overflowed when the operands had different signs and the
/// result has the sign of the first one. That is the sign bit of
/// `(a ^ b) & (a ^ result)`.
pub fn sub_overflow(a: &Expr, b: &Expr, result: &Expr, bits: u8) -> Expr {
    Expr::Bin {
        op: BinOp::And,
        lhs: Box::new(Expr::Bin {
            op: BinOp::Xor,
            lhs: Box::new(a.clone()),
            rhs: Box::new(b.clone()),
            bits,
        }),
        rhs: Box::new(Expr::Bin {
            op: BinOp::Xor,
            lhs: Box::new(a.clone()),
            rhs: Box::new(result.clone()),
            bits,
        }),
        bits,
    }
}

/// The value a comparison operand is, without the width wrapper.
///
/// A comparison of a 32-bit view reads the low half of a 64-bit register,
/// and the IR says so with a truncation. The comparison's own width says
/// the same thing, so the wrapper is dropped and the type is printed once.
pub fn comparison_operand(e: &Expr, bits: u8) -> &Expr {
    if let Expr::Cast {
        kind: CastKind::Trunc,
        arg,
        from,
        to,
    } = e
        && *to == bits
        && *from > *to
    {
        arg
    } else {
        e
    }
}

/// The negation of a boolean.
pub fn not(e: Expr) -> Expr {
    Expr::Bool {
        op: BoolOp::Not,
        args: vec![e],
    }
}

/// Two booleans that differ.
fn differ(a: &Expr, b: &Expr) -> Expr {
    Expr::Bool {
        op: BoolOp::Ne,
        args: vec![a.clone(), b.clone()],
    }
}

/// Two booleans that agree.
fn agree(a: &Expr, b: &Expr) -> Expr {
    Expr::Bool {
        op: BoolOp::Eq,
        args: vec![a.clone(), b.clone()],
    }
}

/// An operation with no arithmetic spelling.
///
/// Each of these is a single machine instruction whose result cannot be
/// written as a combination of the arithmetic operations, because the
/// arithmetic operations work on one machine word and these do not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intrinsic {
    /// Rotate left.
    Rol,
    /// Rotate right.
    Ror,
    /// Rotate left through the carry, which joins the value as its top bit.
    Rcl,
    /// Rotate right through the carry, which joins the value as its top bit.
    Rcr,
    /// Swap the bytes of a value.
    Bswap,
    /// Index of the lowest set bit.
    Bsf,
    /// Index of the highest set bit.
    Bsr,
    /// Whether the low byte of a value has even parity.
    ///
    /// The parity flag is one, when the number of set bits in the low byte
    /// is even. It has no spelling in the arithmetic the IR has, because
    /// it is a question about a count and not about a relation between two
    /// values.
    ParityEven,
    /// The carry out of an addition.
    ///
    /// The sum did not fit. The result is smaller than the value that was
    /// there before, which is the only way an addition can lose a bit.
    Carry,
    /// The borrow out of a subtraction.
    ///
    /// The difference did not fit. The value that was there before was
    /// smaller than the value subtracted from it.
    Borrow,
    /// The overflow of an addition.
    ///
    /// The sum has a sign the operands do not share. The operands take
    /// `previous`, `operand` and `result`, in that order.
    OverflowAdd,
    /// The overflow of a subtraction.
    ///
    /// The difference has a sign the minuend does not share. The operands
    /// take `previous`, `operand` and `result`, in that order.
    OverflowSub,
    /// Number of trailing zero bits.
    Tzcnt,
    /// Number of leading zero bits.
    Lzcnt,
    /// Number of set bits.
    ///
    /// A count of the bits that are one, which the arithmetic the rest of
    /// the IR has cannot say: it is a question about a population count
    /// and not about a relation between two values.
    Popcnt,
    /// The magnitude of one value with the sign of another.
    ///
    /// The two operands are a magnitude and a sign: the result takes the
    /// sign bit of the second and every other bit of the first.
    CopySign,
    /// The sign bits of the lanes, gathered into an integer.
    MovMask,
    /// Divide a two-word dividend.
    Div,
    /// Divide a two-word dividend, read as a signed number.
    IDiv,
    /// The remainder of a two-word division.
    DivRem,
    /// The remainder of a two-word division, read as a signed number.
    IDivRem,
    /// The low half of a double-width multiplication.
    MulLo,
    /// The high half of a double-width multiplication.
    ///
    /// A one-operand `mul` produces twice the operand width, and the upper
    /// half is the whole point of the instruction: it is what tells the reader
    /// whether the product fitted. Keeping only the low half and calling it
    /// the product is a semantic error that shows up as a wrong `rax` for
    /// every operand pair large enough to carry.
    MulHi,
    /// `a + b + c` at the stated width, where `c` is the incoming carry.
    AdcSum,
    /// Whether `a + b + c` carried out of the width.
    AdcCarry,
    /// `a - b - c` at the stated width, where `c` is the incoming borrow.
    SbbDiff,
    /// Whether `a - b - c` borrowed.
    SbbBorrow,
    /// The low half of the product of two signed numbers.
    IMulLo,
    /// The high half of the product of two signed numbers.
    IMulHi,
    /// A parallel addition over the lanes of the operands.
    ///
    /// The word is cut into lanes of a fixed width and each lane is added
    /// on its own, so the carries of one lane never reach the next. The
    /// width of the lane is given by `bits`, and the width of the word by
    /// the number of operands: two operands are a word wide.
    ParallelAdd,
    /// A parallel subtraction over the lanes of the operands.
    ParallelSub,
    /// A parallel addition that saturates each lane.
    ///
    /// A lane that would carry out of its own width stops at the largest
    /// value the lane can hold instead of wrapping around.
    ParallelAddSat,
    /// A parallel subtraction that saturates each lane.
    ParallelSubSat,
    /// A parallel addition that halves each lane's result.
    ///
    /// The halved sum is taken after the addition, so a lane whose sum
    /// overflowed gives the largest value the lane can hold, divided by
    /// two.
    ParallelAddHalf,
    /// A parallel subtraction that halves each lane's result.
    ParallelSubHalf,
    /// A parallel addition over the lanes, reading them as unsigned.
    ParallelAddUns,
    /// A parallel subtraction over the lanes, reading them as unsigned.
    ParallelSubUns,
    /// A parallel addition over the lanes that saturates and reads them as
    /// unsigned.
    ParallelAddSatUns,
    /// A parallel subtraction over the lanes that saturates and reads them
    /// as unsigned.
    ParallelSubSatUns,
    /// A parallel addition over the lanes that halves the result and reads
    /// them as unsigned.
    ParallelAddHalfUns,
    /// A parallel subtraction over the lanes that halves the result and
    /// reads them as unsigned.
    ParallelSubHalfUns,
    /// The bytes of a value in reverse order.
    ///
    /// The full byte swap of the word.
    RevBytes,
    /// The halves of a value in reverse order.
    ///
    /// The sixteen-bit halves are swapped and the bytes inside each half
    /// are not.
    RevHalves,
    /// The halves of a value in reverse order, sign extended.
    ///
    /// The sixteen-bit halves are swapped and the lower one is then read as
    /// a signed number.
    RevHalvesSigned,
    /// The bits of a value in reverse order.
    RevBits,
    /// A choice between two values, taken bit by bit.
    ///
    /// The first operand is the mask: a set bit in it takes the bit of the
    /// second operand, a clear bit takes the bit of the third.
    Select,
    /// A checksum that folds a value into a running one.
    ///
    /// The bits are folded in one at a time, from the bottom of the value
    /// up, and every bit that comes out of the top goes back in through a
    /// polynomial.  The polynomial is the one that the ordinary checksum of
    /// a zip file uses, with its bits the other way round.
    Crc32,
    /// A checksum that folds a value into a running one, with the other
    /// polynomial.
    ///
    /// This is the polynomial that the network protocols use.
    Crc32c,
    /// A saturated addition of two values.
    ///
    /// The sum that would not fit stops at the largest value the width can
    /// hold.
    SatAdd,
    /// A saturated subtraction of two values.
    SatSub,
    /// A parallel addition of two extended values.
    ///
    /// Each half of the second operand is extended and added to the
    /// matching half of the first, and the carry of one half never
    /// reaches the other.
    ExtendAdd,
    /// A parallel addition of two extended values, read as signed numbers.
    ExtendAddSigned,
    /// A parallel addition over the halves of the operands, exchanged.
    ///
    /// The high half of the result is the addition of the high half of
    /// the first operand and the low half of the second, and the low half
    /// of the result is the addition of the low half of the first operand
    /// and the high half of the second.
    ExchangeAdd,
    /// A parallel subtraction over the halves of the operands, exchanged.
    ///
    /// The high half of the result is the subtraction of the high half of
    /// the first operand and the low half of the second, and the low half
    /// of the result is the subtraction of the low half of the first
    /// operand and the high half of the second.
    ExchangeSub,
    /// A parallel addition over the halves of the operands, exchanged, that
    /// halves each half of the result.
    ExchangeAddHalf,
    /// A parallel subtraction over the halves of the operands, exchanged,
    /// that halves each half of the result.
    ExchangeSubHalf,
    /// A parallel addition over the halves of the operands, exchanged, that
    /// saturates each half of the result.
    ExchangeAddSat,
    /// A parallel subtraction over the halves of the operands, exchanged,
    /// that saturates each half of the result.
    ExchangeSubSat,
    /// A parallel addition over the halves of the operands, exchanged, that
    /// reads the halves as unsigned numbers.
    ExchangeAddUns,
    /// A parallel subtraction over the halves of the operands, exchanged,
    /// that reads the halves as unsigned numbers.
    ExchangeSubUns,
    /// A parallel addition over the halves of the operands, exchanged, that
    /// saturates and reads the halves as unsigned numbers.
    ExchangeAddSatUns,
    /// A parallel subtraction over the halves of the operands, exchanged,
    /// that saturates and reads the halves as unsigned numbers.
    ExchangeSubSatUns,
    /// A parallel addition over the halves of the operands, exchanged, that
    /// halves the result and reads the halves as unsigned numbers.
    ExchangeAddHalfUns,
    /// A parallel subtraction over the halves of the operands, exchanged,
    /// that halves the result and reads the halves as unsigned numbers.
    ExchangeSubHalfUns,
}

impl Intrinsic {
    /// The name the operation is printed under.
    ///
    /// The byte swap names its width, because swapping four bytes and
    /// swapping eight are different operations and the reader has to know
    /// which one happened.
    pub fn name_for(self, bits: u8) -> String {
        match self {
            Intrinsic::Bswap => format!("bswap{}", bits * 8),
            // The sign-copying operation works on a float, and two floats of
            // different widths are two different operations.
            Intrinsic::CopySign => format!("copysign{}", bits * 8),
            // The parallel operations name the width of the lane they work
            // on, because the lanes are what the operation is about: the
            // same word read as four bytes and as two halves is two
            // different operations.
            Intrinsic::ParallelAdd => format!("padd{}", bits * 8),
            Intrinsic::ParallelSub => format!("psub{}", bits * 8),
            Intrinsic::ParallelAddSat => format!("qadd{}", bits * 8),
            Intrinsic::ParallelSubSat => format!("qsub{}", bits * 8),
            Intrinsic::ParallelAddHalf => format!("shadd{}", bits * 8),
            Intrinsic::ParallelSubHalf => format!("shsub{}", bits * 8),
            Intrinsic::ParallelAddUns => format!("uadd{}", bits * 8),
            Intrinsic::ParallelSubUns => format!("usub{}", bits * 8),
            Intrinsic::ParallelAddSatUns => format!("uqadd{}", bits * 8),
            Intrinsic::ParallelSubSatUns => format!("uqsub{}", bits * 8),
            Intrinsic::ParallelAddHalfUns => format!("uhadd{}", bits * 8),
            Intrinsic::ParallelSubHalfUns => format!("uhsub{}", bits * 8),
            other => other.name().to_string(),
        }
    }

    /// The name the operation is printed under, without the width.
    pub const fn name(self) -> &'static str {
        match self {
            Intrinsic::Rol => "rol",
            Intrinsic::Ror => "ror",
            Intrinsic::Rcl => "rcl",
            Intrinsic::Rcr => "rcr",
            Intrinsic::Bswap => "bswap",
            Intrinsic::Bsf => "bsf",
            Intrinsic::Bsr => "bsr",
            Intrinsic::Tzcnt => "tzcnt",
            Intrinsic::Lzcnt => "lzcnt",
            Intrinsic::Popcnt => "popcnt",
            Intrinsic::CopySign => "copysign",
            Intrinsic::MovMask => "movmask",
            Intrinsic::ParityEven => "parity_even",
            Intrinsic::Carry => "carry",
            Intrinsic::Borrow => "borrow",
            Intrinsic::OverflowAdd => "overflow_add",
            Intrinsic::OverflowSub => "overflow_sub",
            Intrinsic::Div => "div",
            Intrinsic::IDiv => "idiv",
            Intrinsic::DivRem => "divrem",
            Intrinsic::IDivRem => "idivrem",
            Intrinsic::MulLo => "mul_lo",
            Intrinsic::MulHi => "mul_hi",
            Intrinsic::IMulLo => "imul_lo",
            Intrinsic::IMulHi => "imul_hi",
            Intrinsic::AdcSum => "adc_sum",
            Intrinsic::AdcCarry => "adc_carry",
            Intrinsic::SbbDiff => "sbb_diff",
            Intrinsic::SbbBorrow => "sbb_borrow",
            Intrinsic::RevBytes => "rev_bytes",
            Intrinsic::RevHalves => "rev_halves",
            Intrinsic::RevHalvesSigned => "rev_halves_signed",
            Intrinsic::RevBits => "rev_bits",
            Intrinsic::Select => "select",
            Intrinsic::Crc32 => "crc32",
            Intrinsic::Crc32c => "crc32c",
            Intrinsic::SatAdd => "sat_add",
            Intrinsic::ExtendAdd => "extend_add",
            Intrinsic::ExtendAddSigned => "extend_add_signed",
            Intrinsic::SatSub => "sat_sub",
            Intrinsic::ExchangeAdd => "exchange_add",
            Intrinsic::ExchangeSub => "exchange_sub",
            Intrinsic::ExchangeAddHalf => "exchange_add_half",
            Intrinsic::ExchangeSubHalf => "exchange_sub_half",
            Intrinsic::ExchangeAddSat => "exchange_add_sat",
            Intrinsic::ExchangeSubSat => "exchange_sub_sat",
            Intrinsic::ExchangeAddUns => "exchange_add_uns",
            Intrinsic::ExchangeSubUns => "exchange_sub_uns",
            Intrinsic::ExchangeAddSatUns => "exchange_add_sat_uns",
            Intrinsic::ExchangeSubSatUns => "exchange_sub_sat_uns",
            Intrinsic::ExchangeAddHalfUns => "exchange_add_half_uns",
            Intrinsic::ExchangeSubHalfUns => "exchange_sub_half_uns",
            Intrinsic::ParallelAdd => "parallel_add",
            Intrinsic::ParallelSub => "parallel_sub",
            Intrinsic::ParallelAddSat => "parallel_add_sat",
            Intrinsic::ParallelSubSat => "parallel_sub_sat",
            Intrinsic::ParallelAddHalf => "parallel_add_half",
            Intrinsic::ParallelSubHalf => "parallel_sub_half",
            Intrinsic::ParallelAddUns => "parallel_add_uns",
            Intrinsic::ParallelSubUns => "parallel_sub_uns",
            Intrinsic::ParallelAddSatUns => "parallel_add_sat_uns",
            Intrinsic::ParallelSubSatUns => "parallel_sub_sat_uns",
            Intrinsic::ParallelAddHalfUns => "parallel_add_half_uns",
            Intrinsic::ParallelSubHalfUns => "parallel_sub_half_uns",
        }
    }
}

impl Expr {
    /// Convenience constructor for a constant.
    pub fn imm(v: u64) -> Expr {
        Expr::Const(v)
    }

    /// Convenience constructor for a variable reference.
    pub fn var(name: impl Into<String>) -> Expr {
        Expr::Var(name.into())
    }

    /// Rewrite every variable name through `f`, leaving everything else.
    ///
    /// `f` returns `None` for a name it does not rename. The walk covers every
    /// place a value can be buried, including phi operands and the flag state
    /// a condition reads: a rename that skipped either would leave the body
    /// half-renamed, which reads worse than not renaming at all.
    pub fn rename_vars(&self, f: &dyn Fn(&str) -> Option<String>) -> Expr {
        match self {
            Expr::Const(_) | Expr::Unknown(_) => self.clone(),
            Expr::Var(n) => f(n).map_or_else(|| self.clone(), Expr::Var),
            Expr::Load { addr, size, signed } => Expr::Load {
                addr: Box::new(addr.rename_vars(f)),
                size: *size,
                signed: *signed,
            },
            Expr::Bin { op, lhs, rhs, bits } => Expr::Bin {
                op: *op,
                lhs: Box::new(lhs.rename_vars(f)),
                rhs: Box::new(rhs.rename_vars(f)),
                bits: *bits,
            },
            Expr::Un { op, arg, bits } => Expr::Un {
                op: *op,
                arg: Box::new(arg.rename_vars(f)),
                bits: *bits,
            },
            Expr::Cond { op, lhs, rhs, bits } => Expr::Cond {
                op: *op,
                lhs: Box::new(lhs.rename_vars(f)),
                rhs: Box::new(rhs.rename_vars(f)),
                bits: *bits,
            },
            Expr::Bool { op, args } => Expr::Bool {
                op: *op,
                args: args.iter().map(|a| a.rename_vars(f)).collect(),
            },
            Expr::FlagCond { cond, state, bits } => Expr::FlagCond {
                cond: *cond,
                state: Box::new(state.rename_vars(f)),
                bits: *bits,
            },
            Expr::Cast {
                kind,
                arg,
                from,
                to,
            } => Expr::Cast {
                kind: *kind,
                arg: Box::new(arg.rename_vars(f)),
                from: *from,
                to: *to,
            },
            Expr::Select { cond, yes, no } => Expr::Select {
                cond: Box::new(cond.rename_vars(f)),
                yes: Box::new(yes.rename_vars(f)),
                no: Box::new(no.rename_vars(f)),
            },
            Expr::Member {
                base,
                offset,
                size,
                name,
            } => Expr::Member {
                base: Box::new(base.rename_vars(f)),
                offset: *offset,
                size: *size,
                name: name.clone(),
            },
            Expr::Phi(args) => Expr::Phi(
                args.iter()
                    .map(|(b, e)| (*b, e.rename_vars(f)))
                    .collect(),
            ),
            Expr::Intrinsic { op, args, bits } => Expr::Intrinsic {
                op: *op,
                args: args.iter().map(|a| a.rename_vars(f)).collect(),
                bits: *bits,
            },
        }
    }

    /// Wrap in parentheses when the expression needs them at `min` precedence.
    fn fmt_prec(&self, f: &mut fmt::Formatter<'_>, min: u8) -> fmt::Result {
        let (prec, needs) = match self {
            Expr::Bin { op, .. } => (op.precedence(), !op.is_float()),
            Expr::Cond { .. } => (0, true),
            Expr::Bool { op, .. } => (op.precedence(), true),
            Expr::FlagCond { .. } => (0, true),
            Expr::Un { .. } => (6, true),
            Expr::Cast { .. } => (7, true),
            Expr::Select { .. } => (0, true),
            _ => (9, false),
        };
        let paren = needs && prec < min;
        if paren {
            write!(f, "(")?;
        }
        match self {
            Expr::Const(v) => {
                // Every constant is printed in hex, including small ones. In a
                // disassembly the same number appears as an immediate, a size
                // and an address, and a single radix is what lets a reader
                // compare them without converting in their head.
                write!(f, "{v:#x}")?;
            }
            Expr::Var(name) => write!(f, "{name}")?,
            Expr::Load { addr, size, signed } => {
                let ty = type_name(*size, *signed);
                write!(f, "*({ty}*)")?;
                addr.fmt_prec(f, 7)?;
            }
            Expr::Bin { op, lhs, rhs, bits } => {
                if let Some(name) = op.float_name(*bits) {
                    // A floating-point operation is printed as a call: the
                    // operands are bit patterns, and `a + b` would claim they
                    // are numbers.
                    write!(f, "{name}(")?;
                    lhs.fmt_prec(f, 0)?;
                    if op.float_arity() == 2 {
                        write!(f, ", ")?;
                        rhs.fmt_prec(f, 0)?;
                    }
                    write!(f, ")")?;
                } else {
                    lhs.fmt_prec(f, op.precedence())?;
                    write!(f, " {} ", op.symbol())?;
                    rhs.fmt_prec(f, op.precedence() + 1)?;
                }
            }
            Expr::Un { op, arg, .. } => {
                let sym = match op {
                    UnOp::Neg => "-",
                    UnOp::Not => "~",
                    UnOp::LNot => "!",
                };
                write!(f, "{sym}")?;
                arg.fmt_prec(f, 6)?;
            }
            Expr::Cond {
                op,
                lhs,
                rhs,
                bits,
            } => {
                // The width decides how the operands are read, and the
                // output has to say so: the same 32 bits are a negative
                // number and a large positive one, and a branch that asks
                // "below" asks one of them.
                let lhs = comparison_operand(lhs, *bits);
                let rhs = comparison_operand(rhs, *bits);
                if *bits < 8 {
                    let ty = type_name(*bits, op.signed());
                    // A constant is the value it is, whatever the width.
                    if !matches!(lhs, Expr::Const(_)) {
                        write!(f, "({ty})")?;
                        lhs.fmt_prec(f, 7)?;
                    } else {
                        lhs.fmt_prec(f, 7)?;
                    }
                    write!(f, " {} ", op.symbol())?;
                    if !matches!(rhs, Expr::Const(_)) {
                        write!(f, "({ty})")?;
                        rhs.fmt_prec(f, 7)?;
                    } else {
                        rhs.fmt_prec(f, 7)?;
                    }
                } else {
                    lhs.fmt_prec(f, 1)?;
                    write!(f, " {} ", op.symbol())?;
                    rhs.fmt_prec(f, 1)?;
                }
            }
            Expr::Bool { op, args } => {
                if *op == BoolOp::Not {
                    write!(f, "!")?;
                    args[0].fmt_prec(f, 6)?;
                } else {
                    args[0].fmt_prec(f, op.precedence())?;
                    write!(f, " {} ", op.symbol())?;
                    args[1].fmt_prec(f, op.precedence() + 1)?;
                }
            }
            Expr::FlagCond { cond, state, .. } => {
                // The condition on the flags is a question about the flags,
                // and the flags answer it with the values they hold.
                match state.condition(*cond) {
                    Some(e) => e.fmt_prec(f, min)?,
                    None => write!(f, "<flags>")?,
                }
            }
            Expr::Cast { kind, arg, from, to } => {
                let k = match kind {
                    CastKind::Zext => "u",
                    CastKind::Sext => "s",
                    CastKind::Trunc => "u",
                };
                let _ = k;
                write!(f, "({})", type_name(*to, *kind == CastKind::Sext))?;
                let _ = from;
                arg.fmt_prec(f, 7)?;
            }
            Expr::Select { cond, yes, no } => {
                cond.fmt_prec(f, 1)?;
                write!(f, " ? ")?;
                yes.fmt_prec(f, 0)?;
                write!(f, " : ")?;
                no.fmt_prec(f, 0)?;
            }
            Expr::Member { base, name, .. } => {
                base.fmt_prec(f, 0)?;
                write!(f, "->{name}")?;
            }
            Expr::Phi(args) => {
                write!(f, "phi(")?;
                for (i, (bb, e)) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "bb{bb}: ")?;
                    e.fmt_prec(f, 0)?;
                }
                write!(f, ")")?;
            }
            Expr::Intrinsic { op, args, bits } => {
                write!(f, "{}(", op.name_for(*bits))?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    a.fmt_prec(f, 0)?;
                }
                write!(f, ")")?;
            }
            Expr::Unknown(text) => write!(f, "<{text}>")?,
        }
        if paren {
            write!(f, ")")?;
        }
        Ok(())
    }

    /// All variables referenced.
    pub fn vars(&self, out: &mut Vec<String>) {
        match self {
            Expr::Var(name) => out.push(name.clone()),
            Expr::Load { addr, .. } => addr.vars(out),
            Expr::Bin { lhs, rhs, .. } => {
                lhs.vars(out);
                rhs.vars(out);
            }
            Expr::Un { arg, .. } | Expr::Cast { arg, .. } => arg.vars(out),
            Expr::Cond { lhs, rhs, .. } => {
                lhs.vars(out);
                rhs.vars(out);
            }
            Expr::Bool { args, .. } => {
                for a in args {
                    a.vars(out);
                }
            }
            Expr::FlagCond { state, .. } => state.vars(out),
            Expr::Select { cond, yes, no } => {
                cond.vars(out);
                yes.vars(out);
                no.vars(out);
            }
            Expr::Member { base, .. } => base.vars(out),
            Expr::Phi(args) => {
                for (_, e) in args {
                    e.vars(out);
                }
            }
            Expr::Intrinsic { args, .. } => {
                for a in args {
                    a.vars(out);
                }
            }
            Expr::Const(_) | Expr::Unknown(_) => {}
        }
    }

    /// Substitute variables by name.
    pub fn substitute(&self, map: &std::collections::HashMap<String, Expr>) -> Expr {
        match self {
            Expr::Var(name) => map.get(name).cloned().unwrap_or_else(|| self.clone()),
            Expr::Load { addr, size, signed } => Expr::Load {
                addr: Box::new(addr.substitute(map)),
                size: *size,
                signed: *signed,
            },
            Expr::Bin { op, lhs, rhs, bits } => Expr::Bin {
                op: *op,
                lhs: Box::new(lhs.substitute(map)),
                rhs: Box::new(rhs.substitute(map)),
                bits: *bits,
            },
            Expr::Un { op, arg, bits } => Expr::Un {
                op: *op,
                arg: Box::new(arg.substitute(map)),
                bits: *bits,
            },
            Expr::Cond {
                op,
                lhs,
                rhs,
                bits,
            } => Expr::Cond {
                op: *op,
                lhs: Box::new(lhs.substitute(map)),
                rhs: Box::new(rhs.substitute(map)),
                bits: *bits,
            },
            Expr::Bool { op, args } => Expr::Bool {
                op: *op,
                args: args.iter().map(|a| a.substitute(map)).collect(),
            },
            Expr::FlagCond {
                cond,
                state,
                bits,
            } => Expr::FlagCond {
                cond: *cond,
                state: Box::new(state.substitute(map)),
                bits: *bits,
            },
            Expr::Cast {
                kind,
                arg,
                from,
                to,
            } => Expr::Cast {
                kind: *kind,
                arg: Box::new(arg.substitute(map)),
                from: *from,
                to: *to,
            },
            Expr::Select { cond, yes, no } => Expr::Select {
                cond: Box::new(cond.substitute(map)),
                yes: Box::new(yes.substitute(map)),
                no: Box::new(no.substitute(map)),
            },
            Expr::Member {
                base,
                offset,
                size,
                name,
            } => Expr::Member {
                base: Box::new(base.substitute(map)),
                offset: *offset,
                size: *size,
                name: name.clone(),
            },
            Expr::Phi(args) => Expr::Phi(
                args.iter()
                    .map(|(bb, e)| (*bb, e.substitute(map)))
                    .collect(),
            ),
            Expr::Intrinsic { op, args, bits } => Expr::Intrinsic {
                op: *op,
                args: args.iter().map(|a| a.substitute(map)).collect(),
                bits: *bits,
            },
            Expr::Const(_) | Expr::Unknown(_) => self.clone(),
        }
    }

    /// Fold constant subexpressions and drop redundant casts.
    pub fn simplify(&self) -> Expr {
        match self {
            Expr::Bin { op, lhs, rhs, bits } => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                if let (Expr::Const(a), Expr::Const(b)) = (&lhs, &rhs)
                    && let Some(v) = fold(*op, *a, *b, *bits)
                {
                    return Expr::Const(v);
                }
                // x ^ x == 0, x - x == 0, x & 0 == 0.
                if lhs == rhs {
                    match op {
                        BinOp::Xor | BinOp::Sub => return Expr::Const(0),
                        BinOp::And | BinOp::Or => return lhs,
                        _ => {}
                    }
                }
                // Adding or subtracting zero, multiplying or dividing by one,
                // and the annihilators that survive optimisation.
                if let Expr::Const(v) = &rhs {
                    match op {
                        BinOp::Add | BinOp::Or | BinOp::Xor if *v == 0 => return lhs,
                        BinOp::Sub if *v == 0 => return lhs,
                        BinOp::Mul if *v == 1 => return lhs,
                        BinOp::Mul | BinOp::And if *v == 0 => return Expr::Const(0),
                        BinOp::Shl | BinOp::Shr | BinOp::Sar if *v == 0 => return lhs,
                        _ => {}
                    }
                }
                Expr::Bin {
                    op: *op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    bits: *bits,
                }
            }
            Expr::Un { op, arg, bits } => {
                let arg = arg.simplify();
                if let Expr::Const(v) = &arg {
                    // The operation happens at the width of the operand: the
                    // complement of zero in four bytes is `0xffffffff`, not
                    // the fourteen f's of a 64-bit word.
                    let width = usize::from(*bits).clamp(1, 8);
                    return Expr::Const(match op {
                        UnOp::Neg => truncate(v.wrapping_neg(), width),
                        UnOp::Not => truncate(!*v, width),
                        UnOp::LNot => u64::from(*v == 0),
                    });
                }
                // The negation of a comparison is the opposite comparison.
                // A loop that ends on "not equal" is written as the
                // negation of "equal" by the machine, and the reader wants
                // to see the question the branch actually asks.
                if *op == UnOp::LNot {
                    match &arg {
                        Expr::Cond {
                            op,
                            lhs,
                            rhs,
                            bits,
                        } => {
                            return Expr::Cond {
                                op: op.negated(),
                                lhs: lhs.clone(),
                                rhs: rhs.clone(),
                                bits: *bits,
                            };
                        }
                        Expr::Bool {
                            op: BoolOp::Not,
                            args,
                        } => return args[0].clone(),
                        Expr::Bool {
                            op: BoolOp::Eq,
                            args,
                        } => {
                            return Expr::Bool {
                                op: BoolOp::Ne,
                                args: args.clone(),
                            };
                        }
                        Expr::Bool {
                            op: BoolOp::Ne,
                            args,
                        } => {
                            return Expr::Bool {
                                op: BoolOp::Eq,
                                args: args.clone(),
                            };
                        }
                        _ => {}
                    }
                }
                Expr::Un {
                    op: *op,
                    arg: Box::new(arg),
                    bits: *bits,
                }
            }
            Expr::Cast {
                kind,
                arg,
                from,
                to,
            } => {
                let arg = arg.simplify();
                if from == to {
                    return arg;
                }
                if let Expr::Const(v) = &arg {
                    // The kind decides what happens to the bits above the
                    // source width. Folding every cast to a truncation lost
                    // the sign: `sext(0xff, 1 byte)` is
                    // `0xffffffffffffffff`, not `0xff`.
                    return match kind {
                        CastKind::Zext => Expr::Const(truncate(*v, usize::from(*from))),
                        CastKind::Sext => Expr::Const(sign_extend(*v, *from) as u64),
                        CastKind::Trunc => Expr::Const(truncate(*v, usize::from(*to))),
                    };
                }
                // A widening cast of a widening cast collapses to the wider one.
                if let Expr::Cast {
                    kind: inner_kind,
                    arg: inner,
                    to: inner_to,
                    ..
                } = &arg
                    && inner_kind == kind
                    && *inner_to <= *from
                {
                    return Expr::Cast {
                        kind: *kind,
                        arg: inner.clone(),
                        from: *from,
                        to: *to,
                    };
                }
                Expr::Cast {
                    kind: *kind,
                    arg: Box::new(arg),
                    from: *from,
                    to: *to,
                }
            }
            Expr::Cond {
                op,
                lhs,
                rhs,
                bits,
            } => Expr::Cond {
                op: *op,
                lhs: Box::new(lhs.simplify()),
                rhs: Box::new(rhs.simplify()),
                bits: *bits,
            },
            Expr::Bool { op, args } => {
                let args: Vec<Expr> = args.iter().map(|a| a.simplify()).collect();
                match op {
                    BoolOp::Not => match &args[0] {
                        Expr::Const(v) => Expr::Const(u64::from(*v == 0)),
                        // The negation of a comparison is the opposite
                        // comparison, and the machine asks about the flags
                        // often enough that the spelling matters.
                        Expr::Cond {
                            op,
                            lhs,
                            rhs,
                            bits,
                        } => Expr::Cond {
                            op: op.negated(),
                            lhs: lhs.clone(),
                            rhs: rhs.clone(),
                            bits: *bits,
                        },
                        Expr::Bool {
                            op: BoolOp::Not,
                            args,
                        } => args[0].clone(),
                        Expr::Bool {
                            op: BoolOp::Eq,
                            args,
                        } => Expr::Bool {
                            op: BoolOp::Ne,
                            args: args.clone(),
                        },
                        Expr::Bool {
                            op: BoolOp::Ne,
                            args,
                        } => Expr::Bool {
                            op: BoolOp::Eq,
                            args: args.clone(),
                        },
                        _ => Expr::Bool {
                            op: BoolOp::Not,
                            args,
                        },
                    },
                    BoolOp::And | BoolOp::Or => {
                        // The conjunction has a unit and an absorber, and the
                        // disjunction has them the other way round.
                        let (unit, absorber) = if *op == BoolOp::And {
                            (1u64, 0u64)
                        } else {
                            (0u64, 1u64)
                        };
                        let mut kept: Vec<Expr> = Vec::with_capacity(args.len());
                        for a in args {
                            if let Expr::Const(v) = &a {
                                if *v == absorber {
                                    return Expr::Const(absorber);
                                }
                                if *v == unit {
                                    continue;
                                }
                            }
                            if kept.contains(&a) {
                                continue;
                            }
                            kept.push(a);
                        }
                        match kept.len() {
                            0 => Expr::Const(unit),
                            1 => kept.pop().expect("one operand"),
                            _ => Expr::Bool { op: *op, args: kept },
                        }
                    }
                    BoolOp::Eq | BoolOp::Ne => {
                        let (a, b) = (&args[0], &args[1]);
                        if a == b {
                            return Expr::Const(u64::from(*op == BoolOp::Eq));
                        }
                        // A boolean compared against zero is the boolean
                        // itself, and compared against one it is its negation.
                        if let Expr::Const(v) = b {
                            match (*op, *v) {
                                (BoolOp::Ne, 0) => return a.clone(),
                                (BoolOp::Eq, 1) => return a.clone(),
                                (BoolOp::Eq, 0) | (BoolOp::Ne, 1) => {
                                    return Expr::Bool {
                                        op: BoolOp::Not,
                                        args: vec![a.clone()],
                                    };
                                }
                                _ => {}
                            }
                        }
                        Expr::Bool { op: *op, args }
                    }
                }
            }
            Expr::FlagCond {
                cond,
                state,
                bits,
            } => {
                let state = state.simplify();
                // The condition the flags answer is the condition; keeping
                // both would say the same thing twice.
                match state.condition(*cond) {
                    Some(e) => e.simplify(),
                    None => Expr::FlagCond {
                        cond: *cond,
                        state: Box::new(state),
                        bits: *bits,
                    },
                }
            }
            Expr::Select { cond, yes, no } => {
                let cond = cond.simplify();
                let yes = yes.simplify();
                let no = no.simplify();
                if yes == no {
                    return yes;
                }
                Expr::Select {
                    cond: Box::new(cond),
                    yes: Box::new(yes),
                    no: Box::new(no),
                }
            }
            Expr::Member {
                base,
                offset,
                size,
                name,
            } => Expr::Member {
                base: Box::new(base.simplify()),
                offset: *offset,
                size: *size,
                name: name.clone(),
            },
            Expr::Load { addr, size, signed } => Expr::Load {
                addr: Box::new(addr.simplify()),
                size: *size,
                signed: *signed,
            },
            Expr::Intrinsic { op, args, bits } => {
                let args: Vec<Expr> = args.iter().map(|a| a.simplify()).collect();
                // An operation with every operand known is a value, and the
                // value is worth having.
                if args.iter().all(|a| matches!(a, Expr::Const(_)))
                    && let Some(v) = fold_intrinsic(*op, &args, *bits)
                {
                    return Expr::Const(v);
                }
                Expr::Intrinsic {
                    op: *op,
                    args,
                    bits: *bits,
                }
            }
            other => other.clone(),
        }
    }

    /// True when this is a constant.
    pub fn as_const(&self) -> Option<u64> {
        match self {
            Expr::Const(v) => Some(*v),
            _ => None,
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_prec(f, 0)
    }
}

/// Evaluate the intrinsics that need more than one operand.
///
/// These are the operations that work on parts of a word — the parallel
/// additions, the saturating ones, and the byte reversals — and they are
/// written once so that the folder and the interpreter cannot drift apart
/// in what they mean.
///
/// `bits` is the width of the operation in bytes.  For the parallel
/// operations it is also the width of the lane they work on, because that
/// is what the lane is.
pub fn eval_special(op: Intrinsic, values: &[u64], bits: u8) -> Option<u64> {
    let width = usize::from(bits) * 8;
    let limit = if width >= 64 { 64 } else { width };
    let value = values.first().copied().unwrap_or(0) & mask(bits);
    let result = match op {
        Intrinsic::RevBytes => {
            // The bytes come back the other way round. A word of four
            // bytes has them in the order three, two, one, zero.
            let bytes = value.to_le_bytes();
            let mut out = [0u8; 8];
            for (i, b) in bytes.iter().enumerate().take(usize::from(bits)) {
                out[usize::from(bits) - 1 - i] = *b;
            }
            u64::from_le_bytes(out)
        }
        Intrinsic::RevHalves => {
            // The halves are swapped and the bytes inside each half stay
            // where they are.
            let lane = u64::from(bits) * 4;
            let mask = if lane >= 64 { u64::MAX } else { (1u64 << lane) - 1 };
            let low = value & mask;
            let high = (value >> lane) & mask;
            (low << lane) | high
        }
        Intrinsic::RevHalvesSigned => {
            // The halves are swapped and the one that comes down is read as
            // a signed number.
            let lane = u64::from(bits) * 4;
            let mask = if lane >= 64 { u64::MAX } else { (1u64 << lane) - 1 };
            let low = value & mask;
            let high = (value >> lane) & mask;
            let swapped = (low << lane) | high;
            // The sign is the top bit of the half that came down, and the
            // result keeps it in every bit above the half.
            let sign = if lane >= 64 || (swapped >> (lane - 1)) & 1 == 1 {
                !mask
            } else {
                0
            };
            (swapped & mask) | sign
        }
        Intrinsic::RevBits => {
            // The bits come back the other way round: the bottom bit
            // becomes the top one.
            let mut out = 0u64;
            for i in 0..limit {
                if (value >> i) & 1 == 1 {
                    out |= 1u64 << (limit - 1 - i);
                }
            }
            out
        }
        Intrinsic::Select => {
            // The mask says which of the two values each bit comes from.
            let (Some(mask_value), Some(then_value), Some(else_value)) = (
                values.first().copied(),
                values.get(1).copied(),
                values.get(2).copied(),
            ) else {
                return None;
            };
            (mask_value & then_value) | (!mask_value & else_value) & mask(bits)
        }
        Intrinsic::Crc32 | Intrinsic::Crc32c => {
            // The checksum folds the second operand into the first, one
            // bit at a time.  Every bit that comes out of the bottom
            // goes back in through the polynomial, and the polynomial
            // is the one the ordinary checksum of a zip file uses with
            // its bits the other way round.
            let acc = values.first().copied().unwrap_or(0);
            // The width of the data is the width of the operand that is
            // folded in: a byte, a half word or a word.  It is given as
            // the third operand, because the width of the result is the
            // width of the machine's word and not the width of the data.
            let width = values.get(2).copied().unwrap_or(32).clamp(1, 32) as u32;
            let data_mask = if width >= 32 {
                0xffff_ffffu64
            } else {
                (1u64 << width) - 1
            };
            let data = values.get(1).copied().unwrap_or(0) & data_mask;
            let poly: u64 = if op == Intrinsic::Crc32c {
                0x82f6_3b78
            } else {
                0xedb8_8320
            };
            let mut value = (acc ^ data) & 0xffff_ffff;
            for _ in 0..width {
                if value & 1 == 1 {
                    value = (value >> 1) ^ poly;
                } else {
                    value >>= 1;
                }
            }
            value & 0xffff_ffff
        }
        Intrinsic::SatAdd => {
            let a = signed_int(values.first().copied().unwrap_or(0), bits);
            let b = signed_int(values.get(1).copied().unwrap_or(0), bits);
            let top = (1i64 << (limit as i32 - 1)) - 1;
            let bottom = -(1i64 << (limit as i32 - 1));
            (a.saturating_add(b).clamp(bottom, top) as u64) & mask(bits)
        }
        Intrinsic::SatSub => {
            let a = signed_int(values.first().copied().unwrap_or(0), bits);
            let b = signed_int(values.get(1).copied().unwrap_or(0), bits);
            let top = (1i64 << (limit as i32 - 1)) - 1;
            let bottom = -(1i64 << (limit as i32 - 1));
            (a.saturating_sub(b).clamp(bottom, top) as u64) & mask(bits)
        }
        Intrinsic::ExchangeAdd
        | Intrinsic::ExchangeSub
        | Intrinsic::ExchangeAddSat
        | Intrinsic::ExchangeSubSat
        | Intrinsic::ExchangeAddHalf
        | Intrinsic::ExchangeSubHalf
        | Intrinsic::ExchangeAddUns
        | Intrinsic::ExchangeSubUns
        | Intrinsic::ExchangeAddSatUns
        | Intrinsic::ExchangeSubSatUns
        | Intrinsic::ExchangeAddHalfUns
        | Intrinsic::ExchangeSubHalfUns => {
            // The exchanged operations take the halves of the two
            // operands in the opposite order: the high half of the result
            // is the operation on the high half of the first operand and
            // the low half of the second, and the low half of the result
            // is the operation the other way round.
            let lane_bits = usize::from(bits) * 8;
            if lane_bits == 0 || lane_bits > 16 {
                return None;
            }
            let a = values.first().copied().unwrap_or(0);
            let b = values.get(1).copied().unwrap_or(0);
            let lane_mask = (1u64 << lane_bits) - 1;
            // The high half of the first operand and the low half of the
            // second make the high half of the result, and the low half of
            // the first and the high half of the second make the low half
            // of it.
            let high_x = (a >> lane_bits) & lane_mask;
            let low_x = a & lane_mask;
            let second_low = b & lane_mask;
            let second_high = (b >> lane_bits) & lane_mask;
            let is_add = matches!(
                op,
                Intrinsic::ExchangeAdd
                    | Intrinsic::ExchangeAddSat
                    | Intrinsic::ExchangeAddHalf
                    | Intrinsic::ExchangeAddUns
                    | Intrinsic::ExchangeAddSatUns
                    | Intrinsic::ExchangeAddHalfUns
            );
            let saturated = matches!(
                op,
                Intrinsic::ExchangeAddSat
                    | Intrinsic::ExchangeSubSat
                    | Intrinsic::ExchangeAddSatUns
                    | Intrinsic::ExchangeSubSatUns
            );
            let halved = matches!(
                op,
                Intrinsic::ExchangeAddHalf
                    | Intrinsic::ExchangeSubHalf
                    | Intrinsic::ExchangeAddHalfUns
                    | Intrinsic::ExchangeSubHalfUns
            );
            let unsigned = matches!(
                op,
                Intrinsic::ExchangeAddUns
                    | Intrinsic::ExchangeSubUns
                    | Intrinsic::ExchangeAddSatUns
                    | Intrinsic::ExchangeSubSatUns
                    | Intrinsic::ExchangeAddHalfUns
                    | Intrinsic::ExchangeSubHalfUns
            );
            // The high half of the result takes the high half of the
            // first operand and the low half of the second, and the low
            // half of the result takes them the other way round.
            let high = parallel_lane(
                high_x,
                second_low,
                lane_bits,
                is_add,
                unsigned,
                saturated,
                halved,
            );
            let low = parallel_lane(
                low_x,
                second_high,
                lane_bits,
                !is_add,
                unsigned,
                saturated,
                halved,
            );
            ((high & lane_mask) << lane_bits) | (low & lane_mask)
        }
        Intrinsic::ExtendAdd | Intrinsic::ExtendAddSigned => {
            // The word is cut into halves and each half of the second
            // operand is extended and added to the matching half of the
            // first.  The carry of one half never reaches the other, so
            // this is not one addition but two.
            //
            // The extension takes the low byte of the half, and the
            // half is sixteen bits wide.
            let lane_bits = 16usize;
            let a = values.first().copied().unwrap_or(0);
            let b = values.get(1).copied().unwrap_or(0);
            let lane_mask = (1u64 << lane_bits) - 1;
            let signed = op == Intrinsic::ExtendAddSigned;
            let mut out = 0u64;
            for lane in 0..2 {
                let shift = lane * lane_bits;
                let x = (a >> shift) & lane_mask;
                let y = (b >> shift) & lane_mask;
                // Only the low byte of the half is extended.
                let extended = if signed {
                    i128::from(sign_extend_bits(y & 0xff, 8))
                } else {
                    i128::from(y & 0xff)
                };
                let value = (i128::from(x) + extended) & lane_mask as i128;
                out |= (value as u64) << shift;
            }
            out & 0xffff_ffff
        }
        Intrinsic::ParallelAdd
        | Intrinsic::ParallelSub
        | Intrinsic::ParallelAddSat
        | Intrinsic::ParallelSubSat
        | Intrinsic::ParallelAddHalf
        | Intrinsic::ParallelSubHalf
        | Intrinsic::ParallelAddUns
        | Intrinsic::ParallelSubUns
        | Intrinsic::ParallelAddSatUns
        | Intrinsic::ParallelSubSatUns
        | Intrinsic::ParallelAddHalfUns
        | Intrinsic::ParallelSubHalfUns => {
            // The word is cut into lanes and each lane is worked on by
            // itself, so that the carry of one lane never reaches the
            // next.
            //
            // The width of the lane is the one the operation was given;
            // the width of the word is the one of the operands, which is
            // the width of the machine's word.
            let lane_bits = usize::from(bits) * 8;
            if lane_bits == 0 || lane_bits > 32 {
                return None;
            }
            let a = values.first().copied().unwrap_or(0);
            let b = values.get(1).copied().unwrap_or(0);
            let word_bits = 32;
            let lanes = word_bits / lane_bits;
            let lane_mask = (1u64 << lane_bits) - 1;
            let is_add = matches!(
                op,
                Intrinsic::ParallelAdd
                    | Intrinsic::ParallelAddSat
                    | Intrinsic::ParallelAddHalf
                    | Intrinsic::ParallelAddUns
                    | Intrinsic::ParallelAddSatUns
                    | Intrinsic::ParallelAddHalfUns
            );
            let saturated = matches!(
                op,
                Intrinsic::ParallelAddSat
                    | Intrinsic::ParallelSubSat
                    | Intrinsic::ParallelAddSatUns
                    | Intrinsic::ParallelSubSatUns
            );
            let halved = matches!(
                op,
                Intrinsic::ParallelAddHalf
                    | Intrinsic::ParallelSubHalf
                    | Intrinsic::ParallelAddHalfUns
                    | Intrinsic::ParallelSubHalfUns
            );
            let unsigned = matches!(
                op,
                Intrinsic::ParallelAddUns
                    | Intrinsic::ParallelSubUns
                    | Intrinsic::ParallelAddSatUns
                    | Intrinsic::ParallelSubSatUns
                    | Intrinsic::ParallelAddHalfUns
                    | Intrinsic::ParallelSubHalfUns
            );
            let mut out = 0u64;
            for lane in 0..lanes {
                let shift = lane * lane_bits;
                let x = (a >> shift) & lane_mask;
                let y = (b >> shift) & lane_mask;
                let v = parallel_lane(x, y, lane_bits, is_add, unsigned, saturated, halved);
                out |= v << shift;
            }
            out & 0xffff_ffff
        }
        _ => return None,
    };
    // The parallel operations work on the lanes of a word, so their
    // result is the whole word and not the width of a lane.
    let result = match op {
        Intrinsic::ParallelAdd
        | Intrinsic::ParallelSub
        | Intrinsic::ParallelAddSat
        | Intrinsic::ParallelSubSat
        | Intrinsic::ParallelAddHalf
        | Intrinsic::ParallelSubHalf
        | Intrinsic::ParallelAddUns
        | Intrinsic::ParallelSubUns
        | Intrinsic::ParallelAddSatUns
        | Intrinsic::ParallelSubSatUns
        | Intrinsic::ParallelAddHalfUns
        | Intrinsic::ParallelSubHalfUns
        | Intrinsic::ExchangeAdd
        | Intrinsic::ExchangeSub
        | Intrinsic::ExchangeAddSat
        | Intrinsic::ExchangeSubSat
        | Intrinsic::ExchangeAddHalf
        | Intrinsic::ExchangeSubHalf
        | Intrinsic::ExchangeAddUns
        | Intrinsic::ExchangeSubUns
        | Intrinsic::ExchangeAddSatUns
        | Intrinsic::ExchangeSubSatUns
        | Intrinsic::ExchangeAddHalfUns
        | Intrinsic::ExchangeSubHalfUns => result & 0xffff_ffff,
        _ => result & mask(bits),
    };
    Some(result)
}

/// Fold an intrinsic whose operands are all known.
///
/// The semantics are the machine's, not the arithmetic's: a rotate masks
/// its count first, a bit scan of zero leaves the destination alone, and a
/// count of trailing zeros of zero is the width of the operand rather than
/// an error.
fn fold_intrinsic(op: Intrinsic, args: &[Expr], bits: u8) -> Option<u64> {
    let values: Vec<u64> = args
        .iter()
        .map(|a| match a {
            Expr::Const(v) => Some(*v),
            _ => None,
        })
        .collect::<Option<Vec<u64>>>()?;
    let width = usize::from(bits) * 8;
    let limit = if width >= 64 { 64 } else { width };
    let value = values[0] & mask(bits);
    Some(match op {
        Intrinsic::Rol | Intrinsic::Ror => {
            // The count is masked to the width of the operand before the
            // rotation, so a count of 64 on a 32-bit operand is a no-op.
            let count = values.get(1).copied().unwrap_or(0) & (limit as u64 - 1);
            if count == 0 {
                value
            } else {
                let n = count as u32;
                if op == Intrinsic::Rol {
                    ((value << n) | (value >> (limit as u32 - n))) & mask(bits)
                } else {
                    ((value >> n) | (value << (limit as u32 - n))) & mask(bits)
                }
            }
        }
        Intrinsic::Rcl | Intrinsic::Rcr => {
            // The carry is a ninth bit on a byte, a seventeenth on a word,
            // and so on: it joins the value at the top and comes back out of
            // the bottom, so the count is taken modulo one more than the
            // width.
            let count = values.get(1).copied().unwrap_or(0) & 31;
            let carry = values.get(2).copied().unwrap_or(0) & 1;
            let span = limit as u32 + 1;
            let n = (count as u32) % span;
            if n == 0 {
                value
            } else if span >= 64 {
                // A 64-bit rotate through the carry does not fit in a word;
                // leave it to the interpreter rather than fold it wrong.
                return None;
            } else {
                let all = (1u128 << span) - 1;
                let wide = (u128::from(value) | (u128::from(carry) << limit)) & all;
                let turned = if op == Intrinsic::Rcl {
                    ((wide << n) | (wide >> (span - n))) & all
                } else {
                    ((wide >> n) | (wide << (span - n))) & all
                };
                (turned as u64) & mask(bits)
            }
        }
        Intrinsic::Bswap => {
            let bytes = value.to_le_bytes();
            let mut out = [0u8; 8];
            for (i, b) in bytes.iter().enumerate().take(usize::from(bits)) {
                out[usize::from(bits) - 1 - i] = *b;
            }
            u64::from_le_bytes(out)
        }
        Intrinsic::Bsf => {
            if value == 0 {
                // The machine leaves the destination alone when there is no
                // set bit, so there is no value to report.
                return None;
            }
            value.trailing_zeros() as u64
        }
        Intrinsic::Bsr => {
            if value == 0 {
                return None;
            }
            (limit as u32 - 1 - value.leading_zeros()) as u64
        }
        Intrinsic::ParityEven => u64::from((value as u8).count_ones() % 2 == 0),
        Intrinsic::Carry => {
            // The sum did not fit when the result is smaller than the value
            // that was there before.
            let previous = values.get(1).copied().unwrap_or(0) & mask(bits);
            u64::from(value < previous)
        }
        Intrinsic::Borrow => {
            // The difference did not fit when the minuend was smaller than
            // the value subtracted from it.
            let previous = values.get(1).copied().unwrap_or(0) & mask(bits);
            u64::from(value < previous)
        }
        Intrinsic::OverflowAdd => {
            // The sum has the other sign than the operands: the sign bit of
            // `(a ^ result) & (b ^ result)`.
            let a = values.first().copied().unwrap_or(0) & mask(bits);
            let b = values.get(1).copied().unwrap_or(0) & mask(bits);
            let r = values.get(2).copied().unwrap_or(0) & mask(bits);
            u64::from((a ^ r) & (b ^ r) & (1u64 << (limit as u32 - 1)) != 0)
        }
        Intrinsic::OverflowSub => {
            // The difference has the other sign than the minuend: the sign
            // bit of `(a ^ b) & (a ^ result)`.
            let a = values.first().copied().unwrap_or(0) & mask(bits);
            let b = values.get(1).copied().unwrap_or(0) & mask(bits);
            let r = values.get(2).copied().unwrap_or(0) & mask(bits);
            u64::from((a ^ b) & (a ^ r) & (1u64 << (limit as u32 - 1)) != 0)
        }
        Intrinsic::Tzcnt => value.trailing_zeros() as u64,
        Intrinsic::Lzcnt => (limit as u32 - value.leading_zeros()) as u64,
        Intrinsic::Popcnt => value.count_ones() as u64,
        Intrinsic::CopySign => {
            // The magnitude of the first operand with the sign of the
            // second: every bit but the top one comes from the first, and
            // the top one from the second.
            let top = 1u64 << (limit.saturating_sub(1));
            let sign = values.get(1).copied().unwrap_or(0) & top;
            (value & !top) | sign
        }
        Intrinsic::MovMask => {
            // Each lane contributes its sign bit. The width is the lane's,
            // not the machine's.
            let lane_bits = usize::from(bits) * 8;
            let mut mask_value = 0u64;
            for (i, lane) in values.iter().enumerate() {
                let sign = (lane >> (lane_bits - 1)) & 1;
                mask_value |= sign << i;
            }
            mask_value
        }
        Intrinsic::MulLo | Intrinsic::MulHi => {
            let a = values.first().copied().unwrap_or(0) & mask(bits);
            let b = values.get(1).copied().unwrap_or(0) & mask(bits);
            let product = u128::from(a) * u128::from(b);
            if op == Intrinsic::MulLo {
                (product as u64) & mask(bits)
            } else {
                ((product >> (usize::from(bits) * 8)) as u64) & mask(bits)
            }
        }
        Intrinsic::IMulLo | Intrinsic::IMulHi => {
            // Signed, so the product of two negatives is positive and the high
            // half is not the upper bits of an unsigned product.
            let a = i128::from(signed_int(values.first().copied().unwrap_or(0), bits));
            let b = i128::from(signed_int(values.get(1).copied().unwrap_or(0), bits));
            let product = a.wrapping_mul(b);
            if op == Intrinsic::IMulLo {
                (product as u64) & mask(bits)
            } else {
                ((product >> (usize::from(bits) * 8)) as u64) & mask(bits)
            }
        }
        Intrinsic::AdcSum | Intrinsic::AdcCarry | Intrinsic::SbbDiff | Intrinsic::SbbBorrow => {
            let a = values.first().copied().unwrap_or(0) & mask(bits);
            let b = values.get(1).copied().unwrap_or(0) & mask(bits);
            let c = values.get(2).copied().unwrap_or(0) & 1;
            match op {
                Intrinsic::AdcSum => a.wrapping_add(b).wrapping_add(c) & mask(bits),
                Intrinsic::AdcCarry => {
                    u64::from(u128::from(a) + u128::from(b) + u128::from(c) > u128::from(mask(bits)))
                }
                Intrinsic::SbbDiff => a.wrapping_sub(b).wrapping_sub(c) & mask(bits),
                _ => u64::from(u128::from(a) < u128::from(b) + u128::from(c)),
            }
        }
        Intrinsic::Div | Intrinsic::IDiv | Intrinsic::DivRem | Intrinsic::IDivRem => {
            let (Some(high), Some(low), Some(divisor)) = (
                values.first().copied(),
                values.get(1).copied(),
                values.get(2).copied(),
            ) else {
                return None;
            };
            if divisor == 0 {
                // The machine faults; there is no result.
                return None;
            }
            // The pair is one number of twice the operand width, and the
            // halves are shifted by that width, not by 64: a byte division
            // divides a word, not a double word.
            let span = u32::from(bits) * 2;
            let dividend =
                (u128::from(truncate(high, usize::from(bits))) << span) | u128::from(low);
            let wants_remainder = op == Intrinsic::DivRem || op == Intrinsic::IDivRem;
            let signed = op == Intrinsic::IDiv || op == Intrinsic::IDivRem;
            if signed {
                // The sign of the dividend is the top bit of the pair, and the
                // divisor is signed at the operand width.
                let dividend = if span >= 128 {
                    dividend as i128
                } else if (dividend >> (span - 1)) & 1 == 1 {
                    (dividend | (!0u128 << span)) as i128
                } else {
                    dividend as i128
                };
                let divisor = i128::from(signed_int(divisor, bits));
                if wants_remainder {
                    dividend.wrapping_rem(divisor) as u64
                } else {
                    dividend.wrapping_div(divisor) as u64
                }
            } else if wants_remainder {
                (dividend % u128::from(divisor)) as u64
            } else {
                (dividend / u128::from(divisor)) as u64
            }
        }
        Intrinsic::RevBytes
        | Intrinsic::RevHalves
        | Intrinsic::RevHalvesSigned
        | Intrinsic::RevBits
        | Intrinsic::Select
        | Intrinsic::Crc32
        | Intrinsic::Crc32c
        | Intrinsic::SatAdd
        | Intrinsic::SatSub
        | Intrinsic::ExtendAdd
        | Intrinsic::ExtendAddSigned
        | Intrinsic::ParallelAdd
        | Intrinsic::ParallelSub
        | Intrinsic::ParallelAddSat
        | Intrinsic::ParallelSubSat
        | Intrinsic::ParallelAddHalf
        | Intrinsic::ParallelSubHalf
        | Intrinsic::ParallelAddUns
        | Intrinsic::ParallelSubUns
        | Intrinsic::ParallelAddSatUns
        | Intrinsic::ParallelSubSatUns
        | Intrinsic::ParallelAddHalfUns
        | Intrinsic::ParallelSubHalfUns
        | Intrinsic::ExchangeAdd
        | Intrinsic::ExchangeSub
        | Intrinsic::ExchangeAddSat
        | Intrinsic::ExchangeSubSat
        | Intrinsic::ExchangeAddHalf
        | Intrinsic::ExchangeSubHalf
        | Intrinsic::ExchangeAddUns
        | Intrinsic::ExchangeSubUns
        | Intrinsic::ExchangeAddSatUns
        | Intrinsic::ExchangeSubSatUns
        | Intrinsic::ExchangeAddHalfUns
        | Intrinsic::ExchangeSubHalfUns => eval_special(op, &values, bits)?,
    })
}

/// One lane of a parallel operation.
///
/// The lane is worked on by itself: the addition or the subtraction does
/// not carry into the next lane.  Three things can happen to the result
/// before it goes back into the lane, and the machine applies them in this
/// order:
///
/// * the lanes are read as signed numbers or as unsigned ones;
/// * the result is held at the ends of the lane's range, or it wraps
///   around;
/// * the result is halved, and the halving is an arithmetic one on a
///   result that is one bit wider than the lane, so that the bit that was
///   carried out of the lane is kept.
fn parallel_lane(
    x: u64,
    y: u64,
    lane_bits: usize,
    add: bool,
    unsigned: bool,
    saturated: bool,
    halved: bool,
) -> u64 {
    let mask = (1u64 << lane_bits) - 1;
    // The lanes are read as signed numbers or as unsigned ones, and the
    // operation is carried out at one bit more than the lane, because
    // the halving is done on that wider value: the bit that was
    // carried out of the lane has to survive it.
    //
    // The wider value is signed even when the lanes are read as
    // unsigned ones.  The machine takes the difference of two
    // unsigned lanes and then halves it, and a difference that went
    // below zero is halved as a negative number, so `0 - 1` halved is
    // `-1` and not `0x7f`.  The two are the same number written
    // differently, but they are not the same number.
    let (sx, sy, low, high) = if unsigned {
        (i128::from(x), i128::from(y), 0, i128::from(mask))
    } else {
        (
            i128::from(sign_extend_bits(x, lane_bits as u32)),
            i128::from(sign_extend_bits(y, lane_bits as u32)),
            -(1i128 << (lane_bits - 1)),
            (1i128 << (lane_bits - 1)) - 1,
        )
    };
    let total = if add { sx + sy } else { sx - sy };
    let value = if saturated {
        total.clamp(low, high)
    } else {
        total
    };
    if halved {
        // The halving is done on the value of the operation before it
        // is cut back to the width of the lane, and the shift is an
        // arithmetic one, so that the bit which was carried out of the
        // lane is kept and a result that went below the smallest lane
        // value stays negative.
        return ((value >> 1) as u64) & mask;
    }
    (value as u64) & mask
}

/// Fold one binary operation at a given width.
fn fold(op: BinOp, a: u64, b: u64, bits: u8) -> Option<u64> {
    if op.is_float() {
        return fold_float(op, a, b, bits);
    }
    let shift = u32::from(bits.saturating_mul(8).clamp(8, 64));
    // The operation happens at the width of the operand. The bits above it
    // are not part of the value, so they are dropped before the operation
    // and not after it: `0xffffffff + 1` in 32 bits is zero, and the flags
    // the operation leaves have to agree with that.
    let width = usize::from(bits).clamp(1, 8);
    let a = truncate(a, width);
    let b = truncate(b, width);
    let value = match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::Div => {
            if b == 0 {
                return None;
            }
            (a as i64).wrapping_div(b as i64) as u64
        }
        BinOp::UDiv => {
            if b == 0 {
                return None;
            }
            a / b
        }
        BinOp::Mod => {
            if b == 0 {
                return None;
            }
            (a as i64).wrapping_rem(b as i64) as u64
        }
        BinOp::UMod => {
            if b == 0 {
                return None;
            }
            a % b
        }
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        BinOp::Shl => {
            if b >= u64::from(shift) {
                0
            } else {
                a << b
            }
        }
        BinOp::Shr => {
            if b >= u64::from(shift) {
                0
            } else {
                (truncate(a, bits as usize) & mask(bits)) >> b
            }
        }
        BinOp::Sar => {
            if b >= u64::from(shift) {
                if (a as i64) < 0 { u64::MAX } else { 0 }
            } else {
                let signed = sign_extend(a, bits);
                ((signed >> b) as u64) & mask(bits)
            }
        }
        // Handled before the match: the floating-point operations have their
        // own semantics and their own width.
        _ => return None,
    };
    Some(truncate(value, width))
}

/// Fold a floating-point operation.
///
/// The operands are bit patterns. A width of four bytes is a `float`, eight
/// bytes a `double`; the result of a comparison is zero or one, and the
/// result of an arithmetic operation is the bit pattern of the value.
///
/// The comparisons are the *ordered* ones: a comparison involving a
/// not-a-number is false. The unordered case has its own operation, because
/// the machine reports it through a different flag and a branch can ask
/// about it separately.
fn fold_float(op: BinOp, a: u64, b: u64, bits: u8) -> Option<u64> {
    match bits {
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
                BinOp::FMin => u64::from(min_f32(x, y).to_bits()),
                BinOp::FMax => u64::from(max_f32(x, y).to_bits()),
                BinOp::FSqrt => u64::from(x.sqrt().to_bits()),
                BinOp::I2F32 => u64::from((signed_int(a, bits) as f32).to_bits()),
                BinOp::I2F64 => (signed_int(a, bits) as f64).to_bits(),
                BinOp::F2I32 => u64::from((x as i32) as u32),
                BinOp::F2I64 => (x as i64) as u64,
                BinOp::F2U32 => u64::from(x as u32),
                BinOp::F2U64 => x as u64,
                BinOp::FCeil => u64::from(x.ceil().to_bits()),
                BinOp::FFloor => u64::from(x.floor().to_bits()),
                BinOp::FRound => u64::from(x.round_ties_even().to_bits()),
                BinOp::F2F32 => u64::from(x.to_bits()),
                BinOp::F2F64 => (x as f64).to_bits(),
                BinOp::FNeg => u64::from((-x).to_bits()),
                BinOp::FAbs => u64::from(x.abs().to_bits()),
                // An 80-bit value does not fit a machine word, so it is never
                // folded.
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
                BinOp::FMin => min_f64(x, y).to_bits(),
                BinOp::FMax => max_f64(x, y).to_bits(),
                BinOp::FSqrt => x.sqrt().to_bits(),
                BinOp::I2F32 => u64::from((signed_int(a, bits) as f32).to_bits()),
                BinOp::I2F64 => (signed_int(a, bits) as f64).to_bits(),
                BinOp::F2I32 => u64::from((x as i32) as u32),
                BinOp::F2I64 => (x as i64) as u64,
                BinOp::F2U32 => u64::from(x as u32),
                BinOp::F2U64 => x as u64,
                BinOp::FCeil => x.ceil().to_bits(),
                BinOp::FFloor => x.floor().to_bits(),
                BinOp::FRound => x.round_ties_even().to_bits(),
                BinOp::F2F32 => u64::from((x as f32).to_bits()),
                BinOp::F2F64 => x.to_bits(),
                BinOp::FNeg => (-x).to_bits(),
                BinOp::FAbs => x.abs().to_bits(),
                // An 80-bit value does not fit a machine word, so it is never
                // folded.
                _ => return None,
            })
        }
        _ => None,
    }
}

/// The smaller of two single-precision values, the way the machine
/// computes it.
///
/// The machine's minimum is not a comparison: it returns the second
/// operand when either one is a not-a-number, and when both are zero.
/// The sign of the zero it returns follows the second operand, which is
/// what the rule says and what a comparison-based minimum does not do.
fn min_f32(x: f32, y: f32) -> f32 {
    if x.is_nan() || y.is_nan() {
        return y;
    }
    if x == 0.0 && y == 0.0 {
        return y;
    }
    if x < y { x } else { y }
}

/// The larger of two single-precision values, the way the machine
/// computes it.
fn max_f32(x: f32, y: f32) -> f32 {
    if x.is_nan() || y.is_nan() {
        return y;
    }
    if x == 0.0 && y == 0.0 {
        return y;
    }
    if x > y { x } else { y }
}

/// The smaller of two double-precision values, the way the machine
/// computes it.
fn min_f64(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return y;
    }
    if x == 0.0 && y == 0.0 {
        return y;
    }
    if x < y { x } else { y }
}

/// The larger of two double-precision values, the way the machine
/// computes it.
fn max_f64(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return y;
    }
    if x == 0.0 && y == 0.0 {
        return y;
    }
    if x > y { x } else { y }
}

/// Read a value as a signed integer of the given width.
fn signed_int(v: u64, bits: u8) -> i64 {
    sign_extend(v, bits)
}

/// Mask for a width in bytes, saturating at 64 bits.
pub fn mask(bits: u8) -> u64 {
    if bits >= 8 {
        u64::MAX
    } else {
        (1u64 << (u32::from(bits) * 8)) - 1
    }
}

/// Truncate to a width, as the machine would.
pub fn truncate(v: u64, bits: usize) -> u64 {
    if bits >= 8 { v } else { v & ((1u64 << (bits * 8)) - 1) }
}

/// Sign-extend from a width.
pub fn sign_extend(v: u64, bits: u8) -> i64 {
    let bits = u32::from(bits.clamp(1, 8)) * 8;
    if bits >= 64 {
        v as i64
    } else {
        let shift = 64 - bits;
        ((v << shift) as i64) >> shift
    }
}

/// Sign extend a value whose width is given in bits rather than in bytes.
///
/// The parallel operations work on lanes that are narrower than the word
/// they live in, so the sign of a lane is not the sign of the word.
fn sign_extend_bits(v: u64, bits: u32) -> i64 {
    if bits >= 64 {
        v as i64
    } else if bits == 0 {
        0
    } else {
        let shift = 64 - bits;
        ((v << shift) as i64) >> shift
    }
}

/// Compare two values the way a comparison of `bits`-wide operands does.
///
/// The width is not decoration. The same bits read as a signed number and
/// as an unsigned number are two different numbers, and a branch that asks
/// "below" asks one of them. The operands are truncated first, because the
/// machine compared the low bits and nothing else.
pub fn holds_at(op: CondOp, a: u64, b: u64, bits: u8) -> bool {
    let bytes = usize::from(bits).clamp(1, 8);
    let a = truncate(a, bytes);
    let b = truncate(b, bytes);
    match op {
        CondOp::Eq => a == b,
        CondOp::Ne => a != b,
        CondOp::Slt => sign_extend(a, bits) < sign_extend(b, bits),
        CondOp::Sle => sign_extend(a, bits) <= sign_extend(b, bits),
        CondOp::Sgt => sign_extend(a, bits) > sign_extend(b, bits),
        CondOp::Sge => sign_extend(a, bits) >= sign_extend(b, bits),
        CondOp::Ult => a < b,
        CondOp::Ule => a <= b,
        CondOp::Ugt => a > b,
        CondOp::Uge => a >= b,
    }
}

/// C type name for a width.
pub fn type_name(size: u8, signed: bool) -> &'static str {
    match (size, signed) {
        (1, false) => "u8",
        (1, true) => "i8",
        (2, false) => "u16",
        (2, true) => "i16",
        (4, false) => "u32",
        (4, true) => "i32",
        (8, false) => "u64",
        (8, true) => "i64",
        // The x87 stack's extended precision: ten bytes, with a five-byte
        // exponent and an explicit integer bit.
        (10, false) => "u80",
        (10, true) => "i80",
        _ => "u8",
    }
}

/// How a block ends.
#[derive(Clone, Debug, PartialEq)]
pub enum Term {
    /// Falls into one block.
    Jump(BlockId),
    /// Two-way branch.
    ///
    /// A side is `None` when that path leaves the function: a compiler
    /// puts the cold half of a branch in a separate chunk, and a branch
    /// into it is a jump out of the function. Naming the other side
    /// anyway is what keeps the fall-through: without it the block has no
    /// successor, the next block loses its predecessor, and every value
    /// the branch carried into it is undefined.
    Branch {
        /// Condition.
        cond: Expr,
        /// Taken when the condition holds.
        then_bb: Option<BlockId>,
        /// Taken otherwise.
        else_bb: Option<BlockId>,
    },
    /// Multi-way branch on `index`: case `i` goes to `targets[i]`.
    ///
    /// Produced only when the jump table behind the branch was recovered, so
    /// the number of cases and their order are the ones the compiler emitted.
    Switch {
        /// Case index.
        index: Expr,
        /// One target per case, in table order.
        targets: Vec<BlockId>,
        /// The case value of the first entry, when the code said which one
        /// it is.
        ///
        /// A compiler that sees `case 100..106` subtracts 100 from the index
        /// before it looks anything up, so the table holds positions and the
        /// cases are the positions plus 100. The subtraction is the only
        /// statement that says so, and without it a switch renders its
        /// cases numbered from zero whatever they were.
        first_case: Option<i64>,
    },
    /// Function exit.
    Ret(Option<Expr>),
    /// Computed jump whose target no recovered table explains.
    ///
    /// `target` is the address the jump reads, as an expression over the
    /// registers the block computed: `goto *rax` carries `rax`, and
    /// `goto *(%rdx,%rax,8)` carries the load. The edge itself is unknown —
    /// which is why this has no successors — but the *value* is not, and a
    /// client that reasons about values (a dispatch table, a switch the
    /// syntactic resolver missed) needs it.
    Indirect {
        /// Address the jump goes to.
        target: Expr,
    },
    /// Control leaves in a way the lifter could not model.
    Unreachable,
}

/// One basic block after lifting.
#[derive(Clone, Debug)]
pub struct Block {
    /// Index within the function.
    pub id: BlockId,
    /// Address of the first instruction.
    pub addr: Addr,
    /// Straight-line statements, in program order.
    pub stmts: Vec<Stmt>,
    /// How control leaves.
    pub term: Term,
}

/// One lifted statement.
#[derive(Clone, Debug)]
pub enum Stmt {
    /// SSA definition; the destination is a versioned name.
    Assign {
        /// Destination variable name.
        dest: String,
        /// Value assigned.
        value: Expr,
    },
    /// Memory write.
    Store {
        /// Address written.
        addr: Expr,
        /// Value stored.
        value: Expr,
        /// Access width in bytes.
        size: u8,
    },
    /// Call to a known or unknown target.
    Call {
        /// Rendered callee name.
        name: String,
        /// Arguments, in ABI order.
        args: Vec<Expr>,
        /// Destination for the return value, when the result is used.
        ret: Option<String>,
        /// Address of the call instruction.
        addr: Addr,
        /// Callee, when the instruction names one directly.
        ///
        /// `addr` is where the call is, not where it goes: a symbolic
        /// executor that wants to run the callee needs the target, and
        /// recovering it from the printed name would be a guess.
        target: Option<Addr>,
    },
    /// A statement that runs only when a condition holds.
    ///
    /// ARM32 executes most instructions conditionally, and `streq r0, [r1]`
    /// is one instruction that writes memory on one path and does nothing on
    /// the other. A select cannot express it — the store either happens or
    /// it does not — and a comment would drop the write. The bodies hold
    /// stores and calls, which is what a conditional instruction can do;
    /// conditional values go through [`Expr::Select`].
    If {
        /// Predicate.
        cond: Expr,
        /// Statements when it holds.
        then_body: Vec<Stmt>,
        /// Statements when it does not.
        else_body: Vec<Stmt>,
    },
    /// Side-effecting operation the lifter keeps as a comment.
    Comment(String),
}

/// Structured statement, the shape the renderer prints.
#[derive(Clone, Debug)]
pub enum Node {
    /// Assignment to a named variable.
    Assign {
        /// Destination name.
        dest: String,
        /// Value.
        value: Expr,
    },
    /// Memory write.
    Store {
        /// Address.
        addr: Expr,
        /// Value.
        value: Expr,
        /// Width in bytes.
        size: u8,
    },
    /// Call statement or call expression.
    Call {
        /// Callee.
        name: String,
        /// Arguments.
        args: Vec<Expr>,
        /// Destination, when the result is assigned.
        ret: Option<String>,
    },
    /// Conditional.
    If {
        /// Condition.
        cond: Expr,
        /// Body when true.
        then_body: Vec<Node>,
        /// Body when false.
        else_body: Vec<Node>,
    },
    /// Pre-tested loop.
    While {
        /// Condition.
        cond: Expr,
        /// Body.
        body: Vec<Node>,
    },
    /// Post-tested loop.
    DoWhile {
        /// Body.
        body: Vec<Node>,
        /// Condition.
        cond: Expr,
    },
    /// Multi-way branch on a value.
    Switch {
        /// Discriminant.
        expr: Expr,
        /// Case values and bodies.
        cases: Vec<(u64, Vec<Node>)>,
        /// Body when nothing matches.
        default: Vec<Node>,
    },
    /// Function return.
    Return(Option<Expr>),
    /// Unconditional jump to a label.
    Goto(String),
    /// Unconditional jump to a computed address.
    ComputedGoto(Expr),
    /// Label definition.
    Label(String),
    /// Break out of the innermost loop.
    Break,
    /// Next iteration of the innermost loop.
    Continue,
    /// A statement that does nothing but is kept for shape.
    Nop,
    /// A remark about something that was not lifted.
    Comment(String),
}

/// A decompiled function.
#[derive(Clone, Debug)]
pub struct Decompiled {
    /// Entry address.
    pub entry: Addr,
    /// Name used in the signature.
    pub name: String,
    /// Parameter names, in ABI order.
    pub params: Vec<String>,
    /// Declared type of each parameter, parallel to [`params`](Self::params).
    ///
    /// Empty when the image carries no debug information. An entry is `None`
    /// when the debug info names the parameter but not its type.
    pub param_types: Vec<Option<String>>,
    /// Register names to the parameter names they were given.
    ///
    /// The renderer renames a value that arrives in a register to the
    /// parameter name, so the body reads `a0` where the signature says `a0`.
    /// The mapping is per-architecture: `rdi` on x86-64, `x0` on AArch64,
    /// `a0` on RISC-V.
    pub param_aliases: Vec<(String, String)>,
    /// The type of the value the function returns, when the body says so.
    pub ret_type: Option<String>,
    /// Structured body.
    pub body: Vec<Node>,
    /// Blocks that could not be structured and are reached by `goto`.
    pub unstructured: usize,
    /// Human-readable notes about approximations.
    pub notes: Vec<String>,
    /// Source range the function covers, when debug information says.
    pub source: Option<crate::debug::SourceRef>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A float operation on two constants.
    fn float_op(op: BinOp, a: f64, b: f64) -> Expr {
        Expr::Bin {
            op,
            lhs: Box::new(Expr::Const(a.to_bits())),
            rhs: Box::new(Expr::Const(b.to_bits())),
            bits: 8,
        }
        .simplify()
    }

    #[test]
    fn floating_point_arithmetic_folds_on_the_bit_patterns() {
        let sum = float_op(BinOp::FAdd, 1.5, 2.25);
        assert_eq!(sum, Expr::Const(3.75f64.to_bits()));
        let product = float_op(BinOp::FMul, 2.0, 4.0);
        assert_eq!(product, Expr::Const(8.0f64.to_bits()));
    }

    #[test]
    fn a_not_a_number_makes_the_ordered_comparisons_false() {
        let nan = Expr::Const(f64::NAN.to_bits());
        let one = Expr::Const(1.0f64.to_bits());
        let eq = Expr::Bin {
            op: BinOp::FEq,
            lhs: Box::new(nan.clone()),
            rhs: Box::new(one.clone()),
            bits: 8,
        }
        .simplify();
        assert_eq!(eq, Expr::Const(0));
        let unordered = Expr::Bin {
            op: BinOp::FUnordered,
            lhs: Box::new(nan),
            rhs: Box::new(one),
            bits: 8,
        }
        .simplify();
        assert_eq!(unordered, Expr::Const(1));
    }

    #[test]
    fn a_floating_point_operation_is_printed_as_a_call() {
        let e = Expr::Bin {
            op: BinOp::FAdd,
            lhs: Box::new(Expr::var("x")),
            rhs: Box::new(Expr::var("y")),
            bits: 8,
        };
        assert_eq!(e.to_string(), "f64_add(x, y)");
        let e = Expr::Bin {
            op: BinOp::FMul,
            lhs: Box::new(Expr::var("x")),
            rhs: Box::new(Expr::var("y")),
            bits: 4,
        };
        assert_eq!(e.to_string(), "f32_mul(x, y)");
    }
    /// The parallel operations, checked against what the machine does.
    ///
    /// The values here were read off a table that the instructions
    /// themselves produced: every case was run on an emulator and the
    /// result written down.  The table covers the lanes of a byte and of
    /// a halfword, the saturated and the halved forms, and the ones that
    /// exchange the halves of their operands.
    #[test]
    fn the_parallel_operations_do_what_the_machine_does() {
        // (operation, lane width in bytes, a, b, expected)
        let cases: &[(Intrinsic, u8, u64, u64, u64)] = &[
            // The signed additions over the bytes.
            (Intrinsic::ParallelAdd, 1, 0x8080_8080, 0x0101_0101, 0x8181_8181),
            (Intrinsic::ParallelAdd, 1, 0x7f7f_7f7f, 0x0101_0101, 0x8080_8080),
            (Intrinsic::ParallelAdd, 1, 0x0102_0304, 0x8080_8080, 0x8182_8384),
            // The saturating ones hold at the ends of the lane's range:
            // a lane of 0x80 is -128, and subtracting one from it stays
            // at -128 rather than wrapping around to 127.
            (Intrinsic::ParallelAddSat, 1, 0x8080_8080, 0x0101_0101, 0x8181_8181),
            (Intrinsic::ParallelSubSat, 1, 0x8080_8080, 0x0101_0101, 0x8080_8080),
            (Intrinsic::ParallelSubSat, 1, 0x7f7f_7f7f, 0x0101_0101, 0x7e7e_7e7e),
            // The unsigned ones read the lane as a number and not as a
            // pattern.
            (Intrinsic::ParallelAddUns, 1, 0x8080_8080, 0x0101_0101, 0x8181_8181),
            (Intrinsic::ParallelSubUns, 1, 0x8080_8080, 0x0101_0101, 0x7f7f_7f7f),
            // The halved ones keep the bit that was carried out of the
            // lane, so the sum is taken a bit wider than the lane and the
            // shift is an arithmetic one.
            (Intrinsic::ParallelAddHalf, 1, 0x8080_8080, 0x0101_0101, 0xc0c0_c0c0),
            (Intrinsic::ParallelAddHalf, 1, 0x7f7f_7f7f, 0x0101_0101, 0x4040_4040),
            (Intrinsic::ParallelSubHalf, 1, 0x8080_8080, 0x0101_0101, 0xbfbf_bfbf),
            (Intrinsic::ParallelAddHalfUns, 1, 0x8080_8080, 0x0101_0101, 0x4040_4040),
            // The same over the halfwords.
            (Intrinsic::ParallelAdd, 2, 0xfedc_ba98, 0x1234_5678, 0x1110_1110),
            (Intrinsic::ParallelSub, 2, 0xfedc_ba98, 0x1234_5678, 0xeca8_6420),
            (Intrinsic::ParallelAddSat, 2, 0xfedc_ba98, 0x1234_5678, 0x1110_1110),
            (Intrinsic::ParallelSubSat, 2, 0xfedc_ba98, 0x1234_5678, 0xeca8_8000),
        ];
        for (op, lane, a, b, want) in cases {
            let got = eval_special(*op, &[*a, *b], *lane);
            assert_eq!(
                got,
                Some(*want),
                "{:?} over {} bits of {:#x} and {:#x}",
                op,
                lane * 8,
                a,
                b
            );
        }
    }

    /// The exchanged operations, checked against what the machine does.
    ///
    /// The high half of the result is the operation on the high half of
    /// the first operand and the low half of the second, and the low half
    /// of the result is the operation the other way round.
    #[test]
    fn the_exchanged_operations_do_what_the_machine_does() {
        let cases: &[(Intrinsic, u8, u64, u64, u64)] = &[
            (Intrinsic::ExchangeAdd, 2, 0xfedc_ba98, 0x1234_5678, 0x5554_a864),
            (Intrinsic::ExchangeSub, 2, 0xfedc_ba98, 0x1234_5678, 0xa864_cccc),
            (Intrinsic::ExchangeAddSat, 2, 0xfedc_ba98, 0x1234_5678, 0x5554_a864),
            (Intrinsic::ExchangeSubSat, 2, 0xfedc_ba98, 0x1234_5678, 0xa864_cccc),
            (Intrinsic::ExchangeAddHalf, 2, 0xfedc_ba98, 0x1234_5678, 0x2aaa_d432),
            (Intrinsic::ExchangeAddUns, 2, 0xfedc_ba98, 0x1234_5678, 0x5554_a864),
            (Intrinsic::ExchangeSubUns, 2, 0xfedc_ba98, 0x1234_5678, 0xa864_cccc),
            (Intrinsic::ExchangeAddHalfUns, 2, 0xfedc_ba98, 0x1234_5678, 0xaaaa_5432),
            (Intrinsic::ExchangeSubHalfUns, 2, 0xfedc_ba98, 0x1234_5678, 0x5432_6666),
        ];
        for (op, lane, a, b, want) in cases {
            let got = eval_special(*op, &[*a, *b], *lane);
            assert_eq!(
                got,
                Some(*want),
                "{:?} over {} bits of {:#x} and {:#x}",
                op,
                lane * 8,
                a,
                b
            );
        }
    }
    /// Every parallel and exchanged operation, over every case that was
    /// run on the machine.
    ///
    /// The cases come from running the instructions under an emulator
    /// of the hardware and reading back what it produced, so they say
    /// what the machine does rather than what the manual says it does.
    /// There are nine hundred of them, which is enough for a mistake
    /// in one corner of the arithmetic to show up.
    #[test]
    fn every_parallel_and_exchanged_operation_matches_the_machine() {
        let cases: &[(Intrinsic, u8, u64, u64, u64)] = &[
            (Intrinsic::ParallelAdd, 1, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelAddSat, 1, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelAddHalf, 1, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelAddUns, 1, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelAddSatUns, 1, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelAddHalfUns, 1, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelAdd, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelAddSat, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelAddHalf, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelAddUns, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelAddSatUns, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelAddHalfUns, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelSub, 1, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelSubSat, 1, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelSubHalf, 1, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelSubUns, 1, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelSubSatUns, 1, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 1, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelSub, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelSubSat, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelSubHalf, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelSubUns, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelSubSatUns, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ExchangeAdd, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ExchangeSub, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ExchangeAddSat, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ExchangeSubSat, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ExchangeAddHalf, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ExchangeSubHalf, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ExchangeAddUns, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ExchangeSubUns, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ExchangeAddSatUns, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ExchangeSubSatUns, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ParallelAdd, 1, 0x00000001, 0x00000001, 0x00000002),
            (Intrinsic::ParallelAddSat, 1, 0x00000001, 0x00000001, 0x00000002),
            (Intrinsic::ParallelAddHalf, 1, 0x00000001, 0x00000001, 0x00000001),
            (Intrinsic::ParallelAddUns, 1, 0x00000001, 0x00000001, 0x00000002),
            (Intrinsic::ParallelAddSatUns, 1, 0x00000001, 0x00000001, 0x00000002),
            (Intrinsic::ParallelAddHalfUns, 1, 0x00000001, 0x00000001, 0x00000001),
            (Intrinsic::ParallelAdd, 2, 0x00000001, 0x00000001, 0x00000002),
            (Intrinsic::ParallelAddSat, 2, 0x00000001, 0x00000001, 0x00000002),
            (Intrinsic::ParallelAddHalf, 2, 0x00000001, 0x00000001, 0x00000001),
            (Intrinsic::ParallelAddUns, 2, 0x00000001, 0x00000001, 0x00000002),
            (Intrinsic::ParallelAddSatUns, 2, 0x00000001, 0x00000001, 0x00000002),
            (Intrinsic::ParallelAddHalfUns, 2, 0x00000001, 0x00000001, 0x00000001),
            (Intrinsic::ParallelSub, 1, 0x00000001, 0x00000001, 0x00000000),
            (Intrinsic::ParallelSubSat, 1, 0x00000001, 0x00000001, 0x00000000),
            (Intrinsic::ParallelSubHalf, 1, 0x00000001, 0x00000001, 0x00000000),
            (Intrinsic::ParallelSubUns, 1, 0x00000001, 0x00000001, 0x00000000),
            (Intrinsic::ParallelSubSatUns, 1, 0x00000001, 0x00000001, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 1, 0x00000001, 0x00000001, 0x00000000),
            (Intrinsic::ParallelSub, 2, 0x00000001, 0x00000001, 0x00000000),
            (Intrinsic::ParallelSubSat, 2, 0x00000001, 0x00000001, 0x00000000),
            (Intrinsic::ParallelSubHalf, 2, 0x00000001, 0x00000001, 0x00000000),
            (Intrinsic::ParallelSubUns, 2, 0x00000001, 0x00000001, 0x00000000),
            (Intrinsic::ParallelSubSatUns, 2, 0x00000001, 0x00000001, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 2, 0x00000001, 0x00000001, 0x00000000),
            (Intrinsic::ExchangeAdd, 2, 0x00000001, 0x00000001, 0x00010001),
            (Intrinsic::ExchangeSub, 2, 0x00000001, 0x00000001, 0xffff0001),
            (Intrinsic::ExchangeAddSat, 2, 0x00000001, 0x00000001, 0x00010001),
            (Intrinsic::ExchangeSubSat, 2, 0x00000001, 0x00000001, 0xffff0001),
            (Intrinsic::ExchangeAddHalf, 2, 0x00000001, 0x00000001, 0x00000000),
            (Intrinsic::ExchangeSubHalf, 2, 0x00000001, 0x00000001, 0xffff0000),
            (Intrinsic::ExchangeAddUns, 2, 0x00000001, 0x00000001, 0x00010001),
            (Intrinsic::ExchangeSubUns, 2, 0x00000001, 0x00000001, 0xffff0001),
            (Intrinsic::ExchangeAddSatUns, 2, 0x00000001, 0x00000001, 0x00010001),
            (Intrinsic::ExchangeSubSatUns, 2, 0x00000001, 0x00000001, 0x00000001),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x00000001, 0x00000001, 0x00000000),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x00000001, 0x00000001, 0xffff0000),
            (Intrinsic::ParallelAdd, 1, 0x00000002, 0x00000003, 0x00000005),
            (Intrinsic::ParallelAddSat, 1, 0x00000002, 0x00000003, 0x00000005),
            (Intrinsic::ParallelAddHalf, 1, 0x00000002, 0x00000003, 0x00000002),
            (Intrinsic::ParallelAddUns, 1, 0x00000002, 0x00000003, 0x00000005),
            (Intrinsic::ParallelAddSatUns, 1, 0x00000002, 0x00000003, 0x00000005),
            (Intrinsic::ParallelAddHalfUns, 1, 0x00000002, 0x00000003, 0x00000002),
            (Intrinsic::ParallelAdd, 2, 0x00000002, 0x00000003, 0x00000005),
            (Intrinsic::ParallelAddSat, 2, 0x00000002, 0x00000003, 0x00000005),
            (Intrinsic::ParallelAddHalf, 2, 0x00000002, 0x00000003, 0x00000002),
            (Intrinsic::ParallelAddUns, 2, 0x00000002, 0x00000003, 0x00000005),
            (Intrinsic::ParallelAddSatUns, 2, 0x00000002, 0x00000003, 0x00000005),
            (Intrinsic::ParallelAddHalfUns, 2, 0x00000002, 0x00000003, 0x00000002),
            (Intrinsic::ParallelSub, 1, 0x00000002, 0x00000003, 0x000000ff),
            (Intrinsic::ParallelSubSat, 1, 0x00000002, 0x00000003, 0x000000ff),
            (Intrinsic::ParallelSubHalf, 1, 0x00000002, 0x00000003, 0x000000ff),
            (Intrinsic::ParallelSubUns, 1, 0x00000002, 0x00000003, 0x000000ff),
            (Intrinsic::ParallelSubSatUns, 1, 0x00000002, 0x00000003, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 1, 0x00000002, 0x00000003, 0x000000ff),
            (Intrinsic::ParallelSub, 2, 0x00000002, 0x00000003, 0x0000ffff),
            (Intrinsic::ParallelSubSat, 2, 0x00000002, 0x00000003, 0x0000ffff),
            (Intrinsic::ParallelSubHalf, 2, 0x00000002, 0x00000003, 0x0000ffff),
            (Intrinsic::ParallelSubUns, 2, 0x00000002, 0x00000003, 0x0000ffff),
            (Intrinsic::ParallelSubSatUns, 2, 0x00000002, 0x00000003, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 2, 0x00000002, 0x00000003, 0x0000ffff),
            (Intrinsic::ExchangeAdd, 2, 0x00000002, 0x00000003, 0x00030002),
            (Intrinsic::ExchangeSub, 2, 0x00000002, 0x00000003, 0xfffd0002),
            (Intrinsic::ExchangeAddSat, 2, 0x00000002, 0x00000003, 0x00030002),
            (Intrinsic::ExchangeSubSat, 2, 0x00000002, 0x00000003, 0xfffd0002),
            (Intrinsic::ExchangeAddHalf, 2, 0x00000002, 0x00000003, 0x00010001),
            (Intrinsic::ExchangeSubHalf, 2, 0x00000002, 0x00000003, 0xfffe0001),
            (Intrinsic::ExchangeAddUns, 2, 0x00000002, 0x00000003, 0x00030002),
            (Intrinsic::ExchangeSubUns, 2, 0x00000002, 0x00000003, 0xfffd0002),
            (Intrinsic::ExchangeAddSatUns, 2, 0x00000002, 0x00000003, 0x00030002),
            (Intrinsic::ExchangeSubSatUns, 2, 0x00000002, 0x00000003, 0x00000002),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x00000002, 0x00000003, 0x00010001),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x00000002, 0x00000003, 0xfffe0001),
            (Intrinsic::ParallelAdd, 1, 0xffffffff, 0x00000001, 0xffffff00),
            (Intrinsic::ParallelAddSat, 1, 0xffffffff, 0x00000001, 0xffffff00),
            (Intrinsic::ParallelAddHalf, 1, 0xffffffff, 0x00000001, 0xffffff00),
            (Intrinsic::ParallelAddUns, 1, 0xffffffff, 0x00000001, 0xffffff00),
            (Intrinsic::ParallelAddSatUns, 1, 0xffffffff, 0x00000001, 0xffffffff),
            (Intrinsic::ParallelAddHalfUns, 1, 0xffffffff, 0x00000001, 0x7f7f7f80),
            (Intrinsic::ParallelAdd, 2, 0xffffffff, 0x00000001, 0xffff0000),
            (Intrinsic::ParallelAddSat, 2, 0xffffffff, 0x00000001, 0xffff0000),
            (Intrinsic::ParallelAddHalf, 2, 0xffffffff, 0x00000001, 0xffff0000),
            (Intrinsic::ParallelAddUns, 2, 0xffffffff, 0x00000001, 0xffff0000),
            (Intrinsic::ParallelAddSatUns, 2, 0xffffffff, 0x00000001, 0xffffffff),
            (Intrinsic::ParallelAddHalfUns, 2, 0xffffffff, 0x00000001, 0x7fff8000),
            (Intrinsic::ParallelSub, 1, 0xffffffff, 0x00000001, 0xfffffffe),
            (Intrinsic::ParallelSubSat, 1, 0xffffffff, 0x00000001, 0xfffffffe),
            (Intrinsic::ParallelSubHalf, 1, 0xffffffff, 0x00000001, 0xffffffff),
            (Intrinsic::ParallelSubUns, 1, 0xffffffff, 0x00000001, 0xfffffffe),
            (Intrinsic::ParallelSubSatUns, 1, 0xffffffff, 0x00000001, 0xfffffffe),
            (Intrinsic::ParallelSubHalfUns, 1, 0xffffffff, 0x00000001, 0x7f7f7f7f),
            (Intrinsic::ParallelSub, 2, 0xffffffff, 0x00000001, 0xfffffffe),
            (Intrinsic::ParallelSubSat, 2, 0xffffffff, 0x00000001, 0xfffffffe),
            (Intrinsic::ParallelSubHalf, 2, 0xffffffff, 0x00000001, 0xffffffff),
            (Intrinsic::ParallelSubUns, 2, 0xffffffff, 0x00000001, 0xfffffffe),
            (Intrinsic::ParallelSubSatUns, 2, 0xffffffff, 0x00000001, 0xfffffffe),
            (Intrinsic::ParallelSubHalfUns, 2, 0xffffffff, 0x00000001, 0x7fff7fff),
            (Intrinsic::ExchangeAdd, 2, 0xffffffff, 0x00000001, 0x0000ffff),
            (Intrinsic::ExchangeSub, 2, 0xffffffff, 0x00000001, 0xfffeffff),
            (Intrinsic::ExchangeAddSat, 2, 0xffffffff, 0x00000001, 0x0000ffff),
            (Intrinsic::ExchangeSubSat, 2, 0xffffffff, 0x00000001, 0xfffeffff),
            (Intrinsic::ExchangeAddHalf, 2, 0xffffffff, 0x00000001, 0x0000ffff),
            (Intrinsic::ExchangeSubHalf, 2, 0xffffffff, 0x00000001, 0xffffffff),
            (Intrinsic::ExchangeAddUns, 2, 0xffffffff, 0x00000001, 0x0000ffff),
            (Intrinsic::ExchangeSubUns, 2, 0xffffffff, 0x00000001, 0xfffeffff),
            (Intrinsic::ExchangeAddSatUns, 2, 0xffffffff, 0x00000001, 0xffffffff),
            (Intrinsic::ExchangeSubSatUns, 2, 0xffffffff, 0x00000001, 0xfffeffff),
            (Intrinsic::ExchangeAddHalfUns, 2, 0xffffffff, 0x00000001, 0x80007fff),
            (Intrinsic::ExchangeSubHalfUns, 2, 0xffffffff, 0x00000001, 0x7fff7fff),
            (Intrinsic::ParallelAdd, 1, 0x00000000, 0xffffffff, 0xffffffff),
            (Intrinsic::ParallelAddSat, 1, 0x00000000, 0xffffffff, 0xffffffff),
            (Intrinsic::ParallelAddHalf, 1, 0x00000000, 0xffffffff, 0xffffffff),
            (Intrinsic::ParallelAddUns, 1, 0x00000000, 0xffffffff, 0xffffffff),
            (Intrinsic::ParallelAddSatUns, 1, 0x00000000, 0xffffffff, 0xffffffff),
            (Intrinsic::ParallelAddHalfUns, 1, 0x00000000, 0xffffffff, 0x7f7f7f7f),
            (Intrinsic::ParallelAdd, 2, 0x00000000, 0xffffffff, 0xffffffff),
            (Intrinsic::ParallelAddSat, 2, 0x00000000, 0xffffffff, 0xffffffff),
            (Intrinsic::ParallelAddHalf, 2, 0x00000000, 0xffffffff, 0xffffffff),
            (Intrinsic::ParallelAddUns, 2, 0x00000000, 0xffffffff, 0xffffffff),
            (Intrinsic::ParallelAddSatUns, 2, 0x00000000, 0xffffffff, 0xffffffff),
            (Intrinsic::ParallelAddHalfUns, 2, 0x00000000, 0xffffffff, 0x7fff7fff),
            (Intrinsic::ParallelSub, 1, 0x00000000, 0xffffffff, 0x01010101),
            (Intrinsic::ParallelSubSat, 1, 0x00000000, 0xffffffff, 0x01010101),
            (Intrinsic::ParallelSubHalf, 1, 0x00000000, 0xffffffff, 0x00000000),
            (Intrinsic::ParallelSubUns, 1, 0x00000000, 0xffffffff, 0x01010101),
            (Intrinsic::ParallelSubSatUns, 1, 0x00000000, 0xffffffff, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 1, 0x00000000, 0xffffffff, 0x80808080),
            (Intrinsic::ParallelSub, 2, 0x00000000, 0xffffffff, 0x00010001),
            (Intrinsic::ParallelSubSat, 2, 0x00000000, 0xffffffff, 0x00010001),
            (Intrinsic::ParallelSubHalf, 2, 0x00000000, 0xffffffff, 0x00000000),
            (Intrinsic::ParallelSubUns, 2, 0x00000000, 0xffffffff, 0x00010001),
            (Intrinsic::ParallelSubSatUns, 2, 0x00000000, 0xffffffff, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 2, 0x00000000, 0xffffffff, 0x80008000),
            (Intrinsic::ExchangeAdd, 2, 0x00000000, 0xffffffff, 0xffff0001),
            (Intrinsic::ExchangeSub, 2, 0x00000000, 0xffffffff, 0x0001ffff),
            (Intrinsic::ExchangeAddSat, 2, 0x00000000, 0xffffffff, 0xffff0001),
            (Intrinsic::ExchangeSubSat, 2, 0x00000000, 0xffffffff, 0x0001ffff),
            (Intrinsic::ExchangeAddHalf, 2, 0x00000000, 0xffffffff, 0xffff0000),
            (Intrinsic::ExchangeSubHalf, 2, 0x00000000, 0xffffffff, 0x0000ffff),
            (Intrinsic::ExchangeAddUns, 2, 0x00000000, 0xffffffff, 0xffff0001),
            (Intrinsic::ExchangeSubUns, 2, 0x00000000, 0xffffffff, 0x0001ffff),
            (Intrinsic::ExchangeAddSatUns, 2, 0x00000000, 0xffffffff, 0xffff0000),
            (Intrinsic::ExchangeSubSatUns, 2, 0x00000000, 0xffffffff, 0x0000ffff),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x00000000, 0xffffffff, 0x7fff8000),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x00000000, 0xffffffff, 0x80007fff),
            (Intrinsic::ParallelAdd, 1, 0xffffffff, 0xffffffff, 0xfefefefe),
            (Intrinsic::ParallelAddSat, 1, 0xffffffff, 0xffffffff, 0xfefefefe),
            (Intrinsic::ParallelAddHalf, 1, 0xffffffff, 0xffffffff, 0xffffffff),
            (Intrinsic::ParallelAddUns, 1, 0xffffffff, 0xffffffff, 0xfefefefe),
            (Intrinsic::ParallelAddSatUns, 1, 0xffffffff, 0xffffffff, 0xffffffff),
            (Intrinsic::ParallelAddHalfUns, 1, 0xffffffff, 0xffffffff, 0xffffffff),
            (Intrinsic::ParallelAdd, 2, 0xffffffff, 0xffffffff, 0xfffefffe),
            (Intrinsic::ParallelAddSat, 2, 0xffffffff, 0xffffffff, 0xfffefffe),
            (Intrinsic::ParallelAddHalf, 2, 0xffffffff, 0xffffffff, 0xffffffff),
            (Intrinsic::ParallelAddUns, 2, 0xffffffff, 0xffffffff, 0xfffefffe),
            (Intrinsic::ParallelAddSatUns, 2, 0xffffffff, 0xffffffff, 0xffffffff),
            (Intrinsic::ParallelAddHalfUns, 2, 0xffffffff, 0xffffffff, 0xffffffff),
            (Intrinsic::ParallelSub, 1, 0xffffffff, 0xffffffff, 0x00000000),
            (Intrinsic::ParallelSubSat, 1, 0xffffffff, 0xffffffff, 0x00000000),
            (Intrinsic::ParallelSubHalf, 1, 0xffffffff, 0xffffffff, 0x00000000),
            (Intrinsic::ParallelSubUns, 1, 0xffffffff, 0xffffffff, 0x00000000),
            (Intrinsic::ParallelSubSatUns, 1, 0xffffffff, 0xffffffff, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 1, 0xffffffff, 0xffffffff, 0x00000000),
            (Intrinsic::ParallelSub, 2, 0xffffffff, 0xffffffff, 0x00000000),
            (Intrinsic::ParallelSubSat, 2, 0xffffffff, 0xffffffff, 0x00000000),
            (Intrinsic::ParallelSubHalf, 2, 0xffffffff, 0xffffffff, 0x00000000),
            (Intrinsic::ParallelSubUns, 2, 0xffffffff, 0xffffffff, 0x00000000),
            (Intrinsic::ParallelSubSatUns, 2, 0xffffffff, 0xffffffff, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 2, 0xffffffff, 0xffffffff, 0x00000000),
            (Intrinsic::ExchangeAdd, 2, 0xffffffff, 0xffffffff, 0xfffe0000),
            (Intrinsic::ExchangeSub, 2, 0xffffffff, 0xffffffff, 0x0000fffe),
            (Intrinsic::ExchangeAddSat, 2, 0xffffffff, 0xffffffff, 0xfffe0000),
            (Intrinsic::ExchangeSubSat, 2, 0xffffffff, 0xffffffff, 0x0000fffe),
            (Intrinsic::ExchangeAddHalf, 2, 0xffffffff, 0xffffffff, 0xffff0000),
            (Intrinsic::ExchangeSubHalf, 2, 0xffffffff, 0xffffffff, 0x0000ffff),
            (Intrinsic::ExchangeAddUns, 2, 0xffffffff, 0xffffffff, 0xfffe0000),
            (Intrinsic::ExchangeSubUns, 2, 0xffffffff, 0xffffffff, 0x0000fffe),
            (Intrinsic::ExchangeAddSatUns, 2, 0xffffffff, 0xffffffff, 0xffff0000),
            (Intrinsic::ExchangeSubSatUns, 2, 0xffffffff, 0xffffffff, 0x0000ffff),
            (Intrinsic::ExchangeAddHalfUns, 2, 0xffffffff, 0xffffffff, 0xffff0000),
            (Intrinsic::ExchangeSubHalfUns, 2, 0xffffffff, 0xffffffff, 0x0000ffff),
            (Intrinsic::ParallelAdd, 1, 0x0000007f, 0x00000001, 0x00000080),
            (Intrinsic::ParallelAddSat, 1, 0x0000007f, 0x00000001, 0x0000007f),
            (Intrinsic::ParallelAddHalf, 1, 0x0000007f, 0x00000001, 0x00000040),
            (Intrinsic::ParallelAddUns, 1, 0x0000007f, 0x00000001, 0x00000080),
            (Intrinsic::ParallelAddSatUns, 1, 0x0000007f, 0x00000001, 0x00000080),
            (Intrinsic::ParallelAddHalfUns, 1, 0x0000007f, 0x00000001, 0x00000040),
            (Intrinsic::ParallelAdd, 2, 0x0000007f, 0x00000001, 0x00000080),
            (Intrinsic::ParallelAddSat, 2, 0x0000007f, 0x00000001, 0x00000080),
            (Intrinsic::ParallelAddHalf, 2, 0x0000007f, 0x00000001, 0x00000040),
            (Intrinsic::ParallelAddUns, 2, 0x0000007f, 0x00000001, 0x00000080),
            (Intrinsic::ParallelAddSatUns, 2, 0x0000007f, 0x00000001, 0x00000080),
            (Intrinsic::ParallelAddHalfUns, 2, 0x0000007f, 0x00000001, 0x00000040),
            (Intrinsic::ParallelSub, 1, 0x0000007f, 0x00000001, 0x0000007e),
            (Intrinsic::ParallelSubSat, 1, 0x0000007f, 0x00000001, 0x0000007e),
            (Intrinsic::ParallelSubHalf, 1, 0x0000007f, 0x00000001, 0x0000003f),
            (Intrinsic::ParallelSubUns, 1, 0x0000007f, 0x00000001, 0x0000007e),
            (Intrinsic::ParallelSubSatUns, 1, 0x0000007f, 0x00000001, 0x0000007e),
            (Intrinsic::ParallelSubHalfUns, 1, 0x0000007f, 0x00000001, 0x0000003f),
            (Intrinsic::ParallelSub, 2, 0x0000007f, 0x00000001, 0x0000007e),
            (Intrinsic::ParallelSubSat, 2, 0x0000007f, 0x00000001, 0x0000007e),
            (Intrinsic::ParallelSubHalf, 2, 0x0000007f, 0x00000001, 0x0000003f),
            (Intrinsic::ParallelSubUns, 2, 0x0000007f, 0x00000001, 0x0000007e),
            (Intrinsic::ParallelSubSatUns, 2, 0x0000007f, 0x00000001, 0x0000007e),
            (Intrinsic::ParallelSubHalfUns, 2, 0x0000007f, 0x00000001, 0x0000003f),
            (Intrinsic::ExchangeAdd, 2, 0x0000007f, 0x00000001, 0x0001007f),
            (Intrinsic::ExchangeSub, 2, 0x0000007f, 0x00000001, 0xffff007f),
            (Intrinsic::ExchangeAddSat, 2, 0x0000007f, 0x00000001, 0x0001007f),
            (Intrinsic::ExchangeSubSat, 2, 0x0000007f, 0x00000001, 0xffff007f),
            (Intrinsic::ExchangeAddHalf, 2, 0x0000007f, 0x00000001, 0x0000003f),
            (Intrinsic::ExchangeSubHalf, 2, 0x0000007f, 0x00000001, 0xffff003f),
            (Intrinsic::ExchangeAddUns, 2, 0x0000007f, 0x00000001, 0x0001007f),
            (Intrinsic::ExchangeSubUns, 2, 0x0000007f, 0x00000001, 0xffff007f),
            (Intrinsic::ExchangeAddSatUns, 2, 0x0000007f, 0x00000001, 0x0001007f),
            (Intrinsic::ExchangeSubSatUns, 2, 0x0000007f, 0x00000001, 0x0000007f),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x0000007f, 0x00000001, 0x0000003f),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x0000007f, 0x00000001, 0xffff003f),
            (Intrinsic::ParallelAdd, 1, 0x00000080, 0x00000001, 0x00000081),
            (Intrinsic::ParallelAddSat, 1, 0x00000080, 0x00000001, 0x00000081),
            (Intrinsic::ParallelAddHalf, 1, 0x00000080, 0x00000001, 0x000000c0),
            (Intrinsic::ParallelAddUns, 1, 0x00000080, 0x00000001, 0x00000081),
            (Intrinsic::ParallelAddSatUns, 1, 0x00000080, 0x00000001, 0x00000081),
            (Intrinsic::ParallelAddHalfUns, 1, 0x00000080, 0x00000001, 0x00000040),
            (Intrinsic::ParallelAdd, 2, 0x00000080, 0x00000001, 0x00000081),
            (Intrinsic::ParallelAddSat, 2, 0x00000080, 0x00000001, 0x00000081),
            (Intrinsic::ParallelAddHalf, 2, 0x00000080, 0x00000001, 0x00000040),
            (Intrinsic::ParallelAddUns, 2, 0x00000080, 0x00000001, 0x00000081),
            (Intrinsic::ParallelAddSatUns, 2, 0x00000080, 0x00000001, 0x00000081),
            (Intrinsic::ParallelAddHalfUns, 2, 0x00000080, 0x00000001, 0x00000040),
            (Intrinsic::ParallelSub, 1, 0x00000080, 0x00000001, 0x0000007f),
            (Intrinsic::ParallelSubSat, 1, 0x00000080, 0x00000001, 0x00000080),
            (Intrinsic::ParallelSubHalf, 1, 0x00000080, 0x00000001, 0x000000bf),
            (Intrinsic::ParallelSubUns, 1, 0x00000080, 0x00000001, 0x0000007f),
            (Intrinsic::ParallelSubSatUns, 1, 0x00000080, 0x00000001, 0x0000007f),
            (Intrinsic::ParallelSubHalfUns, 1, 0x00000080, 0x00000001, 0x0000003f),
            (Intrinsic::ParallelSub, 2, 0x00000080, 0x00000001, 0x0000007f),
            (Intrinsic::ParallelSubSat, 2, 0x00000080, 0x00000001, 0x0000007f),
            (Intrinsic::ParallelSubHalf, 2, 0x00000080, 0x00000001, 0x0000003f),
            (Intrinsic::ParallelSubUns, 2, 0x00000080, 0x00000001, 0x0000007f),
            (Intrinsic::ParallelSubSatUns, 2, 0x00000080, 0x00000001, 0x0000007f),
            (Intrinsic::ParallelSubHalfUns, 2, 0x00000080, 0x00000001, 0x0000003f),
            (Intrinsic::ExchangeAdd, 2, 0x00000080, 0x00000001, 0x00010080),
            (Intrinsic::ExchangeSub, 2, 0x00000080, 0x00000001, 0xffff0080),
            (Intrinsic::ExchangeAddSat, 2, 0x00000080, 0x00000001, 0x00010080),
            (Intrinsic::ExchangeSubSat, 2, 0x00000080, 0x00000001, 0xffff0080),
            (Intrinsic::ExchangeAddHalf, 2, 0x00000080, 0x00000001, 0x00000040),
            (Intrinsic::ExchangeSubHalf, 2, 0x00000080, 0x00000001, 0xffff0040),
            (Intrinsic::ExchangeAddUns, 2, 0x00000080, 0x00000001, 0x00010080),
            (Intrinsic::ExchangeSubUns, 2, 0x00000080, 0x00000001, 0xffff0080),
            (Intrinsic::ExchangeAddSatUns, 2, 0x00000080, 0x00000001, 0x00010080),
            (Intrinsic::ExchangeSubSatUns, 2, 0x00000080, 0x00000001, 0x00000080),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x00000080, 0x00000001, 0x00000040),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x00000080, 0x00000001, 0xffff0040),
            (Intrinsic::ParallelAdd, 1, 0x000000ff, 0x00000001, 0x00000000),
            (Intrinsic::ParallelAddSat, 1, 0x000000ff, 0x00000001, 0x00000000),
            (Intrinsic::ParallelAddHalf, 1, 0x000000ff, 0x00000001, 0x00000000),
            (Intrinsic::ParallelAddUns, 1, 0x000000ff, 0x00000001, 0x00000000),
            (Intrinsic::ParallelAddSatUns, 1, 0x000000ff, 0x00000001, 0x000000ff),
            (Intrinsic::ParallelAddHalfUns, 1, 0x000000ff, 0x00000001, 0x00000080),
            (Intrinsic::ParallelAdd, 2, 0x000000ff, 0x00000001, 0x00000100),
            (Intrinsic::ParallelAddSat, 2, 0x000000ff, 0x00000001, 0x00000100),
            (Intrinsic::ParallelAddHalf, 2, 0x000000ff, 0x00000001, 0x00000080),
            (Intrinsic::ParallelAddUns, 2, 0x000000ff, 0x00000001, 0x00000100),
            (Intrinsic::ParallelAddSatUns, 2, 0x000000ff, 0x00000001, 0x00000100),
            (Intrinsic::ParallelAddHalfUns, 2, 0x000000ff, 0x00000001, 0x00000080),
            (Intrinsic::ParallelSub, 1, 0x000000ff, 0x00000001, 0x000000fe),
            (Intrinsic::ParallelSubSat, 1, 0x000000ff, 0x00000001, 0x000000fe),
            (Intrinsic::ParallelSubHalf, 1, 0x000000ff, 0x00000001, 0x000000ff),
            (Intrinsic::ParallelSubUns, 1, 0x000000ff, 0x00000001, 0x000000fe),
            (Intrinsic::ParallelSubSatUns, 1, 0x000000ff, 0x00000001, 0x000000fe),
            (Intrinsic::ParallelSubHalfUns, 1, 0x000000ff, 0x00000001, 0x0000007f),
            (Intrinsic::ParallelSub, 2, 0x000000ff, 0x00000001, 0x000000fe),
            (Intrinsic::ParallelSubSat, 2, 0x000000ff, 0x00000001, 0x000000fe),
            (Intrinsic::ParallelSubHalf, 2, 0x000000ff, 0x00000001, 0x0000007f),
            (Intrinsic::ParallelSubUns, 2, 0x000000ff, 0x00000001, 0x000000fe),
            (Intrinsic::ParallelSubSatUns, 2, 0x000000ff, 0x00000001, 0x000000fe),
            (Intrinsic::ParallelSubHalfUns, 2, 0x000000ff, 0x00000001, 0x0000007f),
            (Intrinsic::ExchangeAdd, 2, 0x000000ff, 0x00000001, 0x000100ff),
            (Intrinsic::ExchangeSub, 2, 0x000000ff, 0x00000001, 0xffff00ff),
            (Intrinsic::ExchangeAddSat, 2, 0x000000ff, 0x00000001, 0x000100ff),
            (Intrinsic::ExchangeSubSat, 2, 0x000000ff, 0x00000001, 0xffff00ff),
            (Intrinsic::ExchangeAddHalf, 2, 0x000000ff, 0x00000001, 0x0000007f),
            (Intrinsic::ExchangeSubHalf, 2, 0x000000ff, 0x00000001, 0xffff007f),
            (Intrinsic::ExchangeAddUns, 2, 0x000000ff, 0x00000001, 0x000100ff),
            (Intrinsic::ExchangeSubUns, 2, 0x000000ff, 0x00000001, 0xffff00ff),
            (Intrinsic::ExchangeAddSatUns, 2, 0x000000ff, 0x00000001, 0x000100ff),
            (Intrinsic::ExchangeSubSatUns, 2, 0x000000ff, 0x00000001, 0x000000ff),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x000000ff, 0x00000001, 0x0000007f),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x000000ff, 0x00000001, 0xffff007f),
            (Intrinsic::ParallelAdd, 1, 0x00000001, 0x000000ff, 0x00000000),
            (Intrinsic::ParallelAddSat, 1, 0x00000001, 0x000000ff, 0x00000000),
            (Intrinsic::ParallelAddHalf, 1, 0x00000001, 0x000000ff, 0x00000000),
            (Intrinsic::ParallelAddUns, 1, 0x00000001, 0x000000ff, 0x00000000),
            (Intrinsic::ParallelAddSatUns, 1, 0x00000001, 0x000000ff, 0x000000ff),
            (Intrinsic::ParallelAddHalfUns, 1, 0x00000001, 0x000000ff, 0x00000080),
            (Intrinsic::ParallelAdd, 2, 0x00000001, 0x000000ff, 0x00000100),
            (Intrinsic::ParallelAddSat, 2, 0x00000001, 0x000000ff, 0x00000100),
            (Intrinsic::ParallelAddHalf, 2, 0x00000001, 0x000000ff, 0x00000080),
            (Intrinsic::ParallelAddUns, 2, 0x00000001, 0x000000ff, 0x00000100),
            (Intrinsic::ParallelAddSatUns, 2, 0x00000001, 0x000000ff, 0x00000100),
            (Intrinsic::ParallelAddHalfUns, 2, 0x00000001, 0x000000ff, 0x00000080),
            (Intrinsic::ParallelSub, 1, 0x00000001, 0x000000ff, 0x00000002),
            (Intrinsic::ParallelSubSat, 1, 0x00000001, 0x000000ff, 0x00000002),
            (Intrinsic::ParallelSubHalf, 1, 0x00000001, 0x000000ff, 0x00000001),
            (Intrinsic::ParallelSubUns, 1, 0x00000001, 0x000000ff, 0x00000002),
            (Intrinsic::ParallelSubSatUns, 1, 0x00000001, 0x000000ff, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 1, 0x00000001, 0x000000ff, 0x00000081),
            (Intrinsic::ParallelSub, 2, 0x00000001, 0x000000ff, 0x0000ff02),
            (Intrinsic::ParallelSubSat, 2, 0x00000001, 0x000000ff, 0x0000ff02),
            (Intrinsic::ParallelSubHalf, 2, 0x00000001, 0x000000ff, 0x0000ff81),
            (Intrinsic::ParallelSubUns, 2, 0x00000001, 0x000000ff, 0x0000ff02),
            (Intrinsic::ParallelSubSatUns, 2, 0x00000001, 0x000000ff, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 2, 0x00000001, 0x000000ff, 0x0000ff81),
            (Intrinsic::ExchangeAdd, 2, 0x00000001, 0x000000ff, 0x00ff0001),
            (Intrinsic::ExchangeSub, 2, 0x00000001, 0x000000ff, 0xff010001),
            (Intrinsic::ExchangeAddSat, 2, 0x00000001, 0x000000ff, 0x00ff0001),
            (Intrinsic::ExchangeSubSat, 2, 0x00000001, 0x000000ff, 0xff010001),
            (Intrinsic::ExchangeAddHalf, 2, 0x00000001, 0x000000ff, 0x007f0000),
            (Intrinsic::ExchangeSubHalf, 2, 0x00000001, 0x000000ff, 0xff800000),
            (Intrinsic::ExchangeAddUns, 2, 0x00000001, 0x000000ff, 0x00ff0001),
            (Intrinsic::ExchangeSubUns, 2, 0x00000001, 0x000000ff, 0xff010001),
            (Intrinsic::ExchangeAddSatUns, 2, 0x00000001, 0x000000ff, 0x00ff0001),
            (Intrinsic::ExchangeSubSatUns, 2, 0x00000001, 0x000000ff, 0x00000001),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x00000001, 0x000000ff, 0x007f0000),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x00000001, 0x000000ff, 0xff800000),
            (Intrinsic::ParallelAdd, 1, 0x0000ffff, 0x00000001, 0x0000ff00),
            (Intrinsic::ParallelAddSat, 1, 0x0000ffff, 0x00000001, 0x0000ff00),
            (Intrinsic::ParallelAddHalf, 1, 0x0000ffff, 0x00000001, 0x0000ff00),
            (Intrinsic::ParallelAddUns, 1, 0x0000ffff, 0x00000001, 0x0000ff00),
            (Intrinsic::ParallelAddSatUns, 1, 0x0000ffff, 0x00000001, 0x0000ffff),
            (Intrinsic::ParallelAddHalfUns, 1, 0x0000ffff, 0x00000001, 0x00007f80),
            (Intrinsic::ParallelAdd, 2, 0x0000ffff, 0x00000001, 0x00000000),
            (Intrinsic::ParallelAddSat, 2, 0x0000ffff, 0x00000001, 0x00000000),
            (Intrinsic::ParallelAddHalf, 2, 0x0000ffff, 0x00000001, 0x00000000),
            (Intrinsic::ParallelAddUns, 2, 0x0000ffff, 0x00000001, 0x00000000),
            (Intrinsic::ParallelAddSatUns, 2, 0x0000ffff, 0x00000001, 0x0000ffff),
            (Intrinsic::ParallelAddHalfUns, 2, 0x0000ffff, 0x00000001, 0x00008000),
            (Intrinsic::ParallelSub, 1, 0x0000ffff, 0x00000001, 0x0000fffe),
            (Intrinsic::ParallelSubSat, 1, 0x0000ffff, 0x00000001, 0x0000fffe),
            (Intrinsic::ParallelSubHalf, 1, 0x0000ffff, 0x00000001, 0x0000ffff),
            (Intrinsic::ParallelSubUns, 1, 0x0000ffff, 0x00000001, 0x0000fffe),
            (Intrinsic::ParallelSubSatUns, 1, 0x0000ffff, 0x00000001, 0x0000fffe),
            (Intrinsic::ParallelSubHalfUns, 1, 0x0000ffff, 0x00000001, 0x00007f7f),
            (Intrinsic::ParallelSub, 2, 0x0000ffff, 0x00000001, 0x0000fffe),
            (Intrinsic::ParallelSubSat, 2, 0x0000ffff, 0x00000001, 0x0000fffe),
            (Intrinsic::ParallelSubHalf, 2, 0x0000ffff, 0x00000001, 0x0000ffff),
            (Intrinsic::ParallelSubUns, 2, 0x0000ffff, 0x00000001, 0x0000fffe),
            (Intrinsic::ParallelSubSatUns, 2, 0x0000ffff, 0x00000001, 0x0000fffe),
            (Intrinsic::ParallelSubHalfUns, 2, 0x0000ffff, 0x00000001, 0x00007fff),
            (Intrinsic::ExchangeAdd, 2, 0x0000ffff, 0x00000001, 0x0001ffff),
            (Intrinsic::ExchangeSub, 2, 0x0000ffff, 0x00000001, 0xffffffff),
            (Intrinsic::ExchangeAddSat, 2, 0x0000ffff, 0x00000001, 0x0001ffff),
            (Intrinsic::ExchangeSubSat, 2, 0x0000ffff, 0x00000001, 0xffffffff),
            (Intrinsic::ExchangeAddHalf, 2, 0x0000ffff, 0x00000001, 0x0000ffff),
            (Intrinsic::ExchangeSubHalf, 2, 0x0000ffff, 0x00000001, 0xffffffff),
            (Intrinsic::ExchangeAddUns, 2, 0x0000ffff, 0x00000001, 0x0001ffff),
            (Intrinsic::ExchangeSubUns, 2, 0x0000ffff, 0x00000001, 0xffffffff),
            (Intrinsic::ExchangeAddSatUns, 2, 0x0000ffff, 0x00000001, 0x0001ffff),
            (Intrinsic::ExchangeSubSatUns, 2, 0x0000ffff, 0x00000001, 0x0000ffff),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x0000ffff, 0x00000001, 0x00007fff),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x0000ffff, 0x00000001, 0xffff7fff),
            (Intrinsic::ParallelAdd, 1, 0x00000001, 0x0000ffff, 0x0000ff00),
            (Intrinsic::ParallelAddSat, 1, 0x00000001, 0x0000ffff, 0x0000ff00),
            (Intrinsic::ParallelAddHalf, 1, 0x00000001, 0x0000ffff, 0x0000ff00),
            (Intrinsic::ParallelAddUns, 1, 0x00000001, 0x0000ffff, 0x0000ff00),
            (Intrinsic::ParallelAddSatUns, 1, 0x00000001, 0x0000ffff, 0x0000ffff),
            (Intrinsic::ParallelAddHalfUns, 1, 0x00000001, 0x0000ffff, 0x00007f80),
            (Intrinsic::ParallelAdd, 2, 0x00000001, 0x0000ffff, 0x00000000),
            (Intrinsic::ParallelAddSat, 2, 0x00000001, 0x0000ffff, 0x00000000),
            (Intrinsic::ParallelAddHalf, 2, 0x00000001, 0x0000ffff, 0x00000000),
            (Intrinsic::ParallelAddUns, 2, 0x00000001, 0x0000ffff, 0x00000000),
            (Intrinsic::ParallelAddSatUns, 2, 0x00000001, 0x0000ffff, 0x0000ffff),
            (Intrinsic::ParallelAddHalfUns, 2, 0x00000001, 0x0000ffff, 0x00008000),
            (Intrinsic::ParallelSub, 1, 0x00000001, 0x0000ffff, 0x00000102),
            (Intrinsic::ParallelSubSat, 1, 0x00000001, 0x0000ffff, 0x00000102),
            (Intrinsic::ParallelSubHalf, 1, 0x00000001, 0x0000ffff, 0x00000001),
            (Intrinsic::ParallelSubUns, 1, 0x00000001, 0x0000ffff, 0x00000102),
            (Intrinsic::ParallelSubSatUns, 1, 0x00000001, 0x0000ffff, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 1, 0x00000001, 0x0000ffff, 0x00008081),
            (Intrinsic::ParallelSub, 2, 0x00000001, 0x0000ffff, 0x00000002),
            (Intrinsic::ParallelSubSat, 2, 0x00000001, 0x0000ffff, 0x00000002),
            (Intrinsic::ParallelSubHalf, 2, 0x00000001, 0x0000ffff, 0x00000001),
            (Intrinsic::ParallelSubUns, 2, 0x00000001, 0x0000ffff, 0x00000002),
            (Intrinsic::ParallelSubSatUns, 2, 0x00000001, 0x0000ffff, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 2, 0x00000001, 0x0000ffff, 0x00008001),
            (Intrinsic::ExchangeAdd, 2, 0x00000001, 0x0000ffff, 0xffff0001),
            (Intrinsic::ExchangeSub, 2, 0x00000001, 0x0000ffff, 0x00010001),
            (Intrinsic::ExchangeAddSat, 2, 0x00000001, 0x0000ffff, 0xffff0001),
            (Intrinsic::ExchangeSubSat, 2, 0x00000001, 0x0000ffff, 0x00010001),
            (Intrinsic::ExchangeAddHalf, 2, 0x00000001, 0x0000ffff, 0xffff0000),
            (Intrinsic::ExchangeSubHalf, 2, 0x00000001, 0x0000ffff, 0x00000000),
            (Intrinsic::ExchangeAddUns, 2, 0x00000001, 0x0000ffff, 0xffff0001),
            (Intrinsic::ExchangeSubUns, 2, 0x00000001, 0x0000ffff, 0x00010001),
            (Intrinsic::ExchangeAddSatUns, 2, 0x00000001, 0x0000ffff, 0xffff0001),
            (Intrinsic::ExchangeSubSatUns, 2, 0x00000001, 0x0000ffff, 0x00000001),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x00000001, 0x0000ffff, 0x7fff0000),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x00000001, 0x0000ffff, 0x80000000),
            (Intrinsic::ParallelAdd, 1, 0x00008000, 0x00000001, 0x00008001),
            (Intrinsic::ParallelAddSat, 1, 0x00008000, 0x00000001, 0x00008001),
            (Intrinsic::ParallelAddHalf, 1, 0x00008000, 0x00000001, 0x0000c000),
            (Intrinsic::ParallelAddUns, 1, 0x00008000, 0x00000001, 0x00008001),
            (Intrinsic::ParallelAddSatUns, 1, 0x00008000, 0x00000001, 0x00008001),
            (Intrinsic::ParallelAddHalfUns, 1, 0x00008000, 0x00000001, 0x00004000),
            (Intrinsic::ParallelAdd, 2, 0x00008000, 0x00000001, 0x00008001),
            (Intrinsic::ParallelAddSat, 2, 0x00008000, 0x00000001, 0x00008001),
            (Intrinsic::ParallelAddHalf, 2, 0x00008000, 0x00000001, 0x0000c000),
            (Intrinsic::ParallelAddUns, 2, 0x00008000, 0x00000001, 0x00008001),
            (Intrinsic::ParallelAddSatUns, 2, 0x00008000, 0x00000001, 0x00008001),
            (Intrinsic::ParallelAddHalfUns, 2, 0x00008000, 0x00000001, 0x00004000),
            (Intrinsic::ParallelSub, 1, 0x00008000, 0x00000001, 0x000080ff),
            (Intrinsic::ParallelSubSat, 1, 0x00008000, 0x00000001, 0x000080ff),
            (Intrinsic::ParallelSubHalf, 1, 0x00008000, 0x00000001, 0x0000c0ff),
            (Intrinsic::ParallelSubUns, 1, 0x00008000, 0x00000001, 0x000080ff),
            (Intrinsic::ParallelSubSatUns, 1, 0x00008000, 0x00000001, 0x00008000),
            (Intrinsic::ParallelSubHalfUns, 1, 0x00008000, 0x00000001, 0x000040ff),
            (Intrinsic::ParallelSub, 2, 0x00008000, 0x00000001, 0x00007fff),
            (Intrinsic::ParallelSubSat, 2, 0x00008000, 0x00000001, 0x00008000),
            (Intrinsic::ParallelSubHalf, 2, 0x00008000, 0x00000001, 0x0000bfff),
            (Intrinsic::ParallelSubUns, 2, 0x00008000, 0x00000001, 0x00007fff),
            (Intrinsic::ParallelSubSatUns, 2, 0x00008000, 0x00000001, 0x00007fff),
            (Intrinsic::ParallelSubHalfUns, 2, 0x00008000, 0x00000001, 0x00003fff),
            (Intrinsic::ExchangeAdd, 2, 0x00008000, 0x00000001, 0x00018000),
            (Intrinsic::ExchangeSub, 2, 0x00008000, 0x00000001, 0xffff8000),
            (Intrinsic::ExchangeAddSat, 2, 0x00008000, 0x00000001, 0x00018000),
            (Intrinsic::ExchangeSubSat, 2, 0x00008000, 0x00000001, 0xffff8000),
            (Intrinsic::ExchangeAddHalf, 2, 0x00008000, 0x00000001, 0x0000c000),
            (Intrinsic::ExchangeSubHalf, 2, 0x00008000, 0x00000001, 0xffffc000),
            (Intrinsic::ExchangeAddUns, 2, 0x00008000, 0x00000001, 0x00018000),
            (Intrinsic::ExchangeSubUns, 2, 0x00008000, 0x00000001, 0xffff8000),
            (Intrinsic::ExchangeAddSatUns, 2, 0x00008000, 0x00000001, 0x00018000),
            (Intrinsic::ExchangeSubSatUns, 2, 0x00008000, 0x00000001, 0x00008000),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x00008000, 0x00000001, 0x00004000),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x00008000, 0x00000001, 0xffff4000),
            (Intrinsic::ParallelAdd, 1, 0x00000001, 0x00008000, 0x00008001),
            (Intrinsic::ParallelAddSat, 1, 0x00000001, 0x00008000, 0x00008001),
            (Intrinsic::ParallelAddHalf, 1, 0x00000001, 0x00008000, 0x0000c000),
            (Intrinsic::ParallelAddUns, 1, 0x00000001, 0x00008000, 0x00008001),
            (Intrinsic::ParallelAddSatUns, 1, 0x00000001, 0x00008000, 0x00008001),
            (Intrinsic::ParallelAddHalfUns, 1, 0x00000001, 0x00008000, 0x00004000),
            (Intrinsic::ParallelAdd, 2, 0x00000001, 0x00008000, 0x00008001),
            (Intrinsic::ParallelAddSat, 2, 0x00000001, 0x00008000, 0x00008001),
            (Intrinsic::ParallelAddHalf, 2, 0x00000001, 0x00008000, 0x0000c000),
            (Intrinsic::ParallelAddUns, 2, 0x00000001, 0x00008000, 0x00008001),
            (Intrinsic::ParallelAddSatUns, 2, 0x00000001, 0x00008000, 0x00008001),
            (Intrinsic::ParallelAddHalfUns, 2, 0x00000001, 0x00008000, 0x00004000),
            (Intrinsic::ParallelSub, 1, 0x00000001, 0x00008000, 0x00008001),
            (Intrinsic::ParallelSubSat, 1, 0x00000001, 0x00008000, 0x00007f01),
            (Intrinsic::ParallelSubHalf, 1, 0x00000001, 0x00008000, 0x00004000),
            (Intrinsic::ParallelSubUns, 1, 0x00000001, 0x00008000, 0x00008001),
            (Intrinsic::ParallelSubSatUns, 1, 0x00000001, 0x00008000, 0x00000001),
            (Intrinsic::ParallelSubHalfUns, 1, 0x00000001, 0x00008000, 0x0000c000),
            (Intrinsic::ParallelSub, 2, 0x00000001, 0x00008000, 0x00008001),
            (Intrinsic::ParallelSubSat, 2, 0x00000001, 0x00008000, 0x00007fff),
            (Intrinsic::ParallelSubHalf, 2, 0x00000001, 0x00008000, 0x00004000),
            (Intrinsic::ParallelSubUns, 2, 0x00000001, 0x00008000, 0x00008001),
            (Intrinsic::ParallelSubSatUns, 2, 0x00000001, 0x00008000, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 2, 0x00000001, 0x00008000, 0x0000c000),
            (Intrinsic::ExchangeAdd, 2, 0x00000001, 0x00008000, 0x80000001),
            (Intrinsic::ExchangeSub, 2, 0x00000001, 0x00008000, 0x80000001),
            (Intrinsic::ExchangeAddSat, 2, 0x00000001, 0x00008000, 0x80000001),
            (Intrinsic::ExchangeSubSat, 2, 0x00000001, 0x00008000, 0x7fff0001),
            (Intrinsic::ExchangeAddHalf, 2, 0x00000001, 0x00008000, 0xc0000000),
            (Intrinsic::ExchangeSubHalf, 2, 0x00000001, 0x00008000, 0x40000000),
            (Intrinsic::ExchangeAddUns, 2, 0x00000001, 0x00008000, 0x80000001),
            (Intrinsic::ExchangeSubUns, 2, 0x00000001, 0x00008000, 0x80000001),
            (Intrinsic::ExchangeAddSatUns, 2, 0x00000001, 0x00008000, 0x80000001),
            (Intrinsic::ExchangeSubSatUns, 2, 0x00000001, 0x00008000, 0x00000001),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x00000001, 0x00008000, 0x40000000),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x00000001, 0x00008000, 0xc0000000),
            (Intrinsic::ParallelAdd, 1, 0x80000000, 0x00000001, 0x80000001),
            (Intrinsic::ParallelAddSat, 1, 0x80000000, 0x00000001, 0x80000001),
            (Intrinsic::ParallelAddHalf, 1, 0x80000000, 0x00000001, 0xc0000000),
            (Intrinsic::ParallelAddUns, 1, 0x80000000, 0x00000001, 0x80000001),
            (Intrinsic::ParallelAddSatUns, 1, 0x80000000, 0x00000001, 0x80000001),
            (Intrinsic::ParallelAddHalfUns, 1, 0x80000000, 0x00000001, 0x40000000),
            (Intrinsic::ParallelAdd, 2, 0x80000000, 0x00000001, 0x80000001),
            (Intrinsic::ParallelAddSat, 2, 0x80000000, 0x00000001, 0x80000001),
            (Intrinsic::ParallelAddHalf, 2, 0x80000000, 0x00000001, 0xc0000000),
            (Intrinsic::ParallelAddUns, 2, 0x80000000, 0x00000001, 0x80000001),
            (Intrinsic::ParallelAddSatUns, 2, 0x80000000, 0x00000001, 0x80000001),
            (Intrinsic::ParallelAddHalfUns, 2, 0x80000000, 0x00000001, 0x40000000),
            (Intrinsic::ParallelSub, 1, 0x80000000, 0x00000001, 0x800000ff),
            (Intrinsic::ParallelSubSat, 1, 0x80000000, 0x00000001, 0x800000ff),
            (Intrinsic::ParallelSubHalf, 1, 0x80000000, 0x00000001, 0xc00000ff),
            (Intrinsic::ParallelSubUns, 1, 0x80000000, 0x00000001, 0x800000ff),
            (Intrinsic::ParallelSubSatUns, 1, 0x80000000, 0x00000001, 0x80000000),
            (Intrinsic::ParallelSubHalfUns, 1, 0x80000000, 0x00000001, 0x400000ff),
            (Intrinsic::ParallelSub, 2, 0x80000000, 0x00000001, 0x8000ffff),
            (Intrinsic::ParallelSubSat, 2, 0x80000000, 0x00000001, 0x8000ffff),
            (Intrinsic::ParallelSubHalf, 2, 0x80000000, 0x00000001, 0xc000ffff),
            (Intrinsic::ParallelSubUns, 2, 0x80000000, 0x00000001, 0x8000ffff),
            (Intrinsic::ParallelSubSatUns, 2, 0x80000000, 0x00000001, 0x80000000),
            (Intrinsic::ParallelSubHalfUns, 2, 0x80000000, 0x00000001, 0x4000ffff),
            (Intrinsic::ExchangeAdd, 2, 0x80000000, 0x00000001, 0x80010000),
            (Intrinsic::ExchangeSub, 2, 0x80000000, 0x00000001, 0x7fff0000),
            (Intrinsic::ExchangeAddSat, 2, 0x80000000, 0x00000001, 0x80010000),
            (Intrinsic::ExchangeSubSat, 2, 0x80000000, 0x00000001, 0x80000000),
            (Intrinsic::ExchangeAddHalf, 2, 0x80000000, 0x00000001, 0xc0000000),
            (Intrinsic::ExchangeSubHalf, 2, 0x80000000, 0x00000001, 0xbfff0000),
            (Intrinsic::ExchangeAddUns, 2, 0x80000000, 0x00000001, 0x80010000),
            (Intrinsic::ExchangeSubUns, 2, 0x80000000, 0x00000001, 0x7fff0000),
            (Intrinsic::ExchangeAddSatUns, 2, 0x80000000, 0x00000001, 0x80010000),
            (Intrinsic::ExchangeSubSatUns, 2, 0x80000000, 0x00000001, 0x7fff0000),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x80000000, 0x00000001, 0x40000000),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x80000000, 0x00000001, 0x3fff0000),
            (Intrinsic::ParallelAdd, 1, 0x00000001, 0x80000000, 0x80000001),
            (Intrinsic::ParallelAddSat, 1, 0x00000001, 0x80000000, 0x80000001),
            (Intrinsic::ParallelAddHalf, 1, 0x00000001, 0x80000000, 0xc0000000),
            (Intrinsic::ParallelAddUns, 1, 0x00000001, 0x80000000, 0x80000001),
            (Intrinsic::ParallelAddSatUns, 1, 0x00000001, 0x80000000, 0x80000001),
            (Intrinsic::ParallelAddHalfUns, 1, 0x00000001, 0x80000000, 0x40000000),
            (Intrinsic::ParallelAdd, 2, 0x00000001, 0x80000000, 0x80000001),
            (Intrinsic::ParallelAddSat, 2, 0x00000001, 0x80000000, 0x80000001),
            (Intrinsic::ParallelAddHalf, 2, 0x00000001, 0x80000000, 0xc0000000),
            (Intrinsic::ParallelAddUns, 2, 0x00000001, 0x80000000, 0x80000001),
            (Intrinsic::ParallelAddSatUns, 2, 0x00000001, 0x80000000, 0x80000001),
            (Intrinsic::ParallelAddHalfUns, 2, 0x00000001, 0x80000000, 0x40000000),
            (Intrinsic::ParallelSub, 1, 0x00000001, 0x80000000, 0x80000001),
            (Intrinsic::ParallelSubSat, 1, 0x00000001, 0x80000000, 0x7f000001),
            (Intrinsic::ParallelSubHalf, 1, 0x00000001, 0x80000000, 0x40000000),
            (Intrinsic::ParallelSubUns, 1, 0x00000001, 0x80000000, 0x80000001),
            (Intrinsic::ParallelSubSatUns, 1, 0x00000001, 0x80000000, 0x00000001),
            (Intrinsic::ParallelSubHalfUns, 1, 0x00000001, 0x80000000, 0xc0000000),
            (Intrinsic::ParallelSub, 2, 0x00000001, 0x80000000, 0x80000001),
            (Intrinsic::ParallelSubSat, 2, 0x00000001, 0x80000000, 0x7fff0001),
            (Intrinsic::ParallelSubHalf, 2, 0x00000001, 0x80000000, 0x40000000),
            (Intrinsic::ParallelSubUns, 2, 0x00000001, 0x80000000, 0x80000001),
            (Intrinsic::ParallelSubSatUns, 2, 0x00000001, 0x80000000, 0x00000001),
            (Intrinsic::ParallelSubHalfUns, 2, 0x00000001, 0x80000000, 0xc0000000),
            (Intrinsic::ExchangeAdd, 2, 0x00000001, 0x80000000, 0x00008001),
            (Intrinsic::ExchangeSub, 2, 0x00000001, 0x80000000, 0x00008001),
            (Intrinsic::ExchangeAddSat, 2, 0x00000001, 0x80000000, 0x00007fff),
            (Intrinsic::ExchangeSubSat, 2, 0x00000001, 0x80000000, 0x00008001),
            (Intrinsic::ExchangeAddHalf, 2, 0x00000001, 0x80000000, 0x00004000),
            (Intrinsic::ExchangeSubHalf, 2, 0x00000001, 0x80000000, 0x0000c000),
            (Intrinsic::ExchangeAddUns, 2, 0x00000001, 0x80000000, 0x00008001),
            (Intrinsic::ExchangeSubUns, 2, 0x00000001, 0x80000000, 0x00008001),
            (Intrinsic::ExchangeAddSatUns, 2, 0x00000001, 0x80000000, 0x00000000),
            (Intrinsic::ExchangeSubSatUns, 2, 0x00000001, 0x80000000, 0x00008001),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x00000001, 0x80000000, 0x0000c000),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x00000001, 0x80000000, 0x00004000),
            (Intrinsic::ParallelAdd, 1, 0x7fffffff, 0x00000001, 0x7fffff00),
            (Intrinsic::ParallelAddSat, 1, 0x7fffffff, 0x00000001, 0x7fffff00),
            (Intrinsic::ParallelAddHalf, 1, 0x7fffffff, 0x00000001, 0x3fffff00),
            (Intrinsic::ParallelAddUns, 1, 0x7fffffff, 0x00000001, 0x7fffff00),
            (Intrinsic::ParallelAddSatUns, 1, 0x7fffffff, 0x00000001, 0x7fffffff),
            (Intrinsic::ParallelAddHalfUns, 1, 0x7fffffff, 0x00000001, 0x3f7f7f80),
            (Intrinsic::ParallelAdd, 2, 0x7fffffff, 0x00000001, 0x7fff0000),
            (Intrinsic::ParallelAddSat, 2, 0x7fffffff, 0x00000001, 0x7fff0000),
            (Intrinsic::ParallelAddHalf, 2, 0x7fffffff, 0x00000001, 0x3fff0000),
            (Intrinsic::ParallelAddUns, 2, 0x7fffffff, 0x00000001, 0x7fff0000),
            (Intrinsic::ParallelAddSatUns, 2, 0x7fffffff, 0x00000001, 0x7fffffff),
            (Intrinsic::ParallelAddHalfUns, 2, 0x7fffffff, 0x00000001, 0x3fff8000),
            (Intrinsic::ParallelSub, 1, 0x7fffffff, 0x00000001, 0x7ffffffe),
            (Intrinsic::ParallelSubSat, 1, 0x7fffffff, 0x00000001, 0x7ffffffe),
            (Intrinsic::ParallelSubHalf, 1, 0x7fffffff, 0x00000001, 0x3fffffff),
            (Intrinsic::ParallelSubUns, 1, 0x7fffffff, 0x00000001, 0x7ffffffe),
            (Intrinsic::ParallelSubSatUns, 1, 0x7fffffff, 0x00000001, 0x7ffffffe),
            (Intrinsic::ParallelSubHalfUns, 1, 0x7fffffff, 0x00000001, 0x3f7f7f7f),
            (Intrinsic::ParallelSub, 2, 0x7fffffff, 0x00000001, 0x7ffffffe),
            (Intrinsic::ParallelSubSat, 2, 0x7fffffff, 0x00000001, 0x7ffffffe),
            (Intrinsic::ParallelSubHalf, 2, 0x7fffffff, 0x00000001, 0x3fffffff),
            (Intrinsic::ParallelSubUns, 2, 0x7fffffff, 0x00000001, 0x7ffffffe),
            (Intrinsic::ParallelSubSatUns, 2, 0x7fffffff, 0x00000001, 0x7ffffffe),
            (Intrinsic::ParallelSubHalfUns, 2, 0x7fffffff, 0x00000001, 0x3fff7fff),
            (Intrinsic::ExchangeAdd, 2, 0x7fffffff, 0x00000001, 0x8000ffff),
            (Intrinsic::ExchangeSub, 2, 0x7fffffff, 0x00000001, 0x7ffeffff),
            (Intrinsic::ExchangeAddSat, 2, 0x7fffffff, 0x00000001, 0x7fffffff),
            (Intrinsic::ExchangeSubSat, 2, 0x7fffffff, 0x00000001, 0x7ffeffff),
            (Intrinsic::ExchangeAddHalf, 2, 0x7fffffff, 0x00000001, 0x4000ffff),
            (Intrinsic::ExchangeSubHalf, 2, 0x7fffffff, 0x00000001, 0x3fffffff),
            (Intrinsic::ExchangeAddUns, 2, 0x7fffffff, 0x00000001, 0x8000ffff),
            (Intrinsic::ExchangeSubUns, 2, 0x7fffffff, 0x00000001, 0x7ffeffff),
            (Intrinsic::ExchangeAddSatUns, 2, 0x7fffffff, 0x00000001, 0x8000ffff),
            (Intrinsic::ExchangeSubSatUns, 2, 0x7fffffff, 0x00000001, 0x7ffeffff),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x7fffffff, 0x00000001, 0x40007fff),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x7fffffff, 0x00000001, 0x3fff7fff),
            (Intrinsic::ParallelAdd, 1, 0x00010000, 0x00000001, 0x00010001),
            (Intrinsic::ParallelAddSat, 1, 0x00010000, 0x00000001, 0x00010001),
            (Intrinsic::ParallelAddHalf, 1, 0x00010000, 0x00000001, 0x00000000),
            (Intrinsic::ParallelAddUns, 1, 0x00010000, 0x00000001, 0x00010001),
            (Intrinsic::ParallelAddSatUns, 1, 0x00010000, 0x00000001, 0x00010001),
            (Intrinsic::ParallelAddHalfUns, 1, 0x00010000, 0x00000001, 0x00000000),
            (Intrinsic::ParallelAdd, 2, 0x00010000, 0x00000001, 0x00010001),
            (Intrinsic::ParallelAddSat, 2, 0x00010000, 0x00000001, 0x00010001),
            (Intrinsic::ParallelAddHalf, 2, 0x00010000, 0x00000001, 0x00000000),
            (Intrinsic::ParallelAddUns, 2, 0x00010000, 0x00000001, 0x00010001),
            (Intrinsic::ParallelAddSatUns, 2, 0x00010000, 0x00000001, 0x00010001),
            (Intrinsic::ParallelAddHalfUns, 2, 0x00010000, 0x00000001, 0x00000000),
            (Intrinsic::ParallelSub, 1, 0x00010000, 0x00000001, 0x000100ff),
            (Intrinsic::ParallelSubSat, 1, 0x00010000, 0x00000001, 0x000100ff),
            (Intrinsic::ParallelSubHalf, 1, 0x00010000, 0x00000001, 0x000000ff),
            (Intrinsic::ParallelSubUns, 1, 0x00010000, 0x00000001, 0x000100ff),
            (Intrinsic::ParallelSubSatUns, 1, 0x00010000, 0x00000001, 0x00010000),
            (Intrinsic::ParallelSubHalfUns, 1, 0x00010000, 0x00000001, 0x000000ff),
            (Intrinsic::ParallelSub, 2, 0x00010000, 0x00000001, 0x0001ffff),
            (Intrinsic::ParallelSubSat, 2, 0x00010000, 0x00000001, 0x0001ffff),
            (Intrinsic::ParallelSubHalf, 2, 0x00010000, 0x00000001, 0x0000ffff),
            (Intrinsic::ParallelSubUns, 2, 0x00010000, 0x00000001, 0x0001ffff),
            (Intrinsic::ParallelSubSatUns, 2, 0x00010000, 0x00000001, 0x00010000),
            (Intrinsic::ParallelSubHalfUns, 2, 0x00010000, 0x00000001, 0x0000ffff),
            (Intrinsic::ExchangeAdd, 2, 0x00010000, 0x00000001, 0x00020000),
            (Intrinsic::ExchangeSub, 2, 0x00010000, 0x00000001, 0x00000000),
            (Intrinsic::ExchangeAddSat, 2, 0x00010000, 0x00000001, 0x00020000),
            (Intrinsic::ExchangeSubSat, 2, 0x00010000, 0x00000001, 0x00000000),
            (Intrinsic::ExchangeAddHalf, 2, 0x00010000, 0x00000001, 0x00010000),
            (Intrinsic::ExchangeSubHalf, 2, 0x00010000, 0x00000001, 0x00000000),
            (Intrinsic::ExchangeAddUns, 2, 0x00010000, 0x00000001, 0x00020000),
            (Intrinsic::ExchangeSubUns, 2, 0x00010000, 0x00000001, 0x00000000),
            (Intrinsic::ExchangeAddSatUns, 2, 0x00010000, 0x00000001, 0x00020000),
            (Intrinsic::ExchangeSubSatUns, 2, 0x00010000, 0x00000001, 0x00000000),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x00010000, 0x00000001, 0x00010000),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x00010000, 0x00000001, 0x00000000),
            (Intrinsic::ParallelAdd, 1, 0x00000001, 0x00010000, 0x00010001),
            (Intrinsic::ParallelAddSat, 1, 0x00000001, 0x00010000, 0x00010001),
            (Intrinsic::ParallelAddHalf, 1, 0x00000001, 0x00010000, 0x00000000),
            (Intrinsic::ParallelAddUns, 1, 0x00000001, 0x00010000, 0x00010001),
            (Intrinsic::ParallelAddSatUns, 1, 0x00000001, 0x00010000, 0x00010001),
            (Intrinsic::ParallelAddHalfUns, 1, 0x00000001, 0x00010000, 0x00000000),
            (Intrinsic::ParallelAdd, 2, 0x00000001, 0x00010000, 0x00010001),
            (Intrinsic::ParallelAddSat, 2, 0x00000001, 0x00010000, 0x00010001),
            (Intrinsic::ParallelAddHalf, 2, 0x00000001, 0x00010000, 0x00000000),
            (Intrinsic::ParallelAddUns, 2, 0x00000001, 0x00010000, 0x00010001),
            (Intrinsic::ParallelAddSatUns, 2, 0x00000001, 0x00010000, 0x00010001),
            (Intrinsic::ParallelAddHalfUns, 2, 0x00000001, 0x00010000, 0x00000000),
            (Intrinsic::ParallelSub, 1, 0x00000001, 0x00010000, 0x00ff0001),
            (Intrinsic::ParallelSubSat, 1, 0x00000001, 0x00010000, 0x00ff0001),
            (Intrinsic::ParallelSubHalf, 1, 0x00000001, 0x00010000, 0x00ff0000),
            (Intrinsic::ParallelSubUns, 1, 0x00000001, 0x00010000, 0x00ff0001),
            (Intrinsic::ParallelSubSatUns, 1, 0x00000001, 0x00010000, 0x00000001),
            (Intrinsic::ParallelSubHalfUns, 1, 0x00000001, 0x00010000, 0x00ff0000),
            (Intrinsic::ParallelSub, 2, 0x00000001, 0x00010000, 0xffff0001),
            (Intrinsic::ParallelSubSat, 2, 0x00000001, 0x00010000, 0xffff0001),
            (Intrinsic::ParallelSubHalf, 2, 0x00000001, 0x00010000, 0xffff0000),
            (Intrinsic::ParallelSubUns, 2, 0x00000001, 0x00010000, 0xffff0001),
            (Intrinsic::ParallelSubSatUns, 2, 0x00000001, 0x00010000, 0x00000001),
            (Intrinsic::ParallelSubHalfUns, 2, 0x00000001, 0x00010000, 0xffff0000),
            (Intrinsic::ExchangeAdd, 2, 0x00000001, 0x00010000, 0x00000000),
            (Intrinsic::ExchangeSub, 2, 0x00000001, 0x00010000, 0x00000002),
            (Intrinsic::ExchangeAddSat, 2, 0x00000001, 0x00010000, 0x00000000),
            (Intrinsic::ExchangeSubSat, 2, 0x00000001, 0x00010000, 0x00000002),
            (Intrinsic::ExchangeAddHalf, 2, 0x00000001, 0x00010000, 0x00000000),
            (Intrinsic::ExchangeSubHalf, 2, 0x00000001, 0x00010000, 0x00000001),
            (Intrinsic::ExchangeAddUns, 2, 0x00000001, 0x00010000, 0x00000000),
            (Intrinsic::ExchangeSubUns, 2, 0x00000001, 0x00010000, 0x00000002),
            (Intrinsic::ExchangeAddSatUns, 2, 0x00000001, 0x00010000, 0x00000000),
            (Intrinsic::ExchangeSubSatUns, 2, 0x00000001, 0x00010000, 0x00000002),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x00000001, 0x00010000, 0x00000000),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x00000001, 0x00010000, 0x00000001),
            (Intrinsic::ParallelAdd, 1, 0xfedcba98, 0x12345678, 0x10101010),
            (Intrinsic::ParallelAddSat, 1, 0xfedcba98, 0x12345678, 0x10101010),
            (Intrinsic::ParallelAddHalf, 1, 0xfedcba98, 0x12345678, 0x08080808),
            (Intrinsic::ParallelAddUns, 1, 0xfedcba98, 0x12345678, 0x10101010),
            (Intrinsic::ParallelAddSatUns, 1, 0xfedcba98, 0x12345678, 0xffffffff),
            (Intrinsic::ParallelAddHalfUns, 1, 0xfedcba98, 0x12345678, 0x88888888),
            (Intrinsic::ParallelAdd, 2, 0xfedcba98, 0x12345678, 0x11101110),
            (Intrinsic::ParallelAddSat, 2, 0xfedcba98, 0x12345678, 0x11101110),
            (Intrinsic::ParallelAddHalf, 2, 0xfedcba98, 0x12345678, 0x08880888),
            (Intrinsic::ParallelAddUns, 2, 0xfedcba98, 0x12345678, 0x11101110),
            (Intrinsic::ParallelAddSatUns, 2, 0xfedcba98, 0x12345678, 0xffffffff),
            (Intrinsic::ParallelAddHalfUns, 2, 0xfedcba98, 0x12345678, 0x88888888),
            (Intrinsic::ParallelSub, 1, 0xfedcba98, 0x12345678, 0xeca86420),
            (Intrinsic::ParallelSubSat, 1, 0xfedcba98, 0x12345678, 0xeca88080),
            (Intrinsic::ParallelSubHalf, 1, 0xfedcba98, 0x12345678, 0xf6d4b290),
            (Intrinsic::ParallelSubUns, 1, 0xfedcba98, 0x12345678, 0xeca86420),
            (Intrinsic::ParallelSubSatUns, 1, 0xfedcba98, 0x12345678, 0xeca86420),
            (Intrinsic::ParallelSubHalfUns, 1, 0xfedcba98, 0x12345678, 0x76543210),
            (Intrinsic::ParallelSub, 2, 0xfedcba98, 0x12345678, 0xeca86420),
            (Intrinsic::ParallelSubSat, 2, 0xfedcba98, 0x12345678, 0xeca88000),
            (Intrinsic::ParallelSubHalf, 2, 0xfedcba98, 0x12345678, 0xf654b210),
            (Intrinsic::ParallelSubUns, 2, 0xfedcba98, 0x12345678, 0xeca86420),
            (Intrinsic::ParallelSubSatUns, 2, 0xfedcba98, 0x12345678, 0xeca86420),
            (Intrinsic::ParallelSubHalfUns, 2, 0xfedcba98, 0x12345678, 0x76543210),
            (Intrinsic::ExchangeAdd, 2, 0xfedcba98, 0x12345678, 0x5554a864),
            (Intrinsic::ExchangeSub, 2, 0xfedcba98, 0x12345678, 0xa864cccc),
            (Intrinsic::ExchangeAddSat, 2, 0xfedcba98, 0x12345678, 0x5554a864),
            (Intrinsic::ExchangeSubSat, 2, 0xfedcba98, 0x12345678, 0xa864cccc),
            (Intrinsic::ExchangeAddHalf, 2, 0xfedcba98, 0x12345678, 0x2aaad432),
            (Intrinsic::ExchangeSubHalf, 2, 0xfedcba98, 0x12345678, 0xd432e666),
            (Intrinsic::ExchangeAddUns, 2, 0xfedcba98, 0x12345678, 0x5554a864),
            (Intrinsic::ExchangeSubUns, 2, 0xfedcba98, 0x12345678, 0xa864cccc),
            (Intrinsic::ExchangeAddSatUns, 2, 0xfedcba98, 0x12345678, 0xffffa864),
            (Intrinsic::ExchangeSubSatUns, 2, 0xfedcba98, 0x12345678, 0xa864cccc),
            (Intrinsic::ExchangeAddHalfUns, 2, 0xfedcba98, 0x12345678, 0xaaaa5432),
            (Intrinsic::ExchangeSubHalfUns, 2, 0xfedcba98, 0x12345678, 0x54326666),
            (Intrinsic::ParallelAdd, 1, 0x0000ffff, 0x0000ffff, 0x0000fefe),
            (Intrinsic::ParallelAddSat, 1, 0x0000ffff, 0x0000ffff, 0x0000fefe),
            (Intrinsic::ParallelAddHalf, 1, 0x0000ffff, 0x0000ffff, 0x0000ffff),
            (Intrinsic::ParallelAddUns, 1, 0x0000ffff, 0x0000ffff, 0x0000fefe),
            (Intrinsic::ParallelAddSatUns, 1, 0x0000ffff, 0x0000ffff, 0x0000ffff),
            (Intrinsic::ParallelAddHalfUns, 1, 0x0000ffff, 0x0000ffff, 0x0000ffff),
            (Intrinsic::ParallelAdd, 2, 0x0000ffff, 0x0000ffff, 0x0000fffe),
            (Intrinsic::ParallelAddSat, 2, 0x0000ffff, 0x0000ffff, 0x0000fffe),
            (Intrinsic::ParallelAddHalf, 2, 0x0000ffff, 0x0000ffff, 0x0000ffff),
            (Intrinsic::ParallelAddUns, 2, 0x0000ffff, 0x0000ffff, 0x0000fffe),
            (Intrinsic::ParallelAddSatUns, 2, 0x0000ffff, 0x0000ffff, 0x0000ffff),
            (Intrinsic::ParallelAddHalfUns, 2, 0x0000ffff, 0x0000ffff, 0x0000ffff),
            (Intrinsic::ParallelSub, 1, 0x0000ffff, 0x0000ffff, 0x00000000),
            (Intrinsic::ParallelSubSat, 1, 0x0000ffff, 0x0000ffff, 0x00000000),
            (Intrinsic::ParallelSubHalf, 1, 0x0000ffff, 0x0000ffff, 0x00000000),
            (Intrinsic::ParallelSubUns, 1, 0x0000ffff, 0x0000ffff, 0x00000000),
            (Intrinsic::ParallelSubSatUns, 1, 0x0000ffff, 0x0000ffff, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 1, 0x0000ffff, 0x0000ffff, 0x00000000),
            (Intrinsic::ParallelSub, 2, 0x0000ffff, 0x0000ffff, 0x00000000),
            (Intrinsic::ParallelSubSat, 2, 0x0000ffff, 0x0000ffff, 0x00000000),
            (Intrinsic::ParallelSubHalf, 2, 0x0000ffff, 0x0000ffff, 0x00000000),
            (Intrinsic::ParallelSubUns, 2, 0x0000ffff, 0x0000ffff, 0x00000000),
            (Intrinsic::ParallelSubSatUns, 2, 0x0000ffff, 0x0000ffff, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 2, 0x0000ffff, 0x0000ffff, 0x00000000),
            (Intrinsic::ExchangeAdd, 2, 0x0000ffff, 0x0000ffff, 0xffffffff),
            (Intrinsic::ExchangeSub, 2, 0x0000ffff, 0x0000ffff, 0x0001ffff),
            (Intrinsic::ExchangeAddSat, 2, 0x0000ffff, 0x0000ffff, 0xffffffff),
            (Intrinsic::ExchangeSubSat, 2, 0x0000ffff, 0x0000ffff, 0x0001ffff),
            (Intrinsic::ExchangeAddHalf, 2, 0x0000ffff, 0x0000ffff, 0xffffffff),
            (Intrinsic::ExchangeSubHalf, 2, 0x0000ffff, 0x0000ffff, 0x0000ffff),
            (Intrinsic::ExchangeAddUns, 2, 0x0000ffff, 0x0000ffff, 0xffffffff),
            (Intrinsic::ExchangeSubUns, 2, 0x0000ffff, 0x0000ffff, 0x0001ffff),
            (Intrinsic::ExchangeAddSatUns, 2, 0x0000ffff, 0x0000ffff, 0xffffffff),
            (Intrinsic::ExchangeSubSatUns, 2, 0x0000ffff, 0x0000ffff, 0x0000ffff),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x0000ffff, 0x0000ffff, 0x7fff7fff),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x0000ffff, 0x0000ffff, 0x80007fff),
            (Intrinsic::ParallelAdd, 1, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ParallelAddSat, 1, 0x80008000, 0x80008000, 0x80008000),
            (Intrinsic::ParallelAddHalf, 1, 0x80008000, 0x80008000, 0x80008000),
            (Intrinsic::ParallelAddUns, 1, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ParallelAddSatUns, 1, 0x80008000, 0x80008000, 0xff00ff00),
            (Intrinsic::ParallelAddHalfUns, 1, 0x80008000, 0x80008000, 0x80008000),
            (Intrinsic::ParallelAdd, 2, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ParallelAddSat, 2, 0x80008000, 0x80008000, 0x80008000),
            (Intrinsic::ParallelAddHalf, 2, 0x80008000, 0x80008000, 0x80008000),
            (Intrinsic::ParallelAddUns, 2, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ParallelAddSatUns, 2, 0x80008000, 0x80008000, 0xffffffff),
            (Intrinsic::ParallelAddHalfUns, 2, 0x80008000, 0x80008000, 0x80008000),
            (Intrinsic::ParallelSub, 1, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ParallelSubSat, 1, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ParallelSubHalf, 1, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ParallelSubUns, 1, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ParallelSubSatUns, 1, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 1, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ParallelSub, 2, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ParallelSubSat, 2, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ParallelSubHalf, 2, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ParallelSubUns, 2, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ParallelSubSatUns, 2, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 2, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ExchangeAdd, 2, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ExchangeSub, 2, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ExchangeAddSat, 2, 0x80008000, 0x80008000, 0x80000000),
            (Intrinsic::ExchangeSubSat, 2, 0x80008000, 0x80008000, 0x00008000),
            (Intrinsic::ExchangeAddHalf, 2, 0x80008000, 0x80008000, 0x80000000),
            (Intrinsic::ExchangeSubHalf, 2, 0x80008000, 0x80008000, 0x00008000),
            (Intrinsic::ExchangeAddUns, 2, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ExchangeSubUns, 2, 0x80008000, 0x80008000, 0x00000000),
            (Intrinsic::ExchangeAddSatUns, 2, 0x80008000, 0x80008000, 0xffff0000),
            (Intrinsic::ExchangeSubSatUns, 2, 0x80008000, 0x80008000, 0x0000ffff),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x80008000, 0x80008000, 0x80000000),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x80008000, 0x80008000, 0x00008000),
            (Intrinsic::ParallelAdd, 1, 0x7fff0000, 0x0000ffff, 0x7fffffff),
            (Intrinsic::ParallelAddSat, 1, 0x7fff0000, 0x0000ffff, 0x7fffffff),
            (Intrinsic::ParallelAddHalf, 1, 0x7fff0000, 0x0000ffff, 0x3fffffff),
            (Intrinsic::ParallelAddUns, 1, 0x7fff0000, 0x0000ffff, 0x7fffffff),
            (Intrinsic::ParallelAddSatUns, 1, 0x7fff0000, 0x0000ffff, 0x7fffffff),
            (Intrinsic::ParallelAddHalfUns, 1, 0x7fff0000, 0x0000ffff, 0x3f7f7f7f),
            (Intrinsic::ParallelAdd, 2, 0x7fff0000, 0x0000ffff, 0x7fffffff),
            (Intrinsic::ParallelAddSat, 2, 0x7fff0000, 0x0000ffff, 0x7fffffff),
            (Intrinsic::ParallelAddHalf, 2, 0x7fff0000, 0x0000ffff, 0x3fffffff),
            (Intrinsic::ParallelAddUns, 2, 0x7fff0000, 0x0000ffff, 0x7fffffff),
            (Intrinsic::ParallelAddSatUns, 2, 0x7fff0000, 0x0000ffff, 0x7fffffff),
            (Intrinsic::ParallelAddHalfUns, 2, 0x7fff0000, 0x0000ffff, 0x3fff7fff),
            (Intrinsic::ParallelSub, 1, 0x7fff0000, 0x0000ffff, 0x7fff0101),
            (Intrinsic::ParallelSubSat, 1, 0x7fff0000, 0x0000ffff, 0x7fff0101),
            (Intrinsic::ParallelSubHalf, 1, 0x7fff0000, 0x0000ffff, 0x3fff0000),
            (Intrinsic::ParallelSubUns, 1, 0x7fff0000, 0x0000ffff, 0x7fff0101),
            (Intrinsic::ParallelSubSatUns, 1, 0x7fff0000, 0x0000ffff, 0x7fff0000),
            (Intrinsic::ParallelSubHalfUns, 1, 0x7fff0000, 0x0000ffff, 0x3f7f8080),
            (Intrinsic::ParallelSub, 2, 0x7fff0000, 0x0000ffff, 0x7fff0001),
            (Intrinsic::ParallelSubSat, 2, 0x7fff0000, 0x0000ffff, 0x7fff0001),
            (Intrinsic::ParallelSubHalf, 2, 0x7fff0000, 0x0000ffff, 0x3fff0000),
            (Intrinsic::ParallelSubUns, 2, 0x7fff0000, 0x0000ffff, 0x7fff0001),
            (Intrinsic::ParallelSubSatUns, 2, 0x7fff0000, 0x0000ffff, 0x7fff0000),
            (Intrinsic::ParallelSubHalfUns, 2, 0x7fff0000, 0x0000ffff, 0x3fff8000),
            (Intrinsic::ExchangeAdd, 2, 0x7fff0000, 0x0000ffff, 0x7ffe0000),
            (Intrinsic::ExchangeSub, 2, 0x7fff0000, 0x0000ffff, 0x80000000),
            (Intrinsic::ExchangeAddSat, 2, 0x7fff0000, 0x0000ffff, 0x7ffe0000),
            (Intrinsic::ExchangeSubSat, 2, 0x7fff0000, 0x0000ffff, 0x7fff0000),
            (Intrinsic::ExchangeAddHalf, 2, 0x7fff0000, 0x0000ffff, 0x3fff0000),
            (Intrinsic::ExchangeSubHalf, 2, 0x7fff0000, 0x0000ffff, 0x40000000),
            (Intrinsic::ExchangeAddUns, 2, 0x7fff0000, 0x0000ffff, 0x7ffe0000),
            (Intrinsic::ExchangeSubUns, 2, 0x7fff0000, 0x0000ffff, 0x80000000),
            (Intrinsic::ExchangeAddSatUns, 2, 0x7fff0000, 0x0000ffff, 0xffff0000),
            (Intrinsic::ExchangeSubSatUns, 2, 0x7fff0000, 0x0000ffff, 0x00000000),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x7fff0000, 0x0000ffff, 0xbfff0000),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x7fff0000, 0x0000ffff, 0xc0000000),
            (Intrinsic::ParallelAdd, 1, 0x0000ff00, 0x000000ff, 0x0000ffff),
            (Intrinsic::ParallelAddSat, 1, 0x0000ff00, 0x000000ff, 0x0000ffff),
            (Intrinsic::ParallelAddHalf, 1, 0x0000ff00, 0x000000ff, 0x0000ffff),
            (Intrinsic::ParallelAddUns, 1, 0x0000ff00, 0x000000ff, 0x0000ffff),
            (Intrinsic::ParallelAddSatUns, 1, 0x0000ff00, 0x000000ff, 0x0000ffff),
            (Intrinsic::ParallelAddHalfUns, 1, 0x0000ff00, 0x000000ff, 0x00007f7f),
            (Intrinsic::ParallelAdd, 2, 0x0000ff00, 0x000000ff, 0x0000ffff),
            (Intrinsic::ParallelAddSat, 2, 0x0000ff00, 0x000000ff, 0x0000ffff),
            (Intrinsic::ParallelAddHalf, 2, 0x0000ff00, 0x000000ff, 0x0000ffff),
            (Intrinsic::ParallelAddUns, 2, 0x0000ff00, 0x000000ff, 0x0000ffff),
            (Intrinsic::ParallelAddSatUns, 2, 0x0000ff00, 0x000000ff, 0x0000ffff),
            (Intrinsic::ParallelAddHalfUns, 2, 0x0000ff00, 0x000000ff, 0x00007fff),
            (Intrinsic::ParallelSub, 1, 0x0000ff00, 0x000000ff, 0x0000ff01),
            (Intrinsic::ParallelSubSat, 1, 0x0000ff00, 0x000000ff, 0x0000ff01),
            (Intrinsic::ParallelSubHalf, 1, 0x0000ff00, 0x000000ff, 0x0000ff00),
            (Intrinsic::ParallelSubUns, 1, 0x0000ff00, 0x000000ff, 0x0000ff01),
            (Intrinsic::ParallelSubSatUns, 1, 0x0000ff00, 0x000000ff, 0x0000ff00),
            (Intrinsic::ParallelSubHalfUns, 1, 0x0000ff00, 0x000000ff, 0x00007f80),
            (Intrinsic::ParallelSub, 2, 0x0000ff00, 0x000000ff, 0x0000fe01),
            (Intrinsic::ParallelSubSat, 2, 0x0000ff00, 0x000000ff, 0x0000fe01),
            (Intrinsic::ParallelSubHalf, 2, 0x0000ff00, 0x000000ff, 0x0000ff00),
            (Intrinsic::ParallelSubUns, 2, 0x0000ff00, 0x000000ff, 0x0000fe01),
            (Intrinsic::ParallelSubSatUns, 2, 0x0000ff00, 0x000000ff, 0x0000fe01),
            (Intrinsic::ParallelSubHalfUns, 2, 0x0000ff00, 0x000000ff, 0x00007f00),
            (Intrinsic::ExchangeAdd, 2, 0x0000ff00, 0x000000ff, 0x00ffff00),
            (Intrinsic::ExchangeSub, 2, 0x0000ff00, 0x000000ff, 0xff01ff00),
            (Intrinsic::ExchangeAddSat, 2, 0x0000ff00, 0x000000ff, 0x00ffff00),
            (Intrinsic::ExchangeSubSat, 2, 0x0000ff00, 0x000000ff, 0xff01ff00),
            (Intrinsic::ExchangeAddHalf, 2, 0x0000ff00, 0x000000ff, 0x007fff80),
            (Intrinsic::ExchangeSubHalf, 2, 0x0000ff00, 0x000000ff, 0xff80ff80),
            (Intrinsic::ExchangeAddUns, 2, 0x0000ff00, 0x000000ff, 0x00ffff00),
            (Intrinsic::ExchangeSubUns, 2, 0x0000ff00, 0x000000ff, 0xff01ff00),
            (Intrinsic::ExchangeAddSatUns, 2, 0x0000ff00, 0x000000ff, 0x00ffff00),
            (Intrinsic::ExchangeSubSatUns, 2, 0x0000ff00, 0x000000ff, 0x0000ff00),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x0000ff00, 0x000000ff, 0x007f7f80),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x0000ff00, 0x000000ff, 0xff807f80),
            (Intrinsic::ParallelAdd, 1, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelAddSat, 1, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelAddHalf, 1, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelAddUns, 1, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelAddSatUns, 1, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelAddHalfUns, 1, 0x0000ffff, 0x00000000, 0x00007f7f),
            (Intrinsic::ParallelAdd, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelAddSat, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelAddHalf, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelAddUns, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelAddSatUns, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelAddHalfUns, 2, 0x0000ffff, 0x00000000, 0x00007fff),
            (Intrinsic::ParallelSub, 1, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelSubSat, 1, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelSubHalf, 1, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelSubUns, 1, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelSubSatUns, 1, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelSubHalfUns, 1, 0x0000ffff, 0x00000000, 0x00007f7f),
            (Intrinsic::ParallelSub, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelSubSat, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelSubHalf, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelSubUns, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelSubSatUns, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ParallelSubHalfUns, 2, 0x0000ffff, 0x00000000, 0x00007fff),
            (Intrinsic::ExchangeAdd, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ExchangeSub, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ExchangeAddSat, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ExchangeSubSat, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ExchangeAddHalf, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ExchangeSubHalf, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ExchangeAddUns, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ExchangeSubUns, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ExchangeAddSatUns, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ExchangeSubSatUns, 2, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x0000ffff, 0x00000000, 0x00007fff),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x0000ffff, 0x00000000, 0x00007fff),
            (Intrinsic::ParallelAdd, 1, 0x00ff00ff, 0xff00ff00, 0xffffffff),
            (Intrinsic::ParallelAddSat, 1, 0x00ff00ff, 0xff00ff00, 0xffffffff),
            (Intrinsic::ParallelAddHalf, 1, 0x00ff00ff, 0xff00ff00, 0xffffffff),
            (Intrinsic::ParallelAddUns, 1, 0x00ff00ff, 0xff00ff00, 0xffffffff),
            (Intrinsic::ParallelAddSatUns, 1, 0x00ff00ff, 0xff00ff00, 0xffffffff),
            (Intrinsic::ParallelAddHalfUns, 1, 0x00ff00ff, 0xff00ff00, 0x7f7f7f7f),
            (Intrinsic::ParallelAdd, 2, 0x00ff00ff, 0xff00ff00, 0xffffffff),
            (Intrinsic::ParallelAddSat, 2, 0x00ff00ff, 0xff00ff00, 0xffffffff),
            (Intrinsic::ParallelAddHalf, 2, 0x00ff00ff, 0xff00ff00, 0xffffffff),
            (Intrinsic::ParallelAddUns, 2, 0x00ff00ff, 0xff00ff00, 0xffffffff),
            (Intrinsic::ParallelAddSatUns, 2, 0x00ff00ff, 0xff00ff00, 0xffffffff),
            (Intrinsic::ParallelAddHalfUns, 2, 0x00ff00ff, 0xff00ff00, 0x7fff7fff),
            (Intrinsic::ParallelSub, 1, 0x00ff00ff, 0xff00ff00, 0x01ff01ff),
            (Intrinsic::ParallelSubSat, 1, 0x00ff00ff, 0xff00ff00, 0x01ff01ff),
            (Intrinsic::ParallelSubHalf, 1, 0x00ff00ff, 0xff00ff00, 0x00ff00ff),
            (Intrinsic::ParallelSubUns, 1, 0x00ff00ff, 0xff00ff00, 0x01ff01ff),
            (Intrinsic::ParallelSubSatUns, 1, 0x00ff00ff, 0xff00ff00, 0x00ff00ff),
            (Intrinsic::ParallelSubHalfUns, 1, 0x00ff00ff, 0xff00ff00, 0x807f807f),
            (Intrinsic::ParallelSub, 2, 0x00ff00ff, 0xff00ff00, 0x01ff01ff),
            (Intrinsic::ParallelSubSat, 2, 0x00ff00ff, 0xff00ff00, 0x01ff01ff),
            (Intrinsic::ParallelSubHalf, 2, 0x00ff00ff, 0xff00ff00, 0x00ff00ff),
            (Intrinsic::ParallelSubUns, 2, 0x00ff00ff, 0xff00ff00, 0x01ff01ff),
            (Intrinsic::ParallelSubSatUns, 2, 0x00ff00ff, 0xff00ff00, 0x00000000),
            (Intrinsic::ParallelSubHalfUns, 2, 0x00ff00ff, 0xff00ff00, 0x80ff80ff),
            (Intrinsic::ExchangeAdd, 2, 0x00ff00ff, 0xff00ff00, 0xffff01ff),
            (Intrinsic::ExchangeSub, 2, 0x00ff00ff, 0xff00ff00, 0x01ffffff),
            (Intrinsic::ExchangeAddSat, 2, 0x00ff00ff, 0xff00ff00, 0xffff01ff),
            (Intrinsic::ExchangeSubSat, 2, 0x00ff00ff, 0xff00ff00, 0x01ffffff),
            (Intrinsic::ExchangeAddHalf, 2, 0x00ff00ff, 0xff00ff00, 0xffff00ff),
            (Intrinsic::ExchangeSubHalf, 2, 0x00ff00ff, 0xff00ff00, 0x00ffffff),
            (Intrinsic::ExchangeAddUns, 2, 0x00ff00ff, 0xff00ff00, 0xffff01ff),
            (Intrinsic::ExchangeSubUns, 2, 0x00ff00ff, 0xff00ff00, 0x01ffffff),
            (Intrinsic::ExchangeAddSatUns, 2, 0x00ff00ff, 0xff00ff00, 0xffff0000),
            (Intrinsic::ExchangeSubSatUns, 2, 0x00ff00ff, 0xff00ff00, 0x0000ffff),
            (Intrinsic::ExchangeAddHalfUns, 2, 0x00ff00ff, 0xff00ff00, 0x7fff80ff),
            (Intrinsic::ExchangeSubHalfUns, 2, 0x00ff00ff, 0xff00ff00, 0x80ff7fff),
        ];
        for (op, lane, a, b, want) in cases {
            let got = eval_special(*op, &[*a, *b], *lane);
            assert_eq!(
                got,
                Some(*want),
                "{:?} over {} bits of {:#010x} and {:#010x}",
                op,
                lane * 8,
                a,
                b
            );
        }
    }
    /// The checksums, over every case that was run on the machine.
    ///
    /// The two families differ only in the polynomial they use, and the
    /// width says how much of the second operand is folded in.
    #[test]
    fn every_checksum_matches_the_machine() {
        let cases: &[(Intrinsic, u64, u64, u32, u64)] = &[
            (Intrinsic::Crc32, 0x00000000, 0x00000000, 8, 0x00000000),
            (Intrinsic::Crc32, 0x00000000, 0x00000000, 16, 0x00000000),
            (Intrinsic::Crc32, 0x00000000, 0x00000000, 32, 0x00000000),
            (Intrinsic::Crc32c, 0x00000000, 0x00000000, 8, 0x00000000),
            (Intrinsic::Crc32c, 0x00000000, 0x00000000, 16, 0x00000000),
            (Intrinsic::Crc32c, 0x00000000, 0x00000000, 32, 0x00000000),
            (Intrinsic::Crc32, 0xffffffff, 0x00000000, 8, 0x2dfd1072),
            (Intrinsic::Crc32, 0xffffffff, 0x00000000, 16, 0xbe26ed00),
            (Intrinsic::Crc32, 0xffffffff, 0x00000000, 32, 0xdebb20e3),
            (Intrinsic::Crc32c, 0xffffffff, 0x00000000, 8, 0xad82acae),
            (Intrinsic::Crc32c, 0xffffffff, 0x00000000, 16, 0x0e9e882d),
            (Intrinsic::Crc32c, 0xffffffff, 0x00000000, 32, 0xb798b438),
            (Intrinsic::Crc32, 0x00000000, 0x00000001, 8, 0x77073096),
            (Intrinsic::Crc32, 0x00000000, 0x00000001, 16, 0x191b3141),
            (Intrinsic::Crc32, 0x00000000, 0x00000001, 32, 0xb8bc6765),
            (Intrinsic::Crc32c, 0x00000000, 0x00000001, 8, 0xf26b8303),
            (Intrinsic::Crc32c, 0x00000000, 0x00000001, 16, 0x13a29877),
            (Intrinsic::Crc32c, 0x00000000, 0x00000001, 32, 0xdd45aab8),
            (Intrinsic::Crc32, 0xffffffff, 0x00000001, 8, 0x5afa20e4),
            (Intrinsic::Crc32, 0xffffffff, 0x00000001, 16, 0xa73ddc41),
            (Intrinsic::Crc32, 0xffffffff, 0x00000001, 32, 0x66074786),
            (Intrinsic::Crc32c, 0xffffffff, 0x00000001, 8, 0x5fe92fad),
            (Intrinsic::Crc32c, 0xffffffff, 0x00000001, 16, 0x1d3c105a),
            (Intrinsic::Crc32c, 0xffffffff, 0x00000001, 32, 0x6add1e80),
            (Intrinsic::Crc32, 0x12345678, 0x00000000, 8, 0x5ecccd58),
            (Intrinsic::Crc32, 0x12345678, 0x00000000, 16, 0x65ee150b),
            (Intrinsic::Crc32, 0x12345678, 0x00000000, 32, 0x8e2958ce),
            (Intrinsic::Crc32c, 0x12345678, 0x00000000, 8, 0xfb533894),
            (Intrinsic::Crc32c, 0x12345678, 0x00000000, 16, 0x55c93830),
            (Intrinsic::Crc32c, 0x12345678, 0x00000000, 32, 0xfa745634),
            (Intrinsic::Crc32, 0x12345678, 0x000000ff, 8, 0x73ce22d5),
            (Intrinsic::Crc32, 0x12345678, 0x000000ff, 16, 0xf6cae879),
            (Intrinsic::Crc32, 0x12345678, 0x000000ff, 32, 0x502c5e3f),
            (Intrinsic::Crc32c, 0x12345678, 0x000000ff, 8, 0x562e6bc5),
            (Intrinsic::Crc32c, 0x12345678, 0x000000ff, 16, 0xf62a1cb3),
            (Intrinsic::Crc32c, 0x12345678, 0x000000ff, 32, 0xd277be76),
            (Intrinsic::Crc32, 0x12345678, 0x0000ffff, 8, 0x73ce22d5),
            (Intrinsic::Crc32, 0x12345678, 0x0000ffff, 16, 0xdbc807f4),
            (Intrinsic::Crc32, 0x12345678, 0x0000ffff, 32, 0xeeb46ad2),
            (Intrinsic::Crc32c, 0x12345678, 0x0000ffff, 8, 0x562e6bc5),
            (Intrinsic::Crc32c, 0x12345678, 0x0000ffff, 16, 0x5b574fe2),
            (Intrinsic::Crc32c, 0x12345678, 0x0000ffff, 32, 0x437295de),
            (Intrinsic::Crc32, 0x12345678, 0xffffffff, 8, 0x73ce22d5),
            (Intrinsic::Crc32, 0x12345678, 0xffffffff, 16, 0xdbc807f4),
            (Intrinsic::Crc32, 0x12345678, 0xffffffff, 32, 0x5092782d),
            (Intrinsic::Crc32c, 0x12345678, 0xffffffff, 8, 0x562e6bc5),
            (Intrinsic::Crc32c, 0x12345678, 0xffffffff, 16, 0x5b574fe2),
            (Intrinsic::Crc32c, 0x12345678, 0xffffffff, 32, 0x4dece20c),
            (Intrinsic::Crc32, 0xdeadbeef, 0x00000000, 8, 0x306b5257),
            (Intrinsic::Crc32, 0xdeadbeef, 0x00000000, 16, 0xf53faf05),
            (Intrinsic::Crc32, 0xdeadbeef, 0x00000000, 32, 0x3b1ebf03),
            (Intrinsic::Crc32c, 0xdeadbeef, 0x00000000, 8, 0xbdfd3980),
            (Intrinsic::Crc32c, 0xdeadbeef, 0x00000000, 16, 0x824bc641),
            (Intrinsic::Crc32c, 0xdeadbeef, 0x00000000, 32, 0x09991d14),
            (Intrinsic::Crc32, 0x00000000, 0xffffffff, 8, 0x2d02ef8d),
            (Intrinsic::Crc32, 0x00000000, 0xffffffff, 16, 0xbe2612ff),
            (Intrinsic::Crc32, 0x00000000, 0xffffffff, 32, 0xdebb20e3),
            (Intrinsic::Crc32c, 0x00000000, 0xffffffff, 8, 0xad7d5351),
            (Intrinsic::Crc32c, 0x00000000, 0xffffffff, 16, 0x0e9e77d2),
            (Intrinsic::Crc32c, 0x00000000, 0xffffffff, 32, 0xb798b438),
            (Intrinsic::Crc32, 0x00000001, 0xffffffff, 8, 0x5a05df1b),
            (Intrinsic::Crc32, 0x00000001, 0xffffffff, 16, 0xa73d23be),
            (Intrinsic::Crc32, 0x00000001, 0xffffffff, 32, 0x66074786),
            (Intrinsic::Crc32c, 0x00000001, 0xffffffff, 8, 0x5f16d052),
            (Intrinsic::Crc32c, 0x00000001, 0xffffffff, 16, 0x1d3cefa5),
            (Intrinsic::Crc32c, 0x00000001, 0xffffffff, 32, 0x6add1e80),
            (Intrinsic::Crc32, 0x89abcdef, 0x00000000, 8, 0x303c5424),
            (Intrinsic::Crc32, 0x89abcdef, 0x00000000, 16, 0x3c33d885),
            (Intrinsic::Crc32, 0x89abcdef, 0x00000000, 32, 0xcefc0adb),
            (Intrinsic::Crc32c, 0x89abcdef, 0x00000000, 8, 0xbdaa3ff3),
            (Intrinsic::Crc32c, 0x89abcdef, 0x00000000, 16, 0xe08336be),
            (Intrinsic::Crc32c, 0x89abcdef, 0x00000000, 32, 0x5914342a),
        ];
        for (op, a, b, width, want) in cases {
            let got = eval_special(*op, &[*a, *b, u64::from(*width)], 4);
            assert_eq!(
                got,
                Some(*want),
                "{:?} over {} bits of {:#010x} and {:#010x}",
                op,
                width,
                a,
                b
            );
        }
    }
    /// The extended additions, over every case that was run on the machine.
    ///
    /// The `16` forms work on both halves of the word at once, and they
    /// are a parallel operation: the carry of one half never reaches the
    /// other.
    #[test]
    fn every_extended_addition_matches_the_machine() {
        let cases: &[(Intrinsic, u64, u64, u64)] = &[
            (Intrinsic::ExtendAddSigned, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ExtendAdd, 0x00000000, 0x00000000, 0x00000000),
            (Intrinsic::ExtendAddSigned, 0xffffffff, 0x00000000, 0xffffffff),
            (Intrinsic::ExtendAdd, 0xffffffff, 0x00000000, 0xffffffff),
            (Intrinsic::ExtendAddSigned, 0x00000000, 0xffffffff, 0xffffffff),
            (Intrinsic::ExtendAdd, 0x00000000, 0xffffffff, 0x00ff00ff),
            (Intrinsic::ExtendAddSigned, 0xffffffff, 0xffffffff, 0xfffefffe),
            (Intrinsic::ExtendAdd, 0xffffffff, 0xffffffff, 0x00fe00fe),
            (Intrinsic::ExtendAddSigned, 0x12345678, 0x00000000, 0x12345678),
            (Intrinsic::ExtendAdd, 0x12345678, 0x00000000, 0x12345678),
            (Intrinsic::ExtendAddSigned, 0x12345678, 0x000000ff, 0x12345677),
            (Intrinsic::ExtendAdd, 0x12345678, 0x000000ff, 0x12345777),
            (Intrinsic::ExtendAddSigned, 0x12345678, 0x0000ffff, 0x12345677),
            (Intrinsic::ExtendAdd, 0x12345678, 0x0000ffff, 0x12345777),
            (Intrinsic::ExtendAddSigned, 0x12345678, 0xffffffff, 0x12335677),
            (Intrinsic::ExtendAdd, 0x12345678, 0xffffffff, 0x13335777),
            (Intrinsic::ExtendAddSigned, 0x00010000, 0x00000001, 0x00010001),
            (Intrinsic::ExtendAdd, 0x00010000, 0x00000001, 0x00010001),
            (Intrinsic::ExtendAddSigned, 0x80008000, 0x80008000, 0x80008000),
            (Intrinsic::ExtendAdd, 0x80008000, 0x80008000, 0x80008000),
            (Intrinsic::ExtendAddSigned, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ExtendAdd, 0x0000ffff, 0x00000000, 0x0000ffff),
            (Intrinsic::ExtendAddSigned, 0x00007fff, 0x00000000, 0x00007fff),
            (Intrinsic::ExtendAdd, 0x00007fff, 0x00000000, 0x00007fff),
        ];
        for (op, a, b, want) in cases {
            let got = eval_special(*op, &[*a, *b], 4);
            assert_eq!(
                got,
                Some(*want),
                "{:?} of {:#010x} and {:#010x}",
                op,
                a,
                b
            );
        }
    }
}
