//! The abstract values, and the arithmetic over them.
//!
//! Two questions decide the shape of this domain: *can this value name a jump
//! target*, and *can this value name a table*. A jump target has to be a
//! concrete address, so the only useful answer is a finite set of them. A table
//! has to be a base plus a stride, so the useful answer is an arithmetic
//! progression. Everything else is [`Num::Top`], and the analysis says so
//! rather than inventing a number.
//!
//! The sets are capped. A set larger than [`MAX_EXACT`] is not listed but
//! described as the progression through its members, which is an
//! over-approximation: it loses the ability to name a target, and it never
//! invents one.

use std::collections::BTreeSet;

/// Most values a finite set holds before it is described as a progression.
pub const MAX_EXACT: usize = 16;

/// A set of machine words.
///
/// Addresses are integers here, negative ones included: an offset from a table
/// base is often negative, and the sign is what makes `movsxd` on a table entry
/// meaningful.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Num {
    /// A finite set, listed in full.
    Exact(BTreeSet<i64>),
    /// `lo + k * stride` for `k` in `0..=steps`, `stride > 0`.
    Stride {
        /// Smallest value.
        lo: i64,
        /// Distance between neighbouring values.
        stride: u64,
        /// How many steps past `lo` the set reaches.
        steps: u64,
    },
    /// A value that is not known, but is related to one unknown.
    ///
    /// `scale * x + bias`, where `x` is the value of the symbol named
    /// here. The symbol is the variable the value came from, and
    /// keeping the name is what tells two unknowns apart: the
    /// difference between `index - 100` and `other - 100` is the
    /// difference between a case label and a wrong one.
    ///
    /// The bias is what a compiler adds before it looks anything up:
    /// `case 100..106` becomes `index - 100` and then a table read, and
    /// the 100 is what says which case the first entry is. Losing it
    /// loses the labels, and the targets alone do not say.
    Sym {
        /// The symbol's name.
        name: String,
        /// How much of the symbol the value carries.
        scale: i64,
        /// What the compiler added to the symbol.
        bias: i64,
    },
    /// Nothing known.
    Top,
}

impl Num {
    /// A single value.
    pub fn one(value: i64) -> Num {
        let mut set = BTreeSet::new();
        set.insert(value);
        Num::Exact(set)
    }

    /// Build from an arbitrary collection, keeping it exact when it is small
    /// enough and describing it as a progression otherwise.
    pub fn from_set(values: BTreeSet<i64>) -> Num {
        if values.is_empty() {
            return Num::Top;
        }
        if values.len() <= MAX_EXACT {
            return Num::Exact(values);
        }
        let lo = *values.iter().next().unwrap_or(&0);
        let hi = *values.iter().next_back().unwrap_or(&0);
        let mut stride = 0u64;
        let mut previous: Option<i64> = None;
        for v in &values {
            if let Some(p) = previous {
                let d = v.wrapping_sub(p).unsigned_abs();
                stride = gcd(stride, d);
            }
            previous = Some(*v);
        }
        if stride == 0 {
            return Num::Exact(values);
        }
        let steps = (hi.wrapping_sub(lo).unsigned_abs()) / stride;
        Num::Stride { lo, stride, steps }
    }

    /// The single value this set holds, if it holds exactly one.
    pub fn single(&self) -> Option<i64> {
        match self {
            Num::Exact(set) if set.len() == 1 => set.iter().next().copied(),
            Num::Stride { lo, steps: 0, .. } => Some(*lo),
            // A symbol is unknown by construction: there is no value to
            // name, and pretending there is would be a guess.
            _ => None,
        }
    }

    /// Every value, when the set is small enough to list.
    pub fn listed(&self) -> Option<Vec<i64>> {
        match self {
            Num::Exact(set) => Some(set.iter().copied().collect()),
            Num::Stride { .. } | Num::Sym { .. } | Num::Top => None,
        }
    }

    /// True when nothing is known.
    pub fn is_top(&self) -> bool {
        matches!(self, Num::Top)
    }

    /// The symbol this value hangs off, when it hangs off one.
    pub fn symbol(&self) -> Option<(&str, i64, i64)> {
        match self {
            Num::Sym {
                name,
                scale,
                bias,
            } => Some((name.as_str(), *scale, *bias)),
            _ => None,
        }
    }

    /// Least upper bound.
    pub fn join(&self, other: &Num) -> Num {
        match (self, other) {
            (Num::Top, _) | (_, Num::Top) => Num::Top,
            // Two symbols are one value only when they are the same symbol
            // with the same offset. The same symbol with a different offset
            // is two values, and two different symbols are two values.
            (
                Num::Sym {
                    name: a,
                    scale: as_,
                    bias: ab,
                },
                Num::Sym {
                    name: b,
                    scale: bs,
                    bias: bb,
                },
            ) if a == b && as_ == bs && ab == bb => Num::Sym {
                name: a.clone(),
                scale: *as_,
                bias: *ab,
            },
            (Num::Sym { .. }, _) | (_, Num::Sym { .. }) => Num::Top,
            (Num::Exact(a), Num::Exact(b)) => {
                let mut union = a.clone();
                union.extend(b.iter().copied());
                Num::from_set(union)
            }
            (Num::Exact(a), s @ Num::Stride { .. }) | (s @ Num::Stride { .. }, Num::Exact(a)) => {
                let Num::Stride { lo, stride, steps } = s else {
                    return Num::Top;
                };
                let mut lo = *lo;
                let mut hi = lo + (stride * steps) as i64;
                for v in a {
                    lo = lo.min(*v);
                    hi = hi.max(*v);
                }
                let mut stride = *stride;
                for v in a {
                    stride = gcd(stride, v.wrapping_sub(lo).unsigned_abs());
                }
                if stride == 0 {
                    return Num::one(lo);
                }
                Num::Stride {
                    lo,
                    stride,
                    steps: (hi.wrapping_sub(lo).unsigned_abs()) / stride,
                }
            }
            (
                Num::Stride {
                    lo: alo,
                    stride: astride,
                    steps: asteps,
                },
                Num::Stride {
                    lo: blo,
                    stride: bstride,
                    steps: bsteps,
                },
            ) => {
                if astride == bstride && alo.wrapping_add((*astride * asteps) as i64) >= *blo {
                    // The two runs sit on the same progression.
                    let lo = (*alo).min(*blo);
                    let hi = alo
                        .wrapping_add((*astride * asteps) as i64)
                        .max(blo.wrapping_add((*bstride * bsteps) as i64));
                    return Num::Stride {
                        lo,
                        stride: *astride,
                        steps: (hi.wrapping_sub(lo).unsigned_abs()) / *astride,
                    };
                }
                // Different strides: the progression through both runs, one
                // step of widening, still an over-approximation.
                let stride = gcd(*astride, *bstride);
                let lo = (*alo).min(*blo);
                let hi = alo
                    .wrapping_add((*astride * asteps) as i64)
                    .max(blo.wrapping_add((*bstride * bsteps) as i64));
                Num::Stride {
                    lo,
                    stride,
                    steps: (hi.wrapping_sub(lo).unsigned_abs()) / stride.max(1),
                }
            }
        }
    }

    /// Least upper bound, losing precision on purpose so a loop terminates.
    ///
    /// The domain has infinite ascending chains -- a counter that gains one
    /// every turn walks through every integer -- so a join at a back edge is
    /// replaced by a jump: one step to a progression, the next to nothing.
    pub fn widen(&self, other: &Num) -> Num {
        if self == other {
            return self.clone();
        }
        match self {
            Num::Top => Num::Top,
            Num::Exact(_) => {
                let joined = self.join(other);
                match joined {
                    // A progression that still describes the values is worth
                    // keeping; a set that is merely larger is not.
                    Num::Stride { .. } => joined,
                    Num::Exact(_) => match (self, other) {
                        (Num::Exact(a), Num::Exact(b)) => {
                            let mut union = a.clone();
                            union.extend(b.iter().copied());
                            let lo = *union.iter().next().unwrap_or(&0);
                            let hi = *union.iter().next_back().unwrap_or(&0);
                            Num::Stride {
                                lo,
                                stride: 1,
                                steps: hi.wrapping_sub(lo).unsigned_abs(),
                            }
                        }
                        _ => Num::Top,
                    },
                    Num::Top | Num::Sym { .. } => Num::Top,
                }
            }
            Num::Stride { .. } => Num::Top,
            // A symbol has no progression to widen into: the next round may
            // see another offset, and then the value is not a symbol at all.
            Num::Sym { .. } => Num::Top,
        }
    }

    /// Apply a function to every member, keeping the shape when it survives.
    pub fn map(&self, f: impl Fn(i64) -> i64) -> Num {
        match self {
            Num::Exact(set) => {
                let mapped: BTreeSet<i64> = set.iter().map(|v| f(*v)).collect();
                Num::from_set(mapped)
            }
            Num::Stride { lo, stride, steps } => {
                let first = f(*lo);
                let second = f(lo.wrapping_add(*stride as i64));
                let step = second.wrapping_sub(first).unsigned_abs();
                if step == 0 {
                    return Num::one(first);
                }
                Num::Stride {
                    lo: first.min(second),
                    stride: step,
                    steps: *steps,
                }
            }
            // A symbol has no members to map, but the offset it carries is
            // a number like any other and the functions this pass applies
            // to it are the ones a compiler applies to a value.
            Num::Sym { name, scale, bias } => Num::Sym {
                name: name.clone(),
                scale: *scale,
                bias: f(*bias),
            },
            Num::Top => Num::Top,
        }
    }

    /// Pairwise combination, exact when both sides are listed.
    pub fn zip(&self, other: &Num, f: impl Fn(i64, i64) -> i64) -> Num {
        let (Num::Exact(a), Num::Exact(b)) = (self, other) else {
            return Num::Top;
        };
        if a.len() * b.len() > MAX_EXACT * 4 {
            return Num::Top;
        }
        let mut out = BTreeSet::new();
        for x in a {
            for y in b {
                out.insert(f(*x, *y));
            }
        }
        Num::from_set(out)
    }

    /// Sum.
    pub fn add(&self, other: &Num) -> Num {
        if let Some(sum) = self.symbol_add(other, 1) {
            return sum;
        }
        if let Some(b) = other.single() {
            // A constant added to a progression moves the progression.
            return self.map(|a| a.wrapping_add(b));
        }
        self.zip(other, |a, b| a.wrapping_add(b))
    }

    /// Difference.
    pub fn sub(&self, other: &Num) -> Num {
        if let Some(diff) = self.symbol_add(other, -1) {
            return diff;
        }
        if let Some(b) = other.single() {
            return self.map(|a| a.wrapping_sub(b));
        }
        self.zip(other, |a, b| a.wrapping_sub(b))
    }

    /// The sum or difference of a symbol and a number or the same symbol.
    ///
    /// `sign` is `1` for a sum and `-1` for a difference. The result is
    /// `None` when the two sides have nothing in common, and that is the
    /// answer: `index - 100` is a symbol with an offset, `index - other`
    /// is not a symbol at all.
    fn symbol_add(&self, other: &Num, sign: i64) -> Option<Num> {
        match (self, other) {
            (
                Num::Sym {
                    name,
                    scale,
                    bias,
                },
                Num::Exact(set),
            ) if set.len() == 1 => {
                let k = *set.iter().next()?;
                Some(Num::Sym {
                    name: name.clone(),
                    scale: *scale,
                    bias: bias.wrapping_add(sign.wrapping_mul(k)),
                })
            }
            (
                Num::Exact(set),
                Num::Sym {
                    name,
                    scale,
                    bias,
                },
            ) if set.len() == 1 && sign > 0 => {
                let k = *set.iter().next()?;
                Some(Num::Sym {
                    name: name.clone(),
                    scale: *scale,
                    bias: bias.wrapping_add(k),
                })
            }
            (
                Num::Sym {
                    name: a,
                    scale: as_,
                    bias: ab,
                },
                Num::Sym {
                    name: b,
                    scale: bs,
                    bias: bb,
                },
            ) if a == b && as_ == bs && sign > 0 => Some(Num::Sym {
                name: a.clone(),
                scale: *as_,
                bias: ab.wrapping_add(*bb),
            }),
            _ => None,
        }
    }

    /// Product.
    pub fn mul(&self, other: &Num) -> Num {
        if let (Some(a), Some(b)) = (self.single(), other.single()) {
            return Num::one(a.wrapping_mul(b));
        }
        // A progression times a constant is the progression with a wider step.
        if let (Num::Stride { lo, stride, steps }, Some(k)) = (self, other.single()) {
            let step = stride.wrapping_mul(k.unsigned_abs());
            if step == 0 {
                return Num::one(0);
            }
            let first = lo.wrapping_mul(k);
            return Num::Stride {
                lo: first.min(first.wrapping_add((step * steps) as i64)),
                stride: step,
                steps: *steps,
            };
        }
        if let (Some(k), Num::Stride { .. }) = (self.single(), other) {
            return other.mul(&Num::one(k));
        }
        // A symbol times a constant is the same symbol, scaled: the offset
        // scales with it, and the symbol stays the one it was.
        if let (Num::Sym { name, scale, bias }, Some(k)) = (self, other.single()) {
            return Num::Sym {
                name: name.clone(),
                scale: scale.wrapping_mul(k),
                bias: bias.wrapping_mul(k),
            };
        }
        if let (Some(k), Num::Sym { .. }) = (self.single(), other) {
            return other.mul(&Num::one(k));
        }
        self.zip(other, |a, b| a.wrapping_mul(b))
    }

    /// Shift left by a constant.
    pub fn shl(&self, k: u32) -> Num {
        if k >= 63 {
            return Num::Top;
        }
        self.map(|v| v.wrapping_shl(k))
    }

    /// Bitwise and with a mask. Only a constant mask keeps any precision.
    pub fn and(&self, other: &Num) -> Num {
        if let (Some(mask), Num::Stride { lo, stride, steps }) = (other.single(), self) {
            // A progression clipped to a mask keeps its stride only when the
            // mask is wider than the run; otherwise the bits fold.
            let top = lo.wrapping_add((*stride * *steps) as i64);
            if mask >= 0 && top >= 0 && (top as u64) <= (mask as u64) {
                return self.clone();
            }
        }
        self.zip(other, |a, b| a & b)
    }

    /// Bitwise or.
    pub fn or(&self, other: &Num) -> Num {
        self.zip(other, |a, b| a | b)
    }

    /// Bitwise exclusive or.
    pub fn xor(&self, other: &Num) -> Num {
        self.zip(other, |a, b| a ^ b)
    }

    /// Truncate to `bits`, signed or not.
    pub fn truncate(&self, bits: u32, signed: bool) -> Num {
        if bits == 0 || bits > 64 {
            return Num::Top;
        }
        let keep = |v: i64| -> i64 {
            if bits == 64 {
                return v;
            }
            let masked = (v as u64) & ((1u64 << bits) - 1);
            if signed && masked >> (bits - 1) != 0 {
                (masked | !((1u64 << bits) - 1)) as i64
            } else {
                masked as i64
            }
        };
        self.map(keep)
    }
}

/// Greatest common divisor, with zero as the neutral element.
pub fn gcd(a: u64, b: u64) -> u64 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// What a value is, as far as the analysis can tell.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    /// A number that is not known to be an address.
    Num(Num),
    /// An address inside a named object, plus an offset.
    Global {
        /// Object name, for the report.
        name: String,
        /// Address the object starts at.
        at: i64,
        /// Offset from that start.
        off: Num,
    },
    /// A run of words read out of the image, with the shape it was read with.
    Table {
        /// Address the first entry sits at.
        base: i64,
        /// Distance between entries in bytes.
        stride: i64,
        /// Constant the index carried before it was scaled.
        ///
        /// `case 100..106` compiles to `sub index, 100` and then an indexed
        /// load, so the position in the table is the case value minus 100 and
        /// the constant is the link back. It is not the table's address — a
        /// table at a lower address would be a different table — so it stays a
        /// separate field rather than being folded into `base`.
        bias: i64,
        /// Entry values, as read.
        entries: BTreeSet<i64>,
    },
    /// A frame slot.
    Stack {
        /// Offset from the frame base.
        off: i64,
    },
    /// A heap object, named by the call that produced it.
    Heap {
        /// Address of the allocating call.
        site: i64,
    },
    /// Nothing known.
    Top,
}

impl Value {
    /// A plain number.
    pub fn num(value: i64) -> Value {
        Value::Num(Num::one(value))
    }

    /// True when nothing is known.
    pub fn is_top(&self) -> bool {
        matches!(self, Value::Top)
    }

    /// The single address this value names, if it names exactly one.
    pub fn single(&self) -> Option<i64> {
        match self {
            Value::Num(n) => n.single(),
            Value::Global { at, off, .. } => off.single().map(|o| at.wrapping_add(o)),
            Value::Stack { off } => Some(*off),
            Value::Heap { site } => Some(*site),
            Value::Table { .. } | Value::Top => None,
        }
    }

    /// Least upper bound.
    pub fn join(&self, other: &Value) -> Value {
        match (self, other) {
            (Value::Top, _) | (_, Value::Top) => Value::Top,
            (Value::Num(a), Value::Num(b)) => Value::Num(a.join(b)),
            (
                Value::Global {
                    name: an,
                    at: aa,
                    off: ao,
                },
                Value::Global {
                    name: bn,
                    at: ba,
                    off: bo,
                },
            ) if an == bn && aa == ba => Value::Global {
                name: an.clone(),
                at: *aa,
                off: ao.join(bo),
            },
            (Value::Stack { off: a }, Value::Stack { off: b }) if a == b => Value::Stack { off: *a },
            (Value::Heap { site: a }, Value::Heap { site: b }) if a == b => Value::Heap { site: *a },
            (
                Value::Table {
                    base: ab,
                    stride: as_,
                    bias: abias,
                    entries: ae,
                },
                Value::Table {
                    base: bb,
                    stride: bs,
                    bias: bbias,
                    entries: be,
                },
            ) if ab == bb && as_ == bs && abias == bbias => {
                let mut entries = ae.clone();
                entries.extend(be.iter().copied());
                Value::Table {
                    base: *ab,
                    stride: *as_,
                    bias: *abias,
                    entries,
                }
            }
            _ => Value::Top,
        }
    }

    /// Least upper bound that terminates on a loop.
    pub fn widen(&self, other: &Value) -> Value {
        if self == other {
            return self.clone();
        }
        match (self, other) {
            (Value::Num(a), Value::Num(b)) => Value::Num(a.widen(b)),
            _ => Value::Top,
        }
    }

    /// Add a number to every address this value can name.
    pub fn offset(&self, delta: &Num) -> Value {
        match self {
            Value::Num(n) => Value::Num(n.add(delta)),
            Value::Global { name, at, off } => Value::Global {
                name: name.clone(),
                at: *at,
                off: off.add(delta),
            },
            Value::Table {
                base,
                stride,
                bias,
                entries,
            } => {
                let Some(k) = delta.single() else {
                    return Value::Top;
                };
                // The table did not move: what the compiler added is the
                // offset stored in each entry. This is the last step of a
                // relative jump table, where the table holds positions and
                // the base turns them into addresses.
                let moved: BTreeSet<i64> = entries.iter().map(|e| e.wrapping_add(k)).collect();
                Value::Table {
                    base: *base,
                    stride: *stride,
                    bias: *bias,
                    entries: moved,
                }
            }
            Value::Stack { off } => match delta.single() {
                Some(k) => Value::Stack {
                    off: off.wrapping_add(k),
                },
                None => Value::Top,
            },
            _ => Value::Top,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(values: &[i64]) -> Num {
        Num::from_set(values.iter().copied().collect())
    }

    #[test]
    fn a_small_set_is_listed() {
        assert_eq!(set(&[1, 5]).listed(), Some(vec![1, 5]));
    }

    #[test]
    fn a_large_progression_keeps_its_stride() {
        let values: Vec<i64> = (0..40).map(|i| i * 4).collect();
        assert_eq!(
            set(&values),
            Num::Stride {
                lo: 0,
                stride: 4,
                steps: 39
            }
        );
    }

    #[test]
    fn joining_two_progressions_with_one_stride_extends_the_run() {
        let a = Num::Stride {
            lo: 0,
            stride: 8,
            steps: 2,
        };
        let b = Num::Stride {
            lo: 24,
            stride: 8,
            steps: 1,
        };
        assert_eq!(
            a.join(&b),
            Num::Stride {
                lo: 0,
                stride: 8,
                steps: 4
            }
        );
    }

    #[test]
    fn a_small_set_stays_a_set() {
        // A set that fits is listed, and listing it is what lets the
        // analysis name every member of it.
        let run = set(&[0, 4, 8]);
        assert_eq!(run.listed(), Some(vec![0, 4, 8]));
        assert_eq!(run.add(&Num::one(0x1000)).listed(), Some(vec![0x1000, 0x1004, 0x1008]));
    }

    #[test]
    fn widening_a_counter_stops_at_nothing() {
        let one = Num::one(0);
        let next = Num::one(1);
        let widened = one.widen(&next);
        assert!(matches!(widened, Num::Stride { .. }));
        assert_eq!(widened.widen(&Num::one(2)), Num::Top);
    }

    #[test]
    fn adding_a_constant_moves_a_progression() {
        let run = Num::Stride {
            lo: 0,
            stride: 4,
            steps: 2,
        };
        assert_eq!(
            run.add(&Num::one(0x1000)),
            Num::Stride {
                lo: 0x1000,
                stride: 4,
                steps: 2
            }
        );
    }

    #[test]
    fn multiplying_a_progression_widens_the_step() {
        let run = Num::Stride {
            lo: 0,
            stride: 1,
            steps: 3,
        };
        assert_eq!(
            run.mul(&Num::one(8)),
            Num::Stride {
                lo: 0,
                stride: 8,
                steps: 3
            }
        );
    }

    #[test]
    fn a_symbol_keeps_its_offset_when_a_constant_is_added() {
        // `index - 100` is the same unknown, moved: the 100 is what says
        // which case the first entry of the table is.
        let index = Num::Sym {
            name: "rdi".into(),
            scale: 1,
            bias: 0,
        };
        let moved = index.sub(&Num::one(100));
        assert_eq!(
            moved,
            Num::Sym {
                name: "rdi".into(),
                scale: 1,
                bias: -100,
            }
        );
    }

    #[test]
    fn two_symbols_are_not_one_value() {
        // `index - other` is not a symbol with an offset: the two unknowns
        // have nothing in common, and claiming an offset would name a case
        // that is not there.
        let index = Num::Sym {
            name: "rdi".into(),
            scale: 1,
            bias: 0,
        };
        let other = Num::Sym {
            name: "rsi".into(),
            scale: 1,
            bias: 0,
        };
        assert_eq!(index.sub(&other), Num::Top);
        assert_eq!(index.join(&other), Num::Top);
    }

    #[test]
    fn the_same_symbol_with_another_offset_is_not_one_value() {
        let index = Num::Sym {
            name: "rdi".into(),
            scale: 1,
            bias: 0,
        };
        let moved = Num::Sym {
            name: "rdi".into(),
            scale: 1,
            bias: -100,
        };
        assert_eq!(index.join(&moved), Num::Top);
        assert_eq!(index.join(&index.clone()), index);
    }

    #[test]
    fn truncation_keeps_the_low_bits_and_the_sign() {
        assert_eq!(Num::one(0x1234_5678).truncate(16, false).single(), Some(0x5678));
        assert_eq!(Num::one(0xffff).truncate(8, true).single(), Some(-1));
    }

    #[test]
    fn joining_two_different_globals_gives_up() {
        let a = Value::Global {
            name: "table".into(),
            at: 0x4000,
            off: Num::one(0),
        };
        let b = Value::Global {
            name: "other".into(),
            at: 0x5000,
            off: Num::one(0),
        };
        assert_eq!(a.join(&b), Value::Top);
    }

    #[test]
    fn offsetting_a_table_moves_the_values_it_holds() {
        let mut entries = BTreeSet::new();
        entries.insert(16);
        entries.insert(32);
        let table = Value::Table {
            base: 0x5000,
            stride: 4,
            bias: 0,
            entries,
        };
        let moved = table.offset(&Num::one(0x1000));
        let Value::Table { base, entries, .. } = moved else {
            panic!("still a table");
        };
        assert_eq!(base, 0x5000);
        assert_eq!(entries.iter().copied().collect::<Vec<_>>(), vec![0x1010, 0x1020]);
    }
}
