//! Turning a path's branch decisions into concrete inputs.
//!
//! The solver is a bounded model search, not a decision procedure. It builds a
//! small candidate domain per free variable out of the constants the path
//! actually compares against, tries the cartesian product, and checks every
//! candidate against the whole constraint list with concrete arithmetic. That
//! makes two answers sound by construction:
//!
//! * `Satisfied` — the model is printed and was verified, never guessed.
//! * `Unsatisfiable` — reported only when a constraint has no free variables at
//!   all and is false, which no input can fix.
//!
//! Everything else is `Unknown`, with a note saying how much was searched. A
//! solver that returned "unreachable" for a path it merely failed to solve
//! would be worse than useless: it would silently hide reachable code.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use occ_core::Addr;
use occ_decompile::ir::{BoolOp, CastKind, CondOp, Expr, holds_at};

use crate::value::eval_bool;

/// One branch decision, recorded so the path can be replayed.
#[derive(Clone, Debug)]
pub struct Constraint {
    /// The condition expression the branch tested.
    pub expr: Expr,
    /// Which way it was taken.
    pub truth: bool,
    /// Address of the branching instruction.
    pub at: Addr,
}

impl Constraint {
    /// The condition as a predicate that must hold, with `!` folded into the
    /// comparison so the printed form reads the way the source would.
    pub fn predicate(&self) -> Expr {
        match (&self.expr, self.truth) {
            (Expr::Cond {
                op,
                lhs,
                rhs,
                bits,
            }, true) => Expr::Cond {
                op: *op,
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                bits: *bits,
            },
            (Expr::Cond {
                op,
                lhs,
                rhs,
                bits,
            }, false) => Expr::Cond {
                op: op.negated(),
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                bits: *bits,
            },
            (Expr::Bool { op, args }, true) => Expr::Bool {
                op: *op,
                args: args.clone(),
            },
            (Expr::Bool { op, args }, false) => negate_bool(*op, args),
            (Expr::FlagCond {
                cond,
                state,
                bits,
            }, true) => Expr::FlagCond {
                cond: *cond,
                state: state.clone(),
                bits: *bits,
            },
            (Expr::FlagCond {
                cond,
                state,
                bits,
            }, false) => Expr::FlagCond {
                cond: cond.negated(),
                state: state.clone(),
                bits: *bits,
            },
            (e, true) => Expr::Cond {
                op: CondOp::Ne,
                lhs: Box::new(e.clone()),
                rhs: Box::new(Expr::Const(0)),
                bits: 8,
            },
            (e, false) => Expr::Cond {
                op: CondOp::Eq,
                lhs: Box::new(e.clone()),
                rhs: Box::new(Expr::Const(0)),
                bits: 8,
            },
        }
    }
}

/// The negation of a boolean combination.
///
/// The negation of a conjunction is a disjunction of negations, and the
/// negation of a comparison is the opposite comparison.
fn negate_bool(op: BoolOp, args: &[Expr]) -> Expr {
    match op {
        BoolOp::Not => args.first().cloned().unwrap_or(Expr::Const(0)),
        BoolOp::Eq => Expr::Bool {
            op: BoolOp::Ne,
            args: args.to_vec(),
        },
        BoolOp::Ne => Expr::Bool {
            op: BoolOp::Eq,
            args: args.to_vec(),
        },
        BoolOp::And => Expr::Bool {
            op: BoolOp::Or,
            args: args.iter().map(negate_bool_expr).collect(),
        },
        BoolOp::Or => Expr::Bool {
            op: BoolOp::And,
            args: args.iter().map(negate_bool_expr).collect(),
        },
    }
}

/// The negation of an expression that is used as a condition.
fn negate_bool_expr(e: &Expr) -> Expr {
    match e {
        Expr::Bool { op, args } => negate_bool(*op, args),
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
        other => other.clone(),
    }
}

impl Constraint {
    /// Human-readable form, e.g. `rdi != 0`.
    pub fn render(&self) -> String {
        self.predicate().to_string()
    }
}

/// What the search concluded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// A verified assignment exists; see [`Model::inputs`].
    Satisfied,
    /// No assignment can work, because a closed constraint is false.
    Unsatisfiable,
    /// The search space was not exhausted, or a value was not modellable.
    Unknown,
}

/// The result of a search.
#[derive(Clone, Debug)]
pub struct Model {
    /// Assignments, sorted by name. Empty for a closed path.
    pub inputs: Vec<(String, u64)>,
    /// What the search concluded.
    pub verdict: Verdict,
    /// How many candidate assignments were evaluated.
    pub tried: usize,
    /// Why the answer is `Unknown`, when it is.
    pub note: Option<String>,
}

impl Model {
    /// Assignment as a lookup table.
    pub fn map(&self) -> HashMap<String, u64> {
        self.inputs.iter().cloned().collect()
    }
}

/// Search for a model of `conds`.
///
/// `cap` bounds the number of candidate assignments tried; 4096 is enough for
/// the branch conditions a single path accumulates in practice, because each
/// one contributes only its own comparison constants.
pub fn solve(conds: &[Constraint], cap: usize) -> Model {
    if conds.is_empty() {
        return Model {
            inputs: Vec::new(),
            verdict: Verdict::Satisfied,
            tried: 0,
            note: None,
        };
    }
    let preds: Vec<Expr> = conds.iter().map(|c| c.predicate()).collect();

    // A constraint with no free variables is decided here, and decides the
    // path: this is the only case where "no model" can be claimed.
    let empty: HashMap<String, u64> = HashMap::new();
    for p in &preds {
        if !has_vars(p) {
            match eval_bool(p, &empty) {
                Some(true) => {}
                Some(false) => {
                    return Model {
                        inputs: Vec::new(),
                        verdict: Verdict::Unsatisfiable,
                        tried: 0,
                        note: Some(format!("`{p}` is false regardless of the inputs")),
                    };
                }
                None => {
                    return Model {
                        inputs: Vec::new(),
                        verdict: Verdict::Unknown,
                        tried: 0,
                        note: Some(format!("`{p}` reads memory and cannot be decided")),
                    };
                }
            }
        }
    }

    let mut names: BTreeSet<String> = BTreeSet::new();
    for p in &preds {
        collect_vars(p, &mut names);
    }
    if names.is_empty() {
        return Model {
            inputs: Vec::new(),
            verdict: Verdict::Satisfied,
            tried: 0,
            note: None,
        };
    }

    // Candidate domains. Every constant the path compares against is a
    // candidate, plus its neighbours: comparisons are almost always `<`/`<=`
    // at a boundary, and the boundary itself is the constant.
    let mut domain: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    for name in &names {
        domain.insert(name.clone(), vec![0, 1]);
    }
    let mut global: Vec<u64> = Vec::new();
    for p in &preds {
        harvest(p, &mut domain, &mut global);
    }
    for pool in domain.values_mut() {
        for v in &global {
            pool.push(*v);
        }
        // Small values first: a model of `argc >= 2` that reads `2` is easier
        // to check by hand than one that reads `0x7fff_ffff`.
        let mut seen = BTreeSet::new();
        let mut out = vec![0u64, 1];
        for v in pool.iter() {
            if seen.insert(*v) && *v != 0 && *v != 1 {
                out.push(*v);
            }
        }
        out.truncate(16);
        *pool = out;
    }

    let names: Vec<String> = names.into_iter().collect();
    let mut tried = 0usize;
    let mut saw_unmodelled = false;
    let mut cursor = vec![0usize; names.len()];
    loop {
        if tried >= cap {
            return Model {
                inputs: Vec::new(),
                verdict: Verdict::Unknown,
                tried,
                note: Some(format!(
                    "search capped at {cap} of {} candidates for {} input(s)",
                    names
                        .iter()
                        .map(|n| domain[n].len())
                        .fold(1usize, |a, b| a.saturating_mul(b)),
                    names.len()
                )),
            };
        }
        let mut assignment = HashMap::with_capacity(names.len());
        let mut inputs = Vec::with_capacity(names.len());
        for (i, name) in names.iter().enumerate() {
            let v = domain[name][cursor[i]];
            assignment.insert(name.clone(), v);
            inputs.push((name.clone(), v));
        }
        tried += 1;

        let mut ok = true;
        for p in &preds {
            match eval_bool(p, &assignment) {
                Some(true) => {}
                Some(false) => {
                    ok = false;
                    break;
                }
                None => {
                    saw_unmodelled = true;
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            return Model {
                inputs,
                verdict: Verdict::Satisfied,
                tried,
                note: None,
            };
        }

        // Odometer over the product, last variable fastest.
        let mut i = names.len();
        loop {
            if i == 0 {
                return Model {
                    inputs: Vec::new(),
                    verdict: Verdict::Unknown,
                    tried,
                    note: Some(if saw_unmodelled {
                        "no candidate satisfied the path; some conditions read memory".into()
                    } else {
                        "no candidate satisfied the path within the search bounds".into()
                    }),
                };
            }
            i -= 1;
            cursor[i] += 1;
            if cursor[i] < domain[&names[i]].len() {
                break;
            }
            cursor[i] = 0;
        }
    }
}

fn has_vars(e: &Expr) -> bool {
    let mut v = Vec::new();
    e.vars(&mut v);
    !v.is_empty()
}

fn collect_vars(e: &Expr, out: &mut BTreeSet<String>) {
    let mut v = Vec::new();
    e.vars(&mut v);
    for name in v {
        out.insert(name);
    }
}

/// Pull comparison constants out of a predicate into the domains they belong
/// to, and everything else into the shared pool.
fn harvest(e: &Expr, domain: &mut BTreeMap<String, Vec<u64>>, global: &mut Vec<u64>) {
    match e {
        Expr::Cond {
            op,
            lhs,
            rhs,
            ..
        } => match (&**lhs, &**rhs) {
            (Expr::Var(name), Expr::Const(c)) => {
                add_candidates(domain, name, *op, *c, 8);
                global.push(*c);
            }
            (Expr::Const(c), Expr::Var(name)) => {
                add_candidates(domain, name, op.swapped(), *c, 8);
                global.push(*c);
            }
            // A variable the comparison reads through a width change compares
            // at that width, not at the width of the register it lives in.
            (view, Expr::Const(c)) => {
                if let Some(view) = narrow_view(view) {
                    add_candidates(domain, &view.name, *op, *c, view.width);
                    global.push(*c);
                } else {
                    harvest(lhs, domain, global);
                    harvest(rhs, domain, global);
                }
            }
            (Expr::Const(c), view) => {
                if let Some(view) = narrow_view(view) {
                    add_candidates(domain, &view.name, op.swapped(), *c, view.width);
                    global.push(*c);
                } else {
                    harvest(lhs, domain, global);
                    harvest(rhs, domain, global);
                }
            }
            _ => {
                harvest(lhs, domain, global);
                harvest(rhs, domain, global);
            }
        },
        Expr::Bin { lhs, rhs, .. } => {
            harvest(lhs, domain, global);
            harvest(rhs, domain, global);
        }
        Expr::Un { arg, .. } | Expr::Cast { arg, .. } => harvest(arg, domain, global),
        Expr::Select { cond, yes, no } => {
            harvest(cond, domain, global);
            harvest(yes, domain, global);
            harvest(no, domain, global);
        }
        Expr::Bool { args, .. } => {
            for a in args {
                harvest(a, domain, global);
            }
        }
        Expr::FlagCond { state, .. } => {
            for value in [
                state.cf.as_ref(),
                state.zf.as_ref(),
                state.sf.as_ref(),
                state.of.as_ref(),
                state.pf.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                harvest(value, domain, global);
            }
        }
        Expr::Const(c) => global.push(*c),
        _ => {}
    }
}

/// A variable the comparison reads through a width change.
///
/// A 32-bit comparison on x86 reads the low half of a 64-bit register, and
/// the IR says so with a cast. The solver has to compare at that width: the
/// value that makes `x < 0` true in 32-bit terms is a different number from
/// the one that makes it true in 64-bit terms, and the machine means the
/// narrow one.
struct View {
    /// The variable's name.
    name: String,
    /// The width the comparison reads, in bytes.
    width: u8,
}

/// Recognise a variable that a comparison reads through a width change.
fn narrow_view(e: &Expr) -> Option<View> {
    let Expr::Cast { kind, arg, from, to } = e else {
        return None;
    };
    let Expr::Var(name) = &**arg else {
        return None;
    };
    let width = match kind {
        // A sign or zero extension widens a narrow value.
        CastKind::Sext | CastKind::Zext => *from,
        // A truncation narrows a wide one.
        CastKind::Trunc => *to,
    };
    if width == 0 || width >= 8 {
        return None;
    }
    Some(View {
        name: name.clone(),
        width,
    })
}

/// Add the values worth trying for a variable compared against `c`.
///
/// `width` is the width the comparison reads; eight means the whole register.
fn add_candidates(
    domain: &mut BTreeMap<String, Vec<u64>>,
    name: &str,
    op: CondOp,
    c: u64,
    width: u8,
) {
    let Some(pool) = domain.get_mut(name) else {
        return;
    };
    for v in boundary_values(op, c, width) {
        pool.push(v);
    }
}

/// The values worth trying for a variable compared against `c`.
///
/// The comparison's boundary is where the interesting answers live: a
/// comparison is almost always at an edge, and the edge is the constant. The
/// values next to it are the ones just inside and just outside.
///
/// A narrow width wraps. `x < 0` in 32-bit terms is answered by
/// `0xffff_ffff`, not by the 64-bit value one below it: the machine compared
/// 32 bits, and only 32 bits of the answer were read.
fn boundary_values(op: CondOp, c: u64, width: u8) -> Vec<u64> {
    let mask = width_mask(width);
    let cn = c & mask;
    // The two sides of the boundary: one that satisfies the comparison and
    // one that does not.
    let deltas: [i64; 2] = match op {
        CondOp::Eq | CondOp::Ne => [0, 1],
        CondOp::Slt | CondOp::Sle | CondOp::Ult | CondOp::Ule => [-1, 0],
        CondOp::Sgt | CondOp::Sge | CondOp::Ugt | CondOp::Uge => [1, 0],
    };
    let mut out = vec![cn];
    for d in deltas {
        let v = cn.wrapping_add(d as u64) & mask;
        if !out.contains(&v) {
            out.push(v);
        }
    }
    out
}

/// The bits a width in bytes covers.
fn width_mask(width: u8) -> u64 {
    match width {
        0 | 8.. => u64::MAX,
        w => (1u64 << (u32::from(w) * 8)) - 1,
    }
}

/// Check one assignment against constraints, for callers that already have a
/// candidate (used by the tests and by `sym.explore` when the caller supplies
/// inputs).
pub fn check(conds: &[Constraint], assignment: &HashMap<String, u64>) -> Option<bool> {
    let mut all = true;
    for c in conds {
        match eval_bool(&c.predicate(), assignment) {
            Some(true) => {}
            Some(false) => return Some(false),
            None => all = false,
        }
    }
    if all { Some(true) } else { None }
}

/// Evaluate a predicate under a concrete assignment, for reporting.
pub fn holds_under(e: &Expr, assignment: &HashMap<String, u64>) -> bool {
    match (e, eval_bool(e, assignment)) {
        (_, Some(v)) => v,
        (
            Expr::Cond {
                op,
                lhs,
                rhs,
                bits,
            },
            None,
        ) => {
            let a = crate::value::eval_word(lhs, assignment).unwrap_or(0);
            let b = crate::value::eval_word(rhs, assignment).unwrap_or(0);
            holds_at(*op, a, b, *bits)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(e: Expr, truth: bool) -> Constraint {
        Constraint {
            expr: e,
            truth,
            at: Addr::NULL,
        }
    }

    fn cmp(op: CondOp, lhs: Expr, rhs: Expr) -> Expr {
        Expr::Cond {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            bits: 8,
        }
    }

    #[test]
    fn an_equality_yields_the_compared_constant() {
        let conds = [c(cmp(CondOp::Eq, Expr::var("rdi"), Expr::Const(0x2a)), true)];
        let m = solve(&conds, 4096);
        assert_eq!(m.verdict, Verdict::Satisfied);
        assert_eq!(m.inputs, vec![("rdi".to_string(), 0x2a)]);
    }

    #[test]
    fn a_strict_bound_lands_just_past_the_edge() {
        // `rdi > 5` must not be answered with 5.
        let conds = [c(cmp(CondOp::Sgt, Expr::var("rdi"), Expr::Const(5)), true)];
        let m = solve(&conds, 4096);
        assert_eq!(m.verdict, Verdict::Satisfied);
        assert_eq!(m.inputs[0].1, 6);
    }

    #[test]
    fn contradictory_bounds_are_not_called_unsatisfiable() {
        let conds = [
            c(cmp(CondOp::Sgt, Expr::var("rdi"), Expr::Const(10)), true),
            c(cmp(CondOp::Ult, Expr::var("rdi"), Expr::Const(3)), true),
        ];
        let m = solve(&conds, 4096);
        // Contradictory, but proving that needs arithmetic this does not do.
        assert_eq!(m.verdict, Verdict::Unknown);
    }

    #[test]
    fn a_closed_false_constraint_is_proven() {
        let conds = [c(cmp(CondOp::Eq, Expr::Const(1), Expr::Const(2)), true)];
        let m = solve(&conds, 4096);
        assert_eq!(m.verdict, Verdict::Unsatisfiable);
    }

    #[test]
    fn two_inputs_are_solved_together() {
        let conds = [
            c(cmp(CondOp::Uge, Expr::var("rdi"), Expr::Const(3)), true),
            c(cmp(CondOp::Eq, Expr::var("rsi"), Expr::Const(7)), true),
        ];
        let m = solve(&conds, 4096);
        assert_eq!(m.verdict, Verdict::Satisfied);
        assert_eq!(m.inputs, vec![("rdi".into(), 3), ("rsi".into(), 7)]);
    }

    /// A 32-bit view of a 64-bit register, the way x86's `js` sees one.
    fn low_half(name: &str) -> Expr {
        Expr::Cast {
            kind: CastKind::Sext,
            arg: Box::new(Expr::var(name)),
            from: 4,
            to: 8,
        }
    }

    #[test]
    fn a_narrow_comparison_is_answered_at_its_own_width() {
        // `js` after `test %edi,%edi` asks whether the low half of the
        // register is negative. The answer is a 32-bit pattern, and a solver
        // that looked at 64 bits would offer `0xffff_ffff_ffff_ffff` — a
        // value the machine never compared.
        let conds = [c(cmp(CondOp::Slt, low_half("rdi"), Expr::Const(0)), true)];
        let m = solve(&conds, 4096);
        assert_eq!(m.verdict, Verdict::Satisfied);
        assert_eq!(m.inputs, vec![("rdi".to_string(), 0xffff_ffff)]);
    }

    #[test]
    fn a_narrow_comparison_finds_the_value_just_past_the_edge() {
        // `jle` after `cmp $9,%edi` with the branch not taken: the input has
        // to be 10, and the solver has to look past 9 to find it.
        let conds = [c(cmp(CondOp::Sle, low_half("rdi"), Expr::Const(9)), false)];
        let m = solve(&conds, 4096);
        assert_eq!(m.verdict, Verdict::Satisfied);
        assert_eq!(m.inputs, vec![("rdi".to_string(), 10)]);
    }

    #[test]
    fn a_truncated_comparison_keeps_its_unsigned_order() {
        // `je` after `test %edi,%edi` compares the low half against zero, and
        // the low half is an unsigned number here.
        let conds = [c(
            cmp(
                CondOp::Eq,
                Expr::Cast {
                    kind: CastKind::Trunc,
                    arg: Box::new(Expr::var("rdi")),
                    from: 8,
                    to: 4,
                },
                Expr::Const(0),
            ),
            true,
        )];
        let m = solve(&conds, 4096);
        assert_eq!(m.verdict, Verdict::Satisfied);
        assert_eq!(m.inputs, vec![("rdi".to_string(), 0)]);
    }
}
