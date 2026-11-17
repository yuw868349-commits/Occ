//! Value-set analysis: what an indirect branch or a call can reach.
//!
//! The decompiler lifts a function to statements over named values, and the
//! symbolic executor walks those statements. This crate answers a different
//! question about the same statements: *which concrete addresses can a value
//! hold at this point* -- and it answers it for every value in the function at
//! once, not for one path at a time.
//!
//! # Why the question is worth a pass of its own
//!
//! A computed jump is the one construct a linear recovery cannot follow. The
//! existing resolver reads it syntactically: it recognises the three-instruction
//! shape a compiler emits for a `switch` and reads the table it names. That
//! works for the shape it was written for and for nothing else -- a table whose
//! base is computed, a jump through a pointer that was loaded earlier, a call
//! through a function pointer are all rendered as `(*rax)` and lost.
//!
//! This pass works on values instead of shapes. It says: `rax` here is one of
//! seven addresses, all of them inside this function, and here is the table
//! they came from. A target is reported only when it is a basic block that the
//! recovery already found, so the answer cannot invent one.
//!
//! # The domain
//!
//! [`domain::Num`] is a finite set of integers that collapses into an
//! arithmetic progression once it grows past [`domain::MAX_EXACT`]; a
//! progression is what a table index looks like. [`domain::Value`] adds the
//! cases that are not numbers: a global address plus an offset, a frame slot, a
//! heap object, and a run of words already read out of the image.
//!
//! # What it does not do
//!
//! * It does not track memory that the function writes; a table that the
//!   program builds at run time is [`domain::Value::Top`] by the time it is
//!   read.
//! * It does not model calls. A value that comes back from a call is unknown,
//!   except that the return of `malloc`-like functions is a heap object.
//! * It does not resolve a target that falls outside the recovered code. A
//!   table with an entry pointing into the middle of an instruction is reported
//!   as unresolved rather than guessed at.

#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod domain;

use std::collections::{BTreeMap, BTreeSet};

use occ_analyze::{Analysis, View};
use occ_core::{Addr, FlowKind, Function};
use occ_decompile::ir::{BinOp, BlockId, CastKind, Expr, Stmt, Term, UnOp};
use occ_decompile::{Program, build_program};

use domain::{Num, Value};

/// Limits, all of them budgets rather than guesses.
#[derive(Clone, Debug)]
pub struct Options {
    /// Most blocks to lift per function.
    pub max_blocks: usize,
    /// Most rounds of the fixed point per function.
    pub max_rounds: usize,
    /// Longest run of words to read as a table.
    pub max_table_entries: usize,
    /// Deepest expression to evaluate before giving up.
    pub max_depth: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            max_blocks: 20_000,
            max_rounds: 24,
            max_table_entries: 1024,
            max_depth: 64,
        }
    }
}

/// One indirect call and what it can reach.
#[derive(Clone, Debug)]
pub struct CallSite {
    /// Function the call is in.
    pub in_function: Addr,
    /// Address of the call instruction.
    pub at: Addr,
    /// Register the call goes through.
    pub via: String,
    /// Targets that are known basic blocks.
    pub targets: Vec<Addr>,
    /// Why the target set looks the way it does.
    pub evidence: String,
}

/// One computed jump, with its cases when a table was read.
#[derive(Clone, Debug)]
pub struct JumpSite {
    /// Function the jump is in.
    pub in_function: Addr,
    /// Address of the jump instruction.
    pub at: Addr,
    /// Register the jump goes through.
    pub via: String,
    /// Table the targets were read from, when there is one.
    pub table: Option<Table>,
    /// Cases, in table order.
    pub cases: Vec<Case>,
    /// Second-level tables behind a first-level one.
    pub two_level: bool,
    /// Every target is a function entry, so the jump is really a tail call.
    pub tail: bool,
    /// Why the answer is what it is, or what stopped it.
    pub evidence: String,
}

/// The shape of a table that was read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Table {
    /// Address of the first entry.
    pub base: Addr,
    /// Distance between entries in bytes.
    pub stride: i64,
    /// Constant the index carried before it was scaled.
    pub bias: i64,
    /// Entries read.
    pub entries: usize,
}

/// One case of a computed jump.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Case {
    /// Position in the table.
    pub index: usize,
    /// Case value the code compares against, when the index is an affine
    /// function of it.
    pub label: i64,
    /// Block the case reaches.
    pub target: Addr,
}

/// Counters, so a claim about coverage can be checked.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// Functions walked.
    pub functions: usize,
    /// Blocks walked.
    pub blocks: usize,
    /// Indirect call sites seen.
    pub call_sites: usize,
    /// Indirect call sites with at least one target.
    pub calls_resolved: usize,
    /// Indirect jumps seen.
    pub jump_sites: usize,
    /// Indirect jumps with at least one target.
    pub jumps_resolved: usize,
    /// Distinct tables read.
    pub tables: usize,
    /// Jumps that needed a second table.
    pub two_level: usize,
    /// Functions whose fixed point was cut off by the round budget.
    pub unconverged: usize,
    /// Tables found by the syntactic resolver in the same image, for
    /// comparison.
    pub syntactic_tables: usize,
}

impl Stats {
    /// Share of call sites that reached a known block, `0.0..=1.0`.
    pub fn call_rate(&self) -> f64 {
        if self.call_sites == 0 {
            return 0.0;
        }
        self.calls_resolved as f64 / self.call_sites as f64
    }

    /// Share of jump sites that reached a known block, `0.0..=1.0`.
    pub fn jump_rate(&self) -> f64 {
        if self.jump_sites == 0 {
            return 0.0;
        }
        self.jumps_resolved as f64 / self.jump_sites as f64
    }
}

/// Everything the pass found.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// Indirect calls.
    pub calls: Vec<CallSite>,
    /// Computed jumps.
    pub jumps: Vec<JumpSite>,
    /// Counters.
    pub stats: Stats,
}

/// Run the pass over every function the analysis recovered.
pub fn analyze(view: &View, analysis: &Analysis, opts: &Options) -> Report {
    let mut walker = Walker::new(view, analysis, opts);
    let mut report = Report::default();
    report.stats.functions = analysis.functions.len();
    report.stats.syntactic_tables = analysis
        .functions
        .iter()
        .map(|f| f.jump_tables.len())
        .sum();
    for func in &analysis.functions {
        walker.function(func, &mut report);
    }
    report.stats = Stats {
        functions: analysis.functions.len(),
        ..report.stats
    };
    report
}

/// One block's incoming values.
type State = BTreeMap<String, Value>;

/// The analysis proper.
struct Walker<'a> {
    view: &'a View<'a>,
    opts: &'a Options,
    /// Block starts of every recovered function, the only addresses a target
    /// may have.
    blocks: BTreeSet<u64>,
    /// Function entries, used to tell a tail call from a switch.
    entries: BTreeSet<u64>,
}

impl<'a> Walker<'a> {
    fn new(view: &'a View<'a>, analysis: &Analysis, opts: &'a Options) -> Walker<'a> {
        let mut blocks = BTreeSet::new();
        let mut entries = BTreeSet::new();
        for f in &analysis.functions {
            entries.insert(f.entry.0);
            for b in &f.blocks {
                blocks.insert(b.start.0);
            }
        }
        Walker {
            view,
            opts,
            blocks,
            entries,
        }
    }

    /// True when the address starts a block the recovery found.
    fn is_target(&self, at: u64) -> bool {
        self.blocks.contains(&at)
    }

    /// True when the address is a function entry.
    fn is_entry(&self, at: u64) -> bool {
        self.entries.contains(&at)
    }

    fn function(&mut self, func: &Function, report: &mut Report) {
        if func.blocks.is_empty() || func.insns.is_empty() {
            return;
        }
        let program = build_program(
            self.view.image(),
            func,
            &occ_decompile::Options {
                max_blocks: self.opts.max_blocks,
                keep_comments: false,
                // The pass reads values, not statements: a block that ends in
                // a branch whose target was not recovered has no terminator to
                // use anything, so liveness would delete every definition in
                // it, including the one the computed jump goes through.
                keep_dead: true,
                debug: None,
            },
            &|a: Addr| format!("sub_{:x}", a.0),
        );
        if program.blocks.is_empty() {
            return;
        }
        report.stats.blocks += program.blocks.len();
        let out = self.fixpoint(&program);
        if out.is_none() {
            report.stats.unconverged += 1;
        }
        let out = out.unwrap_or_default();
        self.sites(func, &program, &out, report);
    }

    /// Iterate the blocks until the incoming values stop changing.
    fn fixpoint(&self, program: &Program) -> Option<Vec<Option<State>>> {
        let count = program.blocks.len();
        let mut out: Vec<Option<State>> = vec![None; count];
        for round in 0..self.opts.max_rounds {
            let mut changed = false;
            for id in 0..count {
                let mut state = self.entry_state(program, id, &out);
                self.run_block(program, id, &mut state, &out);
                let merged = match &out[id] {
                    None => Some(state),
                    Some(previous) => {
                        let joined = self.join_states(previous, &state, round);
                        if joined == *previous {
                            None
                        } else {
                            Some(joined)
                        }
                    }
                };
                if let Some(merged) = merged {
                    out[id] = Some(merged);
                    changed = true;
                }
            }
            if !changed {
                return Some(out);
            }
        }
        Some(out)
    }

    /// Join two states, widening once the round budget is half spent.
    fn join_states(&self, previous: &State, next: &State, round: usize) -> State {
        let widen = round + 1 >= self.opts.max_rounds / 2;
        let mut merged = previous.clone();
        for (name, value) in next {
            match merged.get(name) {
                None => {
                    merged.insert(name.clone(), value.clone());
                }
                Some(old) => {
                    let joined = if widen {
                        old.widen(value)
                    } else {
                        old.join(value)
                    };
                    merged.insert(name.clone(), joined);
                }
            }
        }
        merged
    }

    /// Values a block starts with: the join of its predecessors' outputs.
    fn entry_state(&self, program: &Program, id: BlockId, out: &[Option<State>]) -> State {
        let mut state = State::new();
        for pred in &program.preds[id] {
            let Some(previous) = &out[*pred] else {
                continue;
            };
            for (name, value) in previous {
                match state.get(name) {
                    None => {
                        state.insert(name.clone(), value.clone());
                    }
                    Some(old) => {
                        let joined = old.join(value);
                        state.insert(name.clone(), joined);
                    }
                }
            }
        }
        state
    }

    fn run_block(&self, program: &Program, id: BlockId, state: &mut State, out: &[Option<State>]) {
        for stmt in &program.blocks[id].stmts {
            self.step(stmt, state, out);
        }
    }

    fn step(&self, stmt: &Stmt, state: &mut State, out: &[Option<State>]) {
        match stmt {
            Stmt::Assign { dest, value } => {
                let value = self.eval(value, state, out, 0);
                state.insert(dest.clone(), value);
            }
            // A call returns something this pass does not model, except for
            // the allocators, whose result is a heap object named by the call.
            Stmt::Call { name, ret, addr, .. } => {
                if let Some(ret) = ret {
                    let value = if is_allocator(name) {
                        Value::Heap {
                            site: addr.0 as i64,
                        }
                    } else {
                        Value::Top
                    };
                    state.insert(ret.clone(), value);
                }
            }
            // A store is dropped: a table the program writes at run time is
            // not something this pass can read back.
            Stmt::Store { .. } | Stmt::Comment(_) => {}
        }
    }

    fn eval(&self, e: &Expr, state: &State, out: &[Option<State>], depth: usize) -> Value {
        if depth > self.opts.max_depth {
            return Value::Top;
        }
        match e {
            Expr::Const(v) => Value::num(*v as i64),
            // A variable the state does not know holds a value this pass
            // does not know, but it is still one value, and it is not the
            // same value as another variable's. Naming it keeps the two
            // apart: `index - 100` and `other - 100` differ, and only one
            // of them names a case.
            Expr::Var(name) => state.get(name).cloned().unwrap_or_else(|| {
                Value::Num(Num::Sym {
                    name: name.clone(),
                    scale: 1,
                    bias: 0,
                })
            }),
            Expr::Member {
                base,
                offset,
                size,
                ..
            } => {
                let addr = Expr::Bin {
                    op: BinOp::Add,
                    lhs: base.clone(),
                    rhs: Box::new(Expr::Const(*offset)),
                    bits: 8,
                };
                self.load(&addr, *size, false, state, out, depth)
            }
            Expr::Load { addr, size, signed } => self.load(addr, *size, *signed, state, out, depth),
            Expr::Bin { op, lhs, rhs, bits } => {
                let a = self.eval(lhs, state, out, depth + 1);
                let b = self.eval(rhs, state, out, depth + 1);
                self.binary(*op, &a, &b, *bits)
            }
            Expr::Un { op, arg, bits } => {
                let a = self.eval(arg, state, out, depth + 1);
                match (op, &a) {
                    (UnOp::Neg, Value::Num(n)) => Value::Num(Num::one(0).sub(n)),
                    (UnOp::Not, Value::Num(n)) => {
                        let mask = if *bits >= 8 {
                            u64::MAX as i64
                        } else {
                            (1i64 << (u16::from(*bits) * 8)) - 1
                        };
                        Value::Num(Num::one(mask).sub(n))
                    }
                    _ => Value::Top,
                }
            }
            Expr::Cast {
                kind,
                arg,
                from,
                to,
            } => {
                let value = self.eval(arg, state, out, depth + 1);
                self.cast(*kind, value, *from, *to)
            }
            // A comparison answers zero or one, and the analysis does not try
            // to know which: the branches are both walked.
            Expr::Cond { .. } | Expr::Bool { .. } | Expr::FlagCond { .. } => {
                let mut set = BTreeSet::new();
                set.insert(0);
                set.insert(1);
                Value::Num(Num::from_set(set))
            }
            Expr::Select { yes, no, .. } => {
                let a = self.eval(yes, state, out, depth + 1);
                let b = self.eval(no, state, out, depth + 1);
                a.join(&b)
            }
            Expr::Phi(args) => {
                let mut value = Value::Top;
                let mut first = true;
                for (pred, operand) in args {
                    let from = out
                        .get(*pred)
                        .and_then(|s| s.as_ref())
                        .map(|s| self.simple(operand, s))
                        .unwrap_or(Value::Top);
                    value = if first {
                        first = false;
                        from
                    } else {
                        value.join(&from)
                    };
                }
                value
            }
            Expr::Intrinsic { .. } | Expr::Unknown(_) => Value::Top,
        }
    }

    /// Evaluate an expression in a predecessor's state.
    ///
    /// A phi operand is read where it was computed, which is the predecessor.
    /// Only a name or a constant is read exactly; anything else is unknown,
    /// because evaluating it here would use the wrong state and could invent a
    /// value.
    fn simple(&self, e: &Expr, state: &State) -> Value {
        match e {
            Expr::Const(v) => Value::num(*v as i64),
            Expr::Var(name) => state.get(name).cloned().unwrap_or(Value::Top),
            _ => Value::Top,
        }
    }

    fn binary(&self, op: BinOp, a: &Value, b: &Value, bits: u8) -> Value {
        // A table plus an address is the table moved: this is the `add rax, rdx`
        // that turns a table of offsets into a table of addresses.
        if let Value::Table { .. } = a {
            return a.offset(&self.as_num(b));
        }
        if let Value::Table { .. } = b {
            return b.offset(&self.as_num(a));
        }
        // An address plus a constant stays an address.
        if let (Value::Global { .. }, Value::Num(y)) = (a, b) {
            return a.offset(y);
        }
        if let (Value::Num(x), Value::Global { .. }) = (a, b) {
            return b.offset(x);
        }
        let (Value::Num(x), Value::Num(y)) = (a, b) else {
            return Value::Top;
        };
        let result = match op {
            BinOp::Add => x.add(y),
            BinOp::Sub => x.sub(y),
            BinOp::Mul => x.mul(y),
            BinOp::Shl => match y.single() {
                Some(k) if k >= 0 => x.shl(k as u32),
                _ => Num::Top,
            },
            BinOp::And => x.and(y),
            BinOp::Or => x.or(y),
            BinOp::Xor => x.xor(y),
            BinOp::Div | BinOp::UDiv | BinOp::Mod | BinOp::UMod => match (x.single(), y.single()) {
                (Some(_), Some(0)) | (_, None) => Num::Top,
                (Some(p), Some(q)) => {
                    let v = if matches!(op, BinOp::Mod | BinOp::UMod) {
                        p.wrapping_rem(q)
                    } else {
                        p.wrapping_div(q)
                    };
                    Num::one(v)
                }
                _ => Num::Top,
            },
            _ => Num::Top,
        };
        let _ = bits;
        Value::Num(result)
    }

    /// The numeric part of a value, for table arithmetic.
    fn as_num(&self, v: &Value) -> Num {
        match v {
            Value::Num(n) => n.clone(),
            Value::Global { at, off, .. } => off.map(|o| at.wrapping_add(o)),
            Value::Stack { off } => Num::one(*off),
            Value::Heap { site } => Num::one(*site),
            Value::Table { .. } | Value::Top => Num::Top,
        }
    }

    fn cast(&self, kind: CastKind, value: Value, from: u8, to: u8) -> Value {
        let from_bits = u32::from(from) * 8;
        let to_bits = u32::from(to) * 8;
        match (kind, &value) {
            (CastKind::Trunc, Value::Num(n)) => {
                Value::Num(n.truncate(to_bits.min(from_bits), false))
            }
            (CastKind::Zext, Value::Num(n)) => Value::Num(n.truncate(to_bits, false)),
            (CastKind::Sext, Value::Num(n)) => Value::Num(n.truncate(to_bits, true)),
            // A table of narrow offsets read with `movslq` is still a table:
            // the cast is applied to each entry, not to the shape. This is the
            // `movslq (%rdx,%rdi,4),%rax` of a relative jump table, whose
            // entries are 32-bit offsets from the table base.
            (
                CastKind::Sext,
                Value::Table {
                    base,
                    stride,
                    bias,
                    entries,
                },
            ) => Value::Table {
                base: *base,
                stride: *stride,
                bias: *bias,
                entries: entries
                    .iter()
                    .map(|n| extend(*n as u64, (from_bits / 8) as u8, true))
                    .collect(),
            },
            (
                CastKind::Trunc,
                Value::Table {
                    base,
                    stride,
                    bias,
                    entries,
                },
            ) => Value::Table {
                base: *base,
                stride: *stride,
                bias: *bias,
                entries: entries
                    .iter()
                    .map(|n| extend(*n as u64, (to_bits.min(from_bits) / 8) as u8, false))
                    .collect(),
            },
            // Widening an address keeps it an address.
            (CastKind::Zext, Value::Global { .. }) if to >= from => value.clone(),
            _ => Value::Top,
        }
    }

    fn load(
        &self,
        addr: &Expr,
        size: u8,
        signed: bool,
        state: &State,
        out: &[Option<State>],
        depth: usize,
    ) -> Value {
        if size == 0 || size > 8 {
            return Value::Top;
        }
        // A known set of addresses: read each cell.
        let value = self.eval(addr, state, out, depth + 1);
        let mut cells = BTreeSet::new();
        let mut known = true;
        match &value {
            Value::Num(n) => match n.listed() {
                Some(list) => cells.extend(list.iter().copied()),
                None => known = false,
            },
            Value::Global { at, off, .. } => match off.listed() {
                Some(list) => cells.extend(list.iter().map(|o| at.wrapping_add(*o))),
                None => known = false,
            },
            _ => known = false,
        }
        if known && !cells.is_empty() {
            let mut values = BTreeSet::new();
            for cell in &cells {
                let Some(raw) = self.view.read_uint(Addr(*cell as u64), usize::from(size)) else {
                    return Value::Top;
                };
                values.insert(extend(raw, size, signed));
            }
            return Value::Num(Num::from_set(values));
        }
        // Otherwise it may be a table: base plus a stride.
        let Some(affine) = self.affine(addr, state, depth) else {
            return Value::Top;
        };
        if affine.stride == 0 {
            return Value::Top;
        }
        let mut entries = BTreeSet::new();
        for e in self.table_run(affine.base, affine.stride, size, signed) {
            entries.insert(e);
        }
        if entries.len() < 2 {
            return Value::Top;
        }
        Value::Table {
            base: affine.base,
            stride: affine.stride,
            bias: affine.bias,
            entries,
        }
    }

    /// The shape of an address: a base plus one unknown term of a known stride.
    ///
    /// This is what makes a table readable when the index is a function
    /// parameter: `table + rdi*8` has an unknown value but a known shape, and
    /// the shape is the table.
    ///
    /// The address is `base + stride * position`, where `position` is the
    /// index's own value: `base` is where the entry at position zero sits,
    /// and `stride` is how far apart the entries are. The label of an entry
    /// is `position - bias`, because `bias` is what the compiler added to
    /// the index before it looked anything up.
    fn affine(&self, e: &Expr, state: &State, depth: usize) -> Option<Affine> {
        if depth > self.opts.max_depth {
            return None;
        }
        match e {
            Expr::Const(v) => Some(Affine {
                base: *v as i64,
                stride: 0,
                bias: 0,
            }),
            Expr::Var(name) => match state.get(name) {
                Some(Value::Num(n)) => match n.single() {
                    Some(v) => Some(Affine {
                        base: v,
                        stride: 0,
                        bias: 0,
                    }),
                    // An index the analysis knows the shape of: the symbol
                    // carries the constant the compiler added to it, and
                    // that is what names the cases.
                    None => {
                        let (scale, bias) = n.symbol().map(|(_, s, b)| (s, b)).unwrap_or((1, 0));
                        Some(Affine {
                            base: 0,
                            stride: 1,
                            // The cases are consecutive only when the index
                            // advances by one at a time; when it does not,
                            // the offset says nothing about the labels.
                            bias: if scale == 1 { bias } else { 0 },
                        })
                    }
                },
                Some(Value::Global { at, off, .. }) => off.single().map(|o| Affine {
                    base: at.wrapping_add(o),
                    stride: 0,
                    bias: 0,
                }),
                // An unknown value is one unknown term with a stride of one:
                // the caller scales it.
                _ => Some(Affine {
                    base: 0,
                    stride: 1,
                    bias: 0,
                }),
            },
            Expr::Bin { op, lhs, rhs, .. } => {
                let a = self.affine(lhs, state, depth + 1)?;
                let b = self.affine(rhs, state, depth + 1)?;
                match op {
                    BinOp::Add => match (a.stride == 0, b.stride == 0) {
                        // Two known values: a constant.
                        (true, true) => Some(Affine {
                            base: a.base.wrapping_add(b.base),
                            stride: 0,
                            bias: 0,
                        }),
                        // One known value and one index: the known value is
                        // the base, wherever the index's own zero lands.
                        (true, false) => Some(Affine {
                            base: b.base.wrapping_add(a.base),
                            stride: b.stride,
                            bias: b.bias,
                        }),
                        (false, true) => Some(Affine {
                            base: a.base.wrapping_add(b.base),
                            stride: a.stride,
                            bias: a.bias,
                        }),
                        // Two index terms: the address does not depend on
                        // one index, so no entry can be named.
                        (false, false) => Some(Affine {
                            base: a.base.wrapping_add(b.base),
                            stride: a.stride.checked_add(b.stride)?,
                            bias: 0,
                        }),
                    },
                    BinOp::Sub => match (a.stride == 0, b.stride == 0) {
                        (true, true) => Some(Affine {
                            base: a.base.wrapping_sub(b.base),
                            stride: 0,
                            bias: 0,
                        }),
                        // `index - 100`: the constant shifts the index, not
                        // the table.
                        (false, true) => Some(Affine {
                            base: a.base.wrapping_sub(b.base),
                            stride: a.stride,
                            bias: a.bias,
                        }),
                        _ => None,
                    },
                    BinOp::Mul | BinOp::Shl => {
                        let b = self.eval(rhs, state, &[], depth + 1);
                        let k = match (op, b) {
                            (BinOp::Mul, Value::Num(n)) => n.single()?,
                            (BinOp::Shl, Value::Num(n)) => 1i64.checked_shl(n.single()? as u32)?,
                            _ => return None,
                        };
                        // Scaling the index scales its stride and its base
                        // together; the constant the index carries is inside
                        // the index, so it is untouched.
                        Some(Affine {
                            base: a.base.wrapping_mul(k),
                            stride: a.stride.checked_mul(k)?,
                            bias: a.bias,
                        })
                    }
                    _ => None,
                }
            }
            Expr::Cast {
                kind: CastKind::Zext | CastKind::Sext,
                arg,
                ..
            } => self.affine(arg, state, depth + 1),
            _ => None,
        }
    }

    /// Walk the blocks a second time and record what the indirect sites reach.
    fn sites(&self, func: &Function, program: &Program, out: &[Option<State>], report: &mut Report) {
        for (id, block) in program.blocks.iter().enumerate() {
            let mut state = self.entry_state(program, id, out);
            for stmt in &block.stmts {
                if let Stmt::Call { name, addr, .. } = stmt
                    && name.starts_with("(*")
                {
                    self.call_site(func, *addr, &state, report);
                }
                self.step(stmt, &mut state, out);
            }
            // The terminator is the last instruction of the block.
            let Some(last) = last_instruction(func, block.addr) else {
                continue;
            };
            if last.flow == FlowKind::IndirectBranch {
                self.jump_site(func, program, id, last, &state, report);
            }
        }
    }

    fn call_site(
        &self,
        func: &Function,
        at: Addr,
        state: &State,
        report: &mut Report,
    ) {
        let Some(insn) = func.insns.iter().find(|i| i.addr == at) else {
            return;
        };
        let Some(via) = register_of(&insn.text) else {
            return;
        };
        report.stats.call_sites += 1;
        // The name in the state is the one the block's last definition gave
        // it, and the definition is the last sighting of the register before
        // the call.
        let value = self.register_value(state, &via);
        let mut targets = Vec::new();
        let mut dropped = 0usize;
        for at in self.addresses_of(&value) {
            if self.is_target(at as u64) {
                targets.push(Addr(at as u64));
            } else {
                dropped += 1;
            }
        }
        targets.sort_unstable();
        targets.dedup();
        if !targets.is_empty() {
            report.stats.calls_resolved += 1;
        }
        report.calls.push(CallSite {
            in_function: func.entry,
            at,
            via: via.clone(),
            targets,
            evidence: self.describe(&value, dropped),
        });
    }

    /// The value a register held, by the name the renaming gave it.
    ///
    /// The renaming gives every definition its own name — `rax`, `rax.1`,
    /// `rax.2` — and the value a call goes through is the one the last
    /// definition before it produced. The versions go up with the
    /// definitions, so the highest one is the last. Looking the register
    /// up by name is what finds the value across a block boundary: the
    /// definition is usually in the block before, and the call is the
    /// first thing in its own.
    fn register_value(&self, state: &State, reg: &str) -> Value {
        let prefix = format!("{reg}.");
        let mut best: Option<(u64, &Value)> = None;
        for (name, value) in state {
            let version = if name == reg {
                0
            } else if let Some(rest) = name.strip_prefix(&prefix) {
                match rest.parse::<u64>() {
                    Ok(version) => version,
                    Err(_) => continue,
                }
            } else {
                continue;
            };
            match best {
                Some((seen, _)) if seen >= version => {}
                _ => best = Some((version, value)),
            }
        }
        best.map(|(_, value)| value.clone()).unwrap_or(Value::Top)
    }

    fn jump_site(
        &self,
        func: &Function,
        program: &Program,
        block: BlockId,
        insn: &occ_core::Instruction,
        state: &State,
        report: &mut Report,
    ) {
        report.stats.jump_sites += 1;
        let Some(entry) = program.blocks.get(block) else {
            return;
        };
        let via = register_of(&insn.text).unwrap_or_else(|| "memory".into());
        match &entry.term {
            // The analyzer recovered the table: the cases are in table
            // order, and the index says which case each position is.
            Term::Switch {
                index,
                targets,
                first_case,
            } => {
                let value = self.eval(index, state, &[], 0);
                let shift = self.label_shift(&value);
                let mut cases = Vec::with_capacity(targets.len());
                let mut dropped = 0usize;
                let mut disagree = None;
                for (i, target) in targets.iter().enumerate() {
                    let Some(body) = program.blocks.get(*target) else {
                        dropped += 1;
                        continue;
                    };
                    // Two readings of the same fact: the constant the
                    // compiler took off the index, and the offset the index
                    // carries. They have to agree, and when they do not,
                    // the site says so instead of picking one.
                    let label = match first_case {
                        Some(first) if shift != -first => {
                            disagree = Some((*first, shift));
                            (i as i64).wrapping_sub(shift)
                        }
                        Some(first) => (i as i64).wrapping_add(*first),
                        None => (i as i64).wrapping_sub(shift),
                    };
                    cases.push(Case {
                        index: i,
                        label,
                        target: body.addr,
                    });
                }
                if !cases.is_empty() {
                    report.stats.jumps_resolved += 1;
                }
                let mut evidence = format!("the table was recovered: {} cases", cases.len());
                if shift != 0 {
                    evidence.push_str(&format!(
                        ", and the index carries {shift}, so the first case is {}",
                        -shift
                    ));
                }
                if let Some((first, shift)) = disagree {
                    evidence.push_str(&format!(
                        "; the code subtracts {first} from the index but the index carries {shift}, \
                         and the two readings do not agree"
                    ));
                }
                if dropped > 0 {
                    evidence.push_str(&format!(
                        "; {dropped} case(s) named no block and were dropped"
                    ));
                }
                report.jumps.push(JumpSite {
                    in_function: func.entry,
                    at: insn.addr,
                    via,
                    table: None,
                    cases,
                    two_level: false,
                    tail: false,
                    evidence,
                });
            }
            // No table was recovered, but the jump still reads an address,
            // and the value of that address is what this pass is for.
            Term::Indirect { .. } => {
                let value = self.jump_value(program, block, state);
                let table = match &value {
                    Value::Table {
                        base,
                        stride,
                        bias,
                        entries,
                    } => Some(Table {
                        base: Addr(*base as u64),
                        stride: *stride,
                        bias: *bias,
                        entries: entries.len(),
                    }),
                    _ => None,
                };
                let mut dropped = 0usize;
                let mut targets = Vec::new();
                for at in self.addresses_of(&value) {
                    if self.is_target(at as u64) {
                        targets.push(at);
                    } else {
                        dropped += 1;
                    }
                }
                // A run of stubs that name a second table is one switch, not
                // a jump to a dispatcher: the cases are behind the stubs.
                let (mut cases, two_level) = match self.two_level(program, &targets) {
                    Some(resolved) => (resolved, true),
                    None => (
                        targets
                            .iter()
                            .enumerate()
                            .map(|(i, t)| Case {
                                index: i,
                                label: i as i64,
                                target: Addr(*t as u64),
                            })
                            .collect::<Vec<_>>(),
                        false,
                    ),
                };
                // The table position is not the case value. A compiler that
                // sees `case 100..106` subtracts 100 before indexing, and the
                // constant it subtracted is in the table's shape, so the
                // label is the position shifted back by exactly that.
                if let Some(shift) = table.as_ref().map(|t| t.bias).filter(|b| *b != 0) {
                    for case in &mut cases {
                        case.label = case.label.wrapping_sub(shift);
                    }
                }
                if !cases.is_empty() {
                    report.stats.jumps_resolved += 1;
                }
                if two_level {
                    report.stats.two_level += 1;
                }
                if table.is_some() {
                    report.stats.tables += 1;
                }
                let tail = !targets.is_empty() && targets.iter().all(|t| self.is_entry(*t as u64));
                let mut evidence = self.describe(&value, dropped);
                if two_level {
                    evidence.push_str("; the cases sit one table behind a run of stubs");
                }
                if tail {
                    evidence.push_str("; every target is a function entry, so this is a tail call");
                }
                report.jumps.push(JumpSite {
                    in_function: func.entry,
                    at: insn.addr,
                    via,
                    table,
                    cases,
                    two_level,
                    tail,
                    evidence,
                });
            }
            _ => {}
        }
    }

    /// How far the table position is from the case value.
    ///
    /// The index of a switch is not the case value: a compiler that sees
    /// `case 100..106` subtracts 100 and looks the result up. The constant
    /// it subtracted is the index's own offset, and it is what turns a list
    /// of targets into a list of cases.
    fn label_shift(&self, value: &Value) -> i64 {
        match value {
            Value::Num(n) => n.symbol().map(|(_, scale, bias)| {
                if scale == 1 {
                    bias
                } else {
                    0
                }
            }).unwrap_or(0),
            _ => 0,
        }
    }

    /// The value a computed jump goes through.
    ///
    /// The terminator carries the address the jump reads, with the names the
    /// block's own statements defined, so evaluating it is the whole job: no
    /// reading of the instruction text, and no second parser to disagree with
    /// the lifter about what `jmp *(%rdx,%rax,8)` reads.
    fn jump_value(&self, program: &Program, block: BlockId, state: &State) -> Value {
        let Some(entry) = program.blocks.get(block) else {
            return Value::Top;
        };
        let Term::Indirect { target } = &entry.term else {
            return Value::Top;
        };
        self.eval(target, state, &[], 0)
    }

    /// Read a run of words from one section, stopping at its end.
    ///
    /// The bound is the section, not the file: a table of ten entries followed
    /// by unrelated read-only data is a table of ten, and reading past its end
    /// invents cases that the switch never had. Anything past the section is
    /// not part of the same object.
    /// Read a run of words that all point at code, stopping at the first that
    /// does not.
    ///
    /// Two bounds, and the tighter one wins. The section says where the data
    /// could physically be, which is not the same as where the table ends: a
    /// ten-entry table of relative offsets can sit in front of unrelated
    /// constants in the same section, and reading into them invents cases the
    /// switch never had. What ends the table is the first entry that names no
    /// block, because every entry of a jump table is a jump target. An entry
    /// is taken as an address both ways — a table of pointers holds addresses
    /// and a table of relative offsets holds positions — and the caller's
    /// arithmetic is what picks between them.
    fn table_run(&self, base: i64, stride: i64, size: u8, signed: bool) -> Vec<i64> {
        let mut entries = Vec::new();
        if stride == 0 || size == 0 || size > 8 {
            return entries;
        }
        let Some(section) = self.view.section_at(Addr(base as u64)) else {
            return entries;
        };
        let end = section.addr.0.wrapping_add(section.size);
        for k in 0..self.opts.max_table_entries {
            let at = base.wrapping_add(stride.wrapping_mul(k as i64));
            if at < 0 || (at as u64).wrapping_add(u64::from(size)) > end {
                break;
            }
            let Some(raw) = self.view.read_uint(Addr(at as u64), usize::from(size)) else {
                break;
            };
            // A relative table stores `target - base`, which is negative for
            // every case before the table. The load that reads it does not
            // sign-extend — `movslq` is a separate instruction — so both
            // readings are tried and the one that lands on a block is kept.
            let unsigned = extend(raw, size, false);
            let signed_value = extend(raw, size, true);
            let target = |v: i64| self.is_target(v as u64) || self.is_target(base.wrapping_add(v) as u64);
            let entry = if target(unsigned) {
                unsigned
            } else if signed {
                // The load already sign-extends, so an unsigned reading that
                // walked off the code is the only one left.
                break;
            } else if target(signed_value) {
                signed_value
            } else {
                break;
            };
            entries.push(entry);
        }
        entries
    }

    /// Resolve a first-level table of stub blocks into the cases behind them.
    ///
    /// A first-level entry is a stub that puts a constant in a register and
    /// jumps to one shared block, which reads the second table with that
    /// constant. Reading the constant gives the case label; reading the second
    /// table at that constant gives the body. Both have to be true for the
    /// answer to be reported.
    fn two_level(&self, program: &Program, targets: &[i64]) -> Option<Vec<Case>> {
        if targets.len() < 3 {
            return None;
        }
        let mut constants = Vec::with_capacity(targets.len());
        let mut common: Option<BlockId> = None;
        for target in targets {
            let id = program.block_at(Addr(*target as u64))?;
            let (value, next) = stub(program, id)?;
            match common {
                None => common = Some(next),
                Some(seen) if seen == next => {}
                Some(_) => return None,
            }
            constants.push(value);
        }
        let common = common?;
        // The shared block reads the second table, indexed by the constant.
        let index = program.blocks[common]
            .stmts
            .iter()
            .find(|s| matches!(s, Stmt::Assign { .. }))
            .map(|s| match s {
                Stmt::Assign { dest, .. } => dest.clone(),
                _ => String::new(),
            });
        let _ = index;
        let mut cases = Vec::with_capacity(constants.len());
        for (i, label) in constants.iter().enumerate() {
            let target = self.second_table_target(program, common, *label)?;
            cases.push(Case {
                index: i,
                label: *label,
                target,
            });
        }
        Some(cases)
    }

    /// The block a `label` reaches through the second table of `block`.
    fn second_table_target(&self, program: &Program, block: BlockId, label: i64) -> Option<Addr> {
        let block = program.blocks.get(block)?;
        // The second table is read by a load whose address is
        // `base + label * stride`; the block that ends in the indirect jump.
        for stmt in &block.stmts {
            let Stmt::Assign { value, .. } = stmt else {
                continue;
            };
            let Expr::Load { addr, size, signed } = value else {
                continue;
            };
            let Some(affine) = self.affine(addr, &State::new(), 0) else {
                continue;
            };
            if affine.stride == 0 {
                continue;
            }
            let at = affine.base.wrapping_add(affine.stride.wrapping_mul(label));
            let raw = self.view.read_uint(Addr(at as u64), usize::from(*size))?;
            let entry = extend(raw, *size, *signed);
            // A second-level table of offsets is relative to its own base.
            for candidate in [entry, affine.base.wrapping_add(entry)] {
                if self.is_target(candidate as u64) {
                    return Some(Addr(candidate as u64));
                }
            }
        }
        None
    }

    /// Every address a value can name.
    fn addresses_of(&self, value: &Value) -> Vec<i64> {
        match value {
            Value::Num(n) => n.listed().unwrap_or_default(),
            Value::Global { at, off, .. } => off
                .listed()
                .map(|list| list.iter().map(|o| at.wrapping_add(*o)).collect())
                .unwrap_or_default(),
            Value::Table { entries, .. } => entries.iter().copied().collect(),
            Value::Stack { off } => vec![*off],
            Value::Heap { site } => vec![*site],
            Value::Top => Vec::new(),
        }
    }

    /// A sentence about where a value came from.
    fn describe(&self, value: &Value, dropped: usize) -> String {
        let mut text = match value {
            Value::Num(n) => match n {
                Num::Exact(set) => format!("{} addresses", set.len()),
                Num::Stride { lo, stride, steps } => {
                    format!("a run of {} at {lo:#x} step {stride}", steps + 1)
                }
                Num::Sym {
                    name,
                    scale,
                    bias,
                } => format!(
                    "the value of {name}{}{}",
                    if *scale == 1 {
                        String::new()
                    } else {
                        format!(" times {scale}")
                    },
                    if *bias == 0 {
                        String::new()
                    } else {
                        format!(" less {bias}")
                    }
                ),
                Num::Top => "no value".into(),
            },
            Value::Global { name, at, off } => match off {
                Num::Exact(_) | Num::Stride { .. } => format!("{name}+{:#x}", at),
                Num::Sym { .. } => format!("{name}+?"),
                Num::Top => format!("{name}+?"),
            },
            Value::Table {
                base,
                stride,
                bias,
                entries,
            } => format!(
                "a table at {base:#x}, stride {stride}, {} entries{}",
                entries.len(),
                if *bias == 0 {
                    String::new()
                } else {
                    format!(", indexed after {} so the labels start at {}", -bias, -bias)
                }
            ),
            Value::Stack { off } => format!("frame slot {off}"),
            Value::Heap { site } => format!("heap object from {site:#x}"),
            Value::Top => "no value".into(),
        };
        if dropped > 0 {
            text.push_str(&format!(
                "; {dropped} candidate(s) fell outside the recovered code and were dropped"
            ));
        }
        text
    }
}

/// The shape of an address that reads a table.
///
/// The address is `base + stride * position`, where `position` is the
/// index's own value: `base` is where the entry at position zero sits,
/// `stride` is how far apart the entries are, and `bias` is what the
/// compiler added to the index before it looked anything up. Keeping the
/// three apart is what lets `table[i - 100]` read the table instead of
/// reading 400 bytes before it, and what lets the entry at position zero
/// be called case 100.
#[derive(Clone, Copy, Debug)]
struct Affine {
    /// Address of the entry at position zero.
    base: i64,
    /// Bytes between neighbouring entries; zero when there is no index.
    stride: i64,
    /// Constant the compiler added to the index: the label of the entry at
    /// position zero is `-bias`.
    bias: i64,
}

/// The register an indirect branch or call goes through, read off the text.
fn register_of(text: &str) -> Option<String> {
    let operand = text.split_once(' ').map(|(_, rest)| rest.trim())?;
    // A memory operand is not a register, and the value that matters is the
    // one loaded from it, which the block computed earlier.
    let name = operand.split('[').next().unwrap_or(operand).trim();
    let name = name.strip_prefix("qword ptr ").unwrap_or(name);
    let name = name.trim_start_matches('*').trim();
    if name.starts_with('(') || name.contains('(') {
        // A memory operand is not a register.
        return None;
    }
    let name = name.trim_start_matches('%');
    if name.is_empty() || !name.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    if name.contains(' ') || name.contains(',') {
        return None;
    }
    Some(name.to_string())
}

/// Sign or zero extend a word read from the image.
fn extend(raw: u64, size: u8, signed: bool) -> i64 {
    let bits = u32::from(size) * 8;
    if size >= 8 {
        return raw as i64;
    }
    let masked = raw & ((1u64 << bits) - 1);
    if signed && masked >> (bits - 1) != 0 {
        (masked | !((1u64 << bits) - 1)) as i64
    } else {
        masked as i64
    }
}

/// The last instruction of a block.
fn last_instruction(func: &Function, start: Addr) -> Option<&occ_core::Instruction> {
    let block = func.blocks.iter().find(|b| b.start == start)?;
    func.insns
        .get(block.insn_end.saturating_sub(1))
        .filter(|i| i.addr < block.end)
}

/// A block that only puts a constant in a register and jumps on.
///
/// This is what a first-level table entry points at when the switch needs two
/// tables: the entry names a stub, the stub names the case, and the case is an
/// index into the second table.
fn stub(program: &Program, id: BlockId) -> Option<(i64, BlockId)> {
    let block = program.blocks.get(id)?;
    let mut constant = None;
    let mut assignments = 0;
    for stmt in &block.stmts {
        match stmt {
            Stmt::Assign { value, .. } => {
                assignments += 1;
                match value {
                    Expr::Const(v) => constant = Some(*v as i64),
                    _ => return None,
                }
            }
            Stmt::Comment(_) => {}
            _ => return None,
        }
    }
    if assignments != 1 {
        return None;
    }
    let Term::Jump(next) = block.term else {
        return None;
    };
    Some((constant?, next))
}

/// True for the calls whose result is a fresh object.
fn is_allocator(name: &str) -> bool {
    const NAMES: &[&str] = &["malloc", "calloc", "realloc", "operator new", "new"];
    NAMES.iter().any(|n| name.contains(n))
}
