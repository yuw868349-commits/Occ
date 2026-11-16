//! Path exploration over the decompiler IR.
//!
//! The executor walks the same blocks the structurer does — not the instructions
//! — so a path it reports can be checked against `code.decompile` line for line.
//! Every branch whose condition depends on an input forks the state and records
//! the decision; when a state returns, the decisions plus the returned value are
//! one row of the result.
//!
//! Three things bound the search, and all three are reported rather than hidden:
//! a state budget, a per-path step budget, and a limit on how often one block
//! may repeat on a single path (which is how a loop is unrolled). A path that
//! hits a bound is `Truncated`, never `Unreachable` — the difference matters,
//! because the second one is a claim about the program.

use std::collections::HashMap;

use occ_core::{Addr, Function, Image};
use occ_decompile::ir::{BinOp, BlockId, CondOp, Expr, Stmt, Term};
use occ_decompile::{Options, Program, build_program};

use crate::mem::Memory;
use crate::solve::{self, Constraint, Model, Verdict};
use crate::value::Value;

/// How unknown inputs are treated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Anything not supplied becomes a symbolic input and branches fork. This
    /// is the default, and the only mode that can answer "which inputs reach
    /// this branch".
    Symbolic,
    /// Anything not supplied aborts the path and is named in the result. Use
    /// this to check one concrete call: the report then lists exactly which
    /// registers and addresses still need values.
    Concrete,
}

/// Limits and initial state.
#[derive(Clone, Debug)]
pub struct Config {
    /// Maximal number of paths kept in the result.
    pub max_paths: usize,
    /// Total states explored, including forks. Bounds the whole run.
    pub max_states: usize,
    /// Blocks executed per path, across all its iterations.
    pub max_steps: usize,
    /// How often one block may be executed on a single path before the path is
    /// truncated. This is the loop unrolling depth.
    pub max_loop: u16,
    /// Candidate assignments the constraint solver may try per path.
    pub solve_cap: usize,
    /// Input values known up front, by IR name (`rdi`, `rsi`, …).
    pub inputs: Vec<(String, u64)>,
    /// Memory initialised up front.
    pub memory: Vec<(Addr, Vec<u8>)>,
    /// Unknown-input policy.
    pub mode: Mode,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_paths: 24,
            max_states: 256,
            max_steps: 20_000,
            max_loop: 4,
            solve_cap: 4096,
            inputs: Vec::new(),
            memory: Vec::new(),
            mode: Mode::Symbolic,
        }
    }
}

/// Reads initialised bytes from the image's loadable data, for addresses the
/// executor did not write itself. `None` means the address is not file-backed.
pub type StaticRead<'a> = &'a dyn Fn(Addr, u8) -> Option<u64>;

/// Call a path made, in order.
#[derive(Clone, Debug)]
pub struct CallInfo {
    /// Rendered callee name.
    pub name: String,
    /// Address of the call instruction.
    pub addr: Addr,
    /// Arguments as evaluated at the call.
    pub args: Vec<Expr>,
}

/// How a path ended.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// Reached a `ret`, possibly with a value.
    Return,
    /// Reached a terminator the lifter marked as not returning.
    Unreachable,
    /// Hit a search bound.
    Truncated(&'static str),
    /// `Mode::Concrete` needed an input that was not supplied.
    Missing(Vec<String>),
}

/// One complete path through the function.
#[derive(Clone, Debug)]
pub struct Path {
    /// How it ended.
    pub outcome: Outcome,
    /// Branch decisions, in order.
    pub conds: Vec<Constraint>,
    /// Solved input assignment, when one was found.
    pub model: Model,
    /// Block addresses visited, in order.
    pub blocks: Vec<Addr>,
    /// Calls made, in order.
    pub calls: Vec<CallInfo>,
    /// Returned value, when there was one.
    pub ret: Option<String>,
}

impl Path {
    /// True when the constraints are satisfiable and the path is a real
    /// execution. Truncated and missing-input paths are not counted.
    pub fn is_feasible(&self) -> bool {
        self.model.verdict == Verdict::Satisfied
            && !matches!(self.outcome, Outcome::Truncated(_) | Outcome::Missing(_))
    }

    /// Branch decisions in printed form.
    pub fn conditions(&self) -> Vec<String> {
        self.conds.iter().map(|c| c.render()).collect()
    }
}

/// Result of one exploration.
#[derive(Clone, Debug)]
pub struct Report {
    /// Entry address.
    pub entry: Addr,
    /// Function name as the caller resolved it.
    pub name: String,
    /// Parameter names the body reads, as the pseudo-C prints them.
    pub params: Vec<String>,
    /// Argument registers for those parameters: what `inputs` must name.
    pub param_regs: Vec<String>,
    /// Paths, in discovery order.
    pub paths: Vec<Path>,
    /// Blocks executed in total.
    pub steps: usize,
    /// States dropped at the budget.
    pub dropped: usize,
    /// Notes about approximations.
    pub notes: Vec<String>,
    /// Leftovers from lifting.
    pub lift_notes: Vec<String>,
}

impl Report {
    /// Number of paths whose constraints were satisfied.
    pub fn feasible(&self) -> usize {
        self.paths.iter().filter(|p| p.is_feasible()).count()
    }

    /// Distinct return values across feasible paths, printed.
    pub fn returns(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .paths
            .iter()
            .filter(|p| p.is_feasible())
            .filter_map(|p| p.ret.clone())
            .collect();
        out.sort();
        out.dedup();
        out
    }
}

/// One executor run.
#[derive(Clone)]
struct State {
    env: HashMap<String, Value>,
    mem: Memory,
    conds: Vec<Constraint>,
    visits: HashMap<BlockId, u16>,
    block: BlockId,
    blocks: Vec<BlockId>,
    calls: Vec<CallInfo>,
    /// Set when `Mode::Concrete` needs an input.
    missing: Vec<String>,
}

impl State {
    fn new(block: BlockId) -> State {
        State {
            env: HashMap::new(),
            mem: Memory::new(),
            conds: Vec::new(),
            visits: HashMap::new(),
            block,
            blocks: Vec::new(),
            calls: Vec::new(),
            missing: Vec::new(),
        }
    }
}

struct Runner<'a> {
    program: Program,
    cfg: Config,
    read_static: StaticRead<'a>,
    notes: Vec<String>,
    paths: Vec<Path>,
    steps: usize,
    budget: usize,
    dropped: usize,
}

/// Explore one function.
pub fn explore<'a>(
    image: &Image,
    func: &Function,
    opts: &Options,
    name_of: &dyn Fn(Addr) -> String,
    read_static: StaticRead<'a>,
    cfg: &Config,
) -> Report {
    let program = build_program(image, func, opts, name_of);
    let lift_notes = program.notes.clone();
    let mut runner = Runner {
        steps: 0,
        budget: cfg.max_states.max(1),
        dropped: 0,
        notes: Vec::new(),
        paths: Vec::new(),
        read_static,
        cfg: cfg.clone(),
        program,
    };
    runner.run();
    Report {
        entry: func.entry,
        name: func.display_name(),
        params: runner.program.params.clone(),
        param_regs: runner.program.param_regs.clone(),
        paths: runner.paths,
        steps: runner.steps,
        dropped: runner.dropped,
        notes: runner.notes,
        lift_notes,
    }
}

impl Runner<'_> {
    fn run(&mut self) {
        if self.program.blocks.is_empty() {
            self.notes
                .push("no blocks: nothing to execute (is the architecture lifted?)".into());
            return;
        }
        let mut start = State::new(self.program.entry);
        for (name, value) in &self.cfg.inputs {
            start.env.insert(name.clone(), Value::Const(*value));
        }
        for (addr, bytes) in &self.cfg.memory {
            start.mem.write_bytes(*addr, bytes);
        }
        let mut work = vec![start];
        while let Some(state) = work.pop() {
            if self.paths.len() >= self.cfg.max_paths {
                self.dropped += work.len() + 1;
                self.notes.push(format!(
                    "stopped at {} paths; {} state(s) were not finished",
                    self.cfg.max_paths,
                    self.dropped
                ));
                break;
            }
            self.run_path(state, &mut work);
        }
        self.notes.push(format!(
            "{} block execution(s), {} path(s), {} state(s) dropped",
            self.steps,
            self.paths.len(),
            self.dropped
        ));
    }

    fn run_path(&mut self, mut state: State, work: &mut Vec<State>) {
        loop {
            self.steps += 1;
            if self.steps > self.cfg.max_steps {
                self.finish(state, Outcome::Truncated("step budget"));
                return;
            }
            if state.block >= self.program.blocks.len() {
                self.finish(state, Outcome::Truncated("edge leaves the function"));
                return;
            }
            let block = self.program.blocks[state.block].clone();
            let addr = block.addr;
            state.blocks.push(state.block);
            let seen = state.visits.entry(state.block).or_insert(0);
            *seen += 1;
            if *seen > self.cfg.max_loop {
                self.finish(state, Outcome::Truncated("loop bound"));
                return;
            }

            for stmt in &block.stmts {
                self.stmt(&mut state, stmt);
                if !state.missing.is_empty() {
                    let missing = std::mem::take(&mut state.missing);
                    self.finish(state, Outcome::Missing(missing));
                    return;
                }
            }

            match &block.term {
                Term::Ret(value) => {
                    let ret = value.as_ref().map(|e| {
                        let v = self.eval(e, &mut state);
                        v.to_expr().to_string()
                    });
                    let mut path = self.snapshot(&state, Outcome::Return);
                    path.ret = ret;
                    self.push(path);
                    return;
                }
                // The symbolic executor walks the blocks the recovery found
                // and has nowhere to go through a computed jump.
                Term::Unreachable | Term::Indirect { .. } => {
                    self.finish(state, Outcome::Unreachable);
                    return;
                }
                Term::Jump(target) => state.block = *target,
                Term::Switch {
                    index, targets, ..
                } => {
                    // A dispatch: a known index picks one case, an unknown one
                    // makes every case a possible path.
                    let value = self.eval(index, &mut state);
                    if !state.missing.is_empty() {
                        let missing = std::mem::take(&mut state.missing);
                        self.finish(state, Outcome::Missing(missing));
                        return;
                    }
                    if let Some(k) = value.as_const() {
                        match targets.get(k as usize) {
                            Some(t) => state.block = *t,
                            None => {
                                self.finish(state, Outcome::Unreachable);
                                return;
                            }
                        }
                    } else {
                        let cases: Vec<BlockId> = targets.clone();
                        for (i, t) in cases.into_iter().enumerate() {
                            let mut other = state.clone();
                            let cond = Expr::Cond {
                                op: CondOp::Eq,
                                lhs: Box::new(index.clone()),
                                rhs: Box::new(Expr::Const(i as u64)),
                                bits: 8,
                            };
                            other.conds.push(Constraint {
                                expr: cond.clone(),
                                truth: true,
                                at: addr,
                            });
                            other.block = t;
                            state.conds.push(Constraint {
                                expr: cond,
                                truth: true,
                                at: addr,
                            });
                            state.block = t;
                            if self.budget == 0 {
                                self.dropped += 1;
                                self.note_once(
                                    "state budget reached; further switch cases were not followed"
                                        .into(),
                                );
                            } else {
                                self.budget -= 1;
                                work.push(other);
                            }
                        }
                    }
                }
                Term::Branch {
                    cond,
                    then_bb,
                    else_bb,
                } => {
                    let value = self.eval(cond, &mut state);
                    if !state.missing.is_empty() {
                        let missing = std::mem::take(&mut state.missing);
                        self.finish(state, Outcome::Missing(missing));
                        return;
                    }
                    match value.truth() {
                        Some(true) => match then_bb {
                            Some(block) => state.block = *block,
                            // The branch left the function on this side.
                            None => {
                                self.finish(state, Outcome::Unreachable);
                                return;
                            }
                        },
                        Some(false) => match else_bb {
                            Some(block) => state.block = *block,
                            None => {
                                self.finish(state, Outcome::Unreachable);
                                return;
                            }
                        },
                        None => {
                            // The condition is unknown, so both arms are
                            // possible. An arm that leaves the function has
                            // no block to continue in, so it is not
                            // followed: the state that would have gone
                            // there ends at the branch.
                            let mut other = None;
                            if let Some(block) = else_bb {
                                let mut alt = state.clone();
                                alt.conds.push(Constraint {
                                    expr: cond.clone(),
                                    truth: false,
                                    at: addr,
                                });
                                alt.block = *block;
                                other = Some(alt);
                            }
                            match then_bb {
                                Some(block) => {
                                    state.conds.push(Constraint {
                                        expr: cond.clone(),
                                        truth: true,
                                        at: addr,
                                    });
                                    state.block = *block;
                                    if let Some(alt) = other {
                                        if self.budget == 0 {
                                            self.dropped += 1;
                                            self.note_once(
                                                "state budget reached; further branches were not followed"
                                                    .into(),
                                            );
                                        } else {
                                            self.budget -= 1;
                                            work.push(alt);
                                        }
                                    }
                                }
                                // The taken arm leaves the function: follow
                                // the other one, and only that one.
                                None => match other {
                                    Some(alt) => state = alt,
                                    None => {
                                        self.finish(state, Outcome::Unreachable);
                                        return;
                                    }
                                },
                            }
                        }
                    }
                }
            }
        }
    }

    /// One statement.
    fn stmt(&mut self, state: &mut State, stmt: &Stmt) {
        match stmt {
            Stmt::Assign { dest, value } => {
                let v = self.eval(value, state);
                if !state.missing.is_empty() {
                    return;
                }
                state.env.insert(dest.clone(), v);
            }
            Stmt::Store { addr, value, size } => {
                let a = self.eval(addr, state);
                let v = self.eval(value, state);
                if !state.missing.is_empty() {
                    return;
                }
                match a.as_const() {
                    Some(at) => state.mem.store(at, *size, &v),
                    // A store through a symbolic pointer would need a memory
                    // that maps expressions; without one it is dropped, and
                    // the note is what keeps that visible.
                    None => {
                        let note = format!(
                            "store through `{}` was not traced (symbolic address)",
                            a.to_expr()
                        );
                        self.note_once(note);
                    }
                }
            }
            Stmt::Call {
                name,
                args,
                ret,
                addr,
            } => {
                let mut values = Vec::with_capacity(args.len());
                for a in args {
                    values.push(self.eval(a, state).to_expr());
                    if !state.missing.is_empty() {
                        return;
                    }
                }
                state.calls.push(CallInfo {
                    name: name.clone(),
                    addr: *addr,
                    args: values,
                });
                if let Some(dest) = ret {
                    // The callee is not interpreted, so its result is an
                    // unconstrained value. Naming it after the call keeps the
                    // printed form readable and the origin obvious.
                    state.env.insert(
                        dest.clone(),
                        Value::Sym(Expr::Unknown(format!("{name}()"))),
                    );
                }
            }
            Stmt::Comment(_) => {}
        }
    }

    /// Evaluate an expression under a state.
    fn eval(&mut self, e: &Expr, state: &mut State) -> Value {
        match e {
            Expr::Const(v) => Value::Const(*v),
            Expr::Var(name) => {
                if let Some(v) = state.env.get(name) {
                    return v.clone();
                }
                match self.cfg.mode {
                    // A name that was never assigned is an input: a parameter
                    // register, or a slot the caller owns.
                    Mode::Symbolic => Value::Sym(Expr::var(name.clone())),
                    Mode::Concrete => {
                        push_missing(state, name);
                        Value::Const(0)
                    }
                }
            }
            Expr::Load { addr, size, signed } => {
                let a = self.eval(addr, state);
                let Some(at) = a.as_const() else {
                    // A load through a symbolic pointer is a real read of
                    // unknown memory, so it stays as a load rather than being
                    // replaced by a made-up value.
                    return Value::Sym(Expr::Load {
                        addr: Box::new(a.to_expr()),
                        size: *size,
                        signed: *signed,
                    });
                };
                if let Some(v) = state.mem.load(at, *size, *signed) {
                    return v;
                }
                if let Some(raw) = (self.read_static)(Addr(at), *size) {
                    return Value::Const(if *signed {
                        sign_fit(raw, *size)
                    } else {
                        raw
                    });
                }
                match self.cfg.mode {
                    Mode::Symbolic => {
                        self.note_once(format!(
                            "read of unmapped 0x{at:x} ({size} byte) stayed symbolic"
                        ));
                        Value::Sym(Expr::Unknown(format!("*(u{}*)0x{at:x}", u32::from(*size) * 8)))
                    }
                    Mode::Concrete => {
                        push_missing(state, &format!("memory at 0x{at:x} ({size} byte)"));
                        Value::Const(0)
                    }
                }
            }
            Expr::Bin { op, lhs, rhs, bits } => {
                let a = self.eval(lhs, state);
                let b = self.eval(rhs, state);
                if let Some(0) = b.as_const()
                    && matches!(op, BinOp::Div | BinOp::UDiv | BinOp::Mod | BinOp::UMod)
                {
                    self.note_once(format!("division by zero on a path at {e}"));
                    return Value::Sym(Expr::Unknown("div by zero".into()));
                }
                Value::bin(*op, &a, &b, *bits)
            }
            Expr::Un { op, arg, bits } => {
                let a = self.eval(arg, state);
                Value::un(*op, &a, *bits)
            }
            Expr::Cond {
                op,
                lhs,
                rhs,
                bits,
            } => {
                let a = self.eval(lhs, state);
                let b = self.eval(rhs, state);
                Value::compare(*op, &a, &b, *bits)
            }
            Expr::Bool { op, args } => {
                let values: Vec<Value> = args.iter().map(|a| self.eval(a, state)).collect();
                Value::bool_combination(*op, &values)
            }
            Expr::FlagCond { cond, state: flags, .. } => {
                // The condition on the flags is answered by the flags.
                match flags.condition(*cond) {
                    Some(e) => self.eval(&e, state),
                    None => Value::Sym(Expr::Unknown("flags".into())),
                }
            }
            Expr::Cast {
                kind,
                arg,
                from,
                to,
            } => {
                let a = self.eval(arg, state);
                Value::cast(*kind, &a, *from, *to)
            }
            Expr::Select { cond, yes, no } => {
                let c = self.eval(cond, state);
                match c.truth() {
                    Some(true) => self.eval(yes, state),
                    Some(false) => self.eval(no, state),
                    None => {
                        let y = self.eval(yes, state);
                        let n = self.eval(no, state);
                        Value::Sym(
                            Expr::Select {
                                cond: Box::new(c.to_expr()),
                                yes: Box::new(y.to_expr()),
                                no: Box::new(n.to_expr()),
                            }
                            .simplify(),
                        )
                    }
                }
            }
            Expr::Phi(_) => {
                self.note_once("a phi survived SSA elimination; its value is unknown".into());
                Value::Sym(Expr::Unknown("phi".into()))
            }
            Expr::Intrinsic { op, args, bits } => {
                // The operation is known but its operands may not be, so the
                // result is the operation itself.
                Value::Sym(
                    Expr::Intrinsic {
                        op: *op,
                        args: args.iter().map(|a| self.eval(a, state).to_expr()).collect(),
                        bits: *bits,
                    }
                    .simplify(),
                )
            }
            Expr::Unknown(text) => Value::Sym(Expr::Unknown(text.clone())),
        }
    }

    fn note_once(&mut self, note: String) {
        if !self.notes.contains(&note) {
            self.notes.push(note);
        }
    }

    fn snapshot(&mut self, state: &State, outcome: Outcome) -> Path {
        let model = if state.conds.is_empty() {
            Model {
                inputs: Vec::new(),
                verdict: Verdict::Satisfied,
                tried: 0,
                note: None,
            }
        } else {
            solve::solve(&state.conds, self.cfg.solve_cap)
        };
        Path {
            outcome,
            conds: state.conds.clone(),
            model,
            blocks: state
                .blocks
                .iter()
                .map(|b| self.program.addr_of(*b))
                .collect(),
            calls: state.calls.clone(),
            ret: None,
        }
    }

    fn finish(&mut self, state: State, outcome: Outcome) {
        let path = self.snapshot(&state, outcome);
        self.push(path);
    }

    fn push(&mut self, path: Path) {
        if self.paths.len() < self.cfg.max_paths {
            self.paths.push(path);
        } else {
            self.dropped += 1;
        }
    }
}

/// Record one input the concrete run still needs, once.
fn push_missing(state: &mut State, name: &str) {
    if !state.missing.iter().any(|m| m == name) {
        state.missing.push(name.to_string());
    }
}

fn sign_fit(raw: u64, size: u8) -> u64 {
    occ_decompile::ir::sign_extend(raw, size) as u64
}
