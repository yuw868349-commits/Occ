//! Where does one input end up?
//!
//! A forward may-analysis over the same block graph the decompiler structures.
//! Each program point carries a set of source labels for every variable and
//! every addressable cell, joins at merges are unions, and the iteration runs
//! to a fixpoint over the strongly-connected parts of the graph.
//!
//! The result is a list of flows, each naming one source, one sink, the argument
//! position when the sink is a call, and a block path from the entry. It answers
//! the question a reader actually has in front of a strange function: *does this
//! argument reach that call, and through what*.
//!
//! Three rules are deliberately generous, and each is a place where a stricter
//! analysis would miss real bugs:
//!
//! * a store through an unknown address taints every later load from an
//!   unknown address, because the analysis cannot prove they differ;
//! * a load through a tainted pointer is tainted, so `*p` inherits `p`;
//! * a call's return value inherits its arguments, because the callee is not
//!   interpreted.
//!
//! All three over-approximate, which is the safe direction: a flow reported is
//! a flow that may exist, while a flow missed would be a false negative presented
//! as a clean bill of health.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use occ_core::{Addr, Function, Image};
use occ_decompile::ir::{BlockId, Expr, Stmt, Term};
use occ_decompile::{Options, Program, build_program};

/// Something the analysis treats as attacker-controlled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// A name in the IR: a parameter register, or any value the function reads
    /// before writing.
    Name(String),
    /// A memory range.
    Memory {
        /// First address.
        addr: Addr,
        /// Width in bytes.
        bytes: u8,
    },
}

impl Source {
    /// Stable label used in output.
    pub fn label(&self) -> String {
        match self {
            Source::Name(n) => format!("name:{n}"),
            Source::Memory { addr, bytes } => format!("mem:{}:{bytes}", addr),
        }
    }
}

/// Which events count as sinks.
#[derive(Clone, Debug)]
pub struct SinkFilter {
    /// Report every call, whatever its name.
    pub any_call: bool,
    /// Call names to report, matched as substrings. Ignored when `any_call`.
    pub calls: Vec<String>,
    /// Report branch conditions that depend on a source.
    pub branches: bool,
    /// Report memory writes whose value or address depends on a source.
    pub stores: bool,
    /// What a callee forwards to, keyed by the callee's own name.
    ///
    /// A call site names the function it reaches, and in a stripped image that
    /// name is `sub_4a1c0`. If the same body then calls `memcpy`, the argument
    /// passed at the call site does reach `memcpy`, and a caller filtering for
    /// it should be told so. Without this map, a name filter can only match
    /// calls that are named at the site, which in a stripped binary is none.
    pub forwards: std::collections::HashMap<String, Vec<String>>,
}

impl Default for SinkFilter {
    fn default() -> Self {
        // Everything on: a caller that wants less says so, and a default that
        // silently hid stores would make "no flows" ambiguous.
        SinkFilter {
            any_call: true,
            calls: Vec::new(),
            branches: true,
            stores: true,
            forwards: std::collections::HashMap::new(),
        }
    }
}

impl SinkFilter {
    /// True when a call to `name` is a sink.
    pub fn wants_call(&self, name: &str) -> bool {
        if self.any_call {
            return true;
        }
        if self.calls.iter().any(|c| name.contains(c.as_str())) {
            return true;
        }
        self.forwards.get(name).is_some_and(|next| {
            next.iter()
                .any(|n| self.calls.iter().any(|c| n.contains(c.as_str())))
        })
    }

    /// What to call this sink in a flow report.
    ///
    /// A call the filter names directly is reported under its own name. A call
    /// that only reaches the requested sink through its body is reported with
    /// the names it forwards to, because `run` says nothing about why a flow
    /// looking for `system` stopped there.
    pub fn sink_label(&self, name: &str) -> String {
        if self.any_call || self.calls.iter().any(|c| name.contains(c.as_str())) {
            return name.to_string();
        }
        let Some(next) = self.forwards.get(name) else {
            return name.to_string();
        };
        let mut hits: Vec<&str> = next
            .iter()
            .map(String::as_str)
            .filter(|n| self.calls.iter().any(|c| n.contains(c.as_str())))
            .collect();
        hits.sort_unstable();
        hits.dedup();
        if hits.is_empty() {
            name.to_string()
        } else {
            format!("{name} -> {}", hits.join(" | "))
        }
    }
}

/// One source-to-sink flow.
#[derive(Clone, Debug)]
pub struct Flow {
    /// Source label.
    pub source: String,
    /// Sink description.
    pub sink: String,
    /// Address of the call, branch or store.
    pub at: Addr,
    /// Argument position, for calls.
    pub arg: Option<usize>,
    /// The value that arrived, printed.
    pub value: String,
    /// Block addresses from the entry to the sink, one per block.
    pub path: Vec<Addr>,
}

/// Result of one taint run.
#[derive(Clone, Debug)]
pub struct Report {
    /// Entry address.
    pub entry: Addr,
    /// Function name.
    pub name: String,
    /// Parameter names the body reads, as the pseudo-C prints them.
    pub params: Vec<String>,
    /// Argument registers for those parameters: what a source may name.
    pub param_regs: Vec<String>,
    /// Sources as they were labelled.
    pub sources: Vec<String>,
    /// Flows found, ordered by address.
    pub flows: Vec<Flow>,
    /// Blocks the analysis visited.
    pub blocks: usize,
    /// Iterations until the fixpoint.
    pub iterations: usize,
    /// False when the iteration cap stopped the analysis early.
    pub converged: bool,
    /// True when a source reaches the value the function returns.
    ///
    /// A caller that follows this call can then treat the return as tainted
    /// without looking at the callee's body again.
    pub ret_tainted: bool,
    /// Notes about approximations.
    pub notes: Vec<String>,
    /// Leftovers from lifting.
    pub lift_notes: Vec<String>,
}

impl Report {
    /// Distinct sink descriptions, for a summary line.
    pub fn sinks(&self) -> Vec<String> {
        let mut out: Vec<String> = self.flows.iter().map(|f| f.sink.clone()).collect();
        out.sort();
        out.dedup();
        out
    }

    /// Distinct source labels that reached something.
    pub fn reached(&self) -> Vec<String> {
        let mut out: Vec<String> = self.flows.iter().map(|f| f.source.clone()).collect();
        out.sort();
        out.dedup();
        out
    }
}

/// Labels are indices into the source list.
type Labels = BTreeSet<u16>;

/// Taint state at one program point.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Taint {
    vars: BTreeMap<String, Labels>,
    mem: BTreeMap<u64, Labels>,
    /// Taint that may live at an address the analysis could not pin down.
    anywhere: Labels,
}

impl Taint {
    /// Union another state into this one; true when something changed.
    fn join(&mut self, other: &Taint) -> bool {
        let mut changed = false;
        for (name, labels) in &other.vars {
            let slot = self.vars.entry(name.clone()).or_default();
            let before = slot.len();
            slot.extend(labels.iter().copied());
            changed |= slot.len() != before;
        }
        for (addr, labels) in &other.mem {
            let slot = self.mem.entry(*addr).or_default();
            let before = slot.len();
            slot.extend(labels.iter().copied());
            changed |= slot.len() != before;
        }
        let before = self.anywhere.len();
        self.anywhere.extend(other.anywhere.iter().copied());
        changed |= self.anywhere.len() != before;
        changed
    }
}

/// Limits.
#[derive(Clone, Debug)]
pub struct Config {
    /// Hard cap on fixpoint rounds.
    pub max_iterations: usize,
    /// Which events are sinks.
    pub sinks: SinkFilter,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_iterations: 64,
            sinks: SinkFilter::default(),
        }
    }
}

/// Run the analysis.
pub fn trace(
    image: &Image,
    func: &Function,
    opts: &Options,
    name_of: &dyn Fn(Addr) -> String,
    sources: &[Source],
    cfg: &Config,
    functions: &[Function],
) -> Report {
    trace_with(image, func, opts, name_of, sources, cfg, functions, 0)
}

/// What one callee does with one tainted argument.
#[derive(Clone, Debug, Default)]
struct CalleeSummary {
    /// The argument reaches the value the callee returns.
    ret: bool,
    /// Sinks the argument reaches, by the label the callee's own run gave
    /// them.
    sinks: Vec<(String, Addr)>,
}

/// State the cross-function part of the analysis carries.
struct Calls<'a> {
    image: &'a Image,
    opts: &'a Options,
    name_of: &'a dyn Fn(Addr) -> String,
    sinks: SinkFilter,
    /// Lifted callees, by entry address.
    programs: HashMap<Addr, std::rc::Rc<Program>>,
    /// Summaries already computed, by callee and argument position.
    cache: HashMap<(Addr, usize), Option<CalleeSummary>>,
    /// Callees being summarised, outermost first.
    active: Vec<Addr>,
    depth: usize,
    /// True once one callee was summarised, for the run's notes.
    used: bool,
}

impl Calls<'_> {
    /// The callee's lifted program, lifted once per run.
    fn program(&mut self, callee: &Function) -> std::rc::Rc<Program> {
        if let Some(p) = self.programs.get(&callee.entry) {
            return p.clone();
        }
        let lifted = std::rc::Rc::new(build_program(self.image, callee, self.opts, self.name_of));
        self.programs.insert(callee.entry, lifted.clone());
        lifted
    }

    /// Summarise `callee` for a tainted argument at `arg`.
    ///
    /// `None` means the callee's body is not available, or that summarising
    /// it would recurse: the caller then keeps the conservative rule that a
    /// tainted argument makes the result tainted.
    fn summary(
        &mut self,
        functions: &[Function],
        callee: &Function,
        arg: usize,
    ) -> Option<CalleeSummary> {
        if self.depth >= 2 || self.active.contains(&callee.entry) {
            return None;
        }
        if let Some(hit) = self.cache.get(&(callee.entry, arg)) {
            return hit.clone();
        }
        let program = self.program(callee);
        let Some(reg) = program.param_regs.get(arg).cloned() else {
            self.cache.insert((callee.entry, arg), None);
            return None;
        };
        self.active.push(callee.entry);
        let report = trace_with(
            self.image,
            callee,
            self.opts,
            self.name_of,
            &[Source::Name(reg)],
            &Config {
                max_iterations: 64,
                sinks: self.sinks.clone(),
            },
            functions,
            self.depth + 1,
        );
        self.active.pop();
        let summary = CalleeSummary {
            ret: report.ret_tainted,
            sinks: report
                .flows
                .iter()
                .map(|f| (f.sink.clone(), f.at))
                .collect(),
        };
        self.used = true;
        self.cache
            .insert((callee.entry, arg), Some(summary.clone()));
        Some(summary)
    }
}

#[allow(clippy::too_many_arguments)]
fn trace_with(
    image: &Image,
    func: &Function,
    opts: &Options,
    name_of: &dyn Fn(Addr) -> String,
    sources: &[Source],
    cfg: &Config,
    functions: &[Function],
    depth: usize,
) -> Report {
    let program = build_program(image, func, opts, name_of);
    let lift_notes = program.notes.clone();
    let mut report = Report {
        entry: func.entry,
        name: func.display_name(),
        params: program.params.clone(),
        param_regs: program.param_regs.clone(),
        sources: sources.iter().map(|s| s.label()).collect(),
        flows: Vec::new(),
        ret_tainted: false,
        blocks: program.blocks.len(),
        iterations: 0,
        converged: true,
        notes: Vec::new(),
        lift_notes,
    };
    if program.blocks.is_empty() {
        report
            .notes
            .push("no blocks: nothing to analyse (is the architecture lifted?)".into());
        return report;
    }
    if sources.is_empty() {
        report
            .notes
            .push("no sources given, so nothing can be tainted".into());
        return report;
    }

    // Seed the entry state with one label per source.
    let mut entry = Taint::default();
    for (i, source) in sources.iter().enumerate() {
        let bit = i as u16;
        match source {
            Source::Name(name) => {
                entry.vars.entry(name.clone()).or_default().insert(bit);
            }
            Source::Memory { addr, .. } => {
                entry.mem.entry(addr.get()).or_default().insert(bit);
            }
        }
    }

    let order = rpo(&program);
    let mut state_in: Vec<Taint> = vec![Taint::default(); program.blocks.len()];
    state_in[program.entry] = entry;
    // Keyed by the flow's identity, and carrying the block it was found in so
    // the witness path points at the block rather than the instruction.
    let mut found: BTreeMap<FlowKey, (Flow, BlockId)> = BTreeMap::new();

    let mut calls = Calls {
        image,
        opts,
        name_of,
        sinks: cfg.sinks.clone(),
        programs: HashMap::new(),
        cache: HashMap::new(),
        active: vec![func.entry],
        depth,
        used: false,
    };

    let mut converged = false;
    for round in 0..cfg.max_iterations.max(1) {
        report.iterations = round + 1;
        let mut changed = false;
        for &b in &order {
            let input = state_in[b].clone();
            if input.vars.is_empty() && input.mem.is_empty() && input.anywhere.is_empty() {
                // Nothing to propagate; the block is unreachable from any source.
                continue;
            }
            let mut out = input.clone();
            transfer(
                &program.blocks[b],
                &mut out,
                sources,
                &cfg.sinks,
                functions,
                &mut calls,
                &mut |flow| {
                    let key = (
                        flow.source.clone(),
                        flow.sink.clone(),
                        flow.at.get(),
                        flow.arg,
                    );
                    // Recording a flow is not a state change: the iteration
                    // stops when the labels stop growing, and a flow found in
                    // round one must not keep the fixpoint alive forever.
                    found.entry(key).or_insert((flow, b));
                },
            );
            for succ in successors(&program.blocks[b].term) {
                if state_in[succ].join(&out) {
                    changed = true;
                }
            }
        }
        if !changed {
            converged = true;
            break;
        }
    }
    report.converged = converged;
    if !converged {
        report.notes.push(format!(
            "iteration cap of {} reached; later blocks may be under-approximated",
            cfg.max_iterations
        ));
    }

    // Whether a source reaches the returned value. Asked after the fixpoint
    // and with a sink that does nothing: this pass is about the value, and
    // re-reporting the same stores would double every flow.
    for b in 0..program.blocks.len() {
        if let occ_decompile::ir::Term::Ret(Some(value)) = &program.blocks[b].term {
            let mut out = state_in[b].clone();
            transfer(
                &program.blocks[b],
                &mut out,
                sources,
                &cfg.sinks,
                functions,
                &mut calls,
                &mut |_| {},
            );
            if !labels_of(value, &out).is_empty() {
                report.ret_tainted = true;
                break;
            }
        }
    }

    let reach = reachable_from_entry(&program);
    let mut flows: Vec<Flow> = Vec::with_capacity(found.len());
    for (mut flow, block) in found.into_values() {
        flow.path = witness_path(&program, &reach, block);
        flows.push(flow);
    }
    flows.sort_by(|a, b| {
        a.at.get()
            .cmp(&b.at.get())
            .then(a.sink.cmp(&b.sink))
            .then(a.source.cmp(&b.source))
    });
    report.flows = flows;
    report
}

/// Propagate one block, reporting sinks through `sink`.
fn transfer(
    block: &occ_decompile::ir::Block,
    state: &mut Taint,
    sources: &[Source],
    filter: &SinkFilter,
    functions: &[Function],
    calls: &mut Calls<'_>,
    sink: &mut dyn FnMut(Flow),
) {
    for stmt in &block.stmts {
        transfer_stmt(
            stmt, block.addr, state, sources, filter, functions, calls, sink,
        );
    }

    if filter.branches
        && let Term::Branch { cond, .. } = &block.term
    {
        let labels = labels_of(cond, state);
        for bit in &labels {
            sink(Flow {
                source: sources[usize::from(*bit)].label(),
                sink: "branch".into(),
                at: block.addr,
                arg: None,
                value: cond.to_string(),
                path: Vec::new(),
            });
        }
    }
}

/// Propagate one statement, descending into the two arms of a conditional.
///
/// A conditional store runs on one path and not the other, so the taint it
/// carries has to reach the sink; joining the states afterwards keeps the
/// variable that was tainted on only one arm tainted on both.
#[allow(clippy::too_many_arguments)]
fn transfer_stmt(
    stmt: &Stmt,
    at: occ_core::Addr,
    state: &mut Taint,
    sources: &[Source],
    filter: &SinkFilter,
    functions: &[Function],
    calls: &mut Calls<'_>,
    sink: &mut dyn FnMut(Flow),
) {
    match stmt {
        Stmt::Assign { dest, value } => {
            let labels = labels_of(value, state);
            state.vars.insert(dest.clone(), labels);
        }
        Stmt::Store { addr, value, size } => {
            let value_labels = labels_of(value, state);
            let addr_labels = labels_of(addr, state);
            if filter.stores && (!value_labels.is_empty() || !addr_labels.is_empty()) {
                let mut all = value_labels.clone();
                all.extend(addr_labels.iter().copied());
                for bit in all {
                    sink(Flow {
                        source: sources[usize::from(bit)].label(),
                        sink: match addr_const(addr) {
                            Some(a) => format!("store to 0x{a:x}"),
                            None => "store through a pointer".into(),
                        },
                        at,
                        arg: None,
                        value: format!("*({size} byte) = {value}"),
                        path: Vec::new(),
                    });
                }
            }
            match addr_const(addr) {
                Some(a) => {
                    state.mem.insert(a, value_labels);
                }
                None => {
                    // The write could land anywhere, and the analysis will
                    // not pretend otherwise.
                    state.anywhere.extend(addr_labels);
                    state.anywhere.extend(value_labels);
                }
            }
        }
        Stmt::Call {
            name,
            args,
            ret,
            addr,
            target,
        } => {
            let mut joined = Labels::new();
            let callee = target.and_then(|t| functions.iter().find(|f| f.entry == t));
            // The call lists only the registers its block wrote, so a
            // call that passes an argument the function never touched
            // reads as a call with no arguments. The callee's own
            // parameter registers are where those arguments live.
            let mut bound: Vec<Expr> = args.clone();
            if bound.is_empty()
                && let Some(c) = callee
            {
                bound = calls.program(c).param_regs.iter().map(Expr::var).collect();
            }
            for (i, a) in bound.iter().enumerate() {
                let labels = labels_of(a, state);
                if labels.is_empty() {
                    continue;
                }
                if filter.wants_call(name) {
                    for bit in &labels {
                        sink(Flow {
                            source: sources[usize::from(*bit)].label(),
                            sink: filter.sink_label(name),
                            at: *addr,
                            arg: Some(i),
                            value: a.to_string(),
                            path: Vec::new(),
                        });
                    }
                }
                // A callee whose body is available says what it does with
                // this argument. Without one, the conservative rule
                // stands: a tainted argument makes the result tainted.
                match callee.and_then(|c| calls.summary(functions, c, i)) {
                    Some(sum) => {
                        if sum.ret {
                            joined.extend(labels.iter().copied());
                        }
                        for (sink_name, at) in &sum.sinks {
                            for bit in &labels {
                                sink(Flow {
                                    source: sources[usize::from(*bit)].label(),
                                    sink: sink_name.clone(),
                                    at: *at,
                                    arg: Some(i),
                                    value: a.to_string(),
                                    path: Vec::new(),
                                });
                            }
                        }
                    }
                    None => joined.extend(labels.iter().copied()),
                }
            }
            if let Some(dest) = ret {
                state.vars.insert(dest.clone(), joined);
            }
        }
        // A comment is not a statement, and treating it as one would let a
        // note about a missing instruction carry data flow.
        Stmt::Comment(_) => {}
        Stmt::If {
            then_body,
            else_body,
            ..
        } => {
            let before = state.clone();
            for s in then_body {
                transfer_stmt(s, at, state, sources, filter, functions, calls, sink);
            }
            let then_state = state.clone();
            *state = before;
            for s in else_body {
                transfer_stmt(s, at, state, sources, filter, functions, calls, sink);
            }
            state.join(&then_state);
        }
    }
}

fn addr_const(e: &Expr) -> Option<u64> {
    match e {
        Expr::Const(v) => Some(*v),
        _ => None,
    }
}

/// Source labels an expression carries.
fn labels_of(e: &Expr, state: &Taint) -> Labels {
    let mut out = Labels::new();
    collect_into(e, state, &mut out);
    out
}

fn collect_into(e: &Expr, state: &Taint, out: &mut Labels) {
    match e {
        Expr::Var(name) => {
            if let Some(labels) = state.vars.get(name) {
                out.extend(labels.iter().copied());
            }
        }
        Expr::Load { addr, .. } => {
            if let Some(a) = addr_const(addr)
                && let Some(labels) = state.mem.get(&a)
            {
                out.extend(labels.iter().copied());
            }
            // A load through a cell the analysis could not resolve may see any
            // value that was written through such a cell.
            out.extend(state.anywhere.iter().copied());
            // And the pointer itself: `*p` inherits `p`.
            collect_into(addr, state, out);
        }
        Expr::Member { base, .. } => collect_into(base, state, out),
        Expr::Bin { lhs, rhs, .. } => {
            collect_into(lhs, state, out);
            collect_into(rhs, state, out);
        }
        Expr::Un { arg, .. } | Expr::Cast { arg, .. } => collect_into(arg, state, out),
        Expr::Cond { lhs, rhs, .. } => {
            collect_into(lhs, state, out);
            collect_into(rhs, state, out);
        }
        Expr::Bool { args, .. } => {
            for arg in args {
                collect_into(arg, state, out);
            }
        }
        Expr::FlagCond { state: flags, .. } => {
            for value in [
                flags.cf.as_ref(),
                flags.zf.as_ref(),
                flags.sf.as_ref(),
                flags.of.as_ref(),
                flags.pf.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                collect_into(value, state, out);
            }
        }
        Expr::Select { cond, yes, no } => {
            collect_into(cond, state, out);
            collect_into(yes, state, out);
            collect_into(no, state, out);
        }
        Expr::Phi(args) => {
            for (_, arg) in args {
                collect_into(arg, state, out);
            }
        }
        Expr::Intrinsic { args, .. } => {
            for arg in args {
                collect_into(arg, state, out);
            }
        }
        Expr::Const(_) | Expr::Unknown(_) => {}
    }
}

fn successors(term: &Term) -> Vec<BlockId> {
    match term {
        Term::Jump(t) => vec![*t],
        Term::Branch {
            then_bb, else_bb, ..
        } => then_bb.iter().chain(else_bb.iter()).copied().collect(),
        _ => Vec::new(),
    }
}

/// Identity of a flow: source label, sink, address, argument index.
type FlowKey = (String, String, u64, Option<usize>);

/// Reverse post-order, entry first, iterating the graph without recursion.
fn rpo(program: &Program) -> Vec<BlockId> {
    let n = program.blocks.len();
    let mut seen = vec![false; n];
    let mut post: Vec<BlockId> = Vec::with_capacity(n);
    let mut stack: Vec<(BlockId, usize)> = vec![(program.entry, 0)];
    seen[program.entry] = true;
    while let Some((b, i)) = stack.pop() {
        let succs = successors(&program.blocks[b].term);
        if i < succs.len() {
            stack.push((b, i + 1));
            let s = succs[i];
            if s < n && !seen[s] {
                seen[s] = true;
                stack.push((s, 0));
            }
        } else {
            post.push(b);
        }
    }
    post.reverse();
    // Blocks unreachable from the entry still get visited, in address order, so
    // a source seeded at one of them is not silently skipped.
    for (b, &hit) in seen.iter().enumerate().take(n) {
        if !hit {
            post.push(b);
        }
    }
    post
}

fn reachable_from_entry(program: &Program) -> Vec<Option<BlockId>> {
    let mut prev = vec![None; program.blocks.len()];
    let mut seen = vec![false; program.blocks.len()];
    let mut queue = VecDeque::new();
    seen[program.entry] = true;
    queue.push_back(program.entry);
    while let Some(b) = queue.pop_front() {
        for s in successors(&program.blocks[b].term) {
            if s < seen.len() && !seen[s] {
                seen[s] = true;
                prev[s] = Some(b);
                queue.push_back(s);
            }
        }
    }
    prev
}

/// A block chain from the entry to `start`.
fn witness_path(program: &Program, prev: &[Option<BlockId>], start: BlockId) -> Vec<Addr> {
    if start >= program.blocks.len() {
        return Vec::new();
    }
    let mut chain = vec![start];
    let mut cur = start;
    while let Some(p) = prev[cur] {
        if chain.len() > program.blocks.len() {
            break;
        }
        chain.push(p);
        cur = p;
    }
    chain.reverse();
    chain.into_iter().map(|b| program.addr_of(b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_union_at_joins() {
        let mut a = Taint::default();
        a.vars.insert("x".into(), [1u16].into_iter().collect());
        let mut b = Taint::default();
        b.vars.insert("x".into(), [2u16].into_iter().collect());
        assert!(a.join(&b));
        assert_eq!(a.vars["x"].len(), 2);
        assert!(!a.join(&b));
    }

    #[test]
    fn a_load_through_a_tainted_pointer_is_tainted() {
        let mut state = Taint::default();
        state.vars.insert("p".into(), [0u16].into_iter().collect());
        let e = Expr::Load {
            addr: Box::new(Expr::var("p")),
            size: 4,
            signed: false,
        };
        assert_eq!(labels_of(&e, &state).len(), 1);
    }

    #[test]
    fn a_wrapper_reports_what_it_forwards_to() {
        // `run` calls `system` in its own body, so a filter looking for
        // `system` matches the call to `run`. Naming the sink `run` alone
        // leaves the reader asking why the flow was reported.
        let mut filter = SinkFilter {
            any_call: false,
            calls: vec!["system".into()],
            branches: false,
            stores: false,
            forwards: std::collections::HashMap::new(),
        };
        filter
            .forwards
            .insert("run".into(), vec!["system@plt".into()]);
        assert!(filter.wants_call("run"));
        assert_eq!(filter.sink_label("run"), "run -> system@plt");
        // A call the filter names itself keeps its own name.
        assert_eq!(filter.sink_label("system@plt"), "system@plt");
    }
}
