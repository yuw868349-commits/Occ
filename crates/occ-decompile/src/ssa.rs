//! SSA construction and the clean-up passes that make the output readable.
//!
//! The algorithm is Braun et al.'s on-the-fly construction: blocks are visited
//! in reverse post-order, a write records a new version in the current block,
//! and a read looks the version up along the dominator chain, creating a phi
//! where two paths disagree. It is linear in practice and, unlike the textbook
//! Cytron version, needs no dominance-frontier precomputation.
//!
//! After the rename three passes run: copy propagation, dead assignment
//! removal, and phi elimination into explicit copies on the incoming edges.

use std::collections::{HashMap, HashSet};

use crate::ir::{Block, BlockId, Expr, Stmt, Term};

/// A control-flow graph ready for SSA.
pub struct Cfg {
    /// Blocks in address order.
    pub blocks: Vec<Block>,
    /// Predecessors of each block.
    pub preds: Vec<Vec<BlockId>>,
    /// Successors of each block.
    pub succs: Vec<Vec<BlockId>>,
}

impl Cfg {
    /// Build the edge lists from the terminators.
    pub fn new(blocks: Vec<Block>) -> Cfg {
        let n = blocks.len();
        let mut succs = vec![Vec::new(); n];
        let mut preds = vec![Vec::new(); n];
        for (i, b) in blocks.iter().enumerate() {
            let out: Vec<BlockId> = match &b.term {
                Term::Jump(t) => vec![*t],
                Term::Branch {
                    then_bb, else_bb, ..
                } => then_bb.iter().chain(else_bb.iter()).copied().collect(),
                Term::Switch { targets, .. } => targets.clone(),
                // A computed jump has no edge the pass can follow, so it has
                // no successor: its target is a value, not a block.
                Term::Ret(_) | Term::Unreachable | Term::Indirect { .. } => Vec::new(),
            };
            for &s in &out {
                if s < n && !succs[i].contains(&s) {
                    succs[i].push(s);
                    preds[s].push(i);
                }
            }
        }
        Cfg {
            blocks,
            preds,
            succs,
        }
    }

    /// Reverse post-order over reachable blocks.
    pub fn rpo(&self, entry: BlockId) -> Vec<BlockId> {
        let mut seen = vec![false; self.blocks.len()];
        let mut post = Vec::new();
        // Iterative DFS: decompiled functions can be large enough to blow a
        // recursive helper's stack.
        let mut stack = vec![(entry, 0usize)];
        seen[entry] = true;
        while let Some((b, i)) = stack.pop() {
            if i < self.succs[b].len() {
                stack.push((b, i + 1));
                let s = self.succs[b][i];
                if !seen[s] {
                    seen[s] = true;
                    stack.push((s, 0));
                }
            } else {
                post.push(b);
            }
        }
        post.reverse();
        post
    }
}

/// Rename every value into SSA form, inserting phi nodes where paths join.
pub fn build(
    blocks: &mut Vec<Block>,
    preds: &[Vec<BlockId>],
    order: &[BlockId],
    entry: BlockId,
) -> Vec<String> {
    let n = blocks.len();
    let mut s = Builder {
        blocks,
        phis: vec![Vec::new(); n],
        counters: HashMap::new(),
        defs: HashMap::new(),
        sealed: HashSet::new(),
        visited: HashSet::new(),
        incomplete: HashMap::new(),
        used: HashSet::new(),
        reading: HashSet::new(),
        notes: Vec::new(),
        preds,
    };
    s.sealed.insert(entry);
    for &b in order {
        s.rename_block(b);
        s.visited.insert(b);
        let succs = s.succs(b);
        for next in succs {
            if s.preds[next].iter().all(|p| s.visited.contains(p)) {
                s.seal_block(next);
            }
        }
    }
    // Unreachable or never-sealed blocks still need their phis resolved.
    for b in 0..n {
        if !s.visited.contains(&b) {
            s.rename_block(b);
        }
        if !s.sealed.contains(&b) {
            s.seal_block(b);
        }
    }
    s.prune_empty_phis();
    let Builder {
        blocks,
        phis,
        notes,
        ..
    } = s;
    for (b, list) in phis.into_iter().enumerate() {
        if list.is_empty() {
            continue;
        }
        let mut stmts = Vec::with_capacity(list.len());
        for (dest, args) in list {
            stmts.push(Stmt::Assign {
                dest,
                value: Expr::Phi(args),
            });
        }
        stmts.extend(std::mem::take(&mut blocks[b].stmts));
        blocks[b].stmts = stmts;
    }
    notes
}

/// Phi candidates for one block: `(variable, [(predecessor, value)])`.
type PhiArgs = Vec<(String, Vec<(BlockId, Expr)>)>;

struct Builder<'a> {
    blocks: &'a mut [Block],
    phis: Vec<PhiArgs>,
    counters: HashMap<String, u32>,
    defs: HashMap<(BlockId, String), Expr>,
    sealed: HashSet<BlockId>,
    visited: HashSet<BlockId>,
    /// Phis created before their block was sealed, as `(variable, phi name)`.
    /// The variable's current definition may have moved on by the time the
    /// block is sealed, so the phi has to be remembered by name.
    incomplete: HashMap<BlockId, Vec<(String, String)>>,
    /// Names read so far in the block being renamed.
    used: HashSet<String>,
    /// `(block, variable)` pairs whose value is being resolved right now.
    ///
    /// A block with one predecessor forwards the read to it, and a cycle of
    /// such blocks forwards forever: the value in a loop is read from the
    /// block the loop came back to. Tracking what is in flight turns that
    /// into a value that is reported as unknown.
    reading: HashSet<(BlockId, String)>,
    /// Things worth telling the reader about.
    notes: Vec<String>,
    preds: &'a [Vec<BlockId>],
}

impl Builder<'_> {
    fn succs(&self, b: BlockId) -> Vec<BlockId> {
        match &self.blocks[b].term {
            crate::ir::Term::Jump(t) => vec![*t],
            crate::ir::Term::Branch {
                then_bb, else_bb, ..
            } => then_bb.iter().chain(else_bb.iter()).copied().collect(),
            crate::ir::Term::Switch { targets, .. } => targets.clone(),
            _ => Vec::new(),
        }
    }

    /// Rename one statement, descending into a conditional body.
    ///
    /// A conditional instruction on ARM writes memory on one path and does
    /// nothing on the other; its body holds no definitions, but the walk is
    /// recursive anyway so that a later lifter cannot put one there and have
    /// it silently escape renaming.
    fn rename_stmt(&mut self, stmt: Stmt, b: BlockId) -> Stmt {
        match stmt {
            Stmt::Assign { dest, value } => {
                let value = self.rewrite(&value, b);
                let version = self.write(b, &dest);
                Stmt::Assign {
                    dest: version,
                    value,
                }
            }
            Stmt::Store { addr, value, size } => Stmt::Store {
                addr: self.rewrite(&addr, b),
                value: self.rewrite(&value, b),
                size,
            },
            Stmt::Call {
                name,
                args,
                ret,
                addr,
                target,
            } => {
                let name = self.rename_in_name(b, &name);
                let args = args.iter().map(|a| self.rewrite(a, b)).collect();
                let ret = ret.map(|r| {
                    let version = self.write(b, &r);
                    // The call result itself is filled in by the renderer;
                    // the name records which call produced it.
                    self.defs.insert((b, r.clone()), Expr::Var(version.clone()));
                    version
                });
                Stmt::Call {
                    name,
                    args,
                    ret,
                    addr,
                    target,
                }
            }
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => Stmt::If {
                cond: self.rewrite(&cond, b),
                then_body: then_body
                    .into_iter()
                    .map(|s| self.rename_stmt(s, b))
                    .collect(),
                else_body: else_body
                    .into_iter()
                    .map(|s| self.rename_stmt(s, b))
                    .collect(),
            },
            Stmt::Comment(c) => Stmt::Comment(c),
        }
    }

    fn rename_block(&mut self, b: BlockId) {
        self.used.clear();
        let stmts = std::mem::take(&mut self.blocks[b].stmts);
        let mut out = Vec::with_capacity(stmts.len());
        for stmt in stmts {
            out.push(self.rename_stmt(stmt, b));
        }
        self.blocks[b].stmts = out;
        let term = self.blocks[b].term.clone();
        self.blocks[b].term = match term {
            crate::ir::Term::Branch {
                cond,
                then_bb,
                else_bb,
            } => crate::ir::Term::Branch {
                cond: self.rewrite(&cond, b),
                then_bb,
                else_bb,
            },
            // The returned value is a read like any other, and has to be
            // renamed or every definition feeding it looks dead.
            crate::ir::Term::Ret(Some(value)) => {
                crate::ir::Term::Ret(Some(self.rewrite(&value, b)))
            }
            crate::ir::Term::Switch {
                index,
                targets,
                first_case,
            } => crate::ir::Term::Switch {
                index: self.rewrite(&index, b),
                targets,
                first_case,
            },
            crate::ir::Term::Indirect { target } => crate::ir::Term::Indirect {
                target: self.rewrite(&target, b),
            },
            other => other,
        };
    }

    /// Rename the registers named inside an indirect call's target.
    ///
    /// The target of a `call rax` is carried as text, so it never passes
    /// through [`Self::rewrite`]: the printed call read `rax` — the value the
    /// register held at the function's entry — however many times the body had
    /// written it since. The identifiers inside the text are the only part
    /// that needs renaming, and they are renamed only when the block already
    /// has a definition for them, so this cannot invent a value.
    fn rename_in_name(&mut self, b: BlockId, name: &str) -> String {
        let Some(inner) = name.strip_prefix("(*").and_then(|r| r.strip_suffix(')')) else {
            return name.to_string();
        };
        let mut out = String::with_capacity(inner.len());
        let mut token = String::new();
        let mut changed = false;
        for c in inner.chars() {
            if c.is_alphanumeric() || c == '_' || c == '.' {
                token.push(c);
                continue;
            }
            if !token.is_empty() {
                changed |= self.rename_token(&mut out, b, &token);
                token.clear();
            }
            out.push(c);
        }
        if !token.is_empty() {
            changed |= self.rename_token(&mut out, b, &token);
        }
        if changed {
            // The identifier is printed as it will be named in the rest of the
            // body, where a version separator is an underscore.
            format!("(*{})", out.replace('.', "_"))
        } else {
            name.to_string()
        }
    }

    fn rename_token(&mut self, out: &mut String, b: BlockId, token: &str) -> bool {
        // Only a name the function defines anywhere is a variable here: the
        // text also holds type names and numbers, and reading those would
        // invent a phi for `u64`.
        let known =
            self.defs.contains_key(&(b, token.to_string())) || self.counters.contains_key(token);
        if !known {
            out.push_str(token);
            return false;
        }
        match self.read(b, token) {
            Expr::Var(v) => {
                out.push_str(&v);
                true
            }
            _ => {
                out.push_str(token);
                false
            }
        }
    }

    fn rewrite(&mut self, e: &Expr, b: BlockId) -> Expr {
        match e {
            Expr::Var(name) => self.read(b, name),
            Expr::Load { addr, size, signed } => Expr::Load {
                addr: Box::new(self.rewrite(addr, b)),
                size: *size,
                signed: *signed,
            },
            Expr::Member {
                base,
                offset,
                size,
                name,
            } => Expr::Member {
                base: Box::new(self.rewrite(base, b)),
                offset: *offset,
                size: *size,
                name: name.clone(),
            },
            Expr::Bin { op, lhs, rhs, bits } => Expr::Bin {
                op: *op,
                lhs: Box::new(self.rewrite(lhs, b)),
                rhs: Box::new(self.rewrite(rhs, b)),
                bits: *bits,
            },
            Expr::Un { op, arg, bits } => Expr::Un {
                op: *op,
                arg: Box::new(self.rewrite(arg, b)),
                bits: *bits,
            },
            Expr::Cond { op, lhs, rhs, bits } => Expr::Cond {
                op: *op,
                lhs: Box::new(self.rewrite(lhs, b)),
                rhs: Box::new(self.rewrite(rhs, b)),
                bits: *bits,
            },
            Expr::Bool { op, args } => Expr::Bool {
                op: *op,
                args: args.iter().map(|a| self.rewrite(a, b)).collect(),
            },
            Expr::FlagCond { cond, state, bits } => Expr::FlagCond {
                cond: *cond,
                state: Box::new(crate::ir::FlagState {
                    cf: state.cf.as_ref().map(|e| self.rewrite(e, b)),
                    zf: state.zf.as_ref().map(|e| self.rewrite(e, b)),
                    sf: state.sf.as_ref().map(|e| self.rewrite(e, b)),
                    of: state.of.as_ref().map(|e| self.rewrite(e, b)),
                    pf: state.pf.as_ref().map(|e| self.rewrite(e, b)),
                }),
                bits: *bits,
            },
            Expr::Cast {
                kind,
                arg,
                from,
                to,
            } => Expr::Cast {
                kind: *kind,
                arg: Box::new(self.rewrite(arg, b)),
                from: *from,
                to: *to,
            },
            Expr::Select { cond, yes, no } => Expr::Select {
                cond: Box::new(self.rewrite(cond, b)),
                yes: Box::new(self.rewrite(yes, b)),
                no: Box::new(self.rewrite(no, b)),
            },
            // Listed rather than caught by a wildcard: a new variant must be
            // renamed like any other, and silently cloning it leaves the values
            // inside it unversioned.
            Expr::Const(_) | Expr::Unknown(_) | Expr::Phi(_) => e.clone(),
            Expr::Intrinsic { op, args, bits } => Expr::Intrinsic {
                op: *op,
                args: args.iter().map(|a| self.rewrite(a, b)).collect(),
                bits: *bits,
            },
        }
    }

    fn read(&mut self, b: BlockId, name: &str) -> Expr {
        let value = match self.defs.get(&(b, name.to_string())) {
            Some(v) => v.clone(),
            None => self.read_recursive(b, name),
        };
        // Track the *version* that was handed out: truncation reuses a phi's
        // name, and doing that after the phi has been read would silently
        // redirect those reads to the new definition.
        if let Expr::Var(v) = &value {
            self.used.insert(v.clone());
        }
        self.used.insert(name.to_string());
        value
    }

    fn read_recursive(&mut self, b: BlockId, name: &str) -> Expr {
        if !self.sealed.contains(&b) {
            let phi = self.new_phi(b, name);
            if let Expr::Var(dest) = &phi {
                self.incomplete
                    .entry(b)
                    .or_default()
                    .push((name.to_string(), dest.clone()));
            }
            return phi;
        }
        let preds = self.preds[b].clone();
        if preds.is_empty() {
            // Function entry: a register still holds its incoming value, which
            // is what the caller passed. A stack slot does not: nothing in this
            // function has written it, so its contents are whatever was left
            // there. Reading it by name produced `local_8_u64.3 =
            // local_8_u64`, which looks like a definition feeding itself and
            // says nothing about where the value came from.
            if name.starts_with("local_") {
                self.note_once(format!(
                    "{name} is read before anything writes it: the frame slot is uninitialised here"
                ));
                return Expr::Unknown("uninitialised slot".into());
            }
            return Expr::Var(name.to_string());
        }
        if preds.len() == 1 {
            let key = (b, name.to_string());
            if !self.reading.insert(key.clone()) {
                // The blocks form a cycle: there is no first definition to
                // read, because every definition reads the next one round.
                // The value there is genuinely unknown, and saying so is the
                // only honest answer.
                self.note_once(format!(
                    "{name} has no first definition: the blocks that define it form a cycle"
                ));
                return Expr::Unknown(format!("{name} in a cycle"));
            }
            let value = self.read(preds[0], name);
            self.reading.remove(&key);
            return value;
        }
        let phi = self.new_phi(b, name);
        if let Expr::Var(dest) = &phi {
            let dest = dest.clone();
            self.add_phi_operands(b, name, &dest);
        }
        phi
    }

    fn new_phi(&mut self, b: BlockId, name: &str) -> Expr {
        let version = self.next_version(name);
        self.phis[b].push((version.clone(), Vec::new()));
        let value = Expr::Var(version.clone());
        self.defs.insert((b, name.to_string()), value.clone());
        value
    }

    fn add_phi_operands(&mut self, b: BlockId, name: &str, dest: &str) {
        let preds = self.preds[b].clone();
        let mut args = Vec::with_capacity(preds.len());
        for p in preds {
            let v = self.read(p, name);
            args.push((p, v));
        }
        if let Some((_, slot)) = self.phis[b].iter_mut().find(|(d, _)| d == dest) {
            slot.extend(args);
        }
    }

    fn seal_block(&mut self, b: BlockId) {
        if !self.sealed.insert(b) {
            return;
        }
        let pending = self.incomplete.remove(&b).unwrap_or_default();
        for (name, dest) in pending {
            self.add_phi_operands(b, &name, &dest);
        }
    }

    /// Drop phis that ended up with no operands because the block never had an
    /// edge that could reach them.
    fn prune_empty_phis(&mut self) {
        for b in 0..self.blocks.len() {
            self.phis[b].retain(|(dest, args)| {
                let keep = !args.is_empty();
                if !keep {
                    // Leave an explicit undefined marker; the read that created
                    // this phi is still in the block.
                    self.blocks[b].stmts.insert(
                        0,
                        Stmt::Assign {
                            dest: dest.clone(),
                            value: Expr::Unknown("undefined".into()),
                        },
                    );
                }
                keep
            });
        }
    }

    /// Record a note, once per distinct text.
    fn note_once(&mut self, note: String) {
        if !self.notes.contains(&note) {
            self.notes.push(note);
        }
    }

    fn write(&mut self, b: BlockId, name: &str) -> String {
        // Braun's truncation: a write to a variable whose current definition in
        // this block is still an operand-less phi fills that phi in instead of
        // adding a second definition. Without this the phi keeps zero operands
        // and the value reads back as undefined.
        // Only when nothing has read it yet: reusing the name of a phi that was
        // already read would make those reads see this write's value instead of
        // the value the phi carried on entry.
        if let Some(Expr::Var(cur)) = self.defs.get(&(b, name.to_string())).cloned()
            && !self.used.contains(&cur)
            && let Some(pos) = self.phis[b]
                .iter()
                .position(|(dest, args)| dest == &cur && args.is_empty())
        {
            self.phis[b].remove(pos);
            return cur;
        }
        let version = self.next_version(name);
        self.defs
            .insert((b, name.to_string()), Expr::Var(version.clone()));
        version
    }

    fn next_version(&mut self, name: &str) -> String {
        let c = self.counters.entry(name.to_string()).or_insert(0);
        *c += 1;
        format!("{name}.{}", *c)
    }
}

/// Collapse phis that only pass a value through.
///
/// Two shapes appear constantly after renaming: a phi whose operand for the
/// back edge is the phi itself (the value did not change on that path), and a
/// phi whose operands are all the same value. Both are the same value under
/// another name, and leaving them in place buries the real data flow under
/// `x_7 = phi(L1: x_7, L2: x_3)` noise.
pub fn simplify_phis(blocks: &mut [Block]) {
    for _ in 0..16 {
        let mut subst: HashMap<String, Expr> = HashMap::new();
        for b in blocks.iter_mut() {
            let mut keep: Vec<Stmt> = Vec::with_capacity(b.stmts.len());
            for stmt in std::mem::take(&mut b.stmts) {
                match stmt {
                    Stmt::Assign {
                        dest,
                        value: Expr::Phi(args),
                    } => {
                        let mut live: Vec<(BlockId, Expr)> = Vec::new();
                        for (pred, v) in args {
                            if v == Expr::Var(dest.clone()) {
                                continue;
                            }
                            if !live.iter().any(|(_, seen)| *seen == v) {
                                live.push((pred, v));
                            }
                        }
                        match live.len() {
                            0 => keep.push(Stmt::Assign {
                                dest,
                                value: Expr::Unknown("undefined".into()),
                            }),
                            1 => {
                                subst.insert(dest, live.remove(0).1);
                            }
                            _ => keep.push(Stmt::Assign {
                                dest,
                                value: Expr::Phi(live),
                            }),
                        }
                    }
                    other => keep.push(other),
                }
            }
            b.stmts = keep;
        }
        if subst.is_empty() {
            return;
        }
        apply_subst(blocks, &subst);
    }
}

/// Rewrite every use of the substituted names.
fn apply_subst(blocks: &mut [Block], subst: &HashMap<String, Expr>) {
    if subst.is_empty() {
        return;
    }
    for b in blocks.iter_mut() {
        for_each_stmt_mut(&mut b.stmts, &mut |s| match s {
            Stmt::Assign { value, .. } => *value = value.substitute(subst),
            Stmt::Store { addr, value, .. } => {
                *addr = addr.substitute(subst);
                *value = value.substitute(subst);
            }
            Stmt::Call { args, .. } => {
                for a in args.iter_mut() {
                    *a = a.substitute(subst);
                }
            }
            Stmt::If { cond, .. } => *cond = cond.substitute(subst),
            Stmt::Comment(_) => {}
        });
        match &mut b.term {
            Term::Branch { cond, .. } => *cond = cond.substitute(subst),
            Term::Switch { index, .. } => *index = index.substitute(subst),
            Term::Ret(Some(v)) => *v = v.substitute(subst),
            Term::Indirect { target } => *target = target.substitute(subst),
            _ => {}
        }
    }
}

/// Replace phi nodes with copies on the incoming edges.
///
/// SSA names are unique, so a sequence of copies cannot clobber a value another
/// copy still needs; no parallel-copy scheduling is required.
///
/// The returned set holds the phi destinations. Those names are defined once per
/// incoming edge, so they are variables in the output, not single-assignment
/// values: folding them away would move a value across the edge it was written
/// on.
pub fn eliminate_phis(blocks: &mut [Block], preds: &[Vec<BlockId>]) -> HashSet<String> {
    let mut moves: HashMap<BlockId, Vec<(String, Expr)>> = HashMap::new();
    let mut opaque: HashSet<String> = HashSet::new();
    for block in blocks.iter_mut() {
        let mut keep = Vec::with_capacity(block.stmts.len());
        for stmt in std::mem::take(&mut block.stmts) {
            match stmt {
                Stmt::Assign {
                    dest,
                    value: Expr::Phi(args),
                } => {
                    if args.is_empty() {
                        // A phi with no operands is an undefined value; keep an
                        // explicit marker rather than dropping it.
                        keep.push(Stmt::Assign {
                            dest,
                            value: Expr::Unknown("undefined".into()),
                        });
                        continue;
                    }
                    opaque.insert(dest.clone());
                    for (p, v) in args {
                        moves.entry(p).or_default().push((dest.clone(), v));
                    }
                }
                other => keep.push(other),
            }
        }
        block.stmts = keep;
    }
    for (p, mut list) in moves {
        // Insert before the terminator, which is implicit: statements come
        // before the block's term, so appending is correct.
        list.sort_by(|a, b| a.0.cmp(&b.0));
        // The copies overwrite the phi names. A terminator condition that reads
        // one of them was written against the value the name held *before* the
        // copy, so it has to be evaluated first or it would read the value meant
        // for the next block.
        let names: Vec<String> = list.iter().map(|(d, _)| d.clone()).collect();
        if term_reads(&blocks[p].term, &names) {
            let tmp = format!("__term{p}");
            let old = match &mut blocks[p].term {
                Term::Branch { cond, .. } => std::mem::replace(cond, Expr::Var(tmp.clone())),
                Term::Switch { index, .. } => std::mem::replace(index, Expr::Var(tmp.clone())),
                Term::Ret(Some(v)) => std::mem::replace(v, Expr::Var(tmp.clone())),
                _ => Expr::Const(0),
            };
            blocks[p].stmts.push(Stmt::Assign {
                dest: tmp.clone(),
                value: old,
            });
            // The temporary must survive copy propagation: folding it back into
            // the terminator would put the read after the very copies it was
            // hoisted above.
            opaque.insert(tmp);
        }
        blocks[p].stmts.extend(
            list.into_iter()
                .map(|(dest, value)| Stmt::Assign { dest, value }),
        );
    }
    let _ = preds;
    opaque
}

/// True when the terminator reads any of `names`.
fn term_reads(term: &Term, names: &[String]) -> bool {
    let mut used = Vec::new();
    match term {
        Term::Branch { cond, .. } => cond.vars(&mut used),
        Term::Switch { index, .. } => index.vars(&mut used),
        Term::Ret(Some(v)) => v.vars(&mut used),
        Term::Indirect { target } => target.vars(&mut used),
        _ => return false,
    }
    used.iter().any(|u| names.contains(u))
}

/// Count how often each SSA name is read.
fn use_counts(blocks: &[Block]) -> HashMap<String, u32> {
    let mut counts: HashMap<String, u32> = HashMap::new();
    let bump = |e: &Expr, counts: &mut HashMap<String, u32>| {
        let mut vars = Vec::new();
        e.vars(&mut vars);
        for v in vars {
            *counts.entry(v).or_insert(0) += 1;
        }
    };
    for b in blocks {
        for_each_stmt(&b.stmts, &mut |s| match s {
            Stmt::Assign { value, .. } => bump(value, &mut counts),
            Stmt::Store { addr, value, .. } => {
                bump(addr, &mut counts);
                bump(value, &mut counts);
            }
            Stmt::Call { args, .. } => {
                for a in args {
                    bump(a, &mut counts);
                }
            }
            Stmt::If { cond, .. } => bump(cond, &mut counts),
            Stmt::Comment(_) => {}
        });
        if let crate::ir::Term::Branch { cond, .. } = &b.term {
            bump(cond, &mut counts);
        }
        if let crate::ir::Term::Switch { index, .. } = &b.term {
            bump(index, &mut counts);
        }
        if let crate::ir::Term::Ret(Some(v)) = &b.term {
            bump(v, &mut counts);
        }
        // The address a computed jump reads is a use like any other: without
        // it, every definition feeding the jump looks dead and the dispatch
        // block renders empty.
        if let crate::ir::Term::Indirect { target } = &b.term {
            bump(target, &mut counts);
        }
    }
    counts
}

/// Count how often each name is assigned.
fn definition_counts(blocks: &[Block]) -> HashMap<String, u32> {
    let mut counts: HashMap<String, u32> = HashMap::new();
    for b in blocks {
        for s in &b.stmts {
            if let Stmt::Assign { dest, .. } = s {
                *counts.entry(dest.clone()).or_insert(0) += 1;
            }
        }
    }
    counts
}

/// Copy propagation followed by dead assignment removal.
///
/// Both passes are repeated until nothing changes: removing a dead assignment
/// can turn another into a copy, and folding a copy can make a third dead.
///
/// `keep_dead` skips the removal, and is for the clients that read the values
/// a block computes rather than the statements it renders: see
/// [`crate::Options::keep_dead`].
pub fn cleanup(blocks: &mut [Block], opaque: &HashSet<String>, keep_dead: bool) {
    for _ in 0..8 {
        let changed = propagate_copies(blocks, opaque);
        let folded = propagate_constants(blocks, opaque);
        let removed = if keep_dead {
            false
        } else {
            remove_dead(blocks)
        };
        if !changed && !folded && !removed {
            break;
        }
    }
}

/// Replace a name with the constant it was assigned.
///
/// A constant has no dependency of its own, so unlike a copy it can replace
/// the name in any block, not only in the one that defines it. That is what
/// turns `x1 = 0x491000; *(u32*)(x1 + 0x988)` into a plain address.
///
/// The name must have exactly one definition: a name the phi elimination
/// wrote on several edges holds a different value on each of them, and
/// replacing it with one constant would pick an edge and hide the rest.
fn propagate_constants(blocks: &mut [Block], opaque: &HashSet<String>) -> bool {
    let defs = definition_counts(blocks);
    let mut consts: HashMap<String, Expr> = HashMap::new();
    for b in blocks.iter() {
        for s in &b.stmts {
            let Stmt::Assign { dest, value } = s else {
                continue;
            };
            let Expr::Const(v) = value else { continue };
            if opaque.contains(dest) || defs.get(dest).copied().unwrap_or(0) != 1 {
                continue;
            }
            consts.insert(dest.clone(), Expr::Const(*v));
        }
    }
    if consts.is_empty() {
        return false;
    }
    let mut changed = false;
    for b in blocks.iter_mut() {
        for_each_stmt_mut(&mut b.stmts, &mut |s| match s {
            Stmt::Assign { value, .. } => {
                let n = value.substitute(&consts).simplify();
                if n != *value {
                    *value = n;
                    changed = true;
                }
            }
            Stmt::Store { addr, value, .. } => {
                let na = addr.substitute(&consts).simplify();
                let nv = value.substitute(&consts).simplify();
                if na != *addr || nv != *value {
                    *addr = na;
                    *value = nv;
                    changed = true;
                }
            }
            Stmt::Call { args, .. } => {
                for a in args.iter_mut() {
                    let n = a.substitute(&consts).simplify();
                    if n != *a {
                        *a = n;
                        changed = true;
                    }
                }
            }
            Stmt::If { cond, .. } => {
                let n = cond.substitute(&consts).simplify();
                if n != *cond {
                    *cond = n;
                    changed = true;
                }
            }
            Stmt::Comment(_) => {}
        });
        match &mut b.term {
            Term::Branch { cond, .. } => {
                let n = cond.substitute(&consts).simplify();
                if n != *cond {
                    *cond = n;
                    changed = true;
                }
            }
            Term::Switch { index, .. } => {
                let n = index.substitute(&consts).simplify();
                if n != *index {
                    *index = n;
                    changed = true;
                }
            }
            Term::Ret(Some(v)) => {
                let n = v.substitute(&consts).simplify();
                if n != *v {
                    *v = n;
                    changed = true;
                }
            }
            Term::Indirect { target } => {
                let n = target.substitute(&consts).simplify();
                if n != *target {
                    *target = n;
                    changed = true;
                }
            }
            _ => {}
        }
    }
    changed
}

fn propagate_copies(blocks: &mut [Block], opaque: &HashSet<String>) -> bool {
    let total = use_counts(blocks);
    let mut changed = false;
    for block in blocks.iter_mut() {
        let local = use_counts(std::slice::from_ref(block));
        // Two sound cases: the copy feeds exactly one site anywhere, or every
        // use of it lives in the block that defines it. The second case is what
        // phi elimination produces, where the same name is defined once per
        // incoming edge and read once per edge.
        let mut subst: HashMap<String, Expr> = HashMap::new();
        for s in &block.stmts {
            let Stmt::Assign { dest, value } = s else {
                continue;
            };
            let Expr::Var(src) = value else { continue };
            if dest == src || subst.contains_key(src) || opaque.contains(dest) {
                continue;
            }
            let here = local.get(dest).copied().unwrap_or(0);
            let everywhere = total.get(dest).copied().unwrap_or(0);
            if everywhere == 1 || (here > 0 && here == everywhere) {
                subst.insert(dest.clone(), Expr::Var(src.clone()));
            }
        }
        if subst.is_empty() {
            continue;
        }
        for_each_stmt_mut(&mut block.stmts, &mut |s| match s {
            Stmt::Assign { dest, value } => {
                // The definition stays; readers are rewritten and the
                // dead pass removes it once nothing refers to it.
                if !subst.contains_key(dest) {
                    let new = value.substitute(&subst);
                    if new != *value {
                        *value = new;
                        changed = true;
                    }
                }
            }
            Stmt::Store { addr, value, .. } => {
                let na = addr.substitute(&subst);
                let nv = value.substitute(&subst);
                if na != *addr || nv != *value {
                    *addr = na;
                    *value = nv;
                    changed = true;
                }
            }
            Stmt::Call { args, .. } => {
                for a in args.iter_mut() {
                    let n = a.substitute(&subst);
                    if n != *a {
                        *a = n;
                        changed = true;
                    }
                }
            }
            Stmt::If { cond, .. } => {
                let n = cond.substitute(&subst);
                if n != *cond {
                    *cond = n;
                    changed = true;
                }
            }
            Stmt::Comment(_) => {}
        });
        match &mut block.term {
            Term::Branch { cond, .. } => {
                let n = cond.substitute(&subst);
                if n != *cond {
                    *cond = n;
                    changed = true;
                }
            }
            Term::Indirect { target } => {
                let n = target.substitute(&subst);
                if n != *target {
                    *target = n;
                    changed = true;
                }
            }
            Term::Switch { index, .. } => {
                let n = index.substitute(&subst);
                if n != *index {
                    *index = n;
                    changed = true;
                }
            }
            Term::Ret(Some(v)) => {
                let n = v.substitute(&subst);
                if n != *v {
                    *v = n;
                    changed = true;
                }
            }
            _ => {}
        }
    }
    if changed {
        remove_dead(blocks);
    }
    changed
}

/// Fold a copy that feeds the block's terminator into the terminator itself:
/// `x = y; return x;` becomes `return y;`. Only when `x` is read once, so no
/// other path depends on the name.
pub fn fold_tail_copies(blocks: &mut [Block], opaque: &HashSet<String>) -> bool {
    let counts = use_counts(blocks);
    let mut changed = false;
    for b in blocks.iter_mut() {
        let Some(last) = b.stmts.last() else { continue };
        let Stmt::Assign {
            dest,
            value: Expr::Var(src),
        } = last
        else {
            continue;
        };
        if opaque.contains(dest) || counts.get(dest).copied().unwrap_or(0) != 1 {
            continue;
        }
        let dest_clone = dest.clone();
        let replace = |e: &Expr| -> Option<Expr> {
            if *e == Expr::Var(dest_clone.clone()) {
                Some(Expr::Var(src.clone()))
            } else {
                None
            }
        };
        let mut new_term = None;
        match &b.term {
            Term::Ret(Some(v)) => new_term = replace(v).map(|e| Term::Ret(Some(e))),
            Term::Indirect { target } => {
                new_term = replace(target).map(|t| Term::Indirect { target: t });
            }
            Term::Switch {
                index,
                targets,
                first_case,
            } => {
                if let Some(i) = replace(index) {
                    new_term = Some(Term::Switch {
                        index: i,
                        targets: targets.clone(),
                        first_case: *first_case,
                    });
                }
            }
            Term::Branch {
                cond,
                then_bb,
                else_bb,
            } => {
                if let Some(c) = replace(cond) {
                    new_term = Some(Term::Branch {
                        cond: c,
                        then_bb: *then_bb,
                        else_bb: *else_bb,
                    });
                }
            }
            _ => {}
        }
        if let Some(t) = new_term {
            b.term = t;
            b.stmts.pop();
            changed = true;
        }
    }
    changed
}

fn remove_dead(blocks: &mut [Block]) -> bool {
    let counts = use_counts(blocks);
    let mut removed = false;
    for b in blocks.iter_mut() {
        let before = b.stmts.len();
        // Stores and calls stay: their effect is not a value anyone has to
        // read back.
        b.stmts.retain(|s| match s {
            Stmt::Assign { dest, .. } => counts.get(dest).copied().unwrap_or(0) > 0,
            _ => true,
        });
        if b.stmts.len() != before {
            removed = true;
        }
    }
    removed
}

/// Merge a phi copy back into the definition that feeds it.
///
/// Phi elimination leaves `x.2 = f(x.1)` followed by `x.1 = x.2` at the end of a
/// loop body, which is correct but reads as if the loop had two variables. When
/// the copy source is defined in the same block and nothing else can observe the
/// intermediate name, renaming the definition to the phi name turns that pair
/// into a single update: `x.1 = f(x.1)`.
pub fn coalesce(blocks: &mut [Block], opaque: &HashSet<String>) -> bool {
    let total = use_counts(blocks);
    let mut changed = false;
    for block in blocks.iter_mut() {
        // Phi copies are appended after the block's own statements.
        let mut first_copy = block.stmts.len();
        while first_copy > 0 {
            match &block.stmts[first_copy - 1] {
                Stmt::Assign {
                    dest,
                    value: Expr::Var(_),
                } if opaque.contains(dest) => first_copy -= 1,
                _ => break,
            }
        }
        for idx in first_copy..block.stmts.len() {
            let Stmt::Assign {
                dest,
                value: Expr::Var(src),
            } = &block.stmts[idx]
            else {
                continue;
            };
            let (dest, src) = (dest.clone(), src.clone());
            if dest == src {
                continue;
            }
            let Some(def) = block
                .stmts
                .iter()
                .take(idx)
                .rposition(|s| matches!(s, Stmt::Assign { dest: d, .. } if *d == src))
            else {
                continue;
            };
            // The intermediate name must not escape this block.
            let mut here = 0u32;
            let mut elsewhere = false;
            for (j, s) in block.stmts.iter().enumerate() {
                let mut uses = Vec::new();
                stmt_uses(s, &mut uses);
                let n = uses.iter().filter(|u| **u == src).count() as u32;
                if n > 0 {
                    if j <= idx {
                        here += n;
                    } else {
                        elsewhere = true;
                    }
                }
            }
            // The terminator runs after the copy, so its reads count as local.
            let mut tail = Vec::new();
            match &block.term {
                Term::Branch { cond, .. } => cond.vars(&mut tail),
                Term::Switch { index, .. } => index.vars(&mut tail),
                Term::Ret(Some(v)) => v.vars(&mut tail),
                Term::Indirect { target } => target.vars(&mut tail),
                _ => {}
            }
            here += tail.iter().filter(|u| **u == src).count() as u32;
            if elsewhere || here != total.get(&src).copied().unwrap_or(0) {
                continue;
            }
            // The phi name must not be read between the definition and the copy,
            // or the rename would move the value across those reads.
            let mut blocked = false;
            for s in block.stmts.iter().take(idx).skip(def + 1) {
                let mut uses = Vec::new();
                stmt_uses(s, &mut uses);
                if uses.contains(&dest) {
                    blocked = true;
                    break;
                }
            }
            if blocked {
                continue;
            }
            if let Stmt::Assign { dest: d, .. } = &mut block.stmts[def] {
                *d = dest.clone();
            }
            for s in block.stmts.iter_mut().take(idx).skip(def + 1) {
                rename_in_stmt(s, &src, &dest);
            }
            // The copy is about to disappear, so anything the terminator read
            // from the intermediate name has to read the phi name instead.
            let mut map = HashMap::new();
            map.insert(src.clone(), Expr::Var(dest.clone()));
            match &mut block.term {
                Term::Branch { cond, .. } => *cond = cond.substitute(&map),
                Term::Switch { index, .. } => *index = index.substitute(&map),
                Term::Ret(Some(v)) => *v = v.substitute(&map),
                Term::Indirect { target } => *target = target.substitute(&map),
                _ => {}
            }
            block.stmts.remove(idx);
            changed = true;
            break;
        }
    }
    changed
}

/// Visit every statement, descending into the bodies of conditional ones.
fn for_each_stmt(stmts: &[Stmt], f: &mut dyn FnMut(&Stmt)) {
    for s in stmts {
        f(s);
        if let Stmt::If {
            then_body,
            else_body,
            ..
        } = s
        {
            for_each_stmt(then_body, f);
            for_each_stmt(else_body, f);
        }
    }
}

/// The mutable form. The visitor sees the conditional statement itself and is
/// responsible for its condition; the bodies are visited for it.
fn for_each_stmt_mut(stmts: &mut [Stmt], f: &mut dyn FnMut(&mut Stmt)) {
    for s in stmts {
        f(s);
        if let Stmt::If {
            then_body,
            else_body,
            ..
        } = s
        {
            for_each_stmt_mut(then_body, f);
            for_each_stmt_mut(else_body, f);
        }
    }
}

/// Every name read by a statement.
fn stmt_uses(s: &Stmt, out: &mut Vec<String>) {
    match s {
        Stmt::Assign { value, .. } => value.vars(out),
        Stmt::Store { addr, value, .. } => {
            addr.vars(out);
            value.vars(out);
        }
        Stmt::Call { args, .. } => {
            for a in args {
                a.vars(out);
            }
        }
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => {
            cond.vars(out);
            for_each_stmt(then_body, &mut |s| stmt_uses(s, out));
            for_each_stmt(else_body, &mut |s| stmt_uses(s, out));
        }
        Stmt::Comment(_) => {}
    }
}

/// Rename one variable inside a statement.
fn rename_in_stmt(s: &mut Stmt, from: &str, to: &str) {
    let mut map = HashMap::new();
    map.insert(from.to_string(), Expr::Var(to.to_string()));
    match s {
        Stmt::Assign { value, .. } => *value = value.substitute(&map),
        Stmt::Store { addr, value, .. } => {
            *addr = addr.substitute(&map);
            *value = value.substitute(&map);
        }
        Stmt::Call { args, .. } => {
            for a in args.iter_mut() {
                *a = a.substitute(&map);
            }
        }
        Stmt::If { cond, .. } => *cond = cond.substitute(&map),
        Stmt::Comment(_) => {}
    }
    if let Stmt::If {
        then_body,
        else_body,
        ..
    } = s
    {
        for_each_stmt_mut(then_body, &mut |s| rename_in_stmt_body(s, from, to));
        for_each_stmt_mut(else_body, &mut |s| rename_in_stmt_body(s, from, to));
    }
}

/// `rename_in_stmt` without the descent, for the visitor above: the walk
/// already reaches every nested statement.
fn rename_in_stmt_body(s: &mut Stmt, from: &str, to: &str) {
    let mut map = HashMap::new();
    map.insert(from.to_string(), Expr::Var(to.to_string()));
    match s {
        Stmt::Assign { value, .. } => *value = value.substitute(&map),
        Stmt::Store { addr, value, .. } => {
            *addr = addr.substitute(&map);
            *value = value.substitute(&map);
        }
        Stmt::Call { args, .. } => {
            for a in args.iter_mut() {
                *a = a.substitute(&map);
            }
        }
        Stmt::If { cond, .. } => *cond = cond.substitute(&map),
        Stmt::Comment(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Block, Expr, Stmt, Term};

    /// A builder over a graph whose blocks form a ring, each with exactly one
    /// predecessor.
    ///
    /// `0`'s predecessor is `2`, `1`'s is `0` and `2`'s is `1`. A block with
    /// one predecessor forwards the read to it, so reading a variable in `0`
    /// walks `0 -> 2 -> 1 -> 0` and, before the cycle was noticed, never
    /// came back.
    fn ring_builder<'a>(blocks: &'a mut Vec<Block>, preds: &'a [Vec<BlockId>]) -> Builder<'a> {
        let n = blocks.len();
        Builder {
            blocks,
            phis: vec![Vec::new(); n],
            counters: HashMap::new(),
            defs: HashMap::new(),
            sealed: HashSet::new(),
            visited: HashSet::new(),
            incomplete: HashMap::new(),
            used: HashSet::new(),
            reading: HashSet::new(),
            notes: Vec::new(),
            preds,
        }
    }

    #[test]
    fn a_ring_of_single_predecessors_terminates() {
        let mut blocks = vec![
            Block {
                id: 0,
                addr: occ_core::Addr(0x1000),
                stmts: Vec::new(),
                term: Term::Jump(1),
            },
            Block {
                id: 1,
                addr: occ_core::Addr(0x1010),
                stmts: Vec::new(),
                term: Term::Jump(2),
            },
            Block {
                id: 2,
                addr: occ_core::Addr(0x1020),
                stmts: vec![Stmt::Assign {
                    dest: "rax".into(),
                    value: Expr::Const(1),
                }],
                term: Term::Jump(0),
            },
        ];
        let preds = vec![vec![2usize], vec![0], vec![1]];
        let mut b = ring_builder(&mut blocks, &preds);
        b.sealed.insert(0);
        b.sealed.insert(1);
        b.sealed.insert(2);
        let value = b.read(0, "rax");
        // The value in a ring has no first definition; it is reported as
        // unknown rather than followed forever.
        assert!(
            matches!(&value, Expr::Unknown(_)),
            "a value in a ring is unknown, got {value:?}"
        );
        assert!(
            b.notes.iter().any(|n| n.contains("cycle")),
            "the cycle is reported: {:?}",
            b.notes
        );
    }
}
