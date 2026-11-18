//! Recovering structured control flow from the block graph.
//!
//! The output is C-shaped: `if`/`else`, `while`, `do`/`while`, `break`,
//! `continue`, `return`. That covers what compilers emit for C, C++ and Rust at
//! every optimisation level in practice. Irreducible graphs — mostly hand
//! written assembly and obfuscated binaries — fall back to labelled `goto`, and
//! the count of those is reported rather than hidden.
//!
//! Two analyses do the work: dominators (to find loops and their headers) and
//! post-dominators (to find where the two arms of a branch rejoin).

use std::collections::{HashMap, HashSet};

use crate::ir::{Block, BlockId, Expr, Node, Stmt, Term};

/// Control-flow facts the structurer needs.
pub struct Graph {
    /// Immediate dominator of each block, itself for the entry.
    pub idom: Vec<BlockId>,
    /// Immediate post-dominator; `None` when control does not rejoin.
    pub ipdom: Vec<Option<BlockId>>,
    /// Natural loops by header: every block in the loop.
    pub loops: HashMap<BlockId, HashSet<BlockId>>,
}

impl Graph {
    /// Compute dominators, post-dominators and natural loops.
    pub fn new(blocks: &[Block], preds: &[Vec<BlockId>], succs: &[Vec<BlockId>], entry: BlockId) -> Graph {
        let idom = dominators(blocks.len(), preds, entry);
        let ipdom = post_dominators(blocks, succs);
        let mut loops: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();
        for (u, block) in blocks.iter().enumerate() {
            for target in successors(block) {
                if dominates(&idom, target, u) {
                    // Back edge u -> target: the natural loop is target plus
                    // every block that reaches u without passing through it.
                    let mut set = HashSet::new();
                    set.insert(target);
                    let mut stack = vec![u];
                    while let Some(x) = stack.pop() {
                        if !set.insert(x) {
                            continue;
                        }
                        for p in &preds[x] {
                            if *p != target {
                                stack.push(*p);
                            }
                        }
                    }
                    loops.entry(target).or_default().extend(set);
                }
            }
        }
        Graph {
            idom,
            ipdom,
            loops,
        }
    }

    /// True when `a` dominates `b`.
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        dominates(&self.idom, a, b)
    }
}

/// Successors of a block, as the structurer sees them.
pub fn succs_of(b: &Block) -> Vec<BlockId> {
    successors(b)
}

fn successors(b: &Block) -> Vec<BlockId> {
    match &b.term {
        Term::Jump(t) => vec![*t],
        Term::Branch { then_bb, else_bb, .. } => {
            then_bb.iter().chain(else_bb.iter()).copied().collect()
        }
        Term::Switch { targets, .. } => targets.clone(),
        _ => Vec::new(),
    }
}

/// Cooper-Harvey-Kennedy dominators over a reverse post-order.
fn dominators(n: usize, preds: &[Vec<BlockId>], entry: BlockId) -> Vec<BlockId> {
    let mut order = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    let mut stack = vec![(entry, 0usize)];
    seen[entry] = true;
    // Build a successor list from preds so the DFS does not need the CFG twice.
    let mut succs = vec![Vec::new(); n];
    for (b, ps) in preds.iter().enumerate() {
        for p in ps {
            succs[*p].push(b);
        }
    }
    while let Some((b, i)) = stack.pop() {
        if i < succs[b].len() {
            stack.push((b, i + 1));
            let s = succs[b][i];
            if !seen[s] {
                seen[s] = true;
                stack.push((s, 0));
            }
        } else {
            order.push(b);
        }
    }
    order.reverse();

    let mut rpo = vec![usize::MAX; n];
    for (i, b) in order.iter().enumerate() {
        rpo[*b] = i;
    }
    let mut idom = vec![usize::MAX; n];
    idom[entry] = entry;
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &order {
            if b == entry {
                continue;
            }
            let mut new = usize::MAX;
            for p in &preds[b] {
                if idom[*p] == usize::MAX {
                    continue;
                }
                new = if new == usize::MAX {
                    *p
                } else {
                    intersect(&idom, &rpo, new, *p)
                };
            }
            if new != usize::MAX && idom[b] != new {
                idom[b] = new;
                changed = true;
            }
        }
    }
    // Unreachable blocks dominate only themselves.
    for (b, d) in idom.iter_mut().enumerate() {
        if *d == usize::MAX {
            *d = b;
        }
    }
    idom
}

fn intersect(idom: &[BlockId], rpo: &[usize], mut a: BlockId, mut b: BlockId) -> BlockId {
    while a != b {
        // The reverse post-order numbers the entry lowest, so the deeper
        // node is the one with the larger number, and the deeper node is
        // the one that walks up. Walking the shallower one instead climbs
        // past the meeting point and lands on a block that dominates both
        // — or, in the reversed graph the post-dominators use, on the
        // virtual exit, which then reads as "control never rejoins here".
        while rpo[a] > rpo[b] {
            let d = idom[a];
            if d == a {
                return a;
            }
            a = d;
        }
        while rpo[b] > rpo[a] {
            let d = idom[b];
            if d == b {
                return b;
            }
            b = d;
        }
    }
    a
}

/// Post-dominators, computed on the reversed graph from a virtual exit.
///
/// The reversed graph has an edge `u -> v` for every original edge `v -> u`,
/// so the *predecessors* of a block in it are the block's original
/// successors. Passing the original predecessor lists here instead computed
/// dominators of the original graph from a root nothing reaches, which left
/// every post-dominator undefined — and a branch with no join is a branch
/// the structurer cannot close, so it fell back to `goto` for joins that
/// were perfectly ordinary.
fn post_dominators(blocks: &[Block], succs: &[Vec<BlockId>]) -> Vec<Option<BlockId>> {
    let n = blocks.len();
    let mut rev_preds = vec![Vec::new(); n];
    for (b, ss) in succs.iter().enumerate() {
        if ss.is_empty() {
            // A virtual exit at index `n` with an edge from every real exit.
            rev_preds[b].push(n);
        }
        for s in ss {
            rev_preds[b].push(*s);
        }
    }
    let idom = dominators(n + 1, &rev_preds, n);
    idom.iter()
        .take(n)
        .enumerate()
        .map(|(b, &d)| if d == b || d == n { None } else { Some(d) })
        .collect()
}

fn dominates(idom: &[BlockId], a: BlockId, b: BlockId) -> bool {
    let mut cur = b;
    for _ in 0..4096 {
        if cur == a {
            return true;
        }
        let next = idom[cur];
        if next == cur {
            return false;
        }
        cur = next;
    }
    false
}

/// Context for one region being structured.
struct Region {
    /// Block that ends the region, if any.
    stop: Option<BlockId>,
    /// Header of the innermost loop, so a jump back becomes `continue`.
    loop_header: Option<BlockId>,
    /// Exit blocks of the innermost loop: a branch there becomes `break`.
    loop_exits: HashSet<BlockId>,
}

/// The structurer itself.
pub struct Structurer<'a> {
    blocks: &'a [Block],
    graph: &'a Graph,
    /// Blocks already placed in the output, so a second visit becomes a label.
    placed: HashSet<BlockId>,
    /// Blocks a `goto` points at, in the order they were first needed.
    ///
    /// A second visit to a block the structure already emitted cannot be
    /// inlined again without repeating its statements, so it becomes a jump.
    /// The jump needs a target: the block is emitted once more at the end of
    /// the function, under a label, and the jump goes there.
    goto_targets: Vec<BlockId>,
    /// Blocks that must be reached by a jump rather than inlined.
    ///
    /// The first pass over the function finds out which blocks the structure
    /// has to reach twice. The second pass treats them as boundaries: the
    /// block is emitted once, under a label, and every path that reaches it
    /// jumps there. Without the second pass the block appears twice — once
    /// inlined into the path that got there first, once at the end — and the
    /// reader has to work out that the two copies are the same code.
    boundary: HashSet<BlockId>,
    /// Blocks whose labelled copy is being emitted right now.
    ///
    /// A block inside its own copy is not a boundary: it is already there.
    expanding: HashSet<BlockId>,
    /// Number of blocks that needed a `goto`.
    pub unstructured: usize,
    /// Notes about approximations.
    pub notes: Vec<String>,
}

impl Region {
    /// The outermost context: no stop block, no loop.
    fn root() -> Region {
        Region {
            stop: None,
            loop_header: None,
            loop_exits: HashSet::new(),
        }
    }
}

impl<'a> Structurer<'a> {
    /// Build a structurer over a graph.
    pub fn new(blocks: &'a [Block], graph: &'a Graph) -> Self {
        Structurer {
            blocks,
            graph,
            placed: HashSet::new(),
            goto_targets: Vec::new(),
            boundary: HashSet::new(),
            expanding: HashSet::new(),
            unstructured: 0,
            notes: Vec::new(),
        }
    }

    /// Structure the whole function starting at `entry`.
    pub fn run(&mut self, entry: BlockId) -> Vec<Node> {
        // First pass: find out which blocks the structure has to reach twice.
        // The result is thrown away; only the set of shared blocks matters.
        let _ = self.region(entry, &Region::root());
        self.boundary = self.goto_targets.iter().copied().collect();

        // Second pass: a shared block is labelled where it already sits on
        // the first path that reaches it, and every later path jumps to that
        // label. The first version emitted the block twice — once in place,
        // once at the end under a label — and the copy of a loop header read
        // `L1: goto L1`, a spin loop the original code did not contain.
        self.placed.clear();
        self.goto_targets.clear();
        self.unstructured = 0;
        let mut body = self.region(entry, &Region::root());

        // A shared block that no path reached still needs a definition: a
        // jump to a label that is never defined cannot be compiled, and the
        // reader cannot follow it either.
        let missing: Vec<BlockId> = self
            .boundary
            .iter()
            .copied()
            .filter(|b| !self.placed.contains(b))
            .collect();
        for b in missing {
            self.expanding.insert(b);
            body.push(Node::Label(label(b)));
            body.extend(self.region(b, &Region::root()));
            self.expanding.remove(&b);
        }
        body
    }

    /// Label a shared block the first time control reaches it.
    ///
    /// Returns `false` when the block was already emitted, in which case a
    /// `goto` has been written and the caller must stop walking this path.
    fn label_once(&mut self, b: BlockId, nodes: &mut Vec<Node>) -> bool {
        if !self.boundary.contains(&b) || self.expanding.contains(&b) {
            return true;
        }
        if self.placed.contains(&b) {
            nodes.push(Node::Goto(label(b)));
            self.unstructured += 1;
            return false;
        }
        nodes.push(Node::Label(label(b)));
        true
    }

    fn region(&mut self, start: BlockId, ctx: &Region) -> Vec<Node> {
        let mut nodes = Vec::new();
        let mut cur = Some(start);
        let mut guard = 0usize;
        let mut entered = false;
        while let Some(b) = cur {
            guard += 1;
            if guard > 200_000 {
                self.notes.push("structuring stopped after 200k blocks".into());
                break;
            }
            if Some(b) == ctx.stop {
                break;
            }
            // A block that walks back into the loop header is a `continue`,
            // whether it got there by a jump or by falling out of a branch
            // that returned the header as its continuation.
            if entered && ctx.loop_header == Some(b) {
                nodes.push(Node::Continue);
                break;
            }
            entered = true;
            // A loop is structured as one unit *before* its header's statements
            // are emitted. Emitting the header first would leave every value it
            // computes outside the loop that uses it.
            if ctx.loop_header != Some(b) && self.graph.loops.contains_key(&b) {
                cur = self.structure_loop(b, &mut nodes);
                continue;
            }
            if !self.label_once(b, &mut nodes) {
                break;
            }
            if !self.placed.insert(b) {
                // Already emitted earlier: this is a join the structure could
                // not express, so name it. The label is defined at the end of
                // the function, where the block is emitted a second time — a
                // jump to a label that is never defined is worse than no
                // structure at all, because the reader cannot follow it.
                if !self.goto_targets.contains(&b) {
                    self.goto_targets.push(b);
                }
                nodes.push(Node::Goto(label(b)));
                self.unstructured += 1;
                break;
            }
            let block = &self.blocks[b];
            nodes.extend(self.stmts_to_nodes(&block.stmts));
            match block.term.clone() {
                Term::Ret(value) => {
                    nodes.push(Node::Return(value));
                    break;
                }
                Term::Unreachable => break,
                Term::Indirect { target } => {
                    nodes.push(Node::ComputedGoto(target));
                    break;
                }
                Term::Jump(t) => {
                    if ctx.loop_header == Some(t) && ctx.stop != Some(t) {
                        nodes.push(Node::Continue);
                        break;
                    }
                    // Leaving the loop through its exit block is a `break`;
                    // letting `cur` walk into the exit would emit the code
                    // after the loop inside the loop body.
                    if ctx.loop_header.is_some() && ctx.loop_exits.contains(&t) {
                        nodes.push(Node::Break);
                        break;
                    }
                    cur = Some(t);
                }
                Term::Branch {
                    cond,
                    then_bb,
                    else_bb,
                } => {
                    cur = self.branch(b, cond, then_bb, else_bb, ctx, &mut nodes);
                }
                Term::Switch {
                    index,
                    targets,
                    first_case,
                } => {
                    let (node, join) = self.switch(index, &targets, first_case, ctx);
                    nodes.push(node);
                    cur = join;
                }
            }
        }
        nodes
    }

    /// Structure a multi-way branch into a `switch`.
    ///
    /// Each case is structured on its own. When every case ends by jumping to
    /// the same block, that block is the join the cases share: the cases stop
    /// in front of it and it is emitted once after the switch, which is what
    /// the compiler's layout means. Without that, the first case would consume
    /// the join and the rest would fall back to `goto`.
    fn switch(
        &mut self,
        index: Expr,
        targets: &[BlockId],
        first_case: Option<i64>,
        ctx: &Region,
    ) -> (Node, Option<BlockId>) {
        let join = common_successor(self.blocks, targets);
        let inner = Region {
            stop: join,
            loop_header: ctx.loop_header,
            loop_exits: ctx.loop_exits.clone(),
        };
        let mut cases = Vec::with_capacity(targets.len());
        for (i, t) in targets.iter().enumerate() {
            let body = self.region(*t, &inner);
            // The table holds positions, and the case value of a position is
            // the position plus whatever the compiler took off the index
            // before it looked the table up.
            let value = match first_case {
                Some(first) => first.wrapping_add(i as i64),
                None => i as i64,
            };
            cases.push((value as u64, body));
        }
        (
            Node::Switch {
                expr: index,
                cases,
                default: Vec::new(),
            },
            join,
        )
    }

    /// Structure the loop headed at `header` and return where control resumes
    /// after it, when that is a single block.
    ///
    /// The header's own statements are placed *inside* the body: they are what
    /// recomputes the loop condition on every iteration, so hoisting them out
    /// would be wrong, not merely ugly.
    fn structure_loop(&mut self, header: BlockId, nodes: &mut Vec<Node>) -> Option<BlockId> {
        if !self.label_once(header, nodes) {
            return None;
        }
        if !self.placed.insert(header) {
            // The loop header was already emitted. The jump needs a target
            // like any other, so the header goes on the list of blocks that
            // are emitted again at the end.
            if !self.goto_targets.contains(&header) {
                self.goto_targets.push(header);
            }
            nodes.push(Node::Goto(label(header)));
            self.unstructured += 1;
            return None;
        }
        let body_blocks = self.graph.loops.get(&header).cloned().unwrap_or_default();
        let exits = self.exit_blocks(&body_blocks);
        let inner = Region {
            stop: Some(header),
            loop_header: Some(header),
            loop_exits: exits.clone(),
        };
        let stmts = self.stmts_to_nodes(&self.blocks[header].stmts);
        match self.blocks[header].term.clone() {
            Term::Branch {
                cond,
                then_bb,
                else_bb,
            } => {
                let then_in = then_bb.is_some_and(|b| body_blocks.contains(&b));
                let else_in = else_bb.is_some_and(|b| body_blocks.contains(&b));
                if then_in == else_in {
                    // Both edges stay inside, or both leave: the header is
                    // not the loop test, and the test is deeper in the body.
                    // Treating the header as a plain conditional put the
                    // body's inputs outside the loop and left the back edge
                    // as a `goto`. The loop is `for (;;)` and the test is a
                    // `break`, which is what the machine does.
                    let _ = cond;
                    self.placed.remove(&header);
                    let inner = Region {
                        stop: None,
                        loop_header: Some(header),
                        loop_exits: exits.clone(),
                    };
                    nodes.push(Node::While {
                        cond: Expr::Const(1),
                        body: self.region(header, &inner),
                    });
                    let mut outs: Vec<BlockId> = exits.iter().copied().collect();
                    outs.sort_unstable();
                    return if outs.len() == 1 { Some(outs[0]) } else { None };
                }
                let (body_entry, exit, body_cond) = if then_in {
                    (then_bb.unwrap_or_default(), else_bb, cond)
                } else {
                    (else_bb.unwrap_or_default(), then_bb, negate(cond))
                };
                if body_entry == header {
                    // The header branches to itself: the test runs after the
                    // body, which is a `do`/`while`.
                    nodes.push(Node::DoWhile {
                        body: stmts,
                        cond: body_cond,
                    });
                    return exit;
                }
                let mut body = stmts;
                body.extend(self.region(body_entry, &inner));
                nodes.push(Node::While {
                    cond: body_cond,
                    body,
                });
                exit
            }
            Term::Switch {
                index,
                targets,
                first_case,
            } => {
                // A dispatch at the loop header: the cases are the body.
                let mut body = stmts;
                let (node, join) = self.switch(index, &targets, first_case, &inner);
                body.push(node);
                nodes.extend(body);
                join
            }
            Term::Jump(t) => {
                // No test at the header: an unconditional loop whose only exits
                // are `break`s and returns.
                let mut body = stmts;
                if t != header {
                    body.extend(self.region(t, &inner));
                }
                nodes.push(Node::While {
                    cond: Expr::Const(1),
                    body,
                });
                let mut outs: Vec<BlockId> = exits.iter().copied().collect();
                outs.sort_unstable();
                if outs.len() == 1 { Some(outs[0]) } else { None }
            }
            _ => {
                nodes.extend(stmts);
                None
            }
        }
    }

    /// Emit a conditional and return where control continues, if anywhere.
    fn branch(
        &mut self,
        b: BlockId,
        cond: Expr,
        then_bb: Option<BlockId>,
        else_bb: Option<BlockId>,
        ctx: &Region,
        nodes: &mut Vec<Node>,
    ) -> Option<BlockId> {
        // A side that leaves the function has no block to walk into. The
        // branch is still two-way, and the sides are not symmetric: the one
        // that stays is where the code continues, the other one is a
        // transfer out. Saying so is better than showing an arm that ends
        // without a return, which reads as a value the code never produced.
        let (then_bb, else_bb) = match (then_bb, else_bb) {
            (Some(t), Some(f)) => (t, f),
            (None, Some(f)) => {
                let body = self.region(f, &Region {
                    stop: ctx.stop,
                    loop_header: ctx.loop_header,
                    loop_exits: ctx.loop_exits.clone(),
                });
                nodes.push(Node::If {
                    cond: negate(cond),
                    then_body: body,
                    else_body: vec![Node::Comment(
                        "this path leaves the function".into(),
                    )],
                });
                return None;
            }
            (Some(t), None) => {
                let body = self.region(t, &Region {
                    stop: ctx.stop,
                    loop_header: ctx.loop_header,
                    loop_exits: ctx.loop_exits.clone(),
                });
                nodes.push(Node::If {
                    cond,
                    then_body: body,
                    else_body: vec![Node::Comment(
                        "this path leaves the function".into(),
                    )],
                });
                return None;
            }
            (None, None) => {
                nodes.push(Node::Comment(
                    "both sides of this branch leave the function".into(),
                ));
                return None;
            }
        };
        // Inside a loop, an edge can be the back edge, an exit, or neither.
        // Both sides are read before either is written, because the four
        // combinations are different statements: `if (c) continue;`,
        // `if (c) break;`, `if (c) continue; else break;` — and the last one
        // matters, because walking into the exit from inside the loop puts
        // the code after the loop inside its body.
        if let Some(h) = ctx.loop_header {
            let then_back = then_bb == h;
            let else_back = else_bb == h;
            let then_exit = !then_back && ctx.loop_exits.contains(&then_bb);
            let else_exit = !else_back && ctx.loop_exits.contains(&else_bb);
            let then_out = if then_back {
                Some(Node::Continue)
            } else if then_exit {
                Some(Node::Break)
            } else {
                None
            };
            let else_out = if else_back {
                Some(Node::Continue)
            } else if else_exit {
                Some(Node::Break)
            } else {
                None
            };
            match (then_out, else_out) {
                (Some(t), Some(e)) => {
                    nodes.push(Node::If {
                        cond,
                        then_body: vec![t],
                        else_body: vec![e],
                    });
                    return None;
                }
                (Some(t), None) => {
                    nodes.push(Node::If {
                        cond,
                        then_body: vec![t],
                        else_body: Vec::new(),
                    });
                    return Some(else_bb);
                }
                (None, Some(e)) => {
                    nodes.push(Node::If {
                        cond: negate(cond),
                        then_body: vec![e],
                        else_body: Vec::new(),
                    });
                    return Some(then_bb);
                }
                (None, None) => {}
            }
        }

        // Plain conditional: structure both arms up to their join point.
        let join = self.graph.ipdom[b].filter(|j| Some(*j) != ctx.stop);
        match join {
            Some(j) if j == then_bb => {
                // `if (!cond) { else }` then continue at the join.
                let inner = Region {
                    stop: Some(j),
                    ..clone_ctx(ctx)
                };
                let body = self.region(else_bb, &inner);
                nodes.push(Node::If {
                    cond: negate(cond),
                    then_body: body,
                    else_body: Vec::new(),
                });
                Some(j)
            }
            Some(j) if j == else_bb => {
                let inner = Region {
                    stop: Some(j),
                    ..clone_ctx(ctx)
                };
                let body = self.region(then_bb, &inner);
                nodes.push(Node::If {
                    cond,
                    then_body: body,
                    else_body: Vec::new(),
                });
                Some(j)
            }
            Some(j) => {
                let inner = Region {
                    stop: Some(j),
                    ..clone_ctx(ctx)
                };
                let then_body = self.region(then_bb, &inner);
                let else_body = self.region(else_bb, &inner);
                nodes.push(Node::If {
                    cond,
                    then_body,
                    else_body,
                });
                Some(j)
            }
            None => {
                // No join, so at most one arm can carry on past the branch,
                // and the other one has to end the region. Both are still
                // emitted: the arm that ends does it with a `return` or a
                // jump, and leaving it out dropped that `return` — the
                // function then read as returning nothing when its condition
                // held. The shape is a plain `if`/`else` whose arms both
                // terminate, which is exactly what the machine does.
                nodes.push(Node::If {
                    cond,
                    then_body: self.region(then_bb, &Region {
                        stop: ctx.stop,
                        loop_header: ctx.loop_header,
                        loop_exits: ctx.loop_exits.clone(),
                    }),
                    else_body: self.region(else_bb, &Region {
                        stop: ctx.stop,
                        loop_header: ctx.loop_header,
                        loop_exits: ctx.loop_exits.clone(),
                    }),
                });
                None
            }
        }
    }

    fn exit_blocks(&self, loop_blocks: &HashSet<BlockId>) -> HashSet<BlockId> {
        let mut out = HashSet::new();
        for b in loop_blocks {
            for s in successors(&self.blocks[*b]) {
                if !loop_blocks.contains(&s) {
                    out.insert(s);
                }
            }
        }
        out
    }

    fn stmts_to_nodes(&self, stmts: &[Stmt]) -> Vec<Node> {
        stmts
            .iter()
            .map(|s| match s {
                Stmt::Assign { dest, value } => Node::Assign {
                    dest: dest.clone(),
                    value: value.clone(),
                },
                Stmt::Store { addr, value, size } => Node::Store {
                    addr: addr.clone(),
                    value: value.clone(),
                    size: *size,
                },
                Stmt::Call {
                    name, args, ret, ..
                } => Node::Call {
                    name: name.clone(),
                    args: args.clone(),
                    ret: ret.clone(),
                },
                Stmt::If {
                    cond,
                    then_body,
                    else_body,
                } => Node::If {
                    cond: cond.clone(),
                    then_body: self.stmts_to_nodes(then_body),
                    else_body: self.stmts_to_nodes(else_body),
                },
                Stmt::Comment(c) => Node::Comment(c.clone()),
            })
            .collect()
    }
}

fn clone_ctx(ctx: &Region) -> Region {
    Region {
        stop: ctx.stop,
        loop_header: ctx.loop_header,
        loop_exits: ctx.loop_exits.clone(),
    }
}

/// The block every one of `targets` jumps to, when there is exactly one.
///
/// A compiler lays a `switch` out as the case bodies followed by the code
/// they all fall into. Recovering that shared block is what lets the cases be
/// printed as a `switch` instead of a chain of `goto`s.
fn common_successor(blocks: &[Block], targets: &[BlockId]) -> Option<BlockId> {
    let mut it = targets.iter();
    let first = *it.next()?;
    let mut common = successors(&blocks[first]);
    for t in it {
        let succs = successors(&blocks[*t]);
        common.retain(|c| succs.contains(c));
    }
    if common.len() == 1 {
        Some(common[0])
    } else {
        None
    }
}

fn label(b: BlockId) -> String {
    format!("L{b}")
}

/// Logical negation of a condition, kept readable.
pub fn negate(e: Expr) -> Expr {
    match e {
        Expr::Cond {
            op,
            lhs,
            rhs,
            bits,
        } => Expr::Cond {
            op: op.negated(),
            lhs,
            rhs,
            bits,
        },
        Expr::Bool { op, args } => match op {
            crate::ir::BoolOp::Not => args.into_iter().next().unwrap_or(Expr::Const(0)),
            crate::ir::BoolOp::Eq => Expr::Bool {
                op: crate::ir::BoolOp::Ne,
                args,
            },
            crate::ir::BoolOp::Ne => Expr::Bool {
                op: crate::ir::BoolOp::Eq,
                args,
            },
            // The negation of a conjunction is a disjunction of negations.
            crate::ir::BoolOp::And => Expr::Bool {
                op: crate::ir::BoolOp::Or,
                args: args.into_iter().map(negate).collect(),
            },
            crate::ir::BoolOp::Or => Expr::Bool {
                op: crate::ir::BoolOp::And,
                args: args.into_iter().map(negate).collect(),
            },
        },
        Expr::FlagCond {
            cond,
            state,
            bits,
        } => Expr::FlagCond {
            cond: cond.negated(),
            state,
            bits,
        },
        other => Expr::Un {
            op: crate::ir::UnOp::LNot,
            arg: Box::new(other),
            bits: 1,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Block, Expr, Node, Stmt, Term};
    use occ_core::Addr;

    fn block(id: usize, term: Term) -> Block {
        Block {
            id,
            addr: Addr(0x1000 + id as u64 * 0x10),
            stmts: Vec::new(),
            term,
        }
    }

    /// Block `id` with a distinctive statement, so the tests can tell which
    /// blocks reached the output.
    fn marked(id: usize, term: Term) -> Block {
        Block {
            stmts: vec![Stmt::Assign {
                dest: format!("v{id}"),
                value: Expr::Const(id as u64),
            }],
            ..block(id, term)
        }
    }

    fn graph(blocks: &[Block], entry: BlockId) -> (Graph, Vec<Vec<BlockId>>) {
        let mut preds = vec![Vec::new(); blocks.len()];
        let mut succs = vec![Vec::new(); blocks.len()];
        for b in blocks {
            for s in successors(b) {
                succs[b.id].push(s);
                preds[s].push(b.id);
            }
        }
        (Graph::new(blocks, &preds, &succs, entry), preds)
    }

    fn flatten(nodes: &[Node], out: &mut Vec<String>) {
        for n in nodes {
            match n {
                Node::Assign { dest, .. } => out.push(dest.clone()),
                Node::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    flatten(then_body, out);
                    flatten(else_body, out);
                }
                Node::While { body, .. } | Node::DoWhile { body, .. } => flatten(body, out),
                Node::Switch { cases, default, .. } => {
                    for (_, body) in cases {
                        flatten(body, out);
                    }
                    flatten(default, out);
                }
                Node::Return(Some(Expr::Const(v))) => out.push(format!("ret{v}")),
                Node::Label(l) => out.push(format!("label:{l}")),
                Node::Goto(l) => out.push(format!("goto:{l}")),
                _ => {}
            }
        }
    }

    /// A diamond: `0` branches to `1` and `2`, both jump to `3`.
    fn diamond() -> Vec<Block> {
        vec![
            marked(
                0,
                Term::Branch {
                    cond: Expr::Var("c".into()),
                    then_bb: Some(1),
                    else_bb: Some(2),
                },
            ),
            marked(1, Term::Jump(3)),
            marked(2, Term::Jump(3)),
            marked(3, Term::Ret(None)),
        ]
    }

    #[test]
    fn post_dominators_find_the_join_of_a_diamond() {
        // The join is what tells the structurer where an `if` ends. Computing
        // it on the original predecessor lists instead of the reversed ones
        // left every post-dominator undefined, and a branch with no join fell
        // back to `goto` for the most ordinary shape there is.
        let blocks = diamond();
        let (g, _) = graph(&blocks, 0);
        assert_eq!(g.ipdom[0], Some(3), "the branch rejoins at 3");
        assert_eq!(g.ipdom[1], Some(3));
        assert_eq!(g.ipdom[2], Some(3));
        assert_eq!(g.ipdom[3], None, "3 returns; nothing post-dominates it");
        assert_eq!(g.idom[3], 0);
    }

    /// The classic irreducible shape: `1` and `3` form a cycle with two
    /// entries, from `0` directly and through `2`.
    ///
    /// ```text
    ///   0 -> 1, 2
    ///   1 -> 3
    ///   2 -> 3
    ///   3 -> 1, 4
    /// ```
    ///
    /// `1` has two predecessors and one of them is inside the cycle, so no
    /// single header dominates the loop and the structure has to name blocks.
    fn irreducible() -> Vec<Block> {
        vec![
            marked(
                0,
                Term::Branch {
                    cond: Expr::Var("c".into()),
                    then_bb: Some(1),
                    else_bb: Some(2),
                },
            ),
            marked(1, Term::Jump(3)),
            marked(2, Term::Jump(3)),
            marked(
                3,
                Term::Branch {
                    cond: Expr::Var("d".into()),
                    then_bb: Some(1),
                    else_bb: Some(4),
                },
            ),
            marked(4, Term::Ret(None)),
        ]
    }

    #[test]
    fn a_shared_block_is_labelled_where_it_is_not_copied_at_the_end() {
        // The copy used to be emitted after the body, and the copy of a block
        // that jumps to itself read `L1: goto L1` — a spin loop the original
        // code did not have. Labelling the block where it already sits costs
        // one label and no duplicate.
        let blocks = irreducible();
        let (g, _) = graph(&blocks, 0);
        let mut s = Structurer::new(&blocks, &g);
        let body = s.run(0);
        let mut seen = Vec::new();
        flatten(&body, &mut seen);

        let labels: Vec<&String> = seen
            .iter()
            .filter(|n| n.starts_with("label:"))
            .collect();
        for l in &labels {
            assert_eq!(
                labels.iter().filter(|x| *x == l).count(),
                1,
                "{l} is defined once: {seen:?}"
            );
        }
        for (i, n) in seen.iter().enumerate() {
            if let Some(l) = n.strip_prefix("label:") {
                let next = seen.get(i + 1).map(String::as_str);
                assert_ne!(
                    next,
                    Some(format!("goto:{l}").as_str()),
                    "{l} jumps to itself, which is a loop the code did not have: {seen:?}"
                );
            }
        }
        // Every jump has somewhere to land.
        for n in &seen {
            if let Some(l) = n.strip_prefix("goto:") {
                assert!(
                    seen.iter().any(|x| x == &format!("label:{l}")),
                    "goto {l} has no label: {seen:?}"
                );
            }
        }
    }

    #[test]
    fn a_join_two_arms_share_is_not_a_goto() {
        let blocks = diamond();
        let (g, _) = graph(&blocks, 0);
        let mut s = Structurer::new(&blocks, &g);
        let body = s.run(0);
        let mut seen = Vec::new();
        flatten(&body, &mut seen);
        assert_eq!(s.unstructured, 0, "no goto: {body:?}");
        assert!(!seen.iter().any(|n| n.starts_with("goto:")), "{seen:?}");
        // Both arms and the join are in the output, once each.
        for m in ["v0", "v1", "v2", "v3"] {
            assert_eq!(
                seen.iter().filter(|n| *n == m).count(),
                1,
                "{m} appears once: {seen:?}"
            );
        }
    }

    #[test]
    fn an_if_whose_arms_both_return_keeps_both_returns() {
        // `if (c) return 1; else return 2;` — the two arms leave the function
        // and rejoin nowhere. Emitting only one of them dropped a return, and
        // the function then read as returning nothing on that path.
        let blocks = vec![
            marked(
                0,
                Term::Branch {
                    cond: Expr::Var("c".into()),
                    then_bb: Some(1),
                    else_bb: Some(2),
                },
            ),
            marked(1, Term::Ret(Some(Expr::Const(1)))),
            marked(2, Term::Ret(Some(Expr::Const(2)))),
        ];
        let (g, _) = graph(&blocks, 0);
        assert_eq!(g.ipdom[0], None, "no block post-dominates the branch");
        let mut s = Structurer::new(&blocks, &g);
        let body = s.run(0);
        let mut seen = Vec::new();
        flatten(&body, &mut seen);
        assert!(seen.contains(&"ret1".to_string()), "{seen:?}");
        assert!(seen.contains(&"ret2".to_string()), "{seen:?}");
        assert_eq!(s.unstructured, 0, "{body:?}");
    }

    #[test]
    fn a_loop_whose_test_is_not_at_its_header_is_still_a_loop() {
        // The header runs the body and branches to two blocks that are both
        // inside the loop; the real test is one of them. Treating the header
        // as a plain conditional put the body's inputs outside the loop and
        // left the back edge as a `goto`.
        let blocks = vec![
            marked(0, Term::Jump(1)),
            marked(
                1,
                Term::Branch {
                    cond: Expr::Var("c".into()),
                    then_bb: Some(2),
                    else_bb: Some(3),
                },
            ),
            marked(2, Term::Jump(4)),
            marked(3, Term::Jump(4)),
            marked(
                4,
                Term::Branch {
                    cond: Expr::Var("d".into()),
                    then_bb: Some(1),
                    else_bb: Some(5),
                },
            ),
            marked(5, Term::Ret(None)),
        ];
        let (g, _) = graph(&blocks, 0);
        assert!(g.loops.contains_key(&1), "1 heads a loop");
        let mut s = Structurer::new(&blocks, &g);
        let body = s.run(0);
        let mut seen = Vec::new();
        flatten(&body, &mut seen);
        assert_eq!(s.unstructured, 0, "no goto: {body:?}");
        assert!(
            body.iter().any(|n| matches!(n, Node::While { .. })),
            "the loop is a loop: {body:?}"
        );
    }

    #[test]
    fn a_chain_of_blocks_needs_no_jumps() {
        let blocks = vec![
            marked(0, Term::Jump(1)),
            marked(1, Term::Jump(2)),
            marked(2, Term::Ret(None)),
        ];
        let (g, _) = graph(&blocks, 0);
        let mut s = Structurer::new(&blocks, &g);
        let body = s.run(0);
        assert_eq!(s.unstructured, 0, "{body:?}");
        assert_eq!(body.len(), 4, "three blocks and a return: {body:?}");
    }
}
