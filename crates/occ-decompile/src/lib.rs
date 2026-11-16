//! Decompilation: recovered instructions to pseudo-C.
//!
//! The pipeline is four passes, each a value that can be inspected on its own:
//!
//! 1. **Lift** — each instruction becomes statements over named values
//!    ([`ir::Stmt`]). Stack slots become `local_N`, flags become the condition
//!    of the branch that consumes them.
//! 2. **SSA** — every assignment gets a version, and phi nodes appear where
//!    paths join ([`ssa`]).
//! 3. **Clean** — copy propagation and dead assignment removal, repeated until
//!    the graph stops changing.
//! 4. **Structure** — dominators find loops, post-dominators find join points,
//!    and the blocks become `if`/`while`/`return` ([`struct_`]).
//!
//! What this is not: a type reconstructor, a C++ demangler, or an oracle. Every
//! instruction the lifter does not model shows up as a comment in the output,
//! and the count of constructs that had to fall back to `goto` is reported with
//! the result.


#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod exec;
pub mod ir;
pub mod lift;
pub mod render;
pub mod ssa;
pub mod struct_;

use std::collections::HashMap;

use occ_core::{Addr, FlowKind, Function, Image};

use ir::{Block, BlockId, Decompiled, Expr, Stmt, Term};
use lift::Lifted;

/// Options for one decompilation.
#[derive(Clone, Debug)]
pub struct Options {
    /// Stop after this many blocks; guards against pathological graphs.
    pub max_blocks: usize,
    /// Keep `// comment` statements for instructions that were not lifted.
    pub keep_comments: bool,
    /// Keep definitions that nothing in the function reads.
    ///
    /// A statement is dead when no later statement and no terminator uses the
    /// name it defines, and the clean-up pass drops those, which is what makes
    /// the rendered output readable. A client that reads *values* instead of
    /// statements cannot afford that: a block that ends in a branch the
    /// recovery could not follow (`ja .cold`) has no terminator to use
    /// anything, so every definition in it looks dead, and the value the
    /// branch tested is gone. Such a client sets this and reads the block as
    /// lifted.
    pub keep_dead: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            max_blocks: 20_000,
            keep_comments: true,
            keep_dead: false,
        }
    }
}

/// Decompile one recovered function.
///
/// `name_of` resolves call targets to readable names; passing a closure that
/// returns `sub_<addr>` is always safe.
pub fn decompile(
    image: &Image,
    func: &Function,
    opts: &Options,
    name_of: &dyn Fn(Addr) -> String,
) -> Decompiled {
    if !is_liftable(image.arch) {
        return outline(func, name_of, image.arch);
    }
    let program = build_program(image, func, opts, name_of);
    if program.blocks.is_empty() {
        return Decompiled {
            entry: func.entry,
            name: func.display_name(),
            params: Vec::new(),
            param_aliases: Vec::new(),
            ret_type: None,
            body: Vec::new(),
            unstructured: 0,
            notes: program.notes,
        };
    }
    let mut notes = program.notes.clone();
    let graph = struct_::Graph::new(&program.blocks, &program.preds, &program.succs, program.entry);
    let mut structurer = struct_::Structurer::new(&program.blocks, &graph);
    let body = structurer.run(program.entry);
    notes.extend(structurer.notes.iter().cloned());

    Decompiled {
        entry: func.entry,
        name: func.display_name(),
        params: program.params.clone(),
        param_aliases: program.param_aliases.clone(),
        ret_type: program.ret_type.clone(),
        body,
        unstructured: structurer.unstructured,
        notes,
    }
}

/// Whether the decompiler can lift this architecture's instructions.
pub fn is_liftable(arch: occ_core::Arch) -> bool {
    matches!(
        arch,
        occ_core::Arch::X86_64
            | occ_core::Arch::Aarch64
            | occ_core::Arch::RiscV32
            | occ_core::Arch::RiscV64
    )
}

/// The lifter for an architecture, or `None` when the architecture has none.
fn lifter_for<'a>(
    arch: occ_core::Arch,
    name_of: &'a dyn Fn(Addr) -> String,
    symbols: &'a dyn Fn(&str) -> Option<u64>,
) -> Option<Box<dyn lift::Lifter + 'a>> {
    match arch {
        occ_core::Arch::X86_64 => {
            static TARGET: lift::x86::X86Target = lift::x86::X86Target;
            Some(Box::new(lift::x86::Lifter::new(&TARGET, name_of, symbols)))
        }
        occ_core::Arch::Aarch64 => {
            static TARGET: lift::aarch64::Aarch64Target = lift::aarch64::Aarch64Target;
            Some(Box::new(lift::aarch64::Lifter::new(&TARGET, name_of, symbols)))
        }
        occ_core::Arch::RiscV32 | occ_core::Arch::RiscV64 => {
            static TARGET: lift::riscv::RiscvTarget = lift::riscv::RiscvTarget;
            Some(Box::new(lift::riscv::Lifter::new(&TARGET, name_of, symbols)))
        }
        _ => None,
    }
}

/// The address of a symbol, for the registers a linker fixes.
///
/// RISC-V's `gp` points at `__global_pointer$`, and every access to a global
/// through `gp` is unreadable without it. The symbol table states the value
/// directly, where following the startup code that installs it would mean
/// analysing a function that is not the one being decompiled.
fn symbol_lookup(image: &Image) -> impl Fn(&str) -> Option<u64> + '_ {
    move |name: &str| {
        image
            .symbols
            .iter()
            .find(|s| s.defined && s.name == name)
            .map(|s| s.addr.get())
    }
}

/// The lowered function, before any control-flow recovery.
///
/// This is the shared front end: the decompiler structures it, and the symbolic
/// executor walks it block by block. Both need the same lifting, or they would
/// disagree about what the function does.
pub struct Program {
    /// Block control enters at.
    pub entry: BlockId,
    /// Statements and terminators after SSA construction and clean-up.
    pub blocks: Vec<Block>,
    /// Predecessors of each block.
    pub preds: Vec<Vec<BlockId>>,
    /// Successors of each block.
    pub succs: Vec<Vec<BlockId>>,
    /// Parameter names the body reads, as the renderer prints them (`a0`, …).
    pub params: Vec<String>,
    /// ABI register for each entry of `params` (`rdi` for `a0`). Callers that
    /// need to supply an argument, rather than read one, use these.
    pub param_regs: Vec<String>,
    /// Register names to the names they are printed under.
    pub param_aliases: Vec<(String, String)>,
    /// The type of the value the function returns, when the body says so.
    pub ret_type: Option<String>,
    /// Notes about what was approximated while lifting.
    pub notes: Vec<String>,
}

impl Program {
    /// Block index for an instruction address, if a block starts there.
    pub fn block_at(&self, addr: Addr) -> Option<BlockId> {
        self.blocks.iter().position(|b| b.addr == addr)
    }

    /// Address of a block.
    pub fn addr_of(&self, b: BlockId) -> Addr {
        self.blocks.get(b).map(|b| b.addr).unwrap_or(Addr::NULL)
    }
}

/// The terminator of a computed jump: no edge, but a target value.
fn indirect(lifted: &Lifted) -> Term {
    Term::Indirect {
        target: lifted
            .indirect
            .clone()
            .unwrap_or_else(|| Expr::Unknown("indirect target".into())),
    }
}

fn show(stage: &str, blocks: &[Block]) {
    for b in blocks {
        eprintln!("[{stage}] block {} {:#x} term={:?}", b.id, b.addr.0, b.term);
        for s in &b.stmts {
            eprintln!("[{stage}]    {s:?}");
        }
    }
}

/// Lift and normalise one function, without recovering control flow.
pub fn build_program(
    image: &Image,
    func: &Function,
    opts: &Options,
    name_of: &dyn Fn(Addr) -> String,
) -> Program {
    let mut notes: Vec<String> = Vec::new();
    let symbols = symbol_lookup(image);
    let Some(mut lifter) = lifter_for(image.arch, name_of, &symbols) else {
        notes.push(format!(
            "lifting is implemented for x86-64, AArch64 and RISC-V, this is {}",
            image.arch.label()
        ));
        return Program {
            entry: 0,
            blocks: Vec::new(),
            preds: Vec::new(),
            succs: Vec::new(),
            params: Vec::new(),
            param_regs: Vec::new(),
            param_aliases: Vec::new(),
            ret_type: None,
            notes,
        };
    };

    // Map block entry addresses to indices, so branch targets become indices.
    let mut by_addr: HashMap<Addr, BlockId> = HashMap::new();
    for (i, b) in func.blocks.iter().enumerate() {
        by_addr.insert(b.start, i);
    }

    let mut blocks: Vec<Block> = Vec::with_capacity(func.blocks.len());
    let mut tail: Vec<Stmt> = Vec::new();
    let arg_reg_names = lifter.arg_reg_names();
    for (i, bb) in func.blocks.iter().enumerate() {
        if i >= opts.max_blocks {
            notes.push(format!("stopped after {} blocks", opts.max_blocks));
            break;
        }
        let insns = func
            .insns
            .get(bb.insn_start..bb.insn_end)
            .unwrap_or_default();
        let lifted = lifter.lift_block(insns);
        let last = insns.last();
        let term = match last {
            None => Term::Unreachable,
            Some(insn) => match insn.flow {
                FlowKind::Return => Term::Ret(Some(Expr::var(String::from("rax")))),
                FlowKind::UnconditionalBranch => match insn.target {
                    Some(occ_core::CallTarget::Direct { to }) => match by_addr.get(&to) {
                        Some(t) => Term::Jump(*t),
                        None => {
                            // A jump out of the function is a tail call: the
                            // arguments are already in their registers.
                            tail.push(Stmt::Call {
                                name: name_of(to),
                                args: lifter.call_args(),
                                ret: None,
                                addr: insn.addr,
                            });
                            Term::Ret(None)
                        }
                    },
                    Some(occ_core::CallTarget::Indirect { via: Some(slot) }) => {
                        tail_slot(image, &mut tail, slot);
                        // The frame never comes back, so there is no `return`
                        // statement to print: the callee's result is the
                        // callee's.
                        Term::Unreachable
                    }
                    _ => Term::Unreachable,
                },
                FlowKind::IndirectBranch => {
                    // A computed jump whose table was recovered is a `switch`.
                    // The case bodies are already part of the function; the
                    // dispatch is the one thing a linear listing cannot show.
                    match func.jump_tables.iter().find(|t| t.at == insn.addr) {
                        Some(table) => {
                            let mut targets = Vec::with_capacity(table.targets.len());
                            let mut complete = true;
                            for a in &table.targets {
                                match by_addr.get(a) {
                                    Some(b) => targets.push(*b),
                                    None => {
                                        // A case that was not recovered would
                                        // silently become a missing branch.
                                        complete = false;
                                        break;
                                    }
                                }
                            }
                            if complete && !targets.is_empty() {
                                let index = match &table.index_reg {
                                    Some(reg) => Expr::var(reg.clone()),
                                    None => Expr::Unknown("switch index".into()),
                                };
                                Term::Switch {
                                    index,
                                    targets,
                                    first_case: table.first_case,
                                }
                            } else {
                                indirect(&lifted)
                            }
                        }
                        // No table was recovered, but the jump still reads an
                        // address, and that address is the one thing a later
                        // pass can work with.
                        None => indirect(&lifted),
                    }
                }
                FlowKind::ConditionalBranch => {
                    let cond = lifted
                        .branch
                        .clone()
                        .unwrap_or_else(|| Expr::Unknown("condition".into()));
                    let then_bb = match insn.target {
                        Some(occ_core::CallTarget::Direct { to }) => by_addr.get(&to).copied(),
                        _ => None,
                    };
                    let else_bb = by_addr.get(&bb.end).copied();
                    match (then_bb, else_bb) {
                        (Some(t), Some(f)) => Term::Branch {
                            cond,
                            then_bb: Some(t),
                            else_bb: Some(f),
                        },
                        // One side of the branch leaves the function: a
                        // compiler puts the cold half of a branch in its own
                        // chunk. The other side is still a successor, and
                        // dropping it takes the fall-through with it. The
                        // block after the branch then has no predecessor, and
                        // every value the branch carried into it is undefined.
                        (Some(t), None) => Term::Branch {
                            cond,
                            then_bb: Some(t),
                            else_bb: None,
                        },
                        (None, Some(f)) => Term::Branch {
                            cond,
                            then_bb: None,
                            else_bb: Some(f),
                        },
                        (None, None) => Term::Unreachable,
                    }
                }
                _ => match by_addr.get(&bb.end) {
                    Some(next) => Term::Jump(*next),
                    // A call to a function that never returns is the end of the
                    // block, not a fall-through into a `return`.
                    None if calls_noreturn(insn, name_of) => Term::Unreachable,
                    None => Term::Ret(None),
                },
            },
        };
        let mut stmts = lifted.stmts;
        stmts.append(&mut tail);
        if !opts.keep_comments {
            stmts.retain(|s| !matches!(s, Stmt::Comment(_)));
        }
        if std::env::var("OCC_DEBUG_LIFT").is_ok() {
            eprintln!("lift {:#x} term={term:?}", bb.start.0);
            for s in &stmts {
                eprintln!("   {s:?}");
            }
        }
        blocks.push(Block {
            id: i,
            addr: bb.start,
            stmts,
            term,
        });
    }

    // Which register the value comes back in depends on the function's
    // type, which the code does not carry; the body says where the
    // function left its last value, and that is settled only once every
    // block has been lifted.
    let ret_reg = lifter.ret_reg();
    let ret_type = lifter.ret_type().map(str::to_string);
    // A function that never wrote the register the value comes from has
    // no value to return, and printing one would claim the function chose
    // it.
    let ret_defined = lifter.ret_defined();
    for b in blocks.iter_mut() {
        if let Term::Ret(Some(_)) = b.term {
            b.term = if ret_defined {
                Term::Ret(Some(Expr::var(ret_reg.clone())))
            } else {
                Term::Ret(None)
            };
        }
    }

    for (addr, note) in lifter.notes() {
        notes.push(format!("{addr}: {note}"));
    }
    if blocks.is_empty() {
        return Program {
            entry: 0,
            blocks,
            preds: Vec::new(),
            succs: Vec::new(),
            params: Vec::new(),
            param_regs: Vec::new(),
            param_aliases: Vec::new(),
            ret_type: None,
            notes,
        };
    }

    let debug = std::env::var("OCC_DEBUG").is_ok();
    split_return_blocks(&mut blocks);
    let mut cfg = ssa::Cfg::new(blocks);
    // The entry block is the one at the function's entry address, not block 0:
    // with `-freorder-blocks` the entry sits in the middle of the layout.
    let mut entry = cfg
        .blocks
        .iter()
        .position(|b| b.addr == func.entry)
        .unwrap_or(0);
    // A function whose entry is also a jump target has a predecessor from
    // outside the function: the caller. Without a block to represent that edge,
    // a `do { } while` at the top of a function looks like it is entered from
    // itself, and the value read at the loop head resolves to the value the
    // loop is about to define — an infinite regress in the rename pass and,
    // before that, a plain stack overflow. One empty block restores the missing
    // edge and keeps every other index untouched.
    if !cfg.preds[entry].is_empty() {
        let id = cfg.blocks.len();
        cfg.blocks.push(Block {
            id,
            addr: func.entry,
            stmts: Vec::new(),
            term: Term::Jump(entry),
        });
        cfg = ssa::Cfg::new(cfg.blocks);
        entry = id;
    }
    let order = cfg.rpo(entry);
    let trace = std::env::var("OCC_DEBUG_LIFT").is_ok();
    notes.extend(ssa::build(&mut cfg.blocks, &cfg.preds, &order, entry));
    if trace {
        show("build", &cfg.blocks);
    }
    // Trivial phis first: they carry no information, and folding them now keeps
    // the copy pass from chasing names that only exist because two paths
    // happened to agree.
    ssa::simplify_phis(&mut cfg.blocks);
    if trace {
        show("phis", &cfg.blocks);
    }
    let opaque = ssa::eliminate_phis(&mut cfg.blocks, &cfg.preds);
    if trace {
        show("eliminate", &cfg.blocks);
    }
    ssa::coalesce(&mut cfg.blocks, &opaque);
    if trace {
        show("coalesce", &cfg.blocks);
    }
    ssa::cleanup(&mut cfg.blocks, &opaque, opts.keep_dead);
    if trace {
        show("cleanup", &cfg.blocks);
    }
    ssa::fold_tail_copies(&mut cfg.blocks, &opaque);
    if trace {
        show("tail", &cfg.blocks);
    }

    if debug {
        for b in &cfg.blocks {
            eprintln!("block {} {:#x} term={:?}", b.id, b.addr.0, b.term);
            for s in &b.stmts {
                eprintln!("   {s:?}");
            }
        }
    }
    let (params, param_regs) = parameter_names(&cfg.blocks, &arg_reg_names);
    let mut param_aliases: Vec<(String, String)> = param_regs
        .iter()
        .cloned()
        .zip(params.iter().cloned())
        .collect();
    // The stack pointer at entry is the frame base, not a live pointer.
    param_aliases.push((lifter.sp_reg_name(), lifter.sp_alias().to_string()));
    Program {
        entry,
        blocks: cfg.blocks,
        preds: cfg.preds,
        succs: cfg.succs,
        params,
        param_regs,
        param_aliases,
        ret_type,
        notes,
    }
}


/// Describe a tail jump through a GOT slot.
///
/// A PLT stub is `endbr64; jmp [rip+slot]`, which the decoder reports as an
/// unconditional branch whose target is the slot. Without a body of its own it
/// rendered as `u64 free@plt(void) { }`, which reads as a function that does
/// nothing — the opposite of what the two instructions say. The slot is named
/// through the import table when the container records one, through the symbol
/// table when that names it, and printed as an address otherwise.
fn tail_slot(image: &Image, tail: &mut Vec<Stmt>, slot: Addr) {
    let name = image
        .import_by_slot(slot)
        .map(|i| i.name.clone())
        .or_else(|| image.symbol_at(slot).map(|s| s.name.clone()));
    tail.push(Stmt::Comment(match name {
        Some(import) => format!(
            "tail jump to {import} through GOT slot {slot}; the arguments are whatever the caller left in registers"
        ),
        None => format!("tail jump through GOT slot {slot}; no import or symbol names it"),
    }));
}

/// True when the instruction calls a function that does not come back.
fn calls_noreturn(insn: &occ_core::Instruction, name_of: &dyn Fn(Addr) -> String) -> bool {
    let Some(occ_core::CallTarget::Direct { to }) = insn.target else {
        return false;
    };
    let name = name_of(to);
    let base = name.split('@').next().unwrap_or(&name);
    matches!(
        base,
        "__stack_chk_fail"
            | "abort"
            | "exit"
            | "_exit"
            | "_Exit"
            | "quick_exit"
            | "__assert_fail"
            | "__cxa_throw"
            | "longjmp"
            | "siglongjmp"
            | "pthread_exit"
    )
}

/// Give each arm its own copy of a shared `return` block.
///
/// A block whose whole body is `return` is the join point of several arms.
/// Leaving it shared makes the structurer reach an already-emitted block and
/// fall back to `goto`, which is both unreadable and, when the label was
/// consumed by an earlier arm, wrong. Copying it per predecessor costs one
/// statement and keeps the graph a tree of returns.
fn split_return_blocks(blocks: &mut Vec<Block>) {
    let mut preds: Vec<Vec<BlockId>> = vec![Vec::new(); blocks.len()];
    for (i, b) in blocks.iter().enumerate() {
        for s in successors_of(b) {
            if s < blocks.len() {
                preds[s].push(i);
            }
        }
    }
    for r in 0..blocks.len() {
        if preds[r].len() < 2 || !is_bare_return(&blocks[r]) {
            continue;
        }
        for p in preds[r].clone() {
            let clone = Block {
                id: blocks.len(),
                addr: blocks[r].addr,
                stmts: blocks[r].stmts.clone(),
                term: blocks[r].term.clone(),
            };
            let new_id = clone.id;
            blocks.push(clone);
            redirect(&mut blocks[p].term, r, new_id);
        }
    }
}

/// True when a block only computes a return value and returns, so copying it
/// per predecessor is cheap and cannot duplicate an observable effect.
fn is_bare_return(b: &Block) -> bool {
    matches!(b.term, Term::Ret(_))
        && b.stmts.len() <= 6
        && !b
            .stmts
            .iter()
            .any(|s| matches!(s, Stmt::Call { .. } | Stmt::Store { .. }))
}

fn successors_of(b: &Block) -> Vec<BlockId> {
    match &b.term {
        Term::Jump(t) => vec![*t],
        Term::Branch { then_bb, else_bb, .. } => {
            then_bb.iter().chain(else_bb.iter()).copied().collect()
        }
        // Every case of a switch is a successor: leaving them out of the
        // graph makes the phi placement and the rename pass treat the case
        // bodies as unreachable from the dispatch.
        Term::Switch { targets, .. } => targets.clone(),
        _ => Vec::new(),
    }
}

fn redirect(term: &mut Term, from: BlockId, to: BlockId) {
    match term {
        Term::Jump(t) if *t == from => *t = to,
        Term::Branch { then_bb, else_bb, .. } => {
            if then_bb == &Some(from) {
                *then_bb = Some(to);
            }
            if else_bb == &Some(from) {
                *else_bb = Some(to);
            }
        }
        Term::Switch { targets, .. } => {
            for t in targets.iter_mut() {
                if *t == from {
                    *t = to;
                }
            }
        }
        _ => {}
    }
}

/// Which incoming registers the body actually reads.
///
/// The registers are the architecture's argument registers in ABI order; the
/// names come from the architecture, not from a table here.
fn parameter_names(blocks: &[Block], arg_regs: &[String]) -> (Vec<String>, Vec<String>) {
    let mut used = vec![false; arg_regs.len()];
    fn scan(e: &Expr, used: &mut [bool], arg_regs: &[String]) {
        let mut vars = Vec::new();
        e.vars(&mut vars);
        for v in vars {
            for (i, reg) in arg_regs.iter().enumerate() {
                // Only the incoming value counts: `rdi.2` was written inside
                // the function.
                if v == *reg {
                    used[i] = true;
                }
            }
        }
    }
    fn walk(nodes: &[Stmt], used: &mut [bool], arg_regs: &[String], term: &Term) {
        for s in nodes {
            match s {
                Stmt::Assign { value, .. } => scan(value, used, arg_regs),
                Stmt::Store { addr, value, .. } => {
                    scan(addr, used, arg_regs);
                    scan(value, used, arg_regs);
                }
                Stmt::Call { args, .. } => {
                    for a in args {
                        scan(a, used, arg_regs);
                    }
                }
                Stmt::Comment(_) => {}
            }
        }
        if let Term::Branch { cond, .. } = term {
            scan(cond, used, arg_regs);
        }
        if let Term::Switch { index, .. } = term {
            scan(index, used, arg_regs);
        }
        if let Term::Ret(Some(v)) = term {
            scan(v, used, arg_regs);
        }
    }
    for b in blocks {
        walk(&b.stmts, &mut used, arg_regs, &b.term);
    }
    let mut params = Vec::new();
    let mut regs = Vec::new();
    for (i, u) in used.iter().enumerate() {
        if *u {
            params.push(format!("a{i}"));
            regs.push(arg_regs[i].clone());
        }
    }
    (params, regs)
}

/// Structural outline for architectures whose instructions are not lifted.
fn outline(func: &Function, name_of: &dyn Fn(Addr) -> String, arch: occ_core::Arch) -> Decompiled {
    let mut body: Vec<ir::Node> = Vec::new();
    body.push(ir::Node::Comment(format!(
        "body not lifted: the decompiler models x86-64, AArch64 and RISC-V, this is {}",
        arch.label()
    )));
    let mut unstructured = 0;
    for bb in &func.blocks {
        body.push(ir::Node::Label(format!("L_{:x}", bb.start.0)));
        for insn in func
            .insns
            .get(bb.insn_start..bb.insn_end)
            .unwrap_or_default()
        {
            if let Some(occ_core::CallTarget::Direct { to }) = insn.target {
                body.push(ir::Node::Call {
                    name: name_of(to),
                    args: Vec::new(),
                    ret: None,
                });
            }
        }
        match &bb.edges.first() {
            Some(e) => body.push(ir::Node::Goto(format!("L_{:x}", e.to.0))),
            None => body.push(ir::Node::Return(None)),
        }
        unstructured += 1;
    }
    Decompiled {
        entry: func.entry,
        name: func.display_name(),
        params: Vec::new(),
        param_aliases: Vec::new(),
        ret_type: None,
        body,
        unstructured,
        notes: vec![format!(
            "lifting for {} is not implemented yet",
            arch.label()
        )],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_names_stay_distinct_per_width() {
        assert_ne!(
            lift::slot_name(-8, 4),
            lift::slot_name(-8, 8),
            "a 4-byte and an 8-byte slot at the same offset are different variables"
        );
    }

    #[test]
    fn ssa_names_become_identifiers() {
        assert_eq!(crate::render::name("rax.12"), "rax_12");
        assert_eq!(crate::render::name("local_8_u64.2"), "local_8_u64_2");
    }

    /// Blocks of a graph where two arms share a tail.
    ///
    /// `0` branches to `1` and `2`; both jump to `3`. A structurer that
    /// inlines the shared tail into the first arm has to reach it from the
    /// second one somehow.
    fn shared_tail_graph() -> Vec<crate::ir::Block> {
        use crate::ir::{Block, Expr, Stmt, Term};

        vec![
            Block {
                id: 0,
                addr: occ_core::Addr(0x1000),
                stmts: Vec::new(),
                term: Term::Branch {
                    cond: Expr::Var("c".into()),
                    then_bb: Some(1),
                    else_bb: Some(2),
                },
            },
            Block {
                id: 1,
                addr: occ_core::Addr(0x1010),
                stmts: Vec::new(),
                term: Term::Jump(3),
            },
            Block {
                id: 2,
                addr: occ_core::Addr(0x1020),
                stmts: Vec::new(),
                term: Term::Jump(3),
            },
            Block {
                id: 3,
                addr: occ_core::Addr(0x1030),
                stmts: vec![Stmt::Assign {
                    dest: "rax".into(),
                    value: Expr::Const(7),
                }],
                term: Term::Ret(Some(Expr::Var("rax".into()))),
            },
        ]
    }

    #[test]
    fn every_jump_has_a_label_to_land_on() {
        use crate::ir::Term;
        use crate::struct_::{Graph, Structurer};

        let blocks = shared_tail_graph();
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); blocks.len()];
        let mut succs: Vec<Vec<usize>> = vec![Vec::new(); blocks.len()];
        for b in &blocks {
            let out: Vec<usize> = match &b.term {
                Term::Jump(t) => vec![*t],
                Term::Branch { then_bb, else_bb, .. } => {
            then_bb.iter().chain(else_bb.iter()).copied().collect()
        }
                _ => Vec::new(),
            };
            for s in out {
                succs[b.id].push(s);
                preds[s].push(b.id);
            }
        }
        let graph = Graph::new(&blocks, &preds, &succs, 0);
        let mut st = Structurer::new(&blocks, &graph);
        let body = st.run(0);

        // Collect the labels the body defines and the labels it jumps to.
        let mut defined: Vec<String> = Vec::new();
        let mut jumped: Vec<String> = Vec::new();
        fn walk(nodes: &[crate::ir::Node], defined: &mut Vec<String>, jumped: &mut Vec<String>) {
            for n in nodes {
                match n {
                    crate::ir::Node::Label(l) => defined.push(l.clone()),
                    crate::ir::Node::Goto(l) => jumped.push(l.clone()),
                    crate::ir::Node::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        walk(then_body, defined, jumped);
                        walk(else_body, defined, jumped);
                    }
                    crate::ir::Node::Switch { cases, default, .. } => {
                        for (_, body) in cases {
                            walk(body, defined, jumped);
                        }
                        walk(default, defined, jumped);
                    }
                    crate::ir::Node::While { body, .. } | crate::ir::Node::DoWhile { body, .. } => {
                        walk(body, defined, jumped);
                    }
                    _ => {}
                }
            }
        }
        walk(&body, &mut defined, &mut jumped);

        assert!(
            !jumped.is_empty(),
            "the shared tail needs a jump: {body:?}"
        );
        for target in &jumped {
            assert!(
                defined.contains(target),
                "the jump to {target} has no label to land on; labels: {defined:?}"
            );
        }
        // The label has to carry the block it names, not just the name.
        let labelled = body
            .iter()
            .any(|n| matches!(n, crate::ir::Node::Label(l) if l == &jumped[0]));
        assert!(labelled, "the label is in the body");
    }
}
