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

pub mod debug;
pub mod exec;
pub mod ir;
pub mod lift;
pub mod render;
pub mod reshape;
pub mod ssa;
pub mod struct_;
pub mod wasm;

use std::collections::HashMap;

use occ_core::{Addr, FlowKind, Function, Image};

use ir::{BinOp, Block, BlockId, CastKind, Decompiled, Expr, Stmt, Term};
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
    /// Names, types and source lines from debug information.
    ///
    /// `None` means the image has no usable DWARF, and the output keeps the
    /// mechanical names the lifter chose. An empty value is not the same
    /// thing: it says debug info exists but says nothing about this function.
    pub debug: Option<debug::DebugInfo>,
    /// Mark each statement with the source line it came from.
    ///
    /// The line table is keyed by instruction address, so this needs debug
    /// information; without it the body is unchanged. Off by default because
    /// a caller that reads the body as code, rather than as a listing, does
    /// not want a comment between every pair of statements.
    pub source_lines: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            max_blocks: 20_000,
            keep_comments: true,
            keep_dead: false,
            debug: None,
            source_lines: false,
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
    // WebAssembly is a stack machine whose control flow is structured, and
    // the generic pipeline would have to re-derive what the module states.
    // It has a path of its own.
    if image.arch == occ_core::Arch::Wasm {
        return wasm::decompile(image, func, opts, name_of);
    }
    let program = build_program(image, func, opts, name_of);
    let dbg = opts.debug.as_ref();
    // The signature is the one the source declared, when debug info carries
    // it. The recovered name and the inferred word type stay as the fallback.
    let name = dbg
        .and_then(|d| d.name.clone())
        .unwrap_or_else(|| func.display_name());
    let ret_type = dbg
        .and_then(|d| d.return_type.clone())
        .or_else(|| program.ret_type.clone())
        .or_else(|| Some(word_type(image.arch).to_string()));
    let param_types = parameter_types(dbg, program.params.len(), word_type(image.arch));
    let source = dbg.and_then(|d| d.source_range(d.file.as_deref()));
    if program.blocks.is_empty() {
        return Decompiled {
            entry: func.entry,
            name,
            params: program.params,
            param_types,
            param_aliases: Vec::new(),
            ret_type,
            body: Vec::new(),
            unstructured: 0,
            notes: program.notes,
            source,
        };
    }
    let mut notes = program.notes.clone();
    let graph = struct_::Graph::new(
        &program.blocks,
        &program.preds,
        &program.succs,
        program.entry,
    );
    let mut structurer = struct_::Structurer::new(&program.blocks, &graph);
    let body = structurer.run(program.entry);
    notes.extend(structurer.notes.iter().cloned());

    Decompiled {
        entry: func.entry,
        name,
        params: program.params.clone(),
        param_types,
        param_aliases: program.param_aliases.clone(),
        ret_type,
        body,
        unstructured: structurer.unstructured,
        notes,
        source,
    }
}

/// The architectures whose instructions this decompiler lifts.
///
/// One list, used by both the gate and the message that names what is
/// missing: two copies of it drift, and a note that lists the wrong
/// architectures is worse than no note.
const LIFTABLE: &str = "x86-64, x86-32, ARM32, AArch64, RISC-V and WebAssembly";

/// The architectures whose instructions lift into an executable program.
///
/// This is a shorter list than `LIFTABLE`, and the difference is not an
/// oversight. WebAssembly is decompiled by walking its structured
/// regions — `block`, `loop`, `if` — which yields a body, but a body is
/// not the register-level program the symbolic executor explores: there
/// is no register file to fork and no condition flags to read. The two
/// lists are kept apart so that the message names the architectures that
/// can actually be executed, rather than the ones that can be read.
const EXECUTABLE: &str = "x86-64, x86-32, ARM32, AArch64 and RISC-V";

/// The name the machine's general-purpose word has in the output.
///
/// A signature that says nothing about a type still states a width: the
/// register the value arrives in is four bytes wide on a 32-bit target and
/// eight on a 64-bit one, and printing `u64` for both tells a reader the
/// upper half exists.
fn word_type(arch: occ_core::Arch) -> &'static str {
    match arch {
        occ_core::Arch::X86 | occ_core::Arch::Arm | occ_core::Arch::RiscV32 => "u32",
        _ => "u64",
    }
}

/// The block a conditional return jumps to, before it has a number.
///
/// The block is added after the function's own blocks are lifted, so the
/// number it ends up with is only known then.
const RETURN_BLOCK: BlockId = usize::MAX;

/// The types of the parameters, with the machine word where nothing else
/// says.
///
/// Debug information types the parameters it knows about and leaves the rest
/// untyped; an empty slot is not "no type", it is "this one is the width of
/// a register".
fn parameter_types(
    dbg: Option<&debug::DebugInfo>,
    count: usize,
    word: &str,
) -> Vec<Option<String>> {
    let mut types = vec![Some(word.to_string()); count];
    if let Some(d) = dbg {
        for (i, t) in d.param_types().into_iter().enumerate() {
            if let (Some(t), Some(slot)) = (t, types.get_mut(i)) {
                *slot = Some(t);
            }
        }
    }
    types
}

/// Whether the decompiler can lift this architecture's instructions.
pub fn is_liftable(arch: occ_core::Arch) -> bool {
    matches!(
        arch,
        occ_core::Arch::X86
            | occ_core::Arch::X86_64
            | occ_core::Arch::Arm
            | occ_core::Arch::Aarch64
            | occ_core::Arch::RiscV32
            | occ_core::Arch::RiscV64
            | occ_core::Arch::Wasm
    )
}

/// The lifter for an architecture, or `None` when the architecture has none.
///
/// The container format picks the calling convention where it decides one.
/// A Windows x86-64 binary passes its arguments in `rcx`, `rdx`, `r8` and
/// `r9`; reading it with System V's order hands every function the wrong
/// arguments, which is the kind of mistake that produces a plausible
/// decompilation of a program that does not exist.
fn lifter_for<'a>(
    arch: occ_core::Arch,
    format: occ_core::Format,
    name_of: &'a dyn Fn(Addr) -> String,
    symbols: &'a dyn Fn(&str) -> Option<u64>,
) -> Option<Box<dyn lift::Lifter + 'a>> {
    match arch {
        // 32-bit x86: the same instruction set at half the width, and the
        // one place where that changes the meaning of a call is the
        // arguments, which every convention but `fastcall` puts on the
        // stack. `stdcall` is the Windows convention, where the callee
        // removes them; on every other container the caller does.
        occ_core::Arch::X86 => {
            static CDECL: lift::x86::X86Target = lift::x86::X86Target::cdecl();
            static STDCALL: lift::x86::X86Target = lift::x86::X86Target::stdcall();
            let target = match format {
                occ_core::Format::Pe | occ_core::Format::Coff => &STDCALL,
                _ => &CDECL,
            };
            Some(Box::new(lift::x86::Lifter::new(target, name_of, symbols)))
        }
        // ARM32: AAPCS, arguments in r0-r3, return in r0.
        occ_core::Arch::Arm => {
            static ARM: lift::arm::ArmTarget = lift::arm::ArmTarget::new();
            Some(Box::new(lift::arm::Lifter::new(&ARM, name_of, symbols)))
        }
        occ_core::Arch::X86_64 => {
            static SYSV: lift::x86::X86Target = lift::x86::X86Target::sysv();
            static WIN64: lift::x86::X86Target = lift::x86::X86Target::win64();
            let target = match format {
                occ_core::Format::Pe | occ_core::Format::Coff => &WIN64,
                _ => &SYSV,
            };
            Some(Box::new(lift::x86::Lifter::new(target, name_of, symbols)))
        }
        occ_core::Arch::Aarch64 => {
            static TARGET: lift::aarch64::Aarch64Target = lift::aarch64::Aarch64Target;
            Some(Box::new(lift::aarch64::Lifter::new(
                &TARGET, name_of, symbols,
            )))
        }
        occ_core::Arch::RiscV32 | occ_core::Arch::RiscV64 => {
            static TARGET: lift::riscv::RiscvTarget = lift::riscv::RiscvTarget;
            Some(Box::new(lift::riscv::Lifter::new(
                &TARGET, name_of, symbols,
            )))
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

/// Label each statement with the source line the instruction came from.
///
/// A run of statements an instruction or a source line produced is labelled
/// once, at the first statement: a comment before every assignment would say
/// the same thing repeatedly and bury the body. The file name is printed when
/// it changes, which is what makes inlined and multi-file functions readable.
fn with_source_lines(stmts: Vec<Stmt>, at: &[Addr], dbg: Option<&debug::DebugInfo>) -> Vec<Stmt> {
    let Some(dbg) = dbg else {
        return stmts;
    };
    if dbg.lines.is_empty() {
        return stmts;
    }
    let mut out = Vec::with_capacity(stmts.len() + 8);
    let mut last: Option<(String, u32)> = None;
    for (i, s) in stmts.into_iter().enumerate() {
        let addr = at.get(i).copied().unwrap_or(Addr::NULL);
        if let Some((file, line)) = dbg.line_at(addr) {
            let same_file = last.as_ref().is_some_and(|(f, _)| f == file);
            let same_line = last.as_ref().is_some_and(|(_, l)| *l == line);
            if !same_file || !same_line {
                let text = if same_file {
                    format!("line {line}")
                } else {
                    format!("{file}:{line}")
                };
                out.push(Stmt::Comment(text));
                last = Some((file.to_string(), line));
            }
        }
        out.push(s);
    }
    out
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
    let Some(mut lifter) = lifter_for(image.arch, image.format, name_of, &symbols) else {
        notes.push(format!(
            "path exploration is implemented for {EXECUTABLE}, this is {}",
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
    let mut truncated = false;
    // A tail call hands the callee's result straight back to this function's
    // caller, so the return register holds a value even when no instruction in
    // the body wrote it.
    let mut forwards_call_result = false;
    // Jump targets inside the function's own range that its body never
    // decoded. Collected so the lift can say how many paths leave it.
    let mut holes: Vec<Addr> = Vec::new();
    // Where the block a conditional return jumps to will live.
    let mut return_block_at: Option<Addr> = None;
    let arg_reg_names = lifter.arg_reg_names();
    for (i, bb) in func.blocks.iter().enumerate() {
        if i >= opts.max_blocks {
            truncated = true;
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
                // The return register is the architecture's, not x86's:
                // ARM returns in r0 and AArch64 in x0.
                FlowKind::Return => Term::Ret(Some(Expr::var(lifter.ret_reg()))),
                FlowKind::UnconditionalBranch => match insn.target {
                    Some(occ_core::CallTarget::Direct { to }) => match by_addr.get(&to) {
                        Some(t) => Term::Jump(*t),
                        None if inside(func, to) => {
                            // The jump lands on an address this function's own
                            // range covers but its body never decoded. That is
                            // a hole, not a call: naming `sub_140001aff` here
                            // invented a function that does not exist and
                            // claimed this one returns its result.
                            tail.push(Stmt::Comment(format!(
                                "jump to {:#x} was never decoded; it is inside this function, so control leaves the recovered body here",
                                to.get()
                            )));
                            holes.push(to);
                            Term::Unreachable
                        }
                        None => {
                            // A jump out of the function is a tail call: the
                            // arguments are already in their registers. What
                            // the callee returns is what this function returns
                            // — the frame that called it is still on the stack
                            // — so the call carries the return value instead
                            // of dropping it, and the block ends by handing
                            // that value back.
                            if calls_noreturn(insn, name_of) {
                                tail.push(Stmt::Call {
                                    name: name_of(to),
                                    args: lifter.call_args(),
                                    ret: None,
                                    addr: insn.addr,
                                    target: Some(to),
                                });
                                Term::Unreachable
                            } else {
                                let ret = lifter.ret_reg();
                                tail.push(Stmt::Call {
                                    name: name_of(to),
                                    args: lifter.call_args(),
                                    ret: Some(ret.clone()),
                                    addr: insn.addr,
                                    target: Some(to),
                                });
                                forwards_call_result = true;
                                Term::Ret(Some(Expr::var(ret)))
                            }
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
                    let mut then_bb = match insn.target {
                        Some(occ_core::CallTarget::Direct { to }) => by_addr.get(&to).copied(),
                        _ => None,
                    };
                    let else_bb = by_addr.get(&bb.end).copied();
                    // A conditional return leaves the function on one path
                    // and carries on on the other. The path that leaves
                    // returns what the return register holds, so it gets a
                    // block of its own: a branch with no target reads as a
                    // jump into a cold chunk, and the reader is told the
                    // rest of the function is unreachable — which is how a
                    // closed form of a loop loses its body.
                    if then_bb.is_none() && lifted.conditional_return {
                        then_bb = Some(RETURN_BLOCK);
                        return_block_at = Some(insn.addr);
                    }
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
        // Statements the terminator produced come from the last instruction,
        // which is the address the block ends at.
        let mut stmt_at = lifted.stmt_at;
        stmt_at.resize(stmts.len(), bb.end);
        let tail_len = tail.len();
        stmts.append(&mut tail);
        for _ in 0..tail_len {
            stmt_at.push(bb.end);
        }
        if !opts.keep_comments {
            // The two vectors are parallel, so they are filtered together.
            let mut kept_stmts = Vec::with_capacity(stmts.len());
            let mut kept_at = Vec::with_capacity(stmt_at.len());
            for (s, a) in stmts.into_iter().zip(stmt_at) {
                if !matches!(s, Stmt::Comment(_)) {
                    kept_stmts.push(s);
                    kept_at.push(a);
                }
            }
            stmts = kept_stmts;
            stmt_at = kept_at;
        }
        if opts.source_lines {
            stmts = with_source_lines(stmts, &stmt_at, opts.debug.as_ref());
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

    // The block a conditional return jumps to. It is added after the rest,
    // because it is not a block of the function's own code: it is the end
    // of the function, reached from one path and carrying the value the
    // return register holds.
    if let Some(at) = return_block_at {
        let id = blocks.len();
        for b in blocks.iter_mut() {
            if let Term::Branch {
                then_bb: Some(t @ RETURN_BLOCK),
                ..
            } = &mut b.term
            {
                *t = id;
            }
        }
        blocks.push(Block {
            id,
            addr: at,
            stmts: Vec::new(),
            term: Term::Ret(Some(Expr::var(lifter.ret_reg()))),
        });
    }

    // A prefix cut leaves terminators that point at blocks the cap dropped.
    // Those targets are indices into a vector that no longer has them, and the
    // first pass that dereferences one aborts the process. The target exists —
    // it just was not lifted — so the edge is cut here, while that is still
    // known, and the cut is reported rather than hidden.
    if truncated {
        let cut = drop_edges_outside(&mut blocks);
        notes.push(format!(
            "stopped after {} of {} blocks; {cut} edge(s) into the unlifted remainder were cut",
            blocks.len(),
            func.blocks.len(),
        ));
    }

    // A hole is a fact about the recovery, not about this function's source:
    // the bytes are there, the body has a jump into them, and nothing decoded
    // them. Saying so is the difference between an incomplete listing and a
    // wrong one.
    if !holes.is_empty() {
        holes.sort_unstable_by_key(|a| a.get());
        holes.dedup();
        let shown = holes
            .iter()
            .take(4)
            .map(|a| format!("{:#x}", a.get()))
            .collect::<Vec<_>>()
            .join(", ");
        let more = holes.len().saturating_sub(4);
        notes.push(format!(
            "{} jump(s) land inside this function on bytes that were never decoded ({}); the paths stop there",
            holes.len(),
            if more > 0 {
                format!("{shown}, and {more} more")
            } else {
                shown
            }
        ));
    }

    // Which register the value comes back in depends on the function's
    // type, which the code does not carry; the body says where the
    // function left its last value, and that is settled only once every
    // block has been lifted.
    let ret_reg = lifter.ret_reg();
    let mut ret_type = lifter.ret_type().map(str::to_string);
    // A function that never wrote the register the value comes from has
    // no value to return, and printing one would claim the function chose
    // it.
    let ret_defined = lifter.ret_defined() || forwards_call_result;
    for b in blocks.iter_mut() {
        if let Term::Ret(Some(_)) = b.term {
            b.term = if ret_defined {
                Term::Ret(Some(Expr::var(ret_reg.clone())))
            } else {
                Term::Ret(None)
            };
        }
    }

    // A value wider than the machine word comes back in a pair of
    // registers, and the low half alone is not the value: the product of
    // two thirty-two bit numbers has a high half that is not zero, and
    // dropping it makes the function return the wrong number. When the
    // two halves are the two views of one wide value, that value is what
    // the function returns.
    if let Some((lo, hi)) = lifter.ret_pair() {
        for b in blocks.iter_mut() {
            if !matches!(b.term, Term::Ret(Some(_))) {
                continue;
            }
            if let Some((value, wide)) = pair_return(&b.stmts, &lo, &hi) {
                b.term = Term::Ret(Some(value));
                ret_type = Some(wide.to_string());
            }
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
    let mut entry = match cfg.blocks.iter().position(|b| b.addr == func.entry) {
        Some(e) => e,
        // The cap can fall before the entry block. Structuring whatever block
        // happened to be first would present a fragment of the function as the
        // function, so the result says it could not start instead.
        None if truncated => {
            notes.push(format!(
                "the entry block at {:#x} lies outside the {}-block limit; nothing to structure",
                func.entry.0, opts.max_blocks
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
        }
        None => 0,
    };
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
    let (params, param_regs) = parameter_names(
        &cfg.blocks,
        &arg_reg_names,
        &lifter.memory_args(),
        lifter.declared_args(),
    );
    let mut param_aliases: Vec<(String, String)> = param_regs
        .iter()
        .cloned()
        .zip(params.iter().cloned())
        .collect();
    // The stack pointer at entry is the frame base, not a live pointer.
    param_aliases.push((lifter.sp_reg_name(), lifter.sp_alias().to_string()));
    let mut program = Program {
        entry,
        blocks: cfg.blocks,
        preds: cfg.preds,
        succs: cfg.succs,
        params,
        param_regs,
        param_aliases,
        ret_type,
        notes,
    };
    if let Some(dbg) = &opts.debug {
        let frame_base = if dbg.frame_base_is_cfa {
            Some(cfa_slot(image.arch))
        } else {
            dbg.frame_base_reg
                .as_deref()
                .and_then(|r| lifter.frame_base_slot(r))
        };
        debug::apply(&mut program, dbg, frame_base);
    }
    program
}

/// Frame slot of the canonical frame address.
///
/// A `call` on x86 pushes the return address, so the caller's stack pointer —
/// which is what DWARF calls the canonical frame address — sits eight bytes
/// above the callee's entry stack pointer (four in 32-bit mode). AArch64 and
/// RISC-V keep the return address in a register, so their frame base is the
/// entry stack pointer itself.
/// The value a block returns in a pair of registers, when the two halves
/// are the two views of one wide value.
///
/// A multiply that produces twice as many bits as its factors writes the
/// low half of the product in one register and the high half in the next.
/// Read as a pair they are one number; read as the low register alone
/// they are half of one.
fn pair_return(stmts: &[Stmt], lo: &str, hi: &str) -> Option<(Expr, &'static str)> {
    let mut low: Option<&Expr> = None;
    let mut high: Option<&Expr> = None;
    for s in stmts {
        let Stmt::Assign { dest, value } = s else {
            continue;
        };
        if dest == lo {
            low = Some(value);
        }
        if dest == hi {
            high = Some(value);
        }
    }
    let low = low?;
    let high = high?;
    // The low half is the wide value narrowed, and the high half is the
    // same value shifted down.
    let Expr::Cast {
        kind: CastKind::Trunc,
        arg: low_arg,
        from: 8,
        to: 4,
    } = low
    else {
        return None;
    };
    let Expr::Cast {
        kind: CastKind::Trunc,
        arg: high_arg,
        from: 8,
        to: 4,
    } = high
    else {
        return None;
    };
    let Expr::Var(whole) = low_arg.as_ref() else {
        return None;
    };
    let Expr::Bin {
        op: BinOp::Shr,
        lhs: high_lhs,
        rhs: high_rhs,
        bits: 8,
    } = high_arg.as_ref()
    else {
        return None;
    };
    if high_lhs.as_ref() != &Expr::Var(whole.clone()) || high_rhs.as_ref() != &Expr::Const(32) {
        return None;
    }
    // Whether the number is signed is a fact about how it was made.
    let wide = match stmts.iter().find_map(|s| match s {
        Stmt::Assign { dest, value } if dest == whole => Some(value),
        _ => None,
    }) {
        Some(Expr::Bin { lhs, rhs, .. }) => {
            let signed = matches!(
                lhs.as_ref(),
                Expr::Cast {
                    kind: CastKind::Sext,
                    ..
                }
            ) || matches!(
                rhs.as_ref(),
                Expr::Cast {
                    kind: CastKind::Sext,
                    ..
                }
            );
            if signed { "i64" } else { "u64" }
        }
        _ => "u64",
    };
    Some((Expr::Var(whole.clone()), wide))
}

fn cfa_slot(arch: occ_core::Arch) -> i64 {
    match arch {
        occ_core::Arch::X86_64 => 8,
        occ_core::Arch::X86 => 4,
        _ => 0,
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

/// True when `a` falls inside the address range `f` covers.
///
/// A recovered function spans from its entry to the end of its last
/// instruction, and the bytes in between are not all instructions: a body that
/// jumps over a chunk the descent never followed leaves a hole. A branch into
/// that hole is not a call to anything.
fn inside(f: &Function, a: Addr) -> bool {
    a.get() >= f.entry.get() && a.get() < f.end().get()
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
/// Cut every successor that points past the end of `blocks`.
///
/// Returns how many edges were cut. A branch keeps the side that still has a
/// block; a jump that loses its target — and a switch that loses any case —
/// ends as control flow the pass cannot follow, because a switch with missing
/// cases would claim the function never goes there.
fn drop_edges_outside(blocks: &mut [Block]) -> usize {
    let n = blocks.len();
    let mut cut = 0;
    for b in blocks.iter_mut() {
        let term = std::mem::replace(&mut b.term, Term::Unreachable);
        b.term = match term {
            Term::Jump(t) => {
                if t >= n {
                    cut += 1;
                    Term::Unreachable
                } else {
                    Term::Jump(t)
                }
            }
            Term::Branch {
                cond,
                then_bb,
                else_bb,
            } => {
                let then_bb = then_bb.filter(|t| {
                    if *t >= n {
                        cut += 1;
                        false
                    } else {
                        true
                    }
                });
                let else_bb = else_bb.filter(|t| {
                    if *t >= n {
                        cut += 1;
                        false
                    } else {
                        true
                    }
                });
                if then_bb.is_none() && else_bb.is_none() {
                    Term::Unreachable
                } else {
                    Term::Branch {
                        cond,
                        then_bb,
                        else_bb,
                    }
                }
            }
            Term::Switch {
                index,
                targets,
                first_case,
            } => {
                let kept: Vec<BlockId> = targets.iter().copied().filter(|t| *t < n).collect();
                cut += targets.len() - kept.len();
                if kept.len() == targets.len() {
                    Term::Switch {
                        index,
                        targets,
                        first_case,
                    }
                } else {
                    Term::Indirect { target: index }
                }
            }
            other => other,
        };
    }
    cut
}

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
        Term::Branch {
            then_bb, else_bb, ..
        } => then_bb.iter().chain(else_bb.iter()).copied().collect(),
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
        Term::Branch {
            then_bb, else_bb, ..
        } => {
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

/// A variable's name without the version SSA gave it.
///
/// `arg_4_u32.2` is the second value the slot held. The argument itself is
/// the first one, and a function that stores to an argument slot before
/// reading it back still takes that argument.
fn base_name(v: &str) -> &str {
    match v.find('.') {
        Some(i) => &v[..i],
        None => v,
    }
}

/// Which incoming registers and stack slots the body actually reads.
///
/// The locations are the architecture's argument registers in ABI order,
/// followed by the memory slots it passes arguments in; the names come from
/// the architecture, not from a table here. `declared` is the count the
/// epilogue states, when it states one.
fn parameter_names(
    blocks: &[Block],
    arg_regs: &[String],
    memory_args: &[String],
    declared: Option<usize>,
) -> (Vec<String>, Vec<String>) {
    let locations: Vec<String> = arg_regs.iter().chain(memory_args).cloned().collect();
    let index_of = |v: &str| -> Option<usize> {
        // A register counts only as the incoming value: `rdi.2` was written
        // inside the function. A frame slot is the argument itself however
        // many times the body overwrites it, so its versions still name it.
        if let Some(i) = arg_regs.iter().position(|r| r == v) {
            return Some(i);
        }
        let base = base_name(v);
        memory_args
            .iter()
            .position(|m| m == base)
            .map(|i| arg_regs.len() + i)
    };
    let mut used = vec![false; locations.len()];
    fn scan(e: &Expr, used: &mut [bool], index_of: &dyn Fn(&str) -> Option<usize>) {
        let mut vars = Vec::new();
        e.vars(&mut vars);
        for v in vars {
            if let Some(i) = index_of(&v) {
                used[i] = true;
            }
        }
    }
    fn walk(
        nodes: &[Stmt],
        used: &mut [bool],
        index_of: &dyn Fn(&str) -> Option<usize>,
        term: Option<&Term>,
    ) {
        for s in nodes {
            match s {
                Stmt::Assign { value, .. } => scan(value, used, index_of),
                Stmt::Store { addr, value, .. } => {
                    scan(addr, used, index_of);
                    scan(value, used, index_of);
                }
                Stmt::Call { args, .. } => {
                    for a in args {
                        scan(a, used, index_of);
                    }
                }
                Stmt::If {
                    cond,
                    then_body,
                    else_body,
                } => {
                    scan(cond, used, index_of);
                    walk(then_body, used, index_of, None);
                    walk(else_body, used, index_of, None);
                }
                Stmt::Comment(_) => {}
            }
        }
        let Some(term) = term else { return };
        if let Term::Branch { cond, .. } = term {
            scan(cond, used, index_of);
        }
        if let Term::Switch { index, .. } = term {
            scan(index, used, index_of);
        }
        if let Term::Ret(Some(v)) = term {
            scan(v, used, index_of);
        }
    }
    for b in blocks {
        walk(&b.stmts, &mut used, &index_of, Some(&b.term));
    }
    // A parameter is a position, not a set: a function that reads the third
    // argument register takes three arguments, whether or not the body says
    // anything about the first two. Listing only the registers the body reads
    // printed `f(a2)` for a two-argument function, and a caller reading that
    // prototype passes its argument in the wrong register.
    // An epilogue that removes its own arguments states the count even when
    // the body never mentions the last one: a function that ignores its
    // second argument still has one, and its callers still push it.
    let mut arity = used.iter().rposition(|u| *u).map_or(0, |i| i + 1);
    if let Some(n) = declared {
        arity = arity.max(n);
    }
    let arity = arity.min(locations.len());
    let mut params = Vec::with_capacity(arity);
    let mut regs = Vec::with_capacity(arity);
    for (i, loc) in locations.iter().enumerate().take(arity) {
        params.push(format!("a{i}"));
        regs.push(loc.clone());
    }
    (params, regs)
}

/// Structural outline for architectures whose instructions are not lifted.
fn outline(func: &Function, name_of: &dyn Fn(Addr) -> String, arch: occ_core::Arch) -> Decompiled {
    let mut body: Vec<ir::Node> = Vec::new();
    body.push(ir::Node::Comment(format!(
        "body not lifted: the decompiler models {LIFTABLE}, this is {}",
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
        param_types: Vec::new(),
        param_aliases: Vec::new(),
        ret_type: Some(word_type(arch).to_string()),
        body,
        unstructured,
        notes: vec![format!(
            "lifting for {} is not implemented yet",
            arch.label()
        )],
        source: None,
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
                Term::Branch {
                    then_bb, else_bb, ..
                } => then_bb.iter().chain(else_bb.iter()).copied().collect(),
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

        // With a correct post-dominator tree the shared tail is a join, so
        // the two arms close into one `if`/`else` and no jump is needed at
        // all. Either way the rule is the same: a jump has to land on a
        // label that the body defines.
        for target in &jumped {
            assert!(
                defined.contains(target),
                "the jump to {target} has no label to land on; labels: {defined:?}"
            );
            let labelled = body
                .iter()
                .any(|n| matches!(n, crate::ir::Node::Label(l) if l == target));
            assert!(labelled, "the label is in the body: {body:?}");
        }
        // And the tail's statement must appear exactly once, whether it was
        // inlined into both arms or emitted under a label.
        let assigns = format!("{body:?}").matches("Const(7)").count();
        assert_eq!(assigns, 1, "the shared tail runs once: {body:?}");
    }
}
