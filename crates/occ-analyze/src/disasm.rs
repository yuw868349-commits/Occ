//! Function recovery by recursive descent.
//!
//! Seeds come from every trustworthy statement the container makes about where
//! code starts: the entry point, `STT_FUNC` symbols with a size, exports,
//! initialiser arrays, and PLT stubs. Each seed is decoded block by block,
//! following direct branches and recording direct calls as new seeds.
//!
//! The hard part is not following branches, it is knowing when to *stop*.
//! Three bounds are applied, in order of strength:
//!
//! 1. A symbol size, when the container provides one. This is exact.
//! 2. Ownership: an address already inside a previously recovered function is
//!    not re-decoded. Sized functions are recovered first, so their exact
//!    bounds protect the unsized ones.
//! 3. A jump that leaves the function's byte range is a tail call, not an
//!    internal branch.
//!
//! Without (1) and (2) a single tail call merges two functions into one and
//! the reported size doubles; that failure is silent and looks like a large
//! function rather than an error, which is why the bounds are recorded as
//! evidence instead of being applied invisibly.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use occ_core::{
    Addr, Arch, ArmMode, BasicBlock, CallTarget, EdgeKind, Evidence, EvidenceKind, EvidenceSource,
    FlowKind, Function, Image, ImportKind, Instruction, RelocKind, SectionKind, Symbol,
    SymbolKind, Xref, XrefKind,
};
use occ_decode::AnyDecoder;

use crate::cfg::{BranchTargets, build_blocks_with_branch_targets};
use crate::view::View;

/// Decode one instruction at `pc`.
///
/// ARM is the one architecture whose instruction set changes inside a
/// binary: a file can hold a function compiled as A32 beside one compiled
/// as T32, and the file itself says where the change is.  The decoder is
/// therefore chosen by the address rather than by the image, or half of a
/// Thumb binary is read as ARM words that are not instructions.
///
/// An address the file marks as data holds no instruction, and the decoder
/// says so by declining to produce one.
fn decode_at(view: &View, dec: &AnyDecoder, pc: Addr, bytes: &[u8]) -> Option<occ_decode::Decoded> {
    let AnyDecoder::Arm(_) = dec else {
        return dec.decode(pc, bytes);
    };
    match view.image().arm_mode_at(pc).unwrap_or(ArmMode::A32) {
        ArmMode::T32 => occ_decode::arm::Decoder::with_mode(occ_decode::arm::Mode::Thumb)
            .decode(pc, bytes),
        ArmMode::A32 => occ_decode::arm::Decoder::with_mode(occ_decode::arm::Mode::Arm)
            .decode(pc, bytes),
        ArmMode::Data => None,
    }
}

/// Limits and switches for recovery.
#[derive(Clone, Debug)]
pub struct RecoverOptions {
    /// Stop after this many functions.
    pub max_functions: usize,
    /// Stop after this many instructions in total.
    pub max_insns: usize,
    /// Add newly seen direct call targets to the work list.
    pub follow_calls: bool,
    /// Sweep executable bytes that no seed covered, using a prologue check.
    pub sweep_uncovered: bool,
    /// Stop the sweep after examining this many bytes.
    ///
    /// The sweep is the one pass that walks a section one byte at a time, so
    /// its cost is proportional to the sizes the container claims rather than
    /// to the code that exists. A cap turns a hostile header from a hang into
    /// a note.
    pub max_sweep_bytes: usize,
}

impl Default for RecoverOptions {
    fn default() -> Self {
        RecoverOptions {
            max_functions: 20_000,
            max_insns: 4_000_000,
            follow_calls: true,
            sweep_uncovered: true,
            max_sweep_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Everything recovery produced.
pub struct Recovery {
    /// Recovered functions, sorted by entry address.
    pub functions: Vec<Function>,
    /// Cross references, deduplicated.
    pub xrefs: Vec<Xref>,
    /// PLT-style stubs mapped to the import they forward to.
    pub thunks: Vec<(Addr, String)>,
    /// Instructions decoded in total.
    pub decoded_insns: u64,
    /// Bytes of executable content in the image.
    pub code_bytes: u64,
    /// Human-readable notes about decisions that affected the result.
    pub notes: Vec<String>,
}

#[derive(Clone)]
struct Seed {
    addr: Addr,
    size: Option<u64>,
    name: Option<String>,
    /// The container table that stated this extent, when one did.
    source: Option<&'static str>,
}

/// Recover functions by recursive descent over a native instruction decoder.
fn decode_native(
    view: &View,
    dec: &AnyDecoder,
    opts: &RecoverOptions,
    notes: &mut Vec<String>,
) -> (Vec<Function>, Vec<(Addr, String)>) {
    let thunks = find_thunks(view, dec);
    let mut seeds = collect_seeds(view, &thunks);
    if seeds.is_empty() {
        notes.push("image provides no code entry point or function symbols".into());
    }

    let mut functions: Vec<Function> = Vec::new();
    let mut owned: BTreeMap<u64, u64> = BTreeMap::new();
    let mut queue: VecDeque<Seed> = VecDeque::new();
    let mut seen: HashSet<u64> = HashSet::new();

    // Sized seeds first: their exact bounds keep unsized ones from growing into
    // their neighbours.
    seeds.sort_by_key(|s| (s.size.is_none(), s.addr.0));
    for s in seeds.iter() {
        if seen.insert(s.addr.0) {
            queue.push_back(s.clone());
        }
    }

    // Where one function has to stop. A size hint from the symbol table is the
    // best answer when it is present; otherwise the next known entry is. Using
    // the end of the section instead lets a jump into unrelated code extend a
    // body by kilobytes, which shows up as a function whose size dwarfs its
    // instruction count and whose range swallows its neighbours.
    let mut bounds: Vec<u64> = seeds.iter().map(|s| s.addr.0).collect();
    bounds.sort_unstable();
    bounds.dedup();
    let upper_bound = |addr: Addr| -> Addr {
        match bounds.partition_point(|&b| b <= addr.0) {
            i if i < bounds.len() => Addr(bounds[i]),
            _ => section_end(view, addr).unwrap_or_else(|| addr.add(0x10_0000)),
        }
    };

    let mut total_insns = 0usize;
    let mut budget_hit = false;

    'outer: while let Some(seed) = queue.pop_front() {
        if functions.len() >= opts.max_functions {
            notes.push(format!(
                "stopped at max_functions={}; {} seeds were not recovered",
                opts.max_functions,
                queue.len() + 1
            ));
            break;
        }
        if inside_owned(&owned, seed.addr) {
            continue;
        }
        let next_entry = upper_bound(seed.addr);
        let limit = match seed.size {
            Some(size) if size > 0 => {
                let hinted = seed.addr.add(size);
                if hinted < next_entry { hinted } else { next_entry }
            }
            _ => next_entry,
        };
        let outcome = decode_function(view, dec, seed.addr, seed.size, limit, &owned, opts);
        if outcome.func.insns.is_empty() {
            continue;
        }
        total_insns += outcome.func.insns.len();
        if total_insns >= opts.max_insns {
            notes.push(format!(
                "stopped at max_insns={} after {} functions",
                opts.max_insns,
                functions.len() + 1
            ));
            budget_hit = true;
        }
        let mut f = outcome.func;
        if let Some(source) = seed.source {
            f.boundary_confidence = 1.0;
            f.boundary_evidence.push(Evidence::support(
                EvidenceKind::Symbol,
                EvidenceSource::Loader,
                format!(
                    "{source} states a {}-byte body at {:#x}",
                    f.size, f.entry.0
                ),
                1.0,
            ));
        }
        f.name = seed.name.or(f.name);
        f.is_thunk = is_thunk(&f);
        owned.insert(f.entry.0, f.end().0);
        for callee in outcome.callees {
            if opts.follow_calls
                && !inside_owned(&owned, callee)
                && seen.insert(callee.0)
                && view.is_exec(callee)
            {
                queue.push_back(Seed {
                    addr: callee,
                    size: None,
                    name: None,
                    source: None,
                });
            }
        }
        functions.push(f);
        if budget_hit && total_insns >= opts.max_insns {
            break 'outer;
        }
    }

    if opts.sweep_uncovered {
        sweep(view, dec, &mut functions, &mut owned, opts, notes);
    }

    (functions, thunks)
}

/// Recover the functions of a WebAssembly module.
///
/// There is no address space to search and nothing to infer: the container
/// names one symbol per defined function with the body's offset and length, and
/// the body says where it ends. Recovery is a walk over those symbols, which is
/// why every boundary here carries full confidence — guessing would be inventing
/// a problem that does not exist.
fn wasm_functions(view: &View, opts: &RecoverOptions, notes: &mut Vec<String>) -> Vec<Function> {
    let image = view.image();
    let data = view.file();
    // Calls address functions by index, and the index space starts with the
    // imported ones. The module states which index each body has, so that
    // mapping is read from the module rather than inferred from the order
    // the bodies happen to sit in: a code section whose bodies are not in
    // index order would otherwise send every call to the wrong function.
    let imported = image
        .imports
        .iter()
        .filter(|i| i.kind == ImportKind::Normal)
        .count() as u32;
    let mut bodies: Vec<&Symbol> = image
        .symbols
        .iter()
        .filter(|s| s.kind == SymbolKind::Function && s.defined && s.size > 0)
        .collect();
    bodies.sort_by_key(|s| s.addr.0);
    let mut by_index: HashMap<u32, Addr> = HashMap::new();
    for (index, sig) in image.wasm_funcs.iter().enumerate() {
        if let Some(body) = sig.body {
            by_index.insert(index as u32, body);
        }
    }
    if by_index.is_empty() {
        // The module carried no signature table. Falling back to the order
        // the bodies sit in is the best that can be done, and it is right
        // for every module whose code section is in index order.
        for (k, sym) in bodies.iter().enumerate() {
            by_index.insert(imported + k as u32, sym.addr);
        }
    }
    let call_target = |index: u32| by_index.get(&index).copied();

    let mut functions = Vec::new();
    for sym in bodies.iter().take(opts.max_functions) {
        let start = sym.addr.0;
        let end = start.saturating_add(sym.size);
        let slice = usize::try_from(start)
            .ok()
            .zip(usize::try_from(end).ok())
            .and_then(|(a, b)| data.get(a..b));
        let Some(slice) = slice else { continue };
        let body = occ_decode::wasm::decode(slice, start, &call_target, &image.wasm_types);
        if body.insns.is_empty() {
            continue;
        }
        if let Some((at, why)) = &body.truncated {
            notes.push(format!("{}: {at}: {why}", sym.name));
        }
        // `br_table` names one label per case; the map keeps them attached to
        // the instruction that chose between them instead of flattening them,
        // so each branch gets its own edges.
        let targets: BranchTargets = body
            .targets
            .iter()
            .map(|(at, list)| {
                let mut list = list.clone();
                list.sort_by_key(|a| a.get());
                list.dedup();
                (at.0, list)
            })
            .collect();

        let mut f = Function::new(sym.addr);
        f.name = Some(sym.name.clone());
        f.size = sym.size;
        f.blocks = build_blocks_with_branch_targets(&body.insns, sym.addr, &targets);
        f.insns = body.insns;
        f.boundary_confidence = 1.0;
        f.boundary_evidence.push(Evidence::support(
            EvidenceKind::Symbol,
            EvidenceSource::Loader,
            format!(
                "the code section declares a {size}-byte body at {start:#x}",
                size = sym.size
            ),
            1.0,
        ));
        functions.push(f);
    }
    functions
}

/// Recover functions, basic blocks, call edges and cross references.
pub fn recover(view: &View, opts: &RecoverOptions) -> Recovery {
    let image = view.image();
    let mut notes = Vec::new();
    let mut decoder: Option<AnyDecoder> = None;
    let (mut functions, thunks) = if image.arch == Arch::Wasm {
        // A module has no address space to search: every function body is named
        // by a symbol and its extent is declared by the code section.
        (wasm_functions(view, opts, &mut notes), Vec::new())
    } else {
        match AnyDecoder::new(image.arch, image.bits) {
            Some(dec) => {
                let recovered = decode_native(view, &dec, opts, &mut notes);
                decoder = Some(dec);
                recovered
            }
            None => {
                notes.push(format!(
                    "no decoder for {} {}-bit; function recovery skipped",
                    image.arch.label(),
                    image.bits
                ));
                return Recovery {
                    functions: Vec::new(),
                    xrefs: Vec::new(),
                    thunks: Vec::new(),
                    decoded_insns: 0,
                    code_bytes: code_bytes(view),
                    notes,
                };
            }
        }
    };

    if let Some(dec) = &decoder
        && let Some(main_addr) = startup_main(view, dec, &functions)
    {
        let known = functions.iter().any(|f| f.entry == main_addr);
        if !known {
            let owned: BTreeMap<u64, u64> =
                functions.iter().map(|f| (f.entry.0, f.end().0)).collect();
            let limit = functions
                .iter()
                .filter(|f| f.entry > main_addr)
                .map(|f| f.entry.0)
                .min()
                .map(Addr)
                .unwrap_or_else(|| {
                    section_end(view, main_addr).unwrap_or_else(|| main_addr.add(0x10_0000))
                });
            let out = decode_function(view, dec, main_addr, None, limit, &owned, opts);
            if out.func.insns.len() >= 2 {
                functions.push(out.func);
            }
        }
        if let Some(f) = functions.iter_mut().find(|f| f.entry == main_addr) {
            if f.name.is_none() {
                f.name = Some("main".to_string());
            }
            notes.push(format!(
                "main is {main_addr} — the address the entry point passes to the C runtime as \"main\""
            ));
        }
    }

    functions.sort_by_key(|f| f.entry.0);
    resolve_overlaps(&mut functions);
    if let Some(dec) = &decoder {
        let filled = fill_holes(view, dec, &mut functions, opts);
        if filled > 0 {
            notes.push(format!(
                "decoded {filled} instruction(s) behind jumps that landed on bytes no block had claimed"
            ));
        }
    }
    // A function reached only through a call has no seed name; the symbol table
    // may still know it. Filling names here costs one hash lookup per function
    // and turns `sub_4d84` into `main` on every non-stripped image.
    for f in &mut functions {
        if f.name.is_none()
            && let Some(sym) = image.symbol_at(f.entry)
            && sym.kind == SymbolKind::Function
        {
            f.name = Some(sym.name.clone());
        }
    }

    // Second pass: attach callers and resolve names from symbols.
    let mut callers: HashMap<u64, Vec<Addr>> = HashMap::new();
    let mut index: HashMap<u64, usize> = HashMap::new();
    for (i, f) in functions.iter().enumerate() {
        index.insert(f.entry.0, i);
    }
    let by_thunk: HashMap<u64, String> = thunks.iter().map(|(a, n)| (a.0, n.clone())).collect();
    // A PLT stub is the import's local alias. Naming it after the import makes
    // every call site read as the function it really reaches; the `@plt` suffix
    // keeps it distinguishable from an implementation in this image.
    for f in &mut functions {
        if f.name.is_none()
            && let Some(import) = by_thunk.get(&f.entry.0)
        {
            f.name = Some(format!("{import}@plt"));
        }
    }
    for f in &mut functions {
        f.callees.clear();
        for insn in &f.insns {
            if !insn.flow.is_call() {
                continue;
            }
            if let Some(CallTarget::Direct { to }) = &insn.target {
                f.callees.push(*to);
                callers.entry(to.0).or_default().push(f.entry);
            }
            if let Some(CallTarget::Direct { to }) = &insn.target
                && let Some(name) = by_thunk.get(&to.0)
                && !f.import_calls.contains(name)
            {
                f.import_calls.push(name.clone());
            }
            if let Some(CallTarget::Indirect { via: Some(slot) }) = &insn.target
                && let Some(imp) = image.import_by_slot(*slot)
                && !f.import_calls.contains(&imp.name)
            {
                f.import_calls.push(imp.name.clone());
            }
        }
        f.callees.sort_by_key(|a| a.0);
        f.callees.dedup();
    }
    for f in &mut functions {
        f.callers = callers.remove(&f.entry.0).unwrap_or_default();
        f.callers.sort_by_key(|a| a.0);
        f.callers.dedup();
    }

    let xrefs = build_xrefs(&functions, &thunks, image);
    let decoded_insns = functions.iter().map(|f| f.insns.len() as u64).sum();

    if !image.sections.iter().any(|s| s.kind == SectionKind::Code) {
        notes.push("no section is marked executable; recovery ran on segments only".into());
    }
    let code = code_bytes(view);
    if code > 0 {
        let covered: u64 = functions.iter().map(|f| f.size.max(f.end().0 - f.entry.0)).sum();
        let pct = 100.0 * covered as f64 / code as f64;
        if pct < 40.0 {
            notes.push(format!(
                "only {pct:.1}% of executable bytes fall inside a recovered function; \
                 the image is probably stripped and indirect-only"
            ));
        }
    }

    Recovery {
        functions,
        xrefs,
        thunks,
        decoded_insns,
        code_bytes: code,
        notes,
    }
}

struct Outcome {
    func: Function,
    callees: Vec<Addr>,
}

/// The conditions an `it` block imposes on the instructions after it.
///
/// Thumb has no predicated instruction. The condition of a Thumb
/// instruction lives in the `it` block that governs it, and the block
/// governs the one to four instructions that follow it in address order.
///
/// The conditions have to be known *while* the body is being read. A
/// conditional return inside one is a branch, and a branch that is read as
/// a plain return ends the recovery there: the whole rest of the function
/// was never decoded. The conditions are therefore applied to each
/// instruction as it is decoded, not to the finished list afterwards.
struct ItConditions {
    /// The condition of each instruction the block governs, in order.
    pending: Vec<&'static str>,
    /// The address the next governed instruction has to be at.
    ///
    /// The architecture says the governed instructions are the ones that
    /// follow the block, so a gap in the addresses ends the block: the
    /// instructions after the gap are not among the ones it governs.
    next: Addr,
}

impl ItConditions {
    /// Read the conditions an `it` instruction imposes.
    ///
    /// The block's own condition governs the first instruction, and the
    /// letters in its name say what the ones after it do: `itt` runs the
    /// second under the same condition and `ite` runs it under the
    /// opposite one.
    fn from_it(text: &str, next: Addr) -> Option<ItConditions> {
        let block = occ_decode::arm::it_block(text)?;
        let mut pending = Vec::with_capacity(block.conditions.len() + 1);
        pending.push(block.condition);
        pending.extend_from_slice(&block.conditions);
        Some(ItConditions { pending, next })
    }

    /// The condition the instruction at `addr` runs under, if it is one the
    /// block governs.
    ///
    /// The condition is consumed when it is given out: each instruction of
    /// the block has its own, and one that runs under the block's
    /// condition and the next that runs under the opposite one are
    /// different arms of the same choice.
    fn take(&mut self, addr: Addr) -> Option<&'static str> {
        if self.next.0 != addr.0 {
            return None;
        }
        let condition = self.pending.first().copied()?;
        self.pending.remove(0);
        Some(condition)
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_function(
    view: &View,
    dec: &AnyDecoder,
    entry: Addr,
    size_hint: Option<u64>,
    limit: Addr,
    owned: &BTreeMap<u64, u64>,
    opts: &RecoverOptions,
) -> Outcome {

    let mut insns: BTreeMap<u64, Instruction> = BTreeMap::new();
    let mut case_targets: BranchTargets = BranchTargets::new();
    let mut tables: Vec<occ_core::JumpTable> = Vec::new();
    let mut work: VecDeque<Addr> = VecDeque::new();
    let mut visited: HashSet<u64> = HashSet::new();
    let mut callees: Vec<Addr> = Vec::new();
    let mut budget = opts.max_insns.min(200_000);
    // The conditions of the `it` block whose instructions are being read.
    //
    // The block is Thumb's only way of saying "this happens on one path",
    // and the path it names is decided by a condition that is not on the
    // instruction itself. The condition has to be in the instruction's own
    // name before its flow is read, because a conditional return is a
    // branch: read as a plain return it ends the function there, and the
    // body after it is never decoded.
    let mut it: Option<ItConditions> = None;

    work.push_back(entry);
    while let Some(start) = work.pop_front() {
        if !visited.insert(start.0) {
            continue;
        }
        let mut pc = start;
        // Where this block began, so a computed jump can look back at the
        // instructions that built its target.
        let block_start = start;
        loop {
            if pc.0 >= limit.0 || budget == 0 {
                break;
            }
            if pc != entry && owned_past(owned, pc, entry) {
                break;
            }
            if insns.contains_key(&pc.0) {
                break;
            }
            let Some(bytes) = view.bytes_to_end(pc) else {
                break;
            };
            let Some(decoded) = decode_at(view, dec, pc, bytes) else {
                break;
            };
            let mut insn = decoded.insn;
            let end = insn.end();
            if end.0 > limit.0 {
                break;
            }
            budget -= 1;
            // An `it` block is read before the condition is applied, so that
            // the block is recognised even when the block itself is the
            // instruction a previous block governs.
            let block = ItConditions::from_it(&insn.text, end);
            // A block that is broken by a gap governs nothing after the
            // gap: the instructions there are not among the ones it names.
            if it.as_ref().is_some_and(|c| c.next.0 != pc.0) {
                it = None;
            }
            if let Some(block) = it.as_mut()
                && let Some(condition) = block.take(pc)
            {
                occ_decode::arm::apply_condition(&mut insn, condition);
                // The next governed instruction is the one that follows
                // this one, and how far that is depends on how this one is
                // encoded.
                block.next = insn.end();
            }
            let flow = insn.flow;
            let target = insn.target.clone();
            let undecoded = insn.mnemonic.starts_with('.');
            insns.insert(pc.0, insn);
            pc = end;
            // A block that has just been read governs the instructions
            // that follow it.
            if let Some(block) = block {
                it = Some(block);
            }

            if undecoded {
                // A byte the decoder could not classify is skipped, not treated
                // as a function end: padding and jump tables live inside code.
                continue;
            }
            match flow {
                FlowKind::Next | FlowKind::Call | FlowKind::IndirectCall => {
                    if let Some(CallTarget::Direct { to }) = &target
                        && !to.is_null()
                        && *to != entry
                    {
                        callees.push(*to);
                    }
                }
                FlowKind::ConditionalBranch => {
                    if let Some(CallTarget::Direct { to }) = target
                        && to.0 >= entry.0
                        && to.0 < limit.0
                    {
                        work.push_back(to);
                    }
                }
                FlowKind::UnconditionalBranch => {
                    match target {
                        Some(CallTarget::Direct { to })
                            if to.0 >= entry.0
                                && to.0 < limit.0
                                && !(to.0 > pc.0 && owned_past(owned, to, entry)) =>
                        {
                            // Backward jumps are loops; forward jumps inside
                            // the bound are if/else joins. A jump to something
                            // that reads as a function entry — a known body, a
                            // prologue, or an address the compiler padded up
                            // to — is a tail call. Following it folds the other
                            // function into this one and stretches the range
                            // over everything between them.
                            if to != entry
                                && (owned_past(owned, to, entry)
                                    || padding_before(view, to)
                                    || looks_like_prologue(view, to))
                            {
                                callees.push(to);
                                break;
                            }
                            work.push_back(to);
                            break;
                        }
                        Some(CallTarget::Direct { to }) => {
                            if to != entry {
                                callees.push(to);
                            }
                            break;
                        }
                        _ => break,
                    }
                }
                FlowKind::IndirectBranch => {
                    // A `switch` reaches its case bodies only through this
                    // jump. Resolving the table is what keeps the function from
                    // ending here with three quarters of its code missing.
                    let window: Vec<&Instruction> =
                        insns.range(block_start.0..pc.0).map(|(_, i)| i).collect();
                    if let Some(jmp) = window.last().copied()
                        && let Some(table) = crate::jumptable::resolve(view, &window, jmp)
                    {
                        let mut reached = 0usize;
                        for target in &table.targets {
                            if target.0 >= entry.0
                                && target.0 < limit.0
                                && !inside_owned(owned, *target)
                            {
                                work.push_back(*target);
                                reached += 1;
                            }
                        }
                        if reached > 0 {
                            case_targets.insert(jmp.addr.0, table.targets.clone());
                            tables.push(table);
                        }
                    }
                    break;
                }
                // A syscall returns to the next instruction; only a trap that
                // never comes back ends the walk.
                FlowKind::Interrupt => {}
                FlowKind::Return | FlowKind::Exception => break,
            }
        }
    }

    let mut list: Vec<Instruction> = insns.into_values().collect();
    list.sort_by_key(|i| i.addr.0);
    resolve_addresses(view, dec, &mut list);
    // The conditions of an `it` block were written into the instructions as
    // they were read, because a conditional return inside one is a branch
    // and a branch that is read as a return ends the function there. This
    // pass is the safety net for the one case the reading cannot see: a
    // block whose governed instruction was read in a different walk.
    occ_decode::arm::apply_it_conditions(&mut list);

    let mut func = Function::new(entry);
    func.blocks = build_blocks_with_branch_targets(&list, entry, &case_targets);
    func.jump_tables = tables;
    func.size = 0;
    func.insns = list;
    if func.insns.is_empty() {
        return Outcome {
            func,
            callees: Vec::new(),
        };
    }
    let end = func.insns.last().map(|i| i.end()).unwrap_or(entry);
    func.size = end.0.saturating_sub(entry.0);
    let (confidence, evidence) = boundary_evidence(view, entry, size_hint, &func);
    func.boundary_confidence = confidence;
    func.boundary_evidence = evidence;
    callees.sort_by_key(|a| a.0);
    callees.dedup();
    Outcome {
        func,
        callees,
    }
}

/// Make every function's body and range exclusive.
//////
/// A decode can jump over an already-known body to a block laid out beyond it,
/// and the bytes in between belong to that other function. Leaving them in this
/// body makes two functions claim one address: one of them then reports a size
/// its instructions do not fill, and the other reports instructions outside its
/// own range. The later entry is the more specific claim — a call says "a
/// function starts here" — so the earlier body gives up everything from there
/// on, and its block graph is rebuilt from what remains.
fn resolve_overlaps(functions: &mut [Function]) {
    for i in 0..functions.len() {
        let start = functions[i].entry.0;
        let end = functions[i].end().0;
        if end <= start {
            continue;
        }
        // Entries are sorted, so the first following one is the only claim
        // that can start inside this body.
        let Some(next) = functions[i + 1..]
            .iter()
            .map(|f| f.entry.0)
            .find(|&next| next < end)
        else {
            continue;
        };
        functions[i].insns.retain(|insn| insn.addr.0 < next);
        let f = &mut functions[i];
        let mut targets: BranchTargets = BranchTargets::new();
        for table in &f.jump_tables {
            targets.insert(table.at.0, table.targets.clone());
        }
        f.blocks = build_blocks_with_branch_targets(&f.insns, f.entry, &targets);
        let end = f.insns.last().map(|x| x.end().0).unwrap_or(start);
        f.size = end.saturating_sub(start);
    }
}

/// Decode the bytes a function jumps into but never claimed.
///
/// Recursive descent follows a branch only while the target still looks like
/// this function's own code, and an aligned block behind padding looks like
/// somebody else's entry — so the jump survives and the bytes behind it do not.
/// The listing then has an edge that leads nowhere. By this point every range
/// is final, so the question "whose code is this" has an answer: a target
/// inside `[entry, end)` that no instruction covers belongs to this function.
///
/// Only bytes reached from such a target are decoded, and a chain is kept only
/// when it arrives at an instruction the function already had or ends on a
/// terminator. A hole that is really a jump table therefore stays a hole: its
/// bytes do not decode into a path with two ends.
fn fill_holes(
    view: &View,
    dec: &AnyDecoder,
    functions: &mut [Function],
    opts: &RecoverOptions,
) -> usize {
    let mut added = 0usize;
    for f in functions.iter_mut() {
        let start = f.entry.0;
        let end = f.end().0;
        if end <= start || f.insns.is_empty() {
            continue;
        }
        let mut have: HashSet<u64> = f.insns.iter().map(|i| i.addr.0).collect();
        let mut wanted: Vec<u64> = Vec::new();
        for insn in &f.insns {
            if let Some(CallTarget::Direct { to }) = &insn.target
                && to.0 > start
                && to.0 < end
                && !have.contains(&to.0)
            {
                wanted.push(to.0);
            }
        }
        if wanted.is_empty() {
            continue;
        }
        wanted.sort_unstable();
        wanted.dedup();
        let cap = opts.max_insns.min(4096);
        let mut extra: Vec<Instruction> = Vec::new();
        for target in wanted {
            if have.contains(&target) {
                continue;
            }
            let mut chain: Vec<Instruction> = Vec::new();
            let mut pc = Addr(target);
            let mut connected = false;
            while pc.0 < end && !have.contains(&pc.0) && chain.len() < cap {
                let Some(bytes) = view.bytes_to_end(pc) else {
                    break;
                };
                let Some(decoded) = decode_at(view, dec, pc, bytes) else {
                    break;
                };
                let insn = decoded.insn;
                let next = insn.end();
                if next.0 > end {
                    break;
                }
                let terminal = matches!(
                    insn.flow,
                    FlowKind::Return | FlowKind::UnconditionalBranch | FlowKind::IndirectBranch
                );
                chain.push(insn);
                pc = next;
                if terminal {
                    break;
                }
                if have.contains(&pc.0) {
                    connected = true;
                    break;
                }
            }
            let ends_on_terminator = chain.last().is_some_and(|i| {
                matches!(
                    i.flow,
                    FlowKind::Return | FlowKind::UnconditionalBranch | FlowKind::IndirectBranch
                )
            });
            if chain.is_empty() || !(connected || ends_on_terminator) {
                continue;
            }
            for insn in &chain {
                have.insert(insn.addr.0);
            }
            added += chain.len();
            extra.extend(chain);
        }
        if extra.is_empty() {
            continue;
        }
        f.insns.extend(extra);
        f.insns.sort_by_key(|i| i.addr.0);
        f.insns.dedup_by_key(|i| i.addr.0);
        let mut targets: BranchTargets = BranchTargets::new();
        for table in &f.jump_tables {
            targets.insert(table.at.0, table.targets.clone());
        }
        f.blocks = build_blocks_with_branch_targets(&f.insns, f.entry, &targets);
        let end = f.insns.last().map(|x| x.end().0).unwrap_or(start);
        f.size = end.saturating_sub(start);
    }
    added
}

/// Confidence that the recovered range is exactly one function.
fn boundary_evidence(
    view: &View,
    entry: Addr,
    size_hint: Option<u64>,
    func: &Function,
) -> (f32, Vec<Evidence>) {
    let mut ev = Vec::new();
    if view.image().entry == entry && !entry.is_null() {
        // The header names this address as the first instruction the loader
        // jumps to, which is stronger than any shape heuristic: a function
        // starts here even when nothing else about it is known.
        ev.push(
            Evidence::support(
                EvidenceKind::FormatMeta,
                EvidenceSource::Loader,
                "the image header names this address as the entry point",
                0.8,
            )
            .at(entry),
        );
    }
    if let Some(size) = size_hint
        && size > 0
        && size == func.size
    {
        ev.push(
            Evidence::support(
                EvidenceKind::Symbol,
                EvidenceSource::Loader,
                format!("symbol declares a {size}-byte function at this address"),
                0.9,
            )
            .at(entry),
        );
    } else if let Some(sym) = view.image().symbol_at(entry)
        && sym.kind == SymbolKind::Function
    {
        ev.push(
            Evidence::support(
                EvidenceKind::Symbol,
                EvidenceSource::Loader,
                format!("{} is typed as a function", sym.name),
                0.5,
            )
            .at(entry),
        );
    }
    if looks_like_prologue(view, entry) {
        ev.push(
            Evidence::support(
                EvidenceKind::ControlFlow,
                EvidenceSource::Heuristic,
                "entry matches a compiler prologue",
                0.2,
            )
            .at(entry),
        );
    }
    if !func.callers.is_empty() {
        ev.push(
            Evidence::support(
                EvidenceKind::CrossReference,
                EvidenceSource::Heuristic,
                "at least one direct call targets this entry",
                0.3,
            )
            .at(entry),
        );
    }
    (occ_core::combine_confidence(&ev), ev)
}

fn section_end(view: &View, addr: Addr) -> Option<Addr> {
    view.section_at(addr).map(|s| s.addr.add(s.size))
}

fn inside_owned(owned: &BTreeMap<u64, u64>, addr: Addr) -> bool {
    if let Some((_, &end)) = owned.range(..=addr.0).next_back()
        && addr.0 < end
    {
        return true;
    }
    false
}

/// True when `addr` lies in a body that belongs to a function starting at or
/// after `entry`.
///
/// A range that starts *before* the function being decoded cannot be a reason
/// to stop: the later entry wins, [`resolve_overlaps`] clips the earlier body
/// at this entry, and every byte from `entry` on is this function's. Treating
/// those bytes as somebody else's left a hole — valid instructions that no
/// block claimed, with a jump into them that the lift could only describe.
fn owned_past(owned: &BTreeMap<u64, u64>, addr: Addr, entry: Addr) -> bool {
    if let Some((&start, &end)) = owned.range(..=addr.0).next_back()
        && start > entry.0
        && addr.0 < end
    {
        return true;
    }
    false
}

fn code_bytes(view: &View) -> u64 {
    view.image()
        .sections
        .iter()
        .filter(|s| s.is_code())
        .map(|s| s.size)
        .sum()
}

/// True when `addr` is a place a function body can start.
///
/// Zero is the null address in every container this tool reads, so a seed at 0
/// looks like a bug. It is not always one: a relocatable object has no load
/// base, `cc -c` puts `__TEXT,__text` at 0, and `_main` genuinely starts there.
/// Rejecting zero hides every function in an object file. The distinction that
/// matters is whether executable bytes are mapped at 0 — true for an object,
/// false for every linked image, whose text sits at its load base instead.
fn addressable(view: &View, addr: Addr) -> bool {
    view.is_exec(addr)
}

fn collect_seeds(view: &View, thunks: &[(Addr, String)]) -> Vec<Seed> {
    let image = view.image();
    let mut seeds = Vec::new();
    let mut push = |addr: Addr, size: Option<u64>, name: Option<String>| {
        if addressable(view, addr) {
            seeds.push(Seed {
                addr,
                size,
                name,
                source: None,
            });
        }
    };

    if !image.entry.is_null() {
        // An object has no entry point — its `entry` is just where the first
        // code section starts, and calling that `_start` would invent a symbol
        // the linker has not chosen yet.
        let name = (!image.segments.is_empty()).then(|| "_start".to_string());
        push(image.entry, None, name);
    }
    for sym in &image.symbols {
        if sym.kind == SymbolKind::Function && sym.defined {
            let size = (sym.size > 0).then_some(sym.size);
            push(sym.addr, size, Some(sym.name.clone()));
        }
    }
    for e in &image.exports {
        push(e.addr, None, Some(e.name.clone()));
    }
    for (addr, _) in thunks {
        push(*addr, None, None);
    }
    for name in [".init", ".fini", ".text.startup"] {
        if let Some(s) = image.section_by_name(name)
            && s.size > 0
        {
            push(s.addr, Some(s.size), Some(name.trim_start_matches('.').into()));
        }
    }
    // Initialiser arrays hold function pointers.
    for name in [".init_array", ".fini_array", ".preinit_array", ".ctors", ".dtors"] {
        let Some(s) = image.section_by_name(name) else {
            continue;
        };
        let width = usize::from(image.bits / 8);
        let count = (s.file_size as usize) / width;
        for i in 0..count {
            let slot = s.addr.add((i * width) as u64);
            let target = image
                .relocations
                .iter()
                .find(|r| r.offset == slot && matches!(r.kind, RelocKind::Relative | RelocKind::Absolute))
                .and_then(|r| r.addend)
                .map(|a| Addr(a as u64))
                .or_else(|| view.read_ptr(slot).map(Addr));
            if let Some(t) = target
                && addressable(view, t)
            {
                push(t, None, None);
            }
        }
    }
    drop(push);
    // Unwind tables and function-start lists name extents outright. They are
    // seeds, not guesses, and they survive `strip`.
    for b in &image.function_bounds {
        if addressable(view, b.start) {
            let size = (b.end.0 > b.start.0).then_some(b.end.0 - b.start.0);
            seeds.push(Seed {
                addr: b.start,
                size,
                name: None,
                source: Some(b.source),
            });
        }
    }
    seeds
}

/// PLT stubs and their imports.
///
/// A stub is a short sequence whose only meaningful content is a jump through
/// a GOT/IAT slot. Decoding the first few instructions and matching memory
/// operands against known import slots recovers the mapping without parsing
/// the container's PLT layout, which differs between linkers.
fn find_thunks(view: &View, dec: &AnyDecoder) -> Vec<(Addr, String)> {
    let image = view.image();
    let mut out = Vec::new();
    for s in &image.sections {
        if !(s.name.starts_with(".plt") || s.name.starts_with(".iplt")) || s.file_size == 0 {
            continue;
        }
        let mut addr = s.addr;
        let end = s.addr.add(s.file_size);
        while addr.0 < end.0 {
            let Some(bytes) = view.bytes(addr, 16.min((end.0 - addr.0) as usize)) else {
                break;
            };
            let mut pc = addr;
            let mut rest = bytes;
            for _ in 0..4 {
                let Some(d) = decode_at(view, dec, pc, rest) else { break };
                let mut hit = None;
                for m in d.insn.mem_reads.iter() {
                    if let Some(imp) = image.import_by_slot(*m) {
                        hit = Some(imp.name.clone());
                        break;
                    }
                }
                if let Some(name) = hit {
                    out.push((addr, name));
                    break;
                }
                let len = d.insn.len as usize;
                if len == 0 || len >= rest.len() {
                    break;
                }
                pc = pc.add(len as u64);
                rest = &rest[len..];
            }
            // Stubs are 16 bytes on every ABI in scope; stepping by 16 keeps a
            // failed decode from desynchronising the section scan.
            addr = addr.add(16);
        }
    }
    out.sort_by_key(|(a, _)| a.0);
    out.dedup_by_key(|(a, _)| a.0);
    out
}

/// Fill in memory addresses that need more than one instruction to compute.
///
/// On AArch64 an address is materialised as `adrp` + `add`/`ldr`; on RISC-V as
/// `auipc` + `addi`/`ld`. The decoder sees one instruction at a time and cannot
/// know the register holds a page base, so the sequence is resolved here, in
/// address order, and the resulting absolute address is written back onto the
/// instruction that dereferences it. Without this pass a disassembly of either
/// architecture reports no data references at all.
fn resolve_addresses(view: &View, dec: &AnyDecoder, insns: &mut [Instruction]) {
    match dec.arch() {
        occ_core::Arch::Aarch64 => resolve_aarch64(view, insns),
        occ_core::Arch::RiscV32 | occ_core::Arch::RiscV64 => resolve_riscv(view, insns),
        _ => {}
    }
}

fn resolve_aarch64(view: &View, insns: &mut [Instruction]) {
    let mut regs: [Option<u64>; 32] = [None; 32];
    for insn in insns.iter_mut() {
        let b = insn.bytes;
        let w = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let sf = w & 0x8000_0000 != 0;
        let rd = (w & 0x1f) as usize;
        let rn = ((w >> 5) & 0x1f) as usize;
        // ADRP
        if w & 0x9f00_0000 == 0x9000_0000 {
            let immlo = (w >> 29) & 3;
            let immhi = (w >> 5) & 0x7ffff;
            let imm = ((immhi << 2) | immlo) as i64;
            let imm = (imm << 43) >> 43;
            let page = (insn.addr.0 & !0xfff).wrapping_add((imm as u64) << 12);
            regs[rd] = Some(page);
            continue;
        }
        // ADD (immediate, 64-bit), no flags, shift 0 or 12.
        if w & 0x7f00_0000 == 0x1100_0000 || w & 0xff00_0000 == 0x9100_0000 {
            let shift = if w & 0x0040_0000 != 0 { 12 } else { 0 };
            let imm = u64::from((w >> 10) & 0xfff) << shift;
            if let Some(base) = regs[rn] {
                regs[rd] = Some(base.wrapping_add(imm));
            } else {
                regs[rd] = None;
            }
            continue;
        }
        // MOV (register) = ORR rd, xzr, rm
        if w & 0xffe0_ffe0 == 0xaa00_03e0 {
            let rm = ((w >> 16) & 0x1f) as usize;
            regs[rd] = regs[rm];
            continue;
        }
        // LDR/STR (unsigned immediate) and pair forms: resolve the operand.
        let top = (w >> 24) & 0x3f;
        if top == 0x39 || top == 0x38 {
            let size = w >> 30;
            let scale = size & 3;
            let sub = (w >> 10) & 3;
            if top == 0x39 || sub == 0 || sub == 1 || sub == 3 {
                let imm: i64 = if top == 0x39 {
                    (u64::from((w >> 10) & 0xfff) << scale) as i64
                } else {
                    ((((w >> 12) & 0x1ff) as i64) << 55) >> 55
                };
                if let Some(base) = regs[rn] {
                    let addr = base.wrapping_add_signed(imm);
                    let store = (w >> 22) & 3 == 0;
                    if view.is_mapped(Addr(addr)) {
                        if store {
                            insn.mem_writes.push(Addr(addr));
                        } else {
                            insn.mem_reads.push(Addr(addr));
                        }
                    }
                }
            }
            continue;
        }
        // Pair load/store: stp/ldp with signed offset.
        if (w >> 27) & 7 == 0b101 && (w >> 26) & 1 == 0 {
            let load = w & 0x0040_0000 != 0;
            let imm7 = ((w >> 15) & 0x7f) as i64;
            let imm7 = (imm7 << 57) >> 57;
            let opc = w >> 30;
            let width = if opc == 2 { 8 } else { 4 };
            if let Some(base) = regs[rn] {
                let addr = base.wrapping_add_signed(imm7 * width);
                if view.is_mapped(Addr(addr)) {
                    if load {
                        insn.mem_reads.push(Addr(addr));
                    } else {
                        insn.mem_writes.push(Addr(addr));
                    }
                }
            }
            continue;
        }
        // Any other instruction that writes rd invalidates the tracked value.
        if writes_rd(w, sf) {
            regs[rd] = None;
        }
    }
}

/// Conservative test for "this instruction writes its Rd".
fn writes_rd(w: u32, _sf: bool) -> bool {
    let op0 = (w >> 25) & 0xf;
    match op0 {
        0b1000 | 0b1001 => true,         // data processing immediate
        0b0101 | 0b1101 => true,         // data processing register
        _ => {
            // Loads write Rt; stores do not.
            let top = (w >> 24) & 0x3f;
            top == 0x39 && (w >> 22) & 3 != 0 || top == 0x38 && (w >> 22) & 3 != 0
        }
    }
}

fn resolve_riscv(view: &View, insns: &mut [Instruction]) {
    let mut regs: [Option<u64>; 32] = [None; 32];
    for insn in insns.iter_mut() {
        let b = insn.bytes;
        let w = if insn.len >= 4 {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u32::from(u16::from_le_bytes([b[0], b[1]]))
        };
        if insn.len == 2 {
            // Compressed instructions are not tracked; clear the destination
            // register conservatively.
            let rd = ((w >> 7) & 0x1f) as usize;
            regs[rd] = None;
            continue;
        }
        let opcode = w & 0x7f;
        let rd = ((w >> 7) & 0x1f) as usize;
        let rs1 = ((w >> 15) & 0x1f) as usize;
        match opcode {
            0x17 => {
                // AUIPC
                let imm = (w & 0xffff_f000) as i32 as i64;
                regs[rd] = Some(insn.addr.0.wrapping_add_signed(imm));
                continue;
            }
            0x13 => {
                // ADDI
                let imm = (((w >> 20) & 0xfff) as i64) << 52 >> 52;
                regs[rd] = regs[rs1].map(|b| b.wrapping_add_signed(imm));
                if rd == 0 {
                    regs[0] = None;
                }
                continue;
            }
            0x03 | 0x23 => {
                let funct3 = (w >> 12) & 7;
                let width = match funct3 & 3 {
                    0 => 1u64,
                    1 => 2,
                    2 => 4,
                    _ => 8,
                };
                let imm = if opcode == 0x03 {
                    (((w >> 20) & 0xfff) as i64) << 52 >> 52
                } else {
                    let hi = ((w >> 25) & 0x7f) as i64;
                    let lo = ((w >> 7) & 0x1f) as i64;
                    ((hi << 5) | lo) << 52 >> 52
                };
                if let Some(base) = regs[rs1] {
                    let addr = base.wrapping_add_signed(imm);
                    let store = opcode == 0x23;
                    if view.is_mapped(Addr(addr)) && width <= 8 {
                        if store {
                            insn.mem_writes.push(Addr(addr));
                        } else {
                            insn.mem_reads.push(Addr(addr));
                        }
                    }
                }
                if opcode == 0x03 {
                    regs[rd] = None;
                }
                continue;
            }
            _ => {
                if rd != 0 {
                    regs[rd] = None;
                }
            }
        }
    }
}

fn is_thunk(f: &Function) -> bool {
    if f.insns.is_empty() || f.insns.len() > 4 {
        return false;
    }
    let mut saw_transfer = false;
    for insn in &f.insns {
        match insn.flow {
            FlowKind::UnconditionalBranch | FlowKind::IndirectBranch => saw_transfer = true,
            FlowKind::Next => {}
            _ => return false,
        }
    }
    saw_transfer
}

fn build_xrefs(functions: &[Function], thunks: &[(Addr, String)], image: &Image) -> Vec<Xref> {
    let mut out: Vec<Xref> = Vec::new();
    let mut seen: HashSet<(u64, u64, u8)> = HashSet::new();
    let thunk_map: HashMap<u64, ()> = thunks.iter().map(|(a, _)| (a.0, ())).collect();
    let inside: HashMap<u64, u64> = functions
        .iter()
        .map(|f| (f.entry.0, f.entry.0 + f.size))
        .collect();

    let mut add = |from: Addr, to: Addr, kind: XrefKind, from_fn: Option<Addr>| {
        let tag = kind as u8;
        if !seen.insert((from.0, to.0, tag)) {
            return;
        }
        let mut x = Xref::new(from, to, kind);
        x.from_function = from_fn;
        out.push(x);
    };

    for f in functions {
        for insn in &f.insns {
            match (&insn.flow, &insn.target) {
                (FlowKind::Call, Some(CallTarget::Direct { to })) => {
                    add(insn.addr, *to, XrefKind::Call, Some(f.entry));
                }
                (FlowKind::IndirectCall, Some(CallTarget::Indirect { via: Some(slot) })) => {
                    add(insn.addr, *slot, XrefKind::Call, Some(f.entry));
                }
                (FlowKind::ConditionalBranch, Some(CallTarget::Direct { to })) => {
                    add(insn.addr, *to, XrefKind::ConditionalJump, Some(f.entry));
                }
                (FlowKind::UnconditionalBranch, Some(CallTarget::Direct { to })) => {
                    let internal = inside
                        .get(&f.entry.0)
                        .is_some_and(|end| to.0 >= f.entry.0 && to.0 < *end);
                    if internal {
                        add(insn.addr, *to, XrefKind::Jump, Some(f.entry));
                    } else {
                        add(insn.addr, *to, XrefKind::TailCall, Some(f.entry));
                    }
                    if thunk_map.contains_key(&to.0) {
                        add(insn.addr, *to, XrefKind::ImportThunk, Some(f.entry));
                    }
                }
                (FlowKind::IndirectBranch, Some(CallTarget::Indirect { via: Some(slot) })) => {
                    add(insn.addr, *slot, XrefKind::Jump, Some(f.entry));
                }
                _ => {}
            }
            for m in &insn.mem_reads {
                if !image.is_mapped(*m) {
                    continue;
                }
                let looks_code = image
                    .symbol_at(*m)
                    .map(|s| s.kind == SymbolKind::Function)
                    .unwrap_or(false);
                let kind = if looks_code {
                    XrefKind::AddressTaken
                } else {
                    XrefKind::DataRead
                };
                add(insn.addr, *m, kind, Some(f.entry));
            }
            for m in &insn.mem_writes {
                if image.is_mapped(*m) {
                    add(insn.addr, *m, XrefKind::DataWrite, Some(f.entry));
                }
            }
        }
    }
    out.sort_by_key(|x| (x.from.0, x.to.0));
    out
}

/// Heuristic prologue match, used only when no symbol covers the address.
/// Find `main` from the arguments the entry point hands the C runtime.
///
/// Every libc start-up path — glibc and musl, static or dynamic — calls
/// `__libc_start_main` with `main` as the first argument, and the entry point
/// has to put it there with an immediate or a PC-relative `lea`, because no
/// memory holds it yet. When the symbol table is stripped, that instruction is
/// the only record of where the program's own code begins. Without it, a
/// stripped static binary has no `main`: every tool that starts from `main`
/// reports "not found", and the analysis of the program's own code stops at the
/// entry point.
///
/// The tracker is deliberately narrow. It follows one register and gives up on
/// the first instruction that writes it in a form it does not model, so a
/// failed guess is a missing `main`, never a wrong one.
fn startup_main(view: &View, dec: &AnyDecoder, functions: &[Function]) -> Option<Addr> {
    let image = view.image();
    let entry = image.entry;
    if entry.is_null() || !view.is_exec(entry) {
        return None;
    }
    // Prefer the recovered `_start`: it has already been bounded, so it ends
    // before the next function instead of running into it. Fall back to a
    // bounded decode when recovery did not keep it.
    let owned: BTreeMap<u64, u64> = functions.iter().map(|f| (f.entry.0, f.end().0)).collect();
    let body: Vec<Instruction> = match functions.iter().find(|f| f.entry == entry) {
        Some(f) if f.insns.len() >= 2 => f.insns.clone(),
        _ => {
            let limit = section_end(view, entry).unwrap_or_else(|| entry.add(0x1000));
            decode_function(view, dec, entry, None, limit, &owned, &RecoverOptions::default())
                .func
                .insns
        }
    };
    if body.len() < 2 {
        return None;
    }
    let candidate = match image.arch {
        Arch::X86_64 | Arch::X86 => first_argument_x86(&body),
        Arch::Aarch64 => first_argument_aarch64(&body),
        Arch::RiscV32 | Arch::RiscV64 => first_argument_riscv(&body),
        _ => None,
    }?;
    let candidate = Addr(candidate);
    // It has to be code in this image, and not the entry point itself.
    if candidate == entry || !view.is_exec(candidate) {
        return None;
    }
    Some(candidate)
}

/// Strip legacy and REX prefixes, returning the REX byte when present.
fn x86_prefixes(bytes: &[u8]) -> (Option<u8>, &[u8]) {
    let mut i = 0;
    let mut rex = None;
    while i < bytes.len() {
        match bytes[i] {
            0x66 | 0x67 | 0xf0 | 0xf2 | 0xf3 | 0x2e | 0x3e | 0x26 | 0x36 | 0x64 | 0x65 => i += 1,
            0x40..=0x4f => {
                rex = Some(bytes[i]);
                i += 1;
            }
            _ => break,
        }
    }
    (rex, &bytes[i.min(bytes.len())..])
}

fn first_argument_x86(insns: &[Instruction]) -> Option<u64> {
    let mut tracked: Option<u64> = None;
    for insn in insns {
        let raw = &insn.bytes[..(insn.len as usize).min(insn.bytes.len())];
        let (rex, b) = x86_prefixes(raw);
        let wide = rex.is_some_and(|r| r & 0x08 != 0);
        if insn.flow.is_call() {
            return tracked;
        }
        if b.len() >= 2 && b[0] == 0xc7 && b[1] & 0xf8 == 0xc0 && b[1] & 0x07 == 0x07 {
            // mov rdi, imm32 — sign-extended to the operand size.
            if b.len() >= 6 {
                let imm = i32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                tracked = Some(imm as i64 as u64);
                continue;
            }
        }
        if b.len() == 5 && b[0] == 0xbf && !wide {
            // mov edi, imm32 — zero-extended.
            tracked = Some(u64::from(u32::from_le_bytes([b[1], b[2], b[3], b[4]])));
            continue;
        }
        if b.len() == 10 && b[0] == 0xbf && wide {
            tracked = Some(u64::from_le_bytes([
                b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8],
            ]));
            continue;
        }
        if b.len() == 6 && b[0] == 0x8d && b[1] == 0x3d {
            // lea rdi, [rip+disp32].
            let disp = i32::from_le_bytes([b[2], b[3], b[4], b[5]]) as i64;
            tracked = Some(insn.end().0.wrapping_add_signed(disp));
            continue;
        }
        if x86_writes_rdi(b) {
            tracked = None;
        }
    }
    None
}

/// True when the instruction's destination is `rdi`.
fn x86_writes_rdi(b: &[u8]) -> bool {
    let Some(&op) = b.first() else {
        return false;
    };
    // pop rdi
    if op == 0x5f {
        return true;
    }
    // The arithmetic and move opcodes whose `reg` field is a destination.
    let writes_reg = matches!(
        op,
        0x89 | 0x8b | 0x31 | 0x33 | 0x01 | 0x03 | 0x21 | 0x23 | 0x09 | 0x0b | 0x29 | 0x2b
    ) || matches!(op, 0x0f) && b.get(1).is_some_and(|v| matches!(v & 0xf0, 0xb0..=0xbf));
    if !writes_reg {
        return false;
    }
    let modrm = b[1];
    (modrm >> 3) & 7 == 7
}

fn first_argument_aarch64(insns: &[Instruction]) -> Option<u64> {
    let mut x0: Option<u64> = None;
    for insn in insns {
        if insn.flow.is_call() {
            return x0;
        }
        let b = &insn.bytes;
        let w = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        // adrp x0, label
        if w & 0x9f00_0000 == 0x9000_0000 && w & 0x1f == 0 {
            let immlo = ((w >> 29) & 3) as i64;
            let immhi = ((w >> 5) & 0x7ffff) as i64;
            let imm = ((immhi << 2) | immlo) << 43 >> 43;
            x0 = Some((insn.addr.0 & !0xfff).wrapping_add_signed(imm << 12));
            continue;
        }
        // add x0, x0, #imm
        if w & 0xffc0_03ff == 0x9100_0000 && w & 0x1f == 0 {
            let shift = if w & 0x0040_0000 != 0 { 12 } else { 0 };
            let imm = u64::from((w >> 10) & 0xfff) << shift;
            x0 = x0.map(|v| v.wrapping_add(imm));
            continue;
        }
        // movz x0, #imm16, lsl #n
        if w & 0xff80_0000 == 0xd280_0000 && w & 0x1f == 0 {
            let imm = u64::from((w >> 5) & 0xffff) << (((w >> 21) & 3) * 16);
            x0 = Some(imm);
            continue;
        }
        if (w & 0x1f) == 0 && writes_rd(w, true) {
            x0 = None;
        }
    }
    None
}

fn first_argument_riscv(insns: &[Instruction]) -> Option<u64> {
    let mut a0: Option<u64> = None;
    for insn in insns {
        if insn.flow.is_call() {
            return a0;
        }
        let b = &insn.bytes;
        if insn.len < 4 {
            // A compressed instruction may write a0 in a form not modelled
            // here; forget the value rather than carry a stale one.
            let h = u16::from_le_bytes([b[0], b[1]]);
            if (h & 0x7f) == 0x01 {
                a0 = None;
            }
            continue;
        }
        let w = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let rd = (w >> 7) & 0x1f;
        let rs1 = (w >> 15) & 0x1f;
        // auipc a0, imm20
        if w & 0x7f == 0x17 && rd == 10 {
            let hi = (w & 0xffff_f000) as i32 as i64;
            a0 = Some(insn.addr.0.wrapping_add_signed(hi));
            continue;
        }
        // addi a0, a0, imm12
        if w & 0x7f == 0x13 && rd == 10 && rs1 == 10 && (w >> 12) & 7 == 0 {
            let lo = (w as i32) >> 20;
            a0 = a0.map(|v| v.wrapping_add_signed(i64::from(lo)));
            continue;
        }
        // lui a0, imm20
        if w & 0x7f == 0x37 && rd == 10 {
            a0 = Some((w & 0xffff_f000) as i32 as i64 as u64);
            continue;
        }
        if rd == 10 && matches!(w & 0x7f, 0x13 | 0x33 | 0x37 | 0x1b | 0x3b) {
            a0 = None;
        }
    }
    None
}

/// True when the byte before `addr` is padding the compiler emitted between
/// functions rather than code that falls through.
fn padding_before(view: &View, addr: Addr) -> bool {
    if addr.0 == 0 {
        return false;
    }
    matches!(
        view.bytes(addr.offset(-1), 1),
        Some([0x90] | [0xcc] | [0x00] | [0xc3] | [0x66])
    )
}

fn looks_like_prologue(view: &View, addr: Addr) -> bool {
    let Some(b) = view.bytes(addr, 8) else {
        return false;
    };
    match view.image().arch {
        occ_core::Arch::X86_64 | occ_core::Arch::X86 => {
            b[0] == 0x55
                || (b[0] == 0x53)
                || (b[0] == 0x48 && b[1] == 0x83 && b[2] == 0xec)
                || (b[0] == 0x48 && b[1] == 0x81 && b[2] == 0xec)
                || (b[0] == 0xf3 && b[1] == 0x0f && b[2] == 0x1e && b[3] == 0xfa)
        }
        occ_core::Arch::Aarch64 => {
            let w = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            // stp x29, x30, [sp, #-N]! — the frame record. The immediate is
            // the only field allowed to vary, so it is the only one masked out;
            // matching on a mask that also covered Rt2 required xzr there and
            // never fired on real code.
            w & 0xffc0_7fff == 0xa980_7bfd
                // paciasp and pacibsp are fixed encodings with no field to
                // mask off; comparing through a mask made both unreachable.
                || w == 0xd503_233f
                || w == 0xd503_237f
                // sub sp, sp, #imm: 64-bit and 32-bit forms, Rn and Rd both 31.
                || w & 0xffc0_03ff == 0xd100_03ff
                || w & 0xffc0_03ff == 0x5100_03ff
        }
        occ_core::Arch::RiscV64 | occ_core::Arch::RiscV32 => {
            let w = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            let h = w as u16;
            // c.addi16sp and c.addi sp, -N: quadrant 01, rd = x2. The immediate
            // bits are not part of the mask: a mask that overlaps the constant
            // can never match.
            (h & 0xef83) == 0x6101
                || (h & 0xef83) == 0x0101
                // addi sp, sp, -N, full width. `rs1` lives in bits 19:15, which
                // a 16-bit read cannot see, so this branch uses the whole word.
                || ((w & 0x7f) == 0x13
                    && ((w >> 7) & 0x1f) == 2
                    && ((w >> 12) & 0x7) == 0
                    && ((w >> 15) & 0x1f) == 2
                    && ((w >> 20) & 0xfff) != 0)
        }
        _ => false,
    }
}

/// Recover functions from executable bytes no seed covered.
///
/// This exists for stripped images with no entry point metadata at all. It is
/// deliberately conservative: an address only starts a function if its first
/// bytes look like a prologue and the preceding byte is padding or the start
/// of a section. Without that check the sweep turns every jump table into a
/// dozen "functions".
fn sweep(
    view: &View,
    dec: &AnyDecoder,
    functions: &mut Vec<Function>,
    owned: &mut BTreeMap<u64, u64>,
    opts: &RecoverOptions,
    notes: &mut Vec<String>,
) {
    let mut added = 0usize;
    let mut budget = opts.max_sweep_bytes;
    // Every call a swept body makes names a function the sweep has not reached
    // yet. Without following them, a stripped image recovers only the bodies
    // that happen to open with a recognisable prologue, which is less than half
    // of a real binary.
    let mut pending: Vec<Addr> = Vec::new();
    for s in view.image().sections.iter().filter(|s| s.is_code() && s.file_size > 0) {
        let mut addr = s.addr;
        // Only bytes the file holds can be decoded. `size` is the virtual
        // extent, which for `.bss`-like sections is larger than the file — and
        // for a malformed header is whatever the header says. Walking that one
        // byte at a time is a hang that looks like a slow analysis.
        let end = s.addr.add(s.file_size.min(s.size));
        while addr.0 < end.0 {
            if budget == 0 {
                notes.push(format!(
                    "sweep stopped after {} bytes; the sections claim more",
                    opts.max_sweep_bytes
                ));
                return;
            }
            budget -= 1;
            if view.bytes(addr, 1).is_none() {
                // Nothing is mapped here, so nothing after it can be either.
                break;
            }
            if functions.len() >= opts.max_functions {
                notes.push(format!("sweep stopped at {} functions", opts.max_functions));
                return;
            }
            if inside_owned(owned, addr) {
                addr = addr.add(1);
                continue;
            }
            // Padding is not code. Starting a decode in a run of `nop`/`int3`
            // produces a "function" whose body is the padding followed by the
            // real one, and its entry is a byte the compiler never meant as a
            // boundary.
            if addr != s.addr && matches!(view.bytes(addr, 1), Some([0x90] | [0xcc] | [0x00])) {
                addr = addr.add(1);
                continue;
            }
            let boundary = if addr == s.addr {
                true
            } else {
                matches!(view.bytes(addr.offset(-1), 1), Some([0xcc] | [0x90] | [0x00]))
            };
            // Fixed-width architectures do not need a prologue to prove a
            // function starts here: every arm64 instruction is four bytes and
            // a leaf function opens with `add`/`li`, no frame at all. What the
            // sweep can still check is alignment, and that a decode ending in
            // `ret` accounts for the bytes it claims.
            let aligned = match view.image().arch {
                occ_core::Arch::Aarch64 => addr.0 % 4 == 0,
                occ_core::Arch::RiscV32 | occ_core::Arch::RiscV64 => addr.0 % 2 == 0,
                _ => true,
            };
            let prologue = looks_like_prologue(view, addr);
            let fixed_width = matches!(
                view.image().arch,
                occ_core::Arch::Aarch64 | occ_core::Arch::RiscV32 | occ_core::Arch::RiscV64
            );
            // The prologue is evidence, not a requirement: a leaf function
            // compiled with `-O2` starts with a `mov` and ends with a `ret`,
            // and requiring a frame record skipped every one of them. The body
            // still has to end in a return to be accepted, which is what keeps
            // a decode that fell into unrelated bytes from becoming a function.
            let _ = fixed_width;
            if boundary && aligned {
                // The body cannot run past a function that is already placed:
                // those bytes belong to that function, whatever the decoder
                // finds there.
                let next_owned = owned
                    .range((std::ops::Bound::Excluded(addr.0), std::ops::Bound::Unbounded))
                    .next()
                    .map(|(start, _)| Addr(*start))
                    .unwrap_or(end);
                let out = decode_function(view, dec, addr, None, next_owned, owned, opts);
                let ends_in_return = matches!(
                    out.func.insns.last().map(|i| i.flow),
                    Some(FlowKind::Return)
                );
                if out.func.insns.len() >= 3 && (prologue || ends_in_return) {
                    let mut f = out.func;
                    f.is_thunk = is_thunk(&f);
                    owned.insert(f.entry.0, f.end().0);
                    addr = f.end().max(addr.add(1));
                    pending.extend(out.callees.iter().copied());
                    functions.push(f);
                    added += 1;
                    continue;
                }
            }
            addr = addr.add(1);
        }
    }
    // Follow the calls the sweep found. A called address is a function entry
    // by definition, so these need no prologue check; they also cascade, since
    // each one names its own callees.
    let mut followed = 0usize;
    while let Some(addr) = pending.pop() {
        if functions.len() >= opts.max_functions {
            notes.push(format!(
                "call-following stopped at {} functions; {} address(es) were not recovered",
                opts.max_functions,
                pending.len() + 1
            ));
            break;
        }
        if !view.is_exec(addr) || inside_owned(owned, addr) {
            continue;
        }
        let next_owned = owned
            .range((std::ops::Bound::Excluded(addr.0), std::ops::Bound::Unbounded))
            .next()
            .map(|(start, _)| Addr(*start))
            .unwrap_or_else(|| section_end(view, addr).unwrap_or_else(|| addr.add(0x10_0000)));
        let out = decode_function(view, dec, addr, None, next_owned, owned, opts);
        if out.func.insns.is_empty() {
            continue;
        }
        let mut f = out.func;
        f.is_thunk = is_thunk(&f);
        owned.insert(f.entry.0, f.end().0);
        pending.extend(out.callees.iter().copied());
        functions.push(f);
        added += 1;
        followed += 1;
    }
    if followed > 0 {
        notes.push(format!(
            "sweep followed calls to {followed} function(s) that no symbol named"
        ));
    }
    if added > 0 {
        notes.push(format!("sweep recovered {added} functions without symbol support"));
    }
}

/// Attach string references to functions after strings have been extracted.
pub(crate) fn attach_strings(
    functions: &mut [Function],
    strings: &std::collections::HashMap<u64, String>,
) {
    for f in functions.iter_mut() {
        let mut refs: Vec<Addr> = Vec::new();
        for insn in &f.insns {
            for m in insn.mem_reads.iter().chain(insn.mem_writes.iter()) {
                if strings.contains_key(&m.0) {
                    refs.push(*m);
                }
            }
        }
        refs.sort_by_key(|a| a.0);
        refs.dedup();
        f.string_refs = refs;
    }
}

/// Edge kinds reachable from a block, for tests and callers that summarise.
pub fn edge_kinds(blocks: &[BasicBlock]) -> Vec<EdgeKind> {
    let mut v: Vec<EdgeKind> = blocks
        .iter()
        .flat_map(|b| b.edges.iter().map(|e| e.kind))
        .collect();
    v.sort_by_key(|k| *k as u8);
    v.dedup();
    v
}

#[cfg(test)]
mod prologue_tests {
    use super::*;
    use crate::testutil::image_with_code;
    use occ_core::Arch;

    fn looks(arch: Arch, bytes: &[u8]) -> bool {
        // The matcher reads eight bytes; pad short encodings rather than
        // weakening what it asks for.
        let mut padded = bytes.to_vec();
        padded.resize(bytes.len().max(8), 0);
        let mut img = image_with_code(&padded);
        img.arch = arch;
        let view = View::new(&img, &padded);
        looks_like_prologue(&view, Addr(0x1000))
    }

    #[test]
    fn x86_prologues_are_recognised_and_padding_is_not() {
        assert!(looks(Arch::X86_64, &[0x55, 0x48, 0x89, 0xe5]));
        assert!(looks(Arch::X86_64, &[0x48, 0x83, 0xec, 0x20]));
        assert!(!looks(Arch::X86_64, &[0x90, 0x90, 0x90, 0x90]));
    }

    #[test]
    fn aarch64_prologues_are_recognised() {
        // stp x29, x30, [sp, #-16]! — the frame record every compiled function
        // that touches the stack starts with.
        assert!(looks(Arch::Aarch64, &0xa9bf_7bfdu32.to_le_bytes()));
        // paciasp / pacibsp: fixed encodings, and the pair a mask-based match
        // used to reject.
        assert!(looks(Arch::Aarch64, &0xd503_233fu32.to_le_bytes()));
        assert!(looks(Arch::Aarch64, &0xd503_237fu32.to_le_bytes()));
        // sub sp, sp, #0x30
        assert!(looks(Arch::Aarch64, &0xd100_c3ffu32.to_le_bytes()));
        // add x0, x0, #1 — not a prologue.
        assert!(!looks(Arch::Aarch64, &0x9100_0400u32.to_le_bytes()));
        assert!(!looks(Arch::Aarch64, &[0, 0, 0, 0]));
    }

    #[test]
    fn riscv_prologues_are_recognised() {
        // c.addi16sp sp, -32: quadrant 01, funct3 011, rd x2.
        assert!(looks(Arch::RiscV64, &0x7139u16.to_le_bytes()));
        // addi sp, sp, -16, full width: 0xff010113.
        assert!(looks(Arch::RiscV64, &0xff01_0113u32.to_le_bytes()));
        // addi a0, a0, 1 — not a prologue.
        assert!(!looks(Arch::RiscV64, &0x0015_0513u32.to_le_bytes()));
        assert!(!looks(Arch::RiscV64, &[0, 0, 0, 0]));
    }
}
