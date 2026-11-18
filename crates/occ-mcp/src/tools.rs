//! The tool registry.
//!
//! Every tool follows four rules, and the rules are the product:
//!
//! 1. **Return a handle.** The response's `handle` resolves on the next turn,
//!    so a follow-up names one short string instead of re-sending a query.
//! 2. **Answer the question asked, then stop.** `data` carries the result;
//!    `next` carries two or three suggestions, not a directory listing.
//! 3. **Say how sure, and why.** Conclusions carry `evidence` and
//!    `confidence`; heuristics are labelled as heuristics.
//! 4. **Refuse clearly.** A bad handle names the accepted forms; a missing
//!    image lists the open ones. An error a model can read is a retry that
//!    works.

use occ_core::{
    Addr, Depth, Evidence, EvidenceKind, EvidenceSource, Function, Handle,
};
use occ_json::Value;
use std::sync::Arc;
use occ_session::render::{
    Envelope, addr, block_json, fingerprint_json, function_json, hexdump, image_json,
    import_json, insn_json, reloc_json, section_json, string_json, symbol_json, xref_json,
    export_json,
};
use occ_session::session::{Session, Target};
use occ_session::{ImageFile, diff, pattern};

use crate::args::Args;
use occ_session::obj;

/// One exposed tool.
pub struct ToolDef {
    /// Name as the client sees it.
    pub name: &'static str,
    /// Description shown to the model.
    pub description: &'static str,
    /// JSON Schema for `arguments`.
    pub schema: &'static str,
    /// Implementation.
    pub handler: fn(&Session, &Args) -> Result<Value, String>,
}

/// Every tool, in discovery order.
pub fn all() -> &'static [ToolDef] {
    &TOOLS
}

/// Look up a tool by name.
pub fn find(name: &str) -> Option<&'static ToolDef> {
    TOOLS.iter().find(|t| t.name == name)
}

/// The `tools/list` payload.
pub fn list_json() -> Value {
    Value::Array(
        TOOLS
            .iter()
            .map(|t| {
                obj!(
                    "name" => t.name,
                    "description" => t.description,
                    // Schemas are stored as text so the registry stays a plain
                    // static table; parsing here validates them on every list,
                    // which catches a typo at the first client instead of at
                    // the first call.
                    "inputSchema" => occ_json::parse(t.schema).unwrap_or(Value::Null),
                )
            })
            .collect(),
    )
}

macro_rules! tool {
    ($name:literal, $desc:literal, $schema:literal, $handler:path) => {
        ToolDef {
            name: $name,
            description: $desc,
            schema: $schema,
            handler: $handler,
        }
    };
}

static TOOLS: [ToolDef; 38] = [
    tool!(
        "session.info",
        "List open images and their ids, plus hypotheses and notes. Call this first when continuing a conversation to see what state exists.",
        r#"{"type":"object","properties":{},"additionalProperties":false}"#,
        session_info
    ),
    tool!(
        "session.save",
        "Write the session (image paths, hypotheses, notes) to a JSON file so a later process can restore it. Analyses are recomputed on restore, not cached.",
        r#"{"type":"object","properties":{"path":{"type":"string","description":"file to write"}},"required":["path"],"additionalProperties":false}"#,
        session_save
    ),
    tool!(
        "session.restore",
        "Reopen the images recorded in a saved session file and recreate its hypotheses.",
        r#"{"type":"object","properties":{"path":{"type":"string","description":"file written by session.save"}},"required":["path"],"additionalProperties":false}"#,
        session_restore
    ),
    tool!(
        "binary.open",
        "Load a binary and run full analysis (function recovery, strings, cross references). Returns an image id and a summary. Every other tool takes that id, so open once per image.",
        r#"{"type":"object","properties":{"path":{"type":"string","description":"path to an ELF, PE, COFF, Mach-O or WASM file"}},"required":["path"],"additionalProperties":false}"#,
        binary_open
    ),
    tool!(
        "binary.info",
        "Headers, load segments, section table, fingerprint and analysis coverage for one image.",
        r#"{"type":"object","properties":{"image":{"type":"integer","description":"image id from binary.open"}},"additionalProperties":false}"#,
        binary_info
    ),
    tool!(
        "binary.sections",
        "List sections with address, size, permissions and kind. Filter by name substring or kind.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"filter":{"type":"string","description":"substring of the section name"},"kind":{"type":"string","description":"code, data, rodata, bss, unwind, debug, ..."},"limit":{"type":"integer"},"offset":{"type":"integer"}},"additionalProperties":false}"#,
        binary_sections
    ),
    tool!(
        "binary.imports",
        "List imported symbols, including the GOT/IAT slot each one occupies when the container records it.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"filter":{"type":"string"},"limit":{"type":"integer"},"offset":{"type":"integer"}},"additionalProperties":false}"#,
        binary_imports
    ),
    tool!(
        "binary.exports",
        "List exported symbols with their addresses.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"filter":{"type":"string"},"limit":{"type":"integer"},"offset":{"type":"integer"}},"additionalProperties":false}"#,
        binary_exports
    ),
    tool!(
        "binary.symbols",
        "Search the symbol table by name substring. Reports whether the image is stripped, which changes how much the symbols can be trusted.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"filter":{"type":"string"},"defined_only":{"type":"boolean"},"limit":{"type":"integer"},"offset":{"type":"integer"}},"additionalProperties":false}"#,
        binary_symbols
    ),
    tool!(
        "binary.relocations",
        "List relocations with target symbol, kind and addend. Useful for finding function pointers and resolving position-independent data.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"filter":{"type":"string","description":"substring of the symbol name"},"limit":{"type":"integer"},"offset":{"type":"integer"}},"additionalProperties":false}"#,
        binary_relocations
    ),
    tool!(
        "binary.strings",
        "Extracted strings (ASCII and UTF-16) with addresses. Search by substring; results are handle-addressable as str:<addr>.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"pattern":{"type":"string","description":"case-sensitive substring"},"min_len":{"type":"integer"},"section":{"type":"string"},"encoding":{"type":"string","description":"ascii or utf16"},"sort":{"type":"string","description":"addr or length"},"limit":{"type":"integer"},"offset":{"type":"integer"}},"additionalProperties":false}"#,
        binary_strings
    ),
    tool!(
        "binary.hexdump",
        "Hex and ASCII dump of up to 4096 bytes at an address.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"addr":{"type":"string","description":"start address, e.g. 0x401000"},"len":{"type":"integer"},"width":{"type":"integer","description":"bytes per row, default 16"}},"required":["addr"],"additionalProperties":false}"#,
        binary_hexdump
    ),
    tool!(
        "binary.entropy",
        "Per-section entropy, or a sliding-window scan inside one section, with the threshold used. High entropy in executable bytes is the signal; in .rodata it usually is not.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"section":{"type":"string","description":"run a sliding window over this section"},"window":{"type":"integer","description":"window size, default 256"},"threshold":{"type":"number","description":"default 7.2"}},"additionalProperties":false}"#,
        binary_entropy
    ),
    tool!(
        "binary.fingerprint",
        "Compiler, source language, linkage model, runtime and packer suspicion, each with the evidence behind it.",
        r#"{"type":"object","properties":{"image":{"type":"integer"}},"additionalProperties":false}"#,
        binary_fingerprint
    ),
    tool!(
        "code.functions",
        "Recovered functions with size, block count and call degree. Sort by size or instruction count to find the interesting ones; filter by name substring.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"filter":{"type":"string"},"min_insns":{"type":"integer"},"sort":{"type":"string","description":"addr, size, insns, complexity, callers, callees"},"limit":{"type":"integer"},"offset":{"type":"integer"}},"additionalProperties":false}"#,
        code_functions
    ),
    tool!(
        "code.function",
        "Full detail for one function: blocks, edges, callers, callees, imports called, strings referenced, complexity, boundary evidence. Target accepts fn:<addr>, a symbol name, or an address.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"target":{"type":"string"},"depth":{"type":"string","description":"summary, detail, raw"},"include_body":{"type":"boolean"}},"required":["target"],"additionalProperties":false}"#,
        code_function
    ),
    tool!(
        "code.disasm",
        "Disassemble from an address or function entry for up to N instructions. Each instruction reports its flow kind and resolved target.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"target":{"type":"string","description":"function handle, symbol, or address"},"count":{"type":"integer","description":"instructions, default 40, max 2000"},"depth":{"type":"string"}},"required":["target"],"additionalProperties":false}"#,
        code_disasm
    ),
    tool!(
        "code.blocks",
        "Control-flow graph of one function as blocks with typed edges. This is the cheapest way to see loop and branch structure without reading instructions.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"target":{"type":"string"},"include_body":{"type":"boolean"}},"required":["target"],"additionalProperties":false}"#,
        code_blocks
    ),
    tool!(
        "code.decompile",
        "Recover pseudo-C for one function: lifting, SSA, structuring and rendering. Every statement the lifter could not model stays in the output as a comment, and the count of constructs that needed `goto` is reported, so a reader can tell a faithful body from a partial one.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"target":{"type":"string","description":"function handle (fn:0x401000), name, or address inside the function"},"comments":{"type":"boolean","description":"keep `//` notes for instructions that were not lifted (default true)"},"max_blocks":{"type":"integer","description":"stop after this many basic blocks"}},"required":["target"],"additionalProperties":false}"#,
        code_decompile
    ),
    tool!(
        "code.explore",
        "Execute one function along its paths. Branch conditions that depend on an input fork the state, and each completed path reports the inputs that reach it, solved and checked, plus the value it returns. Bounds are reported as `truncated`, never as dead code.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"target":{"type":"string","description":"function handle (fn:0x401000), name, or address inside the function"},"mode":{"type":"string","description":"symbolic (default) forks on unknown inputs; concrete treats an unknown input as missing and lists it"},"inputs":{"type":"object","description":"known input values by IR name, e.g. {\"rdi\": 3, \"rsi\": \"0x4000\"}"},"memory":{"type":"object","description":"initial memory as {\"0x4000\": \"deadbeef\"}"},"max_paths":{"type":"integer","description":"paths kept in the result, default 16"},"max_states":{"type":"integer","description":"states explored in total, default 128"},"unroll":{"type":"integer","description":"how often one block may repeat per path, default 3"}},"required":["target"],"additionalProperties":false}"#,
        code_explore
    ),
    tool!(
        "code.taint",
        "Propagate sources through one function and report where they arrive: which argument of which call, which branch, which store, with a block path from the entry. Sources default to the parameter registers.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"target":{"type":"string","description":"function handle (fn:0x401000), name, or address inside the function"},"sources":{"type":"array","items":{"type":"string"},"description":"source labels: \"rdi\", \"name:rdi\", \"addr:0x4000:8\", or a bare hex address (8 bytes)"},"sinks":{"type":"array","items":{"type":"string"},"description":"what to report: \"call\" (every call, default), \"branch\", \"store\", or call names to match as substrings"}},"required":["target"],"additionalProperties":false}"#,
        code_taint
    ),
    tool!(
        "code.xrefs",
        "Cross references to or from an address. Use direction=to to answer 'who calls this'; direction=from to answer 'what does this touch'.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"target":{"type":"string"},"direction":{"type":"string","description":"to, from, both (default to)"},"kind":{"type":"string","description":"call, tailcall, jump, condjump, dataread, datawrite, addresstaken, thunk"},"limit":{"type":"integer"}},"required":["target"],"additionalProperties":false}"#,
        code_xrefs
    ),
    tool!(
        "code.callgraph",
        "Walk callers or callees breadth-first from a function, bounded by depth and node count. Answers 'what reaches this' and 'what does this depend on'.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"target":{"type":"string"},"direction":{"type":"string","description":"callers or callees, default callers"},"depth":{"type":"integer","description":"1-4, default 2"},"max_nodes":{"type":"integer","description":"default 200"}},"required":["target"],"additionalProperties":false}"#,
        code_callgraph
    ),
    tool!(
        "code.search",
        "Search instructions by mnemonic or text substring, or by immediate/address value appearing in the operand. Returns the containing function for each hit.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"mnemonic":{"type":"string"},"text":{"type":"string"},"value":{"type":"string","description":"hex value to look for in operands"},"limit":{"type":"integer"}},"additionalProperties":false}"#,
        code_search
    ),
    tool!(
        "debug.info",
        "DWARF inventory: compilation units with producer, language, directory, source file list and counts, plus anything that could not be parsed. Call this before line or variable questions to see whether debug info exists at all.",
        r#"{"type":"object","properties":{"image":{"type":"integer"}},"additionalProperties":false}"#,
        debug_info
    ),
    tool!(
        "debug.line",
        "Map addresses to source file:line:column and name the enclosing debug function. Accepts one address, a comma-separated list, a symbol name, or a fn: handle.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"at":{"type":"string","description":"address, symbol, fn: handle or comma-separated list"}},"required":["at"],"additionalProperties":false}"#,
        debug_line
    ),
    tool!(
        "code.indirect",
        "Resolve every computed jump and every call through a register by value-set analysis. Each site reports the register it goes through, the table behind it when there is one, and the blocks it can reach. A target is reported only when it is a block the recovery already found, so the answer cannot invent one.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"function":{"type":"string","description":"only sites inside this function (fn: handle, name or address)"},"kind":{"type":"string","description":"calls, jumps or both (default both)"},"limit":{"type":"integer"},"offset":{"type":"integer"}},"additionalProperties":false}"#,
        code_indirect
    ),
    tool!(
        "debug.functions",
        "Functions described by debug info, with declaration site and parameter/local counts. These are the compiler's own boundaries, not recovered ones.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"filter":{"type":"string"},"sort":{"type":"string","description":"addr, size, params or name"},"limit":{"type":"integer"},"offset":{"type":"integer"}},"additionalProperties":false}"#,
        debug_functions
    ),
    tool!(
        "debug.function",
        "One function's parameters and locals with resolved type names and the location expression valid at `at` (defaults to the entry address, where arguments are still in their incoming registers).",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"target":{"type":"string","description":"fn: handle, name or address"},"at":{"type":"string","description":"address to resolve locations at"}},"required":["target"],"additionalProperties":false}"#,
        debug_function
    ),
    tool!(
        "debug.types",
        "Types from debug info: kind, byte size, member offsets and pointee. Filter by name substring or kind.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"filter":{"type":"string"},"kind":{"type":"string","description":"base, pointer, struct, union, class, enum, array, typedef, function"},"include_members":{"type":"boolean"},"limit":{"type":"integer"},"offset":{"type":"integer"}},"additionalProperties":false}"#,
        debug_types
    ),
    tool!(
        "debug.variables",
        "Every variable the debug info places at an address, with type and the location expression for that instruction. This is how a stack slot or register is turned back into a named source variable.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"at":{"type":"string"}},"required":["at"],"additionalProperties":false}"#,
        debug_variables
    ),
    tool!(
        "hypothesis.create",
        "Record a falsifiable statement about a handle (typically fn:<addr>) and get an id. Evidence added later moves it between confirmed, refuted and inconclusive.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"target":{"type":"string"},"statement":{"type":"string"}},"required":["target","statement"],"additionalProperties":false}"#,
        hypothesis_create
    ),
    tool!(
        "hypothesis.evidence",
        "Attach one observation for or against a hypothesis. Returns the recomputed verdict: confirmed (>=0.75 support), refuted (>=0.6 counter, support <=0.4), or inconclusive.",
        r#"{"type":"object","properties":{"id":{"type":"integer"},"detail":{"type":"string"},"support":{"type":"boolean","description":"true adds support, false refutes"},"weight":{"type":"number","description":"0..1, default 0.5"},"at":{"type":"string","description":"address the observation applies to"},"kind":{"type":"string","description":"bytes, cfg, table, symbol, string, import, const, entropy, xref, trace, format, absence"}},"required":["id","detail"],"additionalProperties":false}"#,
        hypothesis_evidence
    ),
    tool!(
        "hypothesis.list",
        "All hypotheses with their current verdict and evidence count.",
        r#"{"type":"object","properties":{},"additionalProperties":false}"#,
        hypothesis_list
    ),
    tool!(
        "diff.functions",
        "Compare the functions of two open images. Reports identical, modified (same shape, different operands), rewritten, added and removed, matched by name then by structural hash.",
        r#"{"type":"object","properties":{"image_a":{"type":"integer"},"image_b":{"type":"integer"},"only_changed":{"type":"boolean"},"limit":{"type":"integer"}},"required":["image_a","image_b"],"additionalProperties":false}"#,
        diff_functions
    ),
    tool!(
        "scan.pattern",
        "Match YARA-shaped rules against the image. Supports text strings with nocase/wide, hex with ?? wildcards and [n-m] gaps, and conditions with and/or/not, N of them, at <addr>, in <section>.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"rules":{"type":"string","description":"rule source text"},"limit":{"type":"integer"}},"required":["rules"],"additionalProperties":false}"#,
        scan_pattern
    ),
    tool!(
        "trace.run",
        "Run the program under ptrace and record what it actually does: the instruction path, every call edge including indirect ones, syscalls with their path arguments, and signals. Compares the run against the recovered functions and reports which ones never ran, which executed addresses belong to no function, and where each indirect call went. Bounded by a step count and a wall clock; a cut run says so.",
        r#"{"type":"object","properties":{"image":{"type":"integer"},"argv":{"type":"array","items":{"type":"string"},"description":"full argv, argv[0] included; the first entry is the name the program sees as its own, and defaults to the image path"},"steps":{"type":"integer","description":"instruction budget, default 200000, max 2000000"},"millis":{"type":"integer","description":"wall clock budget, default 5000, max 60000"},"from_entry":{"type":"boolean","description":"start at the entry point (default true); false traces the dynamic loader as well"},"syscalls":{"type":"boolean","description":"record syscalls and the strings behind their path arguments (default true)"},"limit":{"type":"integer","description":"rows per list, default 16"}},"required":["image"],"additionalProperties":false}"#,
        trace_run
    ),
    tool!(
        "trace.get",
        "Query a recorded run without running anything again. Without a target it returns the coverage summary; with a function handle or address it returns block-by-block hit counts, which answers 'did this branch ever go the other way' for that run.",
        r#"{"type":"object","properties":{"trace":{"type":"string","description":"trace handle (trace:0); omit for the most recent"},"target":{"type":"string","description":"function handle, name or address for block-level detail"},"limit":{"type":"integer","description":"rows per list, default 16"}},"additionalProperties":false}"#,
        trace_get
    ),
];

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn summary_of(file: &ImageFile) -> String {
    format!(
        "{} — {} functions, {} strings, {:.1}% of code decoded",
        file.image.headline(),
        file.analysis.coverage.functions,
        file.analysis.strings.len(),
        f64::from(file.analysis.coverage.fraction()) * 100.0
    )
}

fn handle_target(session: &Session, id: u32, target: &str) -> Result<Target, String> {
    let file = session.image(id).ok_or("image is not open")?;
    // A function handle, a symbol name, or a bare address.
    if target.starts_with("fn:") {
        return session.resolve(id, target, occ_core::HandleKind::Function).map_err(|e| e.to_string());
    }
    if let Some(addr) = occ_core::addr::parse_addr(target) {
        if let Some(f) = file.analysis.function_covering(addr) {
            return Ok(Target::Function {
                image: id,
                entry: f.entry,
                name: f.display_name(),
                size: f.size,
            });
        }
        return Ok(Target::Insn {
            image: id,
            addr,
            text: String::new(),
        });
    }
    if let Some(f) = file.analysis.function_by_name(target) {
        return Ok(Target::Function {
            image: id,
            entry: f.entry,
            name: f.display_name(),
            size: f.size,
        });
    }
    Err(format!(
        "no function named {target:?} and it is not an address; try code.functions with a filter"
    ))
}

fn function_of<'a>(file: &'a ImageFile, target: &Target) -> Option<&'a Function> {
    match target {
        Target::Function { entry, .. } => file
            .analysis
            .function_at(*entry)
            .or_else(|| file.analysis.function_covering(*entry)),
        Target::Insn { addr, .. } => file.analysis.function_covering(*addr),
        _ => None,
    }
}

fn paginate<T: Clone>(items: &[T], offset: usize, limit: usize) -> (Vec<T>, bool) {
    let slice: Vec<T> = items.iter().skip(offset).take(limit).cloned().collect();
    (slice, items.len() > offset + limit)
}

// ---------------------------------------------------------------------------
// session tools
// ---------------------------------------------------------------------------

fn session_info(session: &Session, _args: &Args) -> Result<Value, String> {
    let images = session
        .stats()
        .into_iter()
        .map(|(id, text)| obj!("image" => Value::from(id), "summary" => text))
        .collect::<Vec<_>>();
    let hyps = session.hypotheses();
    let e = Envelope::ok(format!(
        "{} image(s), {} hypothesis(es)",
        images.len(),
        hyps.len()
    ))
    .field("images", Value::Array(images))
    .field(
        "hypotheses",
        Value::Array(
            hyps.iter()
                .map(|h| {
                    obj!(
                        "id" => Value::from(h.id),
                        "handle" => h.handle.to_string(),
                        "statement" => h.statement.clone(),
                        "verdict" => h.verdict().label(),
                        "confidence" => f64::from(h.confidence),
                    )
                })
                .collect(),
        ),
    )
    .field(
        "notes",
        Value::Array(session.notes().into_iter().map(Value::String).collect()),
    )
    .next(vec![
        "binary.open {path} to add an image".into(),
        "code.functions to start looking at code".into(),
    ]);
    Ok(e.build())
}

fn session_save(session: &Session, args: &Args) -> Result<Value, String> {
    let path = args.req_str("path")?;
    session.save(path).map_err(|e| e.to_string())?;
    Ok(Envelope::ok(format!("saved session to {path}"))
        .field("path", path)
        .build())
}

fn session_restore(session: &Session, args: &Args) -> Result<Value, String> {
    let path = args.req_str("path")?;
    let saved = occ_session::persist::load(path).map_err(|e| e.to_string())?;
    let warnings = session.restore(&saved).map_err(|e| e.to_string())?;
    let e = Envelope::ok(format!(
        "restored {} image(s) and {} hypothesis(es)",
        saved.images.len(),
        saved.hypotheses.len()
    ))
    .field("warnings", Value::Array(warnings.iter().cloned().map(Value::String).collect()))
    .caveats(warnings.clone());
    Ok(e.build())
}

// ---------------------------------------------------------------------------
// binary tools
// ---------------------------------------------------------------------------

fn binary_open(session: &Session, args: &Args) -> Result<Value, String> {
    let path = args.req_str("path")?;
    let ids = session.open_path(path).map_err(|e| e.to_string())?;
    let id = *ids.first().ok_or("the open produced no image")?;
    let file = session.image(id).ok_or("image vanished after open")?;
    let fp = fingerprint_json(&file.analysis.fingerprint);
    let e = Envelope::ok(summary_of(&file))
        .handle(Handle::image(id).to_string())
        .field("image", Value::from(id))
        .field("identity", image_json(&file.image))
        .field("fingerprint", fp)
        .field(
            "coverage",
            obj!(
                "code_bytes" => Value::from(file.analysis.coverage.code_bytes),
                "insn_bytes" => Value::from(file.analysis.coverage.insn_bytes),
                "unparsed_bytes" => Value::from(file.analysis.coverage.unparsed_bytes),
                "decoded_bytes" => Value::from(file.analysis.coverage.decoded_bytes),
                "covered_bytes" => Value::from(file.analysis.coverage.covered_bytes),
                "functions" => Value::from(file.analysis.coverage.functions),
                "insns" => Value::from(file.analysis.coverage.insns),
                "blocks" => Value::from(file.analysis.coverage.blocks),
                "insn_bytes" => Value::from(file.analysis.coverage.insn_bytes),
                "unparsed_bytes" => Value::from(file.analysis.coverage.unparsed_bytes),
                "fraction" => f64::from(file.analysis.coverage.fraction()),
                "accounted_fraction" => f64::from(file.analysis.coverage.accounted_fraction()),
            ),
        )
        .field(
            "notes",
            Value::Array(
                file.analysis
                    .notes
                    .iter()
                    .chain(file.image.warnings.iter())
                    .cloned()
                    .map(Value::String)
                    .collect(),
            ),
        )
        .field(
            "warnings",
            Value::Array(file.image.warnings.iter().cloned().map(Value::String).collect()),
        )
        .field(
            "entry",
            obj!(
                "addr" => addr(file.image.entry),
                "name" => file
                    .analysis
                    .function_at(file.image.entry)
                    .map(|f| f.display_name())
                    .unwrap_or_default(),
            ),
        )
        .next(vec![
            format!("code.functions with image={id}, sort=insns, limit=20"),
            format!("binary.strings with image={id} and a pattern"),
            format!("binary.fingerprint with image={id}"),
        ]);
    // An archive opens as one image per member. The first member answers the
    // summary, and the rest are named here so a caller can address them
    // without knowing the archive was one file.
    let mut e = e;
    if ids.len() > 1 {
        let members = ids
            .iter()
            .map(|mid| match session.image(*mid) {
                Some(f) => obj!(
                    "image" => Value::from(*mid),
                    "name" => Value::String(member_name(&f.image.path, path)),
                    "format" => Value::String(f.image.format.label().to_string()),
                    "functions" => Value::from(f.analysis.coverage.functions),
                ),
                None => obj!("image" => Value::from(*mid)),
            })
            .collect::<Vec<_>>();
        e = e.field("member_count", Value::from(ids.len())).field(
            "members",
            Value::Array(members),
        );
    }
    Ok(e.build())
}

/// The member name inside an archive's synthetic path.
///
/// The session names a member `libfoo.a(member.o)` so that a handle is
/// readable; the listing wants the part inside the parentheses.
fn member_name(path: &str, archive: &str) -> String {
    path.strip_prefix(archive)
        .and_then(|rest| rest.strip_prefix('('))
        .and_then(|rest| rest.strip_suffix(')'))
        .unwrap_or(path)
        .to_string()
}

fn binary_info(session: &Session, args: &Args) -> Result<Value, String> {
    let file = args.file(session)?;
    let id = args.image(session)?;
    let segments = file
        .image
        .segments
        .iter()
        .map(|s| {
            obj!(
                "addr" => addr(s.addr),
                "size" => Value::from(s.size),
                "file_size" => Value::from(s.file_size),
                "perm" => s.perm.to_string(),
            )
        })
        .collect::<Vec<_>>();
    let e = Envelope::ok(file.image.headline())
        .handle(Handle::image(id).to_string())
        .field("identity", image_json(&file.image))
        .field("segments", Value::Array(segments))
        .field("fingerprint", fingerprint_json(&file.analysis.fingerprint))
        .field(
            "coverage",
            obj!(
                "functions" => Value::from(file.analysis.coverage.functions),
                "insns" => Value::from(file.analysis.coverage.insns),
                "blocks" => Value::from(file.analysis.coverage.blocks),
                "insn_bytes" => Value::from(file.analysis.coverage.insn_bytes),
                "unparsed_bytes" => Value::from(file.analysis.coverage.unparsed_bytes),
                "fraction" => f64::from(file.analysis.coverage.fraction()),
            ),
        )
        .field(
            "notes",
            Value::Array(file.analysis.notes.iter().cloned().map(Value::String).collect()),
        );
    Ok(e.build())
}

fn binary_sections(session: &Session, args: &Args) -> Result<Value, String> {
    let file = args.file(session)?;
    let filter = args.opt_str("filter").map(|s| s.to_ascii_lowercase());
    let kind = args.opt_str("kind").map(|s| s.to_ascii_lowercase());
    let (offset, limit) = args.page(200, 2000)?;
    let items: Vec<Value> = file
        .image
        .sections
        .iter()
        .filter(|s| {
            filter
                .as_ref()
                .map(|f| s.name.to_ascii_lowercase().contains(f))
                .unwrap_or(true)
        })
        .filter(|s| {
            kind.as_ref()
                .map(|k| format!("{:?}", s.kind).to_ascii_lowercase() == *k)
                .unwrap_or(true)
        })
        .map(section_json)
        .collect();
    let total = items.len();
    let (page, truncated) = paginate(&items, offset, limit);
    Ok(Envelope::ok(format!("{total} section(s) match"))
        .field("sections", Value::Array(page))
        .field("total", Value::from(total))
        .truncated(truncated)
        .next(vec!["binary.hexdump with a section address".into()])
        .build())
}

fn binary_imports(session: &Session, args: &Args) -> Result<Value, String> {
    let file = args.file(session)?;
    let filter = args.opt_str("filter").map(|s| s.to_ascii_lowercase());
    let (offset, limit) = args.page(200, 2000)?;
    let items: Vec<Value> = file
        .image
        .imports
        .iter()
        .filter(|i| {
            filter
                .as_ref()
                .map(|f| i.name.to_ascii_lowercase().contains(f))
                .unwrap_or(true)
        })
        .map(import_json)
        .collect();
    let total = items.len();
    let (page, truncated) = paginate(&items, offset, limit);
    Ok(Envelope::ok(format!("{total} import(s)"))
        .field("imports", Value::Array(page))
        .field("total", Value::from(total))
        .truncated(truncated)
        .build())
}

fn binary_exports(session: &Session, args: &Args) -> Result<Value, String> {
    let file = args.file(session)?;
    let filter = args.opt_str("filter").map(|s| s.to_ascii_lowercase());
    let (offset, limit) = args.page(200, 2000)?;
    let items: Vec<Value> = file
        .image
        .exports
        .iter()
        .filter(|e| {
            filter
                .as_ref()
                .map(|f| e.name.to_ascii_lowercase().contains(f))
                .unwrap_or(true)
        })
        .map(export_json)
        .collect();
    let total = items.len();
    let (page, truncated) = paginate(&items, offset, limit);
    Ok(Envelope::ok(format!("{total} export(s)"))
        .field("exports", Value::Array(page))
        .field("total", Value::from(total))
        .truncated(truncated)
        .build())
}

fn binary_symbols(session: &Session, args: &Args) -> Result<Value, String> {
    let file = args.file(session)?;
    let filter = args.opt_str("filter").map(|s| s.to_ascii_lowercase());
    let defined_only = args.bool_or("defined_only", false)?;
    let (offset, limit) = args.page(200, 2000)?;
    let items: Vec<Value> = file
        .image
        .symbols
        .iter()
        .filter(|s| !s.name.is_empty())
        .filter(|s| !defined_only || s.defined)
        .filter(|s| {
            filter
                .as_ref()
                .map(|f| s.name.to_ascii_lowercase().contains(f))
                .unwrap_or(true)
        })
        .map(symbol_json)
        .collect();
    let total = items.len();
    let (page, truncated) = paginate(&items, offset, limit);
    Ok(Envelope::ok(format!(
        "{total} symbol(s); image is {}",
        if file.analysis.fingerprint.stripped {
            "stripped"
        } else {
            "not stripped"
        }
    ))
    .field("symbols", Value::Array(page))
    .field("total", Value::from(total))
    .field("stripped", file.analysis.fingerprint.stripped)
    .truncated(truncated)
    .build())
}

fn binary_relocations(session: &Session, args: &Args) -> Result<Value, String> {
    let file = args.file(session)?;
    let filter = args.opt_str("filter").map(|s| s.to_ascii_lowercase());
    let (offset, limit) = args.page(200, 2000)?;
    let items: Vec<Value> = file
        .image
        .relocations
        .iter()
        .filter(|r| {
            filter
                .as_ref()
                .map(|f| {
                    r.symbol
                        .as_ref()
                        .map(|s| s.to_ascii_lowercase().contains(f))
                        .unwrap_or(false)
                })
                .unwrap_or(true)
        })
        .enumerate()
        .map(|(i, r)| {
            let mut v = reloc_json(r);
            if let Value::Object(m) = &mut v {
                m.insert("handle", Handle::reloc(i).to_string());
            }
            v
        })
        .collect();
    let total = items.len();
    let (page, truncated) = paginate(&items, offset, limit);
    Ok(Envelope::ok(format!("{total} relocation(s)"))
        .field("relocations", Value::Array(page))
        .field("total", Value::from(total))
        .truncated(truncated)
        .build())
}

fn binary_strings(session: &Session, args: &Args) -> Result<Value, String> {
    let file = args.file(session)?;
    let pattern = args.opt_str("pattern");
    let min_len = args.u64_in("min_len", 0, 0, u64::MAX)? as usize;
    let section = args.opt_str("section");
    let encoding = args.opt_str("encoding");
    let sort = args.one_of("sort", "addr", &["addr", "length"])?;
    let (offset, limit) = args.page(100, 1000)?;
    let mut hits: Vec<&occ_analyze::strings::Found> = file
        .analysis
        .strings
        .iter()
        .filter(|s| pattern.map(|p| s.text.contains(p)).unwrap_or(true))
        .filter(|s| s.text.chars().count() >= min_len)
        .filter(|s| section.map(|x| s.section == x).unwrap_or(true))
        .filter(|s| {
            encoding
                .map(|e| s.encoding.label().starts_with(e))
                .unwrap_or(true)
        })
        .collect();
    if sort == "length" {
        hits.sort_by_key(|s| std::cmp::Reverse(s.text.len()));
    }
    let items: Vec<Value> = hits.iter().map(|s| string_json(s)).collect();
    let total = items.len();
    let (page, truncated) = paginate(&items, offset, limit);
    Ok(Envelope::ok(format!("{total} string(s) match"))
        .field("strings", Value::Array(page))
        .field("total", Value::from(total))
        .truncated(truncated)
        .next(vec!["code.xrefs with a str:<addr> handle".into()])
        .build())
}

fn binary_hexdump(session: &Session, args: &Args) -> Result<Value, String> {
    let file = args.file(session)?;
    let at = args.req_addr("addr")?;
    let len = args.u64_in("len", 256, 1, 4096)? as usize;
    let width = args.u64_in("width", 16, 1, 32)? as usize;
    let view = occ_analyze::view::View::new(&file.image, &file.data);
    let bytes = view
        .bytes(at, len)
        .ok_or_else(|| format!("{at}..{at}+{len} is not fully mapped"))?;
    Ok(Envelope::ok(format!("{len} byte(s) at {at}"))
        .handle(Handle::new(occ_core::HandleKind::Str, at.to_string()).to_string())
        .field("addr", addr(at))
        .field("bytes_len", Value::from(len))
        .field("hex", hexdump(bytes, at, width))
        .build())
}

fn binary_entropy(session: &Session, args: &Args) -> Result<Value, String> {
    let file = args.file(session)?;
    let e = Envelope::ok("entropy per section");
    if let Some(name) = args.opt_str("section") {
        let s = file
            .image
            .section_by_name(name)
            .ok_or_else(|| format!("no section named {name}"))?;
        let window = args.u64_in("window", 256, 1, 65536)? as usize;
        let threshold = args.raw("threshold").and_then(|v| v.as_f64()).unwrap_or(7.2) as f32;
        let view = occ_analyze::view::View::new(&file.image, &file.data);
        let wins = occ_analyze::entropy::sliding(
            &view,
            s.addr,
            s.file_size as usize,
            window,
            threshold,
        );
        return Ok(e
            .field("section", name)
            .field("window", Value::from(window))
            .field("threshold", f64::from(threshold))
            .field(
                "windows",
                Value::Array(
                    wins.iter()
                        .take(500)
                        .map(|w| {
                            obj!(
                                "offset" => Value::from(w.offset),
                                "addr" => addr(w.addr),
                                "entropy" => f64::from(w.entropy),
                            )
                        })
                        .collect(),
                ),
            )
            .field("truncated_windows", Value::from(wins.len() > 500))
            .build());
    }
    Ok(e
        .field(
            "sections",
            Value::Array(
                file.analysis
                    .entropy
                    .iter()
                    .map(|s| {
                        obj!(
                            "section" => s.name.clone(),
                            "kind" => format!("{:?}", s.kind),
                            "size" => Value::from(s.size),
                            "entropy" => f64::from(s.entropy),
                            "high" => s.entropy >= occ_analyze::entropy::HIGH_ENTROPY,
                        )
                    })
                    .collect(),
            ),
        )
        .build())
}

fn binary_fingerprint(session: &Session, args: &Args) -> Result<Value, String> {
    let file = args.file(session)?;
    let fp = &file.analysis.fingerprint;
    let e = Envelope::ok(format!(
        "compiler={} language={} linkage={}{}",
        fp.compiler.label(),
        fp.language.label(),
        fp.linkage_label(),
        if fp.packer_suspicion.is_some() {
            ", packer suspected"
        } else {
            ""
        }
    ))
    .field("fingerprint", fingerprint_json(fp))
    .evidence(fp.compiler_claim.evidence.clone())
    .confidence(fp.compiler_claim.confidence)
    .caveats(
        fp.packer_suspicion
            .as_ref()
            .map(|c| c.caveats.clone())
            .unwrap_or_default(),
    );
    Ok(e.build())
}

// ---------------------------------------------------------------------------
// code tools
// ---------------------------------------------------------------------------

fn code_functions(session: &Session, args: &Args) -> Result<Value, String> {
    let file = args.file(session)?;
    let filter = args.opt_str("filter").map(|s| s.to_ascii_lowercase());
    let min_insns = args.u64_in("min_insns", 0, 0, u64::MAX)? as usize;
    let sort = args.one_of(
        "sort",
        "addr",
        &["addr", "size", "insns", "complexity", "callers", "callees"],
    )?;
    let (offset, limit) = args.page(100, 2000)?;
    let mut items: Vec<&Function> = file
        .analysis
        .functions
        .iter()
        .filter(|f| f.insns.len() >= min_insns)
        .filter(|f| {
            filter
                .as_ref()
                .map(|x| f.display_name().to_ascii_lowercase().contains(x))
                .unwrap_or(true)
        })
        .collect();
    match sort {
        "size" => items.sort_by_key(|f| std::cmp::Reverse(f.size)),
        "insns" => items.sort_by_key(|f| std::cmp::Reverse(f.insns.len())),
        "complexity" => items.sort_by_key(|f| std::cmp::Reverse(f.cyclomatic_complexity())),
        "callers" => items.sort_by_key(|f| std::cmp::Reverse(f.callers.len())),
        "callees" => items.sort_by_key(|f| std::cmp::Reverse(f.callees.len())),
        _ => items.sort_by_key(|f| f.entry.0),
    }
    let rows: Vec<Value> = items
        .iter()
        .map(|f| {
            obj!(
                "handle" => Handle::function(f.entry).to_string(),
                "name" => f.display_name(),
                "entry" => addr(f.entry),
                "size" => Value::from(f.size),
                "insns" => Value::from(f.insns.len()),
                "blocks" => Value::from(f.blocks.len()),
                "complexity" => Value::from(f.cyclomatic_complexity()),
                "callers" => Value::from(f.callers.len()),
                "callees" => Value::from(f.callees.len()),
                "thunk" => f.is_thunk,
            )
        })
        .collect();
    let total = rows.len();
    let (page, truncated) = paginate(&rows, offset, limit);
    Ok(Envelope::ok(format!(
        "{total} function(s); sorted by {sort}; {} total instructions",
        file.analysis.coverage.insns
    ))
    .field("functions", Value::Array(page))
    .field("total", Value::from(total))
    .truncated(truncated)
    .next(vec!["code.function with a fn: handle".into()])
    .build())
}

fn code_function(session: &Session, args: &Args) -> Result<Value, String> {
    let id = args.image(session)?;
    let file = args.file(session)?;
    let target = args.req_str("target")?;
    let t = handle_target(session, id, target)?;
    let f = function_of(&file, &t)
        .ok_or_else(|| format!("no function contains {target}"))?;
    let depth = args.depth()?;
    let include_body = args.bool_or("include_body", depth == Depth::Raw)?;
    let mut data = function_json(f, &file.analysis, depth, &file.image);
    if include_body
        && let Value::Object(m) = &mut data
    {
        m.insert(
            "body",
            Value::Array(
                f.insns
                    .iter()
                    .take(depth.limits().max_insns)
                    .map(insn_json)
                    .collect(),
            ),
        );
        m.insert("handle", Handle::function(f.entry).to_string());
    }
    Ok(Envelope::ok(format!(
        "{} at {}: {} insns, {} blocks, complexity {}, {} callers",
        f.display_name(),
        f.entry,
        f.insns.len(),
        f.blocks.len(),
        f.cyclomatic_complexity(),
        f.callers.len()
    ))
    .handle(Handle::function(f.entry).to_string())
    .field("function", data)
    .evidence(f.boundary_evidence.clone())
    .confidence(f.boundary_confidence)
    .next(vec![
        format!("code.blocks for {}", Handle::function(f.entry)),
        format!("code.xrefs direction=to target={}", Handle::function(f.entry)),
    ])
    .build())
}


fn code_decompile(session: &Session, args: &Args) -> Result<Value, String> {
    let id = args.image(session)?;
    let file = args.file(session)?;
    let target = args.req_str("target")?;
    let t = handle_target(session, id, target)?;
    let f = function_of(&file, &t).ok_or_else(|| format!("no function contains {target}"))?;
    let opts = occ_decompile::Options {
        max_blocks: args.u64_in("max_blocks", 20_000, 1, 200_000)? as usize,
        keep_comments: args.bool_or("comments", true)?,
        keep_dead: false,
        debug: occ_session::dbginfo::debug_info_for(&file, f),
        // The server prints a listing for a human, so the source lines the
        // debug info has are worth the extra comment.
        source_lines: true,
    };
    // Call targets are resolved through the recovered function table, so a call
    // reads as the name the rest of the tools use for that address.
    let analysis = &file.analysis;
    let name_of = |a: occ_core::Addr| -> String {
        analysis
            .function_at(a)
            .map(|f| f.display_name())
            .unwrap_or_else(|| format!("sub_{:x}", a.get()))
    };
    let d = occ_decompile::decompile(&file.image, f, &opts, &name_of);
    let code = occ_decompile::render::render(&d);
    let lines = code.lines().count();
    let mut notes: Vec<Value> = d.notes.iter().map(|n| Value::from(n.as_str())).collect();
    if d.unstructured > 0 {
        notes.push(Value::from(format!(
            "{} construct(s) fell back to goto: the graph is not reducible",
            d.unstructured
        )));
    }
    if !d.params.is_empty() && opts.debug.is_none() {
        notes.push(Value::from(
            "parameters are inferred from the registers the body reads; the count is a lower bound",
        ));
    }
    let mut e = Envelope::ok(format!(
        "{}: {} line(s) of pseudo-C, {} block(s), {} unstructured",
        d.name,
        lines,
        f.blocks.len(),
        d.unstructured
    ))
    .handle(Handle::function(f.entry).to_string())
    .field(
        "function",
        obj!(
            "entry" => Value::from(f.entry.to_string()),
            "name" => Value::from(d.name.as_str()),
            "params" => Value::Array(d.params.iter().map(|p| Value::from(p.as_str())).collect()),
            "unstructured" => Value::from(d.unstructured),
            "lines" => Value::from(lines),
        ),
    )
    .field("code", Value::from(code))
    .field("notes", Value::Array(notes))
    .evidence(f.boundary_evidence.clone())
    .confidence((f.boundary_confidence * 0.9).clamp(0.0, 1.0));
    match &opts.debug {
        Some(dbg) => {
            e = e.field(
                "debug",
                obj!(
                    "matched" => Value::from(true),
                    "source" => dbg.file.as_deref().map_or(Value::Null, Value::from),
                    "params" => Value::from(dbg.params.len() as u64),
                    "locals" => Value::from(dbg.locals.len() as u64),
                    "types" => Value::from(dbg.types.len() as u64),
                    "frame_base" => Value::from(if dbg.frame_base_is_cfa {
                        "cfa"
                    } else {
                        dbg.frame_base_reg.as_deref().unwrap_or("unknown")
                    }),
                ),
            );
        }
        None => {
            // Saying "no match" is different from saying nothing: an image
            // with DWARF but no DIE for this function is a coverage gap, and
            // the caller should know the names are mechanical.
            e = e.field(
                "debug",
                obj!(
                    "matched" => Value::from(false),
                    "reason" => Value::from(if file.dwarf.is_some() {
                        "the image has DWARF but no function entry covers this address"
                    } else {
                        "the image has no DWARF"
                    }),
                ),
            );
        }
    }
    Ok(e
        .next(vec![
            format!("code.disasm target={} for the instruction listing", Handle::function(f.entry)),
            "debug.variables for names and types when the image carries DWARF".into(),
        ])
        .build())
}

/// Turn a solver verdict into a word a model can branch on.
fn verdict_label(v: occ_sym::Verdict) -> &'static str {
    match v {
        occ_sym::Verdict::Satisfied => "satisfied",
        occ_sym::Verdict::Unsatisfiable => "unsatisfiable",
        occ_sym::Verdict::Unknown => "unknown",
    }
}

fn outcome_label(outcome: &occ_sym::Outcome) -> &'static str {
    match outcome {
        occ_sym::Outcome::Return => "return",
        occ_sym::Outcome::Unreachable => "unreachable",
        occ_sym::Outcome::Truncated(_) => "truncated",
        occ_sym::Outcome::Missing(_) => "missing_inputs",
    }
}

fn code_explore(session: &Session, args: &Args) -> Result<Value, String> {
    let id = args.image(session)?;
    let file = args.file(session)?;
    let target = args.req_str("target")?;
    let t = handle_target(session, id, target)?;
    let f = function_of(&file, &t).ok_or_else(|| format!("no function contains {target}"))?;

    let mode = match args.opt_str("mode") {
        Some("concrete") => occ_sym::Mode::Concrete,
        Some("symbolic") | None => occ_sym::Mode::Symbolic,
        Some(other) => {
            return Err(format!(
                "`mode` is {other:?}; use \"symbolic\" (default) or \"concrete\""
            ));
        }
    };
    let mut cfg = occ_sym::ExecConfig {
        mode,
        max_paths: args.u64_in("max_paths", 16, 1, 200)? as usize,
        max_states: args.u64_in("max_states", 128, 1, 4096)? as usize,
        max_loop: args.u64_in("unroll", 3, 1, 64)? as u16,
        ..occ_sym::ExecConfig::default()
    };
    // Inputs arrive as `{"rdi": 3, "rsi": "0x4000"}`; both forms are common in
    // the turns that produce them.
    if let Some(value) = args.raw("inputs") {
        let map = value
            .as_object()
            .ok_or("`inputs` must be an object of name to value")?;
        for (name, v) in map.iter() {
            let word = v
                .as_u64()
                .or_else(|| v.as_i64().map(|i| i as u64))
                .or_else(|| {
                    v.as_str()
                        .and_then(occ_core::addr::parse_addr)
                        .map(|a| a.get())
                })
                .ok_or_else(|| {
                    format!("`inputs.{name}` must be an integer or a hex string")
                })?;
            cfg.inputs.push((name.to_string(), word));
        }
    }
    if let Some(value) = args.raw("memory") {
        let map = value
            .as_object()
            .ok_or("`memory` must be an object of address to hex bytes")?;
        for (at, bytes) in map.iter() {
            let addr = occ_core::addr::parse_addr(at)
                .ok_or_else(|| format!("`memory` key {at:?} is not an address"))?;
            let hex = bytes
                .as_str()
                .ok_or_else(|| format!("`memory.{at}` must be a hex byte string"))?;
            cfg.memory.push((addr, parse_hex_bytes(hex)?));
        }
    }

    let analysis = &file.analysis;
    let name_of = |a: Addr| -> String {
        analysis
            .function_at(a)
            .map(|f| f.display_name())
            .unwrap_or_else(|| format!("sub_{:x}", a.get()))
    };
    // Initialised bytes come from the image, so a read of a global resolves to
    // the value in the file instead of turning into an unknown.
    let view = occ_analyze::View::new(&file.image, &file.data);
    let read_static = |at: Addr, bytes: u8| view.read_uint(at, usize::from(bytes));
    let report = occ_sym::explore(
        &file.image,
        f,
        &occ_decompile::Options {
            max_blocks: 20_000,
            keep_comments: false,
            keep_dead: false,
            debug: None,
            source_lines: false,
        },
        &name_of,
        &read_static,
        &file.analysis.functions,
        &cfg,
    );

    let paths: Vec<Value> = report
        .paths
        .iter()
        .map(|p| {
            let mut o = occ_json::Map::new();
            o.insert("outcome", Value::from(outcome_label(&p.outcome)));
            if let occ_sym::Outcome::Truncated(why) = &p.outcome {
                o.insert("truncated_because", Value::from(*why));
            }
            if let occ_sym::Outcome::Missing(names) = &p.outcome {
                o.insert(
                    "missing",
                    Value::Array(names.iter().map(|n| Value::from(n.as_str())).collect()),
                );
            }
            o.insert("solved", Value::from(verdict_label(p.model.verdict)));
            o.insert("feasible", Value::from(p.is_feasible()));
            if !p.model.inputs.is_empty() {
                let mut inputs = occ_json::Map::new();
                for (name, value) in &p.model.inputs {
                    inputs.insert(name.as_str(), Value::from(*value));
                }
                o.insert("inputs", Value::Object(inputs));
            }
            let conditions = p.conditions();
            if !conditions.is_empty() {
                o.insert(
                    "conditions",
                    Value::Array(conditions.into_iter().map(Value::from).collect()),
                );
            }
            o.insert("returns", Value::from(p.ret.clone().unwrap_or_default()));
            o.insert(
                "blocks",
                Value::Array(p.blocks.iter().map(|a| addr(*a)).collect()),
            );
            if !p.calls.is_empty() {
                o.insert(
                    "calls",
                    Value::Array(
                        p.calls
                            .iter()
                            .map(|c| {
                                obj!(
                                    "name" => c.name.clone(),
                                    "at" => addr(c.addr),
                                    "args" => Value::Array(
                                        c.args.iter().map(|a| Value::from(a.to_string())).collect()
                                    ),
                                )
                            })
                            .collect(),
                    ),
                );
            }
            Value::Object(o)
        })
        .collect();

    let feasible = report.feasible();
    let returns = report.returns();
    let mut notes: Vec<Value> = report.notes.iter().map(|n| Value::from(n.as_str())).collect();
    notes.extend(report.lift_notes.iter().map(|n| Value::from(n.as_str())));
    if cfg.mode == occ_sym::Mode::Symbolic {
        notes.push(Value::from(
            "calls are not interpreted: a return value from a callee is unconstrained",
        ));
    }
    let summary = if report.paths.is_empty() {
        format!("{}: no path completed", report.name)
    } else {
        format!(
            "{}: {} of {} path(s) feasible, {} distinct return value(s) over {} block step(s)",
            report.name,
            feasible,
            report.paths.len(),
            returns.len(),
            report.steps
        )
    };
    Ok(Envelope::ok(summary)
        .handle(Handle::function(f.entry).to_string())
        .field(
            "function",
            obj!(
                "entry" => addr(f.entry),
                "name" => Value::from(report.name.as_str()),
                "params" => Value::Array(
                    report.params.iter().map(|p| Value::from(p.as_str())).collect()
                ),
                "param_regs" => Value::Array(
                    report.param_regs.iter().map(|p| Value::from(p.as_str())).collect()
                ),
                "mode" => Value::from(match cfg.mode {
                    occ_sym::Mode::Symbolic => "symbolic",
                    occ_sym::Mode::Concrete => "concrete",
                }),
            ),
        )
        .field("feasible", Value::from(feasible))
        .field("paths", Value::Array(paths))
        .field("notes", Value::Array(notes))
        .evidence(f.boundary_evidence.clone())
        .confidence((f.boundary_confidence * 0.8).clamp(0.0, 1.0))
        .next(vec![
            format!(
                "code.decompile target={} to read the same paths as pseudo-C",
                Handle::function(f.entry)
            ),
            "code.taint for a source-to-sink summary without unrolling".into(),
        ])
        .build())
}

fn parse_hex_bytes(text: &str) -> Result<Vec<u8>, String> {
    let cleaned: String = text
        .chars()
        .filter(|c| !c.is_ascii_whitespace() && *c != '_')
        .collect();
    if cleaned.len() % 2 != 0 || !cleaned.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!(
            "{text:?} is not a hex byte string; expected an even number of hex digits"
        ));
    }
    Ok((0..cleaned.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&cleaned[i..i + 2], 16).unwrap_or(0))
        .collect())
}

fn code_taint(session: &Session, args: &Args) -> Result<Value, String> {
    let id = args.image(session)?;
    let file = args.file(session)?;
    let target = args.req_str("target")?;
    let t = handle_target(session, id, target)?;
    let f = function_of(&file, &t).ok_or_else(|| format!("no function contains {target}"))?;

    // Calls and branches by default: a store of a tainted value is usually the
    // spill of the same value, and reporting it hides the interesting rows.
    let mut sinks = occ_sym::SinkFilter {
        any_call: true,
        calls: Vec::new(),
        branches: true,
        stores: false,
        forwards: std::collections::HashMap::new(),
    };
    if let Some(value) = args.raw("sinks") {
        let list = value.as_array().ok_or("`sinks` must be an array")?;
        if list.is_empty() {
            return Err("`sinks` is empty; drop the argument to report every call".into());
        }
        sinks.any_call = false;
        sinks.branches = false;
        for entry in list {
            let word = entry.as_str().ok_or("`sinks` entries must be strings")?;
            match word {
                "call" | "calls" | "*" => sinks.any_call = true,
                "branch" | "branches" => sinks.branches = true,
                "store" | "stores" => sinks.stores = true,
                other => sinks.calls.push(other.to_string()),
            }
        }
    }

    let analysis = &file.analysis;
    // A call site in a stripped image names a `sub_` body. Recording what each
    // body forwards to lets a name filter match the call that carries the
    // argument to the library function it eventually reaches.
    if !sinks.any_call && !sinks.calls.is_empty() {
        sinks.forwards = callee_forwards(analysis);
    }
    let unmatched: Vec<String> = if sinks.any_call {
        Vec::new()
    } else {
        sinks
            .calls
            .iter()
            .filter(|wanted| !image_names_any(analysis, &file.image, wanted))
            .cloned()
            .collect()
    };
    let name_of = |a: Addr| -> String {
        analysis
            .function_at(a)
            .map(|f| f.display_name())
            .unwrap_or_else(|| format!("sub_{:x}", a.get()))
    };
    let opts = occ_decompile::Options {
        max_blocks: 20_000,
        keep_comments: false,
        keep_dead: false,
        debug: None,
        source_lines: false,
    };
    // Parameters are the default source: the question is nearly always "what
    // does the caller control here".
    let param_regs = occ_decompile::build_program(&file.image, f, &opts, &name_of).param_regs;
    let mut sources = Vec::new();
    match args.raw("sources") {
        Some(value) => {
            let list = value.as_array().ok_or("`sources` must be an array")?;
            for entry in list {
                let word = entry
                    .as_str()
                    .ok_or("`sources` entries must be strings")?;
                sources.push(parse_source(word)?);
            }
        }
        None => {
            for p in &param_regs {
                sources.push(occ_sym::Source::Name(p.clone()));
            }
        }
    }
    if sources.is_empty() {
        return Err(
            "no sources: the function reads no parameter registers, so name one explicitly (e.g. \"rdi\", \"addr:0x4000:8\")"
                .into(),
        );
    }

    let report = occ_sym::trace(
        &file.image,
        f,
        &opts,
        &name_of,
        &sources,
        &occ_sym::TaintConfig {
            max_iterations: 64,
            sinks,
        },
    );
    let flows: Vec<Value> = report
        .flows
        .iter()
        .map(|flow| {
            obj!(
                "source" => Value::from(flow.source.as_str()),
                "sink" => Value::from(flow.sink.as_str()),
                "at" => addr(flow.at),
                "arg" => flow.arg.map_or(Value::Null, |a| Value::from(a as u64)),
                "value" => Value::from(flow.value.as_str()),
                "path" => Value::Array(flow.path.iter().map(|a| addr(*a)).collect()),
            )
        })
        .collect();
    let mut notes: Vec<Value> = report.notes.iter().map(|n| Value::from(n.as_str())).collect();
    notes.extend(report.lift_notes.iter().map(|n| Value::from(n.as_str())));
    notes.push(Value::from(
        "calls are not interpreted: their return values inherit their arguments",
    ));
    let summary = if report.flows.is_empty() {
        format!(
            "{}: no flow from {} source(s) to a sink ({} block(s), {} round(s))",
            report.name,
            report.sources.len(),
            report.blocks,
            report.iterations
        )
    } else {
        format!(
            "{}: {} flow(s) from {} source(s) to {} sink(s), {} block(s), {} round(s)",
            report.name,
            report.flows.len(),
            report.reached().len(),
            report.sinks().len(),
            report.blocks,
            report.iterations
        )
    };
    let mut e = Envelope::ok(summary)
        .handle(Handle::function(f.entry).to_string())
        .field(
            "function",
            obj!(
                "entry" => addr(f.entry),
                "name" => Value::from(report.name.as_str()),
                "params" => Value::Array(
                    report.params.iter().map(|p| Value::from(p.as_str())).collect()
                ),
                "param_regs" => Value::Array(
                    report.param_regs.iter().map(|p| Value::from(p.as_str())).collect()
                ),
            ),
        )
        .field(
            "sources",
            Value::Array(report.sources.iter().map(|s| Value::from(s.as_str())).collect()),
        )
        .field("flows", Value::Array(flows))
        .field("converged", Value::from(report.converged))
        .field("blocks", Value::from(report.blocks as u64))
        .field("iterations", Value::from(report.iterations as u64))
        .field("notes", Value::Array(notes))
        .evidence(f.boundary_evidence.clone())
        .confidence((f.boundary_confidence * 0.85).clamp(0.0, 1.0))
        .next(vec![
            "code.decompile for the statements the flow passes through".into(),
            "code.explore to see which inputs actually reach the branch".into(),
        ]);
    if !unmatched.is_empty() {
        // Saying "no flows" without saying why reads as a clean bill of health,
        // and here the reason is that the filter cannot match anything.
        e = e.caveats(vec![format!(
            "no call site or forwarded name in this image matches {}; it has {} import(s) and {} named function(s), so a name filter has nothing to match. Drop `sinks` to report every call, or pass `sinks: [\"call\"]`",
            unmatched
                .iter()
                .map(|w| format!("`{w}`"))
                .collect::<Vec<_>>()
                .join(", "),
            file.image.imports.len(),
            analysis
                .functions
                .iter()
                .filter(|f| f.name.is_some())
                .count()
        )]);
    }
    Ok(e.build())
}

/// What each recovered function forwards to.
///
/// The value is every name the body can reach in one call: its imports and the
/// names of the functions it calls. A second pass folds in what those forward
/// to, which covers the wrapper-a-call pattern (`main -> sub_x -> memcpy`).
fn callee_forwards(
    analysis: &occ_analyze::Analysis,
) -> std::collections::HashMap<String, Vec<String>> {
    let mut out: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for f in &analysis.functions {
        if f.insns.is_empty() {
            continue;
        }
        let mut names: Vec<String> = f.import_calls.clone();
        for callee in &f.callees {
            if let Some(g) = analysis.function_at(*callee) {
                let n = g.display_name();
                if n != f.display_name() {
                    names.push(n);
                }
            }
        }
        names.sort();
        names.dedup();
        if !names.is_empty() {
            out.insert(f.display_name(), names);
        }
    }
    let snapshot = out.clone();
    for names in out.values_mut() {
        let mut extra: Vec<String> = names
            .iter()
            .flat_map(|n| snapshot.get(n).cloned().unwrap_or_default())
            .collect();
        extra.sort();
        extra.dedup();
        names.extend(extra);
        names.sort();
        names.dedup();
    }
    out
}

/// True when `wanted` names anything the image can reach: an import, an
/// exported name, a symbol, a recovered function, or a name one step behind a
/// recovered function.
fn image_names_any(
    analysis: &occ_analyze::Analysis,
    image: &occ_core::Image,
    wanted: &str,
) -> bool {
    image
        .imports
        .iter()
        .any(|i| i.name.contains(wanted))
        || image
        .exports
        .iter()
        .any(|e| e.name.contains(wanted))
        || image.symbols.iter().any(|s| s.name.contains(wanted))
        || analysis
            .functions
            .iter()
            .any(|f| f.display_name().contains(wanted))
        || callee_forwards(analysis)
            .values()
            .any(|names| names.iter().any(|n| n.contains(wanted)))
}

/// Parse one source label.
///
/// `rdi` and `name:rdi` name a value; `addr:0x4000:8` and a bare `0x4000` name
/// a memory range. Both spellings exist because both are what a caller types.
fn parse_source(word: &str) -> Result<occ_sym::Source, String> {
    if let Some(rest) = word.strip_prefix("name:") {
        return Ok(occ_sym::Source::Name(rest.to_string()));
    }
    if let Some(rest) = word.strip_prefix("addr:") {
        let (at, bytes) = match rest.split_once(':') {
            Some((a, b)) => (a, b.parse::<u8>().unwrap_or(8)),
            None => (rest, 8),
        };
        let addr = occ_core::addr::parse_addr(at)
            .ok_or_else(|| format!("`addr:` expects an address, got {at:?}"))?;
        return Ok(occ_sym::Source::Memory {
            addr,
            bytes: bytes.clamp(1, 8),
        });
    }
    if word.starts_with("0x")
        && let Some(a) = occ_core::addr::parse_addr(word)
    {
        return Ok(occ_sym::Source::Memory {
            addr: a,
            bytes: 8,
        });
    }
    Ok(occ_sym::Source::Name(word.to_string()))
}

fn code_disasm(session: &Session, args: &Args) -> Result<Value, String> {
    let id = args.image(session)?;
    let file = args.file(session)?;
    let target = args.req_str("target")?;
    let count = args.u64_in("count", 40, 1, 2000)? as usize;
    let t = handle_target(session, id, target)?;
    let start = match &t {
        Target::Function { entry, .. } => *entry,
        Target::Insn { addr, .. } => *addr,
        _ => return Err("target must be a function or an address".into()),
    };
    let f = file.analysis.function_covering(start);
    let mut out = Vec::new();
    for i in 0..count {
        let addr = match f {
            Some(f) => {
                let idx = f.insns.iter().position(|x| x.addr == start).unwrap_or(0);
                match f.insns.get(idx + i) {
                    Some(x) => x,
                    None => break,
                }
            }
            None => match file.analysis.insn_at(start) {
                Some(x) if i == 0 => x,
                _ => break,
            },
        };
        let mut v = insn_json(addr);
        // Every instruction says which source line produced it, when the image
        // carries a line table. This is the join that turns a disassembly into
        // something a reader can check against the source.
        if let Some((file, line, _col)) = file
            .dwarf
            .as_ref()
            .and_then(|d| d.line_for(addr.addr))
            && let Some(obj) = v.as_object_mut()
        {
            obj.insert("source", Value::from(format!("{file}:{line}")));
        }
        out.push(v);
    }
    if out.is_empty() {
        return Err(format!("no instruction at {start}"));
    }
    Ok(Envelope::ok(format!(
        "{} instruction(s) from {}",
        out.len(),
        start
    ))
    .handle(Handle::insn(start).to_string())
    .field("addr", addr(start))
    .field("function", f.map(|f| f.display_name()).unwrap_or_default())
    .field("insns", Value::Array(out))
    .build())
}

fn code_blocks(session: &Session, args: &Args) -> Result<Value, String> {
    let id = args.image(session)?;
    let file = args.file(session)?;
    let target = args.req_str("target")?;
    let t = handle_target(session, id, target)?;
    let f = function_of(&file, &t).ok_or_else(|| format!("no function contains {target}"))?;
    let include_body = args.bool_or("include_body", false)?;
    let blocks: Vec<Value> = f
        .blocks
        .iter()
        .map(|b| block_json(b, f, include_body, 40))
        .collect();
    let e = Envelope::ok(format!(
        "{} blocks, {} edges, complexity {}",
        f.blocks.len(),
        f.blocks.iter().map(|b| b.edges.len()).sum::<usize>(),
        f.cyclomatic_complexity()
    ))
    .handle(Handle::function(f.entry).to_string())
    .field("entry", addr(f.entry))
    .field("blocks", Value::Array(blocks))
    .field("loops", Value::from(diff::has_loop(f)))
    .next(vec![format!(
        "code.disasm target={} to read the body",
        Handle::function(f.entry)
    )]);
    Ok(e.build())
}

fn code_indirect(session: &Session, args: &Args) -> Result<Value, String> {
    let file = args.file(session)?;
    let view = occ_analyze::View::new(&file.image, &file.data);
    let report = occ_vsa::analyze(&view, &file.analysis, &occ_vsa::Options::default());
    let only = args.opt_str("function");
    let kind = args.opt_str("kind").unwrap_or("both");
    let wanted = |entry: Addr| -> bool {
        let Some(only) = &only else { return true };
        if let Some(rest) = only.strip_prefix("fn:") {
            if let Ok(v) = u64::from_str_radix(rest.trim_start_matches("0x"), 16) {
                return entry.0 == v;
            }
        }
        file.analysis
            .functions
            .iter()
            .find(|f| f.entry == entry)
            .map(|f| f.display_name() == *only)
            .unwrap_or(false)
    };
    let mut rows: Vec<Value> = Vec::new();
    if kind != "jumps" {
        for call in report.calls.iter().filter(|c| wanted(c.in_function)) {
            rows.push(obj!(
                "kind" => "call",
                "at" => addr(call.at),
                "function" => addr(call.in_function),
                "via" => call.via.clone(),
                "targets" => Value::Array(
                    call.targets.iter().map(|t| addr(*t)).collect()
                ),
                "evidence" => call.evidence.clone(),
            ));
        }
    }
    if kind != "calls" {
        for jump in report.jumps.iter().filter(|j| wanted(j.in_function)) {
            let table = match jump.table {
                Some(t) => obj!(
                    "base" => addr(t.base),
                    "stride" => Value::from(t.stride),
                    "bias" => Value::from(t.bias),
                    "entries" => Value::from(t.entries as u64),
                ),
                None => Value::Null,
            };
            rows.push(obj!(
                "kind" => "jump",
                "at" => addr(jump.at),
                "function" => addr(jump.in_function),
                "via" => jump.via.clone(),
                "table" => table,
                "two_level" => jump.two_level,
                "cases" => Value::Array(
                    jump.cases
                        .iter()
                        .map(|c| obj!(
                            "index" => Value::from(c.index as u64),
                            "label" => Value::from(c.label),
                            "target" => addr(c.target),
                        ))
                        .collect()
                ),
                "evidence" => jump.evidence.clone(),
            ));
        }
    }
    let total = rows.len();
    let (offset, limit) = args.page(50, 500)?;
    let (page, truncated) = paginate(&rows, offset, limit);
    let stats = &report.stats;
    Ok(Envelope::ok(format!(
        "{} indirect site(s): {} of {} calls and {} of {} jumps resolved; \
         {} table(s) read, {} two-level; the syntactic resolver found {} table(s) in the same image",
        total,
        stats.calls_resolved,
        stats.call_sites,
        stats.jumps_resolved,
        stats.jump_sites,
        stats.tables,
        stats.two_level,
        stats.syntactic_tables,
    ))
    .field("sites", Value::Array(page))
    .field("total", Value::from(total as u64))
    .field(
        "stats",
        obj!(
            "functions" => Value::from(stats.functions as u64),
            "blocks" => Value::from(stats.blocks as u64),
            "call_sites" => Value::from(stats.call_sites as u64),
            "calls_resolved" => Value::from(stats.calls_resolved as u64),
            "jump_sites" => Value::from(stats.jump_sites as u64),
            "jumps_resolved" => Value::from(stats.jumps_resolved as u64),
            "tables" => Value::from(stats.tables as u64),
            "two_level" => Value::from(stats.two_level as u64),
            "syntactic_tables" => Value::from(stats.syntactic_tables as u64),
        ),
    )
    .truncated(truncated)
    .next(vec!["code.function on a resolved target".into()])
    .build())
}

fn code_xrefs(session: &Session, args: &Args) -> Result<Value, String> {
    let id = args.image(session)?;
    let file = args.file(session)?;
    let target = args.req_str("target")?;
    let direction = args.opt_str("direction").unwrap_or("to");
    let kind = args.opt_str("kind");
    let limit = args.u64_in("limit", 100, 1, 2000)? as usize;
    let t = handle_target(session, id, target)?;
    let at = match &t {
        Target::Function { entry, .. } => *entry,
        Target::Insn { addr, .. } => *addr,
        Target::Str { addr, .. } => *addr,
        Target::Symbol { addr, .. } => *addr,
        _ => return Err("target must name an address, function, string or symbol".into()),
    };
    let kind_of = |x: &occ_core::Xref| {
        matches!(
            (kind, x.kind),
            (None, _)
                | (Some("call"), occ_core::XrefKind::Call)
                | (Some("tailcall"), occ_core::XrefKind::TailCall)
                | (Some("jump"), occ_core::XrefKind::Jump)
                | (Some("condjump"), occ_core::XrefKind::ConditionalJump)
                | (Some("dataread"), occ_core::XrefKind::DataRead)
                | (Some("datawrite"), occ_core::XrefKind::DataWrite)
                | (Some("addresstaken"), occ_core::XrefKind::AddressTaken)
                | (Some("thunk"), occ_core::XrefKind::ImportThunk)
        )
    };
    let mut rows = Vec::new();
    if direction == "to" || direction == "both" {
        for x in file.analysis.xrefs.to(at).filter(|x| kind_of(x)).take(limit) {
            rows.push(xref_json(x));
        }
    }
    if direction == "from" || direction == "both" {
        for x in file.analysis.xrefs.from(at).filter(|x| kind_of(x)).take(limit) {
            rows.push(xref_json(x));
        }
    }
    Ok(Envelope::ok(format!(
        "{} reference(s) {} {at}",
        rows.len(),
        if direction == "from" { "from" } else { "to" }
    ))
    .handle(Handle::new(occ_core::HandleKind::Insn, at.to_string()).to_string())
    .field("direction", direction)
    .field("xrefs", Value::Array(rows))
    .build())
}

fn code_callgraph(session: &Session, args: &Args) -> Result<Value, String> {
    let id = args.image(session)?;
    let file = args.file(session)?;
    let target = args.req_str("target")?;
    let direction = args.opt_str("direction").unwrap_or("callers");
    let depth = args.u64_in("depth", 2, 1, 4)? as usize;
    let max_nodes = args.u64_in("max_nodes", 200, 1, 5000)? as usize;
    let t = handle_target(session, id, target)?;
    let root = match &t {
        Target::Function { entry, .. } => *entry,
        Target::Insn { addr, .. } => file
            .analysis
            .function_covering(*addr)
            .map(|f| f.entry)
            .ok_or("no function at that address")?,
        _ => return Err("target must be a function".into()),
    };

    let mut seen = std::collections::HashSet::new();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut frontier = vec![(root, 0usize)];
    seen.insert(root.0);
    while let Some((node, d)) = frontier.pop() {
        if nodes.len() >= max_nodes {
            break;
        }
        let Some(f) = file.analysis.function_at(node) else {
            continue;
        };
        nodes.push(obj!(
            "handle" => Handle::function(f.entry).to_string(),
            "name" => f.display_name(),
            "depth" => Value::from(d),
            "insns" => Value::from(f.insns.len()),
        ));
        if d >= depth {
            continue;
        }
        let next: Vec<Addr> = if direction == "callees" {
            f.callees.clone()
        } else {
            f.callers.clone()
        };
        for n in next {
            edges.push(obj!(
                "from" => addr(if direction == "callees" { f.entry } else { n }),
                "to" => addr(if direction == "callees" { n } else { f.entry }),
            ));
            if seen.insert(n.0) {
                frontier.push((n, d + 1));
            }
        }
    }
    Ok(Envelope::ok(format!(
        "{} node(s), {} edge(s), {} direction, depth {depth}",
        nodes.len(),
        edges.len(),
        direction
    ))
    .handle(Handle::function(root).to_string())
    .field("root", addr(root))
    .field("nodes", Value::Array(nodes))
    .field("edges", Value::Array(edges))
    .truncated(seen.len() > max_nodes)
    .build())
}

fn code_search(session: &Session, args: &Args) -> Result<Value, String> {
    let file = args.file(session)?;
    let mnemonic = args.opt_str("mnemonic").map(|s| s.to_ascii_lowercase());
    let text = args.opt_str("text").map(|s| s.to_ascii_lowercase());
    let value = args.opt_addr("value")?;
    if mnemonic.is_none() && text.is_none() && value.is_none() {
        return Err("give at least one of: mnemonic, text, value".into());
    }
    let limit = args.u64_in("limit", 50, 1, 1000)? as usize;
    let mut rows = Vec::new();
    'outer: for f in &file.analysis.functions {
        for i in &f.insns {
            let ok = mnemonic
                .as_ref()
                .map(|m| i.mnemonic.to_ascii_lowercase() == *m || i.mnemonic.to_ascii_lowercase().starts_with(m))
                .unwrap_or(true)
                && text
                    .as_ref()
                    .map(|t| i.text.to_ascii_lowercase().contains(t))
                    .unwrap_or(true)
                && value
                    .map(|v| {
                        i.mem_reads.contains(&v)
                            || i.mem_writes.contains(&v)
                            || i.text.contains(&v.to_string())
                    })
                    .unwrap_or(true);
            if ok {
                rows.push(obj!(
                    "insn" => Handle::insn(i.addr).to_string(),
                    "function" => Handle::function(f.entry).to_string(),
                    "addr" => addr(i.addr),
                    "text" => i.text.clone(),
                ));
                if rows.len() >= limit {
                    break 'outer;
                }
            }
        }
    }
    let truncated = rows.len() >= limit;
    Ok(Envelope::ok(format!("{} matching instruction(s)", rows.len()))
        .field("hits", Value::Array(rows))
        .truncated(truncated)
        .build())
}

// ---------------------------------------------------------------------------
// hypothesis tools
// ---------------------------------------------------------------------------

fn hypothesis_create(session: &Session, args: &Args) -> Result<Value, String> {
    let target = args.req_str("target")?;
    let statement = args.req_str("statement")?;
    // A claim needs a subject that stays the same subject. An address is not
    // one: every PIE image has a `fn:0x401000`. The image is resolved here,
    // where `args.image` already refuses to guess between open images, and
    // recorded with the hypothesis.
    let (handle, image) = if target.contains(':') {
        let h = Handle::parse(target).map_err(|e| e.to_string())?;
        let image = match h.kind() {
            occ_core::HandleKind::Image => {
                let id = h.value().parse::<u32>().map_err(|_| {
                    format!("`target` is {target:?}; image handles carry a numeric id")
                })?;
                Some(id)
            }
            _ => Some(args.image(session)?),
        };
        (h, image)
    } else if let Some(addr) = occ_core::addr::parse_addr(target) {
        (Handle::function(addr), Some(args.image(session)?))
    } else {
        (Handle::symbol(target), Some(args.image(session)?))
    };
    let hyp_id = session
        .hypothesise(handle.clone(), image, statement)
        .map_err(|e| e.to_string())?;
    Ok(Envelope::ok(format!("hypothesis {hyp_id} recorded"))
        .handle(Handle::hypothesis(hyp_id).to_string())
        .field("id", Value::from(hyp_id))
        .field("target", handle.to_string())
        .field("statement", statement)
        .next(vec![format!(
            "hypothesis.evidence id={hyp_id} detail=... support=true weight=0.6"
        )])
        .build())
}

fn hypothesis_evidence(session: &Session, args: &Args) -> Result<Value, String> {
    let id = args.opt_u64("id")?.ok_or("`id` is required")?;
    let detail = args.req_str("detail")?;
    let support = args.bool_or("support", true)?;
    // A weight outside the unit interval is not a weaker observation, it is a
    // number in the wrong range: say so instead of clamping it into agreement.
    let weight = match args.raw("weight") {
        None => 0.5,
        Some(v) => {
            let w = v
                .as_f64()
                .ok_or_else(|| format!("`weight` must be a number, got {}", v.type_name()))?;
            if !(0.0..=1.0).contains(&w) {
                return Err(format!("`weight` must be between 0 and 1, got {w}"));
            }
            w as f32
        }
    };
    let at = args.opt_addr("at")?;
    let kind = match args.one_of(
        "kind",
        "bytes",
        &[
            "bytes", "cfg", "table", "symbol", "string", "import", "const", "entropy", "xref",
            "trace", "format", "absence",
        ],
    )? {
        "cfg" => EvidenceKind::ControlFlow,
        "table" => EvidenceKind::Table,
        "symbol" => EvidenceKind::Symbol,
        "string" => EvidenceKind::StringRef,
        "import" => EvidenceKind::ImportCall,
        "const" => EvidenceKind::Constant,
        "entropy" => EvidenceKind::Entropy,
        "xref" => EvidenceKind::CrossReference,
        "trace" => EvidenceKind::DynamicTrace,
        "format" => EvidenceKind::FormatMeta,
        "absence" => EvidenceKind::Absence,
        _ => EvidenceKind::Bytes,
    };
    let mut ev = if support {
        Evidence::support(kind, EvidenceSource::DataFlow, detail, weight)
    } else {
        Evidence::refute(kind, EvidenceSource::DataFlow, detail, weight)
    };
    if let Some(a) = at {
        ev = ev.at(a);
    }
    let (verdict, confidence) = session.add_evidence(id, ev).map_err(|e| e.to_string())?;
    Ok(Envelope::ok(format!("hypothesis {id}: {}", verdict.label()))
        .handle(Handle::hypothesis(id).to_string())
        .field("verdict", verdict.label())
        .field("confidence", f64::from(confidence))
        .build())
}

fn hypothesis_list(session: &Session, _args: &Args) -> Result<Value, String> {
    let hyps = session.hypotheses();
    Ok(Envelope::ok(format!("{} hypothesis(es)", hyps.len()))
        .field(
            "hypotheses",
            Value::Array(
                hyps.iter()
                    .map(|h| {
                        obj!(
                            "id" => Value::from(h.id),
                            // The hypothesis' own handle, not the handle of what
                            // it is about: `hyp:3` is what the other tools take,
                            // and printing `fn:0x4020` here gave the caller a
                            // handle it could not use for the next call.
                            "handle" => h.handle().to_string(),
                            "target" => h.handle.to_string(),
                            "statement" => h.statement.clone(),
                            "verdict" => h.verdict().label(),
                            "confidence" => f64::from(h.confidence),
                            "evidence" => Value::from(h.evidence.len()),
                        )
                    })
                    .collect(),
            ),
        )
        .build())
}

// ---------------------------------------------------------------------------
// diff and scan tools
// ---------------------------------------------------------------------------

fn diff_functions(session: &Session, args: &Args) -> Result<Value, String> {
    let a_id = args.opt_u64("image_a")?.ok_or("`image_a` is required")? as u32;
    let b_id = args.opt_u64("image_b")?.ok_or("`image_b` is required")? as u32;
    let fa = session.image(a_id).ok_or_else(|| format!("image {a_id} is not open"))?;
    let fb = session.image(b_id).ok_or_else(|| format!("image {b_id} is not open"))?;
    let d = diff::diff(&fa.analysis, &fb.analysis);
    let only_changed = args.bool_or("only_changed", true)?;
    let limit = args.u64_in("limit", 100, 1, 2000)? as usize;
    let selected: Vec<&diff::Pair> = if only_changed {
        d.changed().into_iter().take(limit).collect()
    } else {
        d.pairs.iter().take(limit).collect()
    };
    let rows: Vec<Value> = selected
        .iter()
        .map(|p| {
            obj!(
                "name" => p.name(),
                "change" => p.change.label(),
                "a" => match &p.a {
                    Some(s) => obj!(
                        "handle" => Handle::function(s.entry).to_string(),
                        "entry" => addr(s.entry),
                        "size" => Value::from(s.size),
                        "insns" => Value::from(s.insns),
                    ),
                    None => Value::Null,
                },
                "b" => match &p.b {
                    Some(s) => obj!(
                        "handle" => Handle::function(s.entry).to_string(),
                        "entry" => addr(s.entry),
                        "size" => Value::from(s.size),
                        "insns" => Value::from(s.insns),
                    ),
                    None => Value::Null,
                },
            )
        })
        .collect();
    Ok(Envelope::ok(d.summary())
        .field("summary", d.summary())
        .field("pairs", Value::Array(rows))
        .field("counts", obj!(
            "identical" => Value::from(d.count(diff::Change::Identical)),
            "modified" => Value::from(d.count(diff::Change::Modified)),
            "rewritten" => Value::from(d.count(diff::Change::Rewritten)),
            "added" => Value::from(d.count(diff::Change::Added)),
            "removed" => Value::from(d.count(diff::Change::Removed)),
        ))
        .truncated(d.pairs.len() > limit)
        .build())
}

fn scan_pattern(session: &Session, args: &Args) -> Result<Value, String> {
    let file = args.file(session)?;
    let source = args.req_str("rules")?;
    let rules = pattern::parse(source).map_err(|e| format!("rule parse error: {e}"))?;
    let sections: Vec<pattern::SectionRange> = file
        .image
        .sections
        .iter()
        .filter(|s| s.file_size > 0)
        .map(|s| pattern::SectionRange {
            name: s.name.clone(),
            start: s.addr.get(),
            size: s.file_size,
        })
        .collect();
    let limit = args.u64_in("limit", 100, 1, 1000)? as usize;

    // One row per (rule, address). A rule whose string occurs five times in a
    // section is five findings; the previous shape kept only the lowest
    // address and reported one, so the caller could not tell a single hit from
    // a sweep.
    let mut rows: Vec<(String, String, Addr, Vec<String>)> = Vec::new();
    for s in &sections {
        let Some(bytes) = file
            .image
            .section_by_name(&s.name)
            .and_then(|sec| {
                let view = occ_analyze::view::View::new(&file.image, &file.data);
                view.bytes(sec.addr, sec.file_size as usize)
            })
        else {
            continue;
        };
        for m in pattern::scan(bytes, s.start, &rules, &sections) {
            for a in &m.addrs {
                let ids: Vec<String> = m
                    .strings
                    .iter()
                    .filter(|(_, at)| at.contains(a))
                    .map(|(id, _)| id.clone())
                    .collect();
                rows.push((m.rule.clone(), s.name.clone(), *a, ids));
            }
        }
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.2 .0.cmp(&b.2 .0)));

    let known_strings: std::collections::HashSet<u64> =
        file.analysis.strings.iter().map(|s| s.addr.0).collect();
    let total = rows.len();
    let shown: Vec<Value> = rows
        .iter()
        .take(limit)
        .map(|(rule, section, at, ids)| {
            let mut m = occ_json::Map::new();
            m.insert("rule", Value::String(rule.clone()));
            m.insert("addr", addr(*at));
            m.insert("section", Value::String(section.clone()));
            m.insert(
                "strings",
                Value::Array(ids.iter().cloned().map(Value::String).collect()),
            );
            // A hit on a string the string pass recovered can be addressed by
            // handle; a hit in anonymous bytes cannot, and inventing an
            // instruction handle for data would resolve to the wrong thing.
            if known_strings.contains(&at.0) {
                m.insert(
                    "handle",
                    Value::String(Handle::new(occ_core::HandleKind::Str, at.0.to_string()).to_string()),
                );
            }
            Value::Object(m)
        })
        .collect();

    let mut per_rule: Vec<(String, usize)> = Vec::new();
    for (rule, _, _, _) in &rows {
        match per_rule.iter_mut().find(|(name, _)| name == rule) {
            Some((_, n)) => *n += 1,
            None => per_rule.push((rule.clone(), 1)),
        }
    }
    let rule_rows: Vec<Value> = per_rule
        .iter()
        .map(|(name, n)| obj!("rule" => name.clone(), "matches" => Value::from(*n as u64)))
        .collect();

    let truncated = total > shown.len();
    let e = Envelope::ok(format!(
        "{total} match(es) across {} rule(s){}",
        rules.len(),
        if truncated {
            format!("; showing the first {}", shown.len())
        } else {
            String::new()
        }
    ))
    .field("matches", Value::Array(shown))
    .field("rules", Value::Array(rule_rows))
    .field("total", Value::from(total as u64));
    Ok(e.truncated(truncated).build())
}

// ---------------------------------------------------------------------------
// debug info tools
// ---------------------------------------------------------------------------

/// Debug info for an image, or an explanation of why there is none.
fn dwarf_of(file: &ImageFile) -> Result<&occ_dwarf::Dwarf, String> {
    file.dwarf.as_ref().ok_or_else(|| {
        "no DWARF in this image: it was built without -g, or the debug sections were stripped"
            .to_string()
    })
}

/// JSON string or null, for optional metadata.
fn opt_value(v: Option<&str>) -> Value {
    match v {
        Some(s) => Value::from(s),
        None => Value::Null,
    }
}

/// Turn a debug function into the same shape `code.function` uses, so a caller
/// can hop between the two views.
fn dwarf_function_json(f: &occ_dwarf::DwarfFunction, at: Addr) -> Value {
    obj!(
        "handle" => Handle::function(f.low_pc).to_string(),
        "name" => f.name.clone(),
        "linkage_name" => opt_value(f.linkage_name.as_deref()),
        "low" => addr(f.low_pc),
        "high" => addr(f.high_pc),
        "size" => Value::from(f.size()),
        "decl_file" => opt_value(f.decl_file.as_deref()),
        "decl_line" => f.decl_line.map(Value::from).unwrap_or(Value::Null),
        "at" => addr(at),
        "params" => Value::Array(f.params.iter().map(|v| variable_json(v, at)).collect()),
        "locals" => Value::Array(f.locals.iter().map(|v| variable_json(v, at)).collect()),
    )
}

fn variable_json(v: &occ_dwarf::Variable, at: Addr) -> Value {
    let location = v.location_at(at);
    obj!(
        "name" => v.name.clone(),
        "type" => opt_value(v.type_name.as_deref()),
        "location" => opt_value(location),
        "live" => location.is_some(),
        "ranges" => Value::from(v.ranges.len()),
        "decl_line" => v.decl_line.map(Value::from).unwrap_or(Value::Null),
        "param" => v.is_param,
    )
}

fn debug_info(session: &Session, args: &Args) -> Result<Value, String> {
    let file = args.file(session)?;
    let dw = dwarf_of(&file)?;
    let units: Vec<Value> = dw
        .units
        .iter()
        .map(|u| {
            obj!(
                "offset" => Value::from(u.offset),
                "name" => opt_value(u.name.as_deref()),
                "producer" => opt_value(u.producer.as_deref()),
                "language" => u.language.map(Value::from).unwrap_or(Value::Null),
                "comp_dir" => opt_value(u.comp_dir.as_deref()),
                "functions" => Value::from(u.functions.len()),
                "types" => Value::from(u.types.len()),
                "files" => Value::from(u.lines.files.len()),
                "rows" => Value::from(u.lines.rows.len()),
            )
        })
        .collect();
    let files: Vec<Value> = dw.source_files().iter().map(|f| Value::from(f.as_str())).collect();
    let notes: Vec<Value> = dw.notes.iter().map(|n| Value::from(n.as_str())).collect();
    let funcs = dw.function_count();
    Ok(Envelope::ok(format!(
        "{} compilation unit(s), {funcs} function(s) with ranges, {} type(s), {} source file(s)",
        dw.units.len(),
        dw.types().count(),
        dw.source_files().len()
    ))
    .field("units", Value::Array(units))
    .field("source_files", Value::Array(files))
    .field("notes", Value::Array(notes))
    .next(vec![
        "debug.functions to list the compiler's functions".into(),
        "debug.line with an address to get file:line".into(),
    ])
    .build())
}

fn debug_line(session: &Session, args: &Args) -> Result<Value, String> {
    let id = args.image(session)?;
    let file = args.file(session)?;
    let dw = dwarf_of(&file)?;
    let at = args.req_str("at")?;
    let mut rows = Vec::new();
    for part in at.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        // Accept a bare address, a symbol name, or a handle.
        let target = occ_core::addr::parse_addr(part);
        let addr_value = match target {
            Some(a) => a,
            None => {
                let t = handle_target(session, id, part)?;
                match t {
                    Target::Function { entry, .. } => entry,
                    Target::Insn { addr, .. } => addr,
                    Target::Image { .. } => {
                        return Err(format!("{part} names the image, not an address"));
                    }
                    _ => return Err(format!("{part} does not resolve to an address")),
                }
            }
        };
        let loc = dw.line_for(addr_value);
        let func = dw.function_for(addr_value);
        rows.push(obj!(
            "addr" => addr(addr_value),
            "file" => loc.as_ref().map(|(f, _, _)| Value::from(f.as_str())).unwrap_or(Value::Null),
            "line" => loc.as_ref().map(|(_, l, _)| Value::from(*l)).unwrap_or(Value::Null),
            "column" => loc.as_ref().map(|(_, _, c)| Value::from(*c)).unwrap_or(Value::Null),
            "function" => func.map(|f| Value::from(f.name.as_str())).unwrap_or(Value::Null),
            "decl" => func
                .and_then(|f| f.decl_file.clone().map(|file| format!("{file}:{}", f.decl_line.unwrap_or(0))))
                .map(Value::from)
                .unwrap_or(Value::Null),
        ));
    }
    let found = rows
        .iter()
        .filter(|r| r.get("file").is_some_and(|v| !v.is_null()))
        .count();
    Ok(Envelope::ok(format!("{found}/{} address(es) have a source line", rows.len()))
        .field("lines", Value::Array(rows))
        .truncated(false)
        .next(vec!["debug.variables at the same address to see live variables".into()])
        .build())
}

fn debug_functions(session: &Session, args: &Args) -> Result<Value, String> {
    let file = args.file(session)?;
    let dw = dwarf_of(&file)?;
    let filter = args.opt_str("filter").map(|s| s.to_ascii_lowercase());
    let sort = args.one_of("sort", "addr", &["addr", "size", "params", "name"])?;
    let (offset, limit) = args.page(200, 2000)?;
    let mut funcs: Vec<&occ_dwarf::DwarfFunction> = dw
        .units
        .iter()
        .flat_map(|u| u.functions.iter())
        .filter(|f| f.size() > 0)
        .filter(|f| {
            filter
                .as_ref()
                .map(|x| f.name.to_ascii_lowercase().contains(x))
                .unwrap_or(true)
        })
        .collect();
    match sort {
        "size" => funcs.sort_by_key(|f| std::cmp::Reverse(f.size())),
        "params" => funcs.sort_by_key(|f| std::cmp::Reverse(f.params.len())),
        "name" => funcs.sort_by(|a, b| a.name.cmp(&b.name)),
        _ => funcs.sort_by_key(|f| f.low_pc.0),
    }
    let total = funcs.len();
    let rows: Vec<Value> = funcs
        .iter()
        .map(|f| {
            obj!(
                "handle" => Handle::function(f.low_pc).to_string(),
                "name" => f.name.clone(),
                "low" => addr(f.low_pc),
                "size" => Value::from(f.size()),
                "decl_file" => opt_value(f.decl_file.as_deref()),
                "decl_line" => f.decl_line.map(Value::from).unwrap_or(Value::Null),
                "params" => Value::from(f.params.len()),
                "locals" => Value::from(f.locals.len()),
            )
        })
        .collect();
    let (page, truncated) = paginate(&rows, offset, limit);
    Ok(Envelope::ok(format!("{total} function(s) with debug info; sorted by {sort}"))
        .field("functions", Value::Array(page))
        .field("total", Value::from(total))
        .truncated(truncated)
        .next(vec!["debug.function with one of these handles".into()])
        .build())
}

fn debug_function(session: &Session, args: &Args) -> Result<Value, String> {
    let id = args.image(session)?;
    let file = args.file(session)?;
    let dw = dwarf_of(&file)?;
    let target = args.req_str("target")?;
    let at = match args.opt_str("at").or(Some(target)).and_then(occ_core::addr::parse_addr) {
        Some(a) => a,
        None => {
            let t = handle_target(session, id, target)?;
            match t {
                Target::Function { entry, .. } => entry,
                Target::Insn { addr, .. } => addr,
                _ => return Err(format!("{target} is not a function")),
            }
        }
    };
    let f = dw
        .function_by_name(target)
        .or_else(|| dw.function_for(at))
        .ok_or_else(|| format!("no debug entry for {target}"))?;
    let at = if at == f.low_pc { at } else { at.min(f.high_pc) };
    let line = dw.line_for(at).map(|(file, line, col)| format!("{file}:{line}:{col}"));
    let mut env = Envelope::ok(format!(
        "{}: {} param(s), {} local(s) at {}",
        f.name,
        f.params.len(),
        f.locals.len(),
        at
    ))
    .field("function", dwarf_function_json(f, at))
    .truncated(false);
    if let Some(l) = line {
        env = env.field("line", Value::from(l));
    }
    Ok(env
        .next(vec![
            "debug.line for the addresses inside this function".into(),
            "code.function to see the recovered body instead of the source view".into(),
        ])
        .build())
}

fn debug_types(session: &Session, args: &Args) -> Result<Value, String> {
    let file = args.file(session)?;
    let dw = dwarf_of(&file)?;
    let filter = args.opt_str("filter").map(|s| s.to_ascii_lowercase());
    let kind = args.opt_str("kind").map(|s| s.to_ascii_lowercase());
    let include_members = args.bool_or("include_members", true)?;
    let (offset, limit) = args.page(200, 2000)?;
    let mut types: Vec<Value> = dw
        .types()
        .filter(|t| {
            filter
                .as_ref()
                .map(|x| t.name.to_ascii_lowercase().contains(x))
                .unwrap_or(true)
        })
        .filter(|t| kind.as_ref().map(|k| t.kind == *k).unwrap_or(true))
        .map(|t| {
            let mut m = occ_json::Map::new();
            m.insert("name", t.name.clone());
            m.insert("kind", t.kind.clone());
            if let Some(n) = t.byte_size {
                m.insert("bytes", Value::from(n));
            }
            if let Some(target) = &t.target {
                m.insert("target", target.clone());
            }
            if include_members && !t.members.is_empty() {
                let members: Vec<Value> = t
                    .members
                    .iter()
                    .map(|mem| {
                        obj!(
                            "name" => mem.name.clone(),
                            "type" => opt_value(mem.type_name.as_deref()),
                            "offset" => Value::from(mem.offset),
                            "size" => mem.size.map(Value::from).unwrap_or(Value::Null),
                        )
                    })
                    .collect();
                m.insert("members", Value::Array(members));
            }
            Value::Object(m)
        })
        .collect();
    types.sort_by(|a, b| {
        a.get("name")
            .and_then(|v| v.as_str())
            .cmp(&b.get("name").and_then(|v| v.as_str()))
    });
    let total = types.len();
    let (page, truncated) = paginate(&types, offset, limit);
    Ok(Envelope::ok(format!("{total} type(s)"))
        .field("types", Value::Array(page))
        .field("total", Value::from(total))
        .truncated(truncated)
        .next(vec!["debug.function to see which types its variables use".into()])
        .build())
}

fn debug_variables(session: &Session, args: &Args) -> Result<Value, String> {
    let id = args.image(session)?;
    let file = args.file(session)?;
    let dw = dwarf_of(&file)?;
    let target = args.req_str("at")?;
    let at = match occ_core::addr::parse_addr(target) {
        Some(a) => a,
        None => {
            let t = handle_target(session, id, target)?;
            match t {
                Target::Function { entry, .. } => entry,
                Target::Insn { addr, .. } => addr,
                _ => return Err(format!("{target} does not resolve to an address")),
            }
        }
    };
    let f = dw
        .function_for(at)
        .ok_or_else(|| format!("no debug function covers {at}"))?;
    let mut vars: Vec<Value> = f
        .params
        .iter()
        .chain(f.locals.iter())
        .map(|v| variable_json(v, at))
        .collect();
    vars.sort_by_key(|v| std::cmp::Reverse(v.get("live").and_then(|x| x.as_bool()).unwrap_or(false)));
    let live = vars
        .iter()
        .filter(|v| v.get("live").and_then(|x| x.as_bool()).unwrap_or(false))
        .count();
    let line = dw.line_for(at).map(|(file, line, col)| format!("{file}:{line}:{col}"));
    let mut env = Envelope::ok(format!(
        "{} variable(s), {live} with a location at {} inside {}",
        vars.len(),
        at,
        f.name
    ))
    .field("function", f.name.clone())
    .field("variables", Value::Array(vars))
    .truncated(false);
    if let Some(l) = line {
        env = env.field("line", Value::from(l));
    }
    Ok(env
        .next(vec![
            "debug.function for the whole signature".into(),
            "code.disasm at the same address for the instruction that uses them".into(),
        ])
        .build())
}


// ---------------------------------------------------------------------------
// dynamic tracing
// ---------------------------------------------------------------------------

/// Aggregated syscall rows: a trace of a real program has thousands, and the
/// interesting question is "which syscalls, how often, with which paths".
fn syscall_rows(trace: &occ_dynamic::Trace, limit: usize) -> (Vec<Value>, usize) {
    use std::collections::BTreeMap;
    struct Row {
        count: u64,
        failed: u64,
        paths: Vec<String>,
        example: [u64; 6],
        ret: i64,
    }
    let mut rows: BTreeMap<(u64, String), Row> = BTreeMap::new();
    for s in &trace.syscalls {
        let row = rows.entry((s.nr, s.name.clone())).or_insert(Row {
            count: 0,
            failed: 0,
            paths: Vec::new(),
            example: s.args,
            ret: s.ret,
        });
        row.count += 1;
        if s.ret < 0 {
            row.failed += 1;
        }
        for p in &s.paths {
            if row.paths.len() < 4 && !row.paths.contains(p) {
                row.paths.push(p.clone());
            }
        }
        row.example = s.args;
        row.ret = s.ret;
    }
    let total = rows.len();
    let out: Vec<Value> = rows
        .into_iter()
        .take(limit)
        .map(|((nr, name), row)| {
            obj!(
                "name" => Value::from(name.as_str()),
                "nr" => Value::from(nr),
                "count" => Value::from(row.count),
                "failed" => Value::from(row.failed),
                "last_args" => Value::Array(row.example.iter().map(|a| Value::from(*a)).collect()),
                "last_ret" => Value::from(row.ret),
                "paths" => Value::Array(row.paths.iter().map(|p| Value::from(p.as_str())).collect()),
            )
        })
        .collect();
    (out, total)
}

/// The image, analysis and trace behind a `trace:` handle.
fn trace_record(
    session: &Session,
    args: &Args,
) -> Result<(u32, Arc<occ_session::TraceRecord>, Arc<ImageFile>), String> {
    let id = match args.opt_str("trace") {
        Some(text) => {
            let handle = occ_core::Handle::lax(text, occ_core::HandleKind::Trace)
                .map_err(|e| e.to_string())?;
            if handle.kind() != occ_core::HandleKind::Trace {
                return Err(format!(
                    "{text:?} is not a trace handle; expected trace:<id>"
                ));
            }
            handle
                .value()
                .parse::<u32>()
                .map_err(|_| format!("{text:?} is not a trace handle"))?
        }
        None => *session
            .trace_ids()
            .last()
            .ok_or("no run has been recorded in this session; call trace.run first")?,
    };
    let record = session
        .trace(id)
        .ok_or_else(|| format!("trace:{id} is not in this session"))?;
    let file = session
        .image(record.image)
        .ok_or_else(|| format!("image {} of trace:{id} is no longer open", record.image))?;
    Ok((id, record, file))
}

fn coverage_json(corr: &occ_dynamic::Correlated, limit: usize) -> Value {
    obj!(
        "functions" => Value::Array(
            corr.functions
                .iter()
                .filter(|f| f.hits > 0)
                .take(limit)
                .map(|f| obj!(
                    "fn" => Value::from(format!("fn:{}", f.entry)),
                    "name" => Value::from(f.name.as_str()),
                    "insns" => Value::from(u64::from(f.insns)),
                    "total_insns" => Value::from(u64::from(f.total)),
                    "blocks" => Value::from(u64::from(f.blocks)),
                    "total_blocks" => Value::from(u64::from(f.blocks_total)),
                    "hits" => Value::from(f.hits),
                ))
                .collect()
        ),
        "executed_functions" => Value::from(corr.functions.iter().filter(|f| f.hits > 0).count() as u64),
        "missed_functions" => Value::from(corr.missed.len() as u64),
        "missed" => Value::Array(
            corr.missed
                .iter()
                .take(limit)
                .map(|(a, n)| obj!("fn" => Value::from(format!("fn:{a}")), "name" => Value::from(n.as_str())))
                .collect()
        ),
        "orphan_instructions" => Value::from(corr.orphans.len() as u64),
        "orphans" => Value::Array(
            corr.orphans
                .iter()
                .take(limit)
                .map(|o| obj!(
                    "addr" => addr(o.addr),
                    "section" => Value::from(o.section as u64),
                ))
                .collect()
        ),
        "distinct_instructions_in_image" => Value::from(corr.executed_in_image as u64),
        "steps_outside_image" => Value::from(corr.external),
    )
}

fn call_rows(corr: &occ_dynamic::Correlated, limit: usize) -> Vec<Value> {
    corr.calls
        .iter()
        .filter(|c| c.indirect || c.targets.len() > 1)
        .take(limit)
        .map(|c| {
            obj!(
                "site" => addr(c.site),
                "in_image" => Value::from(c.in_image),
                "fn" => c.from.as_deref().map_or(Value::Null, Value::from),
                "indirect" => Value::from(c.indirect),
                "targets" => Value::Array(
                    c.targets
                        .iter()
                        .take(8)
                        .map(|t| obj!(
                            "target" => addr(t.addr),
                            "in_image" => Value::from(t.in_image),
                            "count" => Value::from(u64::from(t.count)),
                        ))
                        .collect()
                ),
                "target_count" => Value::from(c.targets.len() as u64),
            )
        })
        .collect()
}

fn trace_run(session: &Session, args: &Args) -> Result<Value, String> {
    let id = args.image(session)?;
    let file = args.file(session)?;
    let path = std::path::Path::new(&file.image.path);
    if !path.is_file() {
        return Err(format!(
            "image {id} was loaded from {:?}, which is not a file on disk; a trace runs the program itself",
            file.image.path
        ));
    }
    let mut argv = Vec::new();
    if let Some(value) = args.raw("argv") {
        let list = value.as_array().ok_or("`argv` must be an array")?;
        for entry in list {
            argv.push(
                entry
                    .as_str()
                    .ok_or("`argv` entries must be strings")?
                    .to_string(),
            );
        }
    }
    // `argv` is the whole vector, so its first element is the program name the
    // tracee sees. Dropping it would run the file under its path and shift
    // every argument by one.
    let argv0 = if argv.is_empty() {
        None
    } else {
        Some(argv.remove(0))
    };
    let cfg = occ_dynamic::RunConfig {
        argv0,
        args: argv,
        max_steps: args.u64_in("steps", 200_000, 1, 2_000_000)?,
        max_millis: args.u64_in("millis", 5_000, 1, 60_000)?,
        from_entry: args.bool_or("from_entry", true)?,
        trace_syscalls: args.bool_or("syscalls", true)?,
        ..occ_dynamic::RunConfig::default()
    };
    let (_, limit) = args.page(16, 128)?;

    let trace = occ_dynamic::run(path, &file.image, &cfg)
        .map_err(|e| format!("could not trace {}: {e}", file.image.path))?;
    let corr = occ_dynamic::correlate(&trace, &file.image, &file.analysis);
    let trace_id = session
        .add_trace(id, trace.clone())
        .map_err(|e| e.to_string())?;

    let (syscalls, syscall_kinds) = syscall_rows(&trace, limit);
    let exited = matches!(trace.end, occ_dynamic::End::Exited(_));
    let summary = format!(
        "ran {} for {} step(s) in {} ms: {}; {} of {} recovered function(s) executed, {} never did, {} executed address(es) belong to no function, {} indirect call site(s), {} syscall kind(s)",
        file.image.path,
        trace.steps,
        trace.millis,
        trace.end.label(),
        corr.functions.iter().filter(|f| f.hits > 0).count(),
        file.analysis.functions.iter().filter(|f| !f.insns.is_empty()).count(),
        corr.missed.len(),
        corr.orphans.len(),
        corr.calls.iter().filter(|c| c.indirect).count(),
        syscall_kinds
    );

    let mut evidence = Vec::new();
    for c in corr.calls.iter().filter(|c| c.indirect).take(4) {
        evidence.push(Evidence::support(
            EvidenceKind::DynamicTrace,
            EvidenceSource::Tracer,
            format!(
                "{} reached {} target(s) at run time, first {}",
                c.site,
                c.targets.len(),
                c.targets.first().map(|t| t.addr.to_string()).unwrap_or_default()
            ),
            0.9,
        ));
    }
    if !corr.orphans.is_empty() {
        evidence.push(Evidence::support(
            EvidenceKind::DynamicTrace,
            EvidenceSource::Tracer,
            format!(
                "{} executed instruction(s) fall in code sections but in no recovered function",
                corr.orphans.len()
            ),
            0.85,
        ));
    }
    evidence.push(Evidence::support(
        EvidenceKind::DynamicTrace,
        EvidenceSource::Tracer,
        format!("{} instruction step(s) observed under ptrace", trace.steps),
        0.95,
    ));

    let mut notes: Vec<String> = trace.notes.clone();
    if trace.end.truncated() {
        notes.push(format!(
            "the run was cut short ({}), so absence of evidence below is not evidence of absence",
            trace.end.label()
        ));
    }
    if !exited {
        notes.push("the program did not exit normally".into());
    }

    Ok(Envelope::ok(summary)
        .handle(Handle::trace(trace_id).to_string())
        .field(
            "run",
            obj!(
                "program" => Value::from(trace.path.as_str()),
                "argv" => Value::Array(trace.argv.iter().map(|a| Value::from(a.as_str())).collect()),
                "steps" => Value::from(trace.steps),
                "millis" => Value::from(trace.millis),
                "end" => Value::from(trace.end.label()),
                "truncated" => Value::from(trace.end.truncated()),
                "threads" => Value::from(trace.threads.len() as u64),
                "load_base" => Value::from(format!("{:#x}", trace.base)),
            ),
        )
        .field("coverage", coverage_json(&corr, limit))
        .field(
            "indirect_calls",
            Value::Array(call_rows(&corr, limit)),
        )
        .field(
            "syscalls",
            obj!(
                "kinds" => Value::from(syscall_kinds as u64),
                "rows" => Value::Array(syscalls),
            ),
        )
        .field(
            "signals",
            Value::Array(
                trace
                    .signals
                    .iter()
                    .take(limit)
                    .map(|s| obj!(
                        "signal" => Value::from(s.signal as u64),
                        "at" => addr(s.at),
                        "fault" => s.fault.map_or(Value::Null, |f| Value::from(format!("{f:#x}"))),
                    ))
                    .collect(),
            ),
        )
        .field("notes", Value::Array(notes.iter().map(|n| Value::from(n.as_str())).collect()))
        .evidence(evidence)
        .confidence(if trace.end.truncated() { 0.8 } else { 0.95 })
        .truncated(trace.end.truncated())
        .next(vec![
            "trace.get with a fn: handle for block-level hit counts".into(),
            "code.xrefs direction=to on an orphan address to see what refers to it".into(),
            "debug.line for the source line behind an address the run covered".into(),
        ])
        .build())
}

fn trace_get(session: &Session, args: &Args) -> Result<Value, String> {
    let (trace_id, record, file) = trace_record(session, args)?;
    let trace = &record.trace;
    let corr = occ_dynamic::correlate(trace, &file.image, &file.analysis);
    let (_, limit) = args.page(16, 128)?;

    let Some(target) = args.opt_str("target") else {
        let (syscalls, kinds) = syscall_rows(trace, limit);
        let summary = format!(
            "trace:{trace_id}: {} step(s), {}, {} of {} function(s) executed, {} orphan(s), {} syscall kind(s)",
            trace.steps,
            trace.end.label(),
            corr.functions.iter().filter(|f| f.hits > 0).count(),
            file.analysis.functions.iter().filter(|f| !f.insns.is_empty()).count(),
            corr.orphans.len(),
            kinds
        );
        return Ok(Envelope::ok(summary)
            .handle(Handle::trace(trace_id).to_string())
            .field("coverage", coverage_json(&corr, limit))
            .field("indirect_calls", Value::Array(call_rows(&corr, limit)))
            .field(
                "syscalls",
                obj!("kinds" => Value::from(kinds as u64), "rows" => Value::Array(syscalls)),
            )
            .next(vec![
                "trace.get target=fn:<addr> for block-level detail".into(),
                "trace.run again with different argv to see another path".into(),
            ])
            .build());
    };

    let t = handle_target(session, record.image, target)?;
    let f = function_of(&file, &t)
        .ok_or_else(|| format!("no function contains {target}; pass a fn: handle or an address"))?;
    let blocks = occ_dynamic::function_blocks(trace, f);
    let hit = blocks.iter().filter(|b| b.hits > 0).count();
    let rows: Vec<Value> = blocks
        .iter()
        .take(limit)
        .map(|b| {
            obj!(
                "bb" => Value::from(format!("bb:{}", b.start)),
                "start" => addr(b.start),
                "end" => addr(b.end),
                "insns" => Value::from(u64::from(b.insns)),
                "hits" => Value::from(b.hits),
                "entered" => Value::from(b.hits > 0),
            )
        })
        .collect();
    let complete = !trace.end.truncated();
    let summary = format!(
        "{}: {hit} of {} block(s) entered during trace:{trace_id}{}",
        f.display_name(),
        blocks.len(),
        if complete { "" } else { " (run was truncated)" }
    );
    Ok(Envelope::ok(summary)
        .handle(Handle::function(f.entry).to_string())
        .field(
            "function",
            obj!(
                "entry" => addr(f.entry),
                "name" => Value::from(f.display_name()),
                "blocks_entered" => Value::from(hit as u64),
                "blocks_total" => Value::from(blocks.len() as u64),
                "instructions_entered" => Value::from(
                    blocks.iter().filter(|b| b.hits > 0).map(|b| u64::from(b.insns)).sum::<u64>()
                ),
            ),
        )
        .field("blocks", Value::Array(rows))
        .field("trace", Value::from(format!("trace:{trace_id}")))
        .caveats(if complete {
            vec!["a block with zero hits was not entered during this run".into()]
        } else {
            vec![
                "the run was cut short, so a block with zero hits may still be reachable".into(),
            ]
        })
        .next(vec![
            "trace.get without a target for the whole-run summary".into(),
            "code.blocks for the same function's edges".into(),
        ])
        .build())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_has_a_parseable_schema() {
        for t in all() {
            let v = occ_json::parse(t.schema)
                .unwrap_or_else(|e| panic!("{} schema does not parse: {e}", t.name));
            assert_eq!(
                v.get("type").and_then(|x| x.as_str()),
                Some("object"),
                "{} schema is not an object",
                t.name
            );
        }
    }

    #[test]
    fn tool_names_are_unique() {
        let mut names: Vec<&str> = all().iter().map(|t| t.name).collect();
        names.sort_unstable();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n);
    }

    #[test]
    fn list_payload_matches_the_registry() {
        let v = list_json();
        assert_eq!(v.as_array().unwrap().len(), all().len());
    }
}
