//! Session state.
//!
//! A session owns the images a caller has opened, the handles that address
//! entities inside them, and the hypotheses under evaluation. It is designed
//! for the calling pattern an MCP client actually has: many short requests,
//! each referring to something established earlier by handle rather than by
//! repeating a kilobyte of payload.
//!
//! Concurrency model: images are behind an `RwLock` and shared as `Arc`, so a
//! query clones a pointer out and then reads immutable data without holding the
//! lock. The hypothesis table is behind the same lock and is the only mutable
//! state; its sequence numbers come from an atomic counter so ordering does not
//! depend on lock acquisition order.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use occ_analyze::analysis::{Analysis, AnalysisOptions};
use occ_analyze::view::View;
use occ_core::{
    Addr, Evidence, Format, Function, Handle, HandleKind, Hypothesis, Image, ImageId, Result,
    SymbolKind, Verdict, CoreError,
};

/// One loaded image and everything derived from it.
pub struct ImageFile {
    /// Parsed container metadata.
    pub image: Image,
    /// Raw file bytes, kept because every later read goes through them.
    pub data: Vec<u8>,
    /// Recovery and indexing results.
    pub analysis: Analysis,
    /// Debug info, when the file has any.
    pub dwarf: Option<occ_dwarf::Dwarf>,
    /// When the image was opened, as a monotonic session sequence number.
    pub opened_seq: u64,
}

/// One recorded run, kept so later turns can query it without tracing again.
pub struct TraceRecord {
    /// Image the run used.
    pub image: u32,
    /// What the run observed.
    pub trace: occ_dynamic::Trace,
    /// When it was recorded.
    pub seq: u64,
}

struct Inner {
    images: BTreeMap<u32, Arc<ImageFile>>,
    next_id: u32,
    traces: BTreeMap<u32, Arc<TraceRecord>>,
    next_trace: u32,
    hypotheses: BTreeMap<u64, Hypothesis>,
    next_hyp: u64,
    notes: Vec<String>,
}

/// A session.
pub struct Session {
    inner: RwLock<Inner>,
    seq: AtomicU64,
}

/// What a handle resolved to.
#[derive(Clone, Debug)]
pub enum Target {
    /// The image itself.
    Image {
        /// Image id.
        id: u32,
    },
    /// A section.
    Section {
        /// Owning image.
        image: u32,
        /// Section name.
        name: String,
        /// Virtual address.
        addr: Addr,
        /// Size in bytes.
        size: u64,
    },
    /// A symbol.
    Symbol {
        /// Owning image.
        image: u32,
        /// Symbol name.
        name: String,
        /// Address.
        addr: Addr,
        /// Size.
        size: u64,
    },
    /// A recovered function.
    Function {
        /// Owning image.
        image: u32,
        /// Entry address.
        entry: Addr,
        /// Display name.
        name: String,
        /// Size in bytes.
        size: u64,
    },
    /// A basic block.
    Block {
        /// Owning image.
        image: u32,
        /// Containing function entry.
        func: Addr,
        /// Block start.
        start: Addr,
        /// Block end.
        end: Addr,
    },
    /// A single instruction.
    Insn {
        /// Owning image.
        image: u32,
        /// Address.
        addr: Addr,
        /// Rendered text.
        text: String,
    },
    /// A string literal.
    Str {
        /// Owning image.
        image: u32,
        /// Address.
        addr: Addr,
        /// Text.
        text: String,
    },
    /// An import.
    Import {
        /// Owning image.
        image: u32,
        /// Name.
        name: String,
        /// GOT/IAT slot.
        slot: Option<Addr>,
    },
    /// An export.
    Export {
        /// Owning image.
        image: u32,
        /// Name.
        name: String,
        /// Address.
        addr: Addr,
    },
    /// A relocation.
    Reloc {
        /// Owning image.
        image: u32,
        /// Index.
        index: usize,
    },
    /// A hypothesis.
    Hypothesis {
        /// Hypothesis id.
        id: u64,
    },
    /// A cross-reference edge.
    Xref {
        /// Owning image.
        image: u32,
        /// Index into the xref table.
        index: usize,
    },
    /// A recorded run.
    Trace {
        /// Session id of the trace.
        id: u32,
        /// Image it ran.
        image: u32,
    },
    /// A content-addressed blob handle, recorded but not resolvable to bytes.
    Blob {
        /// Hash prefix.
        hash: String,
    },
}

impl Target {
    /// Canonical handle for the target.
    pub fn handle(&self) -> Handle {
        match self {
            Target::Image { id } => Handle::image(*id),
            Target::Section { name, .. } => Handle::section(name),
            Target::Symbol { name, .. } => Handle::symbol(name),
            Target::Function { entry, .. } => Handle::function(*entry),
            Target::Block { start, .. } => Handle::block(*start),
            Target::Insn { addr, .. } => Handle::insn(*addr),
            Target::Str { addr, .. } => Handle::string(*addr),
            Target::Import { name, .. } => Handle::import(name),
            Target::Export { name, .. } => Handle::export(name),
            Target::Reloc { index, .. } => Handle::reloc(*index),
            Target::Hypothesis { id } => Handle::hypothesis(*id),
            Target::Xref { index, .. } => Handle::xref(*index as u64),
            Target::Trace { id, .. } => Handle::trace(*id),
            Target::Blob { hash } => Handle::blob(hash),
        }
    }

    /// Image the target lives in, when it lives in one.
    pub const fn image(&self) -> Option<u32> {
        match self {
            Target::Image { id } => Some(*id),
            Target::Section { image, .. }
            | Target::Symbol { image, .. }
            | Target::Function { image, .. }
            | Target::Block { image, .. }
            | Target::Insn { image, .. }
            | Target::Str { image, .. }
            | Target::Import { image, .. }
            | Target::Export { image, .. }
            | Target::Reloc { image, .. }
            | Target::Xref { image, .. }
            | Target::Trace { image, .. } => Some(*image),
            _ => None,
        }
    }
}

impl Default for Session {
    fn default() -> Self {
        Session::new()
    }
}

impl Session {
    /// An empty session.
    pub fn new() -> Session {
        Session {
            inner: RwLock::new(Inner {
                images: BTreeMap::new(),
                next_id: 0,
                traces: BTreeMap::new(),
                next_trace: 0,
                hypotheses: BTreeMap::new(),
                next_hyp: 1,
                notes: Vec::new(),
            }),
            seq: AtomicU64::new(1),
        }
    }

    /// Next monotonic sequence number.
    pub fn tick(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::Relaxed)
    }

    /// Open a file by path.
    pub fn open_path(&self, path: &str) -> Result<Vec<u32>> {
        let data = std::fs::read(path)
            .map_err(|e| CoreError::io(Path::new(path).display().to_string(), e))?;
        self.open_bytes(data, path)
    }

    /// Open an image from bytes already in memory.
    ///
    /// An `ar` archive is a container of containers, and it opens as one image
    /// per member, named `libfoo.a(member.o)`. That keeps the archive from
    /// becoming a second kind of handle: every tool takes an image id, and a
    /// member has one. The members that do not parse as an object — a linker
    /// index, a resource blob — are skipped, and the open fails only when
    /// nothing in the archive could be read.
    pub fn open_bytes(&self, data: Vec<u8>, path: &str) -> Result<Vec<u32>> {
        let mut inner = self.inner.write().map_err(|_| poisoned())?;
        if occ_bin::probe_format(&data) == Format::Archive {
            let members = occ_bin::ar::members(&data)?;
            let parts: Vec<(String, Vec<u8>)> = members
                .into_iter()
                .map(|m| (m.name.clone(), m.data.to_vec()))
                .collect();
            let mut ids = Vec::with_capacity(parts.len());
            let mut first_error: Option<CoreError> = None;
            for (name, bytes) in parts {
                let id = inner.next_id;
                let label = format!("{path}({name})");
                match occ_bin::load(&bytes, &label, ImageId(id)) {
                    Ok(image) => {
                        let (analysis, dwarf) = {
                            let view = View::new(&image, &bytes);
                            (
                                Analysis::build(&view, &AnalysisOptions::default()),
                                occ_dwarf::Dwarf::load(&view),
                            )
                        };
                        let opened_seq = self.seq.fetch_add(1, Ordering::Relaxed);
                        inner.next_id += 1;
                        inner.images.insert(
                            id,
                            Arc::new(ImageFile {
                                image,
                                data: bytes,
                                analysis,
                                dwarf,
                                opened_seq,
                            }),
                        );
                        ids.push(id);
                    }
                    Err(e) => {
                        if first_error.is_none() {
                            first_error = Some(e);
                        }
                    }
                }
            }
            if ids.is_empty() {
                return Err(first_error.unwrap_or_else(|| {
                    CoreError::UnsupportedFormat(format!(
                        "{path}: the archive holds no member this build can parse"
                    ))
                }));
            }
            return Ok(ids);
        }
        let id = inner.next_id;
        // Parse before taking an id so a failed open does not consume one and
        // leave a hole that a persisted handle would later point into.
        let image = occ_bin::load(&data, path, ImageId(id))?;
        let (analysis, dwarf) = {
            let view = View::new(&image, &data);
            (
                Analysis::build(&view, &AnalysisOptions::default()),
                occ_dwarf::Dwarf::load(&view),
            )
        };
        let opened_seq = self.seq.fetch_add(1, Ordering::Relaxed);
        inner.next_id += 1;
        inner.images.insert(
            id,
            Arc::new(ImageFile {
                image,
                data,
                analysis,
                dwarf,
                opened_seq,
            }),
        );
        Ok(vec![id])
    }

    /// Store a trace and return its handle.
    ///
    /// Traces accumulate for the life of the session: a follow-up question
    /// ("did that run ever enter this block") would otherwise require running
    /// the program again, and a second run is not the same run.
    pub fn add_trace(&self, image: u32, trace: occ_dynamic::Trace) -> Result<u32> {
        let mut inner = self.inner.write().map_err(|_| poisoned())?;
        let id = inner.next_trace;
        inner.next_trace += 1;
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        inner.traces.insert(
            id,
            Arc::new(TraceRecord {
                image,
                trace,
                seq,
            }),
        );
        Ok(id)
    }

    /// Fetch a recorded trace.
    pub fn trace(&self, id: u32) -> Option<Arc<TraceRecord>> {
        let inner = self.inner.read().ok()?;
        inner.traces.get(&id).cloned()
    }

    /// Ids of every recorded trace, newest last.
    pub fn trace_ids(&self) -> Vec<u32> {
        self.inner
            .read()
            .map(|i| i.traces.keys().copied().collect())
            .unwrap_or_default()
    }

    /// Fetch a loaded image.
    pub fn image(&self, id: u32) -> Option<Arc<ImageFile>> {
        let inner = self.inner.read().ok()?;
        inner.images.get(&id).cloned()
    }

    /// Ids of all loaded images.
    pub fn image_ids(&self) -> Vec<u32> {
        self.inner
            .read()
            .map(|i| i.images.keys().copied().collect())
            .unwrap_or_default()
    }

    /// The only loaded image, when exactly one is open.
    ///
    /// Handle strings do not carry an image id, so a session with one image
    /// resolves them implicitly. With several open the caller must say which;
    /// guessing would silently analyse the wrong binary.
    pub fn only_image(&self) -> Option<u32> {
        let ids = self.image_ids();
        if ids.len() == 1 { Some(ids[0]) } else { None }
    }

    /// Resolve a handle string inside `image`.
    pub fn resolve(&self, image: u32, input: &str, default_kind: HandleKind) -> Result<Target> {
        let handle = Handle::lax(input, default_kind)?;
        let file = self.image(image).ok_or(CoreError::NoSuchImage { id: image })?;
        self.resolve_in(&file, image, &handle)
    }

    /// Resolve a parsed handle against an already-fetched image.
    pub fn resolve_in(&self, file: &ImageFile, image: u32, handle: &Handle) -> Result<Target> {
        let value = handle.value().to_string();
        match handle.kind() {
            HandleKind::Image => Ok(Target::Image { id: image }),
            HandleKind::Section => {
                let s = file
                    .image
                    .section_by_name(&value)
                    .or_else(|| {
                        value
                            .parse::<usize>()
                            .ok()
                            .and_then(|i| file.image.sections.iter().find(|s| s.index == i))
                    })
                    .ok_or_else(|| CoreError::NotFound {
                        what: format!("section {value}"),
                    })?;
                Ok(Target::Section {
                    image,
                    name: s.name.clone(),
                    addr: s.addr,
                    size: s.size,
                })
            }
            HandleKind::Symbol => {
                let s = file
                    .image
                    .symbols
                    .iter()
                    .find(|s| s.name == value)
                    .ok_or_else(|| CoreError::NotFound {
                        what: format!("symbol {value}"),
                    })?;
                Ok(Target::Symbol {
                    image,
                    name: s.name.clone(),
                    addr: s.addr,
                    size: s.size,
                })
            }
            HandleKind::Function => {
                let addr = parse_or_err(&value)?;
                let f = file
                    .analysis
                    .function_at(addr)
                    .or_else(|| file.analysis.function_covering(addr))
                    .ok_or_else(|| CoreError::NotFound {
                        what: format!("function at {addr}"),
                    })?;
                Ok(Target::Function {
                    image,
                    entry: f.entry,
                    name: f.display_name(),
                    size: f.size,
                })
            }
            HandleKind::Block => {
                let addr = parse_or_err(&value)?;
                let f = file
                    .analysis
                    .function_covering(addr)
                    .ok_or_else(|| CoreError::NotFound {
                        what: format!("block at {addr}"),
                    })?;
                let b = f.block_at(addr).ok_or_else(|| CoreError::NotFound {
                    what: format!("block at {addr}"),
                })?;
                Ok(Target::Block {
                    image,
                    func: f.entry,
                    start: b.start,
                    end: b.end,
                })
            }
            HandleKind::Insn => {
                let addr = parse_or_err(&value)?;
                let i = file.analysis.insn_at(addr).ok_or_else(|| CoreError::NotFound {
                    what: format!("instruction at {addr}"),
                })?;
                Ok(Target::Insn {
                    image,
                    addr: i.addr,
                    text: i.text.clone(),
                })
            }
            HandleKind::Str => {
                let addr = parse_or_err(&value)?;
                let s = file.analysis.string_at(addr).ok_or_else(|| CoreError::NotFound {
                    what: format!("string at {addr}"),
                })?;
                Ok(Target::Str {
                    image,
                    addr: s.addr,
                    text: s.text.clone(),
                })
            }
            HandleKind::Import => {
                let i = file
                    .image
                    .import_by_name(&value)
                    .ok_or_else(|| CoreError::NotFound {
                        what: format!("import {value}"),
                    })?;
                Ok(Target::Import {
                    image,
                    name: i.name.clone(),
                    slot: i.slot,
                })
            }
            HandleKind::Export => {
                let e = file
                    .image
                    .exports
                    .iter()
                    .find(|e| e.name == value)
                    .ok_or_else(|| CoreError::NotFound {
                        what: format!("export {value}"),
                    })?;
                Ok(Target::Export {
                    image,
                    name: e.name.clone(),
                    addr: e.addr,
                })
            }
            HandleKind::Reloc => {
                let index: usize = value.parse().map_err(|_| CoreError::NotFound {
                    what: format!("relocation {value}"),
                })?;
                if index >= file.image.relocations.len() {
                    return Err(CoreError::NotFound {
                        what: format!("relocation {index}"),
                    });
                }
                Ok(Target::Reloc { image, index })
            }
            HandleKind::Hypothesis => {
                let id: u64 = value.parse().map_err(|_| CoreError::NotFound {
                    what: format!("hypothesis {value}"),
                })?;
                if !self.hypothesis(id).is_some() {
                    return Err(CoreError::NotFound {
                        what: format!("hypothesis {id}"),
                    });
                }
                Ok(Target::Hypothesis { id })
            }
            HandleKind::Xref => {
                let index: usize = value.parse().map_err(|_| CoreError::NotFound {
                    what: format!("xref {value}"),
                })?;
                if index >= file.analysis.xrefs.len() {
                    return Err(CoreError::NotFound {
                        what: format!("xref {index}"),
                    });
                }
                Ok(Target::Xref { image, index })
            }
            HandleKind::Blob => Ok(Target::Blob { hash: value }),
            HandleKind::Trace => {
                let id: u32 = value.parse().map_err(|_| CoreError::NotFound {
                    what: format!("trace {value}"),
                })?;
                let record = self.trace(id).ok_or_else(|| CoreError::NotFound {
                    what: format!("trace {id}"),
                })?;
                Ok(Target::Trace {
                    id,
                    image: record.image,
                })
            }
        }
    }

    /// Resolve something that is either a symbol name or an address.
    ///
    /// Callers pass whatever the model wrote. A symbol name wins when it
    /// exists, because that is the more specific intent; otherwise the text is
    /// parsed as an address.
    pub fn resolve_name_or_addr(&self, image: u32, input: &str) -> Result<Addr> {
        let file = self.image(image).ok_or(CoreError::NoSuchImage { id: image })?;
        if let Some(s) = file.image.symbols.iter().find(|s| s.name == input) {
            return Ok(s.addr);
        }
        if let Some(f) = file.analysis.function_by_name(input) {
            return Ok(f.entry);
        }
        occ_core::addr::parse_addr(input).ok_or_else(|| CoreError::NotFound {
            what: format!("name or address {input}"),
        })
    }

    /// Create a hypothesis about `handle`.
    pub fn hypothesise(&self, handle: Handle, statement: &str) -> Result<u64> {
        let mut inner = self.inner.write().map_err(|_| poisoned())?;
        let id = inner.next_hyp;
        inner.next_hyp += 1;
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        inner
            .hypotheses
            .insert(id, Hypothesis::new(id, handle, statement, seq));
        Ok(id)
    }

    /// Attach evidence to a hypothesis and return the updated verdict.
    pub fn add_evidence(&self, id: u64, evidence: Evidence) -> Result<(Verdict, f32)> {
        let mut inner = self.inner.write().map_err(|_| poisoned())?;
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let h = inner
            .hypotheses
            .get_mut(&id)
            .ok_or(CoreError::NotFound {
                what: format!("hypothesis {id}"),
            })?;
        h.push(evidence, seq);
        Ok((h.verdict(), h.confidence))
    }

    /// Look up a hypothesis.
    pub fn hypothesis(&self, id: u64) -> Option<Hypothesis> {
        self.inner.read().ok()?.hypotheses.get(&id).cloned()
    }

    /// All hypotheses in creation order.
    pub fn hypotheses(&self) -> Vec<Hypothesis> {
        self.inner
            .read()
            .map(|i| i.hypotheses.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Add a session-level note shown in `session.info`.
    pub fn note(&self, text: impl Into<String>) {
        if let Ok(mut inner) = self.inner.write() {
            inner.notes.push(text.into());
        }
    }

    /// Session notes.
    pub fn notes(&self) -> Vec<String> {
        self.inner
            .read()
            .map(|i| i.notes.clone())
            .unwrap_or_default()
    }

    /// Restore images from a saved session by reopening their paths.
    ///
    /// Traces are not part of a saved session, and the warning list says so:
    /// a trace is an observation with a pid and a wall-clock time, and reloading
    /// one that no longer describes anything would be worse than losing it.
    pub fn restore(&self, saved: &crate::persist::SavedSession) -> Result<Vec<String>> {
        let mut warnings = Vec::new();
        for entry in &saved.images {
            match self.open_path(&entry.path) {
                Ok(ids) => {
                    // An archive reopens as one image per member, and a member's
                    // path carries the archive and the member name, so it never
                    // equals the path that was saved. There is nothing to
                    // compare a member's digest against, so the check applies
                    // only to an image that was saved as itself.
                    for id in ids {
                        let Some(file) = self.image(id) else { continue };
                        if file.image.path != entry.path {
                            continue;
                        }
                        if !entry.sha256.is_empty() && file.image.sha256 != entry.sha256 {
                            warnings.push(format!(
                                "{} changed on disk since the session was saved",
                                entry.path
                            ));
                        }
                    }
                }
                Err(e) => warnings.push(format!("{}: {e}", entry.path)),
            }
        }
        for h in &saved.hypotheses {
            if let Ok(mut inner) = self.inner.write() {
                let id = inner.next_hyp;
                inner.next_hyp += 1;
                let _ = id;
                let seq = self.seq.fetch_add(1, Ordering::Relaxed);
                inner.hypotheses.insert(
                    h.id,
                    Hypothesis {
                        id: h.id,
                        handle: Handle::parse(&h.handle).unwrap_or_else(|_| Handle::image(0)),
                        statement: h.statement.clone(),
                        state: occ_core::HypothesisState::Proposed,
                        evidence: Vec::new(),
                        confidence: 0.0,
                        created_seq: seq,
                        updated_seq: seq,
                    },
                );
            }
        }
        if !self.trace_ids().is_empty() {
            warnings.push(
                "traces are observations, not session state: the handles from the saved session do not resolve here"
                    .to_string(),
            );
        }
        Ok(warnings)
    }

    /// Summary numbers for `session.info`.
    pub fn stats(&self) -> Vec<(u32, String)> {
        let inner = match self.inner.read() {
            Ok(i) => i,
            Err(_) => return Vec::new(),
        };
        inner
            .images
            .iter()
            .map(|(id, f)| {
                (
                    *id,
                    format!(
                        "{} | {} | {} funcs | {} strings | {} imports",
                        f.image.headline(),
                        f.image.sha256.chars().take(12).collect::<String>(),
                        f.analysis.coverage.functions,
                        f.analysis.strings.len(),
                        f.image.imports.len()
                    ),
                )
            })
            .collect()
    }
}

fn poisoned() -> CoreError {
    CoreError::Internal("session lock poisoned by a panicking writer".into())
}

fn parse_or_err(s: &str) -> Result<Addr> {
    occ_core::addr::parse_addr(s).ok_or_else(|| CoreError::NotFound {
        what: format!("address {s}"),
    })
}

/// Collect every function name in an image, for symbol lookup and reports.
pub fn function_names(file: &ImageFile) -> Vec<(Addr, String)> {
    file.analysis
        .functions
        .iter()
        .map(|f| (f.entry, f.display_name()))
        .collect()
}

/// Resolve the owner function of an address for error messages.
pub fn owner_name(file: &ImageFile, addr: Addr) -> String {
    match file.analysis.function_covering(addr) {
        Some(f) => f.display_name(),
        None => format!("{addr}"),
    }
}

/// True when a symbol names code the analyser can decode.
pub fn is_code_symbol(image: &Image, sym: &occ_core::Symbol) -> bool {
    sym.kind == SymbolKind::Function && image.perm_at(sym.addr).exec()
}

/// Convenience for callers that need a function by name or address.
pub fn find_function<'a>(file: &'a ImageFile, input: &str) -> Option<&'a Function> {
    if let Some(addr) = occ_core::addr::parse_addr(input)
        && let Some(f) = file.analysis.function_at(addr)
    {
        return Some(f);
    }
    file.analysis.function_by_name(input)
}
