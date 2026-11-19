//! JSON rendering.
//!
//! Every response is the same envelope, so a client can handle a new tool
//! without new parsing code:
//!
//! ```json
//! {"ok": true, "handle": "fn:0x4012a0", "summary": "main, 412 bytes",
//!  "data": {...}, "evidence": [...], "confidence": 0.93,
//!  "alternatives": [], "caveats": [], "next": [...],
//!  "tokens_estimate": 812, "truncated": false}
//! ```
//!
//! Two properties matter more than the shape. `handle` is always present and
//! always resolvable on the next turn, so a caller can drill down without
//! repeating the query. `tokens_estimate` is computed from the actual payload
//! size, so a client can decide to ask for less before the response is sent,
//! rather than discovering it does not fit.

use occ_analyze::analysis::Analysis;
use occ_analyze::strings::Found;
use occ_core::{
    Addr, Alternative, BasicBlock, Claim, Depth, Evidence, EvidenceKind, Function, Hypothesis,
    Image, Import, Instruction, RelocKind, Relocation, Resource, ResourceName, Section, Symbol,
    Xref,
};
use occ_json::{Map, Value};

/// Build an object literal.
#[macro_export]
macro_rules! obj {
    ($($k:expr => $v:expr),* $(,)?) => {{
        let mut m = occ_json::Map::new();
        $( m.insert($k, $v); )*
        occ_json::Value::Object(m)
    }};
}

/// Build an array from anything iterable.
#[macro_export]
macro_rules! arr {
    ($($v:expr),* $(,)?) => {
        occ_json::Value::Array(vec![$(occ_json::Value::from($v)),*])
    };
}

/// Address as the hex string every payload uses.
pub fn addr(a: Addr) -> Value {
    Value::String(a.to_string())
}

/// Optional address.
pub fn addr_opt(a: Option<Addr>) -> Value {
    match a {
        Some(a) => addr(a),
        None => Value::Null,
    }
}

/// Strings as JSON arrays.
pub fn strings(items: &[String]) -> Value {
    Value::Array(items.iter().map(|s| Value::String(s.clone())).collect())
}

/// The standard response envelope.
pub struct Envelope {
    data: Map,
    handle: Option<String>,
    summary: String,
    evidence: Vec<Evidence>,
    confidence: Option<f32>,
    alternatives: Vec<Alternative>,
    caveats: Vec<String>,
    next: Vec<String>,
    truncated: bool,
}

impl Envelope {
    /// Start a successful response.
    pub fn ok(summary: impl Into<String>) -> Envelope {
        Envelope {
            data: Map::new(),
            handle: None,
            summary: summary.into(),
            evidence: Vec::new(),
            confidence: None,
            alternatives: Vec::new(),
            caveats: Vec::new(),
            next: Vec::new(),
            truncated: false,
        }
    }

    /// Start a failed response.
    pub fn err(summary: impl Into<String>) -> Envelope {
        let mut e = Envelope::ok(summary);
        e.data.insert("error", Value::Bool(true));
        e
    }

    /// Attach the handle the result can be addressed by.
    pub fn handle(mut self, h: impl ToString) -> Envelope {
        self.handle = Some(h.to_string());
        self
    }

    /// Add a payload field.
    pub fn field(mut self, key: &str, value: impl Into<Value>) -> Envelope {
        self.data.insert(key, value);
        self
    }

    /// Add several payload fields.
    pub fn fields(mut self, fields: Map) -> Envelope {
        for (k, v) in fields.iter() {
            self.data.insert(k.to_string(), v.clone());
        }
        self
    }

    /// Attach evidence.
    pub fn evidence(mut self, ev: Vec<Evidence>) -> Envelope {
        self.evidence = ev;
        self
    }

    /// Attach a confidence value.
    pub fn confidence(mut self, c: f32) -> Envelope {
        self.confidence = Some(c);
        self
    }

    /// Attach alternatives.
    pub fn alternatives(mut self, a: Vec<Alternative>) -> Envelope {
        self.alternatives = a;
        self
    }

    /// Attach caveats.
    pub fn caveats(mut self, c: Vec<String>) -> Envelope {
        self.caveats = c;
        self
    }

    /// Suggest next calls.
    pub fn next(mut self, n: Vec<String>) -> Envelope {
        self.next = n;
        self
    }

    /// Mark the payload as clipped by a limit.
    pub fn truncated(mut self, t: bool) -> Envelope {
        self.truncated = t;
        self
    }

    /// Render to a JSON value.
    pub fn build(self) -> Value {
        let mut m = Map::new();
        m.insert("ok", Value::Bool(true));
        m.insert(
            "handle",
            match self.handle {
                Some(h) => Value::String(h),
                None => Value::Null,
            },
        );
        m.insert("summary", Value::String(self.summary));
        let data = Value::Object(self.data);
        m.insert("data", data.clone());
        m.insert(
            "evidence",
            Value::Array(self.evidence.iter().map(evidence_json).collect()),
        );
        m.insert(
            "confidence",
            match self.confidence {
                Some(c) => Value::from(f64::from(c * 1000.0).round() / 1000.0),
                None => Value::Null,
            },
        );
        m.insert(
            "alternatives",
            Value::Array(
                self.alternatives
                    .iter()
                    .map(|a| {
                        obj!("statement" => a.statement.clone(), "confidence" => f64::from(a.confidence))
                    })
                    .collect(),
            ),
        );
        m.insert("caveats", strings(&self.caveats));
        m.insert("next", strings(&self.next));
        let estimate = estimate(&data);
        m.insert("tokens_estimate", Value::from(estimate));
        m.insert("truncated", Value::Bool(self.truncated));
        Value::Object(m)
    }
}

/// Rough token count for a payload: four bytes per token, rounded up.
pub fn estimate(v: &Value) -> usize {
    let text = occ_json::to_string(v);
    text.len().div_ceil(4)
}

/// Evidence as JSON.
pub fn evidence_json(e: &Evidence) -> Value {
    obj!(
        "kind" => e.kind.label(),
        "source" => e.source.label(),
        "at" => addr_opt(e.at),
        "span" => match e.span { Some(s) => Value::from(s), None => Value::Null },
        "detail" => e.detail.clone(),
        "weight" => f64::from(e.weight),
        "direction" => if e.is_counter() { "refute" } else { "support" },
    )
}

/// Claim as JSON, without the envelope.
pub fn claim_json(c: &Claim) -> Value {
    obj!(
        "handle" => c.handle.to_string(),
        "statement" => c.statement.clone(),
        "confidence" => f64::from(c.confidence),
        "evidence" => Value::Array(c.evidence.iter().map(evidence_json).collect()),
        "alternatives" => Value::Array(
            c.alternatives.iter().map(|a| obj!(
                "statement" => a.statement.clone(),
                "confidence" => f64::from(a.confidence)
            )).collect()
        ),
        "caveats" => strings(&c.caveats),
    )
}

/// Image identity and headers.
pub fn image_json(image: &Image) -> Value {
    let mut m = Map::new();
    m.insert("id", Value::from(image.id.0));
    m.insert("path", Value::String(image.path.clone()));
    m.insert("format", image.format.label());
    m.insert("arch", image.arch.label());
    m.insert("bits", Value::from(u32::from(image.bits)));
    m.insert(
        "endian",
        match image.endian {
            occ_core::Endian::Little => "little",
            occ_core::Endian::Big => "big",
        },
    );
    m.insert("entry", addr(image.entry));
    m.insert("image_base", addr(image.image_base));
    m.insert("file_size", Value::from(image.file_size));
    m.insert("sha256", Value::String(image.sha256.clone()));
    m.insert("mapped_size", Value::from(image.mapped_size()));
    m.insert("segments", Value::from(image.segments.len()));
    m.insert("sections", Value::from(image.sections.len()));
    m.insert("symbols", Value::from(image.symbols.len()));
    m.insert("imports", Value::from(image.imports.len()));
    m.insert("exports", Value::from(image.exports.len()));
    m.insert("relocations", Value::from(image.relocations.len()));
    m.insert(
        "interpreter",
        match &image.interpreter {
            Some(i) => Value::String(i.clone()),
            None => Value::Null,
        },
    );
    m.insert("linkage", image.fingerprint.linkage_label());
    m.insert("warnings", strings(&image.warnings));
    Value::Object(m)
}

/// Fingerprint as JSON.
pub fn fingerprint_json(fp: &occ_core::Fingerprint) -> Value {
    obj!(
        "compiler" => fp.compiler.label(),
        "compiler_claim" => claim_json(&fp.compiler_claim),
        "language" => fp.language.label(),
        "linkage" => fp.linkage_label(),
        "stripped" => fp.stripped,
        "has_dwarf" => fp.has_dwarf,
        "has_go_buildinfo" => fp.has_go_buildinfo,
        "runtime" => match &fp.runtime { Some(r) => Value::String(r.clone()), None => Value::Null },
        "packer" => match &fp.packer_suspicion { Some(c) => claim_json(c), None => Value::Null },
        "entropy" => Value::Array(
            fp.section_entropy.iter().map(|(n, e)| {
                obj!("section" => n.clone(), "entropy" => f64::from(*e))
            }).collect()
        ),
    )
}

/// Section as JSON.
pub fn section_json(s: &Section) -> Value {
    obj!(
        "index" => Value::from(s.index),
        "name" => s.name.clone(),
        "addr" => addr(s.addr),
        "size" => Value::from(s.size),
        "file_offset" => Value::from(s.file_offset),
        "file_size" => Value::from(s.file_size),
        "kind" => format!("{:?}", s.kind),
        "perm" => s.perm.to_string(),
    )
}

/// Resource leaf as JSON.
pub fn resource_json(r: &Resource) -> Value {
    obj!(
        "type" => r.kind.label(),
        "type_id" => Value::from(r.type_id),
        "name" => match &r.name {
            ResourceName::Id(id) => Value::from(*id),
            ResourceName::String(s) => Value::String(s.clone()),
        },
        "language" => Value::from(r.language),
        "addr" => addr(r.addr),
        "size" => Value::from(r.size),
        "code_page" => Value::from(r.code_page),
    )
}

/// Symbol as JSON.
pub fn symbol_json(s: &Symbol) -> Value {
    obj!(
        "name" => s.name.clone(),
        "addr" => addr(s.addr),
        "size" => Value::from(s.size),
        "kind" => format!("{:?}", s.kind),
        "binding" => format!("{:?}", s.binding),
        "section" => match &s.section { Some(x) => Value::String(x.clone()), None => Value::Null },
        "defined" => s.defined,
    )
}

/// Import as JSON.
pub fn import_json(i: &Import) -> Value {
    obj!(
        "name" => i.name.clone(),
        "library" => match &i.library { Some(x) => Value::String(x.clone()), None => Value::Null },
        "ordinal" => match i.ordinal { Some(x) => Value::from(x), None => Value::Null },
        "slot" => addr_opt(i.slot),
        "kind" => format!("{:?}", i.kind),
    )
}

/// Export as JSON.
pub fn export_json(e: &occ_core::Export) -> Value {
    obj!(
        "name" => e.name.clone(),
        "addr" => addr(e.addr),
        "ordinal" => match e.ordinal { Some(x) => Value::from(x), None => Value::Null },
    )
}

/// Relocation as JSON.
pub fn reloc_json(r: &Relocation) -> Value {
    let kind = match r.kind {
        RelocKind::Absolute => "absolute",
        RelocKind::Relative => "relative",
        RelocKind::Got => "got",
        RelocKind::Tls => "tls",
        RelocKind::ImageRelative => "image-relative",
        RelocKind::Other => "other",
    };
    obj!(
        "offset" => addr(r.offset),
        "kind" => kind,
        "symbol" => match &r.symbol { Some(x) => Value::String(x.clone()), None => Value::Null },
        "addend" => match r.addend { Some(x) => Value::from(x), None => Value::Null },
    )
}

/// String as JSON.
pub fn string_json(s: &Found) -> Value {
    obj!(
        "addr" => addr(s.addr),
        "text" => s.text.clone(),
        "encoding" => s.encoding.label(),
        "bytes" => Value::from(s.byte_len),
        "section" => s.section.clone(),
        "terminated" => s.terminated,
        "kind" => if s.looks_like_path() { "path" }
                  else if s.looks_like_format() { "format" }
                  else { "text" },
    )
}

/// Instruction as JSON.
pub fn insn_json(i: &Instruction) -> Value {
    let mut text = i.text.clone();
    // The decoder appends the resolved RIP-relative address after `#`; keeping
    // it in the text preserves objdump familiarity, and the separate `target`
    // field below gives machine-readable access.
    if text.len() > 160 {
        text.truncate(157);
        text.push_str("...");
    }
    let target = match &i.target {
        Some(occ_core::CallTarget::Direct { to }) => addr(*to),
        Some(occ_core::CallTarget::Import { name, .. }) => Value::String(format!("imp:{name}")),
        Some(occ_core::CallTarget::Indirect { via }) => addr_opt(*via),
        None => Value::Null,
    };
    obj!(
        "addr" => addr(i.addr),
        "text" => text,
        "mnemonic" => i.mnemonic.to_string(),
        "len" => Value::from(i.len),
        "flow" => format!("{:?}", i.flow),
        "target" => target,
        "reads" => Value::Array(i.mem_reads.iter().map(|a| addr(*a)).collect()),
        "writes" => Value::Array(i.mem_writes.iter().map(|a| addr(*a)).collect()),
    )
}

/// Block as JSON, optionally with its instructions.
pub fn block_json(b: &BasicBlock, f: &Function, with_insns: bool, cap: usize) -> Value {
    let mut m = Map::new();
    m.insert("start", addr(b.start));
    m.insert("end", addr(b.end));
    m.insert("insns", Value::from(b.len()));
    m.insert(
        "edges",
        Value::Array(
            b.edges
                .iter()
                .map(|e| obj!("to" => addr(e.to), "kind" => format!("{:?}", e.kind)))
                .collect(),
        ),
    );
    if with_insns {
        let slice = &f.insns[b.insn_start..b.insn_end.min(f.insns.len())];
        m.insert(
            "body",
            Value::Array(slice.iter().take(cap).map(insn_json).collect()),
        );
    }
    Value::Object(m)
}

/// Function as JSON.
pub fn function_json(f: &Function, an: &Analysis, depth: Depth, image: &Image) -> Value {
    let limits = depth.limits();
    let mut m = Map::new();
    m.insert("entry", addr(f.entry));
    m.insert("name", f.display_name());
    m.insert("size", Value::from(f.size));
    // Bytes of the body that no instruction occupies. A large value is not a
    // contradiction: compilers lay a function's cold blocks far from its hot
    // path, and a jump table can sit in `.text` between them. Reporting the
    // number keeps `size` and the instruction count from looking inconsistent
    // when they are not.
    // Union, not sum: a branch into the middle of a decoded instruction makes
    // two instructions claim the same byte, and adding their lengths would
    // count it twice.
    let mut insn_bytes = 0u64;
    let mut gaps = 0u64;
    let mut covered = f.entry.0;
    for insn in &f.insns {
        if insn.addr.0 > covered {
            gaps += 1;
        }
        let from = insn.addr.0.max(covered);
        insn_bytes += insn.end().0.saturating_sub(from);
        covered = covered.max(insn.end().0);
    }
    m.insert("insn_bytes", Value::from(insn_bytes));
    m.insert("gap_bytes", Value::from(f.size.saturating_sub(insn_bytes)));
    m.insert("gaps", Value::from(gaps));
    m.insert("end", addr(f.end()));
    m.insert("insns", Value::from(f.insns.len()));
    m.insert("blocks", Value::from(f.blocks.len()));
    m.insert("complexity", Value::from(f.cyclomatic_complexity()));
    m.insert("is_thunk", f.is_thunk);
    m.insert("confidence", Value::from(f64::from(f.boundary_confidence)));
    m.insert(
        "callers",
        Value::Array(
            f.callers
                .iter()
                .take(limits.max_items)
                .map(|a| addr(*a))
                .collect(),
        ),
    );
    m.insert(
        "callees",
        Value::Array(
            f.callees
                .iter()
                .take(limits.max_items)
                .map(|a| addr(*a))
                .collect(),
        ),
    );
    m.insert("import_calls", strings(&f.import_calls));
    m.insert(
        "strings",
        Value::Array(
            f.string_refs
                .iter()
                .take(limits.max_items)
                .map(|a| match an.string_at(*a) {
                    Some(s) => obj!("addr" => addr(*a), "text" => s.text.clone()),
                    None => obj!("addr" => addr(*a), "text" => Value::Null),
                })
                .collect(),
        ),
    );
    m.insert(
        "evidence",
        Value::Array(f.boundary_evidence.iter().map(evidence_json).collect()),
    );
    if !f.jump_tables.is_empty() {
        m.insert(
            "jump_tables",
            Value::Array(
                f.jump_tables
                    .iter()
                    .take(limits.max_items)
                    .map(|t| {
                        obj!(
                            "at" => addr(t.at),
                            "base" => addr(t.base),
                            "entry_bytes" => Value::from(u32::from(t.entry_bytes)),
                            "relative" => t.relative,
                            "count" => Value::from(t.count),
                            "index_register" => Value::from(t.index_reg.clone()),
                            "targets" => Value::Array(
                                t.targets
                                    .iter()
                                    .take(limits.max_items)
                                    .map(|a| addr(*a))
                                    .collect(),
                            ),
                        )
                    })
                    .collect(),
            ),
        );
    }
    if depth != Depth::Summary {
        m.insert(
            "blocks_detail",
            Value::Array(
                f.blocks
                    .iter()
                    .take(limits.max_items)
                    .map(|b| block_json(b, f, depth == Depth::Raw, limits.max_insns / 4))
                    .collect(),
            ),
        );
    }
    if depth == Depth::Raw {
        m.insert(
            "body",
            Value::Array(
                f.insns
                    .iter()
                    .take(limits.max_insns)
                    .map(insn_json)
                    .collect(),
            ),
        );
    }
    let _ = image;
    Value::Object(m)
}

/// Xref as JSON.
pub fn xref_json(x: &Xref) -> Value {
    obj!(
        "from" => addr(x.from),
        "to" => addr(x.to),
        "kind" => format!("{:?}", x.kind),
        "from_function" => addr_opt(x.from_function),
    )
}

/// Hypothesis as JSON.
pub fn hypothesis_json(h: &Hypothesis) -> Value {
    obj!(
        "id" => Value::from(h.id),
        "handle" => h.handle.to_string(),
        "image" => match h.image {
            Some(id) => Value::from(id),
            None => Value::Null,
        },
        "statement" => h.statement.clone(),
        "state" => format!("{:?}", h.state),
        "verdict" => h.verdict().label(),
        "confidence" => f64::from(h.confidence),
        "evidence" => Value::Array(h.evidence.iter().map(evidence_json).collect()),
        "created_seq" => Value::from(h.created_seq),
        "updated_seq" => Value::from(h.updated_seq),
    )
}

/// Hex dump as text lines, `offset  hex  ascii`.
pub fn hexdump(bytes: &[u8], base: Addr, width: usize) -> String {
    let mut out = String::new();
    for (row, chunk) in bytes.chunks(width.max(1)).enumerate() {
        let offset = base.add((row * width) as u64);
        out.push_str(&format!("{offset:>10}  "));
        for i in 0..width {
            match chunk.get(i) {
                Some(b) => out.push_str(&format!("{b:02x} ")),
                None => out.push_str("   "),
            }
            if i == width / 2 - 1 {
                out.push(' ');
            }
        }
        out.push(' ');
        for b in chunk {
            out.push(if (0x20..0x7f).contains(b) {
                *b as char
            } else {
                '.'
            });
        }
        out.push('\n');
    }
    out
}

/// Evidence kind label, re-exported so tools can construct evidence uniformly.
pub const fn kind_label(k: EvidenceKind) -> &'static str {
    k.label()
}
