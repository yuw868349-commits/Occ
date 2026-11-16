//! Stable, human-legible handles.
//!
//! A handle is the unit of addressability across conversational turns. Instead
//! of re-sending a function body every time a follow-up question touches it, a
//! caller says `fn:0x4012a0` and the engine resolves it against session state.
//!
//! Grammar: `<kind>:<value>`, kinds are lower-case ASCII, the separator is the
//! first `:` only — everything after it is the value verbatim, so symbol names
//! containing `:`, `@`, or `$` (C++ mangling, versioned ELF symbols) survive
//! intact.
//!
//! Values are deliberately not normalised. `sym:main` and `sym:MAIN` are two
//! handles, because symbol lookup in most containers is case-sensitive and
//! silently folding case would make a typo resolve to the wrong function.

use crate::addr::Addr;
use crate::error::{CoreError, Result};
use std::fmt;

/// What a handle points at.
///
/// The set is closed on purpose. An open-ended kind would let a typo create a
/// new namespace that nothing ever resolves, and the failure would surface much
/// later as an empty result rather than a parse error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HandleKind {
    /// A loaded image: `img:0`.
    Image,
    /// A section or segment: `sec:.text`.
    Section,
    /// A named symbol: `sym:main`.
    Symbol,
    /// A recovered function by entry address: `fn:0x4012a0`.
    Function,
    /// A basic block by start address: `bb:0x4012a0`.
    Block,
    /// A single instruction by address: `insn:0x4012b4`.
    Insn,
    /// A string literal recovered from a read-only section: `str:0x402000`.
    Str,
    /// An opaque byte range addressed by content hash: `blob:9f3c1a2b`.
    Blob,
    /// An import slot: `imp:puts`.
    Import,
    /// An export slot: `exp:main`.
    Export,
    /// A relocation entry: `rel:0`.
    Reloc,
    /// A hypothesis under evaluation: `hyp:3`.
    Hypothesis,
    /// A cross-reference edge: `xref:17`.
    Xref,
    /// A recorded run: `trace:0`.
    ///
    /// A trace is not part of any image — it is what one execution observed —
    /// so it gets its own namespace rather than being hung off `img`.
    Trace,
}

impl HandleKind {
    /// Canonical lower-case spelling used in the handle grammar.
    pub const fn as_str(self) -> &'static str {
        match self {
            HandleKind::Image => "img",
            HandleKind::Section => "sec",
            HandleKind::Symbol => "sym",
            HandleKind::Function => "fn",
            HandleKind::Block => "bb",
            HandleKind::Insn => "insn",
            HandleKind::Str => "str",
            HandleKind::Blob => "blob",
            HandleKind::Import => "imp",
            HandleKind::Export => "exp",
            HandleKind::Reloc => "rel",
            HandleKind::Hypothesis => "hyp",
            HandleKind::Xref => "xref",
            HandleKind::Trace => "trace",
        }
    }

    /// Parse the kind prefix.
    pub fn from_prefix(prefix: &str) -> Option<Self> {
        Some(match prefix {
            "img" => HandleKind::Image,
            "sec" => HandleKind::Section,
            "sym" => HandleKind::Symbol,
            "fn" => HandleKind::Function,
            "bb" => HandleKind::Block,
            "insn" => HandleKind::Insn,
            "str" => HandleKind::Str,
            "blob" => HandleKind::Blob,
            "imp" => HandleKind::Import,
            "exp" => HandleKind::Export,
            "rel" => HandleKind::Reloc,
            "hyp" => HandleKind::Hypothesis,
            "xref" => HandleKind::Xref,
            "trace" => HandleKind::Trace,
            _ => return None,
        })
    }

    /// Every accepted prefix, for error messages.
    pub const ALL: &'static str =
        "img, sec, sym, fn, bb, insn, str, blob, imp, exp, rel, hyp, xref, trace";
}

/// A parsed handle.
///
/// Serialises as the plain string, so JSON payloads read `"handle": "fn:0x4012a0"`
/// and a model can copy it back without a round trip.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Handle {
    kind: HandleKind,
    value: String,
}

impl Handle {
    /// Build a handle from its parts. The value is stored verbatim.
    pub fn new(kind: HandleKind, value: impl Into<String>) -> Self {
        Handle {
            kind,
            value: value.into(),
        }
    }

    /// `img:<id>`
    pub fn image(id: u32) -> Self {
        Handle::new(HandleKind::Image, id.to_string())
    }

    /// `sec:<name>`
    pub fn section(name: impl AsRef<str>) -> Self {
        Handle::new(HandleKind::Section, name.as_ref())
    }

    /// `sym:<name>`
    pub fn symbol(name: impl AsRef<str>) -> Self {
        Handle::new(HandleKind::Symbol, name.as_ref())
    }

    /// `fn:<addr>`
    pub fn function(addr: Addr) -> Self {
        Handle::new(HandleKind::Function, format!("{addr}"))
    }

    /// `bb:<addr>`
    pub fn block(addr: Addr) -> Self {
        Handle::new(HandleKind::Block, format!("{addr}"))
    }

    /// `insn:<addr>`
    pub fn insn(addr: Addr) -> Self {
        Handle::new(HandleKind::Insn, format!("{addr}"))
    }

    /// `str:<addr>`
    pub fn string(addr: Addr) -> Self {
        Handle::new(HandleKind::Str, format!("{addr}"))
    }

    /// `blob:<hex>` — content-addressed, so identical bytes share a handle
    /// across images and across runs.
    pub fn blob(hash_prefix: impl AsRef<str>) -> Self {
        Handle::new(HandleKind::Blob, hash_prefix.as_ref())
    }

    /// `imp:<name>`
    pub fn import(name: impl AsRef<str>) -> Self {
        Handle::new(HandleKind::Import, name.as_ref())
    }

    /// `exp:<name>`
    pub fn export(name: impl AsRef<str>) -> Self {
        Handle::new(HandleKind::Export, name.as_ref())
    }

    /// `rel:<index>`
    pub fn reloc(index: usize) -> Self {
        Handle::new(HandleKind::Reloc, index.to_string())
    }

    /// `hyp:<n>`
    pub fn hypothesis(id: u64) -> Self {
        Handle::new(HandleKind::Hypothesis, id.to_string())
    }

    /// `xref:<n>`
    pub fn xref(id: u64) -> Self {
        Handle::new(HandleKind::Xref, id.to_string())
    }

    /// `trace:<id>`
    pub fn trace(id: u32) -> Self {
        Handle::new(HandleKind::Trace, id.to_string())
    }

    /// Which namespace this handle lives in.
    pub const fn kind(&self) -> HandleKind {
        self.kind
    }

    /// The raw value after the first `:`.
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Interpret the value as an address, if the kind implies one.
    pub fn addr(&self) -> Option<Addr> {
        match self.kind {
            HandleKind::Function
            | HandleKind::Block
            | HandleKind::Insn
            | HandleKind::Str
            | HandleKind::Xref => crate::addr::parse_addr(&self.value),
            _ => None,
        }
    }

    /// Parse `kind:value`.
    pub fn parse(input: &str) -> Result<Self> {
        let trimmed = input.trim();
        let (prefix, value) = trimmed
            .split_once(':')
            .ok_or_else(|| CoreError::BadHandle {
                input: trimmed.to_string(),
                kinds: HandleKind::ALL,
            })?;
        let kind = HandleKind::from_prefix(&prefix.to_ascii_lowercase()).ok_or_else(|| {
            CoreError::BadHandle {
                input: trimmed.to_string(),
                kinds: HandleKind::ALL,
            }
        })?;
        if value.is_empty() {
            return Err(CoreError::BadHandle {
                input: trimmed.to_string(),
                kinds: HandleKind::ALL,
            });
        }
        Ok(Handle::new(kind, value))
    }

    /// Accept either a handle string or something that is obviously an address
    /// or a bare name.
    ///
    /// This exists because a model will sometimes answer `0x4012a0` when asked
    /// which function it means. Rejecting that costs a retry and an apology;
    /// accepting it costs one line here. `default_kind` decides what a bare
    /// value means in the calling context.
    pub fn lax(input: &str, default_kind: HandleKind) -> Result<Self> {
        let trimmed = input.trim();
        if trimmed.contains(':') {
            if let Ok(h) = Handle::parse(trimmed) {
                return Ok(h);
            }
        }
        if let Some(addr) = crate::addr::parse_addr(trimmed) {
            return Ok(Handle::new(default_kind, format!("{addr}")));
        }
        if trimmed.is_empty() {
            return Err(CoreError::BadHandle {
                input: trimmed.to_string(),
                kinds: HandleKind::ALL,
            });
        }
        Ok(Handle::new(default_kind, trimmed))
    }
}

impl fmt::Display for Handle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.kind.as_str(), self.value)
    }
}

impl fmt::Debug for Handle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_text() {
        let cases = [
            Handle::function(Addr(0x4012a0)),
            Handle::section(".text"),
            Handle::symbol("_ZN3foo3barEv"),
            Handle::import("puts"),
            Handle::blob("9f3c1a2b"),
        ];
        for h in cases {
            let text = h.to_string();
            let back = Handle::parse(&text).expect("parse");
            assert_eq!(h, back);
        }
    }

    #[test]
    fn value_survives_colons_after_the_first() {
        // C++ mangled names and versioned ELF symbols contain colons.
        let h = Handle::parse("sym:_ZNSt6vectorIiSaIiEE9push_backERKi@GLIBCXX_3.4").expect("parse");
        assert_eq!(h.kind(), HandleKind::Symbol);
        assert_eq!(h.value(), "_ZNSt6vectorIiSaIiEE9push_backERKi@GLIBCXX_3.4");

        let h2 = Handle::parse("sym:foo::bar").expect("parse");
        assert_eq!(h2.value(), "foo::bar");
    }

    #[test]
    fn rejects_unknown_kind() {
        let err = Handle::parse("widget:3");
        assert!(matches!(err, Err(CoreError::BadHandle { .. })));
    }

    #[test]
    fn rejects_empty_value() {
        assert!(Handle::parse("fn:").is_err());
        assert!(Handle::parse("fn").is_err());
    }

    #[test]
    fn renders_as_bare_string() {
        assert_eq!(Handle::function(Addr(0x4012a0)).to_string(), "fn:0x4012a0");
    }

    #[test]
    fn lax_accepts_bare_address() {
        let h = Handle::lax("0x4012a0", HandleKind::Function).expect("lax");
        assert_eq!(h, Handle::function(Addr(0x4012a0)));

        let h = Handle::lax("401000h", HandleKind::Function).expect("lax");
        assert_eq!(h, Handle::function(Addr(0x401000)));
    }

    #[test]
    fn lax_keeps_explicit_kind() {
        let h = Handle::lax("sec:.text", HandleKind::Function).expect("lax");
        assert_eq!(h.kind(), HandleKind::Section);
    }

    #[test]
    fn lax_treats_non_address_as_name() {
        let h = Handle::lax("main", HandleKind::Symbol).expect("lax");
        assert_eq!(h, Handle::symbol("main"));
    }

    #[test]
    fn addr_accessor_only_for_address_kinds() {
        assert_eq!(Handle::function(Addr(0x10)).addr(), Some(Addr(0x10)));
        assert_eq!(Handle::section(".text").addr(), None);
    }
}
