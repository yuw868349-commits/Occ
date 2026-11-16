//! Evidence chains.
//!
//! Every conclusion the engine emits carries the observations that produced it.
//! This is the load-bearing idea of the whole project: an LLM cannot check its
//! own work, but it *can* check whether the byte range cited as evidence
//! actually contains what the evidence claims. Making that check cheap is more
//! effective at suppressing hallucination than any amount of prompt tuning.
//!
//! # How confidence is computed
//!
//! Each [`Evidence`] carries a signed `weight` in `[-1, 1]`: positive supports
//! the statement, negative contradicts it. Independent supporting observations
//! combine by noisy-OR, `1 - Π(1 - wᵢ)` — two weak signals agreeing is stronger
//! than either alone, but never reaches certainty. Contradictions combine the
//! same way among themselves, then multiply the result down:
//!
//! ```text
//! support = 1 - Π(1 - wᵢ)   for wᵢ > 0
//! refute  = 1 - Π(1 - |wᵢ|) for wᵢ < 0
//! confidence = support × (1 - refute)
//! ```
//!
//! So two 0.6 signals give 0.84; adding one 0.5 counter-signal drops that to
//! 0.42. The function is monotone in every input and stays in `[0, 1]`, which
//! is what makes the number defensible when a caller asks "why 0.84?".

use crate::addr::Addr;
use crate::handle::Handle;

/// What kind of observation an evidence item is.
///
/// The taxonomy matters downstream: a caller filtering for high-trust evidence
/// should keep `Symbol` and `DynamicTrace` and discard `Heuristic`. Lumping
/// everything into "evidence" would make that filter impossible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvidenceKind {
    /// Bytes at a specific offset match a known signature.
    Bytes,
    /// The shape of the control flow (loop nesting, unrolled blocks, jump table).
    ControlFlow,
    /// A data table with recognisable structure (S-box, alphabet, dispatch table).
    Table,
    /// A symbol or debug-info entry names it.
    Symbol,
    /// A string literal is referenced from here.
    StringRef,
    /// An import is called from here, and the import name is informative.
    ImportCall,
    /// A distinctive constant appears (magic number, polynomial, CRC seed).
    Constant,
    /// The region's entropy is characteristic of compression or encryption.
    Entropy,
    /// Another location references this one.
    CrossReference,
    /// Observed at runtime under the tracer.
    DynamicTrace,
    /// Structural metadata from the container (section flags, entry point, TLS).
    FormatMeta,
    /// Absence of an expected feature (no relocations, no imports) — negative
    /// evidence is still evidence, and its absence is often the tell.
    Absence,
}

impl EvidenceKind {
    /// Short label used in compact renderings.
    pub const fn label(self) -> &'static str {
        match self {
            EvidenceKind::Bytes => "bytes",
            EvidenceKind::ControlFlow => "cfg",
            EvidenceKind::Table => "table",
            EvidenceKind::Symbol => "symbol",
            EvidenceKind::StringRef => "string",
            EvidenceKind::ImportCall => "import",
            EvidenceKind::Constant => "const",
            EvidenceKind::Entropy => "entropy",
            EvidenceKind::CrossReference => "xref",
            EvidenceKind::DynamicTrace => "trace",
            EvidenceKind::FormatMeta => "format",
            EvidenceKind::Absence => "absence",
        }
    }

    /// Inverse of [`EvidenceKind::label`].
    ///
    /// Returns `None` for a label this build does not know, so a reader can
    /// reject a document instead of guessing at its meaning.
    pub fn from_label(s: &str) -> Option<Self> {
        Some(match s {
            "bytes" => EvidenceKind::Bytes,
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
            _ => return None,
        })
    }
}

/// Which analysis stage produced an observation.
///
/// Separating *what* was observed from *who* observed it lets a caller say
/// "trust the loader, distrust the heuristic" without parsing prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvidenceSource {
    /// The container loader (section table, headers, symbol table).
    Loader,
    /// The disassembler and control-flow reconstruction.
    Disassembler,
    /// Data-flow and cross-reference analysis.
    DataFlow,
    /// Pattern and signature scanning.
    Scanner,
    /// Runtime observation under ptrace.
    Tracer,
    /// A statistical heuristic (entropy, code density).
    Heuristic,
}

impl EvidenceSource {
    /// Short label for compact renderings.
    pub const fn label(self) -> &'static str {
        match self {
            EvidenceSource::Loader => "loader",
            EvidenceSource::Disassembler => "disasm",
            EvidenceSource::DataFlow => "dataflow",
            EvidenceSource::Scanner => "scan",
            EvidenceSource::Tracer => "trace",
            EvidenceSource::Heuristic => "heuristic",
        }
    }

    /// Inverse of [`EvidenceSource::label`].
    pub fn from_label(s: &str) -> Option<Self> {
        Some(match s {
            "loader" => EvidenceSource::Loader,
            "disasm" => EvidenceSource::Disassembler,
            "dataflow" => EvidenceSource::DataFlow,
            "scan" => EvidenceSource::Scanner,
            "trace" => EvidenceSource::Tracer,
            "heuristic" => EvidenceSource::Heuristic,
            _ => return None,
        })
    }
}

/// One observation supporting or contradicting a claim.
#[derive(Debug, Clone)]
pub struct Evidence {
    /// Category of observation.
    pub kind: EvidenceKind,
    /// Stage that produced it.
    pub source: EvidenceSource,
    /// Where it was observed, when the observation is localisable.
    pub at: Option<Addr>,
    /// How many bytes the observation covers, starting at [`Evidence::at`].
    pub span: Option<u64>,
    /// What was observed, in one sentence a caller can verify by reading bytes.
    pub detail: String,
    /// Signed strength in `[-1, 1]`. Positive supports, negative contradicts.
    pub weight: f32,
}

impl Evidence {
    /// A supporting observation.
    pub fn support(
        kind: EvidenceKind,
        source: EvidenceSource,
        detail: impl Into<String>,
        weight: f32,
    ) -> Self {
        Evidence {
            kind,
            source,
            at: None,
            span: None,
            detail: detail.into(),
            weight: weight.clamp(0.0, 1.0),
        }
    }

    /// A contradicting observation.
    pub fn refute(
        kind: EvidenceKind,
        source: EvidenceSource,
        detail: impl Into<String>,
        weight: f32,
    ) -> Self {
        Evidence {
            kind,
            source,
            at: None,
            span: None,
            detail: detail.into(),
            weight: -weight.clamp(0.0, 1.0),
        }
    }

    /// Attach a location.
    pub fn at(mut self, addr: Addr) -> Self {
        self.at = Some(addr);
        self
    }

    /// Attach a byte span starting at [`Evidence::at`].
    pub fn span(mut self, bytes: u64) -> Self {
        self.span = Some(bytes);
        self
    }

    /// True when this observation argues against the claim.
    pub const fn is_counter(&self) -> bool {
        self.weight < 0.0
    }
}

/// A competing explanation with its own confidence.
///
/// Recorded explicitly so that a caller is not presented with a single answer
/// and left to assume it was the only one considered. `"base64 decode" 0.86`
/// next to `"hex decode" 0.05` tells the reader far more than `0.86` alone.
#[derive(Debug, Clone)]
pub struct Alternative {
    /// The competing statement.
    pub statement: String,
    /// Its confidence, computed the same way as the primary claim.
    pub confidence: f32,
}

impl Alternative {
    /// Build one.
    pub fn new(statement: impl Into<String>, confidence: f32) -> Self {
        Alternative {
            statement: statement.into(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

/// Noisy-OR combination of evidence weights.
///
/// Returns a value in `[0, 1]`. An empty slice returns 0.0 — no evidence means
/// no confidence, not "probably fine", which is the failure mode that makes
/// binary-analysis assistants untrustworthy.
pub fn combine(evidence: &[Evidence]) -> f32 {
    let (support, refute) = combine_parts(evidence);
    (support * (1.0 - refute)).clamp(0.0, 1.0)
}

/// Both halves of [`combine`], uncombined.
///
/// The caller that only wants a single number should use [`combine`]. Callers
/// that have to *classify* — "supported", "refuted", "we saw nothing" — need the
/// two sides separately, because a statement with no support and a weak counter
/// argument is not the same finding as one with strong counter-evidence, even
/// though both multiply to a confidence of zero.
pub fn combine_parts(evidence: &[Evidence]) -> (f32, f32) {
    let mut support_product = 1.0f32;
    let mut refute_product = 1.0f32;
    let mut any_support = false;
    let mut any_refute = false;

    for e in evidence {
        let w = e.weight.clamp(-1.0, 1.0);
        if w > 0.0 {
            support_product *= 1.0 - w;
            any_support = true;
        } else if w < 0.0 {
            refute_product *= 1.0 + w; // 1 - |w|
            any_refute = true;
        }
    }

    let support = if any_support {
        1.0 - support_product
    } else {
        0.0
    };
    let refute = if any_refute {
        1.0 - refute_product
    } else {
        0.0
    };

    (support.clamp(0.0, 1.0), refute.clamp(0.0, 1.0))
}

/// A justified conclusion about one entity.
#[derive(Debug, Clone)]
pub struct Claim {
    /// What this claim is about.
    pub handle: Handle,
    /// The assertion, phrased as a verifiable sentence.
    pub statement: String,
    /// Observations behind it.
    pub evidence: Vec<Evidence>,
    /// Derived from [`Claim::evidence`]; present in output, recomputed on load.
    pub confidence: f32,
    /// Other explanations that were considered and rejected or down-weighted.
    pub alternatives: Vec<Alternative>,
    /// Explicit limits on what the claim covers.
    pub caveats: Vec<String>,
}

impl Claim {
    /// Start a claim with no evidence yet.
    pub fn new(handle: Handle, statement: impl Into<String>) -> Self {
        Claim {
            handle,
            statement: statement.into(),
            evidence: Vec::new(),
            confidence: 0.0,
            alternatives: Vec::new(),
            caveats: Vec::new(),
        }
    }

    /// Add an observation and refresh confidence.
    pub fn with(mut self, evidence: Evidence) -> Self {
        self.evidence.push(evidence);
        self.recompute();
        self
    }

    /// Add a competing explanation.
    pub fn or(mut self, alternative: Alternative) -> Self {
        self.alternatives.push(alternative);
        self
    }

    /// Record a caveat.
    pub fn caveat(mut self, note: impl Into<String>) -> Self {
        self.caveats.push(note.into());
        self
    }

    /// Recompute [`Claim::confidence`] from the current evidence.
    ///
    /// Must be called after deserialising, because confidence is derived state
    /// that a client could have edited in transit.
    pub fn recompute(&mut self) -> &mut Self {
        self.confidence = combine(&self.evidence);
        self
    }

    /// True when confidence clears a usefulness threshold.
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(weight: f32) -> Evidence {
        Evidence::support(EvidenceKind::Bytes, EvidenceSource::Scanner, "x", weight)
    }

    #[test]
    fn empty_evidence_is_zero_confidence() {
        assert_eq!(combine(&[]), 0.0);
        let c = Claim::new(Handle::function(Addr(0x10)), "something");
        assert_eq!(c.confidence, 0.0);
    }

    #[test]
    fn single_support_passes_through() {
        assert!((combine(&[ev(0.6)]) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn independent_supports_combine_by_noisy_or() {
        // 1 - (1-0.6)^2 = 0.84
        let c = combine(&[ev(0.6), ev(0.6)]);
        assert!((c - 0.84).abs() < 1e-5, "got {c}");
    }

    #[test]
    fn counter_evidence_reduces_confidence() {
        let mut counter = ev(0.5);
        counter.weight = -0.5;
        // support 0.84, refute 0.5, result 0.42
        let c = combine(&[ev(0.6), ev(0.6), counter]);
        assert!((c - 0.42).abs() < 1e-5, "got {c}");
    }

    #[test]
    fn only_counter_evidence_yields_zero() {
        let mut counter = ev(0.9);
        counter.weight = -0.9;
        assert_eq!(combine(&[counter]), 0.0);
    }

    #[test]
    fn confidence_is_monotone_in_support() {
        let a = combine(&[ev(0.3)]);
        let b = combine(&[ev(0.3), ev(0.3)]);
        let c = combine(&[ev(0.3), ev(0.3), ev(0.3)]);
        assert!(a < b && b < c);
        assert!(c <= 1.0);
    }

    #[test]
    fn weights_are_clamped_on_construction() {
        let e = ev(5.0);
        assert_eq!(e.weight, 1.0);
        let r = Evidence::refute(
            EvidenceKind::Absence,
            EvidenceSource::Heuristic,
            "no relocations",
            9.0,
        );
        assert_eq!(r.weight, -1.0);
        assert!(r.is_counter());
    }

    #[test]
    fn builder_recomputes_confidence() {
        let c = Claim::new(Handle::section(".text"), "is packed")
            .with(ev(0.7))
            .with(ev(0.5));
        assert!((c.confidence - 0.85).abs() < 1e-5, "got {}", c.confidence);
    }

    #[test]
    fn full_support_saturates_at_one() {
        let c = combine(&[ev(1.0), ev(1.0)]);
        assert!((c - 1.0).abs() < 1e-6);
    }
}
