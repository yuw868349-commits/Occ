//! Hypotheses under evaluation.
//!
//! A hypothesis is the inverse of a claim. A claim is something the engine
//! asserts; a hypothesis is something the *caller* asserts and asks the engine
//! to test. Keeping them as separate types stops the two directions of
//! reasoning from being conflated in the serialised output, which matters
//! because a caller reading a transcript needs to know who said what.
//!
//! The state machine has deliberately coarse thresholds. A hypothesis is
//! confirmed at 0.75, refuted at 0.25 *with at least one counter-observation*,
//! and inconclusive otherwise. The asymmetry is intentional: refutation
//! requires evidence rather than merely absent support, because "I found
//! nothing" is the normal outcome of looking in the wrong place.

use crate::evidence::{Evidence, combine, combine_parts};
use crate::handle::Handle;

/// Where a hypothesis stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HypothesisState {
    /// Filed but not yet examined.
    Proposed,
    /// Evidence clears the confirmation threshold.
    Supported,
    /// Counter-evidence clears the refutation threshold.
    Refuted,
    /// Examined, but the evidence does not settle it.
    Inconclusive,
}

impl HypothesisState {
    /// One-word label, as it appears on the wire and in a saved session.
    pub const fn label(self) -> &'static str {
        match self {
            HypothesisState::Proposed => "proposed",
            HypothesisState::Supported => "supported",
            HypothesisState::Refuted => "refuted",
            HypothesisState::Inconclusive => "inconclusive",
        }
    }

    /// Inverse of [`HypothesisState::label`].
    ///
    /// The capitalised spelling is the `Debug` name, which earlier builds put
    /// on the wire; it is accepted so a session written by one of those builds
    /// still restores.
    pub fn from_label(s: &str) -> Option<Self> {
        Some(match s {
            "proposed" | "Proposed" => HypothesisState::Proposed,
            "supported" | "Supported" => HypothesisState::Supported,
            "refuted" | "Refuted" => HypothesisState::Refuted,
            "inconclusive" | "Inconclusive" => HypothesisState::Inconclusive,
            _ => return None,
        })
    }
}

/// Outcome of a single check, reported back to the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Verdict {
    /// The evidence supports the hypothesis.
    Confirmed,
    /// The evidence contradicts the hypothesis.
    Refuted,
    /// The evidence is insufficient either way.
    Inconclusive,
}

impl Verdict {
    /// One-word label.
    pub const fn label(self) -> &'static str {
        match self {
            Verdict::Confirmed => "confirmed",
            Verdict::Refuted => "refuted",
            Verdict::Inconclusive => "inconclusive",
        }
    }
}

/// Confidence at or above which a hypothesis is treated as supported.
pub const CONFIRM_THRESHOLD: f32 = 0.75;

/// Confidence at or below which counter-evidence refutes a hypothesis.
pub const REFUTE_THRESHOLD: f32 = 0.6;

/// Support below which counter-evidence is allowed to refute outright.
///
/// A statement with a plausible supporting observation and a weak counter is
/// unsettled, not disproved; only a near-absent support side lets the counter
/// evidence decide on its own.
pub const REFUTE_SUPPORT_CEILING: f32 = 0.4;

/// A caller-supplied assertion being tested against the image.
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// Numeric id, also the value in the `hyp:<n>` handle.
    pub id: u64,
    /// What the hypothesis is about.
    pub handle: Handle,
    /// Image the claim is about.
    ///
    /// An address alone does not identify a subject once more than one image is
    /// open: `fn:0x401000` exists in every PIE image. Recording the image with
    /// the claim keeps a later turn from resolving the same handle against a
    /// different binary and reading the answer as continuation.
    pub image: Option<u32>,
    /// The assertion, in the caller's own words.
    pub statement: String,
    /// Current standing.
    pub state: HypothesisState,
    /// Observations gathered so far.
    pub evidence: Vec<Evidence>,
    /// Derived from [`Hypothesis::evidence`].
    pub confidence: f32,
    /// Monotonic sequence number of creation, for ordering in a transcript.
    pub created_seq: u64,
    /// Monotonic sequence number of the last update.
    pub updated_seq: u64,
}

impl Hypothesis {
    /// File a new hypothesis.
    pub fn new(
        id: u64,
        handle: Handle,
        image: Option<u32>,
        statement: impl Into<String>,
        seq: u64,
    ) -> Self {
        Hypothesis {
            id,
            handle,
            image,
            statement: statement.into(),
            state: HypothesisState::Proposed,
            evidence: Vec::new(),
            confidence: 0.0,
            created_seq: seq,
            updated_seq: seq,
        }
    }

    /// Handle form of this hypothesis.
    pub fn handle(&self) -> Handle {
        Handle::hypothesis(self.id)
    }

    /// Attach an observation and re-evaluate.
    pub fn push(&mut self, evidence: Evidence, seq: u64) {
        self.evidence.push(evidence);
        self.updated_seq = seq;
        self.evaluate();
    }

    /// Recompute confidence and state.
    ///
    /// Support and counter-evidence are combined separately before being
    /// multiplied, because the product alone cannot tell three different
    /// situations apart: strong support, strong refutation, and a probe that
    /// simply found nothing either way. All three can produce the same number.
    pub fn evaluate(&mut self) {
        let (support, refute) = combine_parts(&self.evidence);
        self.confidence = combine(&self.evidence);

        self.state = if self.evidence.is_empty() {
            HypothesisState::Proposed
        } else if support >= CONFIRM_THRESHOLD && support > refute {
            HypothesisState::Supported
        } else if refute >= REFUTE_THRESHOLD && support < REFUTE_SUPPORT_CEILING {
            HypothesisState::Refuted
        } else {
            HypothesisState::Inconclusive
        };
    }

    /// Current standing as a verdict.
    pub const fn verdict(&self) -> Verdict {
        match self.state {
            HypothesisState::Supported => Verdict::Confirmed,
            HypothesisState::Refuted => Verdict::Refuted,
            HypothesisState::Proposed | HypothesisState::Inconclusive => Verdict::Inconclusive,
        }
    }

    /// Observations that argued against.
    pub fn counter_evidence(&self) -> impl Iterator<Item = &Evidence> {
        self.evidence.iter().filter(|e| e.is_counter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::{EvidenceKind, EvidenceSource};

    fn support(weight: f32) -> Evidence {
        Evidence::support(
            EvidenceKind::Bytes,
            EvidenceSource::Scanner,
            "match",
            weight,
        )
    }

    fn counter(weight: f32) -> Evidence {
        Evidence::refute(
            EvidenceKind::Absence,
            EvidenceSource::Heuristic,
            "missing",
            weight,
        )
    }

    fn fresh() -> Hypothesis {
        Hypothesis::new(
            1,
            Handle::function(crate::addr::Addr(0x4012a0)),
            None,
            "decodes base64",
            0,
        )
    }

    #[test]
    fn starts_proposed_with_zero_confidence() {
        let h = fresh();
        assert_eq!(h.state, HypothesisState::Proposed);
        assert_eq!(h.verdict(), Verdict::Inconclusive);
    }

    #[test]
    fn strong_support_confirms() {
        let mut h = fresh();
        h.push(support(0.8), 1);
        assert_eq!(h.state, HypothesisState::Supported);
        assert_eq!(h.verdict(), Verdict::Confirmed);
    }

    #[test]
    fn two_medium_signals_confirm() {
        let mut h = fresh();
        h.push(support(0.6), 1);
        assert_eq!(h.state, HypothesisState::Inconclusive);
        h.push(support(0.6), 2);
        // 0.84 >= 0.75
        assert_eq!(h.state, HypothesisState::Supported);
    }

    #[test]
    fn counter_evidence_alone_refutes() {
        let mut h = fresh();
        h.push(counter(0.9), 1);
        assert_eq!(h.state, HypothesisState::Refuted);
        assert_eq!(h.verdict(), Verdict::Refuted);
    }

    #[test]
    fn mixed_evidence_is_inconclusive() {
        let mut h = fresh();
        h.push(support(0.7), 1);
        h.push(counter(0.6), 2);
        // 0.7 * 0.4 = 0.28, above threshold but not confirmed
        assert_eq!(h.state, HypothesisState::Inconclusive);
    }

    #[test]
    fn handle_matches_id() {
        let h = fresh();
        assert_eq!(h.handle().to_string(), "hyp:1");
    }

    #[test]
    fn updated_seq_tracks_last_push() {
        let mut h = fresh();
        h.push(support(0.5), 7);
        assert_eq!(h.created_seq, 0);
        assert_eq!(h.updated_seq, 7);
    }
}
