//! Output budgeting.
//!
//! The binding constraint on a language model reading analysis output is its
//! context window, and it is not a soft constraint: exceeding it does not
//! degrade gracefully, it truncates mid-structure. Every tool that returns a
//! variable-size answer therefore takes a [`Depth`] and reports back how much
//! it actually produced.
//!
//! Three levels are enough. Summary answers "what is this"; Detail answers "how
//! do you know"; Raw answers "show me the bytes". A caller that needs more asks
//! again against a narrower handle, which is cheaper than raising the ceiling
//! and hoping.

/// How much detail to emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Depth {
    /// Enough to orient: identity, shape, and the single most telling fact.
    Summary,
    /// Enough to reason about: structure, evidence, and representative samples.
    #[default]
    Detail,
    /// Everything held, including raw bytes and complete listings.
    Raw,
}

impl Depth {
    /// Numeric limits implied by this level.
    pub const fn limits(self) -> RenderLimits {
        match self {
            Depth::Summary => RenderLimits {
                max_items: 12,
                max_bytes: 256,
                max_insns: 24,
                max_evidence: 4,
            },
            Depth::Detail => RenderLimits {
                max_items: 200,
                max_bytes: 8192,
                max_insns: 400,
                max_evidence: 24,
            },
            Depth::Raw => RenderLimits {
                max_items: usize::MAX,
                max_bytes: 1 << 20,
                max_insns: usize::MAX,
                max_evidence: usize::MAX,
            },
        }
    }

    /// Lower-case label used in payloads.
    pub const fn label(self) -> &'static str {
        match self {
            Depth::Summary => "summary",
            Depth::Detail => "detail",
            Depth::Raw => "raw",
        }
    }
}

/// Concrete ceilings derived from a [`Depth`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderLimits {
    /// Maximum collection entries to emit.
    pub max_items: usize,
    /// Maximum bytes of raw content to emit.
    pub max_bytes: usize,
    /// Maximum instructions to emit.
    pub max_insns: usize,
    /// Maximum evidence items to emit.
    pub max_evidence: usize,
}

impl RenderLimits {
    /// Clamp a requested count, reporting whether it was cut.
    pub fn take(&self, requested: usize) -> (usize, bool) {
        if requested > self.max_items {
            (self.max_items, true)
        } else {
            (requested, false)
        }
    }
}

/// Rough token estimate for a byte count.
///
/// English and hex-dump text average close to 3.5 bytes per token on BPE
/// tokenisers; structured JSON is denser, around 3.0. The estimate is used only
/// to tell a caller "this answer costs about N tokens", so being within 20% is
/// sufficient and a real tokeniser would be a dependency for no benefit.
pub fn estimate_tokens(bytes: usize) -> usize {
    bytes / 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depths_are_monotone() {
        let s = Depth::Summary.limits();
        let d = Depth::Detail.limits();
        let r = Depth::Raw.limits();
        assert!(s.max_insns < d.max_insns);
        assert!(d.max_insns <= r.max_insns);
        assert!(s.max_bytes < d.max_bytes);
        assert!(d.max_bytes < r.max_bytes);
    }

    #[test]
    fn take_reports_truncation() {
        let limits = Depth::Summary.limits();
        let (n, cut) = limits.take(5);
        assert_eq!(n, 5);
        assert!(!cut);
        let (n, cut) = limits.take(1_000);
        assert_eq!(n, 12);
        assert!(cut);
    }

    #[test]
    fn token_estimate_is_plausible() {
        assert_eq!(estimate_tokens(0), 0);
        assert_eq!(estimate_tokens(350), 116);
    }

    #[test]
    fn default_depth_is_detail() {
        assert_eq!(Depth::default(), Depth::Detail);
    }
}
