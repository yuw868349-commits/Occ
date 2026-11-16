//! Byte-level statistics used to spot packing, compression and encrypted
//! regions.
//!
//! Entropy alone proves nothing: a `.rodata` full of float constants is as
//! close to uniform as a compressed blob. What makes the number useful is
//! where it sits. A section named `.text` at 7.98 bits per byte is a finding;
//! the same value in `.rodata` is Tuesday. The caller gets both, plus a
//! sliding window so a small encrypted stub inside an otherwise normal section
//! is visible instead of averaged away.

use crate::view::View;
use occ_core::SectionKind;

/// Shannon entropy of `data`, in bits per byte (`0.0..=8.0`).
///
/// Returns zero for empty input, which is the honest value: an empty stream
/// carries no information, and callers dividing by it should not have to
/// special-case `NaN`.
pub fn shannon(data: &[u8]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mut counts = [0u32; 256];
    for b in data {
        counts[*b as usize] += 1;
    }
    let total = data.len() as f32;
    let mut h = 0.0f32;
    for c in counts {
        if c > 0 {
            let p = c as f32 / total;
            h -= p * p.log2();
        }
    }
    h
}

/// Entropy and size of one section.
#[derive(Clone, Debug)]
pub struct SectionEntropy {
    /// Section name.
    pub name: String,
    /// Section kind, which decides how suspicious the value is.
    pub kind: SectionKind,
    /// Bytes measured.
    pub size: u64,
    /// Shannon entropy over those bytes.
    pub entropy: f32,
}

/// Entropy of every section with file content, highest first.
pub fn per_section(view: &View) -> Vec<SectionEntropy> {
    let mut out = Vec::new();
    for s in view.image().sections.iter() {
        if s.file_size == 0 {
            continue;
        }
        let Some(bytes) = view.bytes(s.addr, s.file_size as usize) else {
            continue;
        };
        out.push(SectionEntropy {
            name: s.name.clone(),
            kind: s.kind,
            size: s.file_size,
            entropy: shannon(bytes),
        });
    }
    out.sort_by(|a, b| b.entropy.total_cmp(&a.entropy));
    out
}

/// One window of a sliding entropy scan.
#[derive(Clone, Debug)]
pub struct Window {
    /// File offset where the window starts.
    pub offset: u64,
    /// Address the window starts at, when the region is mapped.
    pub addr: occ_core::Addr,
    /// Entropy in this window.
    pub entropy: f32,
}

/// Sliding-window entropy over a section, reporting only windows at or above
/// `threshold`.
///
/// Adjacent windows above the threshold are merged into runs by the caller;
/// this function reports every window so no boundary judgement is baked in.
pub fn sliding(view: &View, addr: occ_core::Addr, len: usize, window: usize, threshold: f32) -> Vec<Window> {
    if window == 0 {
        return Vec::new();
    }
    let Some(bytes) = view.bytes(addr, len) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + window <= bytes.len() {
        let h = shannon(&bytes[i..i + window]);
        if h >= threshold {
            out.push(Window {
                offset: i as u64,
                addr: addr.add(i as u64),
                entropy: h,
            });
        }
        i += window / 2;
    }
    out
}

/// Conventional threshold above which a section looks compressed or encrypted.
///
/// 7.2 rather than 7.5: real compressed data (deflate, LZMA) sits around
/// 7.4-7.99, and encrypted data at 7.99, while code sits near 6.0-6.6 and
/// ASCII text near 4.5. The gap is wide enough that precision is not the
/// issue; the issue is *which* section, which is why the caller gets the kind.
pub const HIGH_ENTROPY: f32 = 7.2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_bytes_have_zero_entropy() {
        assert_eq!(shannon(&[7u8; 1000]), 0.0);
        assert_eq!(shannon(&[]), 0.0);
    }

    #[test]
    fn all_byte_values_score_eight() {
        let data: Vec<u8> = (0..=255u8).collect();
        let h = shannon(&data);
        assert!((h - 8.0).abs() < 1e-4, "got {h}");
    }

    #[test]
    fn ascii_text_scores_low() {
        let text = b"the quick brown fox jumps over the lazy dog, repeatedly and at length";
        let h = shannon(text);
        assert!(h < 5.0, "got {h}");
    }

    #[test]
    fn two_symbols_score_one() {
        let data: Vec<u8> = (0..1024).map(|i| (i % 2) as u8).collect();
        let h = shannon(&data);
        assert!((h - 1.0).abs() < 1e-4, "got {h}");
    }

    #[test]
    fn window_of_zero_is_rejected() {
        let data = vec![0u8; 32];
        let img = crate::testutil::image_with_data(&data);
        let v = crate::view::View::new(&img, &data);
        assert!(sliding(&v, occ_core::Addr(0x1000), 32, 0, 0.0).is_empty());
    }
}
