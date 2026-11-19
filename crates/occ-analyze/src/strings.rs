//! String extraction.
//!
//! A "string" here is a run of printable bytes long enough that its presence
//! is unlikely to be coincidence. The minimum default of four is what
//! `strings(1)` uses; below that, ordinary machine code produces false
//! positives at a rate that buries the useful results.
//!
//! UTF-16 is scanned separately rather than by decoding every byte pair. A
//! UTF-16LE string is mostly `ASCII, 0x00, ASCII, 0x00`; scanning it as ASCII
//! yields a run of length one per character, which is exactly why the two
//! passes do not interfere.

use crate::view::View;
use occ_core::Addr;

/// Character encoding of a found string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// One byte per character, printable ASCII.
    Ascii,
    /// Two bytes per character, low byte first.
    Utf16Le,
    /// Two bytes per character, high byte first.
    Utf16Be,
}

impl Encoding {
    /// Short label for reports.
    pub const fn label(self) -> &'static str {
        match self {
            Encoding::Ascii => "ascii",
            Encoding::Utf16Le => "utf16le",
            Encoding::Utf16Be => "utf16be",
        }
    }

    /// Bytes per character.
    pub const fn width(self) -> usize {
        match self {
            Encoding::Ascii => 1,
            Encoding::Utf16Le | Encoding::Utf16Be => 2,
        }
    }
}

/// Extraction parameters.
#[derive(Clone, Debug)]
pub struct Options {
    /// Minimum number of characters.
    pub min_len: usize,
    /// Also scan 16-bit encodings.
    pub utf16: bool,
    /// Include executable sections. Off by default: code is full of short
    /// printable runs (`mov`, `push`) that are not strings.
    pub include_code: bool,
    /// Stop after this many strings.
    pub max: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            min_len: 4,
            utf16: true,
            include_code: false,
            max: 100_000,
        }
    }
}

/// One extracted string.
#[derive(Clone, Debug)]
pub struct Found {
    /// Address of the first character.
    pub addr: Addr,
    /// Decoded text.
    pub text: String,
    /// Encoding it was found in.
    pub encoding: Encoding,
    /// Length in bytes, including the terminator when one was seen.
    pub byte_len: u64,
    /// Containing section name.
    pub section: String,
    /// True when the run ended at a NUL rather than at the end of the section.
    pub terminated: bool,
}

impl Found {
    /// Whether the text looks like a filesystem path or URL, which is usually
    /// worth surfacing first.
    pub fn looks_like_path(&self) -> bool {
        self.text.starts_with('/')
            || self.text.starts_with("http://")
            || self.text.starts_with("https://")
            || self.text.contains(":\\")
    }

    /// Whether the text looks like a printf-style format string.
    pub fn looks_like_format(&self) -> bool {
        let bytes = self.text.as_bytes();
        let mut i = 0;
        while i + 1 < bytes.len() {
            if bytes[i] == b'%' && bytes[i + 1] != b'%' {
                return true;
            }
            i += 1;
        }
        false
    }
}

/// Extract strings from every eligible section.
pub fn scan(view: &View, opts: &Options) -> Vec<Found> {
    let mut out = Vec::new();
    for s in view.image().sections.iter() {
        if s.file_size == 0 {
            continue;
        }
        if s.is_code() && !opts.include_code {
            continue;
        }
        let Some(bytes) = view.bytes(s.addr, s.file_size as usize) else {
            continue;
        };
        scan_ascii(bytes, s.addr, &s.name, opts, &mut out);
        if opts.utf16 {
            scan_utf16(bytes, s.addr, &s.name, opts, &mut out);
        }
        if out.len() >= opts.max {
            out.truncate(opts.max);
            break;
        }
    }
    out.sort_by_key(|f| f.addr.0);
    out
}

fn printable(b: u8) -> bool {
    (0x20..0x7f).contains(&b) || b == b'\t'
}

fn scan_ascii(bytes: &[u8], base: Addr, section: &str, opts: &Options, out: &mut Vec<Found>) {
    let mut start = None;
    for (i, b) in bytes.iter().enumerate() {
        if printable(*b) {
            if start.is_none() {
                start = Some(i);
            }
            continue;
        }
        if let Some(s) = start.take() {
            let len = i - s;
            if len >= opts.min_len {
                push(
                    out,
                    base.add(s as u64),
                    &bytes[s..i],
                    Encoding::Ascii,
                    s,
                    len,
                    *b == 0,
                    section,
                );
            }
        }
    }
    if let Some(s) = start {
        let len = bytes.len() - s;
        if len >= opts.min_len {
            push(
                out,
                base.add(s as u64),
                &bytes[s..],
                Encoding::Ascii,
                s,
                len,
                false,
                section,
            );
        }
    }
}

fn scan_utf16(bytes: &[u8], base: Addr, section: &str, opts: &Options, out: &mut Vec<Found>) {
    for (le, enc) in [(true, Encoding::Utf16Le), (false, Encoding::Utf16Be)] {
        let mut i = 0usize;
        let mut start = None;
        while i + 1 < bytes.len() {
            let (lo, hi) = if le {
                (bytes[i], bytes[i + 1])
            } else {
                (bytes[i + 1], bytes[i])
            };
            let ok = hi == 0 && printable(lo);
            if ok {
                if start.is_none() {
                    start = Some(i);
                }
            } else if let Some(s) = start.take() {
                let chars = (i - s) / 2;
                if chars >= opts.min_len {
                    let text: String = (s..i)
                        .step_by(2)
                        .map(|k| (if le { bytes[k] } else { bytes[k + 1] }) as char)
                        .collect();
                    out.push(Found {
                        addr: base.add(s as u64),
                        text,
                        encoding: enc,
                        byte_len: (i - s) as u64,
                        section: section.to_string(),
                        terminated: hi == 0 && lo == 0,
                    });
                }
            }
            i += 2;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push(
    out: &mut Vec<Found>,
    addr: Addr,
    raw: &[u8],
    encoding: Encoding,
    _offset: usize,
    len: usize,
    terminated: bool,
    section: &str,
) {
    if let Ok(text) = std::str::from_utf8(raw) {
        out.push(Found {
            addr,
            text: text.to_string(),
            encoding,
            byte_len: len as u64 + u64::from(terminated),
            section: section.to_string(),
            terminated,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::image_with_data;

    fn scan_bytes(data: &[u8]) -> Vec<Found> {
        let img = image_with_data(data);
        let v = View::new(&img, data);
        scan(&v, &Options::default())
    }

    #[test]
    fn finds_nul_terminated_ascii() {
        let mut data = vec![0u8; 64];
        data[0..6].copy_from_slice(b"hello\0");
        let found = scan_bytes(&data);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].text, "hello");
        assert_eq!(found[0].addr, Addr(0x1000));
        assert!(found[0].terminated);
    }

    #[test]
    fn respects_minimum_length() {
        let mut data = vec![0u8; 32];
        data[0..3].copy_from_slice(b"abc");
        assert!(scan_bytes(&data).is_empty());
        data[0..4].copy_from_slice(b"abcd");
        assert_eq!(scan_bytes(&data).len(), 1);
    }

    #[test]
    fn ignores_high_bytes() {
        let data = vec![0xff; 64];
        assert!(scan_bytes(&data).is_empty());
    }

    #[test]
    fn finds_utf16le() {
        let mut data = vec![0u8; 64];
        for (i, c) in "wide string".chars().enumerate() {
            data[i * 2] = c as u8;
        }
        let found = scan_bytes(&data);
        let wide: Vec<_> = found
            .iter()
            .filter(|f| f.encoding == Encoding::Utf16Le)
            .collect();
        assert_eq!(wide.len(), 1, "got {found:?}");
        assert_eq!(wide[0].text, "wide string");
    }

    #[test]
    fn classifies_paths_and_formats() {
        let mut data = vec![0u8; 64];
        data[0..12].copy_from_slice(b"/etc/passwd\0");
        data[16..24].copy_from_slice(b"%s: %d\0\0");
        let found = scan_bytes(&data);
        assert!(found.iter().any(|f| f.looks_like_path()));
        assert!(found.iter().any(|f| f.looks_like_format()));
    }
}
