//! The analysis facade.
//!
//! [`Analysis::build`] runs recovery, string extraction, fingerprinting and
//! cross-reference indexing exactly once, then exposes O(log n) lookups. Every
//! later question — "what is at this address", "who calls it", "what strings
//! does it touch" — is answered from the stored result, so a session that keeps
//! one `Analysis` alive pays the decode cost once per image instead of once
//! per question.
//!
//! The struct is deliberately plain data plus indices. It can be rebuilt from
//! the same bytes and will make the same decisions, which is what makes caching
//! it across turns of a conversation safe.

use std::collections::HashMap;

use occ_core::{Addr, Fingerprint, Function, Image, Instruction};

use crate::disasm::{RecoverOptions, attach_strings, recover};
use crate::entropy::{SectionEntropy, per_section};
use crate::fingerprint::fingerprint;
use crate::strings::{Found, Options as StringOptions, scan};
use crate::view::View;
use crate::xref::XrefTable;

/// What to compute.
#[derive(Clone, Debug)]
pub struct AnalysisOptions {
    /// Recovery limits.
    pub recover: RecoverOptions,
    /// String extraction parameters.
    pub strings: StringOptions,
    /// Extract strings at all. Turning this off saves a full pass over the
    /// read-only data for callers that only need code structure.
    pub scan_strings: bool,
    /// Compute the toolchain fingerprint.
    pub fingerprint: bool,
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        AnalysisOptions {
            recover: RecoverOptions::default(),
            strings: StringOptions::default(),
            scan_strings: true,
            fingerprint: true,
        }
    }
}

/// How much of the executable content was accounted for.
#[derive(Clone, Copy, Debug, Default)]
pub struct Coverage {
    /// Executable bytes declared by sections.
    pub code_bytes: u64,
    /// Bytes covered by instructions the decoder understood, unioned across
    /// functions.
    ///
    /// This, not [`decoded_bytes`](Self::decoded_bytes), is what "code
    /// decoded" means. A byte the decoder had to emit as `.word` was read but
    /// not understood: counting it as decoded let a NEON kernel report 100%
    /// while five of its six instructions were data directives.
    ///
    /// It is also not the space the functions claim: a body can jump over a
    /// chunk the recovery never followed, and counting the range would report
    /// those bytes as decoded when nothing reads them.
    pub insn_bytes: u64,
    /// Bytes covered by data directives the decoder emitted where it could
    /// not decode an instruction.
    ///
    /// These bytes are accounted for — the stream stayed aligned — but no
    /// instruction was understood, so they are reported separately instead of
    /// inflating the decoded fraction.
    pub unparsed_bytes: u64,
    /// Bytes covered by any decoded unit: instructions plus data directives.
    ///
    /// `decoded_bytes == insn_bytes + unparsed_bytes`; overlaps between the
    /// two are resolved in favour of `insn_bytes`.
    pub decoded_bytes: u64,
    /// Bytes inside a recovered function's range, holes included.
    pub covered_bytes: u64,
    /// Functions recovered.
    pub functions: usize,
    /// Instructions decoded.
    pub insns: usize,
    /// Basic blocks built.
    pub blocks: usize,
}

impl Coverage {
    /// Fraction of executable bytes the decoder understood, `0.0..=1.0`.
    ///
    /// Data directives do not count: a `.word` is a byte that was skipped, not
    /// one that was decoded.
    pub fn fraction(&self) -> f32 {
        if self.code_bytes == 0 {
            0.0
        } else {
            (self.insn_bytes as f32 / self.code_bytes as f32).min(1.0)
        }
    }

    /// Fraction of executable bytes that any decoded unit covers, including
    /// data directives, `0.0..=1.0`.
    ///
    /// This answers "did the recovery walk these bytes", not "did it
    /// understand them".
    pub fn accounted_fraction(&self) -> f32 {
        if self.code_bytes == 0 {
            0.0
        } else {
            (self.decoded_bytes as f32 / self.code_bytes as f32).min(1.0)
        }
    }
}

/// The analysis result for one image.
pub struct Analysis {
    /// Recovered functions, sorted by entry.
    pub functions: Vec<Function>,
    /// Cross references.
    pub xrefs: XrefTable,
    /// Extracted strings, sorted by address.
    pub strings: Vec<Found>,
    /// PLT stubs and the imports they forward to.
    pub thunks: Vec<(Addr, String)>,
    /// Coverage statistics.
    pub coverage: Coverage,
    /// Per-section entropy, highest first.
    pub entropy: Vec<SectionEntropy>,
    /// Toolchain fingerprint.
    pub fingerprint: Fingerprint,
    /// Notes about decisions that shaped the result.
    pub notes: Vec<String>,
    index: HashMap<u64, usize>,
    string_at: HashMap<u64, usize>,
}

impl Analysis {
    /// Run every analysis pass over `view`.
    pub fn build(view: &View, opts: &AnalysisOptions) -> Analysis {
        let image = view.image();
        let rec = recover(view, &opts.recover);

        let strings = if opts.scan_strings {
            scan(view, &opts.strings)
        } else {
            Vec::new()
        };
        let mut by_addr: HashMap<u64, String> = HashMap::with_capacity(strings.len());
        for s in &strings {
            by_addr.insert(s.addr.0, s.text.clone());
        }

        let mut functions = rec.functions;
        attach_strings(&mut functions, &by_addr);

        let mut index = HashMap::with_capacity(functions.len());
        for (i, f) in functions.iter().enumerate() {
            index.insert(f.entry.0, i);
        }
        let mut string_at = HashMap::with_capacity(strings.len());
        for (i, s) in strings.iter().enumerate() {
            string_at.insert(s.addr.0, i);
        }

        // Union, not sum: two functions that overlap (a shared tail, a
        // mis-recovered bound) must not push coverage past 100%. Instructions
        // and data directives are counted separately so that a `.word` cannot
        // pass for understanding.
        let (insn_bytes, unparsed_bytes) = byte_accounting(&functions);
        let decoded_bytes = insn_bytes + unparsed_bytes;
        let covered_bytes = union_len(&functions);
        let blocks = functions.iter().map(|f| f.blocks.len()).sum();
        let insns = functions.iter().map(|f| f.insns.len()).sum();

        let fp = if opts.fingerprint {
            fingerprint(image, view)
        } else {
            image.fingerprint.clone()
        };

        Analysis {
            coverage: Coverage {
                code_bytes: rec.code_bytes,
                insn_bytes,
                unparsed_bytes,
                decoded_bytes,
                covered_bytes,
                functions: functions.len(),
                insns,
                blocks,
            },
            entropy: per_section(view),
            fingerprint: fp,
            functions,
            xrefs: XrefTable::new(rec.xrefs),
            strings,
            thunks: rec.thunks,
            notes: rec.notes,
            index,
            string_at,
        }
    }

    /// Function whose entry is exactly `addr`.
    pub fn function_at(&self, addr: Addr) -> Option<&Function> {
        self.index.get(&addr.0).map(|i| &self.functions[*i])
    }

    /// Function whose body contains `addr`.
    ///
    /// Bodies can overlap after a bad recovery; the smallest containing range
    /// wins, because the tightest match is the most specific claim.
    pub fn function_covering(&self, addr: Addr) -> Option<&Function> {
        let mut best: Option<&Function> = None;
        for f in &self.functions {
            if addr.0 >= f.entry.0 && addr.0 < f.entry.0 + f.size {
                match best {
                    Some(b) if b.size <= f.size => {}
                    _ => best = Some(f),
                }
            }
        }
        best
    }

    /// Function by exact symbol name, then by display name.
    pub fn function_by_name(&self, name: &str) -> Option<&Function> {
        self.functions
            .iter()
            .find(|f| f.name.as_deref() == Some(name))
            .or_else(|| self.functions.iter().find(|f| f.display_name() == name))
    }

    /// String starting exactly at `addr`.
    pub fn string_at(&self, addr: Addr) -> Option<&Found> {
        self.string_at.get(&addr.0).map(|i| &self.strings[*i])
    }

    /// String containing `addr`, when the address falls inside its bytes.
    pub fn string_covering(&self, addr: Addr) -> Option<&Found> {
        if let Some(s) = self.string_at(addr) {
            return Some(s);
        }
        self.strings
            .iter()
            .find(|s| addr.within(s.addr, s.byte_len))
    }

    /// Instruction at `addr`, searching the containing function.
    pub fn insn_at(&self, addr: Addr) -> Option<&Instruction> {
        self.function_covering(addr).and_then(|f| f.insn_at(addr))
    }

    /// Import name a call target resolves to, following thunks.
    pub fn import_for(&self, addr: Addr, image: &Image) -> Option<String> {
        if let Some((_, name)) = self.thunks.iter().find(|(a, _)| *a == addr) {
            return Some(name.clone());
        }
        image.import_by_slot(addr).map(|i| i.name.clone())
    }

    /// Functions that nothing in the image calls directly.
    ///
    /// These are entry points, exported API, or code reached only through
    /// pointers — in a stripped binary, the first place to look.
    pub fn roots(&self) -> Vec<Addr> {
        self.functions
            .iter()
            .filter(|f| f.callers.is_empty())
            .map(|f| f.entry)
            .collect()
    }

    /// Functions sorted by a structural size measure, largest first.
    pub fn largest(&self, n: usize) -> Vec<&Function> {
        let mut v: Vec<&Function> = self.functions.iter().collect();
        v.sort_by_key(|f| std::cmp::Reverse(f.insns.len()));
        v.truncate(n);
        v
    }

    /// One-line summary used in reports.
    pub fn summary(&self) -> String {
        format!(
            "{} functions, {} instructions, {} blocks, {:.1}% of {} code bytes",
            self.coverage.functions,
            self.coverage.insns,
            self.coverage.blocks,
            self.coverage.fraction() * 100.0,
            self.coverage.code_bytes
        )
    }
}

/// Total bytes covered by the union of function ranges.
fn union_len(functions: &[Function]) -> u64 {
    let mut ranges: Vec<(u64, u64)> = functions
        .iter()
        .map(|f| (f.entry.0, f.entry.0 + f.size))
        .filter(|(a, b)| b > a)
        .collect();
    ranges.sort_unstable();
    span_union(ranges)
}

/// Total bytes covered by the union of decoded units, split by whether the
/// decoder understood them.
///
/// A function's range can be larger than the instructions inside it, so this is
/// the only count that answers "was this byte decoded". Overlaps are unioned
/// for the same reason as above: two functions that share a tail are one run of
/// bytes, not two.
///
/// Data directives — `.byte`, `.word`, and the other forms the decoders emit
/// when an encoding is not understood — cover bytes without decoding anything.
/// They are counted in `unparsed_bytes`, never in `insn_bytes`. When a byte is
/// covered by both a real instruction and a directive (one function sharing a
/// tail with another that mis-read it as data), the instruction wins: something
/// did understand that byte.
///
/// Returns `(instruction bytes, data-directive bytes)`.
fn byte_accounting(functions: &[Function]) -> (u64, u64) {
    // (start, end, is_data)
    let mut ranges: Vec<(u64, u64, bool)> = functions
        .iter()
        .flat_map(|f| {
            f.insns.iter().map(|i| {
                (
                    i.addr.0,
                    i.end().0,
                    i.mnemonic.starts_with('.'),
                )
            })
        })
        .filter(|(a, b, _)| b > a)
        .collect();
    if ranges.is_empty() {
        return (0, 0);
    }
    ranges.sort_unstable_by_key(|r| r.0);

    // Sweep the endpoints. Depths are per kind; a byte is understood when at
    // least one instruction covers it, whatever else also covers it.
    let mut events: Vec<(u64, i32, i32)> = Vec::with_capacity(ranges.len() * 2);
    for (a, b, is_data) in ranges {
        if is_data {
            events.push((a, 1, 0));
            events.push((b, -1, 0));
        } else {
            events.push((a, 0, 1));
            events.push((b, 0, -1));
        }
    }
    events.sort_unstable_by_key(|e| e.0);

    let mut insn_depth = 0i64;
    let mut data_depth = 0i64;
    let mut understood = 0u64;
    let mut unparsed = 0u64;
    let mut prev = events[0].0;
    let mut i = 0;
    while i < events.len() {
        let at = events[i].0;
        if at > prev {
            let len = at - prev;
            if insn_depth > 0 {
                understood += len;
            } else if data_depth > 0 {
                unparsed += len;
            }
        }
        while i < events.len() && events[i].0 == at {
            data_depth += i64::from(events[i].1);
            insn_depth += i64::from(events[i].2);
            i += 1;
        }
        prev = at;
    }
    (understood, unparsed)
}

fn span_union(ranges: Vec<(u64, u64)>) -> u64 {
    let mut total = 0u64;
    let mut cur: Option<(u64, u64)> = None;
    for (a, b) in ranges {
        match cur {
            None => cur = Some((a, b)),
            Some((ca, cb)) if a <= cb => cur = Some((ca, cb.max(b))),
            Some((ca, cb)) => {
                total += cb - ca;
                cur = Some((a, b));
            }
        }
    }
    if let Some((a, b)) = cur {
        total += b - a;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::image_with_code;

    #[test]
    fn builds_an_empty_analysis_for_a_non_code_image() {
        let data: Vec<u8> = Vec::new();
        let img = image_with_code(&data);
        let v = View::new(&img, &data);
        let a = Analysis::build(&v, &AnalysisOptions::default());
        assert_eq!(a.coverage.functions, 0);
        assert!(a.functions.is_empty());
    }
}
