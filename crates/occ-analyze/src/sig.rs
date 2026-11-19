//! Function signatures: naming library code in a stripped image.
//!
//! A stripped static executable is mostly a library. Every function is
//! `sub_4f3a20`, and the reader has no way to tell `memcpy` from `__printf_fp`
//! without reading two thousand bodies by hand. The names are gone; the bytes
//! are not.
//!
//! This module is the FLIRT idea without the vendor format: a signature is a
//! prefix of a function's machine code plus a mask that says which bytes must
//! match. Learn signatures from an image that still has its symbol table, apply
//! them to the same code stripped, and the names come back.
//!
//! ```text
//! occ> binary.signatures action=learn path=/usr/lib/libc.a   -> libc.sig
//! occ> binary.signatures action=load  path=libc.sig          (then called for
//!                                                             every image)
//! ```
//!
//! # Why a mask
//!
//! Two builds of one function differ exactly where the linker had a choice:
//! a relative call, an absolute address in a `mov`, a stack offset that moved
//! because one local grew. Masking those bytes lets one signature cover both
//! builds. Masking is per byte and comes from the file, so a signature that was
//! learned from one binary and applied to another says what it compared.
//!
//! # What it refuses to do
//!
//! * It never renames a function that already has a name.
//! * When two signatures of equal length match one function, it names nothing
//!   and records the collision. A wrong name is worse than `sub_4f3a20`: the
//!   reader stops looking.
//! * It does not guess at a partial match. A signature either matches its
//!   whole prefix or it does not match.

use std::collections::HashMap;
use std::fmt::Write as _;

use occ_core::{Image, SymbolKind};

use crate::analysis::Analysis;
use crate::view::View;

/// Bytes a learned signature covers.
///
/// Thirty-two, not sixteen. Sixteen is enough to recognise a function by eye
/// and not enough to tell two apart: a static link holds hundreds of bodies
/// that open with the same `endbr64; push r15; push r14` and diverge later.
/// With a sixteen-byte prefix, learning from one build and applying to the
/// same build stripped produced 468 collisions against 580 names — the
/// database was naming almost nothing it could not also have named wrongly.
pub const DEFAULT_LEN: usize = 32;

/// Shortest prefix worth keeping.
pub const MIN_LEN: usize = 6;

/// One function signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    /// Name to give a matching function.
    pub name: String,
    /// Bytes of the function's first instructions.
    pub bytes: Vec<u8>,
    /// One flag per byte: `true` when it must match, `false` for a wildcard.
    pub mask: Vec<bool>,
}

impl Signature {
    /// Learn a signature from the bytes at the start of a function.
    ///
    /// `code` is the function's first bytes; `len` is how many of them to keep.
    /// Returns `None` when there are too few bytes to say anything.
    pub fn learn(name: &str, code: &[u8], len: usize) -> Option<Signature> {
        if name.is_empty() || code.len() < MIN_LEN.min(len) {
            return None;
        }
        let len = len.min(code.len());
        if len < MIN_LEN {
            return None;
        }
        Some(Signature {
            name: name.to_string(),
            bytes: code[..len].to_vec(),
            mask: vec![true; len],
        })
    }

    /// Length of the prefix.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// True when the signature has no bytes.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// True when `code` starts with this signature.
    pub fn matches(&self, code: &[u8]) -> bool {
        if code.len() < self.bytes.len() {
            return false;
        }
        self.bytes
            .iter()
            .zip(&self.mask)
            .zip(code)
            .all(|((&b, &m), &c)| !m || b == c)
    }

    /// Wildcard every byte of the instructions that name an address.
    ///
    /// A call or a RIP-relative load bakes a link-time address into the
    /// instruction; the same source compiled into a different binary has a
    /// different one. Masking them is what makes a learned signature survive
    /// being applied to another build. `spans` lists the byte ranges to clear,
    /// as offsets into the prefix.
    pub fn wildcard(&mut self, spans: &[(usize, usize)]) {
        for &(at, len) in spans {
            for i in at..(at + len).min(self.mask.len()) {
                self.mask[i] = false;
            }
        }
    }
}

/// A set of signatures.
#[derive(Clone, Debug, Default)]
pub struct SignatureDb {
    /// Entries, in the order they were learned.
    pub entries: Vec<Signature>,
    /// Index by first four bytes, so a match does not compare every signature.
    index: HashMap<[u8; 4], Vec<usize>>,
}

impl SignatureDb {
    /// An empty database.
    pub fn new() -> SignatureDb {
        SignatureDb::default()
    }

    /// Number of signatures.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when there is nothing to apply.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Add a signature, replacing one with the same name and length.
    pub fn insert(&mut self, sig: Signature) {
        if let Some(existing) = self
            .entries
            .iter_mut()
            .find(|s| s.name == sig.name && s.len() == sig.len())
        {
            *existing = sig;
            return;
        }
        let key = key_of(&sig.bytes);
        self.index.entry(key).or_default().push(self.entries.len());
        self.entries.push(sig);
    }

    /// Learn a database from an image that still has names.
    ///
    /// Only defined functions with a name and a body are considered. A symbol
    /// that names data, an undefined import, or a zero-length entry says
    /// nothing about what a body looks like.
    pub fn learn(image: &Image, view: &View, len: usize) -> SignatureDb {
        let mut db = SignatureDb::new();
        // One signature per address. A static libc puts several names on one
        // body — `lseek`, `lseek64`, `__libc_lseek` are one function — and
        // learning all of them would make every one of them a collision with
        // the others, which is a fact about the symbol table and not about
        // the code.
        let mut seen: HashMap<u64, ()> = HashMap::new();
        let mut bodies: Vec<&[u8]> = Vec::new();
        for sym in &image.symbols {
            if !sym.defined || sym.kind != SymbolKind::Function || sym.name.is_empty() {
                continue;
            }
            if seen.insert(sym.addr.0, ()).is_some() {
                continue;
            }
            // A symbol without a size still has code; the next symbol's address
            // bounds it. Reading past the end of the section is not possible,
            // so a read of `len` bytes is safe either way.
            // The symbol's own size is the honest bound: a signature longer
            // than the function would compare the next function's bytes and
            // claim a match that the code does not support.
            let want = if sym.size >= MIN_LEN as u64 {
                (sym.size as usize).min(len)
            } else {
                len
            };
            let Some(code) = view.bytes(sym.addr, want) else {
                continue;
            };
            if let Some(sig) = Signature::learn(&sym.name, code, want) {
                db.insert(sig);
            }
            if let Some(body) = view.bytes(sym.addr, 64) {
                bodies.push(body);
            }
        }
        db.prune_ambiguous(&bodies);
        db
    }

    /// Drop every signature that cannot name a function uniquely.
    ///
    /// Library code contains bodies that are byte-identical apart from the
    /// name: `_IO_puts.cold`, `_IO_fclose.cold` and friends are one `call
    /// abort` each, and no prefix can tell them apart. Keeping all of them
    /// makes every later match a collision, and picking one of the names
    /// would be a wrong answer. Removing the whole conflicting set is the
    /// only honest option: those functions stay unnamed, and the signatures
    /// that are actually distinctive name everything else.
    ///
    /// The comparison is run against the bodies the database was learned
    /// from, using the same rule `match_all` applies at call time, so a
    /// conflict that would be reported later is removed here instead.
    fn prune_ambiguous(&mut self, bodies: &[&[u8]]) {
        let mut doomed: Vec<bool> = vec![false; self.entries.len()];
        for body in bodies {
            let matching: Vec<usize> = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, s)| s.matches(body))
                .map(|(i, _)| i)
                .collect();
            if matching.len() < 2 {
                continue;
            }
            let best = matching.iter().map(|&i| self.entries[i].len()).max().unwrap_or(0);
            let winners: Vec<usize> = matching
                .into_iter()
                .filter(|&i| self.entries[i].len() == best)
                .collect();
            let mut names: Vec<&str> = winners.iter().map(|&i| self.entries[i].name.as_str()).collect();
            names.sort_unstable();
            names.dedup();
            if names.len() > 1 {
                for i in winners {
                    doomed[i] = true;
                }
            }
        }
        if doomed.iter().any(|&d| d) {
            self.entries = std::mem::take(&mut self.entries)
                .into_iter()
                .enumerate()
                .filter(|(i, _)| !doomed[*i])
                .map(|(_, s)| s)
                .collect();
            self.reindex();
        }
    }

    /// Rebuild the first-four-bytes index after the entry list changed.
    fn reindex(&mut self) {
        self.index.clear();
        for i in 0..self.entries.len() {
            let key = key_of(&self.entries[i].bytes);
            self.index.entry(key).or_default().push(i);
        }
    }

    /// Match every unnamed function against the database.
    ///
    /// The names are returned, not written: a caller that holds an analysis
    /// behind a lock or a shared pointer needs the answer without a mutable
    /// borrow, and a caller that owns one can use [`apply`](Self::apply).
    pub fn match_all(&self, analysis: &Analysis, view: &View) -> Applied {
        let mut applied = Applied::default();
        if self.is_empty() {
            return applied;
        }
        for f in &analysis.functions {
            if f.name.is_some() || f.size == 0 {
                continue;
            }
            let Some(head) = view.bytes(f.entry, 4) else {
                continue;
            };
            let mut key = [0u8; 4];
            key.copy_from_slice(head);
            let Some(candidates) = self.index.get(&key) else {
                continue;
            };
            // A body at the end of a section is shorter than the read; fall
            // back rather than skipping it, and let `matches` reject a
            // signature longer than what could be read.
            let Some(code) = view.bytes(f.entry, 64).or_else(|| view.bytes(f.entry, DEFAULT_LEN))
            else {
                continue;
            };
            let mut matches: Vec<&Signature> = candidates
                .iter()
                .map(|&i| &self.entries[i])
                .filter(|s| s.matches(code))
                .collect();
            if matches.is_empty() {
                continue;
            }
            // The longest prefix is the most specific claim. Two signatures of
            // the same length that both match cannot both be right, and
            // picking either one would put a wrong name on the function.
            matches.sort_by_key(|s| std::cmp::Reverse(s.len()));
            let best = matches[0].len();
            let winners: Vec<&&Signature> = matches.iter().filter(|s| s.len() == best).collect();
            if winners.len() > 1 {
                applied.collisions.push(Collision {
                    entry: f.entry,
                    names: winners.iter().map(|s| s.name.clone()).collect(),
                });
                continue;
            }
            applied.renames.push((f.entry, winners[0].name.clone()));
        }
        applied.named = applied.renames.len();
        applied
    }

    /// Name the unnamed functions in `analysis`.
    ///
    /// Returns what happened, including the collisions that were refused.
    pub fn apply(&self, analysis: &mut Analysis, view: &View) -> Applied {
        let applied = self.match_all(analysis, view);
        for (addr, name) in &applied.renames {
            if let Some(f) = analysis.functions.iter_mut().find(|f| f.entry == *addr) {
                f.name = Some(name.clone());
            }
        }
        applied
    }

    /// Serialise to the text format.
    ///
    /// Text, not a binary blob: a signature file is a claim about code, and a
    /// reader should be able to see what it claims and edit it.
    pub fn to_text(&self) -> String {
        let mut out = String::with_capacity(self.entries.len() * 96);
        out.push_str("# occ signatures v1\n");
        out.push_str("# name\tbytes-hex\tmask-bits\n");
        for s in &self.entries {
            let mut hex = String::with_capacity(s.len() * 2);
            let mut mask = String::with_capacity(s.len());
            for (b, m) in s.bytes.iter().zip(&s.mask) {
                let _ = write!(hex, "{b:02x}");
                mask.push(if *m { '1' } else { '0' });
            }
            // A name with a tab or newline would break the format; the symbol
            // table cannot produce one, but a hand-written file can.
            let name = s.name.replace(['\t', '\n'], " ");
            let _ = writeln!(out, "{name}\t{hex}\t{mask}");
        }
        out
    }

    /// Parse the text format.
    pub fn parse(text: &str) -> Result<SignatureDb, String> {
        let mut db = SignatureDb::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split('\t');
            let (Some(name), Some(hex), Some(mask)) = (parts.next(), parts.next(), parts.next())
            else {
                return Err(format!("line {}: expected name, bytes and mask", n + 1));
            };
            let bytes = unhex(hex).ok_or_else(|| format!("line {}: bytes are not hex", n + 1))?;
            if mask.len() != bytes.len() || !mask.bytes().all(|c| c == b'0' || c == b'1') {
                return Err(format!(
                    "line {}: mask must be one 0/1 per byte",
                    n + 1
                ));
            }
            let mask: Vec<bool> = mask.bytes().map(|c| c == b'1').collect();
            if bytes.len() < MIN_LEN {
                return Err(format!("line {}: signature shorter than {MIN_LEN} bytes", n + 1));
            }
            db.insert(Signature {
                name: name.to_string(),
                bytes,
                mask,
            });
        }
        Ok(db)
    }
}

/// What applying a database did.
#[derive(Clone, Debug, Default)]
pub struct Applied {
    /// Functions that received a name, as (entry, name).
    pub renames: Vec<(occ_core::Addr, String)>,
    /// Functions that received a name.
    pub named: usize,
    /// Functions that several signatures of equal length claimed.
    pub collisions: Vec<Collision>,
}

/// One function that several signatures claimed.
#[derive(Clone, Debug)]
pub struct Collision {
    /// The function.
    pub entry: occ_core::Addr,
    /// Names that matched equally well.
    pub names: Vec<String>,
}

fn key_of(bytes: &[u8]) -> [u8; 4] {
    let mut key = [0u8; 4];
    let n = bytes.len().min(4);
    key[..n].copy_from_slice(&bytes[..n]);
    key
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let b = s.as_bytes();
    for i in (0..b.len()).step_by(2) {
        let hi = (b[i] as char).to_digit(16)?;
        let lo = (b[i + 1] as char).to_digit(16)?;
        out.push((hi * 16 + lo) as u8);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_signature_matches_the_bytes_it_was_learned_from() {
        let code = [0x55, 0x48, 0x89, 0xe5, 0x48, 0x83, 0xec, 0x10];
        let sig = Signature::learn("f", &code, 8).unwrap();
        assert!(sig.matches(&code));
        assert!(sig.matches(&[&code[..], &[0xff, 0xff]].concat()));
    }

    #[test]
    fn a_shorter_candidate_does_not_match() {
        let sig = Signature::learn("f", &[0u8; 8], 8).unwrap();
        assert!(!sig.matches(&[0u8; 7]));
    }

    #[test]
    fn a_masked_byte_can_differ() {
        let code = [0x55, 0x48, 0x89, 0xe5, 0xe8, 0x10, 0x00, 0x00, 0x00, 0xc3];
        let mut sig = Signature::learn("f", &code, 10).unwrap();
        // The call displacement is the last four bytes of the `e8` instruction.
        sig.wildcard(&[(5, 4)]);
        let mut other = code;
        other[5..9].copy_from_slice(&[0x99, 0x88, 0x77, 0x66]);
        assert!(sig.matches(&other));
        // What is not masked still has to match.
        other[4] = 0x90;
        assert!(!sig.matches(&other));
    }

    #[test]
    fn text_round_trips() {
        let mut db = SignatureDb::new();
        db.insert(Signature::learn("memcpy", &[1, 2, 3, 4, 5, 6, 7, 8], 8).unwrap());
        let mut wild = Signature::learn("memset", &[1, 2, 3, 4, 5, 6, 7, 8], 8).unwrap();
        wild.wildcard(&[(2, 2)]);
        db.insert(wild);
        let text = db.to_text();
        let back = SignatureDb::parse(&text).expect("parse");
        assert_eq!(back.len(), 2);
        assert_eq!(back.entries[0].bytes, db.entries[0].bytes);
        assert_eq!(back.entries[1].mask, db.entries[1].mask);
    }

    #[test]
    fn a_broken_line_is_refused_with_a_line_number() {
        let e = SignatureDb::parse("ok\t414243444546\t111111\nbad line").unwrap_err();
        assert!(e.contains("line 2"), "{e}");
        let e = SignatureDb::parse("x\tzz\t11").unwrap_err();
        assert!(e.contains("hex"), "{e}");
    }
}
