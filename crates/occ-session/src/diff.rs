//! Function-level diffing.
//!
//! The question is not "which bytes changed" — `cmp` answers that — but "which
//! functions changed, and did any change shape". Two hashes per function give
//! that:
//!
//! * **shape** hashes the mnemonic and control-flow sequence, ignoring operands
//!   and addresses. Equal shape means the code was rewritten with different
//!   addresses but the same instructions.
//! * **exact** hashes the rendered text. Equal exact means byte-identical up to
//!   address resolution, which catches register allocation changes that leave
//!   the shape intact.
//!
//! Matching is by name first, then by exact hash, then by shape hash. Name is
//! the strongest signal when symbols survive; when they do not, the hashes are
//! what is left, and matching by shape is what lets a stripped binary be
//! compared against its own previous version.

use std::collections::HashMap;

use occ_analyze::analysis::Analysis;
use occ_core::{Addr, FlowKind, Function};

/// Structural summary of one function.
#[derive(Clone, Debug)]
pub struct FuncSig {
    /// Entry address.
    pub entry: Addr,
    /// Display name.
    pub name: String,
    /// Byte size.
    pub size: u64,
    /// Block count.
    pub blocks: u32,
    /// Instruction count.
    pub insns: u32,
    /// Hash of mnemonics and flow kinds.
    pub shape: u64,
    /// Hash of rendered instruction text.
    pub exact: u64,
}

/// How a function changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Change {
    /// Same shape and same text.
    Identical,
    /// Same shape, different operands or registers.
    Modified,
    /// Shape changed; the function's logic was edited.
    Rewritten,
    /// Only in the second image.
    Added,
    /// Only in the first image.
    Removed,
}

impl Change {
    /// Lower-case label.
    pub const fn label(self) -> &'static str {
        match self {
            Change::Identical => "identical",
            Change::Modified => "modified",
            Change::Rewritten => "rewritten",
            Change::Added => "added",
            Change::Removed => "removed",
        }
    }
}

/// A matched pair, or a one-sided function.
#[derive(Clone, Debug)]
pub struct Pair {
    /// Function in the first image.
    pub a: Option<FuncSig>,
    /// Function in the second image.
    pub b: Option<FuncSig>,
    /// Classification.
    pub change: Change,
}

impl Pair {
    /// Display name, preferring the first image.
    pub fn name(&self) -> String {
        self.a
            .as_ref()
            .map(|s| s.name.clone())
            .or_else(|| self.b.as_ref().map(|s| s.name.clone()))
            .unwrap_or_default()
    }
}

/// Result of comparing two images.
#[derive(Clone, Debug, Default)]
pub struct Diff {
    /// Every pair and every one-sided function.
    pub pairs: Vec<Pair>,
}

impl Diff {
    /// Count of pairs in a given class.
    pub fn count(&self, change: Change) -> usize {
        self.pairs.iter().filter(|p| p.change == change).count()
    }

    /// Pairs that are not identical, most interesting first: rewritten, then
    /// modified, then added, then removed.
    pub fn changed(&self) -> Vec<&Pair> {
        let mut v: Vec<&Pair> = self
            .pairs
            .iter()
            .filter(|p| p.change != Change::Identical)
            .collect();
        v.sort_by_key(|p| match p.change {
            Change::Rewritten => 0,
            Change::Modified => 1,
            Change::Added => 2,
            Change::Removed => 3,
            Change::Identical => 4,
        });
        v
    }

    /// One-line summary.
    pub fn summary(&self) -> String {
        format!(
            "{} identical, {} modified, {} rewritten, {} added, {} removed",
            self.count(Change::Identical),
            self.count(Change::Modified),
            self.count(Change::Rewritten),
            self.count(Change::Added),
            self.count(Change::Removed)
        )
    }
}

/// Compute the signature of one function.
pub fn signature(f: &Function) -> FuncSig {
    let mut shape = FNV_OFFSET;
    for i in &f.insns {
        shape = hash_bytes(shape, i.mnemonic.as_bytes());
        shape = hash_byte(shape, i.flow as u8);
        // A call's destination moves between builds; its existence does not.
        shape = hash_byte(shape, u8::from(i.flow.is_call()));
    }
    let mut exact = FNV_OFFSET;
    for i in &f.insns {
        exact = hash_bytes(exact, i.text.as_bytes());
        exact = hash_byte(exact, 0xff);
    }
    FuncSig {
        entry: f.entry,
        name: f.display_name(),
        size: f.size,
        blocks: f.blocks.len() as u32,
        insns: f.insns.len() as u32,
        shape,
        exact,
    }
}

/// Compare two analyses.
pub fn diff(a: &Analysis, b: &Analysis) -> Diff {
    let sa: Vec<FuncSig> = a.functions.iter().map(signature).collect();
    let sb: Vec<FuncSig> = b.functions.iter().map(signature).collect();

    let mut used_a = vec![false; sa.len()];
    let mut used_b = vec![false; sb.len()];
    let mut pairs: Vec<Pair> = Vec::new();

    // 1. By name, when both sides have one. `sub_xxx` names are addresses in
    // disguise and must not be used for matching: they would pair unrelated
    // functions that happen to sit at the same offset in two layouts.
    let mut by_name: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, s) in sb.iter().enumerate() {
        if !s.name.starts_with("sub_") {
            by_name.entry(s.name.as_str()).or_default().push(i);
        }
    }
    for (i, s) in sa.iter().enumerate() {
        if s.name.starts_with("sub_") {
            continue;
        }
        if let Some(cands) = by_name.get_mut(s.name.as_str())
            && let Some(j) = cands.pop()
        {
            used_a[i] = true;
            used_b[j] = true;
            pairs.push(pair_of(Some(sa[i].clone()), Some(sb[j].clone())));
        }
    }

    // 2. By exact hash.
    let mut by_exact: HashMap<u64, Vec<usize>> = HashMap::new();
    for (j, s) in sb.iter().enumerate() {
        if !used_b[j] {
            by_exact.entry(s.exact).or_default().push(j);
        }
    }
    for (i, s) in sa.iter().enumerate() {
        if used_a[i] {
            continue;
        }
        if let Some(cands) = by_exact.get_mut(&s.exact)
            && let Some(j) = cands.pop()
        {
            used_a[i] = true;
            used_b[j] = true;
            pairs.push(Pair {
                a: Some(sa[i].clone()),
                b: Some(sb[j].clone()),
                change: Change::Identical,
            });
        }
    }

    // 3. By shape hash, requiring the same instruction count to avoid pairing
    //    a loop body with an unrolled copy of itself.
    let mut by_shape: HashMap<(u64, u32), Vec<usize>> = HashMap::new();
    for (j, s) in sb.iter().enumerate() {
        if !used_b[j] {
            by_shape.entry((s.shape, s.insns)).or_default().push(j);
        }
    }
    for (i, s) in sa.iter().enumerate() {
        if used_a[i] {
            continue;
        }
        if let Some(cands) = by_shape.get_mut(&(s.shape, s.insns))
            && let Some(j) = cands.pop()
        {
            used_a[i] = true;
            used_b[j] = true;
            pairs.push(Pair {
                a: Some(sa[i].clone()),
                b: Some(sb[j].clone()),
                change: Change::Modified,
            });
        }
    }

    // 4. Whatever is left is one-sided.
    for (i, s) in sa.iter().enumerate() {
        if !used_a[i] {
            pairs.push(Pair {
                a: Some(s.clone()),
                b: None,
                change: Change::Removed,
            });
        }
    }
    for (j, s) in sb.iter().enumerate() {
        if !used_b[j] {
            pairs.push(Pair {
                a: None,
                b: Some(s.clone()),
                change: Change::Added,
            });
        }
    }

    pairs.sort_by_key(|p| {
        p.a.as_ref()
            .map(|s| s.entry.0)
            .or_else(|| p.b.as_ref().map(|s| s.entry.0))
            .unwrap_or(0)
    });
    Diff { pairs }
}

fn pair_of(a: Option<FuncSig>, b: Option<FuncSig>) -> Pair {
    let change = match (&a, &b) {
        (Some(x), Some(y)) if x.exact == y.exact => Change::Identical,
        (Some(x), Some(y)) if x.shape == y.shape => Change::Modified,
        (Some(_), Some(_)) => Change::Rewritten,
        (Some(_), None) => Change::Removed,
        (None, Some(_)) => Change::Added,
        (None, None) => Change::Identical,
    };
    Pair { a, b, change }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn hash_byte(h: u64, b: u8) -> u64 {
    (h ^ u64::from(b)).wrapping_mul(FNV_PRIME)
}

fn hash_bytes(mut h: u64, bytes: &[u8]) -> u64 {
    for b in bytes {
        h = hash_byte(h, *b);
    }
    h
}

/// True when the function contains a loop (a block edge that points backwards).
pub fn has_loop(f: &Function) -> bool {
    f.blocks.iter().any(|b| b.edges.iter().any(|e| e.to.0 < b.start.0))
}

/// True when the function ends in a tail call rather than a return.
pub fn ends_in_tail_call(f: &Function) -> bool {
    f.insns
        .last()
        .map(|i| i.flow == FlowKind::UnconditionalBranch)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_is_stable() {
        assert_eq!(hash_bytes(FNV_OFFSET, b"abc"), hash_bytes(FNV_OFFSET, b"abc"));
        assert_ne!(hash_bytes(FNV_OFFSET, b"abc"), hash_bytes(FNV_OFFSET, b"abd"));
    }

    #[test]
    fn change_labels_are_lower_case() {
        assert_eq!(Change::Rewritten.label(), "rewritten");
    }
}
