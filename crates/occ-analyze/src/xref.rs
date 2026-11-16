//! Cross-reference queries.
//!
//! The recovery pass produces a flat, sorted list. Analysts ask the same two
//! questions in a loop — "who calls this" and "what does this function touch" —
//! so the list is indexed once at construction rather than scanned per query.

use occ_core::{Addr, Xref, XrefKind};
use std::collections::HashMap;

/// A cross-reference table with both directions indexed.
pub struct XrefTable {
    all: Vec<Xref>,
    to: HashMap<u64, Vec<usize>>,
    from: HashMap<u64, Vec<usize>>,
}

impl XrefTable {
    /// Index a list of references.
    pub fn new(mut all: Vec<Xref>) -> XrefTable {
        all.sort_by_key(|x| (x.from.0, x.to.0));
        let mut to: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut from: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, x) in all.iter().enumerate() {
            to.entry(x.to.0).or_default().push(i);
            from.entry(x.from.0).or_default().push(i);
        }
        XrefTable { all, to, from }
    }

    /// Every reference, in address order.
    pub fn all(&self) -> &[Xref] {
        &self.all
    }

    /// References whose destination is `addr`.
    pub fn to(&self, addr: Addr) -> impl Iterator<Item = &Xref> {
        self.to
            .get(&addr.0)
            .into_iter()
            .flatten()
            .map(move |i| &self.all[*i])
    }

    /// References originating at `addr`.
    pub fn from(&self, addr: Addr) -> impl Iterator<Item = &Xref> {
        self.from
            .get(&addr.0)
            .into_iter()
            .flatten()
            .map(move |i| &self.all[*i])
    }

    /// References to `addr` of a specific kind.
    pub fn to_kind(&self, addr: Addr, kind: XrefKind) -> impl Iterator<Item = &Xref> {
        self.to(addr).filter(move |x| x.kind == kind)
    }

    /// Number of references.
    pub fn len(&self) -> usize {
        self.all.len()
    }

    /// True when no references were recorded.
    pub fn is_empty(&self) -> bool {
        self.all.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes_both_directions() {
        let t = XrefTable::new(vec![
            Xref::new(Addr(0x10), Addr(0x100), XrefKind::Call),
            Xref::new(Addr(0x20), Addr(0x100), XrefKind::Jump),
            Xref::new(Addr(0x10), Addr(0x200), XrefKind::DataRead),
        ]);
        assert_eq!(t.to(Addr(0x100)).count(), 2);
        assert_eq!(t.from(Addr(0x10)).count(), 2);
        assert_eq!(t.to_kind(Addr(0x100), XrefKind::Call).count(), 1);
        assert_eq!(t.to(Addr(0x999)).count(), 0);
    }
}
