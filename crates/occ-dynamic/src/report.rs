//! Putting a trace next to the static view of the same file.
//!
//! A trace on its own is a list of addresses. What makes it evidence is the
//! comparison: which recovered function actually ran, which basic block was
//! never entered, which indirect call went where, and — the answer static
//! analysis cannot produce — which executed addresses belong to no recovered
//! function at all. That last set is the honest measure of recovery gaps, and it
//! is the reason this module exists rather than leaving the caller to join two
//! tables by hand.
//!
//! Everything here is a set operation over observed addresses. Nothing is
//! inferred from absence beyond the stated bound: a function reported as never
//! executed was not executed *during this run*, and if the run was truncated the
//! caller is told so by the trace, not by this module.

use occ_analyze::analysis::Analysis;
use occ_core::{Addr, BasicBlock, Function, Image};
use std::collections::BTreeMap;

use crate::run::Trace;

/// Coverage of one recovered function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionHit {
    /// Entry address.
    pub entry: Addr,
    /// Display name from the analysis.
    pub name: String,
    /// Distinct instructions executed inside the function.
    pub insns: u32,
    /// Instructions the recovery found in it.
    pub total: u32,
    /// Times its instructions ran, summed.
    pub hits: u64,
    /// Basic blocks entered at least once.
    pub blocks: u32,
    /// Basic blocks the recovery found.
    pub blocks_total: u32,
}

/// An executed address inside the image that no recovered function covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Orphan {
    /// The address, in image coordinates.
    pub addr: Addr,
    /// Index of the section it falls in.
    pub section: usize,
}

/// One target a call site reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    /// Image address when `in_image`, otherwise the runtime address.
    pub addr: Addr,
    /// True when the address belongs to the traced image.
    pub in_image: bool,
    /// Times this site reached it.
    pub count: u32,
}

/// A call site with the targets it was seen to reach.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallSite {
    /// Address of the call instruction: image coordinates when `in_image`,
    /// otherwise the runtime address.
    pub site: Addr,
    /// True when the site itself is in the image.
    pub in_image: bool,
    /// Name of the containing function, when recovery covers the site.
    pub from: Option<String>,
    /// Targets, most frequent first.
    pub targets: Vec<Target>,
    /// True when the instruction is not a direct call.
    pub indirect: bool,
}

/// The comparison of one trace against one image.
#[derive(Clone, Debug)]
pub struct Correlated {
    /// Distinct instructions executed inside the image.
    pub executed_in_image: usize,
    /// Instructions executed elsewhere — the loader, libc, the vdso.
    pub external: u64,
    /// Per-function coverage, most-executed first.
    pub functions: Vec<FunctionHit>,
    /// Recovered functions with no executed instruction.
    pub missed: Vec<(Addr, String)>,
    /// Executed addresses inside the image that no function covers.
    pub orphans: Vec<Orphan>,
    /// Call sites, ordered by address.
    pub calls: Vec<CallSite>,
}

impl Correlated {
    /// Entry addresses of the functions that ran, in address order.
    pub fn hit_entries(&self) -> Vec<Addr> {
        let mut v: Vec<Addr> = self.functions.iter().map(|f| f.entry).collect();
        v.sort_by_key(|a| a.0);
        v
    }

    /// The site with the most targets: usually a dispatch table.
    pub fn busiest_site(&self) -> Option<&CallSite> {
        self.calls.iter().max_by_key(|c| c.targets.len())
    }
}

/// One basic block's share of a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockHit {
    /// Block start.
    pub start: Addr,
    /// One past the last byte.
    pub end: Addr,
    /// Instructions in the block.
    pub insns: u32,
    /// Times its instructions ran, summed.
    pub hits: u64,
}

/// How many times the instructions of `b` ran.
fn block_hits(b: &BasicBlock, f: &Function, hits: &BTreeMap<u64, u32>) -> u64 {
    let end = b.insn_end.min(f.insns.len());
    let mut total = 0u64;
    for insn in &f.insns[b.insn_start.min(end)..end] {
        total += u64::from(*hits.get(&insn.addr.0).unwrap_or(&0));
    }
    total
}

/// Which function covers `addr`, as an index into `functions`.
///
/// `functions` is sorted by entry, so the candidate is the last one that starts
/// at or before the address. Overlapping bodies make that a candidate rather than
/// an answer, hence the bounded walk backwards: the tightest containing function
/// is the most specific claim, and a body nested inside another is at most a few
/// steps away.
fn covering(functions: &[Function], addr: Addr) -> Option<usize> {
    let at = functions.partition_point(|f| f.entry <= addr);
    functions[..at]
        .iter()
        .enumerate()
        .rev()
        .take(16)
        .find(|(_, f)| addr.0 < f.end().0)
        .map(|(i, _)| i)
}

/// Compare `trace` against `image` and `analysis`.
pub fn correlate(trace: &Trace, image: &Image, analysis: &Analysis) -> Correlated {
    let hits = trace.hit_counts();
    let functions = &analysis.functions;

    #[derive(Default, Clone, Copy)]
    struct Acc {
        hits: u64,
        distinct: u32,
    }
    let mut acc = vec![Acc::default(); functions.len()];

    let mut orphans = Vec::new();
    for (&addr, &count) in &hits {
        let a = Addr(addr);
        if let Some(i) = covering(functions, a) {
            acc[i].hits += u64::from(count);
            acc[i].distinct += 1;
            continue;
        }
        if let Some(s) = image.section_at(a)
            && s.is_code()
        {
            orphans.push(Orphan {
                addr: a,
                section: s.index,
            });
        }
    }
    let executed_in_image = hits.len();
    let external = trace.external_steps;

    let mut rows: Vec<FunctionHit> = Vec::new();
    for (i, f) in functions.iter().enumerate() {
        if f.insns.is_empty() {
            continue;
        }
        let blocks = f
            .blocks
            .iter()
            .filter(|b| block_hits(b, f, &hits) > 0)
            .count() as u32;
        rows.push(FunctionHit {
            entry: f.entry,
            name: f.display_name(),
            insns: acc[i].distinct,
            total: f.insns.len() as u32,
            hits: acc[i].hits,
            blocks,
            blocks_total: f.blocks.len() as u32,
        });
    }
    rows.sort_by(|a, b| b.hits.cmp(&a.hits).then(a.entry.cmp(&b.entry)));

    let missed: Vec<(Addr, String)> = functions
        .iter()
        .enumerate()
        .filter(|(i, f)| !f.insns.is_empty() && acc[*i].distinct == 0)
        .map(|(_, f)| (f.entry, f.display_name()))
        .collect();

    // Group the tracer's edges by site; a direct call has one target, an
    // indirect one has as many as the run produced.
    let mut by_site: BTreeMap<u64, CallSite> = BTreeMap::new();
    for e in &trace.calls {
        let site = by_site.entry(e.site.0).or_insert_with(|| CallSite {
            site: e.site,
            in_image: e.site_in_image,
            from: e.site_in_image
                .then(|| covering(functions, e.site))
                .flatten()
                .map(|i| functions[i].display_name()),
            targets: Vec::new(),
            indirect: e.indirect,
        });
        site.indirect |= e.indirect;
        site.targets.push(Target {
            addr: e.target,
            in_image: e.target_in_image,
            count: e.count,
        });
    }
    let mut calls: Vec<CallSite> = by_site.into_values().collect();
    for c in &mut calls {
        c.targets
            .sort_by(|a, b| b.count.cmp(&a.count).then(a.addr.cmp(&b.addr)));
    }

    Correlated {
        executed_in_image,
        external,
        functions: rows,
        missed,
        orphans,
        calls,
    }
}

/// Block-level coverage of one function.
///
/// This is the answer to "did this branch ever go the other way": a block with
/// zero hits was not entered, and if the trace is complete that is a fact about
/// the input, not about the analysis.
pub fn function_blocks(trace: &Trace, func: &Function) -> Vec<BlockHit> {
    let hits = trace.hit_counts();
    func.blocks
        .iter()
        .map(|b| BlockHit {
            start: b.start,
            end: b.end,
            insns: b.len() as u32,
            hits: block_hits(b, func, &hits),
        })
        .collect()
}
