//! Basic block construction.
//!
//! A block is a maximal run of instructions with one entry and one exit. The
//! leaders are: the function entry, every branch target that lands inside the
//! function, the instruction after any block-terminating instruction, and the
//! first instruction after an address gap. That last case matters more than it
//! looks: a caller may supply extra targets that recursive descent did not
//! reach by linear decoding (a WASM `br_table` does this), so the instruction
//! list is not guaranteed to be contiguous even after sorting.

use std::collections::BTreeMap;

use occ_core::{Addr, BasicBlock, CallTarget, Edge, EdgeKind, FlowKind, Instruction};

/// Extra targets per branching instruction: the cases of a computed jump, or
/// the labels of a WebAssembly `br_table`. Keyed by the branch's address.
pub type BranchTargets = BTreeMap<u64, Vec<Addr>>;

/// Build blocks and edges from an address-sorted instruction list.
///
/// Instructions must not overlap. Overlaps are dropped (first encoding wins),
/// because two different decodings of the same byte is a bug in recovery, not
/// a property of the program, and silently keeping both would double every
/// later count.
pub fn build_blocks(insns: &[Instruction], entry: Addr) -> Vec<BasicBlock> {
    build_blocks_with_branch_targets(insns, entry, &BranchTargets::new())
}

/// Build blocks, treating every address in `extra` as a leader and linking the
/// branching instruction to each of them.
///
/// A multi-way branch carries more targets than an [`Instruction`] can hold —
/// WebAssembly's `br_table` names one label per case, and an x86 jump table
/// names one address per case — so the caller that knows them passes them here
/// rather than having them silently ignored.
pub fn build_blocks_with_branch_targets(
    insns: &[Instruction],
    entry: Addr,
    extra: &BranchTargets,
) -> Vec<BasicBlock> {
    if insns.is_empty() {
        return Vec::new();
    }

    // Index instructions by address, dropping overlaps.
    let mut kept: Vec<&Instruction> = Vec::with_capacity(insns.len());
    for insn in insns {
        if let Some(last) = kept.last()
            && insn.addr.0 < last.end().0
        {
            continue;
        }
        kept.push(insn);
    }

    let mut leaders = vec![false; kept.len()];
    // The first instruction is a leader regardless of where it sits.
    leaders[0] = true;
    for (i, insn) in kept.iter().enumerate() {
        let next = i + 1;
        if next >= kept.len() {
            break;
        }
        if kept[next].addr != insn.end() {
            leaders[next] = true;
        }
        if insn.flow.terminates_block() {
            leaders[next] = true;
        }
        if let Some(target) = branch_target(insn)
            && let Some(j) = kept.iter().position(|x| x.addr == target)
        {
            leaders[j] = true;
        }
    }
    // The entry itself must start a block even if recovery placed other
    // instructions before it.
    if let Some(j) = kept.iter().position(|x| x.addr == entry) {
        leaders[j] = true;
    }
    for list in extra.values() {
        for target in list {
            if let Some(j) = kept.iter().position(|x| x.addr == *target) {
                leaders[j] = true;
            }
        }
    }

    let mut blocks = Vec::new();
    let mut i = 0usize;
    while i < kept.len() {
        let start_idx = i;
        i += 1;
        while i < kept.len() && !leaders[i] {
            i += 1;
        }
        let slice = &kept[start_idx..i];
        let start = slice[0].addr;
        let end = slice[slice.len() - 1].end();
        let last = slice[slice.len() - 1];
        let edges = edges_of(last, end, extra.get(&last.addr.0));
        blocks.push(BasicBlock {
            start,
            end,
            insn_start: start_idx,
            insn_end: i,
            edges,
        });
    }
    blocks
}

/// Direct branch target, when the instruction has one that stays in the image.
fn branch_target(insn: &Instruction) -> Option<Addr> {
    match (&insn.flow, &insn.target) {
        (FlowKind::ConditionalBranch | FlowKind::UnconditionalBranch, Some(CallTarget::Direct { to })) => {
            Some(*to)
        }
        _ => None,
    }
}

fn edges_of(last: &Instruction, fallthrough: Addr, cases: Option<&Vec<Addr>>) -> Vec<Edge> {
    let mut edges = Vec::with_capacity(2);
    match last.flow {
        // A trap returns, so what follows is reachable. Leaving it
        // unconnected stranded every instruction after a syscall.
        FlowKind::Next | FlowKind::Call | FlowKind::IndirectCall | FlowKind::Interrupt => {
            edges.push(Edge {
                to: fallthrough,
                kind: EdgeKind::FallThrough,
            });
        }
        FlowKind::ConditionalBranch => {
            if let Some(CallTarget::Direct { to }) = last.target {
                edges.push(Edge {
                    to,
                    kind: EdgeKind::True,
                });
            }
            edges.push(Edge {
                to: fallthrough,
                kind: EdgeKind::False,
            });
        }
        FlowKind::UnconditionalBranch => {
            if let Some(CallTarget::Direct { to }) = last.target {
                edges.push(Edge {
                    to,
                    kind: EdgeKind::Unconditional,
                });
            } else {
                edges.push(Edge {
                    to: Addr::NULL,
                    kind: EdgeKind::Indirect,
                });
            }
        }
        FlowKind::IndirectBranch => match cases {
            // The table was read: every entry is a real destination, and each
            // one is a case edge rather than an unknown.
            Some(list) if !list.is_empty() => {
                for to in list {
                    edges.push(Edge {
                        to: *to,
                        kind: EdgeKind::Indirect,
                    });
                }
            }
            _ => edges.push(Edge {
                to: Addr::NULL,
                kind: EdgeKind::Indirect,
            }),
        },
        FlowKind::Return | FlowKind::Exception => {}
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;
    use occ_core::MAX_INSN_LEN;

    fn insn(addr: u64, len: u32, flow: FlowKind, target: Option<u64>) -> Instruction {
        Instruction {
            addr: Addr(addr),
            len,
            bytes: [0; MAX_INSN_LEN],
            text: String::new(),
            mnemonic: "t".into(),
            flow,
            target: target.map(|to| CallTarget::Direct { to: Addr(to) }),
            mem_reads: Vec::new(),
            mem_writes: Vec::new(),
        }
    }

    #[test]
    fn straight_line_is_one_block() {
        let v = vec![
            insn(0x1000, 4, FlowKind::Next, None),
            insn(0x1004, 4, FlowKind::Next, None),
            insn(0x1008, 1, FlowKind::Return, None),
        ];
        let b = build_blocks(&v, Addr(0x1000));
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].len(), 3);
        assert!(b[0].edges.is_empty());
    }

    #[test]
    fn conditional_splits_and_links_both_ways() {
        let v = vec![
            insn(0x1000, 2, FlowKind::ConditionalBranch, Some(0x1006)),
            insn(0x1002, 4, FlowKind::Next, None),
            insn(0x1006, 1, FlowKind::Return, None),
        ];
        // Three blocks: the branch, its fallthrough, and the target. A
        // conditional ends a block by definition even though the instruction
        // following it is adjacent in memory.
        let b = build_blocks(&v, Addr(0x1000));
        assert_eq!(b.len(), 3);
        assert_eq!(b[0].edges.len(), 2);
        assert_eq!(b[0].edges[0], Edge { to: Addr(0x1006), kind: EdgeKind::True });
        assert_eq!(b[0].edges[1], Edge { to: Addr(0x1002), kind: EdgeKind::False });
    }

    #[test]
    fn gap_forces_a_new_block() {
        let v = vec![
            insn(0x1000, 4, FlowKind::Next, None),
            insn(0x2000, 1, FlowKind::Return, None),
        ];
        let b = build_blocks(&v, Addr(0x1000));
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn overlapping_instruction_is_dropped() {
        let v = vec![
            insn(0x1000, 4, FlowKind::Next, None),
            insn(0x1002, 4, FlowKind::Return, None),
        ];
        let b = build_blocks(&v, Addr(0x1000));
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].len(), 1);
    }

    #[test]
    fn indirect_branch_gets_an_unknown_edge() {
        let v = vec![insn(0x1000, 2, FlowKind::IndirectBranch, None)];
        let b = build_blocks(&v, Addr(0x1000));
        assert_eq!(b[0].edges[0].kind, EdgeKind::Indirect);
        assert_eq!(b[0].edges[0].to, Addr::NULL);
    }
}
