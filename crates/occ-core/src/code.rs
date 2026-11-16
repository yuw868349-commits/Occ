//! Recovered code structure: instructions, basic blocks, functions, xrefs.
//!
//! Nothing here borrows from `iced-x86`. The disassembler speaks its own types;
//! this module is the neutral form the rest of the system reasons over. That
//! costs one decode-time conversion, and buys the ability to swap decoders or
//! add a second architecture without touching the session, MCP, or render code.

use crate::addr::Addr;
use crate::evidence::Evidence;

/// Longest possible x86 instruction, in bytes.
pub const MAX_INSN_LEN: usize = 15;

/// How control leaves an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlowKind {
    /// Falls through to the next instruction.
    Next,
    /// Unconditional direct branch.
    UnconditionalBranch,
    /// Conditional direct branch.
    ConditionalBranch,
    /// Direct or indirect call.
    Call,
    /// Unconditional branch through a register or memory operand.
    IndirectBranch,
    /// Call through a register or memory operand.
    IndirectCall,
    /// Function return.
    Return,
    /// Software interrupt (`int3`, `syscall`, `int 0x80`).
    Interrupt,
    /// Traps, `ud2`, and other faulting instructions.
    Exception,
}

impl FlowKind {
    /// True when execution does not continue at the following address.
    pub const fn terminates_block(self) -> bool {
        matches!(
            self,
            FlowKind::UnconditionalBranch
                | FlowKind::ConditionalBranch
                | FlowKind::Return
                | FlowKind::IndirectBranch
                | FlowKind::Interrupt
                | FlowKind::Exception
        )
    }

    /// True when this is a call of any kind.
    pub const fn is_call(self) -> bool {
        matches!(self, FlowKind::Call | FlowKind::IndirectCall)
    }
}

/// Where a call lands, resolved as far as the static image allows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallTarget {
    /// A direct call to an address inside this image.
    Direct {
        /// Callee address.
        to: Addr,
    },
    /// A call that resolves to a named import through a PLT stub or IAT slot.
    Import {
        /// Imported symbol name.
        name: String,
        /// Providing library, when recorded.
        library: Option<String>,
    },
    /// A call through a computed pointer. `via` is the slot when the memory
    /// operand address was resolvable.
    Indirect {
        /// Address of the pointer cell, when known.
        via: Option<Addr>,
    },
}

/// One decoded instruction.
#[derive(Debug, Clone)]
pub struct Instruction {
    /// Address of the first byte.
    pub addr: Addr,
    /// Encoded length in bytes.
    ///
    /// A fixed-width encoding never exceeds [`MAX_INSN_LEN`], but WebAssembly
    /// does: a `br_table` carries one label index per case. The width is kept
    /// exact so that address arithmetic stays correct, and `bytes` holds as
    /// much of the prefix as fits.
    pub len: u32,
    /// Raw encoding, zero-padded. Kept so a caller can re-decode independently
    /// and check the engine's answer without trusting it.
    pub bytes: [u8; MAX_INSN_LEN],
    /// Full textual form, Intel syntax.
    pub text: String,
    /// Mnemonic alone, for cheap classification without string splitting.
    pub mnemonic: Box<str>,
    /// How control leaves.
    pub flow: FlowKind,
    /// Direct branch or call target when statically known.
    pub target: Option<CallTarget>,
    /// Absolute addresses read by a memory operand. RIP-relative and
    /// GOT-relative operands are already resolved to virtual addresses.
    pub mem_reads: Vec<Addr>,
    /// Absolute addresses written by a memory operand.
    pub mem_writes: Vec<Addr>,
}

impl Instruction {
    /// Encoded bytes actually present, capped at what `bytes` can hold.
    pub fn encoded(&self) -> &[u8] {
        let n = (self.len as usize).min(MAX_INSN_LEN);
        &self.bytes[..n]
    }

    /// Half-open end address.
    pub fn end(&self) -> Addr {
        self.addr.add(u64::from(self.len))
    }

    /// True when the instruction touches memory at an absolute address.
    pub fn touches_memory(&self) -> bool {
        !self.mem_reads.is_empty() || !self.mem_writes.is_empty()
    }
}

/// Why a basic block points at another block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    /// Sequential continuation.
    FallThrough,
    /// Taken unconditional branch.
    Unconditional,
    /// Conditional branch, condition true.
    True,
    /// Conditional branch, condition false.
    False,
    /// Call edge into another function.
    Call,
    /// Return into an unknown caller.
    Return,
    /// Computed branch (jump table or register).
    Indirect,
}

/// A control-flow edge between basic blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    /// Destination block start.
    pub to: Addr,
    /// Why the edge exists.
    pub kind: EdgeKind,
}

/// A maximal straight-line instruction sequence.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Address of the first instruction.
    pub start: Addr,
    /// Address one past the last instruction byte.
    pub end: Addr,
    /// Index into [`Function::insns`] of the first instruction.
    pub insn_start: usize,
    /// Index one past the last instruction.
    pub insn_end: usize,
    /// Outgoing edges.
    pub edges: Vec<Edge>,
}

impl BasicBlock {
    /// Number of instructions.
    pub fn len(&self) -> usize {
        self.insn_end.saturating_sub(self.insn_start)
    }

    /// True when the block holds no instructions.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A function recovered by analysis.
#[derive(Debug, Clone)]
pub struct Function {
    /// Entry address. This is the address in `fn:<addr>` handles.
    pub entry: Addr,
    /// Name from the symbol table or a user assignment.
    pub name: Option<String>,
    /// Recovered extent in bytes. Zero when the boundary is unknown.
    pub size: u64,
    /// Instructions in address order.
    pub insns: Vec<Instruction>,
    /// Basic blocks in address order.
    pub blocks: Vec<BasicBlock>,
    /// Entry addresses of functions that call this one.
    pub callers: Vec<Addr>,
    /// Entry addresses this function calls directly.
    pub callees: Vec<Addr>,
    /// Addresses of string literals referenced.
    pub string_refs: Vec<Addr>,
    /// Names of imports called.
    pub import_calls: Vec<String>,
    /// Confidence that the boundary is correct, in `[0, 1]`.
    pub boundary_confidence: f32,
    /// Evidence behind the boundary decision.
    pub boundary_evidence: Vec<Evidence>,
    /// True when the body is a linker stub rather than real code.
    pub is_thunk: bool,
    /// Dispatch tables recovered from computed jumps in this function.
    pub jump_tables: Vec<JumpTable>,
}

/// A jump table behind a computed branch, and the cases it reaches.
#[derive(Debug, Clone)]
pub struct JumpTable {
    /// Address of the branching instruction that reads the table.
    pub at: Addr,
    /// Address of the first entry.
    pub base: Addr,
    /// Width of one entry in bytes.
    pub entry_bytes: u8,
    /// True when entries are signed offsets from [`JumpTable::base`] rather
    /// than absolute addresses.
    pub relative: bool,
    /// Number of entries followed.
    pub count: u32,
    /// Register holding the case index, when the table read named one. This is
    /// the discriminant a `switch` renders, not the value jumped to.
    pub index_reg: Option<String>,
    /// The case value of the first entry, when the code said which one it is.
    ///
    /// A compiler that sees `case 100..106` subtracts 100 from the index
    /// before it looks anything up, so the table holds positions and the
    /// cases are the positions plus 100. The subtraction is the only
    /// statement that says so, and without it a switch renders its cases
    /// numbered from zero whatever they were.
    pub first_case: Option<i64>,
    /// Case targets, in table order.
    pub targets: Vec<Addr>,
}

impl Function {
    /// Build an empty function shell.
    pub fn new(entry: Addr) -> Self {
        Function {
            entry,
            name: None,
            size: 0,
            insns: Vec::new(),
            blocks: Vec::new(),
            callers: Vec::new(),
            callees: Vec::new(),
            string_refs: Vec::new(),
            import_calls: Vec::new(),
            boundary_confidence: 0.0,
            boundary_evidence: Vec::new(),
            is_thunk: false,
            jump_tables: Vec::new(),
        }
    }

    /// Best available display name.
    pub fn display_name(&self) -> String {
        match &self.name {
            Some(n) => n.clone(),
            None => format!("sub_{:x}", self.entry.get()),
        }
    }

    /// Instruction covering `addr`, if any.
    pub fn insn_at(&self, addr: Addr) -> Option<&Instruction> {
        self.insns
            .binary_search_by(|i| i.addr.cmp(&addr))
            .ok()
            .map(|idx| &self.insns[idx])
            .or_else(|| {
                self.insns
                    .iter()
                    .find(|i| addr.within(i.addr, u64::from(i.len)))
            })
    }

    /// Block containing `addr`, if any.
    pub fn block_at(&self, addr: Addr) -> Option<&BasicBlock> {
        self.blocks
            .iter()
            .find(|b| addr.within(b.start, b.end.get().saturating_sub(b.start.get())))
    }

    /// Half-open end address, derived from the last instruction when size is unset.
    pub fn end(&self) -> Addr {
        if self.size > 0 {
            self.entry.add(self.size)
        } else {
            self.insns.last().map(|i| i.end()).unwrap_or(self.entry)
        }
    }

    /// Cyclomatic complexity: `edges - nodes + 2`, with a virtual exit node.
    ///
    /// The usual formula assumes every path ends somewhere; a block that
    /// returns has no successor edge, so counting only real edges quietly
    /// lowers the result for exactly the functions that branch most. Every
    /// block with no successors therefore contributes one edge to a virtual
    /// exit, which keeps a seven-way `switch` reporting 8 instead of 2.
    ///
    /// A single-block straight-line function reports 1. Anything above ~15 is
    /// a signal that the function is doing too much, and for an analyst it is a
    /// hint that the body may contain state-machine or dispatch logic worth
    /// mapping out separately.
    pub fn cyclomatic_complexity(&self) -> u32 {
        let nodes = self.blocks.len() as u32;
        if nodes == 0 {
            return 0;
        }
        let edges: u32 = self
            .blocks
            .iter()
            .map(|b| (b.edges.len() as u32).max(1))
            .sum();
        // The +1 node is the exit the virtual edges point at.
        edges.saturating_sub(nodes + 1).saturating_add(2)
    }

    /// Instructions that write to memory outside the stack frame.
    pub fn external_writes(&self) -> impl Iterator<Item = &Instruction> {
        self.insns.iter().filter(|i| !i.mem_writes.is_empty())
    }
}

/// Kind of cross-reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum XrefKind {
    /// A call.
    Call,
    /// A jump that transfers control permanently (tail call).
    TailCall,
    /// An unconditional jump within a function.
    Jump,
    /// A conditional jump.
    ConditionalJump,
    /// The address is loaded as data (function pointer table, vtable).
    AddressTaken,
    /// Data at the address is read.
    DataRead,
    /// Data at the address is written.
    DataWrite,
    /// A thunk forwards to an import.
    ImportThunk,
}

impl XrefKind {
    /// True when the reference implies executable control flow.
    pub const fn is_control(&self) -> bool {
        matches!(
            self,
            XrefKind::Call
                | XrefKind::TailCall
                | XrefKind::Jump
                | XrefKind::ConditionalJump
                | XrefKind::ImportThunk
        )
    }
}

/// A directed reference from one address to another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Xref {
    /// Source address.
    pub from: Addr,
    /// Destination address.
    pub to: Addr,
    /// Relationship.
    pub kind: XrefKind,
    /// Enclosing function of the source, when known.
    pub from_function: Option<Addr>,
}

impl Xref {
    /// Build one.
    pub const fn new(from: Addr, to: Addr, kind: XrefKind) -> Self {
        Xref {
            from,
            to,
            kind,
            from_function: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insn(addr: u64, len: u32, flow: FlowKind) -> Instruction {
        Instruction {
            addr: Addr(addr),
            len,
            bytes: [0u8; MAX_INSN_LEN],
            text: "nop".into(),
            mnemonic: "nop".into(),
            flow,
            target: None,
            mem_reads: Vec::new(),
            mem_writes: Vec::new(),
        }
    }

    fn two_block_function() -> Function {
        let mut f = Function::new(Addr(0x1000));
        f.insns = vec![
            insn(0x1000, 2, FlowKind::Next),
            insn(0x1002, 2, FlowKind::ConditionalBranch),
            insn(0x1004, 1, FlowKind::Return),
        ];
        f.blocks = vec![
            BasicBlock {
                start: Addr(0x1000),
                end: Addr(0x1004),
                insn_start: 0,
                insn_end: 2,
                edges: vec![
                    Edge {
                        to: Addr(0x1004),
                        kind: EdgeKind::False,
                    },
                    Edge {
                        to: Addr(0x1004),
                        kind: EdgeKind::True,
                    },
                ],
            },
            BasicBlock {
                start: Addr(0x1004),
                end: Addr(0x1005),
                insn_start: 2,
                insn_end: 3,
                edges: vec![Edge {
                    to: Addr(0x1004),
                    kind: EdgeKind::Return,
                }],
            },
        ];
        f.size = 5;
        f
    }

    #[test]
    fn insn_at_finds_interior_address() {
        let f = two_block_function();
        assert_eq!(f.insn_at(Addr(0x1000)).map(|i| i.addr), Some(Addr(0x1000)));
        assert_eq!(f.insn_at(Addr(0x1001)).map(|i| i.addr), Some(Addr(0x1000)));
        assert_eq!(f.insn_at(Addr(0x1002)).map(|i| i.addr), Some(Addr(0x1002)));
        assert!(f.insn_at(Addr(0x1009)).is_none());
    }

    #[test]
    fn block_at_uses_half_open_range() {
        let f = two_block_function();
        assert_eq!(
            f.block_at(Addr(0x1003)).map(|b| b.start),
            Some(Addr(0x1000))
        );
        assert_eq!(
            f.block_at(Addr(0x1004)).map(|b| b.start),
            Some(Addr(0x1004))
        );
        assert!(f.block_at(Addr(0x1005)).is_none());
    }

    #[test]
    fn cyclomatic_complexity_counts_branches() {
        let f = two_block_function();
        // 3 edges, 2 nodes, plus the virtual exit: 3 - 3 + 2 = 2.
        assert_eq!(f.cyclomatic_complexity(), 2);
        assert_eq!(Function::new(Addr(0)).cyclomatic_complexity(), 0);
    }

    #[test]
    fn a_switch_counts_its_cases() {
        // One dispatch block with seven case edges, seven bodies that return:
        // an if-chain would report the same number, which is the point.
        let mut f = Function::new(Addr(0x1000));
        f.blocks = vec![BasicBlock {
            start: Addr(0x1000),
            end: Addr(0x1006),
            insn_start: 0,
            insn_end: 1,
            edges: (0..7)
                .map(|i| Edge {
                    to: Addr(0x1100 + i * 0x10),
                    kind: EdgeKind::Indirect,
                })
                .collect(),
        }];
        for i in 0..7 {
            f.blocks.push(BasicBlock {
                start: Addr(0x1100 + i * 0x10),
                end: Addr(0x1101 + i * 0x10),
                insn_start: 0,
                insn_end: 0,
                edges: Vec::new(),
            });
        }
        assert_eq!(f.cyclomatic_complexity(), 7);
    }

    #[test]
    fn empty_function_end_is_entry() {
        let f = Function::new(Addr(0x2000));
        assert_eq!(f.end(), Addr(0x2000));
    }

    #[test]
    fn display_name_falls_back_to_address() {
        let f = Function::new(Addr(0x4012a0));
        assert_eq!(f.display_name(), "sub_4012a0");
    }

    #[test]
    fn encoded_returns_only_present_bytes() {
        let mut i = insn(0x1000, 3, FlowKind::Next);
        i.bytes[0] = 0x90;
        i.bytes[1] = 0xcc;
        i.bytes[2] = 0xc3;
        assert_eq!(i.encoded(), &[0x90, 0xcc, 0xc3]);
        assert_eq!(i.end(), Addr(0x1003));
    }

    #[test]
    fn call_target_keeps_the_library_that_supplied_the_symbol() {
        let t = CallTarget::Import {
            name: "puts".into(),
            library: Some("libc.so.6".into()),
        };
        match t {
            CallTarget::Import { name, library } => {
                assert_eq!(name, "puts");
                assert_eq!(library.as_deref(), Some("libc.so.6"));
            }
            _ => unreachable!(),
        }
    }
}
