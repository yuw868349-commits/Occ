//! Domain model for `occ`.
//!
//! This crate holds no I/O, no parsing, and no disassembly. It defines the
//! vocabulary that every other crate speaks: how an address is written, how an
//! entity is addressed across turns, how a conclusion is justified, and how
//! much of it fits in a caller's context window.
//!
//! Two decisions shape the whole design and are worth stating up front:
//!
//! 1. **Addresses serialize as hex strings, not integers.** MCP clients are
//!    frequently JavaScript, where `Number` is an IEEE-754 double. A `u64`
//!    above 2^53 would be silently truncated, and the truncation would look
//!    like a wrong address rather than a transport bug. `Addr` therefore
//!    serializes as `"0x401000"` and deserializes from either form.
//!
//! 2. **Confidence is derived, never asserted.** [`Claim::confidence`] is a
//!    computed value from the evidence attached to it, using a noisy-OR
//!    combination. A caller that wants to argue with a conclusion can inspect
//!    the individual weights instead of trusting one opaque number.


#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
// The unwrap/expect bans exist so production paths cannot panic on hostile
// input. Tests are the opposite case: a panic there is the assertion, and
// propagating errors out of a test only hides which line failed.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod addr;
pub mod code;
pub mod depth;
pub mod error;
pub mod evidence;
pub mod handle;
pub mod hypothesis;
pub mod image;

pub use addr::Addr;
pub use code::{
    BasicBlock, CallTarget, Edge, EdgeKind, FlowKind, Function, Instruction, JumpTable,
    MAX_INSN_LEN, Xref, XrefKind,
};
pub use depth::{Depth, RenderLimits};
pub use error::{CoreError, Result};
pub use evidence::{
    Alternative, Claim, Evidence, EvidenceKind, EvidenceSource, combine as combine_confidence,
};
pub use handle::{Handle, HandleKind};
pub use hypothesis::{Hypothesis, HypothesisState, Verdict};
pub use image::{
    Arch, CompilerFamily, Endian, Export, Fingerprint, Format, Image, ImageId, Import, ImportKind,
    LangGuess, Linkage, Perm, RelocKind, Relocation, Section, SectionKind, Segment, Symbol,
    SymbolBinding, SymbolKind,
};

/// Crate version, surfaced in MCP `serverInfo` so a client can pin behaviour.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
