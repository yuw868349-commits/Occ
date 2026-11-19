//! Static analysis over a loaded [`occ_core::Image`].
//!
//! The layers above the container parsers all live here:
//!
//! * [`view`] — the single translation from virtual address to file bytes.
//! * [`disasm`] — recursive-descent instruction recovery with symbol bounds.
//! * [`cfg`] — basic blocks and edges inside a recovered function.
//! * [`xref`] — who refers to what, in both directions.
//! * [`strings`] — printable runs, ASCII and UTF-16.
//! * [`entropy`] — where the bytes stop looking like code.
//! * [`fingerprint`] — compiler, language, linkage, packer suspicion.
//! * [`analysis`] — the facade that runs all of the above once and indexes the
//!   result for O(log n) lookups.
//!
//! Nothing here mutates the image. An [`analysis::Analysis`] is a value that
//! can be rebuilt from the same bytes and will make the same decisions, which
//! is what makes a cached result safe to reuse across turns of a conversation.

#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod analysis;
pub mod cfg;
pub mod disasm;
pub mod entropy;
pub mod fingerprint;
pub mod jumptable;
pub mod sig;
pub mod strings;
pub mod view;
pub mod xref;

#[cfg(test)]
mod testutil;

pub use analysis::{Analysis, AnalysisOptions, Coverage};
pub use cfg::{BranchTargets, build_blocks, build_blocks_with_branch_targets};
pub use disasm::{RecoverOptions, Recovery, recover};
pub use strings::{Encoding, Found as FoundString, Options as StringOptions, scan as scan_strings};
pub use view::View;
