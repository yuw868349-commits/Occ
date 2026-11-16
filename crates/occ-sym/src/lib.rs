//! Execution and data flow over the decompiler IR.
//!
//! The decompiler answers *what the code looks like*. This crate answers the two
//! questions that come next, on exactly the same lowered program — the same
//! lifting, the same SSA, the same basic blocks — so an answer here can be read
//! against the pseudo-C without a translation step:
//!
//! * **[`exec`]** — run a function. Branch conditions that depend on inputs fork
//!   the state, decisions accumulate as constraints, and each completed path
//!   carries the inputs that reach it ([`solve`]) and the value it returns. A
//!   path that hits a bound says so instead of pretending to be dead code.
//! * **[`taint`]** — propagate a set of sources through the function and report
//!   where they arrive: which argument of which call, which branch, which store,
//!   with a block path from the entry.
//!
//! What this is not: a soundness oracle. Calls are not interpreted, memory is
//! tracked only at addresses the executor can name, and the solver is a bounded
//! search rather than a decision procedure. Every one of those limits is
//! reported in the result, because a reader has to be able to tell "the analysis
//! proved it" from "the analysis gave up here".
//!
//! Both entry points take a function that [`occ_decompile::build_program`]
//! already lowered, a name resolver for call targets, and the limits above.


#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod exec;
pub mod mem;
pub mod solve;
pub mod taint;
pub mod value;

pub use exec::{CallInfo, Config as ExecConfig, Mode, Outcome, Path, Report as ExecReport, explore};
pub use mem::Memory;
pub use solve::{Constraint, Model, Verdict, solve as solve_constraints};
pub use taint::{Config as TaintConfig, Flow, Report as TaintReport, SinkFilter, Source, trace};
pub use value::Value;
