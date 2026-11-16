//! Runtime observation through `ptrace`, and what it proves about a file.
//!
//! This is a debugger, not a sandbox. The traced process runs natively with the
//! caller's privileges and everything it does has real effects. [`run`] means
//! "run this program", because that is exactly what it does.
//!
//! # What a trace adds to static analysis
//!
//! Three questions static analysis answers badly, and this answers directly:
//!
//! * **What does an indirect call reach?** A function pointer resolved at run
//!   time is a register value here, and the tracer records every target each
//!   site produced.
//! * **Which code is real?** Recovery guesses at boundaries; a run says which
//!   instructions executed. Addresses that executed but belong to no recovered
//!   function are recovery gaps, not opinions.
//! * **What did it touch?** Syscall numbers, arguments and the strings behind
//!   them, in order.
//!
//! # Bounds
//!
//! Every run is capped by a step count and a wall clock, and the result says
//! which cap ended it. A truncated trace is still evidence — everything up to
//! the cut was observed — but it cannot support "this never happens". The type
//! keeps that distinction in [`End::truncated`] rather than leaving it to prose.
//!
//! # Platform
//!
//! x86-64 Linux only: the register layout, the syscall table and the ptrace
//! request numbers are all that ABI's. A trace of an AArch64 binary is refused
//! rather than silently misread.

#![deny(unsafe_code)]

#![deny(missing_docs)]

pub mod maps;
pub mod probe;
pub mod report;
pub mod run;
pub mod syscall;

pub use maps::Map;
pub use report::{
    BlockHit, CallSite, Correlated, FunctionHit, Orphan, correlate, function_blocks,
};
pub use run::{CallEdge, End, RunConfig, Signal, Syscall, Thread, Trace, run};
