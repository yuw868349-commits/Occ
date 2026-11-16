//! Linux system-call bindings, declared directly against glibc.
//!
//! The project ships with an empty dependency list, and every dependency it
//! gave up was replaceable except this one: `ptrace` has no `std` equivalent.
//! Calling it means declaring the symbol and letting the linker bind it, which
//! is what this crate does. The `libc` crate is not involved — it is a
//! re-declaration of these same symbols, and re-declaring them here costs
//! sixty lines instead of a dependency.
//!
//! # Safety
//!
//! This crate contains `unsafe` because FFI requires it. The boundaries are
//! narrow by design: each function takes and returns plain integers, performs
//! exactly one system call, and converts the `-1`-plus-`errno` convention into
//! `Result`. No pointer or file descriptor escapes a function except
//! [`mem::Mapped`], which owns its mapping and unmaps it on drop.

#![deny(missing_docs)]

pub mod cleanup;
pub mod errno;
pub mod mem;
pub mod proc;
pub mod ptrace;
pub mod spawn;

pub use errno::{SysError, errno};
pub use mem::{Mapped, page_size};
pub use ptrace::{Event, Registers, SignalInfo, Stop, WaitStatus, wait_any};

/// Signal numbers, re-exported for callers that only need the constants.
pub use ptrace::signal;
