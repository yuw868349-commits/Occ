//! `errno` access without the `libc` crate.
//!
//! glibc and musl both expose the thread-local error number through a
//! function rather than a symbol, and the names differ:
//!
//! * glibc: `__errno_location`
//! * musl: `__errno_location` as well (it is a macro over the same symbol)
//!
//! Both are declared here. The linker resolves whichever the platform's libc
//! provides, so the same source builds against either.

use std::ffi::c_int;
use std::fmt;

unsafe extern "C" {
    /// Pointer to this thread's `errno`.
    fn __errno_location() -> *mut c_int;
}

/// The calling thread's current `errno`.
///
/// Reading this is only meaningful immediately after a call that returned a
/// failure indicator; anything in between may overwrite it.
pub fn errno() -> i32 {
    // SAFETY: `__errno_location` is part of the C runtime every Linux process
    // links, and it always returns a valid pointer to this thread's `errno`.
    unsafe { *__errno_location() }
}

/// Overwrite the calling thread's `errno`.
///
/// Used before a call whose failure mode is "returned -1", so that a stale
/// `errno` from an earlier call cannot be mistaken for this one's.
pub fn set_errno(value: i32) {
    // SAFETY: as above.
    unsafe {
        *__errno_location() = value;
    }
}

/// A failed system call: the operation name plus the `errno` it produced.
#[derive(Clone, Copy, Debug)]
pub struct SysError {
    /// Operation, e.g. `"ptrace(PTRACE_GETREGS)"`.
    pub op: &'static str,
    /// `errno` value.
    pub code: i32,
}

impl SysError {
    /// Capture the current `errno`, naming the operation that failed.
    pub fn now(op: &'static str) -> Self {
        SysError { op, code: errno() }
    }

    /// Symbolic name for the code, when it is one this crate knows.
    pub fn name(&self) -> &'static str {
        match self.code {
            1 => "EPERM",
            2 => "ENOENT",
            3 => "ESRCH",
            4 => "EINTR",
            5 => "EIO",
            6 => "ENXIO",
            9 => "EBADF",
            10 => "ECHILD",
            11 => "EAGAIN",
            12 => "ENOMEM",
            13 => "EACCES",
            14 => "EFAULT",
            16 => "EBUSY",
            19 => "ENODEV",
            21 => "EISDIR",
            22 => "EINVAL",
            25 => "ENOTTY",
            27 => "EFBIG",
            32 => "EPIPE",
            38 => "ENOSYS",
            95 => "EOPNOTSUPP",
            _ => "unknown",
        }
    }
}

impl fmt::Display for SysError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} failed: {} (errno {})",
            self.op,
            self.name(),
            self.code
        )
    }
}

impl std::error::Error for SysError {}

/// `EPERM`: the operation is not permitted.
pub const EPERM: i32 = 1;

/// `EINTR`: the call was interrupted by a signal and may be retried.
pub const EINTR: i32 = 4;

/// `ESRCH`: no such process.
pub const ESRCH: i32 = 3;

/// `ECHILD`: no child has this pid.
pub const ECHILD: i32 = 10;

/// `EIO`: generic I/O failure, which is what `/proc/pid/mem` returns for an
/// unmapped address.
pub const EIO: i32 = 5;

/// `EFAULT`: bad address.
pub const EFAULT: i32 = 14;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errno_location_is_writable_and_stable() {
        let saved = errno();
        set_errno(42);
        assert_eq!(errno(), 42);
        set_errno(saved);
        assert_eq!(errno(), saved);
    }

    #[test]
    fn known_codes_have_names() {
        assert_eq!(SysError { op: "x", code: 3 }.name(), "ESRCH");
        assert_eq!(SysError { op: "x", code: 999 }.name(), "unknown");
        assert!(
            SysError {
                op: "ptrace",
                code: 3
            }
            .to_string()
            .contains("ESRCH")
        );
    }
}
