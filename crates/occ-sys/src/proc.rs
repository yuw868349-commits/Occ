//! Process control: process groups and signals.
//!
//! The tracer needs two things from the kernel that `std::process` does not
//! expose: putting the tracee in its own process group, so a runaway trace can
//! be killed including anything it spawned, and sending that group a signal
//! when a wall-clock limit expires. Both are single system calls, which is
//! cheaper to declare than to take a dependency for.

use crate::errno::SysError;
use std::ffi::c_int;

unsafe extern "C" {
    fn setpgid(pid: c_int, pgid: c_int) -> c_int;
    fn kill(pid: c_int, sig: c_int) -> c_int;
}

/// Put `pid` in its own process group.
///
/// Called from the child between `fork` and `exec`, with `(0, 0)`: pid 0 means
/// "me", so the child becomes the leader of a group named after itself. The
/// tracer can then signal `-pid` and reach every process the tracee created,
/// including ones that escaped by forking.
pub fn setpgid_self() -> Result<(), SysError> {
    // SAFETY: two integers, no pointers.
    let ret = unsafe { setpgid(0, 0) };
    if ret < 0 {
        return Err(SysError::now("setpgid"));
    }
    Ok(())
}

/// Send `sig` to one process.
pub fn kill_pid(pid: i32, sig: i32) -> Result<(), SysError> {
    // SAFETY: two integers, no pointers.
    let ret = unsafe { kill(pid, sig) };
    if ret < 0 {
        return Err(SysError::now("kill"));
    }
    Ok(())
}

/// Send `sig` to a process group.
///
/// `pgid` is the group leader's pid, which is what [`setpgid_self`] arranges.
pub fn kill_group(pgid: i32, sig: i32) -> Result<(), SysError> {
    kill_pid(pgid.saturating_neg(), sig)
}

/// Signals the tracer sends.
pub mod signal {
    /// Uncatchable termination. The only signal a stopped tracee cannot ignore.
    pub const SIGKILL: i32 = 9;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn killing_a_missing_process_is_an_error_not_a_panic() {
        // ESRCH (3) for a pid that cannot exist.
        let err = kill_pid(i32::MAX - 1, signal::SIGKILL).expect_err("no such process");
        assert_eq!(err.code, 3, "{err:?}");
    }
}
