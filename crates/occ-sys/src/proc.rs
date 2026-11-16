//! Process control: process groups, signals and resource limits.
//!
//! The tracer needs three things from the kernel that `std::process` does not
//! expose: putting the tracee in its own process group, so a runaway trace can
//! be killed including anything it spawned, sending that group a signal
//! when a wall-clock limit expires, and lowering the core-dump limit of the
//! tracee so that a program which dies under the tracer leaves a report
//! instead of a core file. Each is a single system call, which is cheaper to
//! declare than to take a dependency for.

use crate::errno::SysError;
use std::ffi::c_int;

unsafe extern "C" {
    fn setpgid(pid: c_int, pgid: c_int) -> c_int;
    fn kill(pid: c_int, sig: c_int) -> c_int;
    fn setrlimit(resource: c_int, rlim: *const RLimit) -> c_int;
    fn getrlimit(resource: c_int, rlim: *mut RLimit) -> c_int;
}

/// `RLIMIT_CORE`: the largest core file the process may create.
const RLIMIT_CORE: c_int = 4;

/// The soft and hard limits of one resource.
#[repr(C)]
struct RLimit {
    cur: u64,
    max: u64,
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

/// Stop the process from writing core files.
///
/// Called in a child between `fork` and `exec`, so the limit is inherited by
/// the program being traced. Without it a program that dies under the tracer
/// leaves a core file in whatever directory it happened to be running in,
/// which is a core file in the middle of someone else's source tree. The
/// tracer's own report already says what happened and where.
///
/// Only the soft limit is lowered. A process may lower its own soft limit
/// freely, but raising the hard limit needs privilege, so the hard limit is
/// read and written back unchanged — `RLIM_INFINITY` is preserved rather than
/// turned into `u64::MAX`, because the kernel treats the two differently and
/// a limit of `u64::MAX` is not the same as "no limit".
pub fn disable_core_dumps() -> Result<(), SysError> {
    let mut current = RLimit { cur: 0, max: 0 };
    // SAFETY: `current` is a live `RLimit` for the duration of the call, and
    // `RLIMIT_CORE` is a valid resource.
    if unsafe { getrlimit(RLIMIT_CORE, &mut current) } < 0 {
        return Err(SysError::now("getrlimit(RLIMIT_CORE)"));
    }
    let limit = RLimit {
        cur: 0,
        max: current.max,
    };
    // SAFETY: as above.
    let ret = unsafe { setrlimit(RLIMIT_CORE, &limit) };
    if ret < 0 {
        return Err(SysError::now("setrlimit(RLIMIT_CORE)"));
    }
    Ok(())
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

    /// Read the current core-dump limits.
    fn core_limits() -> (u64, u64) {
        let mut limit = RLimit { cur: 0, max: 0 };
        // SAFETY: `limit` is a live `RLimit` for the duration of the call.
        let ret = unsafe { getrlimit(RLIMIT_CORE, &mut limit) };
        assert_eq!(ret, 0, "getrlimit(RLIMIT_CORE) failed");
        (limit.cur, limit.max)
    }

    #[test]
    fn core_dumps_can_be_turned_off_without_touching_the_hard_limit() {
        let before = core_limits();
        disable_core_dumps().expect("lower the core-dump limit");
        let after = core_limits();
        assert_eq!(after.0, 0, "the soft limit should be zero");
        assert_eq!(
            after.1, before.1,
            "the hard limit must not change: lowering the soft limit is allowed, \
             changing the hard one is not"
        );
        // Put the limit back so the rest of the suite is unaffected by the
        // order the tests happen to run in.
        let restore = RLimit {
            cur: before.0,
            max: before.1,
        };
        // SAFETY: `restore` is a live `RLimit` for the duration of the call.
        let ret = unsafe { setrlimit(RLIMIT_CORE, &restore) };
        assert_eq!(ret, 0, "restoring the core-dump limit failed");
        assert_eq!(core_limits(), before);
    }
}
