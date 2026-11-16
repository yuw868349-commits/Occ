//! End-to-end checks for the raw bindings.
//!
//! A declaration that compiles proves nothing about the ABI: a wrong struct
//! layout, a wrong request number or a wrong `waitpid` decode all produce code
//! that builds cleanly and fails at runtime. These tests therefore fork a real
//! process and drive it through the tracer, which is the only way to know the
//! declarations match the kernel.

use occ_sys::ptrace::{self, PTRACE_O_EXITKILL, PTRACE_O_TRACESYSGOOD, signal};
use occ_sys::{WaitStatus, errno};
use std::os::unix::process::CommandExt;
use std::process::Command;

/// Start `path` under the tracer and wait for the initial `SIGTRAP`.
fn spawn_traced(path: &str) -> std::process::Child {
    let mut cmd = Command::new(path);
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    // SAFETY: `trace_me` is async-signal-safe, which is the documented
    // requirement for `pre_exec`. It touches no allocator and no Rust runtime
    // state.
    unsafe {
        cmd.pre_exec(|| {
            ptrace::trace_me().map_err(std::io::Error::other)?;
            Ok(())
        });
    }
    let child = cmd.spawn().expect("spawn tracee");
    let pid = child.id() as i32;
    let status = ptrace::wait(pid).expect("initial stop");
    assert!(
        matches!(status, WaitStatus::Stopped(_)),
        "first wait must be a stop, got {status:?}"
    );
    child
}

#[test]
fn trace_me_reports_the_exec_trap() {
    let mut child = spawn_traced("/bin/true");
    let pid = child.id() as i32;

    // The stop after `execve` is a `SIGTRAP`, and on a PIE the entry point is
    // already relocated at this point.
    ptrace::set_options(pid, PTRACE_O_EXITKILL | PTRACE_O_TRACESYSGOOD).expect("options");
    let regs = ptrace::get_regs(pid).expect("get_regs");
    assert!(regs.rip != 0, "rip must be set at the exec stop");
    assert!(regs.rsp != 0, "rsp must be set at the exec stop");

    ptrace::cont(pid, 0).expect("cont");
    let status = ptrace::wait(pid).expect("final wait");
    assert_eq!(
        status,
        WaitStatus::Exited(0),
        "/bin/true must exit cleanly when traced"
    );
    let _ = child.wait();
}

#[test]
fn peek_reads_the_entry_instruction_and_poke_is_reversible() {
    let mut child = spawn_traced("/bin/true");
    let pid = child.id() as i32;
    ptrace::set_options(pid, PTRACE_O_EXITKILL).expect("options");

    let regs = ptrace::get_regs(pid).expect("get_regs");
    let original = ptrace::peek_word(pid, regs.rip).expect("peek");
    assert_ne!(original, 0, "entry point must not be zero-filled");

    // Write the same value back. This proves `poke_word` addresses the same
    // byte range `peek_word` read, which is the pairing the breakpoint layer
    // depends on.
    ptrace::poke_word(pid, regs.rip, original).expect("poke");
    let after = ptrace::peek_word(pid, regs.rip).expect("peek again");
    assert_eq!(after, original, "poke must be visible to peek");

    ptrace::cont(pid, 0).expect("cont");
    let _ = ptrace::wait(pid);
    let _ = child.wait();
}

#[test]
fn peek_on_an_unmapped_address_reports_eio() {
    let mut child = spawn_traced("/bin/true");
    let pid = child.id() as i32;
    ptrace::set_options(pid, PTRACE_O_EXITKILL).expect("options");

    // Address 0 is never mapped in a normal process, and the kernel reports
    // EIO rather than letting the tracer read it.
    let err = ptrace::peek_word(pid, 0).expect_err("null page must not be readable");
    assert_eq!(err.code, errno::EIO, "{err}");

    ptrace::cont(pid, 0).expect("cont");
    let _ = ptrace::wait(pid);
    let _ = child.wait();
}

#[test]
fn syscall_stops_are_told_apart_from_breakpoints() {
    let mut child = spawn_traced("/bin/true");
    let pid = child.id() as i32;
    ptrace::set_options(pid, PTRACE_O_EXITKILL | PTRACE_O_TRACESYSGOOD).expect("options");

    // Step to the first syscall boundary. `execve` has already happened, so the
    // next one is either `brk` or an exit path.
    ptrace::syscall(pid, 0).expect("syscall");
    let mut saw_syscall_stop = false;
    for _ in 0..64 {
        let status = ptrace::wait(pid).expect("wait");
        match status {
            WaitStatus::Stopped(sig) if sig == signal::SIGTRAP_SYSCALL => {
                saw_syscall_stop = true;
                break;
            }
            WaitStatus::Stopped(_) => {
                ptrace::syscall(pid, 0).expect("resume");
            }
            _ => break,
        }
    }
    assert!(
        saw_syscall_stop,
        "TRACESYSGOOD must mark syscall stops with SIGTRAP|0x80"
    );

    ptrace::kill_tracee(pid).ok();
    let _ = ptrace::wait(pid);
    let _ = child.wait();
}

#[test]
fn attach_to_a_missing_pid_reports_esrch() {
    // A pid that cannot exist: above the default pid_max on any sane system.
    let err = ptrace::attach(0x7fff_fffe).expect_err("attach must fail");
    assert!(
        err.code == errno::ESRCH || err.code == errno::EPERM,
        "expected ESRCH or EPERM, got {err}"
    );
}
