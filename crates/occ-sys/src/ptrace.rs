//! Raw `ptrace` and `waitpid` bindings for the Linux x86-64 ABI.
//!
//! Every call here is unsafe at the C level and is wrapped so the rest of the
//! project never touches a raw pointer. The wrapper layer is thin on purpose:
//! it fixes the argument types, translates `-1`-plus-`errno` into a
//! [`SysError`], and nothing else. Policy lives in `occ-dynamic`.
//!
//! The register layout is `struct user_regs_struct` exactly as the kernel
//! defines it. Getting one field out of order would not fail to compile — it
//! would silently report the wrong register, so the struct is `#[repr(C)]` and
//! every field is declared in kernel order.

use crate::errno::{self, SysError};
use std::ffi::{c_int, c_long, c_uint, c_void};

// Request numbers, from `<sys/ptrace.h>`.
const PTRACE_TRACEME: c_uint = 0;
const PTRACE_PEEKDATA: c_uint = 2;
const PTRACE_POKEDATA: c_uint = 5;
const PTRACE_CONT: c_uint = 7;
const PTRACE_KILL: c_uint = 8;
const PTRACE_SINGLESTEP: c_uint = 9;
const PTRACE_GETREGS: c_uint = 12;
const PTRACE_SETREGS: c_uint = 13;
const PTRACE_ATTACH: c_uint = 16;
const PTRACE_DETACH: c_uint = 17;
const PTRACE_SYSCALL: c_uint = 24;
const PTRACE_SETOPTIONS: c_uint = 0x4200;
const PTRACE_GETSIGINFO: c_uint = 0x4202;
const PTRACE_GETEVENTMSG: c_uint = 0x4201;

/// Stop the tracee when it enters or leaves a system call.
///
/// `SIGTRAP | 0x80` instead of a bare `SIGTRAP`, so a syscall stop can be told
/// apart from a breakpoint without consulting `PTRACE_GETSIGINFO`.
pub const PTRACE_O_TRACESYSGOOD: PtraceOptions = 0x0000_0001;

/// Kill the tracee if the tracer dies.
///
/// Without this, a crash in the analysis server leaves the traced program
/// running as an orphan with no way for the user to know where it came from.
pub const PTRACE_O_EXITKILL: PtraceOptions = 0x0010_0000;

/// Send the first signal to the tracee and stop it at `execve`.
pub const PTRACE_O_TRACEEXEC: PtraceOptions = 0x0000_0010;

/// Stop the tracee when it creates a thread, and trace the new one.
///
/// Without this a multithreaded program is only partly observed: the original
/// thread stops at every step, its siblings run free, and the trace describes a
/// program that never existed.
pub const PTRACE_O_TRACECLONE: PtraceOptions = 0x0000_0008;

/// Stop the tracee when it forks, and trace the child.
pub const PTRACE_O_TRACEFORK: PtraceOptions = 0x0000_0002;

/// As [`PTRACE_O_TRACEFORK`], for `vfork`.
pub const PTRACE_O_TRACEVFORK: PtraceOptions = 0x0000_0004;

/// Bit set accepted by `PTRACE_SETOPTIONS`.
pub type PtraceOptions = u64;

unsafe extern "C" {
    fn ptrace(request: c_uint, ...) -> c_long;
    fn waitpid(pid: c_int, status: *mut c_int, options: c_int) -> c_int;
    fn getpid() -> c_int;
}

/// This process's pid.
pub fn pid_self() -> i32 {
    // SAFETY: `getpid` cannot fail.
    unsafe { getpid() }
}

/// Why a tracee stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitStatus {
    /// The process exited normally with this code.
    Exited(i32),
    /// The process was killed by this signal.
    Signaled(i32),
    /// The process stopped, delivering this signal.
    Stopped(i32),
    /// The process was resumed by `SIGCONT`.
    Continued,
    /// No state change is available; the caller may retry.
    StillRunning,
}

impl WaitStatus {
    fn from_raw(status: i32) -> Self {
        // These are the `W*` macros from `<sys/wait.h>`, expanded by hand
        // because the crate has no C headers to include.
        if status == 0xffff {
            return WaitStatus::Continued;
        }
        if status & 0x7f == 0 {
            return WaitStatus::Exited((status >> 8) & 0xff);
        }
        if status & 0xff == 0x7f {
            return WaitStatus::Stopped((status >> 8) & 0xff);
        }
        if (status & 0x7f) != 0 {
            return WaitStatus::Signaled(status & 0x7f);
        }
        WaitStatus::StillRunning
    }

    /// Signal number that caused the stop, when this is a stop.
    pub fn stop_signal(self) -> Option<i32> {
        match self {
            WaitStatus::Stopped(s) => Some(s),
            _ => None,
        }
    }
}

/// Something worth reporting from the tracee.
///
/// A `PTRACE_EVENT_*` value from the upper half of a wait status. `wait` throws
/// that half away; [`wait_any`] keeps it, because "the tracee stopped with
/// `SIGTRAP`" means three different things — a completed step, a breakpoint, or
/// a `clone` — and guessing wrong desynchronises the whole trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    /// `PTRACE_EVENT_FORK`.
    Fork,
    /// `PTRACE_EVENT_VFORK`.
    Vfork,
    /// `PTRACE_EVENT_CLONE`.
    Clone,
    /// `PTRACE_EVENT_EXEC`.
    Exec,
    /// `PTRACE_EVENT_EXIT`, reported before the process is gone.
    Exit,
    /// Any other event number, kept verbatim so an unknown one is visible.
    Other(u32),
}

impl Event {
    fn from_raw(n: u32) -> Option<Self> {
        Some(match n {
            0 => return None,
            1 => Event::Fork,
            2 => Event::Vfork,
            3 => Event::Clone,
            4 => Event::Exec,
            6 => Event::Exit,
            other => Event::Other(other),
        })
    }
}

/// One wait result, with the pid that produced it.
#[derive(Clone, Copy, Debug)]
pub struct Stop {
    /// Thread that changed state.
    pub pid: i32,
    /// Exit, signal or stop, decoded the same way as [`wait`].
    pub status: WaitStatus,
    /// The `PTRACE_EVENT_*` this stop reports, when it reports one.
    pub event: Option<Event>,
    /// Raw `waitpid` status word, for callers that need the upper bits.
    pub raw: i32,
}

impl Stop {
    /// True when this stop is a consequence of our own stepping or of a
    /// breakpoint, rather than a signal the program has to see.
    pub fn is_trap(self) -> bool {
        matches!(self.status, WaitStatus::Stopped(s) if s == signal::SIGTRAP)
    }
}

/// Wait for any traced process to change state.
///
/// `waitpid(-1)` rather than `waitpid(pid)`: once a program has threads, the
/// event may come from any of them, and waiting on one specific thread
/// deadlocks as soon as another stops first.
pub fn wait_any() -> Result<Stop, SysError> {
    let mut status: c_int = 0;
    loop {
        // SAFETY: `status` is a valid out-parameter; `-1` asks for any child.
        let ret = unsafe { waitpid(-1, &mut status, 0) };
        if ret < 0 {
            let err = SysError::now("waitpid");
            if err.code == errno::EINTR {
                continue;
            }
            return Err(err);
        }
        let stop = WaitStatus::from_raw(status);
        let event = match stop {
            WaitStatus::Stopped(sig) if sig == signal::SIGTRAP => {
                Event::from_raw((status as u32) >> 16)
            }
            _ => None,
        };
        return Ok(Stop {
            pid: ret,
            status: stop,
            event,
            raw: status,
        });
    }
}

/// Block until `pid` changes state.
pub fn wait(pid: i32) -> Result<WaitStatus, SysError> {
    let mut status: c_int = 0;
    loop {
        // SAFETY: `status` is a valid out-parameter for the duration of the
        // call, and `0` asks for the default blocking behaviour.
        let ret = unsafe { waitpid(pid, &mut status, 0) };
        if ret < 0 {
            let err = SysError::now("waitpid");
            // A signal delivered to the *tracer* interrupts the wait without
            // being an error in the tracee. Retrying is the documented
            // contract, and treating it as failure would abort traces
            // intermittently under any signal load.
            if err.code == errno::EINTR {
                continue;
            }
            return Err(err);
        }
        return Ok(WaitStatus::from_raw(status));
    }
}

/// Ask the kernel to stop this process at `execve`, making it traceable.
///
/// Must be called by the tracee itself, between `fork` and `exec`. In this
/// project that happens inside `Command::pre_exec`, which runs in the child.
pub fn trace_me() -> Result<(), SysError> {
    // SAFETY: the varargs call passes no trailing arguments, which is what
    // PTRACE_TRACEME expects.
    let ret = unsafe { ptrace(PTRACE_TRACEME) };
    if ret < 0 {
        return Err(SysError::now("ptrace(PTRACE_TRACEME)"));
    }
    Ok(())
}

/// Attach to an already-running process.
pub fn attach(pid: i32) -> Result<(), SysError> {
    // SAFETY: null addr and data are required for PTRACE_ATTACH.
    let ret = unsafe {
        ptrace(
            PTRACE_ATTACH,
            pid,
            std::ptr::null_mut::<c_void>(),
            std::ptr::null_mut::<c_void>(),
        )
    };
    if ret < 0 {
        return Err(SysError::now("ptrace(PTRACE_ATTACH)"));
    }
    Ok(())
}

/// Set trace options.
pub fn set_options(pid: i32, options: PtraceOptions) -> Result<(), SysError> {
    // SAFETY: PTRACE_SETOPTIONS takes the option bits directly as `data`.
    let ret = unsafe {
        ptrace(
            PTRACE_SETOPTIONS,
            pid,
            std::ptr::null_mut::<c_void>(),
            options as *mut c_void,
        )
    };
    if ret < 0 {
        return Err(SysError::now("ptrace(PTRACE_SETOPTIONS)"));
    }
    Ok(())
}

/// Resume the tracee, optionally delivering `signal`.
pub fn cont(pid: i32, signal: i32) -> Result<(), SysError> {
    // SAFETY: `signal` is passed as the data word, as the ABI requires.
    let ret = unsafe {
        ptrace(
            PTRACE_CONT,
            pid,
            std::ptr::null_mut::<c_void>(),
            signal as *mut c_void,
        )
    };
    if ret < 0 {
        return Err(SysError::now("ptrace(PTRACE_CONT)"));
    }
    Ok(())
}

/// Resume and stop at the next signal delivery, including syscall boundaries.
pub fn syscall(pid: i32, signal: i32) -> Result<(), SysError> {
    // SAFETY: as `cont`.
    let ret = unsafe {
        ptrace(
            PTRACE_SYSCALL,
            pid,
            std::ptr::null_mut::<c_void>(),
            signal as *mut c_void,
        )
    };
    if ret < 0 {
        return Err(SysError::now("ptrace(PTRACE_SYSCALL)"));
    }
    Ok(())
}

/// Resume and stop after exactly one instruction.
pub fn singlestep(pid: i32, signal: i32) -> Result<(), SysError> {
    // SAFETY: as `cont`.
    let ret = unsafe {
        ptrace(
            PTRACE_SINGLESTEP,
            pid,
            std::ptr::null_mut::<c_void>(),
            signal as *mut c_void,
        )
    };
    if ret < 0 {
        return Err(SysError::now("ptrace(PTRACE_SINGLESTEP)"));
    }
    Ok(())
}

/// Detach, leaving the process running.
pub fn detach(pid: i32, signal: i32) -> Result<(), SysError> {
    // SAFETY: as `cont`.
    let ret = unsafe {
        ptrace(
            PTRACE_DETACH,
            pid,
            std::ptr::null_mut::<c_void>(),
            signal as *mut c_void,
        )
    };
    if ret < 0 {
        return Err(SysError::now("ptrace(PTRACE_DETACH)"));
    }
    Ok(())
}

/// Kill the tracee.
pub fn kill_tracee(pid: i32) -> Result<(), SysError> {
    // SAFETY: null addr and data.
    let ret = unsafe {
        ptrace(
            PTRACE_KILL,
            pid,
            std::ptr::null_mut::<c_void>(),
            std::ptr::null_mut::<c_void>(),
        )
    };
    if ret < 0 {
        return Err(SysError::now("ptrace(PTRACE_KILL)"));
    }
    Ok(())
}

/// Read one machine word from the tracee's address space.
///
/// `-1` is a legal word value, so failure is detected through `errno` rather
/// than the return value. Clearing `errno` first is what makes that
/// distinction reliable.
pub fn peek_word(pid: i32, addr: u64) -> Result<u64, SysError> {
    errno::set_errno(0);
    // SAFETY: the kernel validates `addr` against the tracee's maps; an
    // invalid address yields -1 with EIO rather than undefined behaviour.
    let ret = unsafe {
        ptrace(
            PTRACE_PEEKDATA,
            pid,
            addr as *mut c_void,
            std::ptr::null_mut::<c_void>(),
        )
    };
    if ret == -1 {
        let code = errno::errno();
        if code != 0 {
            return Err(SysError {
                op: "ptrace(PTRACE_PEEKDATA)",
                code,
            });
        }
    }
    Ok(ret as u64)
}

/// Write one machine word into the tracee's address space.
pub fn poke_word(pid: i32, addr: u64, word: u64) -> Result<(), SysError> {
    // SAFETY: as `peek_word`.
    let ret = unsafe {
        ptrace(
            PTRACE_POKEDATA,
            pid,
            addr as *mut c_void,
            word as *mut c_void,
        )
    };
    if ret < 0 {
        return Err(SysError::now("ptrace(PTRACE_POKEDATA)"));
    }
    Ok(())
}

/// x86-64 general-purpose registers, in kernel order.
///
/// This mirrors `struct user_regs_struct` field for field. Field order is
/// load-bearing: `PTRACE_GETREGS` writes the kernel struct straight into this
/// memory, so a reordering would read as a plausible but wrong register.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Registers {
    /// r15.
    pub r15: u64,
    /// r14.
    pub r14: u64,
    /// r13.
    pub r13: u64,
    /// r12.
    pub r12: u64,
    /// rbp.
    pub rbp: u64,
    /// rbx.
    pub rbx: u64,
    /// r11.
    pub r11: u64,
    /// r10.
    pub r10: u64,
    /// r9.
    pub r9: u64,
    /// r8.
    pub r8: u64,
    /// rax. Holds the syscall number on entry and the result on return.
    pub rax: u64,
    /// rcx.
    pub rcx: u64,
    /// rdx.
    pub rdx: u64,
    /// rsi.
    pub rsi: u64,
    /// rdi.
    pub rdi: u64,
    /// Original rax at syscall entry; the kernel rewrites rax on return.
    pub orig_rax: u64,
    /// Instruction pointer.
    pub rip: u64,
    /// Code segment.
    pub cs: u64,
    /// Flags.
    pub eflags: u64,
    /// Stack pointer.
    pub rsp: u64,
    /// Stack segment.
    pub ss: u64,
    /// Base of the fs segment.
    pub fs_base: u64,
    /// Base of the gs segment.
    pub gs_base: u64,
    /// Data segment.
    pub ds: u64,
    /// Extra segment.
    pub es: u64,
    /// fs selector.
    pub fs: u64,
    /// gs selector.
    pub gs: u64,
}

impl Registers {
    /// Read a register by the name used in disassembly.
    ///
    /// Accepts the 64-bit name only. The 32- and 16-bit aliases are not
    /// separate registers and reading them through this method would return
    /// the full-width value, which is not what a caller asking for `eax` wants.
    pub fn get(&self, name: &str) -> Option<u64> {
        Some(match name {
            "rax" => self.rax,
            "rbx" => self.rbx,
            "rcx" => self.rcx,
            "rdx" => self.rdx,
            "rsi" => self.rsi,
            "rdi" => self.rdi,
            "rbp" => self.rbp,
            "rsp" => self.rsp,
            "rip" => self.rip,
            "r8" => self.r8,
            "r9" => self.r9,
            "r10" => self.r10,
            "r11" => self.r11,
            "r12" => self.r12,
            "r13" => self.r13,
            "r14" => self.r14,
            "r15" => self.r15,
            "eflags" => self.eflags,
            _ => return None,
        })
    }

    /// Set a register by name.
    pub fn set(&mut self, name: &str, value: u64) -> bool {
        match name {
            "rax" => self.rax = value,
            "rbx" => self.rbx = value,
            "rcx" => self.rcx = value,
            "rdx" => self.rdx = value,
            "rsi" => self.rsi = value,
            "rdi" => self.rdi = value,
            "rbp" => self.rbp = value,
            "rsp" => self.rsp = value,
            "rip" => self.rip = value,
            "r8" => self.r8 = value,
            "r9" => self.r9 = value,
            "r10" => self.r10 = value,
            "r11" => self.r11 = value,
            "r12" => self.r12 = value,
            "r13" => self.r13 = value,
            "r14" => self.r14 = value,
            "r15" => self.r15 = value,
            "eflags" => self.eflags = value,
            _ => return false,
        }
        true
    }

    /// Syscall number, valid at a syscall-entry stop.
    pub fn syscall_number(&self) -> u64 {
        self.orig_rax
    }

    /// Syscall arguments, in order.
    pub fn syscall_args(&self) -> [u64; 6] {
        [self.rdi, self.rsi, self.rdx, self.r10, self.r8, self.r9]
    }

    /// Syscall return value, valid at a syscall-exit stop.
    ///
    /// A negative return is an error, encoded as `-errno` in two's complement.
    pub fn syscall_result(&self) -> i64 {
        self.rax as i64
    }
}

/// Fetch general-purpose registers.
pub fn get_regs(pid: i32) -> Result<Registers, SysError> {
    let mut regs = Registers::default();
    // SAFETY: `regs` is a correctly laid out `user_regs_struct`, and the kernel
    // writes exactly `sizeof(struct user_regs_struct)` bytes into it.
    let ret = unsafe {
        ptrace(
            PTRACE_GETREGS,
            pid,
            std::ptr::null_mut::<c_void>(),
            (&raw mut regs).cast::<c_void>(),
        )
    };
    if ret < 0 {
        return Err(SysError::now("ptrace(PTRACE_GETREGS)"));
    }
    Ok(regs)
}

/// Write general-purpose registers.
pub fn set_regs(pid: i32, regs: &Registers) -> Result<(), SysError> {
    // SAFETY: as `get_regs`, with the kernel reading the struct.
    let ret = unsafe {
        ptrace(
            PTRACE_SETREGS,
            pid,
            std::ptr::null_mut::<c_void>(),
            (regs as *const Registers).cast_mut().cast::<c_void>(),
        )
    };
    if ret < 0 {
        return Err(SysError::now("ptrace(PTRACE_SETREGS)"));
    }
    Ok(())
}

/// Read the `siginfo_t` that explains the current stop.
///
/// Only the first three fields are decoded: `si_signo`, `si_errno` and
/// `si_code`. The rest is a union whose meaning depends on the signal, and
/// misinterpreted union members are worse than absent ones.
pub fn get_siginfo(pid: i32) -> Result<SignalInfo, SysError> {
    // `siginfo_t` is 128 bytes on x86-64; over-allocating is harmless and
    // avoids depending on the exact size.
    let mut buf = [0u8; 128];
    // SAFETY: the kernel writes at most `sizeof(siginfo_t)` (128) bytes.
    let ret = unsafe {
        ptrace(
            PTRACE_GETSIGINFO,
            pid,
            std::ptr::null_mut::<c_void>(),
            buf.as_mut_ptr().cast::<c_void>(),
        )
    };
    if ret < 0 {
        return Err(SysError::now("ptrace(PTRACE_GETSIGINFO)"));
    }
    let word = |off: usize| -> i32 {
        let mut b = [0u8; 4];
        b.copy_from_slice(&buf[off..off + 4]);
        i32::from_ne_bytes(b)
    };
    let signal = word(0);
    let code = word(8);
    // `si_addr` sits at offset 16: three ints, four bytes of padding for the
    // pointer alignment, then the union member. It only means anything for the
    // fault signals, and only when the kernel (not `raise`) sent them.
    let addr = if code > 0 && matches!(signal, 4 | 7 | 8 | 11) {
        let mut b = [0u8; 8];
        b.copy_from_slice(&buf[16..24]);
        Some(u64::from_ne_bytes(b))
    } else {
        None
    };
    Ok(SignalInfo {
        signal,
        errno: word(4),
        code,
        addr,
    })
}

/// The decoded head of a `siginfo_t`.
#[derive(Clone, Copy, Debug)]
pub struct SignalInfo {
    /// Signal number.
    pub signal: i32,
    /// `si_errno`.
    pub errno: i32,
    /// `si_code`. Negative values come from the kernel, positive from
    /// userspace, which is how a real fault is told from a `raise`.
    pub code: i32,
    /// `si_addr`, for the four signals whose union member is an address.
    ///
    /// Reading this as an address for any other signal would report whichever
    /// union member happens to live at offset 16 — a pid, a band, a file
    /// descriptor — as a pointer.
    pub addr: Option<u64>,
}

/// Fetch the message associated with the current ptrace event.
pub fn get_event_message(pid: i32) -> Result<u64, SysError> {
    let mut msg: u64 = 0;
    // SAFETY: `msg` is a valid out-parameter for `unsigned long`.
    let ret = unsafe {
        ptrace(
            PTRACE_GETEVENTMSG,
            pid,
            std::ptr::null_mut::<c_void>(),
            (&raw mut msg).cast::<c_void>(),
        )
    };
    if ret < 0 {
        return Err(SysError::now("ptrace(PTRACE_GETEVENTMSG)"));
    }
    Ok(msg)
}

/// Signal numbers used by the tracer.
pub mod signal {
    /// Interrupt.
    pub const SIGINT: i32 = 2;
    /// Kill, uncatchable.
    pub const SIGKILL: i32 = 9;
    /// Segmentation fault.
    pub const SIGSEGV: i32 = 11;
    /// Termination request.
    pub const SIGTERM: i32 = 15;
    /// Child state change.
    pub const SIGCHLD: i32 = 17;
    /// Resume.
    pub const SIGCONT: i32 = 18;
    /// Stop, uncatchable.
    pub const SIGSTOP: i32 = 19;
    /// Breakpoint or, with `TRACESYSGOOD`, a syscall boundary.
    pub const SIGTRAP: i32 = 5;
    /// The value `SIGTRAP` takes in `WSTOPSIG` when `TRACESYSGOOD` is set.
    pub const SIGTRAP_SYSCALL: i32 = SIGTRAP | 0x80;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wait_status_decoding_matches_the_macros() {
        // Exited with code 3.
        assert_eq!(WaitStatus::from_raw(3 << 8), WaitStatus::Exited(3));
        // Killed by SIGSEGV.
        assert_eq!(WaitStatus::from_raw(11), WaitStatus::Signaled(11));
        // Stopped by SIGTRAP.
        assert_eq!(
            WaitStatus::from_raw(0x7f | (5 << 8)),
            WaitStatus::Stopped(5)
        );
        // Resumed.
        assert_eq!(WaitStatus::from_raw(0xffff), WaitStatus::Continued);
    }

    #[test]
    fn registers_are_the_kernel_size() {
        // 27 u64 fields. If this ever changes, `PTRACE_GETREGS` is writing past
        // the end of the struct.
        assert_eq!(std::mem::size_of::<Registers>(), 27 * 8);
        assert_eq!(std::mem::align_of::<Registers>(), 8);
    }

    #[test]
    fn register_lookup_covers_every_field() {
        let mut r = Registers::default();
        for name in [
            "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp", "rip", "r8", "r9", "r10",
            "r11", "r12", "r13", "r14", "r15", "eflags",
        ] {
            assert!(r.set(name, 7), "{name} must be settable");
            assert_eq!(r.get(name), Some(7), "{name} must read back");
        }
        assert_eq!(r.get("eax"), None, "32-bit aliases are not full registers");
    }

    #[test]
    fn syscall_arguments_follow_the_abi_order() {
        let r = Registers {
            rdi: 1,
            rsi: 2,
            rdx: 3,
            r10: 4,
            r8: 5,
            r9: 6,
            ..Default::default()
        };
        assert_eq!(r.syscall_args(), [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn negative_syscall_results_are_errors() {
        let r = Registers {
            rax: (-12i64) as u64,
            ..Default::default()
        };
        assert_eq!(r.syscall_result(), -12);
    }

    #[test]
    fn own_pid_is_positive() {
        assert!(pid_self() > 0);
    }
}
