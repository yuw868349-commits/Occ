//! Forking a child that executes code under a tracer.
//!
//! A differential test needs a process whose registers can be set to arbitrary
//! values and whose next instruction can be a single step. `std::process` can
//! start a program but cannot hand it a register file, so the child is forked
//! here and stopped with `SIGSTOP` before it runs anything.
//!
//! # Why the mappings are made before the fork
//!
//! The parent maps two regions shared with the child and writes through them
//! afterwards. Doing it this way means the address is known to both sides
//! without a pipe, and a write from the parent is visible to the child instead
//! of being lost to copy-on-write.
//!
//! # Why the child does almost nothing
//!
//! After `fork` in a threaded process, only async-signal-safe calls are
//! defined; the allocator's locks may be held by a thread that no longer
//! exists. The child therefore calls `ptrace`, `raise` and `_exit` and nothing
//! else — no allocation, no `Result`, no formatting.

use std::ffi::{c_int, c_void};

use crate::errno::SysError;

const PROT_READ: c_int = 1;
const PROT_WRITE: c_int = 2;
const PROT_EXEC: c_int = 4;
const MAP_SHARED: c_int = 1;
const MAP_ANONYMOUS: c_int = 0x20;
/// Place the mapping at exactly the address asked for, and fail if it is
/// taken. Without this flag `mmap` would evict whatever is there.
const MAP_FIXED_NOREPLACE: c_int = 0x100000;
const SIGSTOP: c_int = 19;

unsafe extern "C" {
    fn fork() -> c_int;
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        off: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
    fn raise(sig: c_int) -> c_int;
    fn _exit(code: c_int) -> !;
}

/// Fork. Returns 0 in the child and the child's pid in the parent.
///
/// # Safety
///
/// In a multithreaded program the child may only call async-signal-safe
/// functions before `exec`. Callers must treat the returned 0 as "run three
/// system calls and exit", not as "continue normally".
pub unsafe fn fork_child() -> Result<i32, SysError> {
    // SAFETY: `fork` takes no arguments.
    let pid = unsafe { fork() };
    if pid < 0 {
        return Err(SysError::now("fork"));
    }
    Ok(pid)
}

/// A region shared with a forked child.
///
/// Shared, so a write from the parent is visible to the child and a write from
/// the child is visible back. `MAP_ANONYMOUS`, so nothing touches the disk.
pub struct Shared {
    addr: *mut u8,
    len: usize,
}

impl Shared {
    /// Map `len` bytes, rounded up to a page.
    ///
    /// `exec` adds `PROT_EXEC`, which is only needed for the region that holds
    /// the instructions. Everything else is mapped without it.
    pub fn map(len: usize, exec: bool) -> Result<Shared, SysError> {
        Shared::map_at(None, len, exec)
    }

    /// Map `len` bytes at a chosen address, if that address is free.
    ///
    /// A fixed address makes a run reproducible. The kernel is free to place
    /// an anonymous mapping anywhere, and where it lands ends up inside the
    /// register values a caller feeds the traced process, so an address that
    /// moves between runs decides different questions — whether a division
    /// overflows depends on bits of it that the kernel chose. Asking for a
    /// specific address removes that.
    ///
    /// The mapping is placed with `MAP_FIXED_NOREPLACE`, so an address that is
    /// already taken is an error rather than a reason to evict whatever was
    /// there. Callers that can do without a fixed address should use
    /// [`Shared::map`] instead; callers that need one should treat a failure
    /// as "not available" and say so, because the run is no longer
    /// reproducible and the difference is invisible in the results.
    pub fn map_at(addr: Option<u64>, len: usize, exec: bool) -> Result<Shared, SysError> {
        let page = crate::mem::page_size();
        let len = len.div_ceil(page) * page;
        let mut prot = PROT_READ | PROT_WRITE;
        if exec {
            prot |= PROT_EXEC;
        }
        let flags = MAP_SHARED | MAP_ANONYMOUS;
        // SAFETY: an anonymous shared mapping with no file descriptor is the
        // documented use of `fd = -1`. The address, when given, is a page
        // boundary the caller chose, and `MAP_FIXED_NOREPLACE` makes the
        // kernel refuse rather than replace anything.
        let addr = unsafe {
            mmap(
                match addr {
                    Some(a) => a as *mut c_void,
                    None => std::ptr::null_mut(),
                },
                len,
                prot,
                match addr {
                    Some(_) => flags | MAP_FIXED_NOREPLACE,
                    None => flags,
                },
                -1,
                0,
            )
        };
        if addr as isize == -1 {
            return Err(SysError::now("mmap"));
        }
        Ok(Shared {
            addr: addr.cast::<u8>(),
            len,
        })
    }

    /// Base address.
    pub fn base(&self) -> u64 {
        self.addr as u64
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when the mapping is empty, which cannot happen after `map`.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Write through the parent's view.
    pub fn write(&mut self, offset: usize, bytes: &[u8]) {
        debug_assert!(offset + bytes.len() <= self.len);
        // SAFETY: the range was checked against the mapping's length.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.addr.add(offset), bytes.len());
        }
    }

    /// Read through the parent's view.
    pub fn read(&self, offset: usize, len: usize) -> &[u8] {
        debug_assert!(offset + len <= self.len);
        // SAFETY: as `write`.
        unsafe { std::slice::from_raw_parts(self.addr.add(offset), len) }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: the pointer and length are the ones `mmap` returned.
        unsafe {
            munmap(self.addr.cast::<c_void>(), self.len);
        }
    }
}

/// Fork a child that has asked to be traced and then stopped itself.
///
/// Returns 0 in the child, which is the caller's cue to do nothing further and
/// let the tracer drive it, and the child's pid in the parent. The child makes
/// three calls before it stops — `ptrace(TRACEME)`, `raise(SIGSTOP)` and a
/// spin — all of them async-signal-safe, which is the only thing a forked
/// child may do.
pub fn fork_stopped_child() -> Result<i32, SysError> {
    // SAFETY: `fork` takes no arguments, and the child performs no allocation.
    let pid = unsafe { fork() };
    if pid < 0 {
        return Err(SysError::now("fork"));
    }
    if pid == 0 {
        // SAFETY: single-step tracing of the calling process, then a stop the
        // parent is waiting for.
        unsafe {
            let _ = crate::ptrace::trace_me();
            raise(SIGSTOP);
        }
        // The tracer resumes this only to execute one instruction at a time.
        loop {
            std::hint::spin_loop();
        }
    }
    Ok(pid)
}

/// Stop the calling thread with `SIGSTOP` so a tracer can take over.
///
/// # Safety
///
/// Only meaningful in a child that just called [`fork_child`] and is about to
/// be traced.
pub unsafe fn stop_self() {
    // SAFETY: a signal to the calling thread.
    unsafe {
        raise(SIGSTOP);
    }
}

/// Leave the process immediately, without running any destructor.
///
/// # Safety
///
/// Only for a forked child: running the parent's destructors there would free
/// objects the parent still owns.
pub unsafe fn exit_now(code: i32) -> ! {
    // SAFETY: `_exit` does not return.
    unsafe { _exit(code) }
}
