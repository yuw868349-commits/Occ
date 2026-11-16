//! Running a program under `ptrace` and recording what it does.
//!
//! # Why single-step, and what it costs
//!
//! Two ways exist to observe a process: breakpoints at addresses you already
//! believe in, or one instruction at a time. Breakpoints are fast but circular —
//! they only confirm the code the static analysis already found, which is the
//! last thing a tool whose job is to check that analysis needs. Single-stepping
//! sees every instruction, including the ones no recovery pass found, so the
//! result can contradict the static view. That is the point of tracing at all.
//!
//! The price is about five system calls per instruction. A 200 000-step budget
//! therefore costs a second or two, which is why every run is bounded by both a
//! step count and a wall clock.
//!
//! # What is recorded
//!
//! * Instruction addresses in execution order — a path, not a set. The set is
//!   derived from it, and the order answers "which branch came first".
//! * Call edges, including indirect ones. The target of `call rax` is a fact
//!   only a running program has.
//! * Syscalls with number, arguments, return value and the string behind each
//!   path argument.
//! * Signals, with the faulting address when the kernel reports one.
//!
//! # What is not
//!
//! Memory writes, register values per instruction, and execution that happened
//! before the first stop. The first two would need a snapshot per step and the
//! third is gone by the time `ptrace` has control. A trace that does not claim
//! them is more useful than one that guesses.

use occ_core::{Addr, Arch, CoreError, Image, Result};
use occ_decode::{AnyDecoder, Decoded};
use occ_sys::proc;
use occ_sys::ptrace::{self, Registers};
use occ_sys::{Event, Stop, WaitStatus};
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::maps::{self, Map};
use crate::syscall;

/// Longest recording of executed instructions. Past this the run is stopped:
/// a path vector is evidence, not a log file.
const MAX_PATH: usize = 4_000_000;

/// One trace at a time, per process.
///
/// `waitpid(-1)` is process-global: it returns the next stop of *any* child, so
/// two threads tracing concurrently would each reap the other's events and both
/// would describe a program that never ran. A lock is the only correct fix that
/// does not involve a second process, and a trace is seconds at most.
///
/// Poisoning is ignored: a trace that panicked should not disable tracing for
/// the rest of the session.
static TRACE_LOCK: Mutex<()> = Mutex::new(());

/// How far to run and what to record.
#[derive(Clone, Debug)]
pub struct RunConfig {
    /// `argv[0]`: the name the program sees as its own.
    ///
    /// Executing the file does not require this to equal the path, and a
    /// program that prints its usage or dispatches on its name reads this
    /// value. `None` uses the path, which is what a shell does.
    pub argv0: Option<String>,
    /// Arguments after the program name.
    pub args: Vec<String>,
    /// Environment entries to add or override.
    pub env: Vec<(String, String)>,
    /// Working directory. `None` inherits.
    pub cwd: Option<PathBuf>,
    /// Instruction budget.
    pub max_steps: u64,
    /// Wall-clock budget, enforced by killing the process group.
    pub max_millis: u64,
    /// Start at the image's entry point instead of at the dynamic loader.
    pub from_entry: bool,
    /// Record syscalls.
    pub trace_syscalls: bool,
    /// Longest path string to read from the tracee.
    pub max_string: usize,
}

impl Default for RunConfig {
    fn default() -> Self {
        RunConfig {
            argv0: None,
            args: Vec::new(),
            env: Vec::new(),
            cwd: None,
            max_steps: 200_000,
            max_millis: 5_000,
            from_entry: true,
            trace_syscalls: true,
            max_string: 256,
        }
    }
}

/// How the run ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum End {
    /// The process called `exit` with this code.
    Exited(i32),
    /// The process died from this signal.
    Signaled(i32),
    /// The step budget ran out; the process was killed.
    StepLimit,
    /// The wall-clock budget ran out; the process was killed.
    TimeLimit,
    /// `ptrace` failed mid-run, with this message.
    Failed(String),
}

impl End {
    /// Short label for output.
    pub fn label(&self) -> String {
        match self {
            End::Exited(c) => format!("exited {c}"),
            End::Signaled(s) => format!("killed by signal {s}"),
            End::StepLimit => "step limit".into(),
            End::TimeLimit => "time limit".into(),
            End::Failed(e) => format!("failed: {e}"),
        }
    }

    /// True when the run stopped because of a budget rather than the program.
    ///
    /// A truncated trace still carries evidence — everything up to the cut is
    /// observed fact — but a conclusion of the form "this never happens" is not
    /// available from it, and the caller needs to be able to tell.
    pub const fn truncated(&self) -> bool {
        matches!(self, End::StepLimit | End::TimeLimit | End::Failed(_))
    }
}

/// A call site and one target it reached.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallEdge {
    /// Address of the call instruction: in image coordinates when
    /// `site_in_image`, otherwise the runtime address.
    pub site: Addr,
    /// Address the call reached, on the same terms as `site`.
    pub target: Addr,
    /// True when `site` is an address of the traced image.
    pub site_in_image: bool,
    /// True when `target` is an address of the traced image.
    pub target_in_image: bool,
    /// Times this site reached this target.
    pub count: u32,
    /// True when the instruction is not a direct call.
    pub indirect: bool,
}

/// One syscall.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Syscall {
    /// Syscall number from `rax`.
    pub nr: u64,
    /// Name from the x86-64 table, or `sys_<nr>`.
    pub name: String,
    /// Arguments in ABI order.
    pub args: [u64; 6],
    /// Negative errno on failure, when `returned` is set.
    pub ret: i64,
    /// True when the step completed, meaning the kernel returned. An
    /// `exit_group` never returns, and saying so is more useful than a zero.
    pub returned: bool,
    /// Address of the `syscall` instruction, in image coordinates.
    pub at: Addr,
    /// Strings read from the path arguments.
    pub paths: Vec<String>,
}

/// A signal delivered to the tracee.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signal {
    /// Signal number.
    pub signal: i32,
    /// Instruction pointer when it arrived, in image coordinates.
    pub at: Addr,
    /// `si_addr`, for faults the kernel reported an address for.
    pub fault: Option<u64>,
}

/// One thread that was traced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Thread {
    /// Kernel thread id.
    pub tid: i32,
    /// Instructions stepped in this thread.
    pub steps: u64,
}

/// Everything one run produced.
#[derive(Clone, Debug)]
pub struct Trace {
    /// Program path as given.
    pub path: String,
    /// Arguments, including the program name.
    pub argv: Vec<String>,
    /// Pid of the initial thread.
    pub pid: i32,
    /// Load base from `/proc/<pid>/maps`.
    pub base: u64,
    /// `base - image_base`: subtracted from a runtime address to get the
    /// address the image itself uses.
    pub delta: i64,
    /// Instructions executed, in order, in image coordinates. Addresses that
    /// belong to another module are counted in `external_steps` instead: a
    /// libc address minus this image's load base is not an address of anything.
    pub path_taken: Vec<Addr>,
    /// Instructions executed outside the image: the loader, libc, the vdso.
    pub external_steps: u64,
    /// Call edges observed.
    pub calls: Vec<CallEdge>,
    /// Syscalls observed, in order.
    pub syscalls: Vec<Syscall>,
    /// Signals observed, in order.
    pub signals: Vec<Signal>,
    /// Threads that ran.
    pub threads: Vec<Thread>,
    /// Total instructions stepped.
    pub steps: u64,
    /// How it ended.
    pub end: End,
    /// Wall-clock duration.
    pub millis: u64,
    /// Problems worth reporting that did not stop the run.
    pub notes: Vec<String>,
}

impl Trace {
    /// Translate a runtime address into the image's coordinates.
    pub fn to_image(&self, runtime: Addr) -> Addr {
        Addr((runtime.0 as i64 - self.delta) as u64)
    }

    /// Distinct instruction addresses executed, sorted.
    ///
    /// The path is kept in order because the order carries information; the set
    /// is what coverage questions ask about, and building it on demand is
    /// cheaper than maintaining both while tracing.
    pub fn coverage(&self) -> Vec<Addr> {
        let mut v = self.path_taken.clone();
        v.sort_by_key(|a| a.0);
        v.dedup();
        v
    }

    /// Times each distinct address was executed.
    pub fn hit_counts(&self) -> BTreeMap<u64, u32> {
        let mut m = BTreeMap::new();
        for a in &self.path_taken {
            *m.entry(a.0).or_insert(0) += 1;
        }
        m
    }
}

/// Run `path` under `ptrace` and record what it does.
///
/// `image` is the file `path` was loaded from; it supplies the entry point and
/// the image base used to translate runtime addresses. A mismatched pair is not
/// prevented, and does not need to be: the load base is matched through
/// `/proc/<pid>/maps`, and a mismatch shows up as every recorded address falling
/// outside the image rather than as silent nonsense.
#[allow(unsafe_code)] // `Command::pre_exec` is unsafe by signature; the closure
                      // body is two async-signal-safe system calls and nothing else.
pub fn run(path: &Path, image: &Image, cfg: &RunConfig) -> Result<Trace> {
    let _serial = TRACE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !cfg!(target_arch = "x86_64") {
        return Err(CoreError::Unavailable(
            "ptrace tracing is implemented for x86-64 hosts only".into(),
        ));
    }
    if !matches!(image.arch, Arch::X86 | Arch::X86_64) {
        return Err(CoreError::Unavailable(format!(
            "the tracee must be x86 or x86-64; the image is {}",
            image.arch.label()
        )));
    }

    let mut cmd = Command::new(path);
    if let Some(a0) = &cfg.argv0 {
        // `Command::new` always execs `path`; `arg0` only changes what the
        // tracee sees as `argv[0]`.
        cmd.arg0(a0);
    }
    cmd.args(&cfg.args);
    if let Some(cwd) = &cfg.cwd {
        cmd.current_dir(cwd);
    }
    for (k, v) in &cfg.env {
        cmd.env(k, v);
    }
    // The tracee's output would otherwise be interleaved into ours, and ours is
    // a JSON-RPC stream. A traced program that writes to stdout has its output
    // dropped; the trace records the write either way.
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    // SAFETY: `pre_exec` runs in the child between fork and exec. Every call
    // is async-signal-safe — `ptrace`, `setpgid` and `setrlimit` are single
    // system calls with no allocation — and nothing else is touched in the
    // closure.
    unsafe {
        cmd.pre_exec(|| {
            ptrace::trace_me().map_err(|e| io::Error::from_raw_os_error(e.code))?;
            proc::setpgid_self().map_err(|e| io::Error::from_raw_os_error(e.code))?;
            // A tracee that dies must not leave a core file behind. The
            // report says what happened; a core file in the caller's source
            // tree says nothing and costs megabytes.
            proc::disable_core_dumps().map_err(|e| io::Error::from_raw_os_error(e.code))?;
            Ok(())
        });
    }
    let child = cmd.spawn().map_err(|e| CoreError::Io {
        path: path.display().to_string(),
        source: e,
    })?;
    let pid = child.id() as i32;
    let started = Instant::now();

    // The first stop is the exec trap: the image is loaded and nothing has run.
    let first = ptrace::wait(pid)?;
    if first != WaitStatus::Stopped(ptrace::signal::SIGTRAP) {
        let _ = proc::kill_group(pid, proc::signal::SIGKILL);
        return Err(CoreError::Backend(format!(
            "the first stop after exec was {first:?}, not a trap"
        )));
    }
    ptrace::set_options(
        pid,
        ptrace::PTRACE_O_EXITKILL
            | ptrace::PTRACE_O_TRACEEXEC
            | ptrace::PTRACE_O_TRACECLONE
            | ptrace::PTRACE_O_TRACEFORK
            | ptrace::PTRACE_O_TRACEVFORK,
    )?;

    let mut notes = Vec::new();
    let table = maps::read(pid)?;
    let mut base = maps::base_of(&table, &path.display().to_string()).unwrap_or(0);
    if base == 0
        && let Some(m) = table
            .iter()
            .filter(|m| m.perm.exec())
            .min_by_key(|m| m.start)
    {
        // A statically linked binary can be mapped at its link address, where no
        // mapping starts at file offset zero. The lowest executable one is the
        // best remaining estimate, and saying so beats a silently wrong offset.
        base = m.start;
        notes.push(format!(
            "no mapping of {} starts at file offset 0; using the lowest executable mapping at {base:#x}",
            path.display()
        ));
    }
    let delta = base as i64 - image.image_base.0 as i64;

    // One breakpoint, at the entry point, so the loader's work is not stepped
    // through: `ld.so` runs tens of thousands of instructions before `main` and
    // the budget is better spent on the program itself.
    let mut entry_bp: Option<(u64, u8)> = None;
    if cfg.from_entry && !image.entry.is_null() {
        let rt = (image.entry.0 as i64 + delta) as u64;
        if table.iter().any(|m| m.perm.exec() && m.contains(rt)) {
            match plant(pid, rt) {
                Ok(orig) => entry_bp = Some((rt, orig)),
                Err(e) => notes.push(format!("could not plant a breakpoint at {rt:#x}: {e}")),
            }
        } else {
            notes.push(format!(
                "entry {} is not mapped executable at run time; tracing from the loader instead",
                image.entry
            ));
        }
    }

    // The watchdog makes a blocking tracee harmless: a program waiting on input
    // the tracer never supplies would otherwise hold the call open forever.
    let stop_flag = Arc::new(AtomicBool::new(false));
    let finished = Arc::new(AtomicBool::new(false));
    {
        let flag = Arc::clone(&stop_flag);
        let done = Arc::clone(&finished);
        let limit = cfg.max_millis;
        std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_millis(limit);
            while Instant::now() < deadline {
                if done.load(Ordering::Relaxed) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            if !done.load(Ordering::Relaxed) {
                flag.store(true, Ordering::SeqCst);
                let _ = proc::kill_group(pid, proc::signal::SIGKILL);
            }
        });
    }

    let mut tracer = Tracer {
        pid,
        image,
        dec: AnyDecoder::new(image.arch, image.bits),
        table,
        cfg,
        steps: 0,
        path_taken: Vec::new(),
        external_steps: 0,
        calls: BTreeMap::new(),
        syscalls: Vec::new(),
        signals: Vec::new(),
        notes,
        threads: BTreeMap::from([(pid, Thread { tid: pid, steps: 0 })]),
        live: BTreeSet::from([pid]),
        state: BTreeMap::from([(pid, State::Stopped)]),
        pending: BTreeMap::new(),
        deliver: BTreeMap::new(),
        newborn: BTreeSet::new(),
        limit: cfg.max_steps,
        start: started,
        delta,
        entry_bp,
        status: None,
    };

    // Nothing has run yet: either run to the entry breakpoint, or take the
    // first step of the initial thread.
    match entry_bp {
        Some(_) => {
            ptrace::cont(pid, 0)?;
            tracer.state.insert(pid, State::Running);
        }
        None => tracer.resume_all()?,
    }

    let end = tracer.loop_until_done(&stop_flag);
    finished.store(true, Ordering::Relaxed);

    // Whatever happened, nothing is left running. A stopped tracee that the
    // caller has forgotten about is a process leak that outlives the server.
    let _ = proc::kill_group(pid, proc::signal::SIGKILL);
    drop(child);

    let mut threads: Vec<Thread> = tracer.threads.values().copied().collect();
    threads.sort_by_key(|t| t.tid);
    Ok(Trace {
        path: path.display().to_string(),
        argv: {
            let mut v = vec![cfg
                .argv0
                .clone()
                .unwrap_or_else(|| path.display().to_string())];
            v.extend(cfg.args.iter().cloned());
            v
        },
        pid,
        base,
        delta,
        path_taken: tracer.path_taken,
        external_steps: tracer.external_steps,
        calls: tracer
            .calls
            .into_iter()
            .map(
                |((site, target), (count, indirect, site_in, target_in))| CallEdge {
                    site: Addr(site),
                    target: Addr(target),
                    site_in_image: site_in,
                    target_in_image: target_in,
                    count,
                    indirect,
                },
            )
            .collect(),
        syscalls: tracer.syscalls,
        signals: tracer.signals,
        threads,
        steps: tracer.steps,
        end,
        millis: started.elapsed().as_millis() as u64,
        notes: tracer.notes,
    })
}

/// Replace the first byte at `addr` with `int3`, returning the byte replaced.
fn plant(pid: i32, addr: u64) -> Result<u8> {
    let word_addr = addr & !7;
    let word = ptrace::peek_word(pid, word_addr)?;
    let shift = ((addr - word_addr) * 8) as u32;
    let orig = ((word >> shift) & 0xff) as u8;
    let patched = (word & !(0xffu64 << shift)) | (0xccu64 << shift);
    ptrace::poke_word(pid, word_addr, patched)?;
    Ok(orig)
}

/// Put `byte` back at `addr`.
fn unplant(pid: i32, addr: u64, byte: u8) -> Result<()> {
    let word_addr = addr & !7;
    let word = ptrace::peek_word(pid, word_addr)?;
    let shift = ((addr - word_addr) * 8) as u32;
    let patched = (word & !(0xffu64 << shift)) | (u64::from(byte) << shift);
    ptrace::poke_word(pid, word_addr, patched)?;
    Ok(())
}

/// What the instruction about to run will do to the flow of control.
enum PendingKind {
    /// A call; the target is known once the step completes.
    Call { site: Addr, indirect: bool },
    /// A syscall; number and arguments are read before the step, the result
    /// after it. The index points at the record already stored, so a call that
    /// never returns still appears in the trace.
    Syscall { index: usize },
    /// Anything else.
    Plain,
}

/// Whether a thread is running under our single-step or parked at a stop.
#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Running,
    Stopped,
}

struct Tracer<'a> {
    pid: i32,
    image: &'a Image,
    dec: Option<AnyDecoder>,
    table: Vec<Map>,
    cfg: &'a RunConfig,
    steps: u64,
    path_taken: Vec<Addr>,
    external_steps: u64,
    calls: BTreeMap<(u64, u64), (u32, bool, bool, bool)>,
    syscalls: Vec<Syscall>,
    signals: Vec<Signal>,
    notes: Vec<String>,
    /// Every thread that ever ran, with its step count. Kept after exit: the
    /// trace describes what happened, not what is still alive.
    threads: BTreeMap<i32, Thread>,
    /// Threads still running. This is what ends the run.
    live: BTreeSet<i32>,
    state: BTreeMap<i32, State>,
    pending: BTreeMap<i32, PendingKind>,
    /// Signals the tracee has to see, held until the thread is resumed.
    deliver: BTreeMap<i32, i32>,
    /// Threads whose first stop has not arrived yet: their `SIGSTOP` is ours.
    newborn: BTreeSet<i32>,
    limit: u64,
    start: Instant,
    delta: i64,
    entry_bp: Option<(u64, u8)>,
    status: Option<End>,
}

impl Tracer<'_> {
    fn to_image(&self, runtime: u64) -> Addr {
        Addr((runtime as i64 - self.delta) as u64)
    }

    /// The image address for `runtime`, when it belongs to this image.
    ///
    /// A libc address minus this image's load base is a number of the right
    /// magnitude to look like an address, so membership is decided by asking
    /// whether the result lands in a section of the image, not by arithmetic
    /// alone.
    fn in_image(&self, runtime: u64) -> Option<Addr> {
        let a = self.to_image(runtime);
        self.image.section_at(a).map(|_| a)
    }

    /// Note the instruction about to run in `tid`, and classify it.
    fn record(&mut self, tid: i32) -> Result<()> {
        let regs = ptrace::get_regs(tid)?;
        let ip = regs.rip;
        match self.in_image(ip) {
            Some(a) => self.path_taken.push(a),
            None => self.external_steps += 1,
        }
        self.steps += 1;
        if let Some(t) = self.threads.get_mut(&tid) {
            t.steps += 1;
        }
        let kind = self.classify(tid, ip, &regs);
        self.pending.insert(tid, kind);
        Ok(())
    }

    /// Classify the instruction at `ip` for the three cases the tracer records.
    fn classify(&mut self, tid: i32, ip: u64, regs: &Registers) -> PendingKind {
        let Some(dec) = self.dec else {
            return PendingKind::Plain;
        };
        let Some(bytes) = read_mem(tid, ip, 16) else {
            return PendingKind::Plain;
        };
        let Some(Decoded { insn, .. }) = dec.decode(Addr(ip), &bytes) else {
            return PendingKind::Plain;
        };
        use occ_core::FlowKind;
        match insn.flow {
            FlowKind::Call => PendingKind::Call {
                site: Addr(ip),
                indirect: false,
            },
            FlowKind::IndirectCall => PendingKind::Call {
                site: Addr(ip),
                indirect: true,
            },
            FlowKind::Interrupt if self.cfg.trace_syscalls && is_syscall(&insn) => {
                // The number lives in `rax` here. `orig_rax` is only filled in
                // by a syscall-stop, which a single-stepping tracer never sees.
                let nr = regs.rax;
                let args = regs.syscall_args();
                let mut paths = Vec::new();
                for idx in syscall::path_args(nr) {
                    if let Some(s) = read_cstr(tid, args[usize::from(*idx)], self.cfg.max_string) {
                        paths.push(s);
                    }
                }
                let at = Addr(ip);
                let index = self.syscalls.len();
                if index < 4096 {
                    self.syscalls.push(Syscall {
                        nr,
                        name: syscall::name(nr),
                        args,
                        ret: 0,
                        returned: false,
                        at,
                        paths,
                    });
                    PendingKind::Syscall { index }
                } else {
                    PendingKind::Plain
                }
            }
            _ => PendingKind::Plain,
        }
    }

    /// Run `tid` for exactly one instruction, delivering any pending signal.
    fn resume(&mut self, tid: i32) -> Result<()> {
        if self.steps >= self.limit || self.path_taken.len() >= MAX_PATH {
            return Ok(());
        }
        if let Err(e) = self.record(tid) {
            // The thread is gone: a stop that beat us to the resume. Nothing to
            // do but forget it, exactly as an exit stop would.
            self.threads.remove(&tid);
            self.state.remove(&tid);
            self.pending.remove(&tid);
            self.notes.push(format!("thread {tid} vanished: {e}"));
            return Ok(());
        }
        let sig = self.deliver.remove(&tid).unwrap_or(0);
        match ptrace::singlestep(tid, sig) {
            Ok(()) => {
                self.state.insert(tid, State::Running);
            }
            Err(e) => {
                self.threads.remove(&tid);
                self.state.remove(&tid);
                self.pending.remove(&tid);
                self.notes.push(format!("could not resume {tid}: {e}"));
            }
        }
        Ok(())
    }

    /// Run every parked thread for one instruction.
    ///
    /// All of them, not just the one that stopped: a thread blocked in `futex`
    /// waiting for a sibling only makes progress if the sibling runs, and a
    /// tracer that steps one thread at a time deadlocks every `pthread_join` in
    /// existence. Threads that have not yet reported their first stop are left
    /// alone — they are stopped in the kernel, not by us.
    fn resume_all(&mut self) -> Result<()> {
        let parked: Vec<i32> = self
            .state
            .iter()
            .filter(|(_, s)| **s == State::Stopped)
            .map(|(tid, _)| *tid)
            .collect();
        for tid in parked {
            self.resume(tid)?;
        }
        Ok(())
    }

    /// Handle one stop. `Ok(false)` means the run is over.
    fn handle(&mut self, stop: Stop) -> Result<bool> {
        let tid = stop.pid;
        self.state.insert(tid, State::Stopped);
        match stop.status {
            WaitStatus::Exited(code) => {
                self.live.remove(&tid);
                self.state.remove(&tid);
                self.pending.remove(&tid);
                if tid == self.pid {
                    self.status = Some(End::Exited(code));
                    return Ok(false);
                }
                Ok(!self.live.is_empty())
            }
            WaitStatus::Signaled(sig) => {
                self.live.remove(&tid);
                self.state.remove(&tid);
                self.pending.remove(&tid);
                if tid == self.pid || self.live.is_empty() {
                    self.status = Some(End::Signaled(sig));
                    return Ok(false);
                }
                Ok(true)
            }
            WaitStatus::Continued | WaitStatus::StillRunning => Ok(true),
            WaitStatus::Stopped(sig) => {
                if let Some(event) = stop.event {
                    self.on_event(tid, event)?;
                    return Ok(true);
                }
                if sig == ptrace::signal::SIGTRAP {
                    return self.on_trap(tid);
                }
                if sig == ptrace::signal::SIGSTOP && self.newborn.remove(&tid) {
                    // The first stop of a thread we were told about: the SIGSTOP
                    // is the kernel's way of handing it to us, not the program's.
                    return Ok(true);
                }
                // A real signal: record it and hand it back, so the program's
                // own handler or the default action sees it.
                let regs = ptrace::get_regs(tid).unwrap_or_default();
                let fault = ptrace::get_siginfo(tid).ok().and_then(|i| i.addr);
                self.signals.push(Signal {
                    signal: sig,
                    at: self.to_image(regs.rip),
                    fault,
                });
                self.pending.remove(&tid);
                self.deliver.insert(tid, sig);
                Ok(true)
            }
        }
    }

    fn on_event(&mut self, tid: i32, event: Event) -> Result<()> {
        match event {
            Event::Clone | Event::Fork | Event::Vfork => {
                let new = ptrace::get_event_message(tid).unwrap_or(0) as i32;
                if new <= 0 {
                    return Ok(());
                }
                if self.live.len() < 64 {
                    self.threads.entry(new).or_insert(Thread { tid: new, steps: 0 });
                    self.live.insert(new);
                    self.newborn.insert(new);
                } else {
                    self.notes
                        .push(format!("more than 64 threads; {new} runs untraced"));
                }
            }
            Event::Exec => {
                // The image changed, so the translation table did. Re-reading
                // the maps is the only way to keep addresses meaningful.
                if let Ok(m) = maps::read(self.pid) {
                    self.table = m;
                }
                let at = ptrace::get_regs(tid).map_or(0, |r| r.rip);
                self.notes.push(format!(
                    "the tracee called exec at {}; addresses after this point belong to the new image",
                    self.to_image(at)
                ));
            }
            Event::Exit | Event::Other(_) => {}
        }
        Ok(())
    }

    fn on_trap(&mut self, tid: i32) -> Result<bool> {
        let regs = ptrace::get_regs(tid)?;
        // A breakpoint we planted reports the trap *after* the `int3`, so the
        // instruction pointer is one byte past it. Put the real byte back and
        // rewind, or the program would execute `int3` and die.
        if let Some((at, orig)) = self.entry_bp
            && regs.rip == at + 1
        {
            unplant(tid, at, orig)?;
            let mut fixed = regs;
            fixed.rip = at;
            ptrace::set_regs(tid, &fixed)?;
            self.pending.remove(&tid);
            self.entry_bp = None;
            return Ok(true);
        }

        match self.pending.remove(&tid) {
            Some(PendingKind::Call { site, indirect }) => {
                let site_in = self.in_image(site.0);
                let target_in = self.in_image(regs.rip);
                // Both coordinates are kept as the image would name them when
                // they are the image's, and as the raw runtime address when they
                // are not. Mixing the two would make an edge into libc look like
                // an edge into a low part of the file.
                let site_key = site_in.unwrap_or(site);
                let target_key = target_in.unwrap_or(Addr(regs.rip));
                let slot = self
                    .calls
                    .entry((site_key.0, target_key.0))
                    .or_insert((0, indirect, site_in.is_some(), target_in.is_some()));
                slot.0 += 1;
                slot.1 |= indirect;
                slot.2 &= site_in.is_some();
                slot.3 &= target_in.is_some();
            }
            Some(PendingKind::Syscall { index }) => {
                let mapped = self
                    .syscalls
                    .get(index)
                    .map(|s| self.in_image(s.at.0).unwrap_or(s.at));
                if let (Some(mapped), Some(s)) = (mapped, self.syscalls.get_mut(index)) {
                    s.ret = regs.rax as i64;
                    s.returned = true;
                    s.at = mapped;
                }
            }
            Some(PendingKind::Plain) | None => {}
        }
        Ok(true)
    }

    fn loop_until_done(&mut self, stop_flag: &AtomicBool) -> End {
        loop {
            if stop_flag.load(Ordering::SeqCst) {
                self.status = Some(End::TimeLimit);
                break;
            }
            if self.steps >= self.limit || self.path_taken.len() >= MAX_PATH {
                self.status = Some(End::StepLimit);
                break;
            }
            if self.start.elapsed() > Duration::from_millis(self.cfg.max_millis) {
                self.status = Some(End::TimeLimit);
                break;
            }
            if self.live.is_empty() {
                break;
            }
            let stop = match ptrace::wait_any() {
                Ok(s) => s,
                Err(e) => {
                    self.status = Some(End::Failed(format!("{e}")));
                    break;
                }
            };
            match self.handle(stop) {
                Ok(true) => {}
                Ok(false) => break,
                Err(e) => {
                    self.status = Some(End::Failed(format!("{e}")));
                    break;
                }
            }
            if let Err(e) = self.resume_all() {
                self.status = Some(End::Failed(format!("{e}")));
                break;
            }
        }
        self.status.take().unwrap_or(End::StepLimit)
    }
}

/// True when this instruction enters the kernel.
fn is_syscall(insn: &occ_core::Instruction) -> bool {
    match &*insn.mnemonic {
        "syscall" | "sysenter" => true,
        "int" => insn.text.contains("0x80"),
        _ => false,
    }
}

/// Read up to `len` bytes of the tracee's memory, starting at `addr`.
fn read_mem(pid: i32, addr: u64, len: usize) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(len + 8);
    let mut at = addr;
    while out.len() < len {
        let word = ptrace::peek_word(pid, at).ok()?;
        out.extend_from_slice(&word.to_le_bytes());
        at = at.checked_add(8)?;
    }
    out.truncate(len);
    Some(out)
}

/// Read a NUL-terminated string from the tracee, bounded by `limit`.
fn read_cstr(pid: i32, addr: u64, limit: usize) -> Option<String> {
    if addr == 0 {
        return None;
    }
    let mut bytes = Vec::new();
    let mut at = addr;
    while bytes.len() < limit {
        let word = ptrace::peek_word(pid, at).ok()?;
        for b in word.to_le_bytes() {
            if b == 0 {
                let text = String::from_utf8_lossy(&bytes).into_owned();
                return (!text.is_empty()).then_some(text);
            }
            bytes.push(b);
            if bytes.len() >= limit {
                break;
            }
        }
        at = at.checked_add(8)?;
    }
    let text = String::from_utf8_lossy(&bytes).into_owned();
    (!text.is_empty()).then_some(text)
}
