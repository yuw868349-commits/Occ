//! Executing one instruction at a time under `ptrace`.
//!
//! This exists to answer one question: given a register file and a memory
//! image, what does the processor do with these bytes? A disassembler says what
//! the bytes mean; only the processor says what they do. Comparing the two is
//! how a lifter's semantics get checked instead of assumed.
//!
//! # Shape of a run
//!
//! A child is forked once and reused. Before the fork the parent maps two
//! shared regions: a small executable one for the instruction, and a larger
//! writable one that the stack pointer aims into. Every step is then
//! `PTRACE_SETREGS` (set the state), `PTRACE_SINGLESTEP`, `wait`, `PTRACE_GETREGS`
//! (read the state). No process is created per instruction, which matters when
//! a sweep runs tens of thousands of them.
//!
//! # What a run will not do
//!
//! An instruction that faults loses the child: the tracer sees `SIGSEGV` and
//! the process is killed rather than resumed, because resuming from a fault
//! with registers that were chosen for a different question proves nothing.
//! The next call forks a new child. The caller sees `Sample::fault` and decides
//! whether the fault was the point of the test.

use occ_sys::ptrace::{self, Registers};
use occ_sys::spawn::{self, Shared};
use occ_sys::{WaitStatus, signal};

/// Size of the executable page holding one instruction.
const CODE_LEN: usize = 4096;
/// Size of the writable region the stack and data operands point into.
const DATA_LEN: usize = 64 * 1024;
/// Offset into the data region used as the initial stack pointer, leaving room
/// for a call to push and a function to use a frame.
pub const STACK_TOP: usize = DATA_LEN / 2;

/// Base address of the first fixed slot.
///
/// A fixed address rather than one the kernel picks, because the address ends
/// up inside the register values the caller feeds the traced process. Where
/// the region lands decides whether a division overflows and whether a memory
/// operand lands inside the region, so an address that moves between runs
/// means the same seed tests a different set of encodings each time. A
/// differential run that cannot be repeated cannot be trusted to have found
/// everything.
///
/// Slots are handed out in order, so the first probe in a process always gets
/// the same pair of addresses no matter how many probes came before it in
/// another process. That is what makes a run repeatable.
const SLOT_BASE: u64 = 0x0000_5000_0000_0000;
/// Layout of one slot: the code region, then the data region, then the next
/// slot's code region.
///
/// The two regions of a slot are 128 KiB apart and the slot is 256 KiB
/// long, so neither region can grow into the other or into the next slot.
const SLOT_CODE_OFFSET: u64 = 0x0000_0000_0000_0000;
const SLOT_DATA_OFFSET: u64 = 0x0000_0000_0002_0000;
const SLOT_STRIDE: u64 = 0x0000_0000_0004_0000;
/// The next slot to hand out.
static NEXT_SLOT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A child process that executes one instruction on request.
pub struct Probe {
    pid: i32,
    code: Shared,
    data: Shared,
    alive: bool,
    /// True when both regions are at their fixed addresses.
    reproducible: bool,
    /// Instructions executed since the child was forked.
    pub steps: u64,
    /// Children forked so far, including replacements after a fault.
    pub forks: u64,
}

/// Map a region at `addr`, falling back to wherever the kernel puts it.
///
/// The fallback exists because a fixed address can be taken by something else
/// in the process — a debugger, a sanitizer, a library that reserves a range.
/// The returned flag says which happened, so the caller can report a run that
/// is not reproducible rather than presenting it as one.
fn map_region(addr: u64, len: usize, exec: bool) -> Result<(Shared, bool), String> {
    match Shared::map_at(Some(addr), len, exec) {
        Ok(region) => Ok((region, true)),
        Err(_) => Shared::map(len, exec)
            .map(|region| (region, false))
            .map_err(|e| e.to_string()),
    }
}

/// What one step produced.
#[derive(Clone, Debug)]
pub struct Sample {
    /// Registers after the instruction.
    pub regs: Registers,
    /// Signal the instruction raised, when it raised one.
    pub fault: Option<i32>,
    /// True when the step completed and the registers are meaningful.
    pub ok: bool,
}

impl Probe {
    /// Fork a child and wait for it to stop.
    ///
    /// The two regions are mapped at fixed addresses so that a run is
    /// reproducible: see [`CODE_ADDR`]. If a fixed address is not available
    /// the region is mapped wherever the kernel puts it, and the run still
    /// works — it just stops being comparable with a previous one, which is
    /// why [`Probe::is_reproducible`] says so out loud.
    pub fn new() -> Result<Probe, String> {
        // The slot counter starts at zero in every process, so the first
        // probe of a run always takes the same addresses. A second probe in
        // the same process takes the next slot, because the first one still
        // holds the first slot's pages.
        let slot = NEXT_SLOT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let base = SLOT_BASE + slot * SLOT_STRIDE;
        let (code, code_fixed) = map_region(base + SLOT_CODE_OFFSET, CODE_LEN, true)?;
        let (data, data_fixed) = map_region(base + SLOT_DATA_OFFSET, DATA_LEN, false)?;
        let mut p = Probe {
            pid: 0,
            code,
            data,
            alive: false,
            steps: 0,
            forks: 0,
            reproducible: code_fixed && data_fixed,
        };
        p.spawn()?;
        Ok(p)
    }

    /// True when both regions sit at their fixed addresses.
    ///
    /// A run that could not take the fixed addresses is still correct — the
    /// processor and the model are still compared on the same input — but the
    /// set of encodings it covers depends on where the kernel put the
    /// regions, so two runs are not the same run. A caller that compares runs
    /// needs to know.
    pub fn is_reproducible(&self) -> bool {
        self.reproducible
    }

    /// Base address of the writable region, for building operand values.
    pub fn data_base(&self) -> u64 {
        self.data.base()
    }

    /// Address of the stack pointer's initial value.
    pub fn stack_addr(&self) -> u64 {
        self.data.base() + STACK_TOP as u64
    }

    /// Read the writable region through the parent's mapping.
    pub fn data(&self) -> &[u8] {
        self.data.read(0, DATA_LEN)
    }

    /// Write the writable region through the parent's mapping.
    pub fn set_data(&mut self, bytes: &[u8]) {
        let n = bytes.len().min(DATA_LEN);
        self.data.write(0, &bytes[..n]);
    }

    /// Fork a replacement child.
    fn spawn(&mut self) -> Result<(), String> {
        let pid = spawn::fork_stopped_child().map_err(|e| e.to_string())?;
        match ptrace::wait(pid) {
            Ok(WaitStatus::Stopped(s)) if s == signal::SIGSTOP => {
                self.pid = pid;
                self.alive = true;
                self.forks += 1;
                Ok(())
            }
            Ok(other) => Err(format!("child stopped unexpectedly: {other:?}")),
            Err(e) => Err(e.to_string()),
        }
    }

    /// Kill the current child, if any.
    fn reap(&mut self) {
        if self.alive {
            let _ = ptrace::kill_tracee(self.pid);
            let _ = ptrace::wait(self.pid);
            self.alive = false;
        }
    }

    /// Execute `code` from a state read out of `regs`.
    ///
    /// `regs` supplies every general-purpose register except `rip` and `rsp`,
    /// which this function sets: `rip` to the code page and `rsp` to
    /// [`Probe::stack_addr`]. `eflags` is taken as given, which is how a test
    /// asks what `adc` does when the carry flag is already set.
    pub fn step(&mut self, code: &[u8], regs: &Registers) -> Result<Sample, String> {
        if !self.alive {
            self.spawn()?;
        }
        if code.len() > CODE_LEN {
            return Err(format!("instruction of {} bytes does not fit", code.len()));
        }
        self.code.write(0, code);

        let mut want = *regs;
        want.rip = self.code.base();
        want.rsp = self.stack_addr();
        // The segment selectors and thread pointers belong to the child; taking
        // them from the caller would be a fault invented by the test.
        if let Ok(current) = ptrace::get_regs(self.pid) {
            want.cs = current.cs;
            want.ss = current.ss;
            want.fs_base = current.fs_base;
            want.gs_base = current.gs_base;
        }
        if ptrace::set_regs(self.pid, &want).is_err() {
            self.reap();
            return Err("PTRACE_SETREGS failed".into());
        }
        if ptrace::singlestep(self.pid, 0).is_err() {
            self.reap();
            return Err("PTRACE_SINGLESTEP failed".into());
        }

        let stop = match ptrace::wait(self.pid) {
            Ok(s) => s,
            Err(e) => {
                self.reap();
                return Err(e.to_string());
            }
        };
        self.steps += 1;

        let fault = match stop {
            WaitStatus::Stopped(sig) if sig == signal::SIGTRAP => None,
            WaitStatus::Stopped(sig) => Some(sig),
            WaitStatus::Exited(code) => {
                self.alive = false;
                return Ok(Sample {
                    regs: Registers::default(),
                    fault: Some(-code),
                    ok: false,
                });
            }
            _ => None,
        };

        let regs = ptrace::get_regs(self.pid).unwrap_or_default();
        if fault.is_some() {
            // A faulted child cannot be resumed into a meaningful state.
            self.reap();
        }
        Ok(Sample {
            regs,
            fault,
            ok: fault.is_none(),
        })
    }

    /// Kill the child.
    pub fn close(mut self) {
        self.reap();
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        self.reap();
    }
}

/// True when a step stopped for a reason other than the single-step trap.
pub fn failed(sample: &Sample) -> bool {
    sample.fault.is_some() || !sample.ok
}
