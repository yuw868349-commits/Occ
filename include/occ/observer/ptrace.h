#pragma once

// The ptrace control plane.
//
// A tracer here is a process that has stopped another process and is
// deciding what it does next. Everything the observer layer does is built on
// this one primitive: a breakpoint is a stop that the tracer arranged, a
// syscall trace is a stop the kernel makes on the tracer's behalf, and a
// memory write is a poke into a stopped process.
//
// The stop states are handled as the small state machine they are. A
// sequence of waitpid calls does not tell a tracer why it stopped; the
// tracer has to ask, and the answers are different kinds of stop that
// require different handling. Treating them as one case is how a tracer ends
// up silently swallowing a signal it was supposed to deliver.
//
// Nothing here is a wrapper around a library. ptrace has six requests and a
// shared memory layout, and the whole of it is a handful of constants and
// one union.

#include <cstdint>
#include <string>
#include <vector>

#include "occ/syscall/syscall.h"

namespace occ::obs {

// ------------------------------------------------------------------ requests
//
// Restated rather than taken from <sys/ptrace.h>, because glibc's header
// renames several of these and the renames differ between architectures.
// The values are the kernel's and are stable.

inline constexpr unsigned long kPtraceTraceme = 0;
inline constexpr unsigned long kPtracePeekText = 1;
inline constexpr unsigned long kPtracePeekData = 2;
inline constexpr unsigned long kPtracePeekUser = 3;
inline constexpr unsigned long kPtracePokeText = 4;
inline constexpr unsigned long kPtracePokeData = 5;
inline constexpr unsigned long kPtracePokeUser = 6;
inline constexpr unsigned long kPtraceCont = 7;
inline constexpr unsigned long kPtraceKill = 8;
inline constexpr unsigned long kPtraceSinglestep = 9;
inline constexpr unsigned long kPtraceGetregs = 12;
inline constexpr unsigned long kPtraceSetregs = 13;
inline constexpr unsigned long kPtraceAttach = 16;
inline constexpr unsigned long kPtraceDetach = 17;
inline constexpr unsigned long kPtraceSyscall = 24;
inline constexpr unsigned long kPtraceSetoptions = 0x4200;
inline constexpr unsigned long kPtraceGetsiginfo = 0x4202;
inline constexpr unsigned long kPtraceSetsiginfo = 0x4203;
inline constexpr unsigned long kPtraceGettid = 0x4204;
inline constexpr unsigned long kPtraceSeize = 0x4206;
inline constexpr unsigned long kPtraceInterrupt = 0x4207;
inline constexpr unsigned long kPtraceListen = 0x4208;
inline constexpr unsigned long kPtraceGeteventmsg = 0x4205;
inline constexpr unsigned long kPtracePeekuserAsText = 0x420b;
inline constexpr unsigned long kPtracePokeuserAsText = 0x420c;

// ------------------------------------------------------------------- options

inline constexpr unsigned long kPtraceO_TRACESYSGOOD = 0x00000001;
inline constexpr unsigned long kPtraceO_TRACEFORK = 0x00000002;
inline constexpr unsigned long kPtraceO_TRACEVFORK = 0x00000004;
inline constexpr unsigned long kPtraceO_TRACECLONE = 0x00000008;
inline constexpr unsigned long kPtraceO_TRACEEXEC = 0x00000010;
inline constexpr unsigned long kPtraceO_TRACEVFORKDONE = 0x00000020;
inline constexpr unsigned long kPtraceO_TRACEEXIT = 0x00000040;
inline constexpr unsigned long kPtraceO_TRACESECCOMP = 0x00000080;
inline constexpr unsigned long kPtraceO_EXITKILL = 0x00100000;
inline constexpr unsigned long kPtraceO_SUSPEND_SECCOMP = 0x00200000;

// The events a stop can carry. These arrive as the status the kernel
// reports, in a range that overlaps the signal numbers, which is why a stop
// has to be classified rather than compared against a signal directly.
inline constexpr int kPtraceEventFork = 1;
inline constexpr int kPtraceEventVfork = 2;
inline constexpr int kPtraceEventClone = 3;
inline constexpr int kPtraceEventExec = 4;
inline constexpr int kPtraceEventVforkDone = 5;
inline constexpr int kPtraceEventExit = 6;
inline constexpr int kPtraceEventSeccomp = 7;
inline constexpr int kPtraceEventStop = 128;
inline constexpr int kSigsys = 31;

// ----------------------------------------------------------------- the state

// Why a trace stopped. Each of these needs different handling, and the
// distinction is the whole reason a tracer is more than a waitpid loop.
enum class StopKind : std::uint8_t {
    // The process was stopped by ptrace or by an arriving signal.
    Signal,
    // A syscall entry or exit stop, requested with PTRACE_SYSCALL.
    SyscallStop,
    // The process was created by fork/vfork/clone and is traced by
    // inheritance. It has not run yet and has to be configured.
    NewChild,
    // An exec completed. The image in memory is no longer the one the
    // tracer was looking at, so every cached address is stale.
    Exec,
    // A seccomp filter with SECCOMP_RET_TRAP fired, or a filter installed
    // with PTRACE_O_TRACESECCOMP matched.
    SeccompTrap,
    // A group-stop, which is a different kind of stop from a signal-stop and
    // has to be continued with a different call.
    GroupStop,
    // The tracee exited, or is a zombie waiting to be reaped.
    Exited,
};

[[nodiscard]] const char* stop_kind_name(StopKind k) noexcept;

// The register file as the kernel reports it for a 64-bit tracee. The
// layout is the order of struct user_regs_struct, which is ABI and does not
// change.
struct Registers {
    std::uint64_t r15, r14, r13, r12, rbp, rbx;
    std::uint64_t r11, r10, r9, r8;
    std::uint64_t rax, rcx, rdx, rsi, rdi;
    std::uint64_t orig_rax;
    std::uint64_t rip, cs, eflags, rsp, ss;
    std::uint64_t fs_base, gs_base, ds, es, fs, gs;
};

namespace detail {
struct Iovec {
    void* base;
    std::size_t len;
};
} // namespace detail

// The outcome of one wait. The three fields are kept apart because a
// consumer that conflates them will mishandle the stop: a signal number is
// not an event, and an exit code is not a signal.
struct Stop {
    StopKind kind = StopKind::Exited;

    // The process the stop is about. For a NewChild stop this is the child,
    // not the process that was waited on, which is how a tracer learns the
    // pid it now has to configure.
    int pid = 0;

    // For a Signal or GroupStop, the signal. For a SeccompTrap this is the
    // guard value the filter returned.
    int signal = 0;

    // True when the signal was generated by the kernel for ptrace's benefit
    // (SIGTRAP and the event stops) rather than delivered to the tracee.
    // A tracer that redelivers one will kill the process.
    bool synthetic = false;

    // True for a syscall-entry stop, false for a syscall-exit stop. Set only
    // when the kind is SyscallStop.
    bool syscall_entry = false;

    // The exit status for an Exited stop, and whether it was a signal.
    int exit_code = 0;
    bool term_signal = false;

    // The raw status word, which is what PTRACE_GETEVENTMSG and the
    // siginfo-free paths need.
    int raw_status = 0;
};

class Tracer {
public:
    Tracer() = default;

    // Takes control of an existing process. Seize rather than Attach: seize
    // works on a process this one did not create and does not fail on a
    // process that is already stopped, which is the case when the container
    // hands over a target that is parked.
    [[nodiscard]] sys::Result seize(int pid) noexcept;
    [[nodiscard]] sys::Result attach(int pid) noexcept;

    // The options every tracer here wants. EXITKILL is included because a
    // tracer that dies and leaves a tracee stopped forever has left the
    // system in a worse state than one that kills it.
    [[nodiscard]] sys::Result set_options(int pid,
                                          unsigned long options) noexcept;

    [[nodiscard]] sys::Result cont(int pid, int signal = 0) noexcept;
    [[nodiscard]] sys::Result syscall(int pid, int signal = 0) noexcept;
    [[nodiscard]] sys::Result singlestep(int pid, int signal = 0) noexcept;
    [[nodiscard]] sys::Result detach(int pid, int signal = 0) noexcept;
    [[nodiscard]] sys::Result kill(int pid) noexcept;
    [[nodiscard]] sys::Result interrupt(int pid) noexcept;

    [[nodiscard]] sys::Result get_regs(int pid, Registers& out) noexcept;
    [[nodiscard]] sys::Result set_regs(int pid, const Registers& in) noexcept;

    // Reading one word at a time is slow and, on a process that is being
    // traced across a syscall boundary, can observe a torn value. It is the
    // fallback for when process_vm_readv is refused.
    [[nodiscard]] sys::Result peek(int pid, std::uint64_t addr) noexcept;
    [[nodiscard]] sys::Result poke(int pid, std::uint64_t addr,
                                   std::uint64_t word) noexcept;

    // Moves a block of memory in or out of the tracee. Uses process_vm_*
    // where the kernel allows it and falls back to word-at-a-time ptrace.
    // Returns the number of bytes moved, or a negative errno.
    [[nodiscard]] sys::Result read_memory(int pid, std::uint64_t addr,
                                          void* buffer,
                                          std::size_t length) noexcept;
    [[nodiscard]] sys::Result write_memory(int pid, std::uint64_t addr,
                                           const void* buffer,
                                           std::size_t length) noexcept;
    [[nodiscard]] sys::Result read_cstring(int pid, std::uint64_t addr,
                                           std::string& out,
                                           std::size_t limit) noexcept;

    // Waits for the next event from any traced process.
    [[nodiscard]] Stop wait(int options = 0) noexcept;
    // Waits for a specific traced process.
    [[nodiscard]] Stop wait_pid(int pid, int options = 0) noexcept;

    // The pid a NewChild stop created. Only valid for that kind of stop.
    [[nodiscard]] sys::Result event_message(int pid) noexcept;

    // The siginfo of the stop currently being handled, which is where the
    // faulting address and the trap number live for a hardware breakpoint.
    [[nodiscard]] sys::Result siginfo(int pid, void* out,
                                      std::size_t size) noexcept;
};

// Classifies a raw wait status into the stop the tracer has to handle.
[[nodiscard]] Stop classify_status(int pid, int status) noexcept;

// ---------------------------------------------------------------- breakpoints
//
// A software breakpoint replaces one byte with 0xcc and remembers what was
// there. It is invisible to the tracee's own code until that byte executes,
// and it has to be removed and put back around every single step, which is
// why the tracker below keeps the save state next to the address.

struct SoftwareBreakpoint {
    std::uint64_t address = 0;
    // The byte that was at the address before the breakpoint was installed.
    std::uint8_t saved = 0;
    bool armed = false;
};

// Installs and removes software breakpoints on a stopped tracee. The
// breakpoint list is owned by the caller so that one tracker can serve
// several processes, which is what a fork-tracing session needs.
class Breakpoints {
public:
    // Writes 0xcc at `address` and records the byte it replaced. The tracee
    // has to be stopped.
    [[nodiscard]] int add(Tracer& tracer, int pid, std::uint64_t address,
                          std::string& detail) noexcept;

    // Puts `address` back to its original byte.
    [[nodiscard]] int remove(Tracer& tracer, int pid,
                             std::uint64_t address) noexcept;

    // Puts every byte back without forgetting the breakpoints. Used to step
    // over a breakpoint: the instruction has to run, which means the byte
    // has to be the original one for exactly one step.
    [[nodiscard]] int disarm_all(Tracer& tracer, int pid) noexcept;

    // Re-installs every breakpoint after a disarm. Addresses that no longer
    // hold a breakpoint are skipped.
    [[nodiscard]] int rearm_all(Tracer& tracer, int pid) noexcept;

    // The breakpoint at `address`, or nullptr. Used to find the original
    // byte when a single step has run over one.
    [[nodiscard]] const SoftwareBreakpoint* find(
        std::uint64_t address) const noexcept;

    // Reduces rip past a software breakpoint. A software breakpoint stops
    // the tracee with rip pointing one past the 0xcc, so a resume without
    // this adjustment would re-execute the following instruction and skip
    // the one at the breakpoint address.
    [[nodiscard]] int step_over(Tracer& tracer, int pid,
                                std::string& detail) noexcept;

    [[nodiscard]] std::size_t size() const noexcept { return points_.size(); }
    [[nodiscard]] const std::vector<SoftwareBreakpoint>& all() const noexcept {
        return points_;
    }

private:
    std::vector<SoftwareBreakpoint> points_;
};

} // namespace occ::obs
