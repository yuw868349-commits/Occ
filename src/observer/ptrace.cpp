#include "occ/observer/ptrace.h"

#include "occ/syscall/errno.h"

#include <cerrno>

#include <sys/uio.h>
#include <sys/wait.h>

namespace occ::obs {

namespace {

// libc's waitpid is used rather than a hand-written wait4 for one reason:
// a tracer's whole job is to be interrupted by signals, and a hand-written
// wait4 loop that mishandles EINTR would lose a stop, which is a hang. The
// wrapper here retries explicitly, which is what makes it correct rather
// than what makes it short.
int wait_status(int pid, int options, int& status) {
    for (;;) {
        const int r = ::waitpid(pid, &status, options);
        if (r >= 0) {
            return r;
        }
        if (errno != EINTR) {
            return -errno;
        }
    }
}

// SIGTRAP is what ptrace uses for its own events unless the stop is one the
// kernel distinguishes some other way.
constexpr int kSigtrap = 5;
constexpr int kSigstop = 19;

} // namespace

const char* stop_kind_name(StopKind k) noexcept {
    switch (k) {
    case StopKind::Signal:
        return "signal";
    case StopKind::SyscallStop:
        return "syscall";
    case StopKind::NewChild:
        return "new-child";
    case StopKind::Exec:
        return "exec";
    case StopKind::SeccompTrap:
        return "seccomp-trap";
    case StopKind::GroupStop:
        return "group-stop";
    case StopKind::Exited:
        return "exited";
    }
    return "unknown";
}

Stop classify_status(int pid, int status) noexcept {
    Stop out;
    out.pid = pid;
    out.raw_status = status;

    if (WIFEXITED(status)) {
        out.kind = StopKind::Exited;
        out.exit_code = WEXITSTATUS(status);
        return out;
    }
    if (WIFSIGNALED(status)) {
        out.kind = StopKind::Exited;
        out.term_signal = true;
        out.exit_code = WTERMSIG(status);
        return out;
    }

    // Anything below here is a stop, not an exit. The order of the tests
    // matters: the event stops share a number range with the signals, so a
    // stop has to be checked for an event before it is read as a signal.
    const int stopsig = WSTOPSIG(status);

    if (stopsig == (kSigtrap | 0x80)) {
        // PTRACE_O_TRACESYSGOOD sets the high bit of SIGTRAP on a syscall
        // stop, which is the only way to tell one from a real SIGTRAP. The
        // entry/exit distinction is not in the status; it is derived from
        // the architecture's syscall state, which the caller tracks.
        out.kind = StopKind::SyscallStop;
        out.signal = kSigtrap;
        out.synthetic = true;
        return out;
    }

    if (stopsig == kSigtrap) {
        // Either a real SIGTRAP delivered to the tracee, or one of the
        // event stops. The event is in the high byte of the status.
        const int event = status >> 8;
        switch (event) {
        case kPtraceEventFork:
        case kPtraceEventVfork:
        case kPtraceEventClone:
            out.kind = StopKind::NewChild;
            out.synthetic = true;
            return out;
        case kPtraceEventExec:
            out.kind = StopKind::Exec;
            out.synthetic = true;
            return out;
        case kPtraceEventSeccomp:
            out.kind = StopKind::SeccompTrap;
            out.synthetic = true;
            return out;
        case kPtraceEventExit:
            // The tracee has exited and this is its final event. It is not
            // reaped yet; that happens on the next wait.
            out.kind = StopKind::Exited;
            out.synthetic = true;
            return out;
        case kPtraceEventVforkDone:
            // The vfork has completed. Nothing to do, but it must still be
            // continued or the tracee stays stopped.
            out.kind = StopKind::Signal;
            out.synthetic = true;
            out.signal = 0;
            return out;
        default:
            break;
        }
        // A plain SIGTRAP with no event is a breakpoint or a single step.
        out.kind = StopKind::Signal;
        out.signal = kSigtrap;
        out.synthetic = true;
        return out;
    }

    if (stopsig == kSigstop) {
        // SIGSTOP is how a group-stop arrives. A group-stop is continued
        // with PTRACE_CONT and a signal of 0, or with SIGCONT to actually
        // deliver the stop, and getting that wrong leaves the tracee in a
        // state where it ignores the next SIGCONT.
        out.kind = StopKind::GroupStop;
        out.signal = kSigstop;
        return out;
    }

    out.kind = StopKind::Signal;
    out.signal = stopsig;
    return out;
}

// ------------------------------------------------------------------ requests

namespace {

sys::Result request(unsigned long req, int pid, void* addr, void* data) {
    return sys::ptrace(req, pid, addr, data);
}

} // namespace

sys::Result Tracer::seize(int pid) noexcept {
    return request(kPtraceSeize, pid, nullptr, nullptr);
}

sys::Result Tracer::attach(int pid) noexcept {
    return request(kPtraceAttach, pid, nullptr, nullptr);
}

sys::Result Tracer::set_options(int pid, unsigned long options) noexcept {
    // The kernel expects the options in the data argument, not in addr.
    return request(kPtraceSetoptions, pid, nullptr,
                   reinterpret_cast<void*>(static_cast<std::uintptr_t>(options)));
}

sys::Result Tracer::cont(int pid, int signal) noexcept {
    return request(kPtraceCont, pid, nullptr,
                   reinterpret_cast<void*>(static_cast<std::uintptr_t>(signal)));
}

sys::Result Tracer::syscall(int pid, int signal) noexcept {
    return request(kPtraceSyscall, pid, nullptr,
                   reinterpret_cast<void*>(static_cast<std::uintptr_t>(signal)));
}

sys::Result Tracer::singlestep(int pid, int signal) noexcept {
    return request(kPtraceSinglestep, pid, nullptr,
                   reinterpret_cast<void*>(static_cast<std::uintptr_t>(signal)));
}

sys::Result Tracer::detach(int pid, int signal) noexcept {
    return request(kPtraceDetach, pid, nullptr,
                   reinterpret_cast<void*>(static_cast<std::uintptr_t>(signal)));
}

sys::Result Tracer::kill(int pid) noexcept {
    return request(kPtraceKill, pid, nullptr, nullptr);
}

sys::Result Tracer::interrupt(int pid) noexcept {
    return request(kPtraceInterrupt, pid, nullptr, nullptr);
}

sys::Result Tracer::get_regs(int pid, Registers& out) noexcept {
    return request(kPtraceGetregs, pid, nullptr, &out);
}

sys::Result Tracer::set_regs(int pid, const Registers& in) noexcept {
    return request(kPtraceSetregs, pid, nullptr,
                   const_cast<Registers*>(&in));
}

sys::Result Tracer::peek(int pid, std::uint64_t addr) noexcept {
    return request(kPtracePeekData, pid,
                   reinterpret_cast<void*>(static_cast<std::uintptr_t>(addr)),
                   nullptr);
}

sys::Result Tracer::poke(int pid, std::uint64_t addr,
                         std::uint64_t word) noexcept {
    return request(kPtracePokeData, pid,
                   reinterpret_cast<void*>(static_cast<std::uintptr_t>(addr)),
                   reinterpret_cast<void*>(static_cast<std::uintptr_t>(word)));
}

sys::Result Tracer::read_memory(int pid, std::uint64_t addr, void* buffer,
                                std::size_t length) noexcept {
    if (length == 0) {
        return sys::Result{0, 0};
    }

    // process_vm_readv is one syscall and does not require the target to be
    // stopped at a syscall boundary, which ptrace's word-at-a-time interface
    // does. It is tried first and the fallback exists for the kernels and
    // the permission configurations that refuse it.
    struct iovec local {};
    local.iov_base = buffer;
    local.iov_len = length;
    struct iovec remote {};
    remote.iov_base = reinterpret_cast<void*>(static_cast<std::uintptr_t>(addr));
    remote.iov_len = length;

    auto r = sys::process_vm_readv(pid, &local, 1, &remote, 1, 0);
    if (r.ok()) {
        return r;
    }
    // A short read is reported as the count that did move, which is what a
    // caller looping for a whole buffer needs.
    if (r.error != sys::kEperm && r.error != sys::kEinval) {
        return r;
    }

    auto* out = static_cast<std::uint8_t*>(buffer);
    std::size_t moved = 0;
    while (moved < length) {
        const std::uint64_t at = addr + moved;
        auto w = peek(pid, at);
        if (w.failed()) {
            if (moved == 0) {
                return w;
            }
            break;
        }
        const std::size_t chunk = length - moved < 8 ? length - moved : 8;
        for (std::size_t i = 0; i < chunk; ++i) {
            out[moved + i] =
                static_cast<std::uint8_t>((w.value >> (8 * i)) & 0xff);
        }
        moved += chunk;
    }
    return sys::Result{static_cast<long>(moved), 0};
}

sys::Result Tracer::write_memory(int pid, std::uint64_t addr,
                                 const void* buffer,
                                 std::size_t length) noexcept {
    if (length == 0) {
        return sys::Result{0, 0};
    }

    struct iovec local {};
    local.iov_base = const_cast<void*>(buffer);
    local.iov_len = length;
    struct iovec remote {};
    remote.iov_base = reinterpret_cast<void*>(static_cast<std::uintptr_t>(addr));
    remote.iov_len = length;

    auto r = sys::process_vm_writev(pid, &local, 1, &remote, 1, 0);
    if (r.ok()) {
        return r;
    }
    if (r.error != sys::kEperm && r.error != sys::kEinval) {
        return r;
    }

    const auto* in = static_cast<const std::uint8_t*>(buffer);
    std::size_t moved = 0;
    while (moved < length) {
        // ptrace writes a whole word, so a partial tail has to be read,
        // merged and written back rather than written directly. Writing
        // only the tail bytes would clobber the rest of the word.
        const std::uint64_t at = addr + moved;
        const std::size_t chunk = length - moved < 8 ? length - moved : 8;

        std::uint64_t word = 0;
        if (chunk < 8) {
            auto cur = peek(pid, at);
            if (cur.failed()) {
                if (moved == 0) {
                    return cur;
                }
                break;
            }
            word = static_cast<std::uint64_t>(cur.value);
        }
        for (std::size_t i = 0; i < chunk; ++i) {
            const std::uint64_t mask = 0xffULL << (8 * i);
            word = (word & ~mask) |
                   ((static_cast<std::uint64_t>(in[moved + i]) << (8 * i)) &
                    mask);
        }

        auto w = poke(pid, at, word);
        if (w.failed()) {
            if (moved == 0) {
                return w;
            }
            break;
        }
        moved += chunk;
    }
    return sys::Result{static_cast<long>(moved), 0};
}

sys::Result Tracer::read_cstring(int pid, std::uint64_t addr, std::string& out,
                                 std::size_t limit) noexcept {
    out.clear();
    std::uint64_t at = addr;
    // Read in word-sized chunks rather than byte at a time: a path is the
    // most common thing read this way and a word per byte makes a syscall
    // storm out of every one.
    while (out.size() < limit) {
        auto w = peek(pid, at);
        if (w.failed()) {
            return out.empty() ? w : sys::Result{static_cast<long>(out.size()), 0};
        }
        for (int i = 0; i < 8; ++i) {
            const char c =
                static_cast<char>((static_cast<std::uint64_t>(w.value) >>
                                   (8 * i)) &
                                  0xff);
            if (c == '\0') {
                return sys::Result{static_cast<long>(out.size()), 0};
            }
            out.push_back(c);
            if (out.size() >= limit) {
                return sys::Result{static_cast<long>(out.size()), 0};
            }
        }
        at += 8;
    }
    return sys::Result{static_cast<long>(out.size()), 0};
}

Stop Tracer::wait(int options) noexcept {
    return wait_pid(-1, options);
}

Stop Tracer::wait_pid(int pid, int options) noexcept {
    int status = 0;
    const int r = wait_status(pid, options, status);
    if (r < 0) {
        Stop out;
        out.kind = StopKind::Exited;
        out.pid = pid;
        // A negative pid here means the wait failed rather than that a
        // process ended. There is no stop to report, so the signal field
        // carries the errno and the caller has to check for it.
        out.signal = -r;
        out.synthetic = true;
        return out;
    }
    if (r == 0) {
        Stop out;
        out.pid = pid;
        out.kind = StopKind::Exited;
        out.signal = sys::kEchild;
        out.synthetic = true;
        return out;
    }
    return classify_status(r, status);
}

sys::Result Tracer::event_message(int pid) noexcept {
    return request(kPtraceGeteventmsg, pid, nullptr, nullptr);
}

sys::Result Tracer::siginfo(int pid, void* out, std::size_t size) noexcept {
    (void)size;
    return request(kPtraceGetsiginfo, pid, nullptr, out);
}

// --------------------------------------------------------------- breakpoints

int Breakpoints::add(Tracer& tracer, int pid, std::uint64_t address,
                     std::string& detail) noexcept {
    for (auto& b : points_) {
        if (b.address == address) {
            b.armed = true;
            return 0;
        }
    }

    auto cur = tracer.peek(pid, address);
    if (cur.failed()) {
        detail = "reading the original byte";
        return cur.error;
    }

    SoftwareBreakpoint bp;
    bp.address = address;
    bp.saved = static_cast<std::uint8_t>(cur.value & 0xff);

    // The breakpoint byte is written into the word that is already there,
    // leaving the other seven bytes alone. A poke that wrote the whole word
    // would replace the instruction stream around the breakpoint.
    const std::uint64_t word =
        (static_cast<std::uint64_t>(cur.value) & ~0xffULL) | 0xccULL;
    auto w = tracer.poke(pid, address, word);
    if (w.failed()) {
        detail = "writing the breakpoint byte";
        return w.error;
    }

    bp.armed = true;
    points_.push_back(bp);
    return 0;
}

const SoftwareBreakpoint* Breakpoints::find(std::uint64_t address) const noexcept {
    for (const auto& b : points_) {
        if (b.address == address) {
            return &b;
        }
    }
    return nullptr;
}

int Breakpoints::remove(Tracer& tracer, int pid, std::uint64_t address) noexcept {
    for (auto it = points_.begin(); it != points_.end(); ++it) {
        if (it->address != address) {
            continue;
        }
        if (it->armed) {
            auto cur = tracer.peek(pid, address);
            if (cur.failed()) {
                return cur.error;
            }
            const std::uint64_t word = (static_cast<std::uint64_t>(cur.value) &
                                        ~0xffULL) |
                                       static_cast<std::uint64_t>(it->saved);
            auto w = tracer.poke(pid, address, word);
            if (w.failed()) {
                return w.error;
            }
        }
        points_.erase(it);
        return 0;
    }
    return 0;
}

int Breakpoints::disarm_all(Tracer& tracer, int pid) noexcept {
    for (auto& b : points_) {
        if (!b.armed) {
            continue;
        }
        auto cur = tracer.peek(pid, b.address);
        if (cur.failed()) {
            return cur.error;
        }
        const std::uint64_t word =
            (static_cast<std::uint64_t>(cur.value) & ~0xffULL) |
            static_cast<std::uint64_t>(b.saved);
        auto w = tracer.poke(pid, b.address, word);
        if (w.failed()) {
            return w.error;
        }
        b.armed = false;
    }
    return 0;
}

int Breakpoints::rearm_all(Tracer& tracer, int pid) noexcept {
    for (auto& b : points_) {
        if (b.armed) {
            continue;
        }
        auto cur = tracer.peek(pid, b.address);
        if (cur.failed()) {
            return cur.error;
        }
        const std::uint64_t word =
            (static_cast<std::uint64_t>(cur.value) & ~0xffULL) | 0xccULL;
        auto w = tracer.poke(pid, b.address, word);
        if (w.failed()) {
            return w.error;
        }
        b.armed = true;
    }
    return 0;
}

int Breakpoints::step_over(Tracer& tracer, int pid, std::string& detail) noexcept {
    Registers regs{};
    auto gr = tracer.get_regs(pid, regs);
    if (gr.failed()) {
        detail = "reading the registers to step over a breakpoint";
        return gr.error;
    }

    // A software breakpoint stops the tracee with rip one past the byte that
    // was replaced, because the trap is raised after the instruction
    // retires. Without this adjustment the following instruction would run
    // twice and the instruction at the breakpoint address would never run.
    const bool at_breakpoint = find(regs.rip - 1) != nullptr;
    if (at_breakpoint) {
        regs.rip -= 1;
        auto sr = tracer.set_regs(pid, regs);
        if (sr.failed()) {
            detail = "rewinding rip past the breakpoint byte";
            return sr.error;
        }
    }

    // The original byte has to be back in place for the instruction to run,
    // or the single step would trap on the breakpoint byte again.
    const int da = disarm_all(tracer, pid);
    if (da != 0) {
        detail = "disarming the breakpoints to step";
        return da;
    }

    auto st = tracer.singlestep(pid, 0);
    if (st.failed()) {
        detail = "the single step over the breakpoint";
        return st.error;
    }

    Stop stop = tracer.wait_pid(pid);
    if (stop.kind == StopKind::Exited) {
        detail = "the tracee exited while stepping over a breakpoint";
        return sys::kEchild;
    }

    const int ra = rearm_all(tracer, pid);
    if (ra != 0) {
        detail = "re-arming the breakpoints after a step";
        return ra;
    }
    return 0;
}

} // namespace occ::obs
