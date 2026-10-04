#pragma once

// The observation session.
//
// A session owns a traced process and the loop that serves it: it waits for
// stops, decides what each one means, reports it, and answers the remote
// debugger. It is the layer that turns three independent mechanisms into
// one running thing.
//
// The loop is deliberately single-threaded and stop-driven. A tracer that
// polled, or that used a second thread to read events, would have two
// readers of the same process state and no ordering between them, and the
// whole point of an observation session is that the record it produces is
// in the order the stops happened.
//
// The session is also the place where the pieces have to agree about what a
// stop means. A breakpoint, a single step override, a seccomp trap and a
// signal all arrive as the same kind of kernel event, and the session is
// what knows which one it arranged and therefore which one it is looking at.

#include <cstdint>
#include <string>
#include <vector>

#include "occ/observer/event.h"
#include "occ/observer/ptrace.h"
#include "occ/observer/rsp.h"
#include "occ/observer/watchpoint.h"
#include "occ/observer/wx.h"

namespace occ::obs {

struct SessionConfig {
    // The process to observe. It has to be stopped, or the session has to
    // be given the right to stop it.
    int pid = 0;

    // Whether to trace syscalls. Turning this on makes the session stop
    // twice per syscall, which is the single largest cost the observer can
    // add to a target.
    bool trace_syscalls = false;

    // Whether to install the fork and exec event options. A session that
    // does not follow children is a session that stops observing at the
    // first fork, which is why this defaults on.
    bool follow_forks = true;

    // Whether to watch for write-then-execute transitions.
    //
    // This costs the target: a hardware watch takes a debug register, there
    // are four, and every write to a watched region stops the session. It is
    // off by default because a run that is not looking for self-modifying
    // code should not pay for the possibility.
    bool track_wx = false;

    // Which regions the tracker watches when track_wx is on. Empty means
    // "work it out from the target's own maps", which is the right answer
    // for a run that does not know the image in advance. A caller that does
    // know -- an unpacker that has already located the staging buffer --
    // names the regions and skips the scan.
    std::vector<WatchTarget> wx_regions;

    // Whether a region has to be anonymous to be watched. A decoder's
    // staging buffer is anonymous, and requiring it cuts the number of
    // candidate regions to roughly one per allocation rather than one per
    // mapping, which matters when there are four debug registers and a
    // linker maps dozens.
    bool wx_anonymous_only = true;

    // The largest total region size the tracker will consider. A target that
    // maps a gigabyte of writable memory is not going to have its decoder
    // found by watching the first four words of it, and a limit keeps the
    // scan from proposing regions that could never be covered.
    std::uint64_t wx_max_region_bytes = 16u * 1024u * 1024u;

    // Whether to serve a remote debugger. When false the session only
    // produces events, and the loop never reads from the client
    // descriptor.
    bool serve_gdb = false;

    // The descriptor the remote debugger speaks on, and the one it reads
    // from. Two descriptors rather than one because the protocol is
    // full-duplex and a debugger may send an interrupt while it is reading
    // a stop reply.
    int gdb_read_fd = -1;
    int gdb_write_fd = -1;
};

// What a session did.
struct SessionResult {
    bool failed = false;
    std::string detail;

    // The number of stops the session handled, which is the honest measure
    // of how much it observed.
    std::uint64_t stops = 0;
    std::uint64_t syscall_stops = 0;
    std::uint64_t breakpoint_hits = 0;
    std::uint64_t signals = 0;
    std::uint64_t transitions = 0;

    // What the write tracker managed. These are reported rather than assumed
    // because the hardware has four debug registers and a candidate region
    // is a whole page: the honest answer is usually "four bytes of the
    // first region", and a caller that was told "tracking" with no numbers
    // would read that as coverage it does not have.
    std::uint64_t wx_regions = 0;
    std::uint64_t wx_watches = 0;
    std::uint64_t wx_bytes_covered = 0;
    std::uint64_t wx_bytes_total = 0;
    std::uint64_t wx_regions_unwatched = 0;
    // Samples the kernel reported as lost, which are accesses the observer
    // never saw. Nonzero means the record is incomplete.
    std::uint64_t wx_lost_samples = 0;
    // True when no hardware watch could be installed at all, which is the
    // normal result on a host whose policy blocks perf_event_open. It is
    // reported as a fact rather than as a failure because the rest of the
    // session is unaffected.
    bool wx_unavailable = false;

    int exit_code = 0;
    int term_signal = 0;
    bool signaled = false;
};

// Runs the observation loop until the traced process ends or the debugger
// detaches.
[[nodiscard]] SessionResult observe(const SessionConfig& config,
                                    Writer& events) noexcept;

// The remote debugger protocol, as a state machine over one process. Kept
// separate from the loop so that the packet handling can be tested by
// feeding it packets, without a live process to trace.
class DebugServer {
public:
    DebugServer(Tracer& tracer, Breakpoints& breakpoints, int pid,
                Writer& events) noexcept
        : tracer_(&tracer), breakpoints_(&breakpoints), pid_(pid),
          events_(&events) {}

    // Handles one decoded packet and produces the response payload. An
    // empty response means the packet is not supported, which the protocol
    // defines as an answer rather than a failure.
    [[nodiscard]] std::string handle(std::string_view packet) noexcept;

    // True when the client asked to detach. The loop uses it to stop.
    [[nodiscard]] bool detached() const noexcept { return detached_; }

    // True when the client asked for a continue or a step, which the loop
    // has to act on rather than answer.
    [[nodiscard]] bool resume_requested() const noexcept {
        return resume_requested_;
    }
    [[nodiscard]] bool step_requested() const noexcept { return step_; }
    void clear_resume() noexcept {
        resume_requested_ = false;
        step_ = false;
        resume_signal_ = 0;
    }
    [[nodiscard]] int resume_signal() const noexcept { return resume_signal_; }

private:
    std::string handle_query(std::string_view kind) noexcept;
    std::string handle_read_memory(std::string_view args) noexcept;
    std::string handle_write_memory(std::string_view args) noexcept;
    std::string handle_read_registers() noexcept;
    std::string handle_write_registers(std::string_view args) noexcept;
    std::string handle_breakpoint(std::string_view args) noexcept;
    std::string handle_thread() noexcept;

    Tracer* tracer_;
    Breakpoints* breakpoints_;
    int pid_;
    Writer* events_;

    bool detached_ = false;
    bool resume_requested_ = false;
    bool step_ = false;
    int resume_signal_ = 0;
};

} // namespace occ::obs
