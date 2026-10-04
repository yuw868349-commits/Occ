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
    bool track_wx = false;

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
