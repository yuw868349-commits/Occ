#pragma once

// A run of a target under observation.
//
// The runner is the layer that knows how the three pieces fit together: the
// parser says what the image wants, the container builds the environment it
// runs in, and the observer reports what it did. None of the three knows
// about the others, which is what keeps each one testable on its own.
//
// What the runner adds is the sequence. It reads the target, decides whether
// the image is one this build can run, builds the container configuration,
// starts the target, reports every fact it has about it, and collects the
// exit status. The order of the reported facts is part of the interface: an
// image_loaded event is written before any mapping event, so a consumer that
// reads the stream top to bottom has a complete picture at every point.

#include <cstdint>
#include <string>
#include <vector>

#include "occ/isolation/container.h"
#include "occ/observer/event.h"
#include "occ/parser/elf.h"

namespace occ::runner {

// What a caller can ask a run to do.
struct RunOptions {
    // Where the new root comes from. Empty means the caller wants a
    // read-only bind of the host's own root, which is the smallest setup
    // that still gives the target a coherent filesystem.
    std::string root_dir;
    std::string upper_dir;
    std::string work_dir;

    isolation::RootKind root_kind = isolation::RootKind::ReadOnlyBind;

    // Resource limits. Zero means unlimited.
    std::uint64_t memory_bytes = 0;
    std::uint64_t pids = 0;
    std::uint32_t cpu_percent = 0;

    // The directory a cgroup is created under. Empty disables the cgroup
    // step, which is what a run without a delegated hierarchy has to do.
    std::string cgroup_parent;

    // Extra bind mounts, applied after the root is assembled.
    std::vector<isolation::ContainerConfig::BindMount> extra_mounts;

    // Whether the target's own output goes to the terminal unchanged. When
    // this is false the target's stdout is still inherited: the event
    // stream goes to its own descriptor, so the two do not interleave.
    bool stream_events = true;

    // Whether to trace the target with ptrace while it runs. A traced
    // target is stopped twice per syscall and cannot be attached to by
    // anything else, so this is off by default: a run that only needs the
    // isolation facts should not pay for observation it did not ask for.
    bool observe = false;

    // Where the target is, for a run that is being set up for a debugger to
    // reach later. Empty for an ordinary run.
    std::string session_id;

    // The descriptor a remote debugger speaks on, when one is being served.
    // Negative means none.
    int gdb_read_fd = -1;
    int gdb_write_fd = -1;

    // Environment for the target. Empty means a minimal environment is
    // built from the host's PATH and TERM, which is enough for a static
    // binary and avoids passing through a variable that changes behaviour.
    std::vector<std::string> env;
};

// How a run ended.
struct RunResult {
    // True when the run could not be started or did not complete. An exit
    // code is not an error: a target that exits 3 is a successful run whose
    // target chose 3. This is a separate flag rather than a test on
    // `error.error`, because a refusal has a reason but no errno, and a
    // caller that tested the errno would read a refusal as a success.
    bool failed = false;

    // The stage and errno of a failure, and the human-readable reason.
    isolation::ContainerError error;

    // The reason a run was refused, set when `failed` is true and the
    // failure was a refusal rather than a failing syscall.
    std::string failure_detail;

    int exit_code = 0;
    int term_signal = 0;
    bool signaled = false;

    // Facts about the image, filled in whether or not the target ran, so a
    // caller can report what it would have run even when it refused to.
    bool image_loaded = false;
    parser::LoadError load_error = parser::LoadError::NotElf;
    std::string load_detail;
    std::uint64_t entry = 0;
    std::uint64_t mapping_count = 0;
    std::size_t events = 0;

    // Filled in only when the run was observed. A caller that did not ask
    // for observation gets zeros here, which is the honest answer: nothing
    // was observed.
    std::uint64_t stops = 0;
    std::uint64_t syscall_stops = 0;
    std::uint64_t breakpoint_hits = 0;
    std::uint64_t observed_signals = 0;
};

// Reads `path`, reports what the image is, and runs it under the container
// described by `options`. Facts go to `events` as they are established.
[[nodiscard]] RunResult run(const std::string& path,
                            const std::vector<std::string>& argv,
                            const RunOptions& options,
                            obs::Writer& events);

// Reads the image and reports it without running anything. Split out so
// that a caller can show what would happen, and so that `occ check` and the
// run path agree on what the image is rather than being two implementations
// that can drift.
[[nodiscard]] parser::ElfImage inspect(const std::string& path,
                                       obs::Writer* events,
                                       std::string& read_error);

} // namespace occ::runner
