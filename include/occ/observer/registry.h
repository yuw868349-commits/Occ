#pragma once

// Running sessions.
//
// A session is a run that did not stop when its process exited: the target
// is still there, or its observation stream is still open, and a second
// invocation of occ has to be able to reach it. The registry is what makes
// that possible.
//
// It is a directory of small files rather than a daemon. A daemon would
// need a protocol, an authentication story, and a way to be restarted, and
// the only thing this registry has to answer is "which sessions exist" and
// "where is the stream". Files are readable by every tool on the system,
// survive a crash of the process that wrote them, and can be inspected by
// hand when something is wrong, which is what an operator wants at the
// moment they need it.
//
// The directory is under the runtime directory, not under /tmp: a registry
// entry is per-user state with a lifetime bounded by the boot, and /tmp is
// neither.

#include <cstdint>
#include <string>
#include <vector>

#include "occ/observer/event.h"

namespace occ::obs {

// What is known about a session that is not this process.
struct SessionRecord {
    std::string id;
    int pid = 0;
    // The process the session observes, which is not the session's own pid.
    std::string target;
    std::uint64_t started_ns = 0;
    // Where the event stream is being written. A session that was started
    // with its stream on a terminal has no path here and reports empty.
    std::string stream_path;
    // The directory the registry entry lives in, so a caller can remove it
    // without re-deriving the path.
    std::string directory;
};

// The directory the registry lives in. Created on first use.
[[nodiscard]] std::string registry_root() noexcept;

// Writes a record for a session this process is running. Returns the id, or
// an empty string when the entry could not be written, which is reported
// rather than treated as fatal: a session that cannot be found later is
// still a session this process can finish.
[[nodiscard]] std::string register_session(const std::string& target, int pid,
                                           const std::string& stream_path,
                                           std::string& error) noexcept;

// Rewrites the record with a new process id, which is what a run that
// replaced its image has to do: the session is the same one but the
// process the registry points at is not.
[[nodiscard]] bool update_session_pid(const std::string& id, int pid,
                                      std::string& error) noexcept;

// Removes the record. Called on every path out of a session, including the
// failing ones, because a stale record sends a later attach to a pid that
// belongs to something else.
void unregister_session(const std::string& id) noexcept;

// Every record whose process is still alive. A record whose process has
// gone is removed rather than returned, because a registry that reported
// dead sessions would make every caller check.
[[nodiscard]] std::vector<SessionRecord> live_sessions() noexcept;

// One record by id, or by the pid it observes if `by_pid` is set.
[[nodiscard]] bool find_session(const std::string& key, bool by_pid,
                                SessionRecord& out) noexcept;

// True when a process with this pid exists and is not a zombie. Uses
// kill(pid, 0) rather than reading /proc, which is one syscall instead of
// an open, a read, and a parse.
[[nodiscard]] bool process_alive(int pid) noexcept;

} // namespace occ::obs
