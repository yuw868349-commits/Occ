#pragma once

// The isolation container.
//
// A container here is a process tree that has been given its own view of the
// system: its own mount table, its own process id space, its own network
// namespace, and its own root directory. There is no hypervisor and no
// kernel module. Everything is composed from primitives the kernel already
// exposes, which is why the setup path reads as a sequence of small
// operations rather than as a call into a container runtime.
//
// Order matters and is not arbitrary:
//
//   1. clone3 with the namespace flags. The new process is created stopped
//      behind a pipe so that it cannot run a single instruction before the
//      rest of the setup has happened.
//   2. Write the uid and gid maps. A user namespace starts with no mapping,
//      so the process inside has no valid credentials until this completes.
//   3. In the child: make the mount table private, so nothing that happens
//      next can propagate back to the host.
//   4. Assemble the new root. Overlayfs where a writable layer was asked
//      for, a plain bind otherwise, then pivot_root.
//   5. Attach to a cgroup, so the resource limits a caller asked for are in
//      place before any real work starts.
//   6. Install the seccomp filter last. It applies to the process that
//      installs it, so everything above has to finish first.
//
// Each step reports a failure at the point it happened. A container that
// half-exists is worse than one that never started, and the caller needs to
// know which capability the host is missing.

#include <cstdint>
#include <string>
#include <vector>

#include "occ/syscall/capability.h"

namespace occ::isolation {

// Which namespaces to create. A container with a partial set is a valid and
// sometimes useful thing: sharing the network namespace with the host is how
// a target is given real network access, and sharing the process id space is
// how a tracer outside the container can still see it by pid.
struct NamespaceSet {
    bool user = true;
    bool mount = true;
    bool pid = true;
    bool ipc = true;
    bool uts = true;
    bool net = false; // off by default: the host's network is reachable
    bool cgroup = true;

    [[nodiscard]] std::uint64_t flags() const noexcept;
};

// How the new root directory is built.
enum class RootKind : std::uint8_t {
    // A read-only bind of an existing directory. The target cannot write
    // anywhere, and nothing it does can survive the run.
    ReadOnlyBind,
    // An overlay with a writable upper layer. The target can write, the
    // writes land in the upper directory, and the lower directory is never
    // modified. This is the mode to use when a run has to be repeatable:
    // throw away the upper directory and the next run starts identical.
    Overlay,
    // A tmpfs mounted over the root, with the given directory bind-mounted
    // inside it. Nothing outside the mount survives, and the target cannot
    // see the host's filesystem at all.
    Tmpfs,
};

struct ResourceLimits {
    // Zero means unlimited, which is also what the kernel reads from a
    // "max" written into a cgroup file.
    std::uint64_t memory_bytes = 0;
    std::uint64_t pids = 0;
    // Percent of one cpu, 1..100. Zero leaves the default weight alone.
    std::uint32_t cpu_percent = 0;
};

struct ContainerConfig {
    // The directory that becomes the new root, or the lower layer when the
    // root kind is Overlay.
    std::string root_dir;
    // Where writes land when the root kind is Overlay. Required for that
    // kind and ignored otherwise.
    std::string upper_dir;
    // The directory the overlay's workdir is created under. Overlayfs
    // requires it to be on the same filesystem as the upper directory.
    std::string work_dir;

    RootKind root_kind = RootKind::ReadOnlyBind;

    NamespaceSet namespaces;

    // The uid and gid the target runs as inside the container. One mapping
    // per identifier is written, so the target sees exactly one user, which
    // is what keeps a setuid binary from finding a privileged identity to
    // change to.
    std::uint32_t inside_uid = 0;
    std::uint32_t inside_gid = 0;

    // The directory a cgroup is created under, or empty to skip. The caller
    // is responsible for that directory being on a cgroup v2 mount with the
    // controllers it wants delegated; writing to subtree_control in a
    // directory that was not delegated fails with EACCES and is reported.
    std::string cgroup_parent;
    ResourceLimits limits;

    // Capabilities the target keeps. Everything not listed is dropped from
    // the bounding, effective, permitted and inheritable sets, so a target
    // that has no use for a capability cannot use it, and cannot regain it
    // through an exec.
    std::vector<sys::Cap> keep_capabilities;

    // Paths bind-mounted read-only into the new root after it is in place.
    // Used for things the target needs but that should not be part of the
    // image: a runtime library directory, a device file, a data directory.
    struct BindMount {
        std::string source;
        std::string target;
        bool writable = false;
    };
    std::vector<BindMount> extra_mounts;
};

// Where each setup step got to. A caller that sees a failure uses this to
// report which primitive the host refused, rather than a bare errno.
enum class Stage : std::uint8_t {
    Clone,
    UidMap,
    MountPropagation,
    RootAssembly,
    PivotRoot,
    Cgroup,
    Capabilities,
    Seccomp,
    Exec,
};

[[nodiscard]] const char* stage_name(Stage s) noexcept;

struct ContainerError {
    Stage stage = Stage::Clone;
    int error = 0; // positive errno
    std::string detail;

    [[nodiscard]] bool ok() const noexcept { return error == 0; }
};

// The result of a container run. The container itself has exited by the
// time this is returned; a caller that wants to observe the target does so
// through the observer layer, which takes control of the process before it
// is released.
struct ContainerResult {
    ContainerError error;
    int exit_code = 0;
    int term_signal = 0;
    // True when the process was stopped by a signal rather than exiting.
    bool signaled = false;
};

// Runs `path` with `argv` inside a container built from `config`.
//
// This is the synchronous form: it returns once the target has exited. A
// caller that needs the process to keep running, or that needs to trace it,
// uses the split form below.
[[nodiscard]] ContainerResult run_container(const ContainerConfig& config,
                                            const std::string& path,
                                            const std::vector<std::string>& argv,
                                            const std::vector<std::string>& envp);

// The split form. `spawn` builds the container and returns once the target
// is stopped at its first instruction, with the child's pid. The caller then
// has the process to itself: it can attach a tracer, install a filter, or
// read its memory. `resume` releases it. `reap` collects the exit status.
//
// Nothing between spawn and resume is allowed to fail silently: if the
// caller abandons the child it stays stopped forever, which is why the
// caller is expected to call reap in every path.
struct SpawnResult {
    ContainerError error;
    int pid = 0;
};

[[nodiscard]] SpawnResult container_spawn(const ContainerConfig& config,
                                          const std::string& path,
                                          const std::vector<std::string>& argv,
                                          const std::vector<std::string>& envp);

// Releases a spawned child. Returns the error if the resume failed.
[[nodiscard]] ContainerError container_resume(int pid) noexcept;

// Waits for the child and reports how it ended. Always call this, including
// on the paths where an earlier step failed, or the child is left stopped.
[[nodiscard]] ContainerResult container_reap(int pid) noexcept;

// Tears down the parts of the container that outlive the process: the
// cgroup directory and, for an overlay root, the upper and work directories.
// Returns the number of things removed, or -1 if any removal failed.
[[nodiscard]] long container_cleanup(const ContainerConfig& config) noexcept;

// Reads the cgroup v2 mount point from the running system's own view of it,
// rather than assuming /sys/fs/cgroup. Returns an empty string when cgroup
// v2 is not mounted.
[[nodiscard]] std::string cgroup2_mount_point() noexcept;

// Enumerates the controllers available in the root cgroup. Reported by
// occ doctor and used here to check a requested limit can be applied.
[[nodiscard]] std::vector<std::string> cgroup2_controllers() noexcept;

} // namespace occ::isolation
