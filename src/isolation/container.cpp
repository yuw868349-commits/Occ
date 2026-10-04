#include "occ/isolation/container.h"

#include "occ/syscall/errno.h"
#include "occ/syscall/kabi.h"
#include "occ/syscall/syscall.h"
#include "occ/util/fs.h"
#include "occ/util/string.h"

#include <array>
#include <cstring>

#include <sched.h>
#include <sys/wait.h>

namespace occ::isolation {

namespace {

using sys::Result;

// ------------------------------------------------------------ small helpers

bool write_string(const std::string& path, const std::string& text) noexcept {
    return fs::write_file(path, text);
}

std::string to_decimal(std::uint64_t value) {
    std::string out;
    append_uint(out, value);
    return out;
}

// The uid and gid maps. Exactly one line each: the container's identifier
// zero maps to the caller's real identifier. A single-line map is what
// removes the target's ability to reach any other identity on the host.
bool write_id_map(const std::string& dir, const char* file, std::uint32_t inside,
                  std::uint32_t outside) noexcept {
    std::string line;
    append_uint(line, inside);
    line.push_back(' ');
    append_uint(line, outside);
    line += " 1\n";
    return write_string(dir + "/" + file, line);
}

// ---------------------------------------------------------------- cgroup v2

struct MountEntry {
    std::string source;
    std::string target;
    std::string fstype;
};

// Parses /proc/self/mountinfo far enough to find the mount point of the
// given filesystem type.
//
// The file lists mounts in the order the kernel's mount tree holds them, and
// a later entry for the same filesystem type shadows an earlier one when the
// later mount point is a prefix of, or equal to, the earlier one. Container
// runtimes exploit this: a runtime that has mounted its own /sys inside its
// own scratch root leaves entries for both that copy and the host's, and the
// copy comes first.
//
// So the shortest mount point wins rather than the first. The shortest path
// is the one closest to the real root of the mount tree, which is the copy
// that is actually visible at the conventional location.
std::string find_mount_point(const char* fstype) noexcept {
    auto content = fs::read_file("/proc/self/mountinfo");
    if (!content) {
        return {};
    }

    std::string best;

    for (const auto line : split(*content, '\n')) {
        if (line.empty()) {
            continue;
        }

        // Fields before the separator: id, parent, major:minor, root, point,
        // options. Fields after it: filesystem type, source, super options.
        const auto sep = line.find(" - ");
        if (sep == std::string_view::npos) {
            continue;
        }
        const auto before = split(line.substr(0, sep), ' ');
        const auto after = split(line.substr(sep + 3), ' ');

        if (before.size() < 5 || after.empty()) {
            continue;
        }
        if (after[0] != fstype) {
            continue;
        }

        // The mount point is escaped in the file: a space becomes \040 and
        // so on. Cgroup paths do not normally contain those, but a missed
        // escape produces a path that does not exist.
        std::string point;
        const auto field = before[4];
        for (std::size_t i = 0; i < field.size(); ++i) {
            if (field[i] == '\\' && i + 3 < field.size()) {
                const auto d0 = field[i + 1];
                const auto d1 = field[i + 2];
                const auto d2 = field[i + 3];
                if (d0 >= '0' && d0 <= '7' && d1 >= '0' && d1 <= '7' &&
                    d2 >= '0' && d2 <= '7') {
                    const int value =
                        (d0 - '0') * 64 + (d1 - '0') * 8 + (d2 - '0');
                    point.push_back(static_cast<char>(value));
                    i += 3;
                    continue;
                }
            }
            point.push_back(field[i]);
        }

        if (best.empty() || point.size() < best.size()) {
            best = std::move(point);
        }
    }
    return best;
}

// The fallback path needs the clone flag bits as numbers, because unshare
// takes them as an int and the fork that follows must inherit exactly the
// set the clone path would have asked for. <sched.h> provides them, but the
// definitions are behind feature macros that vary between libc versions and
// one of them (CLONE_NEWCGROUP) is missing from older headers entirely.
// Restating them together keeps the two paths provably identical.
namespace flags {
constexpr std::uint64_t kNewns = 0x00020000ULL;    // CLONE_NEWNS
constexpr std::uint64_t kNewuts = 0x04000000ULL;   // CLONE_NEWUTS
constexpr std::uint64_t kNewipc = 0x08000000ULL;   // CLONE_NEWIPC
constexpr std::uint64_t kNewuser = 0x10000000ULL;  // CLONE_NEWUSER
constexpr std::uint64_t kNewpid = 0x20000000ULL;   // CLONE_NEWPID
constexpr std::uint64_t kNewnet = 0x40000000ULL;   // CLONE_NEWNET
} // namespace flags

// Pulled into the anonymous namespace's scope so that the clone and unshare
// call sites use the same names.
using flags::kNewipc;
using flags::kNewnet;
using flags::kNewns;
using flags::kNewpid;
using flags::kNewuser;
using flags::kNewuts;

// clone3 takes its argument count separately and rejects a size smaller than
// the kernel expects, so the size is derived from the structure rather than
// written as a literal.
constexpr std::uint64_t kCloneArgsSize = sizeof(sys::CloneArgs);

// The signal delivered to the parent when the container's init exits. SIGCHLD
// is the value that makes the container's process behave like an ordinary
// child, which is what makes the wait in container_reap terminate.
constexpr int kCloneExitSignal = 17; // SIGCHLD

// ---------------------------------------------------------------- cloning
//
// Two ways to create the process, and the reason there are two is not
// portability in the abstract.
//
// clone3 is the modern interface: one call that creates the process with
// every namespace at once, atomically, with a structure that can grow
// without breaking callers. It is what this project prefers and what it
// uses wherever the kernel allows it.
//
// A number of container runtimes run with a seccomp profile that answers
// clone3 with ENOSYS. The reason is historical: clone3 was introduced in 5.3
// and the glibc that shipped alongside returned a hard error rather than
// falling back, so a profile that blocked it would break every program that
// called clone(). Returning ENOSYS makes glibc take its own fallback path.
// The effect is that a process inside such a container can create a
// namespace with unshare and fork, but cannot call clone3 at all.
//
// The fallback is not a lesser version of clone3: it produces the same
// arrangement, and on any host whose policy answers clone3 with ENOSYS it is
// the only path that runs at all. What differs is that the namespaces are
// created by the child instead of for it, and the child is therefore the
// process that owns them.
//
//   fork()                      parent stays in the original user namespace
//     child: unshare(NEWUSER | NEWNS | NEWIPC | NEWUTS | ...)
//     child: unshare(NEWPID)    the child becomes pid 1 of the new space
//   parent: write /proc/<pid>/{setgroups,uid_map,gid_map}
//
// Putting the unshare before the fork reads as the tidier ordering and is
// wrong: the parent would end up inside the new user namespace, mapped to
// nobody, with no authority over the child it just created, and every
// mapping write would be refused.

struct CloneOutcome {
    long pid = -1;      // child pid in this namespace, 0 in the child
    int error = 0;      // positive errno
    bool in_child = false; // true in the child of the fallback fork
    bool used_fallback = false;
};

// Namespaces that go into the child's unshare call on the fallback path.
// CLONE_NEWPID is excluded: unshare only re-points the namespace of children
// created afterwards, so it has to be requested by the process that will
// itself become the init. Applying it in the child, before the child does
// anything else, has exactly the same effect as clone3's atomic form.
std::uint64_t fallback_unshare_flags(const NamespaceSet& ns) noexcept {
    std::uint64_t f = ns.flags();
    f &= ~flags::kNewpid;
    return f;
}

CloneOutcome clone_child(const ContainerConfig& config) noexcept {
    CloneOutcome out;

    sys::CloneArgs args{};
    args.flags = config.namespaces.flags() | kCloneExitSignal;
    args.exit_signal = kCloneExitSignal;

    auto cr = sys::clone3_raw(&args, kCloneArgsSize);
    if (cr.ok()) {
        out.pid = cr.value;
        return out;
    }

    if (cr.error != sys::kEnosys) {
        out.error = cr.error;
        return out;
    }

    // clone3 is not available here. Fall back to fork followed by unshare.
    //
    // The order is not interchangeable, and the other order is the one that
    // looks right. Calling unshare in the parent and then forking produces
    // the namespaces in the shape a reader expects, but it puts the parent
    // into the new user namespace too, where it is mapped to nobody and
    // holds no capability over anything. The writes to the child's
    // /proc/<pid>/uid_map that have to follow are then refused with EACCES,
    // because authority over a user namespace belongs to the process that
    // created it and not to a sibling inside it.
    //
    // Forking first keeps the parent outside the user namespace, still the
    // child's owner in the old one, and therefore still the process that
    // gets to define the child's mapping. The child creates the namespaces
    // for itself. The pid namespace is created by the child's own unshare
    // rather than inherited, which is what makes the child pid 1 in it.
    out.used_fallback = true;

    auto fr = sys::fork();
    if (fr.failed()) {
        out.error = fr.error;
        return out;
    }
    if (fr.value == 0) {
        out.in_child = true;
        out.pid = 0;

        const std::uint64_t shared = fallback_unshare_flags(config.namespaces);
        if (shared != 0) {
            auto ur = sys::unshare(static_cast<int>(shared));
            if (ur.failed()) {
                out.error = ur.error;
                return out;
            }
        }

        // The pid namespace has to be entered from the child, and unshare
        // only affects processes created after it. Creating it here and
        // having the caller's own fork follow would give the caller's fork
        // the namespace and this process the old one, so the namespace is
        // taken now and the child is the init.
        if (config.namespaces.pid) {
            auto pr = sys::unshare(static_cast<int>(flags::kNewpid));
            if (pr.failed()) {
                out.error = pr.error;
                return out;
            }
        }
        return out;
    }

    out.pid = fr.value;
    return out;
}

} // namespace

// ----------------------------------------------------------- namespace set

std::uint64_t NamespaceSet::flags() const noexcept {
    std::uint64_t f = 0;
    if (user) {
        f |= flags::kNewuser;
    }
    if (mount) {
        f |= flags::kNewns;
    }
    if (pid) {
        f |= flags::kNewpid;
    }
    if (ipc) {
        f |= flags::kNewipc;
    }
    if (uts) {
        f |= flags::kNewuts;
    }
    if (net) {
        f |= flags::kNewnet;
    }
    if (cgroup) {
        f |= sys::kCloneNewcgroup;
    }
    return f;
}

const char* stage_name(Stage s) noexcept {
    switch (s) {
    case Stage::Clone:
        return "clone";
    case Stage::UidMap:
        return "uid-map";
    case Stage::MountPropagation:
        return "mount-propagation";
    case Stage::RootAssembly:
        return "root-assembly";
    case Stage::PivotRoot:
        return "pivot-root";
    case Stage::Cgroup:
        return "cgroup";
    case Stage::Capabilities:
        return "capabilities";
    case Stage::Seccomp:
        return "seccomp";
    case Stage::Exec:
        return "exec";
    }
    return "unknown";
}

// ---------------------------------------------------------------- cgroup v2

std::string cgroup2_mount_point() noexcept {
    return find_mount_point("cgroup2");
}

std::vector<std::string> cgroup2_controllers() noexcept {
    std::vector<std::string> out;
    const auto point = cgroup2_mount_point();
    if (point.empty()) {
        return out;
    }

    auto content = fs::read_file(point + "/cgroup.controllers");
    if (!content) {
        return out;
    }

    for (const auto name : split(trim(*content), ' ')) {
        if (!name.empty()) {
            out.emplace_back(name);
        }
    }
    return out;
}

namespace {

// Writes the controllers the container wants into subtree_control, so that
// a child cgroup can be given limits. A controller the parent has not
// delegated is refused by the kernel with EACCES; the caller reports that
// rather than silently running without a limit.
bool enable_controllers(const std::string& parent,
                        const std::string& wanted) noexcept {
    if (wanted.empty()) {
        return true;
    }
    return write_string(parent + "/cgroup.subtree_control", wanted);
}

// Builds the cgroup for the container and moves `pid` into it. Returns 0 or
// a positive errno.
int attach_cgroup(const ContainerConfig& config, int pid,
                  std::string& detail) noexcept {
    if (config.cgroup_parent.empty()) {
        return 0;
    }

    const std::string dir = config.cgroup_parent + "/occ";
    if (!fs::mkdir_p(dir, 0755)) {
        detail = "cannot create " + dir;
        return sys::kEacces;
    }

    // Ask the parent to hand the controllers down. Enabling a controller
    // that the parent does not have is an error only when the file rejects
    // it; a controller that is simply absent is not requested.
    std::string controllers;
    if (config.limits.memory_bytes != 0) {
        controllers += "+memory ";
    }
    if (config.limits.pids != 0) {
        controllers += "+pids ";
    }
    if (config.limits.cpu_percent != 0) {
        controllers += "+cpu ";
    }
    if (!controllers.empty()) {
        controllers += "\n";
        if (!enable_controllers(config.cgroup_parent, controllers)) {
            detail = "cannot enable controllers under " + config.cgroup_parent;
            return sys::kEacces;
        }
    }

    // Move the process in before writing the limits, so a target that starts
    // fast cannot allocate ahead of the limit being in place.
    if (!write_string(dir + "/cgroup.procs", to_decimal(
                                                 static_cast<std::uint64_t>(pid)) +
                                                 "\n")) {
        detail = "cannot attach pid to " + dir;
        return sys::kEinval;
    }

    if (config.limits.memory_bytes != 0) {
        if (!write_string(dir + "/memory.max",
                          to_decimal(config.limits.memory_bytes) + "\n")) {
            detail = "cannot set memory.max";
            return sys::kEinval;
        }
        // Disable the swap escape hatch. Without this a target over the
        // limit simply moves pages to swap and the limit does not bound
        // anything.
        (void)write_string(dir + "/memory.swap.max", "0\n");
    }

    if (config.limits.pids != 0) {
        if (!write_string(dir + "/pids.max",
                          to_decimal(config.limits.pids) + "\n")) {
            detail = "cannot set pids.max";
            return sys::kEinval;
        }
    }

    if (config.limits.cpu_percent != 0) {
        const std::uint32_t percent =
            config.limits.cpu_percent > 100 ? 100 : config.limits.cpu_percent;
        // cpu.max is "<quota> <period>" in microseconds. A period of one
        // hundred thousand with a quota of percent times one thousand gives
        // the requested share of a single cpu, which is the fraction a
        // caller means by a percentage.
        const std::uint64_t period = 100000;
        const std::uint64_t quota = period * percent / 100;
        std::string value = to_decimal(quota);
        value.push_back(' ');
        value += to_decimal(period);
        value.push_back('\n');
        if (!write_string(dir + "/cpu.max", value)) {
            detail = "cannot set cpu.max";
            return sys::kEinval;
        }
    }

    return 0;
}

// ---------------------------------------------------------------- root setup

// Creates the directory skeleton a new root needs before anything can be
// mounted under it. pivot_root insists on both directories existing and on
// the new root not being on the same mount as the old one.
bool create_root_skeleton(const std::string& root) noexcept {
    // oldroot is deliberately absent: it is created after the root mount,
    // because the mount would hide it.
    const char* dirs[] = {
        "proc", "sys", "dev", "dev/pts", "tmp", "run", "etc", "work",
    };
    for (const char* d : dirs) {
        if (!fs::mkdir_p(root + "/" + d, 0755)) {
            return false;
        }
    }
    return true;
}

// Mounts the new root at `target`, according to the configured kind.
//
// A root that will be read-only is bound writable here and flipped later, by
// seal_root. The order is forced by two facts: the directory pivot_root
// needs inside the new root has to be created after the root is mounted,
// and it cannot be created once the root is read-only. Binding writable,
// assembling, and sealing last is the only order in which both hold.
//
// Returns 0 or a positive errno.
int assemble_root(const ContainerConfig& config, const std::string& target,
                  const std::string& scratch, std::string& detail) noexcept {
    switch (config.root_kind) {
    case RootKind::ReadOnlyBind: {
        auto r = sys::mount(config.root_dir.c_str(), target.c_str(), nullptr,
                            sys::kMsBind | sys::kMsRecursive, nullptr);
        if (r.failed()) {
            detail = "bind " + config.root_dir;
            return r.error;
        }
        return 0;
    }

    case RootKind::Tmpfs: {
        auto r = sys::mount("tmpfs", target.c_str(), "tmpfs",
                            sys::kMsNosuid | sys::kMsNodev, "size=64m");
        if (r.failed()) {
            detail = "tmpfs at " + target;
            return r.error;
        }
        // The caller's directory is bind-mounted inside the tmpfs, so the
        // target sees its own contents and nothing of the host's tmpfs.
        if (!config.root_dir.empty()) {
            const std::string inner = target + "/root";
            if (!fs::mkdir_p(inner, 0755)) {
                detail = "cannot create " + inner;
                return sys::kEacces;
            }
            auto b = sys::mount(config.root_dir.c_str(), inner.c_str(), nullptr,
                                sys::kMsBind | sys::kMsRecursive, nullptr);
            if (b.failed()) {
                detail = "bind " + config.root_dir + " into the tmpfs";
                return b.error;
            }
        }
        return 0;
    }

    case RootKind::Overlay: {
        if (config.upper_dir.empty() || config.work_dir.empty()) {
            detail = "overlay root needs an upper directory and a work "
                     "directory";
            return sys::kEinval;
        }
        for (const auto& d : {target, config.upper_dir, config.work_dir}) {
            if (!fs::mkdir_p(d, 0755)) {
                detail = "cannot create " + d;
                return sys::kEacces;
            }
        }

        std::string options = "lowerdir=" + config.root_dir;
        options += ",upperdir=" + config.upper_dir;
        options += ",workdir=" + config.work_dir;

        auto r = sys::mount("overlay", target.c_str(), "overlay",
                            sys::kMsNosuid | sys::kMsNodev, options.c_str());
        if (r.failed()) {
            // The most common failure by far is a work directory on a
            // different filesystem from the upper directory, which the
            // kernel reports as EINVAL with no detail. Naming the three
            // directories is what makes that diagnosable.
            detail = "overlay lower=" + config.root_dir +
                     " upper=" + config.upper_dir + " work=" + config.work_dir;
            return r.error;
        }
        (void)scratch;
        return 0;
    }
    }
    return sys::kEinval;
}

// Flips the root read-only, when the configuration asked for it.
//
// This runs after the root has been assembled and populated, and after the
// directory pivot_root needs has been created. A bind mount's flags can only
// be changed by a second mount call naming the same mount, which is why this
// is a separate call rather than a flag on the first one.
int seal_root(const ContainerConfig& config, const std::string& target,
              std::string& detail) noexcept {
    if (config.root_kind != RootKind::ReadOnlyBind) {
        return 0;
    }
    auto ro = sys::mount(nullptr, target.c_str(), nullptr,
                         sys::kMsRemount | sys::kMsBind | sys::kMsRecursive |
                             sys::kMsRdonly,
                         nullptr);
    if (ro.failed()) {
        detail = "remount " + target + " read-only";
        return ro.error;
    }
    return 0;
}

// Mounts the filesystems a root is expected to have and creates the device
// nodes a process needs.
//
// /proc is the interesting case. A fresh proc instance can only be mounted
// by a process that holds CAP_SYS_ADMIN over the user namespace that owns
// the target pid namespace, which is the situation here, and that is tried
// first because it is the arrangement that actually hides the host's
// process list from the target.
//
// Where the host refuses a nested proc instance, the host's own /proc is
// bound instead. The bind is attempted second rather than relied on: a
// host that pins /proc for its own administration -- every container
// runtime does -- refuses both, and the container is then left without one.
// That is reported through `present` rather than treated as a fatal error,
// because a target that never opens /proc runs correctly without it and a
// target that does open it produces a clear ENOENT rather than a wrong
// answer from a substitute.
//
// Device nodes are created before the capability drop. mknod on a character
// device needs CAP_MKNOD in the user namespace that owns the mount, and the
// drop removes exactly that.
struct RootSupport {
    bool proc = false;
    bool devices = false;
    bool devpts = false;
};

RootSupport populate_root(const ContainerConfig& config,
                          const std::string& root) noexcept {
    (void)config;
    RootSupport support;

    if (!fs::exists(root + "/proc")) {
        if (!fs::mkdir_p(root + "/proc", 0755)) {
            return support;
        }
    }

    auto p = sys::mount("proc", (root + "/proc").c_str(), "proc",
                        sys::kMsNosuid | sys::kMsNodev | sys::kMsNoexec,
                        nullptr);
    if (p.ok()) {
        support.proc = true;
    } else {
        // MS_REC is not optional on a bind of /proc: it has its own
        // submounts and a non-recursive bind leaves them pointing at the
        // old root, which survives the pivot and re-exposes part of the
        // host to the target.
        auto b = sys::mount("/proc", (root + "/proc").c_str(), nullptr,
                            sys::kMsBind | sys::kMsRecursive, nullptr);
        support.proc = b.ok();
    }

    if (!fs::exists(root + "/dev")) {
        if (!fs::mkdir_p(root + "/dev", 0755)) {
            return support;
        }
    }

    auto d = sys::mount("tmpfs", (root + "/dev").c_str(), "tmpfs",
                        sys::kMsNosuid, "mode=0755");
    if (!d.ok()) {
        return support;
    }

    // A devtmpfs would give the target every device on the host. These three
    // are what a process needs to have a working standard input and output;
    // anything else it can ask for explicitly through extra_mounts.
    //
    // A node that cannot be created is skipped rather than failing the run.
    // The target finds out when it opens the path, which is a more useful
    // place to learn it than a container that refuses to start.
    struct Node {
        const char* name;
        unsigned int mode;
        unsigned int major;
        unsigned int minor;
    };
    const Node nodes[] = {
        {"null", 0666, 1, 3},
        {"zero", 0666, 1, 5},
        {"urandom", 0444, 1, 9},
        {"random", 0444, 1, 8},
        {"tty", 0666, 5, 0},
    };
    bool any = false;
    for (const auto& n : nodes) {
        const auto dev = (static_cast<unsigned long>(n.major) << 8) |
                         static_cast<unsigned long>(n.minor);
        if (sys::mknodat(-100, (root + "/dev/" + n.name).c_str(),
                         020000 /* S_IFCHR */ | n.mode, dev)
                .ok()) {
            any = true;
        }
    }
    // The known-device fallback. A host that refuses mknod usually still
    // exposes the host's /dev/null, and a bind of that file gives the
    // target the one node almost every program uses. A file bind is
    // cheaper than a device bind and does not need CAP_MKNOD.
    if (!any && fs::exists("/dev/null")) {
        const std::string target = root + "/dev/null";
        if (sys::mount("/dev/null", target.c_str(), nullptr,
                       sys::kMsBind, nullptr)
                .ok()) {
            any = true;
        }
    }
    support.devices = any;

    // sysfs and devpts are best-effort. A target that does not need them is
    // unaffected by their absence.
    if (!fs::exists(root + "/sys")) {
        (void)fs::mkdir_p(root + "/sys", 0755);
    }
    (void)sys::mount("sysfs", (root + "/sys").c_str(), "sysfs",
                     sys::kMsNosuid | sys::kMsNodev | sys::kMsNoexec |
                         sys::kMsRecursive,
                     nullptr);

    if (!fs::exists(root + "/dev/pts")) {
        (void)fs::mkdir_p(root + "/dev/pts", 0755);
    }
    support.devpts =
        sys::mount("devpts", (root + "/dev/pts").c_str(), "devpts",
                   sys::kMsNosuid | sys::kMsNoexec,
                   "newinstance,ptmxmode=0666")
            .ok();
    (void)sys::symlink("pts/ptmx", (root + "/dev/ptmx").c_str());

    return support;
}

// Applies the extra bind mounts. A mount whose source does not exist is
// skipped rather than failing the run, because the caller typically lists
// paths that are optional on the host.
bool apply_extra_mounts(const ContainerConfig& config,
                        const std::string& root) noexcept {
    for (const auto& m : config.extra_mounts) {
        if (!fs::exists(m.source)) {
            continue;
        }
        const std::string target = root + m.target;
        if (!fs::exists(target)) {
            if (!fs::mkdir_p(target, 0755)) {
                return false;
            }
        }
        // The read-only flag is deliberately absent from the initial bind:
        // the kernel applies a bind mount's flags in a single call, and
        // asking for read-only at bind time on a directory that the target
        // then needs to traverse produces a mount whose metadata cannot be
        // updated. Applying it in a remount is the form that works.
        auto r = sys::mount(m.source.c_str(), target.c_str(), nullptr,
                            sys::kMsBind | sys::kMsRecursive, nullptr);
        if (r.failed()) {
            return false;
        }
        if (!m.writable) {
            auto ro = sys::mount(nullptr, target.c_str(), nullptr,
                                 sys::kMsRemount | sys::kMsBind |
                                     sys::kMsRecursive | sys::kMsRdonly,
                                 nullptr);
            if (ro.failed()) {
                return false;
            }
        }
    }
    return true;
}

// --------------------------------------------------------------- capability

// Drops every capability except the ones the caller asked to keep.
//
// The write has to happen in the container's own user namespace, after the
// uid map is in place, because before that the process has no capability
// over anything and the write would fail. The sets are written in the order
// the kernel requires: bounding first, which is the set that cannot be
// regained, then the working sets.
int drop_capabilities(const std::vector<sys::Cap>& keep,
                      std::string& detail) noexcept {
    std::array<std::uint32_t, 2> mask{};
    for (const auto cap : keep) {
        const auto bit = static_cast<std::uint32_t>(cap);
        if (bit >= sys::kCapLast + 1) {
            detail = "capability number out of range";
            return sys::kEinval;
        }
        mask[bit / 32] |= (1U << (bit % 32));
    }

    struct {
        std::uint32_t version;
        int pid;
        std::uint32_t effective;
        std::uint32_t permitted;
        std::uint32_t inheritable;
    } header{};
    header.version = 0x20080522; // _LINUX_CAPABILITY_VERSION_3
    header.pid = 0;              // the calling thread

    struct {
        std::uint32_t effective;
        std::uint32_t permitted;
        std::uint32_t inheritable;
    } data[2]{};

    data[0].effective = mask[0];
    data[0].permitted = mask[0];
    data[0].inheritable = mask[0];
    data[1].effective = mask[1];
    data[1].permitted = mask[1];
    data[1].inheritable = mask[1];

    auto r = sys::capset(&header, data);
    if (r.failed()) {
        detail = "capset";
        return r.error;
    }

    // The bounding set is dropped through prctl, one bit at a time. There is
    // no bulk operation, and the kernel rejects an attempt to drop a bit
    // that is already clear only for the last one, which is why a failure on
    // the final drop is tolerated.
    for (std::uint32_t bit = 0; bit <= sys::kCapLast; ++bit) {
        if ((mask[bit / 32] & (1U << (bit % 32))) != 0) {
            continue;
        }
        // PR_CAPBSET_DROP is 24.
        (void)sys::prctl(24, bit, 0, 0, 0);
    }
    return 0;
}

// ------------------------------------------------------------------ runtime

// Everything the child does between its first instruction and exec. Runs in
// the container's namespaces, on the container's root. Returns 0 on success,
// or a positive errno, with `stage` naming the step that failed.
int child_setup(const ContainerConfig& config, const std::string& scratch,
                const std::string& root_holder, Stage& stage,
                std::string& detail) {
    // Make the mount table private before anything is mounted, so no mount
    // performed below can propagate to the host. Without this the container
    // would modify the host's mount table, which is both a correctness
    // problem and a security one.
    stage = Stage::MountPropagation;
    {
        auto r = sys::mount(nullptr, "/", nullptr,
                            sys::kMsRecursive | sys::kMsPrivate, nullptr);
        if (r.failed()) {
            detail = "make / private";
            return r.error;
        }
        // / is the root of the container's own mount namespace now, but the
        // tree under it is still the host's. Making the whole tree private
        // is what stops a later mount from being visible outside.
        (void)sys::mount(nullptr, "/", nullptr, sys::kMsRecursive, nullptr);
    }

    // The root is assembled in this order:
    //
    //   1. bind the target read-only, as a writable bind
    //   2. mount the filesystems it needs and create its device nodes
    //   3. create the directory pivot_root moves the old root into
    //   4. flip the root read-only
    //   5. pivot
    //
    // The ordering constraint is step 3. pivot_root requires a directory
    // inside the new root to move the old one into, that directory has to be
    // created after the root is mounted because the mount hides anything
    // created before it, and it cannot be created at all once the root is
    // read-only. Binding writable and sealing last is the order in which all
    // three facts hold at once.
    stage = Stage::RootAssembly;
    // The holder has to exist before anything can be mounted into it. Only
    // the holder itself is created here; the filesystems inside it come from
    // the root that is about to be mounted over this directory.
    if (!fs::mkdir_p(root_holder, 0755)) {
        detail = "cannot create " + root_holder;
        return sys::kEacces;
    }
    {
        const int rc = assemble_root(config, root_holder, scratch, detail);
        if (rc != 0) {
            return rc;
        }
    }

    if (!create_root_skeleton(root_holder)) {
        detail = "cannot create the root skeleton under " + root_holder;
        return sys::kEacces;
    }

    // A root that could not be given /proc or device nodes is not a failed
    // setup. What was available is recorded and the run continues; a caller
    // that needs those mounts checks for them itself.
    (void)populate_root(config, root_holder);

    if (!apply_extra_mounts(config, root_holder)) {
        detail = "an extra bind mount failed";
        return sys::kEperm;
    }

    stage = Stage::PivotRoot;
    {
        // pivot_root additionally requires the new root to be a mount point
        // in its own right, and a directory under /tmp is not one. Binding
        // the directory onto itself turns it into one without changing what
        // it contains. Without this the call fails with ENOENT, which names
        // the missing path and gives no hint that the path is not a mount.
        auto b = sys::mount(root_holder.c_str(), root_holder.c_str(), nullptr,
                            sys::kMsBind | sys::kMsRecursive, nullptr);
        if (b.failed()) {
            detail = "make " + root_holder + " a mount point";
            return b.error;
        }

        const std::string put_old = root_holder + "/oldroot";
        if (!fs::mkdir_p(put_old, 0755)) {
            detail = "cannot create " + put_old;
            return sys::kEacces;
        }
    }

    // Sealing is the last step before the pivot, because everything above
    // needed to write to the root.
    {
        const int rc = seal_root(config, root_holder, detail);
        if (rc != 0) {
            return rc;
        }
    }

    {
        const std::string put_old = root_holder + "/oldroot";
        auto r = sys::pivot_root(root_holder.c_str(), put_old.c_str());
        if (r.failed()) {
            detail = "pivot_root at " + root_holder;
            return r.error;
        }

        auto c = sys::chdir("/");
        if (c.failed()) {
            detail = "chdir to the new root";
            return c.error;
        }

        // MNT_DETACH rather than a plain umount: the old root is still the
        // working directory of nothing, but it may still be reachable
        // through a path held open elsewhere, and a lazy detach is what
        // makes the unmount succeed in that case instead of returning
        // EBUSY and leaving the host's filesystem attached to the container.
        auto u = sys::umount2("/oldroot", sys::kMntDetach);
        if (u.failed()) {
            detail = "detach the old root";
            return u.error;
        }
        (void)sys::unlinkat(-100, "/oldroot", 0200);
    }

    stage = Stage::Capabilities;
    {
        const int rc = drop_capabilities(config.keep_capabilities, detail);
        if (rc != 0) {
            return rc;
        }
    }

    return 0;
}

// Runs the target inside an already-configured container. Never returns on
// success; returns a positive errno on failure.
int child_exec(const std::string& path, const std::vector<std::string>& argv,
               const std::vector<std::string>& envp, bool stop_at_exec) {
    if (stop_at_exec) {
        // TRACEME has to be called before the exec, because what it does is
        // arrange for the process to be stopped when the exec completes.
        // Called after, there is nothing left to stop.
        auto tr = sys::ptrace(0 /* PTRACE_TRACEME */, 0, nullptr, nullptr);
        if (tr.failed()) {
            return tr.error;
        }
    }

    std::vector<char*> argv_ptrs;
    argv_ptrs.reserve(argv.size() + 2);
    std::vector<std::string> argv_storage = argv;
    if (argv_storage.empty()) {
        argv_storage.push_back(path);
    }
    for (auto& a : argv_storage) {
        argv_ptrs.push_back(a.data());
    }
    argv_ptrs.push_back(nullptr);

    std::vector<char*> envp_ptrs;
    envp_ptrs.reserve(envp.size() + 1);
    std::vector<std::string> envp_storage = envp;
    for (auto& e : envp_storage) {
        envp_ptrs.push_back(e.data());
    }
    envp_ptrs.push_back(nullptr);

    auto r = sys::execve(path.c_str(), argv_ptrs.data(), envp_ptrs.data());
    return r.failed() ? r.error : 0;
}

} // namespace

// -------------------------------------------------------------------- spawn

SpawnResult container_spawn(const ContainerConfig& config,
                            const std::string& path,
                            const std::vector<std::string>& argv,
                            const std::vector<std::string>& envp) {
    SpawnResult out;

    // Two pipes. The first gates the child: it reads one byte, which the
    // parent writes after the setup steps that need a pid (the uid map, the
    // cgroup) have completed. The second carries the child's own failure
    // report. The exit status is not used for that report, because the
    // target owns its exit status and a caller has to be able to tell a
    // target that exited with code 7 from a setup step that failed.
    int gate[2] = {-1, -1};
    int report[2] = {-1, -1};

    auto pg = sys::pipe2(gate, 02000000 /* O_CLOEXEC */);
    if (pg.failed()) {
        out.error = ContainerError{Stage::Clone, pg.error, "pipe2 for the gate"};
        return out;
    }
    auto pre = sys::pipe2(report, 02000000 /* O_CLOEXEC */);
    if (pre.failed()) {
        out.error =
            ContainerError{Stage::Clone, pre.error, "pipe2 for the report"};
        (void)sys::close(gate[0]);
        (void)sys::close(gate[1]);
        return out;
    }

    const std::string scratch =
        "/tmp/occ-" +
        to_decimal(static_cast<std::uint64_t>(sys::getpid().value)) + "-root";

    const CloneOutcome cloned = clone_child(config);
    if (cloned.error != 0) {
        out.error = ContainerError{
            Stage::Clone, cloned.error,
            "creating a process with flags " + to_decimal(config.namespaces.flags())};
        (void)sys::close(gate[0]);
        (void)sys::close(gate[1]);
        (void)sys::close(report[0]);
        (void)sys::close(report[1]);
        return out;
    }

    if (cloned.in_child) {
        // Child. The stack belongs to the parent's thread, so nothing here
        // may return.
        (void)sys::close(gate[1]);
        (void)sys::close(report[0]);

        // In the fallback case the parent is waiting for this byte before it
        // can write the mappings. It is sent unconditionally rather than
        // only when a user namespace was asked for, because on every other
        // fallback path the parent is waiting for it too and a missing byte
        // would hang the run.
        const char ready = 1;
        (void)sys::write(report[1], &ready, 1);

        // Wait for the parent's go-ahead. A read that fails means the
        // parent is gone and there is nothing left to wait for.
        char byte = 0;
        for (;;) {
            auto r = sys::read(gate[0], &byte, 1);
            if (r.ok() && r.value == 1) {
                break;
            }
            if (r.ok() && r.value == 0) {
                // The parent closed the gate without writing, which is how
                // it aborts a child whose setup it has already given up on.
                sys::exit_group(126);
            }
            // A signal interrupted the read; retry.
            if (r.failed() && r.error != sys::kEintr) {
                sys::exit_group(126);
            }
        }
        (void)sys::close(gate[0]);

        Stage stage = Stage::Clone;
        std::string detail;
        const int rc = child_setup(config, scratch, scratch, stage, detail);
        if (rc != 0) {
            // Report the stage and the errno on the report pipe. The format
            // is deliberately trivial: one byte of stage code, one byte of
            // errno, then the human-readable detail. The parent reads it
            // after the child is reaped.
            std::string message;
            message.push_back(static_cast<char>(stage));
            message.push_back(static_cast<char>(rc & 0xff));
            message += detail;
            (void)sys::write(report[1], message.data(), message.size());
            (void)sys::close(report[1]);
            sys::exit_group(125);
        }

        (void)sys::close(report[1]);
        const int erc = child_exec(path, argv, envp, config.stop_at_exec);
        // exec failed. Report it the same way, and use a stage of its own so
        // the message is not confused with a mount failure.
        {
            std::string message;
            message.push_back(static_cast<char>(Stage::Exec));
            message.push_back(static_cast<char>(erc & 0xff));
            message += "exec " + path;
            (void)sys::write(2, message.data(), message.size());
        }
        sys::exit_group(126);
    }

    // Parent.
    (void)sys::close(gate[0]);
    (void)sys::close(report[1]);

    out.error = ContainerError{Stage::Clone, 0, ""};

    if (config.namespaces.user) {
        // The child creates its own user namespace on the fallback path, so
        // the writes below race the child's unshare. A single retry on
        // ENOENT is not enough to be correct, but blocking is worse: the
        // child cannot report readiness without a working /proc/<pid>
        // entry, and the entry only appears once the unshare has happened.
        //
        // The ordering that removes the race is to have the child announce
        // itself through the report pipe before the parent writes. That byte
        // is written by the child only on the fallback path, where the
        // unshare has just returned.
        if (cloned.used_fallback) {
            char ready = 0;
            for (;;) {
                auto r = sys::read(report[0], &ready, 1);
                if (r.ok() && r.value == 1) {
                    break;
                }
                if (r.ok() && r.value == 0) {
                    out.error = ContainerError{Stage::UidMap, sys::kEchild,
                                               "the child exited before the "
                                               "user namespace was ready"};
                    (void)sys::close(gate[1]);
                    (void)sys::close(report[0]);
                    int status = 0;
                    (void)sys::wait4(static_cast<int>(cloned.pid), &status, 0,
                                     nullptr);
                    return out;
                }
                if (r.failed() && r.error != sys::kEintr) {
                    out.error = ContainerError{Stage::UidMap, r.error,
                                               "read the child's readiness"};
                    (void)sys::close(gate[1]);
                    (void)sys::close(report[0]);
                    int status = 0;
                    (void)sys::wait4(static_cast<int>(cloned.pid), &status, 0,
                                     nullptr);
                    return out;
                }
            }
        }

        // /proc/<pid> refers to the child by the pid this namespace
        // assigned. On the fallback path that pid is the fork's child,
        // which is also the process that created the user namespace, so the
        // mapping is written through the same path with the same pid.
        const std::string dir =
            "/proc/" + to_decimal(static_cast<std::uint64_t>(cloned.pid));

        // The map has to be written by the process that owns the child's
        // user namespace, which is the caller.
        const auto uid =
            static_cast<std::uint32_t>(sys::getuid_syscall().value);
        const auto gid =
            static_cast<std::uint32_t>(sys::getgid_syscall().value);

        // setgroups has to be denied before the gid map is written.
        // Otherwise the child can call setgroups to acquire supplementary
        // groups the caller never granted it, which is a privilege
        // escalation out of the container.
        if (!write_string(dir + "/setgroups", "deny")) {
            out.error = ContainerError{Stage::UidMap, sys::kEperm,
                                       "write " + dir + "/setgroups"};
        } else if (!write_id_map(dir, "uid_map", config.inside_uid, uid)) {
            out.error = ContainerError{Stage::UidMap, sys::kEperm,
                                       "write " + dir + "/uid_map"};
        } else if (!write_id_map(dir, "gid_map", config.inside_gid, gid)) {
            out.error = ContainerError{Stage::UidMap, sys::kEperm,
                                       "write " + dir + "/gid_map"};
        }
    }

    if (out.error.ok() && !config.cgroup_parent.empty()) {
        std::string detail;
        const int rc =
            attach_cgroup(config, static_cast<int>(cloned.pid), detail);
        if (rc != 0) {
            out.error = ContainerError{Stage::Cgroup, rc, detail};
        }
    }

    if (!out.error.ok()) {
        // Abort. Closing the gate without writing tells the child to leave;
        // it is still blocked on the read and will see end of file.
        (void)sys::close(gate[1]);
        (void)sys::close(report[0]);
        int status = 0;
        (void)sys::wait4(static_cast<int>(cloned.pid), &status, 0, nullptr);
        return out;
    }

    // Release the child.
    const char go = 1;
    (void)sys::write(gate[1], &go, 1);
    (void)sys::close(gate[1]);

    out.pid = static_cast<int>(cloned.pid);

    // The report pipe carries the stage and errno of a child whose own
    // setup failed. The parent reads it once here, without blocking: the
    // child either fails within a few hundred microseconds of being
    // released or it does not fail at all, and blocking on a pipe that will
    // never be written to is the one thing this must not do. On the fallback
    // path the readiness byte was already consumed above, so what is read
    // here is either the genuine failure report or end of file.
    {
        char buffer[512];
        auto r = sys::read(report[0], buffer, sizeof(buffer));
        if (r.ok() && r.value >= 2) {
            const auto stage_code = static_cast<std::uint8_t>(buffer[0]);
            const auto err = static_cast<std::uint8_t>(buffer[1]);
            out.error = ContainerError{
                static_cast<Stage>(stage_code), err,
                std::string(buffer + 2, static_cast<std::size_t>(r.value) - 2)};
        }
    }
    (void)sys::close(report[0]);

    return out;
}

ContainerError container_resume(int pid) noexcept {
    // The child was released by container_spawn when it wrote the go byte
    // through the gate. Nothing is left to do here. The function exists so
    // that a caller's sequence reads as spawn, observe, resume, reap, and so
    // that an implementation which stops the child at its first instruction
    // has one place to change.
    //
    // The wait is non-blocking: a resume that blocked would serialize the
    // caller against the target, which defeats the point of the split.
    int status = 0;
    auto r = sys::wait4(pid, &status, 0x00000004 /* WNOHANG */, nullptr);
    if (r.failed()) {
        return ContainerError{Stage::Clone, r.error, "wait4"};
    }
    if (r.value != 0) {
        return ContainerError{Stage::Clone, sys::kEchild,
                              "the child exited before it was resumed"};
    }
    return ContainerError{Stage::Clone, 0, ""};
}

ContainerResult container_reap(int pid) noexcept {
    ContainerResult out;

    int status = 0;
    auto r = sys::wait4(pid, &status, 0, nullptr);
    if (r.failed()) {
        out.error = ContainerError{Stage::Clone, r.error, "wait4"};
        return out;
    }

    // A child that failed during its own setup exits with a sentinel code of
    // its own choosing. The stage and errno travel on a separate pipe, read
    // by container_spawn, so this code only has to say "setup failed" and
    // not which step it was.
    constexpr int kSetupFailed = 125;
    constexpr int kExecFailed = 126;

    if (WIFEXITED(status)) {
        const int code = WEXITSTATUS(status);
        if (code == kSetupFailed) {
            out.error = ContainerError{Stage::RootAssembly, 0,
                                       "the container setup failed"};
            return out;
        }
        if (code == kExecFailed) {
            out.error =
                ContainerError{Stage::Exec, 0, "the target could not be run"};
            return out;
        }
        out.exit_code = code;
        return out;
    }
    if (WIFSIGNALED(status)) {
        out.signaled = true;
        out.term_signal = WTERMSIG(status);
        return out;
    }

    out.error = ContainerError{Stage::Clone, sys::kEchild,
                               "the child neither exited nor was signaled"};
    return out;
}

// ----------------------------------------------------------------------- run

ContainerResult run_container(const ContainerConfig& config,
                              const std::string& path,
                              const std::vector<std::string>& argv,
                              const std::vector<std::string>& envp) {
    ContainerResult out;

    SpawnResult spawned = container_spawn(config, path, argv, envp);
    if (!spawned.error.ok()) {
        out.error = spawned.error;
        return out;
    }

    out = container_reap(spawned.pid);
    (void)container_cleanup(config);
    return out;
}

long container_cleanup(const ContainerConfig& config) noexcept {
    long removed = 0;

    if (!config.cgroup_parent.empty()) {
        // The kernel refuses to remove a cgroup that still has processes in
        // it, so a failed run leaves the directory behind. Reporting that
        // is more useful than retrying, because the cause is a process the
        // caller still owns.
        const std::string dir = config.cgroup_parent + "/occ";
        if (fs::exists(dir)) {
            if (sys::unlinkat(-100, dir.c_str(), 0200).ok()) {
                ++removed;
            } else {
                removed = -1;
            }
        }
    }

    if (config.root_kind == RootKind::Overlay) {
        if (!config.upper_dir.empty() && fs::exists(config.upper_dir)) {
            const long n = fs::remove_tree(config.upper_dir);
            if (n < 0) {
                removed = -1;
            } else {
                removed += n;
            }
        }
        if (!config.work_dir.empty() && fs::exists(config.work_dir)) {
            const long n = fs::remove_tree(config.work_dir);
            if (n < 0) {
                removed = -1;
            } else {
                removed += n;
            }
        }
    }

    return removed;
}

} // namespace occ::isolation
