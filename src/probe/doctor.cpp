#include "occ/doctor.h"

#include "occ/util/fs.h"
#include "occ/util/string.h"
#include "occ/syscall/syscall.h"

#include <cstring>
#include <string>
#include <vector>

namespace occ {

namespace {

// System identity, via uname(2). Fields are read from a local copy of the
// structure; the kernel ABI for utsname is frozen and the sizes are
// generous, so a plain buffer is safe here.
struct UtsName {
    char sysname[65];
    char nodename[65];
    char release[65];
    char version[65];
    char machine[65];
    char domainname[65];
};

std::string errno_text(int err) {
    char buf[96]{};
    const char* p = ::strerror_r(err, buf, sizeof(buf) - 1);
    return {p};
}

Check ok(std::string name, std::string detail) {
    return Check{std::move(name), CheckStatus::Ok, std::move(detail)};
}
Check warn(std::string name, std::string detail) {
    return Check{std::move(name), CheckStatus::Warn, std::move(detail)};
}
Check fail(std::string name, std::string detail) {
    return Check{std::move(name), CheckStatus::Fail, std::move(detail)};
}

// ---------------------------------------------------------------- probes

Check probe_kernel() {
    UtsName u{};
    auto r = sys::uname(&u);
    if (r.failed()) {
        return warn("kernel", "uname failed: " + errno_text(r.error));
    }
    return ok("kernel", std::string(u.release) + " " + u.machine);
}

Check probe_identity() {
    auto uid = sys::getuid_syscall();
    auto gid = sys::getgid_syscall();
    if (uid.failed() || gid.failed()) {
        return warn("identity", "could not determine uid/gid");
    }

    std::string detail = "uid=";
    append_int(detail, uid.value);
    detail += " gid=";
    append_int(detail, gid.value);

    if (uid.value == 0) {
        detail += " (root)";
        return ok("identity", std::move(detail));
    }
    detail += " (capability set required, see docs/SECURITY.md)";
    return warn("identity", std::move(detail));
}

Check probe_user_namespaces() {
    auto v = fs::read_file("/proc/sys/user/max_user_namespaces");
    if (!v) {
        return warn("user namespaces", "limit not readable");
    }
    const auto n = trim(*v);
    if (n == "0") {
        return fail("user namespaces", "disabled (max_user_namespaces=0)");
    }
    return ok("user namespaces", "max " + std::string(n));
}

Check probe_mount() {
    if (!fs::can_mount_tmpfs()) {
        return fail("mount", "tmpfs mount refused; CAP_SYS_ADMIN and a "
                             "private mount namespace are both required");
    }
    return ok("mount", "tmpfs mount succeeded in a private namespace");
}

Check probe_overlay() {
    // /proc/filesystems lists filesystem types already registered with the
    // kernel. A modular filesystem that has not been loaded yet is absent
    // from that list even though mount(2) will load it on demand. Since
    // overlay is modular on most distributions, the list is not the answer:
    // the only reliable test is to try a mount.
    if (fs::can_mount_overlay()) {
        return ok("overlayfs", "test mount succeeded");
    }

    if (!fs::filesystem_supports("overlay")) {
        return fail("overlayfs",
                    "not registered and a test mount failed; the root "
                    "filesystem layer cannot be built");
    }

    return fail("overlayfs",
                "registered but a test mount failed; the upper and work "
                "directories must be on the same filesystem and neither may "
                "be a descendant of the other");
}

Check probe_cgroup() {
    if (!fs::exists("/sys/fs/cgroup/cgroup.controllers")) {
        return fail("cgroup v2",
                    "/sys/fs/cgroup is not a cgroup2 mount; resource limits "
                    "and process placement are unavailable");
    }

    auto ctrl = fs::read_file("/sys/fs/cgroup/cgroup.controllers");
    if (!ctrl) {
        return warn("cgroup v2", "mounted but the controller list is "
                                 "unreadable");
    }

    std::string detail = "controllers: ";
    detail += trim(*ctrl);

    std::string have;
    auto subtree = fs::read_file("/sys/fs/cgroup/cgroup.subtree_control");
    if (subtree) {
        have = std::string(trim(*subtree));
    }

    if (have.find("memory") == std::string::npos) {
        detail += " | subtree_control lacks +memory; occ enables it at run "
                  "time";
        return warn("cgroup v2", std::move(detail));
    }
    return ok("cgroup v2", std::move(detail));
}

Check probe_seccomp() {
    auto v = fs::read_file("/proc/sys/kernel/seccomp/actions_avail");
    if (!v) {
        return fail("seccomp", "actions_avail missing; the kernel predates "
                               "the seccomp filter interface occ requires");
    }
    return ok("seccomp", "actions: " + std::string(trim(*v)));
}

Check probe_bpf() {
    auto v = fs::read_file("/proc/sys/kernel/unprivileged_bpf_disabled");
    if (!v) {
        return warn("bpf", "unprivileged_bpf_disabled not readable");
    }
    const auto n = std::string(trim(*v));
    if (n == "0") {
        return ok("bpf", "unprivileged bpf enabled");
    }
    return ok("bpf", "unprivileged bpf disabled (value " + n +
                         "); root with CAP_BPF is required");
}

Check probe_perf() {
    auto v = fs::read_file("/proc/sys/kernel/perf_event_paranoid");
    if (!v) {
        return warn("perf_event", "perf_event_paranoid not readable");
    }
    const auto n = std::string(trim(*v));
    if (n == "-1" || n == "0" || n == "1" || n == "2") {
        return ok("perf_event", "paranoid " + n +
                                    "; hardware breakpoints available");
    }
    return warn("perf_event", "paranoid " + n +
                                  "; root required for breakpoints");
}

Check probe_tracefs() {
    for (const char* p : {"/sys/kernel/tracing/events",
                          "/sys/kernel/debug/tracing/events"}) {
        if (fs::exists(p)) {
            return ok("tracefs", std::string(p) +
                                     " available for tracepoint field reads");
        }
    }
    return fail("tracefs", "not mounted; run: mount -t tracefs nodev "
                           "/sys/kernel/tracing");
}

Check probe_binderfs() {
    if (fs::exists("/dev/binderfs")) {
        return ok("binderfs", "/dev/binderfs mounted");
    }
    if (fs::filesystem_supports("binder")) {
        return ok("binderfs", "supported by the kernel, not yet mounted; "
                              "required only by the apk engine");
    }
    return warn("binderfs", "unavailable; the apk engine cannot run on this "
                            "host");
}

Check probe_config() {
    UtsName u{};
    if (sys::uname(&u).ok()) {
        const std::string path = "/boot/config-" + std::string(u.release);
        if (fs::exists(path)) {
            return ok("kernel config", path);
        }
    }
    if (fs::exists("/proc/config.gz")) {
        return ok("kernel config", "/proc/config.gz");
    }
    return warn("kernel config",
                "unavailable; cloud images commonly ship neither "
                "/boot/config-$(uname -r) nor /proc/config.gz, so CONFIG_* "
                "assertions cannot be checked here");
}

Check probe_procfs() {
    if (!fs::exists("/proc/self/stat")) {
        return fail("procfs", "/proc is not mounted");
    }
    return ok("procfs", "mounted");
}

} // namespace

bool DoctorReport::has_failure() const noexcept {
    for (const auto& c : checks) {
        if (c.status == CheckStatus::Fail) {
            return true;
        }
    }
    return false;
}

int DoctorReport::exit_code() const noexcept { return has_failure() ? 1 : 0; }

DoctorReport run_doctor() noexcept {
    DoctorReport report;
    report.checks.reserve(16);

    report.checks.push_back(probe_identity());
    report.checks.push_back(probe_kernel());
    report.checks.push_back(probe_procfs());
    report.checks.push_back(probe_user_namespaces());
    report.checks.push_back(probe_mount());
    report.checks.push_back(probe_overlay());
    report.checks.push_back(probe_cgroup());
    report.checks.push_back(probe_seccomp());
    report.checks.push_back(probe_bpf());
    report.checks.push_back(probe_perf());
    report.checks.push_back(probe_tracefs());
    report.checks.push_back(probe_binderfs());
    report.checks.push_back(probe_config());

    return report;
}

} // namespace occ
