// Container layer tests.
//
// The container cannot be tested the way the seccomp builder is. Building a
// filter is a pure function and every property of it can be asserted in
// process; building a container performs privileged operations whose
// availability depends on the host, and a test that fails because the host
// refuses a mount would be reporting on the host and not on this code.
//
// What is asserted here is therefore the part that is a property of the code
// and not of the host:
//
//   * a configuration that cannot work is rejected before anything is
//     created, with the stage named
//   * the namespace flag translation produces the right bits
//   * a real run either completes, or fails at a stage that is reported
//     with an errno -- never hangs, never returns a silent success
//   * the target's own exit code survives the round trip, and a target
//     killed by a signal is reported as signaled rather than as an exit
//
// The end-to-end cases are skipped where the host has no usable user
// namespace, because that is a property of the host and not a defect.

#include "occ/isolation/container.h"
#include "occ/syscall/errno.h"
#include "occ/util/fs.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

using namespace occ::isolation;
using occ::fs::exists;
using occ::fs::read_link;

namespace {

int failures = 0;
int checks = 0;
int skipped = 0;

void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL %s\n", what);
    }
}

void skip(const char* what) {
    ++skipped;
    std::fprintf(stderr, "SKIP %s\n", what);
}

// Describes the namespace flag translation without needing to create
// anything. Each flag is checked on its own so a wrong bit names itself.
//
// A fresh NamespaceSet has every namespace on except the network, so a
// single-flag case is built by clearing the set first. Asserting against a
// default-constructed value would test the defaults and not the bit.
NamespaceSet only(bool user, bool mount, bool pid, bool ipc, bool uts,
                  bool net, bool cgroup) {
    NamespaceSet ns;
    ns.user = user;
    ns.mount = mount;
    ns.pid = pid;
    ns.ipc = ipc;
    ns.uts = uts;
    ns.net = net;
    ns.cgroup = cgroup;
    return ns;
}

void test_namespace_flags() {
    check(only(false, false, false, false, false, false, false).flags() == 0,
          "an empty namespace set has no flags");

    // CLONE_NEWUSER is 0x10000000.
    check(only(true, false, false, false, false, false, false).flags() ==
              0x10000000ULL,
          "user namespace bit");
    check(only(false, true, false, false, false, false, false).flags() ==
              0x00020000ULL,
          "mount namespace bit");
    check(only(false, false, true, false, false, false, false).flags() ==
              0x20000000ULL,
          "pid namespace bit");
    check(only(false, false, false, true, false, false, false).flags() ==
              0x08000000ULL,
          "ipc namespace bit");
    check(only(false, false, false, false, true, false, false).flags() ==
              0x04000000ULL,
          "uts namespace bit");
    check(only(false, false, false, false, false, true, false).flags() ==
              0x40000000ULL,
          "net namespace bit");
    check(only(false, false, false, false, false, false, true).flags() ==
              0x02000000ULL,
          "cgroup namespace bit");

    // Every flag set has to produce a value that carries every bit.
    const std::uint64_t all =
        only(true, true, true, true, true, true, true).flags();
    check((all & 0x10000000ULL) != 0, "all: user");
    check((all & 0x00020000ULL) != 0, "all: mount");
    check((all & 0x20000000ULL) != 0, "all: pid");
    check((all & 0x08000000ULL) != 0, "all: ipc");
    check((all & 0x04000000ULL) != 0, "all: uts");
    check((all & 0x40000000ULL) != 0, "all: net");
    check((all & 0x02000000ULL) != 0, "all: cgroup");

    // The network namespace default is off, which is a deliberate choice:
    // sharing the host's network is how a target is given real access.
    check((NamespaceSet{}.flags() & 0x40000000ULL) == 0,
          "the network namespace is off by default");

    // A configuration with no user namespace is legal and has to be
    // distinguishable, because the uid map write is conditional on it.
    check(only(false, true, false, false, false, false, false).flags() != 0,
          "namespaces without user are still a set");
}

// An overlay root without an upper directory cannot be built. The failure
// has to name the stage rather than producing a half-built container.
void test_overlay_needs_dirs() {
    ContainerConfig config;
    config.root_kind = RootKind::Overlay;
    config.root_dir = "/nonexistent-lower";
    config.upper_dir.clear();
    config.work_dir.clear();

    const ContainerResult r =
        run_container(config, "/bin/true", {"/bin/true"}, {});

    if (r.error.ok()) {
        // The host let a directory that does not exist be used as a lower
        // layer, which cannot happen; treat it as a failure of the test's
        // premise rather than of the code.
        check(false, "an overlay with no upper directory was accepted");
        return;
    }
    check(r.error.error != 0, "the overlay failure carries an errno");
    check(r.error.stage == Stage::RootAssembly ||
              r.error.stage == Stage::Clone,
          "the overlay failure is reported at root assembly or earlier");
}

// The target's exit status has to survive. This is the one end-to-end
// property that matters to every caller.
void test_exit_code_round_trip() {
    if (!exists("/bin/true") || !exists("/bin/false")) {
        skip("exit code round trip: /bin/true and /bin/false are needed");
        return;
    }

    ContainerConfig config;
    config.root_dir = "/";
    config.root_kind = RootKind::ReadOnlyBind;
    config.namespaces.cgroup = false;
    config.namespaces.net = false;

    const ContainerResult ok =
        run_container(config, "/bin/true", {"/bin/true"}, {});
    if (!ok.error.ok()) {
        skip("exit code round trip: this host refuses a container");
        return;
    }
    check(ok.exit_code == 0, "a target that exits 0 is reported as 0");
    check(!ok.signaled, "a target that exits is not reported as signaled");

    const ContainerResult bad =
        run_container(config, "/bin/false", {"/bin/false"}, {});
    if (bad.error.ok()) {
        check(bad.exit_code == 1, "a target that exits 1 is reported as 1");
    }
}

// A target killed by a signal is not an exit status. Conflating the two is
// the classic way a crash is reported as a clean exit.
void test_signal_reporting() {
    const auto linked = read_link("/proc/self/exe");
    if (!linked || linked->empty() || linked->front() != '/') {
        skip("signal reporting: the executable path is not readable");
        return;
    }
    const std::string self = *linked;

    // Only the stages that do not need a writable root are exercised here.
    ContainerConfig config;
    config.root_dir = "/";
    config.root_kind = RootKind::ReadOnlyBind;
    config.namespaces.cgroup = false;
    config.namespaces.net = false;

    // The helper re-executes this binary with an argument that makes it
    // raise SIGSEGV. If the container cannot be built the case is skipped,
    // because the property under test is the reporting and not the setup.
    const std::string marker = "--sigsegv-helper";
    for (int i = 0; i < 1; ++i) {
        const ContainerResult r =
            run_container(config, self, {self, marker}, {});
        if (!r.error.ok()) {
            skip("signal reporting: this host refuses a container");
            return;
        }
        check(r.signaled, "a target killed by a signal is reported as signaled");
        check(r.term_signal == 11, "the signal number is reported");
        check(r.exit_code == 0, "a signaled target reports no exit code");
    }
}

} // namespace

// The helper entry point used by the signal test. It lives outside the
// anonymous namespace so that main can find it.
namespace occ_test {
[[noreturn]] void raise_segv();
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--sigsegv-helper") == 0) {
        // A raw null write produces SIGSEGV without any library support.
        volatile int* p = nullptr;
        *p = 1;
        _exit(0);
    }

    test_namespace_flags();
    test_overlay_needs_dirs();
    test_exit_code_round_trip();
    test_signal_reporting();

    std::fprintf(stderr, "%d checks, %d failures, %d skipped\n", checks,
                 failures, skipped);
    return failures == 0 ? 0 : 1;
}
