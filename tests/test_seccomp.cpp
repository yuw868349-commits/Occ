// Verifies the seccomp program builder against the running kernel.
//
// Two things make this harder than it looks, and both are properties of the
// system under test rather than of the test.
//
// The first is reporting. A filter that denies write(2) also denies the
// child's own diagnostics, so no amount of care with a file descriptor makes
// the observations visible. The results are written to a shared anonymous
// mapping instead: storing to memory needs no syscall, so every observation
// survives the child, and the parent reads them after a wait.
//
// The second is that the filter applies to the process that installed it,
// including the call that returns from installing it. Everything the child
// needs in order to terminate is therefore part of the policy, and the
// policy is what the test is checking.

#include "occ/isolation/seccomp.h"
#include "occ/syscall/syscall.h"

#include <cstdio>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using occ::sys::build_seccomp;
using occ::sys::SeccompAction;
using occ::sys::SeccompArgTest;
using occ::sys::SeccompCmp;
using occ::sys::SeccompDefault;
using occ::sys::SeccompPolicy;
using occ::sys::SeccompRule;
using occ::sys::highest_known_syscall;

int g_failures = 0;

void check(const char* label, long got, long want) {
    const bool pass = (got == want);
    std::printf("%s  %-52s got %4ld  want %4ld\n", pass ? "PASS" : "FAIL",
                label, got, want);
    if (!pass) {
        ++g_failures;
    }
}

void check_true(const char* label, bool value) {
    std::printf("%s  %s\n", value ? "PASS" : "FAIL", label);
    if (!value) {
        ++g_failures;
    }
}

// What the child observes. Plain data, so filling it needs no syscall.
struct Observations {
    long build_ok;
    long insn_count;
    long no_new_privs;
    long install;
    long write_ret;
    long getpid_ret;
    long openat_plain_ret;
    long openat_create_ret;
    long unlisted_ret;
    long allowed_ret;
};

constexpr std::uint32_t kNrWrite = 1;
constexpr std::uint32_t kNrGetpid = 39;
constexpr std::uint32_t kNrOpenat = 257;
constexpr std::uint32_t kNrGetuid = 102;
constexpr std::uint32_t kNrExitGroup = 231;
constexpr std::uint32_t kNrRtSigreturn = 15;

void run_child(Observations* out) {
    SeccompPolicy policy;
    policy.fallback = SeccompDefault::Errno;
    policy.fallback_error = 38; // ENOSYS
    policy.max_nr = highest_known_syscall();

    // Deny outright, with a number that distinguishes this rule from the
    // others in the policy.
    policy.rules.push_back(SeccompRule{kNrWrite, SeccompAction::Errno, 22, {}});
    policy.rules.push_back(SeccompRule{kNrGetpid, SeccompAction::Errno, 30, {}});

    // Deny only when O_CREAT is among the flags. Every other flag set falls
    // through to the next rule and then to the fallback, which is what makes
    // the argument test observable.
    policy.rules.push_back(SeccompRule{
        kNrOpenat, SeccompAction::Errno, 1,
        {SeccompArgTest{2, SeccompCmp::Masked, 0100}}});

    // Allow with no argument test, to confirm a rule can permit as well as
    // deny.
    policy.rules.push_back(
        SeccompRule{kNrGetuid, SeccompAction::Allow, 0, {}});

    // What a process needs in order to finish cleanly. The filter applies to
    // the exit path, so leaving these out would make the child die on a
    // denied exit_group and report nothing.
    policy.rules.push_back(
        SeccompRule{kNrExitGroup, SeccompAction::Allow, 0, {}});
    policy.rules.push_back(
        SeccompRule{kNrRtSigreturn, SeccompAction::Allow, 0, {}});

    const auto program = build_seccomp(policy);
    out->build_ok = program.valid() ? 1 : 0;
    out->insn_count = static_cast<long>(program.insn_count());
    if (!program.valid()) {
        _exit(1);
    }

    out->no_new_privs = occ::sys::prctl(38 /* PR_SET_NO_NEW_PRIVS */, 1, 0, 0, 0)
                            .value;
    out->install = occ::sys::seccomp_install(program);

    out->write_ret = occ::sys::write(1, "x", 1).value;
    out->getpid_ret = occ::sys::getpid().value;
    out->openat_plain_ret =
        occ::sys::openat(-100, "/nonexistent-occ-xyz", 0, 0).value;
    out->openat_create_ret =
        occ::sys::openat(-100, "/nonexistent-occ-xyz", 0100, 0644).value;
    out->unlisted_ret = occ::sys::nanosleep(nullptr, nullptr).value;
    out->allowed_ret = occ::sys::getuid_syscall().value;

    _exit(0);
}

} // namespace

int main() {
    // A rule that would make a syscall appear to succeed without running.
    // The kernel reads SECCOMP_RET_ERRNO with a zero payload as "return
    // zero", which is neither an error nor a success and which sends a
    // caller of read into an end-of-file loop. The builder refuses it.
    {
        SeccompPolicy policy;
        policy.fallback_error = 38;
        policy.rules.push_back(
            SeccompRule{kNrWrite, SeccompAction::Errno, 0, {}});
        const auto program = build_seccomp(policy);
        check_true("a zero errno is refused", !program.valid());
    }

    // An argument index past the sixth cannot be expressed in seccomp_data
    // and would silently read an adjacent field.
    {
        SeccompPolicy policy;
        policy.fallback_error = 38;
        policy.rules.push_back(SeccompRule{
            kNrWrite, SeccompAction::Errno, 22,
            {SeccompArgTest{9, SeccompCmp::Equal, 0}}});
        const auto program = build_seccomp(policy);
        check_true("an argument index above 5 is refused", !program.valid());
    }

    auto* out = static_cast<Observations*>(
        ::mmap(nullptr, sizeof(Observations), PROT_READ | PROT_WRITE,
               MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    if (out == MAP_FAILED) {
        std::printf("FAIL  mmap\n");
        return 1;
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        std::printf("FAIL  fork\n");
        return 1;
    }
    if (pid == 0) {
        run_child(out);
    }

    int status = 0;
    (void)::waitpid(pid, &status, 0);

    check_true("the policy compiles", out->build_ok == 1);
    check("instruction count is positive", out->insn_count > 0 ? 1 : 0, 1);
    check("PR_SET_NO_NEW_PRIVS succeeds", out->no_new_privs, 0);
    check("the filter installs", out->install, 0);

    check("write(2) returns the rule's errno", out->write_ret, -22);
    check("getpid(2) returns the rule's errno", out->getpid_ret, -30);
    check("openat(2) without O_CREAT reaches the fallback",
          out->openat_plain_ret, -38);
    check("openat(2) with O_CREAT matches the argument test",
          out->openat_create_ret, -1);
    check("a syscall no rule mentions reaches the fallback",
          out->unlisted_ret, -38);
    check("a rule that allows runs the syscall", out->allowed_ret, 0);

    check_true("the child exited normally", WIFEXITED(status) == 1);
    check_true("the child exited with status zero",
               WIFEXITED(status) == 1 && WEXITSTATUS(status) == 0);

    (void)::munmap(out, sizeof(Observations));

    if (g_failures != 0) {
        std::printf("%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
