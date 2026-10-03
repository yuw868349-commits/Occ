#pragma once

// Seccomp-BPF program construction.
//
// The filter bytecode is emitted here rather than delegated to libseccomp,
// because the kernel only ever executes bytecode: a library adds a policy
// compiler and a syscall-name table that are pinned to the library release
// rather than to the kernel the filter runs on, and both of those are things
// this project would then have to trust.
//
// Shape of a generated program:
//
//   load arch
//   if arch != AUDIT_ARCH_X86_64 -> reject every call
//   load nr
//   if nr > max_nr -> reject
//   jump to the block for nr, else fall through
//   ...
//   each block returns its action, or tests argument conditions first
//   fallthrough -> the policy's fallback action
//
// A syscall the policy does not mention is handled by the fallback, which is
// rejection unless the caller says otherwise. Default-closed is the only
// default that can be reasoned about: a syscall added by a future kernel is
// not silently permitted.
//
// The `nr > max_nr` test is not an optimisation. On x86-64 the kernel sets
// bit 30 of the syscall number for x32 callers, so a number seen at the
// seccomp boundary has to be rejected on the full value. A filter that
// dispatched on the low bits would let an x32 caller reach the native entry
// point under a different number.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <linux/filter.h>
#include <linux/seccomp.h>

#include "occ/util/span.h"

namespace occ::sys {

// What a matching syscall does.
enum class SeccompAction : std::uint8_t {
    // Run the syscall normally. A policy that names a syscall only to allow
    // it says something specific: the number is known to be needed and has
    // been considered, which is different from leaving it to the fallback.
    Allow,
    // Return an error to the caller without running the syscall.
    Errno,
    // Deliver SIGSYS and let a tracer decide. Requires a tracer that handles
    // PTRACE_EVENT_SECCOMP; without one the process dies.
    Trap,
    // Deliver SIGSYS and kill the thread regardless of any handler.
    KillThread,
    // Deliver SIGSYS and kill the whole thread group.
    KillProcess,
};

// Comparison applied to one syscall argument.
enum class SeccompCmp : std::uint8_t {
    Equal,
    NotEqual,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
    // True when (arg & value) != 0. This is the bit-test instruction, not an
    // equality test, and is what expresses "any of these flag bits set".
    Masked,
};

// One argument condition. The index runs 0..5 in the order the syscall's
// manual page numbers them, which is also the order they appear in
// seccomp_data.args.
struct SeccompArgTest {
    std::uint8_t index;
    SeccompCmp cmp;
    std::uint32_t value;
};

// One rule. A rule with no argument tests matches on the number alone.
struct SeccompRule {
    std::uint32_t nr;
    SeccompAction action;
    // Used only by Action::Errno. The kernel negates this, so passing 1 here
    // makes the syscall return EPERM to the caller.
    int error;
    std::vector<SeccompArgTest> args;
};

enum class SeccompDefault : std::uint8_t {
    Allow,
    Errno,
    Trap,
    KillProcess,
};

struct SeccompPolicy {
    // Applied to any syscall no rule mentions.
    SeccompDefault fallback = SeccompDefault::Errno;
    // Used only when the fallback is Errno.
    int fallback_error = 38; // ENOSYS
    std::vector<SeccompRule> rules;
    // Anything above this number is rejected without being dispatched, which
    // is what closes the x32 hole described above.
    std::uint32_t max_nr = 0;
};

class SeccompProgram {
public:
    [[nodiscard]] bool valid() const noexcept { return valid_; }
    [[nodiscard]] const char* error() const noexcept { return error_; }

    [[nodiscard]] const struct sock_filter* code() const noexcept {
        return insns_.empty() ? nullptr : insns_.data();
    }
    [[nodiscard]] std::size_t insn_count() const noexcept {
        return insns_.size();
    }
    [[nodiscard]] std::size_t byte_count() const noexcept {
        return insns_.size() * sizeof(struct sock_filter);
    }

private:
    friend SeccompProgram build_seccomp(const SeccompPolicy&) noexcept;

    std::vector<struct sock_filter> insns_;
    bool valid_ = false;
    const char* error_ = "not built";
};

// Compiles the policy into bytecode. Building never throws and never installs
// anything. On failure the result reports valid() == false together with a
// description; that description is a string literal with static storage, so
// it stays valid for the lifetime of the process.
[[nodiscard]] SeccompProgram build_seccomp(const SeccompPolicy& policy) noexcept;

// Installs a compiled program on the calling thread.
//
// SECCOMP_FILTER_FLAG_TSYNC is not used. It can only succeed while the
// process is single-threaded, so a caller that needs it is a caller that has
// already lost, and the kernel's failure mode for the unsupported cases is
// not distinguishable from a filter that is simply wrong.
//
// Returns 0 on success or a negative errno.
[[nodiscard]] int seccomp_install(const SeccompProgram& program) noexcept;

// Installs a compiled program on another thread, which must be stopped in a
// ptrace stop. The tracer uses this to filter a child that it is about to
// release. Returns 0 on success or a negative errno.
[[nodiscard]] int seccomp_install_on_thread(const SeccompProgram& program,
                                            int tid) noexcept;

// The highest syscall number this build knows about. A policy's max_nr
// defaults to this, so the dispatch table is bounded by the build's headers
// rather than by whichever numbers a profile happened to mention.
[[nodiscard]] std::uint32_t highest_known_syscall() noexcept;

} // namespace occ::sys
