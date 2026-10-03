#include "occ/isolation/seccomp.h"

#include "occ/syscall/errno.h"
#include "occ/syscall/syscall.h"

#include <linux/filter.h>
#include <linux/prctl.h>
#include <linux/seccomp.h>

namespace occ::sys {

namespace {

using Insn = struct sock_filter;

// ---------------------------------------------------------------- encoding
//
// Instructions are built from the BPF_* mnemonics in <linux/filter.h> so
// that a disassembly of the finished program and this source can be read side
// by side. `k` is the immediate; `jt` and `jf` are the branch offsets for a
// conditional jump, and `k` carries the offset for an unconditional one.
//
// <linux/filter.h> names the classic BPF tests from the original paper:
// BPF_JEQ, BPF_JGT, BPF_JGE, BPF_JSET. The seccomp language also accepts JNE,
// JLT and JLE, whose encodings are part of the operation field in
// <linux/bpf_common.h>. They are written out because the uapi header does not
// export the names.
constexpr std::uint16_t kJne = 0x50;
constexpr std::uint16_t kJlt = 0xa0;
constexpr std::uint16_t kJle = 0xb0;

[[nodiscard]] constexpr Insn stmt(std::uint16_t code, std::uint32_t k) noexcept {
    return Insn{code, 0, 0, k};
}

[[nodiscard]] constexpr Insn branch(std::uint16_t code, std::uint32_t k,
                                    std::uint8_t jt,
                                    std::uint8_t jf) noexcept {
    return Insn{code, jt, jf, k};
}

// Offsets into struct seccomp_data: nr, arch, instruction_pointer, then six
// 64-bit arguments starting at byte 16.
constexpr std::uint32_t kOffNr = 0;
constexpr std::uint32_t kOffArch = 4;
constexpr std::uint32_t kOffArgs = 16;

[[nodiscard]] constexpr Insn ld_nr() noexcept {
    return stmt(BPF_LD | BPF_W | BPF_ABS, kOffNr);
}

[[nodiscard]] constexpr Insn ld_arch() noexcept {
    return stmt(BPF_LD | BPF_W | BPF_ABS, kOffArch);
}

// seccomp_data holds each argument as 64 bits and is native-endian. A test
// reads the high half first and requires it to be zero, because a 32-bit
// comparison alone also matches a truncated value: a test for arg == 0 would
// otherwise accept arg == 0x1_00000000, which turns a range check on a size
// or a flag test on an address into a bypass.
[[nodiscard]] constexpr Insn ld_arg_lo(std::uint8_t index) noexcept {
    return stmt(BPF_LD | BPF_W | BPF_ABS,
                kOffArgs + static_cast<std::uint32_t>(index) * 8u);
}

[[nodiscard]] constexpr Insn ld_arg_hi(std::uint8_t index) noexcept {
    return stmt(BPF_LD | BPF_W | BPF_ABS,
                kOffArgs + static_cast<std::uint32_t>(index) * 8u + 4u);
}

[[nodiscard]] constexpr std::uint32_t ret_errno(int err) noexcept {
    const auto magnitude =
        static_cast<std::uint32_t>(err < 0 ? -err : err) & 0xffffu;
    return static_cast<std::uint32_t>(SECCOMP_RET_ERRNO) | magnitude;
}

[[nodiscard]] constexpr std::uint32_t ret_for(SeccompAction action,
                                              int error) noexcept {
    switch (action) {
    case SeccompAction::Allow:
        return SECCOMP_RET_ALLOW;
    case SeccompAction::Errno:
        return ret_errno(error);
    case SeccompAction::Trap:
        return SECCOMP_RET_TRAP;
    case SeccompAction::KillThread:
        return SECCOMP_RET_KILL_THREAD;
    case SeccompAction::KillProcess:
        return SECCOMP_RET_KILL_PROCESS;
    }
    return SECCOMP_RET_KILL_PROCESS;
}

[[nodiscard]] constexpr std::uint32_t ret_for(SeccompDefault fallback,
                                              int error) noexcept {
    switch (fallback) {
    case SeccompDefault::Allow:
        return SECCOMP_RET_ALLOW;
    case SeccompDefault::Errno:
        return ret_errno(error);
    case SeccompDefault::Trap:
        return SECCOMP_RET_TRAP;
    case SeccompDefault::KillProcess:
        return SECCOMP_RET_KILL_PROCESS;
    }
    return SECCOMP_RET_KILL_PROCESS;
}

// A classic BPF branch offset is eight bits, so one branch reaches 255
// instructions forward. The layout below keeps every branch within that
// window by construction. A policy too large for it is refused rather than
// emitted with a wrapped offset: a wrapped offset still loads and still
// returns a decision, and it is the wrong one.
constexpr std::size_t kMaxOffset = 255;

// Instructions a rule block occupies per argument test. Reading the high
// half, rejecting a non-zero high half, reading the low half, and comparing:
// five instructions. Keeping this a named constant means the offsets below
// are arithmetic on a value rather than integers that have to be recounted
// every time a test is added.
constexpr std::size_t kInsnPerTest = 5;

// SECCOMP_RET_ERRNO with a zero payload is refused. The kernel reads it as
// "make the syscall return 0", which is neither an error nor a success: a
// caller of read() sees zero bytes, concludes end of file, and loops, and a
// caller flushing a stdio buffer loops forever because the write reports
// success without consuming anything. This is not a hypothetical: it is what
// the first version of this file did to its own self-test, and the symptom
// was a program that hung without printing anything.
[[nodiscard]] bool errno_is_valid(int err) noexcept { return err != 0; }

// The size of the block that implements one rule.
[[nodiscard]] constexpr std::size_t block_size(std::size_t arg_count) noexcept {
    if (arg_count == 0) {
        return 1; // just the action
    }
    return arg_count * kInsnPerTest + 1; // the tests, then the action
}

} // namespace

std::uint32_t highest_known_syscall() noexcept {
    // Native x86-64 numbers run to 463. The ceiling sits above that and below
    // bit 30, which the kernel sets for x32 callers, so a number the filter
    // cannot dispatch is refused rather than compared and an x32 caller
    // cannot reach a native syscall under a shifted number.
    return 0x1ffu;
}

SeccompProgram build_seccomp(const SeccompPolicy& policy) noexcept {
    SeccompProgram out;

    const std::uint32_t max_nr =
        policy.max_nr != 0 ? policy.max_nr : highest_known_syscall();

    const std::size_t rule_count = policy.rules.size();

    if (policy.fallback == SeccompDefault::Errno &&
        !errno_is_valid(policy.fallback_error)) {
        out.error_ = "fallback errno is zero";
        return out;
    }
    for (const auto& rule : policy.rules) {
        if (rule.action == SeccompAction::Errno && !errno_is_valid(rule.error)) {
            out.error_ = "rule errno is zero";
            return out;
        }
        for (const auto& test : rule.args) {
            if (test.index > 5) {
                out.error_ = "argument index above 5";
                return out;
            }
        }
    }

    // Layout. Every branch is forward, which is what a classic BPF program
    // requires: an unconditional jump encodes a positive offset only.
    //
    //   [0] ld arch
    //   [1] jeq AUDIT_ARCH_X86_64 : match skips [2]
    //   [2] ret KILL_PROCESS                     wrong architecture
    //   [3] ld nr
    //   [4] jgt max_nr : over the ceiling jumps [5]
    //   [5] ret FALLBACK                         number out of range
    //   then, for each rule in order:
    //     A_i  jeq nr_i : match skips the pad; a miss jumps past the block
    //     P_i  ja +1    : the landing pad for a match
    //     B_i  the argument tests, then the rule's action
    //   [last] ret FALLBACK                      in range, matched nothing
    //
    // The pad exists because a rule with no argument tests has a one
    // instruction block, and without it the entry's two branches would need
    // offsets that collide: a jump of 1 into the block and a jump of 1 past
    // it are the same number. The pad makes the taken branch a constant 1 and
    // the not-taken branch the block size plus one, and those are never
    // equal.
    //
    // An argument test that fails leaves rule i and resumes at the next
    // rule's entry, so a number that a rule mentions but whose arguments do
    // not match is still resolved by the following rules and finally by the
    // fallback. The distance from a test to the next rule's entry does not
    // depend on the size of any other rule, which is what keeps every offset
    // inside a block a constant.
    //
    // A test occupies five instructions:
    //
    //   0  ld arg_hi
    //   1  jeq 0    : a zero high half continues at 3
    //   2  ja +2    : a non-zero high half abandons the rule
    //   3  ld arg_lo
    //   4  cmp <value> : a match continues at 5; a miss abandons the rule
    //
    // Instruction 1 jumps by 1, which skips instruction 2. When the test is
    // not the last of its rule, instruction 2 jumps by 2 and instruction 4
    // jumps by 1, both landing on the next test's instruction 0. When it is
    // the last, both land past the rule's action instead: instruction 2 by
    // 3, instruction 4 by 1.

    std::vector<Insn> insns;
    insns.reserve(8 + rule_count * 8);

    insns.push_back(ld_arch());
    insns.push_back(branch(BPF_JMP | BPF_JEQ | BPF_K, 0xc000003eu, 1, 0));
    insns.push_back(stmt(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS));
    insns.push_back(ld_nr());
    insns.push_back(branch(BPF_JMP | BPF_JGT | BPF_K, max_nr, 0, 1));
    insns.push_back(stmt(BPF_RET | BPF_K,
                         ret_for(policy.fallback, policy.fallback_error)));

    for (std::size_t i = 0; i < rule_count; ++i) {
        const auto& rule = policy.rules[i];
        const std::size_t args = rule.args.size();
        const std::size_t block = block_size(args);
        const std::size_t skip = block + 1; // the pad and the block

        if (skip > kMaxOffset) {
            out.error_ = "seccomp rule block exceeds the jump range";
            return out;
        }

        // The entry: a match takes the pad, a miss skips pad and block.
        insns.push_back(branch(BPF_JMP | BPF_JEQ | BPF_K, rule.nr, 1,
                               static_cast<std::uint8_t>(skip)));
        // The pad.
        insns.push_back(branch(BPF_JMP | BPF_JA, 1, 0, 0));

        for (std::size_t t = 0; t < args; ++t) {
            const auto& test = rule.args[t];
            const bool last = (t + 1 == args);

            insns.push_back(ld_arg_hi(test.index));
            insns.push_back(branch(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, 0));
            insns.push_back(branch(BPF_JMP | BPF_JA, last ? 3u : 2u, 0, 0));
            insns.push_back(ld_arg_lo(test.index));

            std::uint16_t op = BPF_JEQ;
            switch (test.cmp) {
            case SeccompCmp::Equal:
                op = BPF_JEQ;
                break;
            case SeccompCmp::NotEqual:
                op = kJne;
                break;
            case SeccompCmp::Greater:
                op = BPF_JGT;
                break;
            case SeccompCmp::GreaterOrEqual:
                op = BPF_JGE;
                break;
            case SeccompCmp::Less:
                op = kJlt;
                break;
            case SeccompCmp::LessOrEqual:
                op = kJle;
                break;
            case SeccompCmp::Masked:
                op = BPF_JSET;
                break;
            }
            // A match continues to the next instruction; a miss abandons the
            // rule. When the test is last, the next instruction is the
            // action, so abandoning means stepping past it. When it is not,
            // the next instruction is the following test, which is where a
            // miss belongs.
            insns.push_back(branch(BPF_JMP | op | BPF_K, test.value, 0, 1));
        }

        insns.push_back(stmt(BPF_RET | BPF_K, ret_for(rule.action, rule.error)));
    }

    insns.push_back(stmt(BPF_RET | BPF_K,
                         ret_for(policy.fallback, policy.fallback_error)));

    out.insns_ = std::move(insns);
    out.valid_ = true;
    out.error_ = "";
    return out;
}

int seccomp_install(const SeccompProgram& program) noexcept {
    if (!program.valid() || program.insn_count() == 0) {
        return -kEinval;
    }
    if (program.insn_count() > 0xfffdu) {
        return -kEinval;
    }

    struct sock_fprog prog {};
    prog.len = static_cast<unsigned short>(program.insn_count());
    prog.filter = const_cast<struct sock_filter*>(program.code());

    // PR_SET_NO_NEW_PRIVS is the caller's decision, not this function's: it
    // stops a setuid binary from gaining privilege across an exec, which
    // matters for a target run under isolation and does not matter for a
    // filter installed on a process that is already unprivileged.
    auto r = prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER,
                   reinterpret_cast<unsigned long>(&prog), 0, 0);
    if (r.failed()) {
        return -r.error;
    }
    return 0;
}

int seccomp_install_on_thread(const SeccompProgram& program,
                              int tid) noexcept {
    (void)program;
    (void)tid;

    // seccomp(2) installs on the calling thread only. The one supported way
    // for a tracer to give a traced thread a filter is for that thread to
    // make the call itself, which the isolation layer arranges by compiling
    // in the parent and having the child install before it drops privileges.
    // Reporting the limitation is better than letting a caller believe a
    // filter is active because a call returned success.
    return -kEopnotsupp;
}

} // namespace occ::sys
