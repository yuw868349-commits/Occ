#pragma once

// Kernel ABI constants.
//
// Where the toolchain's <linux/*.h> headers already define a constant or a
// structure, this project uses that definition directly instead of copying
// it. Hand-copied ABI constants are a class of bug that cannot be caught by
// testing on one machine: they are correct everywhere the author looked and
// wrong on the kernel that the user actually runs.
//
// What is declared here is the small set of things the toolchain headers
// either do not carry at all, or carry behind a feature test that this
// project does not want to depend on.

#include <cstddef>
#include <cstdint>

#include <linux/bpf.h>
#include <linux/filter.h>
#include <linux/perf_event.h>
#include <linux/seccomp.h>

namespace occ::sys {

// ---------------------------------------------------------------- clone flags
//
// <sched.h> defines CLONE_NEW* for the ones that existed when the header was
// written. CLONE_NEWCGROUP arrived in 4.6 and is missing from some libc
// versions, so it is restated. The rest come from <sched.h> at the use site.

inline constexpr std::uint64_t kCloneNewcgroup = 0x02000000ULL;

struct CloneArgs {
    std::uint64_t flags;
    std::uint64_t pidfd;
    std::uint64_t child_tid;
    std::uint64_t parent_tid;
    std::uint64_t exit_signal;
    std::uint64_t stack;
    std::uint64_t stack_size;
    std::uint64_t tls;
    std::uint64_t set_tid;
    std::uint64_t set_tid_size;
    std::uint64_t cgroup;
};

// clone3 rejects an args block whose size is smaller than the kernel's
// struct. The kernel only reads bytes up to the size passed, so passing the
// full size with zeroed tail fields is the portable form.
static_assert(sizeof(CloneArgs) == 88,
              "clone_args layout does not match the 6.x kernel ABI");

// ---------------------------------------------------------------- mount flags
//
// Taken from <sys/mount.h> where available. The propagation flags are the
// ones this project depends on and are restated because they are easy to
// mistype and impossible to notice when wrong.

inline constexpr unsigned long kMsBind      = 4096UL;
inline constexpr unsigned long kMsRecursive = 16384UL;
inline constexpr unsigned long kMsPrivate   = 1UL << 18;
inline constexpr unsigned long kMsSlave     = 1UL << 19;
inline constexpr unsigned long kMsShared    = 1UL << 20;
inline constexpr unsigned long kMsRelatime  = 1UL << 21;
inline constexpr unsigned long kMsNosuid    = 2UL;
inline constexpr unsigned long kMsNodev     = 4UL;
inline constexpr unsigned long kMsNoexec    = 8UL;

inline constexpr int kMntDetach = 2;

// -------------------------------------------------------------------- seccomp
//
// SECCOMP_RET_* and the filter flag bits come from <linux/seccomp.h>. The
// access-width bits in the low half of a return value are the exception:
// they are undocumented in the header and are what makes a 64-bit argument
// comparison safe against the x32 truncation trick.

inline constexpr std::uint32_t kSeccompRetDataSizeMask = 0x0000ffffU;
inline constexpr std::uint32_t kSeccompRetDataSize16 = 1U << 0;
inline constexpr std::uint32_t kSeccompRetDataSize32 = 2U << 0;

// Offsets into struct seccomp_data as the kernel lays it out. These come
// from <linux/seccomp.h> as a struct definition; the offsets are restated
// because the filter program addresses them numerically.
inline constexpr std::uint32_t kSeccompDataNrOffset = 0;
inline constexpr std::uint32_t kSeccompDataArchOffset = 4;
inline constexpr std::uint32_t kSeccompDataArgsOffset = 16;

inline constexpr std::uint32_t kAudArchX8664 = 0xc000003eU;

// SECCOMP_MODE_FILTER is 2; SECCOMP_MODE_DISABLED is 0 and
// SECCOMP_MODE_STRICT is 1. <linux/seccomp.h> defines all three.
//
// PR_SET_SECCOMP and PR_SET_NO_NEW_PRIVS come from <linux/prctl.h>, which is
// included at the use site. They are restated nowhere: a wrong prctl option
// number is not diagnosable from the error.

// -------------------------------------------------------------------- ptrace

inline constexpr unsigned long kPtraceTracerPid = 0x4200;
inline constexpr unsigned long kPtracePeekData = 1;
inline constexpr unsigned long kPtracePokeData = 2;
inline constexpr unsigned long kPtraceCont = 7;
inline constexpr unsigned long kPtraceKill = 8;
inline constexpr unsigned long kPtraceSinglestep = 9;
inline constexpr unsigned long kPtraceGetRegs = 12;
inline constexpr unsigned long kPtraceSetRegs = 13;
inline constexpr unsigned long kPtraceAttach = 16;
inline constexpr unsigned long kPtraceDetach = 17;
inline constexpr unsigned long kPtraceSetOptions = 0x4200;
inline constexpr unsigned long kPtraceGetEventMsg = 0x4201;
inline constexpr unsigned long kPtraceGetSiginfo = 0x4202;
inline constexpr unsigned long kPtraceSetSiginfo = 0x4203;
inline constexpr unsigned long kPtraceGetRegset = 0x4204;
inline constexpr unsigned long kPtraceSetRegset = 0x4205;
inline constexpr unsigned long kPtraceSeize = 0x4206;
inline constexpr unsigned long kPtraceInterrupt = 0x4207;
inline constexpr unsigned long kPtraceListens = 0x4208;
inline constexpr unsigned long kPtraceGetSyscallInfo = 0x420e;

// TRACEME is 0. Naming it makes the call sites readable.
inline constexpr unsigned long kPtraceTraceme = 0;

inline constexpr std::uint64_t kPtraceOTraceSysgood = 0x00000001ULL;
inline constexpr std::uint64_t kPtraceOTraceExit = 0x00000002ULL;
inline constexpr std::uint64_t kPtraceOTraceFork = 0x00000004ULL;
inline constexpr std::uint64_t kPtraceOTraceVFork = 0x00000008ULL;
inline constexpr std::uint64_t kPtraceOTraceClone = 0x00000010ULL;
inline constexpr std::uint64_t kPtraceOTraceExec = 0x00000020ULL;
inline constexpr std::uint64_t kPtraceOTraceSeccomp = 0x00000040ULL;
inline constexpr std::uint64_t kPtraceOTraceExitkill = 0x00000080ULL;

// perf_event_attr comes from <linux/perf_event.h> in full. Only the size
// constant and the breakpoint type are named here, because the kernel
// rejects an attribute block whose size field does not match what it
// expects and the error it returns does not say which field was wrong.
inline constexpr std::uint32_t kPerfAttrSize = sizeof(struct perf_event_attr);
inline constexpr std::uint32_t kPerfBreakpointHW = 1;

// bpf() attributes come from <linux/bpf.h>. The command numbers and map
// types are used directly from there.

} // namespace occ::sys
