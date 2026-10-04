#pragma once

// Hardware breakpoints.
//
// A software breakpoint changes the program's own bytes, so it can only stop
// execution at an instruction. A hardware breakpoint uses a debug register,
// which means it can stop on a memory read or write at an address, and it
// leaves the program's bytes untouched, so it cannot be discovered by a
// target that checksums its own code.
//
// The kernel exposes the four debug registers through perf_event_open. Using
// them through perf rather than through ptrace's debug-register requests is
// what makes this work on a process that is not being traced at all, which
// is the case for a target whose syscalls are filtered rather than traced.
//
// There are exactly four of these, in hardware, and no more can be created.
// The kernel reports the count in the debug-register capability, and a fifth
// request fails with ENOSPC rather than being emulated. That limit is the
// reason this layer reports how many are in use rather than pretending the
// number is unbounded.

#include <cstdint>
#include <string>
#include <vector>

#include "occ/syscall/syscall.h"

namespace occ::obs {

// What kind of access to watch for. The encoding is the kernel's
// hw_breakpoint_type.
enum class WatchKind : std::uint8_t {
    // Execution. Stops before the instruction at the address runs, which
    // makes it the hardware equivalent of a software breakpoint and the one
    // to use on a page that cannot be written.
    Execute = 0,
    // A write to the address. This is the W in a W-to-X tracker: it fires
    // while the target is writing, before the new value is committed in a
    // way the target can rely on.
    Write = 1,
    // A read or a write. The kernel has no read-only encoding, so a
    // read watch is a read-or-write watch and the handler has to look at
    // the instruction to tell which happened.
    ReadWrite = 2,
    // Not a hardware type. Present so that a caller can ask for a watch
    // that is only satisfied by a read, and be told it is unavailable
    // rather than silently given the read-or-write one.
    Read = 3,
};

// How much memory one watch covers. A watch is at most eight bytes and has
// to be naturally aligned, so a range is covered by several watches rather
// than by one wide one.
enum class WatchSize : std::uint8_t { Bytes1 = 1, Bytes2 = 2, Bytes4 = 4, Bytes8 = 8 };

struct HardwareWatch {
    int fd = -1;
    int pid = 0;
    std::uint64_t address = 0;
    WatchKind kind = WatchKind::Write;
    WatchSize size = WatchSize::Bytes8;
};

class Watchpoints {
public:
    Watchpoints() = default;
    Watchpoints(const Watchpoints&) = delete;
    Watchpoints& operator=(const Watchpoints&) = delete;
    ~Watchpoints();

    // Registers a watch on `address` in `pid`.
    //
    // The events are delivered on the returned descriptor, which the caller
    // polls. A synchronous SIGTRAP delivery is not used: a signal would
    // arrive in the middle of whatever the target was doing, and the
    // observer would then be reading its state from a handler rather than
    // from a loop that owns it.
    [[nodiscard]] int add(int pid, std::uint64_t address, WatchKind kind,
                          WatchSize size, std::string& detail) noexcept;

    // Removes one watch by address. Returns 0 even when nothing matched,
    // because a caller removing a watch it is not sure it installed should
    // not have to check first.
    int remove(std::uint64_t address) noexcept;

    int remove_all_for(int pid) noexcept;

    [[nodiscard]] const std::vector<HardwareWatch>& all() const noexcept {
        return watches_;
    }
    [[nodiscard]] std::size_t size() const noexcept { return watches_.size(); }

    // How many hardware debug registers exist on this machine. Zero when the
    // host reports none, which is the honest answer for a kernel built
    // without the capability.
    [[nodiscard]] static std::uint32_t hardware_slots() noexcept;

private:
    std::vector<HardwareWatch> watches_;
};

// Reads the instruction at `rip` in a stopped process and reports which
// general-purpose register it uses as a memory operand's base, and how wide
// the access is.
//
// This exists because the kernel's ReadWrite watch cannot distinguish a read
// from a write, and the only way to tell is to decode the faulting
// instruction. Only the subset the tracker needs is decoded: a mov whose
// destination is a register is a read, and a mov whose source is a register
// and whose destination is memory is a write. Anything else is reported as
// undetermined rather than guessed at.
struct AccessDecode {
    bool valid = false;
    bool is_write = false;
    // The register the access is based on, or -1 when the operand is not a
    // simple base-register access.
    int base_register = -1;
    // The byte width of the access, or 0 when it was not decoded.
    std::uint32_t width = 0;
    // The displacement added to the base register.
    std::int64_t displacement = 0;
};

[[nodiscard]] AccessDecode decode_access(const std::uint8_t* code,
                                         std::size_t length) noexcept;

} // namespace occ::obs
