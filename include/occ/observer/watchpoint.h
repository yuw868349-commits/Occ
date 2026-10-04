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

    // The perf ring buffer this watch delivers through, and how far the
    // consumer has read into it. The buffer is mapped once per watch rather
    // than shared, because each watch is its own perf event with its own
    // stream, and a shared ring would make the mapping between a fired
    // address and a watch a matter of bookkeeping rather than of identity.
    void* ring = nullptr;
    std::size_t ring_bytes = 0;
    // Where the record area starts, which is one page past the metadata.
    // It is stored rather than assumed because the page size is a runtime
    // fact and a hard-coded 4096 would silently misread the ring on a
    // machine with a different one.
    std::size_t ring_data_offset = 0;
    std::size_t ring_tail = 0;
};

// One delivered watch event, decoded out of a ring buffer record.
struct WatchEvent {
    int pid = 0;
    // The instruction that performed the access.
    std::uint64_t rip = 0;
    // The data address that matched the watch, when the kernel supplied one.
    // An execute watch has no data address and reports zero.
    std::uint64_t address = 0;
    WatchKind kind = WatchKind::Write;
    // True when the record carried a usable data address. The distinction
    // matters because zero is a legitimate address on some mappings, and
    // treating "no address" as "address zero" would attribute an event to
    // the wrong page.
    bool has_address = false;
};

// How many ring buffer bytes one watch allocates. It has to be a power of two
// number of pages plus one control page, because the kernel's ring buffer
// arithmetic is a mask rather than a division.
inline constexpr std::size_t kWatchRingDataPages = 2;
inline constexpr std::size_t kWatchRingPages = kWatchRingDataPages + 1;

class Watchpoints {
public:
    Watchpoints() = default;
    Watchpoints(const Watchpoints&) = delete;
    Watchpoints& operator=(const Watchpoints&) = delete;
    ~Watchpoints();

    // Registers a watch on `address` in `pid`.
    //
    // The events are delivered through a mapped ring buffer, which the
    // caller drains with read_events(). A synchronous SIGTRAP delivery is
    // not used: a signal would arrive in the middle of whatever the target
    // was doing, and the observer would then be reading its state from a
    // handler rather than from a loop that owns it.
    //
    // On success the watch is enabled. A watch left disabled would accept
    // the event and never report it, which is the one failure mode that
    // cannot be detected from the outside.
    [[nodiscard]] int add(int pid, std::uint64_t address, WatchKind kind,
                          WatchSize size, std::string& detail) noexcept;

    // Removes one watch by address. Returns 0 even when nothing matched,
    // because a caller removing a watch it is not sure it installed should
    // not have to check first.
    int remove(std::uint64_t address) noexcept;

    int remove_all_for(int pid) noexcept;

    // Every watch descriptor, for poll(). The descriptor is the perf event
    // fd itself: it becomes readable when the ring buffer has records the
    // consumer has not taken. Polling the fd is what lets a single loop
    // watch the target's stops, the debugger's packets, and the watch
    // events without any of them being able to starve the others.
    [[nodiscard]] std::vector<int> fds() const noexcept;

    // Takes every record currently in every ring buffer and appends the
    // decoded ones to `out`. Returns the number of events produced.
    //
    // Records that cannot be decoded are skipped rather than guessed at: a
    // ring buffer that wrapped or was written by a kernel with a different
    // record layout is reported through the return value being lower than
    // the record count, and the caller re-syncs its tail.
    std::size_t read_events(std::vector<WatchEvent>& out) noexcept;

    [[nodiscard]] const std::vector<HardwareWatch>& all() const noexcept {
        return watches_;
    }
    [[nodiscard]] std::size_t size() const noexcept { return watches_.size(); }

    // How many samples the kernel reported as lost across every watch. A
    // lost sample is a real access the observer never saw, and in a tracker
    // that means a write it cannot account for. The count is kept rather
    // than discarded because "the ring wrapped" and "no writes happened"
    // are indistinguishable in the output otherwise, and only one of them
    // means the target is clean.
    [[nodiscard]] std::uint64_t lost_samples() const noexcept {
        return lost_samples_;
    }
    void clear_lost_samples() noexcept { lost_samples_ = 0; }

    // How many hardware debug registers exist on this machine. Zero when the
    // host reports none, which is the honest answer for a kernel built
    // without the capability.
    //
    // This counts the registers the machine has, not the ones this object
    // still has free. The distinction matters and the naming cannot carry
    // it, so it is stated here: the count is obtained by creating watches on
    // the calling process until one is refused, so a caller that already
    // holds every register gets zero back. A caller budgeting an allocation
    // needs the machine's capacity and must subtract what it already holds.
    [[nodiscard]] static std::uint32_t machine_slots() noexcept;

    // Runs the probe for machine_slots() without consulting the cache.
    //
    // Exposed because the cached answer and the live answer disagree exactly
    // when the caller is holding watches, and a test that can only see the
    // cached value cannot tell a machine with four registers from one with
    // four free registers. This is the raw question; machine_slots() is the
    // remembered answer.
    [[nodiscard]] static std::uint32_t probe_slots() noexcept;

    // The cached machine capacity.
    [[nodiscard]] static std::uint32_t hardware_slots() noexcept {
        return machine_slots();
    }

    // How many more watches this object could install right now: the machine
    // capacity minus the watches already held.
    //
    // This is the number an installer wants. Using hardware_slots() for it
    // fails in the one case that matters -- reallocating onto a new region
    // while the old watches are still installed reports zero free, and the
    // reallocation silently installs nothing.
    [[nodiscard]] std::uint32_t free_slots() const noexcept;

    // Whether this machine can create hardware watches at all. Separate
    // from hardware_slots() because a machine can report slots and still
    // refuse the event: a seccomp filter that blocks perf_event_open makes
    // every watch impossible while the count stays at four. A caller that
    // only checked the count would arm nothing and believe it had.
    [[nodiscard]] static bool hardware_available() noexcept;

private:
    // Frees the ring buffer and the descriptor of one watch. Shared by the
    // destructor and by remove, because leaking a mapping per watch would
    // outlive the target it was watching.
    static void release(HardwareWatch& w) noexcept;

    std::vector<HardwareWatch> watches_;
    std::uint64_t lost_samples_ = 0;
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
