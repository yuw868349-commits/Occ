#pragma once

// The write-then-execute tracker.
//
// A target that decodes its own code has to write the decoded bytes
// somewhere and then run them, and the run is what the tracker watches for.
// The two halves are observed differently:
//
//   - The write is seen with a hardware watch, which is why the watch has to
//     be on a page whose writability says something. A watch on a page the
//     target maps without execute permission is the interesting case,
//     because the write is then a write to memory that is not yet code.
//   - The execute is seen with a watch of its own on the entry point, or by
//     sampling the permission of the region after the write.
//
// What is reported is a transition, not a single event: a region that was
// written and then became executable. The transitions are deduplicated,
// because a decoder that writes one byte at a time produces a write per
// byte and a report per byte is a report nobody can read.
//
// The tracker does not decide whether a transition is hostile. It reports
// the address, the size, and the fact. A caller that has a policy applies
// it; a tracker that applied one would be a tracker with an opinion about
// what a legitimate program does, and self-modifying code is a legitimate
// technique as well as a hostile one.

#include <cstdint>
#include <string>
#include <vector>

#include "occ/observer/event.h"
#include "occ/observer/ptrace.h"
#include "occ/observer/watchpoint.h"

namespace occ::obs {

// A region the target wrote to and that later became executable.
struct Transition {
    std::uint64_t address = 0;
    // The number of bytes written across every write the tracker saw.
    std::uint64_t bytes_written = 0;
    // How many separate writes were folded into this transition. A decoder
    // that writes one byte at a time produces a large count over a small
    // region, which is what distinguishes a decoder from an ordinary store.
    std::uint64_t write_count = 0;
    // The permission the region had when it was written, and the one it had
    // when it was executed. The pair is the whole point: a region written
    // while already writable-and-executable is a different fact from one
    // written while not executable and made executable afterwards.
    bool was_writable = false;
    bool was_executable = false;
    bool became_executable = false;
};

// What the tracker is watching. A tracker with no regions watches nothing
// and reports nothing, which is the correct behaviour for a target that is
// not suspected of rewriting itself.
struct WatchTarget {
    // The page-aligned base of the region.
    std::uint64_t base = 0;
    std::uint64_t length = 0;
};

// A record of one region's permission, read from the target's own maps. The
// tracker keeps these rather than asking the kernel per write, because a
// decoder produces many writes to one region and re-reading the maps for
// each would turn a report into a syscall storm.
struct RegionPerms {
    bool readable = false;
    bool writable = false;
    bool executable = false;
};

class WriteExecuteTracker {
public:
    WriteExecuteTracker() = default;

    // Starts watching `base` for writes. The region is rounded out to whole
    // pages, because a permission is a page property and a region that
    // shared a page with something else could not have its permission
    // changed on its own.
    [[nodiscard]] int watch(int pid, std::uint64_t base, std::uint64_t length,
                            std::string& detail) noexcept;

    // Records that `bytes` were written at `address` by a process whose
    // faulting instruction was at `rip`. The access is attributed to the
    // region that contains `address`, not to the region that contains the
    // instruction, because the write is what matters.
    void note_write(int pid, std::uint64_t address, std::uint64_t bytes,
                    std::uint64_t rip) noexcept;

    // Records the permission a region has now, and reports a transition if
    // the region was written while not executable and is executable now.
    // Returns true when this call produced a transition that had not
    // already been reported.
    bool note_permission(int pid, std::uint64_t address,
                         const RegionPerms& perms) noexcept;

    // Reports every pending transition to the writer and clears them.
    void flush(int pid, obs::Writer& events) noexcept;

    [[nodiscard]] const std::vector<Transition>& transitions() const noexcept {
        return transitions_;
    }
    [[nodiscard]] std::size_t writes_seen() const noexcept { return writes_seen_; }

    // Installs the hardware watches for every region being tracked, using
    // the slots available. Regions beyond the hardware limit are reported as
    // unwatched rather than silently dropped, because a tracker that
    // covered some of its regions and said nothing would be worse than one
    // that covered none.
    [[nodiscard]] int arm(Watchpoints& watchpoints, int pid,
                          std::vector<WatchTarget>& unwatched) noexcept;

private:
    struct Region {
        WatchTarget target;
        RegionPerms perms;
        bool written = false;
        bool reported = false;
        std::uint64_t bytes = 0;
        std::uint64_t writes = 0;
        bool armed = false;
    };

    Region* find_region(std::uint64_t address) noexcept;

    std::vector<Region> regions_;
    std::vector<Transition> transitions_;
    std::size_t writes_seen_ = 0;
};

// Reads the permission of the page containing `address` from a stopped
// process's own /proc/<pid>/maps. Returns false when the address is not
// mapped, which is itself a fact: a write to an unmapped page is a fault,
// not a transition.
[[nodiscard]] bool read_region_perms(int pid, std::uint64_t address,
                                     RegionPerms& out) noexcept;

} // namespace occ::obs
