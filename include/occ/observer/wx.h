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

// What the arming achieved, reported rather than assumed. A caller that
// asked for a region to be watched has to be able to find out how much of it
// actually is, because "the tracker is on" and "the tracker covers this
// region" are different claims and only the second one is useful.
struct ArmReport {
    // How many hardware watches were installed.
    std::size_t watches_installed = 0;
    // How many bytes of the watched regions are covered by a watch.
    std::uint64_t bytes_covered = 0;
    // How many bytes were asked about.
    std::uint64_t bytes_total = 0;
    // Regions that could not be watched at all, because the debug registers
    // were already gone.
    std::vector<WatchTarget> unwatched;
    // Regions watched only in part, with the covered byte count of each.
    struct Partial {
        WatchTarget target;
        std::uint64_t bytes_covered;
    };
    std::vector<Partial> partial;

    [[nodiscard]] bool complete() const noexcept {
        return unwatched.empty() && partial.empty();
    }
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

    // Installs the hardware watches for every region being tracked, using the
    // slots available.
    //
    // A region is covered by as many watches as it takes: the hardware
    // watches at most eight naturally aligned bytes, so a 4096-byte page is
    // 512 watches and there are four of them. The tracker therefore covers
    // the largest prefix of each region that fits in the available slots,
    // widest watch first, and reports what it could not cover rather than
    // pretending the coverage is total.
    //
    // The strategy is deliberate. Covering the start of a region catches the
    // case that matters most -- a decoder that writes an instruction stream
    // from its beginning -- while spreading four watches thinly over four
    // regions would catch nothing at all in any of them. One region covered
    // deeply beats four regions covered partially, because a partial watch
    // that a target steps over is indistinguishable from no watch.
    //
    // The regions are served shortest first, which is the opposite of what a
    // reader would guess and is the whole reason this works. Four watches of
    // eight bytes cover 32 bytes; a program image has hundreds of kilobytes
    // of writable data, and the largest region would take every slot and
    // cover 32 bytes of a region the target treats as ordinary storage.
    // A short region, by contrast, can be covered *completely* within the
    // budget, and a completely covered region yields certain detection
    // rather than a sample. The slots are scarce enough that certainty is
    // worth more than reach.
    [[nodiscard]] ArmReport arm(Watchpoints& watchpoints, int pid) noexcept;

    // Reallocates the hardware slots onto a region that appeared while the
    // session was running, and returns what the reallocation achieved.
    //
    // This is what makes write-then-execute observable at all. A decoder
    // does not stage its output in a region that existed when the process
    // started: it maps a fresh anonymous page writable, fills it, and then
    // makes that page executable. A tracker that armed its watches once, at
    // startup, would be watching the loader's data segments while the one
    // region that became code went unobserved.
    //
    // Chasing is where the scarcity of debug registers has to be spent
    // deliberately. Watches on regions that have never been written to are
    // released to make room, because a watch that has not fired says only
    // that nothing happened yet, and a region the target has not touched is
    // the weakest evidence available. A region that *has* been written to is
    // kept: it is the only place a transition can still be detected.
    //
    // Returns an empty report's worth of counters when the region could not
    // be added at all. The caller does not treat that as a failure, because
    // a tracker that reports an honest miss is more useful than one that
    // claims a coverage it does not have.
    [[nodiscard]] ArmReport chase(Watchpoints& watchpoints, int pid,
                                 std::uint64_t base,
                                 std::uint64_t length) noexcept;

    // Releases the hardware watches held on regions that have never been
    // written to, until `need` watches have been freed or no cold region is
    // left. Called when a new mapping appears and the slots the startup
    // scan spent are needed elsewhere: a watch on a region the target has
    // not touched is the weakest evidence available, and it is worth less
    // than a watch on the region that just appeared.
    //
    // A region that has been written to is never evicted. The count returned
    // is watches released, not regions visited, because a region may have
    // held several.
    std::size_t evict_cold(Watchpoints& watchpoints, std::size_t need) noexcept;

    // The regions the tracker knows about, for a caller that has to report
    // them.
    [[nodiscard]] const std::vector<WatchTarget>& targets() const noexcept;

    [[nodiscard]] std::size_t region_count() const noexcept {
        return regions_.size();
    }

    // Removes every hardware watch this tracker installed. Called when the
    // target exits: a watch is a kernel resource attached to a process, and
    // leaving the descriptors open past the process's death would leak one
    // set of debug registers per run.
    void release(Watchpoints& watchpoints) noexcept;

private:
    struct Region {
        WatchTarget target;
        RegionPerms perms;
        bool written = false;
        bool reported = false;
        std::uint64_t bytes = 0;
        std::uint64_t writes = 0;
        bool armed = false;
        // How many bytes of this region the installed watches actually
        // cover. It is less than the region length on almost every region,
        // because the hardware has four watches and a page needs 512, and
        // it is kept so the tracker can say "covered 32 of 4096" instead of
        // implying the whole page is watched.
        std::uint64_t covered = 0;
        // The addresses the watches were placed at, in the order they were
        // installed. Kept so a region can be released, and so a caller can
        // see the coverage rather than infer it.
        std::vector<std::uint64_t> watch_addresses;
        // When the region was added, as a monotonically increasing counter.
        // Two regions with watches have to be ordered for eviction, and
        // "least recently watched" is only meaningful with a clock.
        std::uint64_t armed_at = 0;
    };

    Region* find_region(std::uint64_t address) noexcept;

    std::vector<Region> regions_;
    std::vector<Transition> transitions_;
    std::size_t writes_seen_ = 0;
    // The public view of the regions, rebuilt when the set changes rather
    // than on every query.
    mutable std::vector<WatchTarget> targets_cache_;
    // The next value of Region::armed_at.
    std::uint64_t arm_clock_ = 0;
};

// Reads the permission of the page containing `address` from a stopped
// process's own /proc/<pid>/maps. Returns false when the address is not
// mapped, which is itself a fact: a write to an unmapped page is a fault,
// not a transition.
[[nodiscard]] bool read_region_perms(int pid, std::uint64_t address,
                                     RegionPerms& out) noexcept;

// What kind of region a line of maps describes, and why the tracker treats
// the kinds differently.
enum class RegionClass : std::uint8_t {
    // Writable and not executable: the region a decoder writes into. This
    // is what the write watch is for.
    WritableData,
    // Writable and executable: a region that is already both. A write here
    // is not a W-to-X transition but it is still a self-modification, and
    // the caller is told which of the two it saw.
    WritableExecutable,
    // Executable and not writable: the ordinary code region. Watched for
    // writes only when the caller asked, because a write here would be
    // impossible without an mprotect first.
    ExecutableOnly,
    // Neither writable nor executable: not interesting to this tracker.
    Other,
};

struct RegionInfo {
    std::uint64_t base = 0;
    std::uint64_t length = 0;
    RegionPerms perms;
    RegionClass kind = RegionClass::Other;
    // True when the region has no pathname, which is what a decoder's
    // staging buffer looks like. An anonymous region is far more likely to
    // hold generated code than a file mapping is.
    bool anonymous = false;
};

// Every mapping in a process, classified. Reads /proc/<pid>/maps.
[[nodiscard]] std::vector<RegionInfo> process_regions(int pid) noexcept;

} // namespace occ::obs
