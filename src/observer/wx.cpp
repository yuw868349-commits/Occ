#include "occ/observer/wx.h"

#include "occ/syscall/errno.h"
#include "occ/util/fs.h"
#include "occ/util/string.h"

#include <cstring>

namespace occ::obs {

namespace {

constexpr std::uint64_t kPageSize = 4096;

std::uint64_t page_floor(std::uint64_t v) noexcept {
    return v & ~(kPageSize - 1);
}

std::uint64_t page_ceil(std::uint64_t v) noexcept {
    return (v + kPageSize - 1) & ~(kPageSize - 1);
}

// Parses one line of /proc/<pid>/maps into a range and its permissions.
// The format is "start-end perms offset dev inode pathname", and the
// permission field is always four characters, with '-' where a permission
// is absent. Parsing is done by hand rather than with a stream because this
// runs for every write the tracker sees.
bool parse_maps_line(std::string_view line, std::uint64_t& start,
                     std::uint64_t& end, RegionPerms& perms) noexcept {
    // Range.
    const std::size_t dash = line.find('-');
    if (dash == std::string_view::npos) {
        return false;
    }
    std::uint64_t s = 0;
    for (char c : line.substr(0, dash)) {
        if (c >= '0' && c <= '9') {
            s = s * 16 + static_cast<std::uint64_t>(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            s = s * 16 + static_cast<std::uint64_t>(c - 'a' + 10);
        } else {
            return false;
        }
    }

    const std::size_t space = line.find(' ', dash);
    if (space == std::string_view::npos) {
        return false;
    }
    std::uint64_t e = 0;
    for (char c : line.substr(dash + 1, space - dash - 1)) {
        if (c >= '0' && c <= '9') {
            e = e * 16 + static_cast<std::uint64_t>(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            e = e * 16 + static_cast<std::uint64_t>(c - 'a' + 10);
        } else {
            return false;
        }
    }

    const std::size_t perms_at = line.find_first_not_of(' ', space);
    if (perms_at == std::string_view::npos || perms_at + 4 > line.size()) {
        return false;
    }

    RegionPerms p;
    p.readable = line[perms_at] == 'r';
    p.writable = line[perms_at + 1] == 'w';
    p.executable = line[perms_at + 2] == 'x';

    start = s;
    end = e;
    perms = p;
    return true;
}

} // namespace

bool read_region_perms(int pid, std::uint64_t address,
                       RegionPerms& out) noexcept {
    const std::string path = "/proc/" + std::to_string(pid) + "/maps";
    auto content = fs::read_file(path);
    if (!content) {
        return false;
    }

    std::size_t pos = 0;
    const std::string& text = *content;
    while (pos < text.size()) {
        std::size_t end = text.find('\n', pos);
        if (end == std::string::npos) {
            end = text.size();
        }
        const std::string_view line(text.data() + pos, end - pos);

        std::uint64_t start = 0;
        std::uint64_t stop = 0;
        RegionPerms perms;
        if (parse_maps_line(line, start, stop, perms) && address >= start &&
            address < stop) {
            out = perms;
            return true;
        }

        pos = end + 1;
    }
    return false;
}

int WriteExecuteTracker::watch(int pid, std::uint64_t base,
                               std::uint64_t length,
                               std::string& detail) noexcept {
    (void)pid;
    if (length == 0) {
        detail = "the region is empty";
        return sys::kEinval;
    }

    Region r;
    r.target.base = page_floor(base);
    r.target.length = page_ceil(base + length) - r.target.base;

    for (auto& existing : regions_) {
        if (existing.target.base == r.target.base &&
            existing.target.length == r.target.length) {
            return 0;
        }
    }

    regions_.push_back(r);
    detail.clear();
    return 0;
}

WriteExecuteTracker::Region* WriteExecuteTracker::find_region(
    std::uint64_t address) noexcept {
    for (auto& r : regions_) {
        if (address >= r.target.base &&
            address < r.target.base + r.target.length) {
            return &r;
        }
    }
    return nullptr;
}

void WriteExecuteTracker::note_write(int pid, std::uint64_t address,
                                     std::uint64_t bytes,
                                     std::uint64_t rip) noexcept {
    (void)pid;
    (void)rip;
    ++writes_seen_;

    Region* r = find_region(address);
    if (r == nullptr) {
        // A write outside every watched region. The watch hardware fires on
        // an address range, so this happens when the faulting instruction
        // touched a neighbouring address in the same slot, and it is not a
        // transition.
        return;
    }

    r->written = true;
    r->bytes += bytes;
    ++r->writes;

    // A write clears a previous report: the region has been modified again
    // since it was last seen to execute, so whatever is there now has not
    // been accounted for.
    r->reported = false;
}

bool WriteExecuteTracker::note_permission(int pid, std::uint64_t address,
                                          const RegionPerms& perms) noexcept {
    Region* r = find_region(address);
    if (r == nullptr) {
        return false;
    }

    const RegionPerms before = r->perms;
    r->perms = perms;

    // The transition is "written while not executable, executable now". A
    // region that was already executable when it was written is not a
    // transition; it is a program writing to its own code, which is a
    // different observation and one the caller can make from the write
    // record alone.
    if (!r->written || r->reported || !perms.executable) {
        return false;
    }

    Transition t;
    t.address = r->target.base;
    t.bytes_written = r->bytes;
    t.write_count = r->writes;
    t.was_writable = before.writable;
    t.was_executable = before.executable;
    t.became_executable = true;

    transitions_.push_back(t);
    r->reported = true;

    (void)pid;
    return true;
}

void WriteExecuteTracker::flush(int pid, obs::Writer& events) noexcept {
    for (const auto& t : transitions_) {
        auto& e = events.begin(obs::EventKind::MemoryWrite);
        e.add("pid", static_cast<std::uint64_t>(pid));
        e.add_hex("address", t.address);
        e.add("bytes_written", t.bytes_written);
        e.add("write_count", t.write_count);
        e.add("was_writable", t.was_writable);
        e.add("was_executable", t.was_executable);
        e.add("became_executable", t.became_executable);
        events.commit();
    }
    transitions_.clear();
}

int WriteExecuteTracker::arm(Watchpoints& watchpoints, int pid,
                             std::vector<WatchTarget>& unwatched) noexcept {
    unwatched.clear();

    for (auto& r : regions_) {
        if (r.armed) {
            continue;
        }

        // Half of a region may be covered by the last watch when the region
        // is not a multiple of eight. The tail is covered by a narrower
        // watch, and a tail smaller than a watch is left uncovered and
        // reported, because an uncovered byte is a write the tracker would
        // otherwise miss.
        std::string detail;
        const int rc = watchpoints.add(pid, r.target.base, WatchKind::Write,
                                       WatchSize::Bytes8, detail);
        if (rc != 0) {
            WatchTarget t;
            t.base = r.target.base;
            t.length = r.target.length;
            unwatched.push_back(t);
            continue;
        }
        r.armed = true;
    }
    return 0;
}

} // namespace occ::obs
