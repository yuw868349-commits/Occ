#include "occ/observer/event.h"

#include "occ/syscall/errno.h"
#include "occ/syscall/syscall.h"
#include "occ/util/string.h"

namespace occ::obs {

namespace {

// Monotonic time. CLOCK_MONOTONIC is the only clock that is guaranteed not
// to step backwards, and a stream whose times go backwards cannot be sorted.
constexpr int kClockMonotonic = 1;

} // namespace

const char* event_kind_name(EventKind k) noexcept {
    switch (k) {
    case EventKind::SessionStart:
        return "session_start";
    case EventKind::SessionEnd:
        return "session_end";
    case EventKind::ProcessSpawn:
        return "process_spawn";
    case EventKind::ProcessExit:
        return "process_exit";
    case EventKind::ImageLoaded:
        return "image_loaded";
    case EventKind::Mapping:
        return "mapping";
    case EventKind::FileOpened:
        return "file_opened";
    case EventKind::SyscallBlocked:
        return "syscall_blocked";
    case EventKind::BreakpointHit:
        return "breakpoint_hit";
    case EventKind::Signal:
        return "signal";
    case EventKind::MemoryWrite:
        return "memory_write";
    case EventKind::Exec:
        return "exec";
    case EventKind::Note:
        return "note";
    }
    return "unknown";
}

void Event::separator() noexcept {
    if (!body_.empty()) {
        body_.push_back(',');
    }
}

void Event::add(std::string_view key, std::uint64_t value) noexcept {
    separator();
    body_.push_back('"');
    body_.append(key);
    body_.append("\":");
    append_uint(body_, value);
}

void Event::add(std::string_view key, std::int64_t value) noexcept {
    separator();
    body_.push_back('"');
    body_.append(key);
    body_.append("\":");
    append_int(body_, value);
}

void Event::add(std::string_view key, std::string_view value) noexcept {
    separator();
    body_.push_back('"');
    body_.append(key);
    body_.append("\":\"");
    append_json_escaped(body_, value);
    body_.push_back('"');
}

void Event::add(std::string_view key, bool value) noexcept {
    separator();
    body_.push_back('"');
    body_.append(key);
    body_.append("\":");
    body_.append(value ? "true" : "false");
}

void Event::add_hex(std::string_view key, std::uint64_t value) noexcept {
    separator();
    body_.push_back('"');
    body_.append(key);
    // The value is quoted, not a JSON number: JSON has no hexadecimal
    // literal, and an address written in decimal is not what a consumer
    // comparing against disassembly wants. The width is the number of
    // nibbles the writer may use, which is what a 64-bit address needs;
    // leading zeros are not padded, because an address is read as a value
    // and the padded form is harder to compare by eye.
    body_.append("\":\"0x");
    append_hex(body_, value, 16);
    body_.push_back('"');
}

bool Sink::write_line(std::string_view line) noexcept {
    if (fd_ < 0) {
        return false;
    }
    std::string out;
    out.reserve(line.size() + 1);
    out.append(line);
    out.push_back('\n');

    std::size_t off = 0;
    while (off < out.size()) {
        auto r = sys::write(fd_, out.data() + off, out.size() - off);
        if (r.failed()) {
            // A signal before any byte was transferred is worth retrying;
            // anything else means the far end is gone.
            if (r.error == sys::kEintr) {
                continue;
            }
            return false;
        }
        off += static_cast<std::size_t>(r.value);
    }
    return true;
}

std::uint64_t Writer::now_ns() const noexcept {
    // clock_gettime through the syscall layer, so that a run which has
    // replaced its own address space still has a working clock without
    // depending on the vdso having been set up for the new image.
    struct Timespec {
        long sec;
        long nsec;
    };
    Timespec ts{};
    auto r = sys::clock_gettime(kClockMonotonic, &ts);
    if (r.failed()) {
        return 0;
    }
    return static_cast<std::uint64_t>(ts.sec) * 1000000000ULL +
           static_cast<std::uint64_t>(ts.nsec);
}

Event& Writer::begin(EventKind kind) noexcept {
    current_ = Event{};
    // The kind name is handed over as a string_view. A bare const char*
    // would make the overload set prefer the bool one, and every event would
    // claim its kind was true.
    current_.add("kind", std::string_view{event_kind_name(kind)});
    current_.add("session", session_);
    current_.add("t", now_ns());
    open_ = true;
    return current_;
}

void Writer::commit() noexcept {
    if (!open_) {
        return;
    }
    open_ = false;

    std::string line;
    line.reserve(current_.body().size() + 2);
    line.push_back('{');
    line.append(current_.body());
    line.push_back('}');

    if (!sink_.write_line(line)) {
        failed_ = true;
        return;
    }
    ++written_;
}

void Writer::note(std::string_view text) noexcept {
    auto& e = begin(EventKind::Note);
    e.add("text", text);
    commit();
}

} // namespace occ::obs
