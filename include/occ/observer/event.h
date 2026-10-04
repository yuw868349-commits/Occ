#pragma once

// The event stream.
//
// Every fact a run produces leaves through one writer, and the writer emits
// one JSON object per line. A line-oriented format is chosen because it is
// readable while the run is still going: a consumer tails the stream, parses
// one object at a time, and never has to hold the whole run in memory. The
// same bytes are what a terminal shows and what a tool consumes, so there is
// no second code path that only ever gets exercised by one of the two.
//
// Field order is fixed by the encoder rather than by a map, because a
// stable order makes two runs of the same target diff cleanly, and diffing
// two runs is the main thing a caller does with this.
//
// Times are monotonic nanoseconds since the stream was opened. A wall clock
// would make two runs on two machines incomparable, and this stream is
// meant to be compared.

#include <cstdint>
#include <string>
#include <string_view>

#include "occ/util/span.h"

namespace occ::obs {

// What an event is about. The kind is the first field on every line, so a
// consumer can dispatch on it before it has parsed the rest.
enum class EventKind : std::uint8_t {
    SessionStart,
    SessionEnd,
    ProcessSpawn,
    ProcessExit,
    ImageLoaded,
    Mapping,
    FileOpened,
    SyscallBlocked,
    BreakpointHit,
    Signal,
    MemoryWrite,
    Exec,
    Note,
};

[[nodiscard]] const char* event_kind_name(EventKind k) noexcept;

// A single event, held as its already-encoded JSON body. Building the body
// at the point the fact is observed keeps the encoder free of any state
// beyond the output sink, and keeps a struct per event kind from being
// needed for the thirty-odd kinds that end up existing.
class Event {
public:
    Event() = default;

    // Appends `"key":value` to the body. The value overloads are what keep
    // a caller from having to quote a string by hand, which is where an
    // unescaped quote would corrupt the stream.
    void add(std::string_view key, std::uint64_t value) noexcept;
    void add(std::string_view key, std::int64_t value) noexcept;
    void add(std::string_view key, std::string_view value) noexcept;
    void add(std::string_view key, bool value) noexcept;
    // A bare pointer would otherwise convert to bool, which is a conversion
    // the compiler is entitled to make and one that turns a name into a
    // truth value. Naming the overload makes the string win, so the hazard
    // is closed at the type rather than at each call site.
    void add(std::string_view key, const char* value) noexcept {
        add(key, std::string_view{value});
    }
    void add_hex(std::string_view key, std::uint64_t value) noexcept;

    [[nodiscard]] const std::string& body() const noexcept { return body_; }
    [[nodiscard]] bool empty() const noexcept { return body_.empty(); }

private:
    void separator() noexcept;

    std::string body_;
};

// Where the encoded lines go. An fd sink rather than a stream, because the
// stream has to survive the process replacement that the run performs and a
// FILE* buffer would be duplicated or lost across it.
class Sink {
public:
    Sink() = default;
    explicit Sink(int fd) noexcept : fd_(fd) {}

    // Writes one line. A partial write is retried; anything else is
    // reported by a false return. The sink never blocks indefinitely on a
    // dead reader: a broken pipe is a disconnection, not a reason to spin.
    [[nodiscard]] bool write_line(std::string_view line) noexcept;

    [[nodiscard]] int fd() const noexcept { return fd_; }

private:
    int fd_ = -1;
};

// The writer. One of these exists per run, and it owns the stream's identity
// so that a consumer can tell two interleaved runs apart.
class Writer {
public:
    Writer() = default;

    void attach(int fd) noexcept { sink_ = Sink{fd}; }

    // Emits an event with the fields every event carries: the kind, the
    // session it belongs to, and the time it was observed. Callers fill in
    // the rest through the returned reference.
    [[nodiscard]] Event& begin(EventKind kind) noexcept;

    // Encodes and writes the event returned by begin. Safe to call when an
    // event was never begun, in which case nothing is written.
    void commit() noexcept;

    // Convenience for the common case of an event with no extra fields
    // beyond a message.
    void note(std::string_view text) noexcept;

    [[nodiscard]] std::uint64_t events_written() const noexcept {
        return written_;
    }
    [[nodiscard]] bool failed() const noexcept { return failed_; }

private:
    [[nodiscard]] std::uint64_t now_ns() const noexcept;

    Sink sink_;
    Event current_;
    std::uint64_t session_ = 0;
    std::uint64_t written_ = 0;
    bool failed_ = false;
    bool open_ = false;
};

} // namespace occ::obs
