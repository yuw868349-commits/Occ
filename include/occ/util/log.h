#pragma once

// Logging.
//
// No timestamps. The output is meant to be read in a terminal next to the
// command that produced it, and a timestamp makes two runs of the same
// command impossible to diff. When a consumer needs ordering it takes it
// from the event stream, which carries monotonic times as data.

#include <cstddef>
#include <string_view>

namespace occ::log {

enum class Level {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
};

void set_level(Level level) noexcept;
[[nodiscard]] Level level() noexcept;

void error(std::string_view msg) noexcept;
void warn(std::string_view msg) noexcept;
void info(std::string_view msg) noexcept;
void debug(std::string_view msg) noexcept;

// Writes directly to fd 2 and does not go through the locale or stdio.
// Safe to call from a signal handler and from a process that has already
// closed stdio.
void raw(std::string_view msg) noexcept;

// Same, but appends the errno description for err.
void raw_errno(std::string_view msg, int err) noexcept;

} // namespace occ::log
