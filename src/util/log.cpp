#include "occ/util/log.h"
#include "occ/syscall/syscall.h"

#include <array>
#include <cstring>

namespace occ::log {

namespace {

Level g_level = Level::Info;

// A single write per message. Interleaving from multiple threads is a
// problem for the caller to solve; the logger does not take a lock and does
// not pretend to be atomic beyond the level of one write call.
void emit(std::string_view prefix, std::string_view msg) noexcept {
    std::array<char, 1024> buf{};
    std::size_t n = 0;

    const auto copy = [&](std::string_view s) {
        const std::size_t room = buf.size() - n - 2;
        const std::size_t take = s.size() < room ? s.size() : room;
        std::memcpy(buf.data() + n, s.data(), take);
        n += take;
    };

    copy(prefix);
    copy(msg);
    buf[n++] = '\n';

    // A short write on fd 2 is not retried. If the terminal is gone there is
    // no point insisting, and a retry loop inside the error path can hang.
    (void)sys::write(2, buf.data(), n);
}

} // namespace

void set_level(Level level) noexcept { g_level = level; }

Level level() noexcept { return g_level; }

void error(std::string_view msg) noexcept {
    if (g_level >= Level::Error) {
        emit("occ: error: ", msg);
    }
}

void warn(std::string_view msg) noexcept {
    if (g_level >= Level::Warn) {
        emit("occ: warning: ", msg);
    }
}

void info(std::string_view msg) noexcept {
    if (g_level >= Level::Info) {
        emit("occ: ", msg);
    }
}

void debug(std::string_view msg) noexcept {
    if (g_level >= Level::Debug) {
        emit("occ: debug: ", msg);
    }
}

void raw(std::string_view msg) noexcept {
    (void)sys::write(2, msg.data(), msg.size());
}

void raw_errno(std::string_view msg, int err) noexcept {
    char desc[64]{};
    const char* p = ::strerror_r(err, desc, sizeof(desc) - 1);

    (void)sys::write(2, msg.data(), msg.size());
    (void)sys::write(2, ": ", 2);
    (void)sys::write(2, p, std::strlen(p));
    (void)sys::write(2, "\n", 1);
}

} // namespace occ::log
