#pragma once

// Small string and byte helpers used across the codebase.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "occ/util/span.h"

namespace occ {

[[nodiscard]] bool starts_with(std::string_view s, std::string_view prefix) noexcept;

[[nodiscard]] bool ends_with(std::string_view s, std::string_view suffix) noexcept;

// Trims ASCII whitespace from both ends.
[[nodiscard]] std::string_view trim(std::string_view s) noexcept;

// Splits on runs of the delimiter. Empty fields are kept, because a config
// line with two adjacent separators usually means something.
[[nodiscard]] std::vector<std::string_view> split(std::string_view s,
                                                  char delim) noexcept;

// Appends the decimal representation. Faster and locale-free compared to
// going through a stream, which matters in the event encoder.
void append_uint(std::string& out, std::uint64_t value) noexcept;
void append_int(std::string& out, std::int64_t value) noexcept;
void append_hex(std::string& out, std::uint64_t value, int width) noexcept;

// Escapes a byte string for JSON. Control characters below 0x20 become
// \u00XX, quote and backslash are escaped, and bytes at or above 0x80 are
// passed through as-is. The stream is declared UTF-8; invalid sequences are
// the producer's problem to not emit.
void append_json_escaped(std::string& out, std::string_view s) noexcept;

[[nodiscard]] std::string to_hex(std::uint64_t value, int width);

// Reads a little-endian integer from a byte span at an offset. Returns false
// when the span is too short, rather than reading past the end. Every caller
// in the parsers uses the return value.
[[nodiscard]] bool read_u16le(ByteSpan s, std::size_t off, std::uint16_t& out) noexcept;
[[nodiscard]] bool read_u32le(ByteSpan s, std::size_t off, std::uint32_t& out) noexcept;
[[nodiscard]] bool read_u64le(ByteSpan s, std::size_t off, std::uint64_t& out) noexcept;
[[nodiscard]] bool read_u16be(ByteSpan s, std::size_t off, std::uint16_t& out) noexcept;
[[nodiscard]] bool read_u32be(ByteSpan s, std::size_t off, std::uint32_t& out) noexcept;
[[nodiscard]] bool read_u64be(ByteSpan s, std::size_t off, std::uint64_t& out) noexcept;

// Same, but the caller has already established that the range is in bounds.
// Used inside a parser after one bounds check covers a whole record.
[[nodiscard]] constexpr std::uint32_t u32le_at(ByteSpan s, std::size_t off) noexcept {
    return static_cast<std::uint32_t>(s[off]) |
           (static_cast<std::uint32_t>(s[off + 1]) << 8) |
           (static_cast<std::uint32_t>(s[off + 2]) << 16) |
           (static_cast<std::uint32_t>(s[off + 3]) << 24);
}

[[nodiscard]] constexpr std::uint32_t u32be_at(ByteSpan s, std::size_t off) noexcept {
    return (static_cast<std::uint32_t>(s[off]) << 24) |
           (static_cast<std::uint32_t>(s[off + 1]) << 16) |
           (static_cast<std::uint32_t>(s[off + 2]) << 8) |
           static_cast<std::uint32_t>(s[off + 3]);
}

[[nodiscard]] constexpr std::uint16_t u16le_at(ByteSpan s, std::size_t off) noexcept {
    return static_cast<std::uint16_t>(s[off]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(s[off + 1]) << 8);
}

} // namespace occ
