#include "occ/util/string.h"

#include <array>
#include <cstring>
#include <vector>

namespace occ {

bool starts_with(std::string_view s, std::string_view prefix) noexcept {
    return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

bool ends_with(std::string_view s, std::string_view suffix) noexcept {
    return s.size() >= suffix.size() &&
           s.substr(s.size() - suffix.size()) == suffix;
}

std::string_view trim(std::string_view s) noexcept {
    constexpr std::string_view ws = " \t\r\n\v\f";
    const std::size_t first = s.find_first_not_of(ws);
    if (first == std::string_view::npos) {
        return {};
    }
    const std::size_t last = s.find_last_not_of(ws);
    return s.substr(first, last - first + 1);
}

std::vector<std::string_view> split(std::string_view s, char delim) noexcept {
    std::vector<std::string_view> out;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == delim) {
            out.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

void append_uint(std::string& out, std::uint64_t value) noexcept {
    if (value == 0) {
        out.push_back('0');
        return;
    }
    char buf[20];
    std::size_t n = 0;
    while (value != 0) {
        buf[n++] = static_cast<char>('0' + (value % 10));
        value /= 10;
    }
    while (n != 0) {
        out.push_back(buf[--n]);
    }
}

void append_int(std::string& out, std::int64_t value) noexcept {
    if (value < 0) {
        out.push_back('-');
        // Negating INT64_MIN overflows. Doing the conversion on the unsigned
        // side first is the only portable way to print it.
        const auto mag = static_cast<std::uint64_t>(0) -
                         static_cast<std::uint64_t>(value);
        append_uint(out, mag);
        return;
    }
    append_uint(out, static_cast<std::uint64_t>(value));
}

void append_hex(std::string& out, std::uint64_t value, int width) noexcept {
    constexpr char digits[] = "0123456789abcdef";
    int shift = (width - 1) * 4;
    bool started = false;
    for (int i = 0; i < width; ++i) {
        const auto nibble =
            static_cast<std::uint64_t>((value >> (shift - i * 4)) & 0xf);
        if (nibble != 0 || started || i == width - 1) {
            started = true;
            out.push_back(digits[nibble]);
        }
    }
}

std::string to_hex(std::uint64_t value, int width) {
    std::string out;
    append_hex(out, value, width);
    return out;
}

void append_json_escaped(std::string& out, std::string_view s) noexcept {
    constexpr char hexd[] = "0123456789abcdef";
    for (const char raw : s) {
        const auto c = static_cast<unsigned char>(raw);
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20) {
                out += "\\u00";
                out.push_back(hexd[(c >> 4) & 0xf]);
                out.push_back(hexd[c & 0xf]);
            } else {
                out.push_back(raw);
            }
            break;
        }
    }
}

bool read_u16le(ByteSpan s, std::size_t off, std::uint16_t& out) noexcept {
    if (off + 2 > s.size()) {
        return false;
    }
    out = u16le_at(s, off);
    return true;
}

bool read_u32le(ByteSpan s, std::size_t off, std::uint32_t& out) noexcept {
    if (off + 4 > s.size()) {
        return false;
    }
    out = u32le_at(s, off);
    return true;
}

bool read_u64le(ByteSpan s, std::size_t off, std::uint64_t& out) noexcept {
    if (off + 8 > s.size()) {
        return false;
    }
    out = static_cast<std::uint64_t>(u32le_at(s, off)) |
          (static_cast<std::uint64_t>(u32le_at(s, off + 4)) << 32);
    return true;
}

bool read_u16be(ByteSpan s, std::size_t off, std::uint16_t& out) noexcept {
    if (off + 2 > s.size()) {
        return false;
    }
    out = static_cast<std::uint16_t>(static_cast<std::uint16_t>(s[off]) << 8) |
          static_cast<std::uint16_t>(s[off + 1]);
    return true;
}

bool read_u32be(ByteSpan s, std::size_t off, std::uint32_t& out) noexcept {
    if (off + 4 > s.size()) {
        return false;
    }
    out = u32be_at(s, off);
    return true;
}

bool read_u64be(ByteSpan s, std::size_t off, std::uint64_t& out) noexcept {
    if (off + 8 > s.size()) {
        return false;
    }
    out = (static_cast<std::uint64_t>(u32be_at(s, off)) << 32) |
          static_cast<std::uint64_t>(u32be_at(s, off + 4));
    return true;
}

} // namespace occ
