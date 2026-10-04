#include "occ/observer/rsp.h"

#include <limits>

namespace occ::obs {

namespace {

constexpr char kEscape = '}';
constexpr char kStart = '$';
constexpr char kEnd = '#';
constexpr char kAck = '+';
constexpr char kNoAck = '-';

int hex_digit(char c) noexcept {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return -1;
}

char hex_char(int nibble) noexcept {
    return static_cast<char>(nibble < 10 ? '0' + nibble : 'a' + (nibble - 10));
}

} // namespace

std::string unescape(std::string_view in) noexcept {
    std::string out;
    out.reserve(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        if (in[i] != kEscape) {
            out.push_back(in[i]);
            continue;
        }
        if (i + 1 >= in.size()) {
            // A trailing escape is a truncated packet. It is passed through
            // rather than dropped, so the caller sees a payload that is
            // wrong instead of one that is silently short.
            out.push_back(in[i]);
            break;
        }
        out.push_back(static_cast<char>(in[i + 1] ^ 0x20));
        ++i;
    }
    return out;
}

std::string escape(std::string_view in) noexcept {
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        // The three characters that would confuse the framing are escaped,
        // and so is the escape itself. Everything else passes through, which
        // matters because a payload is usually already text.
        if (c == kStart || c == kEnd || c == kEscape) {
            out.push_back(kEscape);
            out.push_back(static_cast<char>(c ^ 0x20));
        } else {
            out.push_back(c);
        }
    }
    return out;
}

std::string encode_packet(std::string_view payload) noexcept {
    const std::string escaped = escape(payload);

    unsigned int sum = 0;
    for (char c : escaped) {
        sum += static_cast<unsigned char>(c);
    }
    sum &= 0xff;

    std::string out;
    out.reserve(escaped.size() + 4);
    out.push_back(kStart);
    out.append(escaped);
    out.push_back(kEnd);
    out.push_back(hex_char(static_cast<int>((sum >> 4) & 0xf)));
    out.push_back(hex_char(static_cast<int>(sum & 0xf)));
    return out;
}

std::size_t PacketDecoder::feed(const char* data, std::size_t length) noexcept {
    const std::size_t before = ready_.size();

    for (std::size_t i = 0; i < length; ++i) {
        const char c = data[i];

        switch (state_) {
        case State::Idle:
            if (c == kStart) {
                payload_.clear();
                raw_.clear();
                raw_.push_back(c);
                state_ = State::Payload;
            } else if (c == kAck) {
                state_ = State::Ack;
            } else if (c == kNoAck) {
                // The far end rejected our last packet and wants it again.
                want_retransmit_ = true;
            }
            // Anything else outside a packet is noise, which the protocol
            // says to ignore rather than to treat as an error.
            break;

        case State::Ack:
            // A '+' followed by a '$' is an acknowledgment and then the
            // start of a packet. Anything else after a '+' is the old
            // three-byte acknowledgment form and is ignored.
            state_ = State::Idle;
            if (c == kStart) {
                payload_.clear();
                raw_.clear();
                raw_.push_back(c);
                state_ = State::Payload;
            }
            break;

        case State::Payload:
            if (c == kEnd) {
                state_ = State::FirstChecksum;
            } else if (c == kStart) {
                // A new start before an end means the previous packet was
                // cut short. Restarting is what the protocol expects of a
                // receiver that sees framing it cannot complete.
                payload_.clear();
                raw_.clear();
                raw_.push_back(c);
            } else {
                payload_.push_back(c);
                raw_.push_back(c);
            }
            break;

        case State::FirstChecksum: {
            const int d = hex_digit(c);
            if (d < 0) {
                // Not a checksum character. The packet is malformed; drop it
                // and resynchronize on the next start byte.
                state_ = State::Idle;
                break;
            }
            checksum_hi_ = d;
            state_ = State::SecondChecksum;
            break;
        }

        case State::SecondChecksum: {
            const int d = hex_digit(c);
            state_ = State::Idle;
            if (d < 0) {
                break;
            }

            unsigned int sum = 0;
            for (char ch : payload_) {
                sum += static_cast<unsigned char>(ch);
            }
            sum &= 0xff;
            const auto expected =
                static_cast<unsigned int>((checksum_hi_ << 4) | d);

            Packet p;
            if (sum == expected) {
                p.data = unescape(payload_);
                p.checksum_ok = true;
                last_ = raw_;
                last_.push_back(kEnd);
                last_.push_back(hex_char(checksum_hi_));
                last_.push_back(hex_char(d));
            } else {
                p.checksum_ok = false;
                want_retransmit_ = true;
            }
            ready_.push_back(std::move(p));
            payload_.clear();
            raw_.clear();
            break;
        }
        }
    }

    return ready_.size() - before;
}

Packet PacketDecoder::take() noexcept {
    if (ready_.empty()) {
        return Packet{};
    }
    Packet p = std::move(ready_.front());
    ready_.erase(ready_.begin());
    return p;
}

// ------------------------------------------------------------------ helpers

std::string hex_u64_le(std::uint64_t value) noexcept {
    std::string out;
    out.reserve(16);
    for (int i = 0; i < 8; ++i) {
        const auto byte = static_cast<unsigned int>((value >> (8 * i)) & 0xff);
        out.push_back(hex_char(static_cast<int>((byte >> 4) & 0xf)));
        out.push_back(hex_char(static_cast<int>(byte & 0xf)));
    }
    return out;
}

std::string hex_u64_be(std::uint64_t value) noexcept {
    std::string out;
    out.reserve(16);
    for (int i = 7; i >= 0; --i) {
        const auto byte = static_cast<unsigned int>((value >> (8 * i)) & 0xff);
        out.push_back(hex_char(static_cast<int>((byte >> 4) & 0xf)));
        out.push_back(hex_char(static_cast<int>(byte & 0xf)));
    }
    return out;
}

std::string hex_u32_le(std::uint32_t value) noexcept {
    std::string out;
    out.reserve(8);
    for (int i = 0; i < 4; ++i) {
        const auto byte = static_cast<unsigned int>((value >> (8 * i)) & 0xff);
        out.push_back(hex_char(static_cast<int>((byte >> 4) & 0xf)));
        out.push_back(hex_char(static_cast<int>(byte & 0xf)));
    }
    return out;
}

std::string hex_u8(std::uint8_t value) noexcept {
    std::string out;
    out.push_back(hex_char(static_cast<int>((value >> 4) & 0xf)));
    out.push_back(hex_char(static_cast<int>(value & 0xf)));
    return out;
}

std::string hex_number(std::uint64_t value) noexcept {
    // The counterpart of parse_hex_number: most significant digit first, and
    // no leading zeros, because that is the form the protocol's readers
    // accept and the form GDB sends. A thread id of 4242 is "1092", not the
    // eight digits a register value of the same number would occupy.
    //
    // Zero is the one value that cannot drop a leading digit, so it is
    // spelled out rather than producing an empty field.
    char digits[16];
    int n = 0;
    if (value == 0) {
        digits[n++] = '0';
    }
    while (value != 0) {
        digits[n++] = hex_char(static_cast<int>(value & 0xf));
        value >>= 4;
    }
    std::string out;
    out.reserve(static_cast<std::size_t>(n));
    for (int i = n - 1; i >= 0; --i) {
        out.push_back(digits[i]);
    }
    return out;
}

bool parse_hex_u64_le(std::string_view s, std::uint64_t& out) noexcept {
    if (s.size() < 16) {
        return false;
    }
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        const int hi = hex_digit(s[static_cast<std::size_t>(2 * i)]);
        const int lo = hex_digit(s[static_cast<std::size_t>(2 * i + 1)]);
        if (hi < 0 || lo < 0) {
            return false;
        }
        v |= static_cast<std::uint64_t>((hi << 4) | lo) << (8 * i);
    }
    out = v;
    return true;
}

bool parse_hex_u32_le(std::string_view s, std::uint32_t& out) noexcept {
    if (s.size() < 8) {
        return false;
    }
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        const int hi = hex_digit(s[static_cast<std::size_t>(2 * i)]);
        const int lo = hex_digit(s[static_cast<std::size_t>(2 * i + 1)]);
        if (hi < 0 || lo < 0) {
            return false;
        }
        v |= static_cast<std::uint32_t>((hi << 4) | lo) << (8 * i);
    }
    out = v;
    return true;
}

bool parse_hex_number(std::string_view s, std::uint64_t& out) noexcept {
    // A protocol number is hexadecimal and variable width, and it is not in
    // the target's byte order: an offset is an offset whichever way the
    // target stores integers, so the digits are read most significant first.
    // "0" is zero, "400" is one thousand and twenty eight, and a leading zero
    // is not significant, which is what lets the two-digit signal numbers the
    // protocol writes as "05" mean five rather than being read as a byte.
    //
    // This is deliberately a different function from parse_hex_u64_le. That
    // one reads a register block, where every field is exactly eight bytes
    // and the byte order is the target's; using it on a number whose width is
    // whatever the sender chose reads the digits in the wrong order and
    // refuses every value short of sixteen digits.
    if (s.empty() || s.size() > 16) {
        return false;
    }
    std::uint64_t v = 0;
    for (const char c : s) {
        const int d = hex_digit(c);
        if (d < 0) {
            return false;
        }
        // The width is capped above, so a full sixteen digits can only
        // overflow if the leading digit is above one, and that case is
        // refused rather than allowed to wrap.
        if (v > (std::numeric_limits<std::uint64_t>::max() >> 4)) {
            return false;
        }
        v = (v << 4) | static_cast<std::uint64_t>(d);
    }
    out = v;
    return true;
}

bool parse_hex_bytes(std::string_view s, std::vector<std::uint8_t>& out) noexcept {
    if (s.size() % 2 != 0) {
        return false;
    }
    out.clear();
    out.reserve(s.size() / 2);
    for (std::size_t i = 0; i + 1 < s.size(); i += 2) {
        const int hi = hex_digit(s[i]);
        const int lo = hex_digit(s[i + 1]);
        if (hi < 0 || lo < 0) {
            return false;
        }
        out.push_back(static_cast<std::uint8_t>((hi << 4) | lo));
    }
    return true;
}

void split_packet(std::string_view data, std::string_view& head,
                  std::string_view& tail) noexcept {
    // A packet is one command letter followed by its arguments. The split is
    // after the first byte, not at a ':', because a ':' appears inside the
    // arguments of several packets (a memory write is M<addr>,<len>:<bytes>)
    // and splitting there would cut a packet in the wrong place.
    if (data.empty()) {
        head = data;
        tail = std::string_view{};
        return;
    }
    head = data.substr(0, 1);
    tail = data.substr(1);
}

std::string encode_stop_reply(int signal, std::uint64_t rip) noexcept {
    // The T form carries the registers a debugger needs to report the stop
    // without a follow-up read. Only rip is included: it is the one
    // register that changes what the stop means, and the rest can be read
    // on demand.
    std::string out = "T";
    out.push_back(hex_char((signal >> 4) & 0xf));
    out.push_back(hex_char(signal & 0xf));
    // Register 16 is rip in the x86-64 register numbering GDB uses, and the
    // number is written as the protocol requires: hexadecimal, without a
    // leading zero, and with its own trailing colon.
    out += "10:";
    out += hex_u64_le(rip);
    out += ";";
    return out;
}

int gdb_signal_for_trap() noexcept {
    return 5; // SIGTRAP
}

int gdb_signal_for_breakpoint() noexcept {
    return 5; // SIGTRAP
}

} // namespace occ::obs
