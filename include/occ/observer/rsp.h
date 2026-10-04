#pragma once

// The GDB Remote Serial Protocol.
//
// The protocol is a byte stream of packets, each framed by '$' and '#',
// with a checksum of the payload after the '#'. It is chosen over inventing
// one because every debugger front end already speaks it: a target running
// under occ can be attached to from gdb, lldb, or any of the editors that
// drive them, without a plugin.
//
// The parser here is a state machine over bytes rather than a line reader,
// because the framing characters can appear inside a payload when it is
// escaped, and a reader that split on '#" would mis-frame a packet that
// contained one. The state is also what lets a partial read be resumed: a
// serial port and a socket both deliver bytes in whatever size chunks they
// feel like, and a parser that assumed a whole packet per read would work
// on a local socket and fail on everything else.
//
// What is implemented is the packet set a debugger needs to attach, read
// and write memory, inspect registers, set breakpoints, and single step.
// Packets outside that set are answered with an empty response, which the
// protocol defines as "not supported" -- an answer a debugger understands,
// rather than a disconnect it would have to guess at.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace occ::obs {

// One decoded packet.
struct Packet {
    // The packet's payload, with the framing and checksum removed and the
    // escapes undone. A command packet begins with a letter; a response
    // packet is whatever was queued.
    std::string data;
    // False when the checksum did not match, in which case the payload is
    // empty and the caller has to ask for a retransmission.
    bool checksum_ok = true;
};

// Incrementally decodes a byte stream into packets. Bytes are fed in as they
// arrive and packets come out as they complete.
class PacketDecoder {
public:
    // Feeds bytes into the decoder. Returns the number of packets that
    // became complete. Callers take them with take().
    std::size_t feed(const char* data, std::size_t length) noexcept;
    std::size_t feed(std::string_view s) noexcept {
        return feed(s.data(), s.size());
    }

    [[nodiscard]] bool has_packet() const noexcept { return !ready_.empty(); }
    [[nodiscard]] Packet take() noexcept;

    // True when the far end asked for a retransmission because a checksum
    // failed. Cleared by the next take().
    [[nodiscard]] bool retransmit_requested() const noexcept {
        return want_retransmit_;
    }
    void clear_retransmit() noexcept { want_retransmit_ = false; }

    // The last packet that was fully decoded, for a retransmission. Stored
    // because the protocol expects the same bytes back, checksum included.
    [[nodiscard]] const std::string& last_packet() const noexcept {
        return last_;
    }

private:
    enum class State : std::uint8_t {
        Idle,
        Payload,
        FirstChecksum,
        SecondChecksum,
        // After a '+', the next byte is read to decide whether it is an
        // acknowledgment or the start of a three-byte packet.
        Ack,
    };

    State state_ = State::Idle;
    std::string payload_;
    std::string raw_;
    int checksum_hi_ = 0;
    bool want_retransmit_ = false;
    std::string last_;
    std::vector<Packet> ready_;
};

// Encodes a payload into a framed packet. The payload is escaped and the
// checksum computed.
[[nodiscard]] std::string encode_packet(std::string_view payload) noexcept;

// Undoes the protocol's escape sequences. '}' is the escape byte and the
// byte after it has 0x20 exclusive-or'd into it.
[[nodiscard]] std::string unescape(std::string_view in) noexcept;
[[nodiscard]] std::string escape(std::string_view in) noexcept;

// ---------------------------------------------------------------- hex helpers
//
// GDB's wire format is hexadecimal for every binary quantity, and the byte
// order is the target's, not the protocol's. The helpers are here rather
// than in a general utility because their byte order is part of the
// protocol and not a general property.

// A register value is sent in the target's byte order. For a little-endian
// target that is least significant byte first, which is why these two are
// not the same as a general hex read.
[[nodiscard]] std::string hex_u64_le(std::uint64_t value) noexcept;
[[nodiscard]] std::string hex_u64_be(std::uint64_t value) noexcept;
[[nodiscard]] std::string hex_u32_le(std::uint32_t value) noexcept;
[[nodiscard]] std::string hex_u8(std::uint8_t value) noexcept;

// Writes a protocol number: hexadecimal, most significant digit first, no
// leading zeros. This is the form thread ids, offsets and signal numbers take
// on the wire, and it is not the form a register value takes.
[[nodiscard]] std::string hex_number(std::uint64_t value) noexcept;

[[nodiscard]] bool parse_hex_u64_le(std::string_view s,
                                    std::uint64_t& out) noexcept;
[[nodiscard]] bool parse_hex_u32_le(std::string_view s,
                                    std::uint32_t& out) noexcept;

// Reads a protocol number: hexadecimal, variable width, most significant
// digit first, with no padding significance. Offsets, lengths, register
// numbers and signals are all this, and all of them arrive narrower than the
// eight or sixteen digits the register-block readers above require.
[[nodiscard]] bool parse_hex_number(std::string_view s,
                                    std::uint64_t& out) noexcept;

[[nodiscard]] bool parse_hex_bytes(std::string_view s,
                                   std::vector<std::uint8_t>& out) noexcept;

// Splits a packet payload at the first ':'. A packet with no ':' yields
// everything in `head` and an empty `tail`, which is what a command with no
// arguments looks like.
void split_packet(std::string_view data, std::string_view& head,
                  std::string_view& tail) noexcept;

// ------------------------------------------------------------------ stop reply
//
// The reply to a continue or a step, and to the question "why did you stop".
// The two forms are T (with registers) and S (with just a signal); the
// tracker uses T because a debugger that has to follow up with a register
// read for every stop doubles the traffic on the wire for no reason.

[[nodiscard]] std::string encode_stop_reply(int signal,
                                            std::uint64_t rip) noexcept;

// The signal number GDB expects for a given ptrace stop. The protocol's
// numbering is the target's, and SIGTRAP is 5 on every Linux target.
[[nodiscard]] int gdb_signal_for_trap() noexcept;
[[nodiscard]] int gdb_signal_for_breakpoint() noexcept;

} // namespace occ::obs
