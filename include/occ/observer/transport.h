#pragma once

// The remote debugger protocol over a socket.
//
// DebugServer answers packets. It does not own a connection, which is what
// makes it testable by feeding it strings, but something has to carry those
// strings to a debugger and read the replies back. That is this header's
// whole job, and it is deliberately the only part of the debugger that knows
// about file descriptors.
//
// The listener binds a loopback port rather than a wildcard one. A stub that
// traced a process can hand out the ability to read and write that process's
// memory to anything that can reach the socket, so the port is not something
// to expose by default: a debugger that is not on this machine is not a
// debugger this tool has a use case for, and "run it in a container" is a
// better answer than "make the port private".

#include <cstdint>
#include <string>

#include "occ/observer/rsp.h"

namespace occ::obs {

// Defined in session.h. Only a reference appears here, so a forward
// declaration is enough and the two headers do not have to include each
// other.
class DebugServer;

// One accepted connection.
//
// The protocol is a request and response loop over a stream with no framing
// beyond the protocol's own, so this owns the buffering that turns the byte
// stream back into packets and knows when the peer went away.
class Connection {
public:
    Connection() noexcept = default;
    ~Connection() noexcept;

    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    Connection(Connection&& other) noexcept;
    Connection& operator=(Connection&& other) noexcept;

    // Adopts an already-accepted descriptor. It is moved in rather than
    // connected so that a caller can set options on the raw descriptor before
    // anything is read from it.
    void adopt(int fd) noexcept;

    [[nodiscard]] bool valid() const noexcept { return fd_ >= 0; }
    [[nodiscard]] int fd() const noexcept { return fd_; }

    // Reads whatever is available and appends it to the packet decoder.
    // Returns false when the peer closed or the descriptor failed, which is
    // the signal to stop serving. A read that would block returns true with
    // nothing added, so a caller can wait for readiness first.
    [[nodiscard]] bool pump() noexcept;

    // True when a complete packet is waiting to be handled. A packet whose
    // checksum failed does not count: there is nothing to handle until the
    // debugger has been asked to send it again.
    [[nodiscard]] bool has_packet() const noexcept;

    // Takes the next packet, if there is one. Returns false when there is
    // not, which is not an error: the stream simply has nothing complete, or
    // the last packet failed its checksum and is waiting to be retransmitted.
    [[nodiscard]] bool take_packet(std::string& out) noexcept;

    // True when the far end asked for the previous packet again because a
    // checksum did not match. The reply has to be the same bytes, so they
    // are kept rather than regenerated.
    [[nodiscard]] bool retransmit_requested() const noexcept;
    void clear_retransmit() noexcept;
    [[nodiscard]] const std::string& last_packet() const noexcept;

    // Sends a payload as one packet and waits for its acknowledgement. The
    // acknowledgement is what makes a reply reliable on a stream that does
    // not promise delivery, and a debugger that never saw it will send the
    // packet again.
    [[nodiscard]] bool send_packet(std::string_view payload) noexcept;

    // Waits up to the timeout for the peer to close or for data to arrive.
    // Used to notice a debugger that has gone away without waiting for the
    // process to stop.
    [[nodiscard]] bool wait_readable(int timeout_ms) noexcept;

    void close() noexcept;

private:
    int fd_ = -1;
    PacketDecoder decoder_;
};

// A listening socket, bound and ready to accept.
//
// The port is chosen by the caller as a number rather than as a string so
// that zero can be assigned by the kernel and read back, which is how the
// caller learns which port it got when it asked for any free one.
class Listener {
public:
    Listener() noexcept = default;
    ~Listener() noexcept;

    Listener(const Listener&) = delete;
    Listener& operator=(const Listener&) = delete;
    Listener(Listener&& other) noexcept;
    Listener& operator=(Listener&& other) noexcept;

    // Binds and listens. On failure returns false and fills `error` with a
    // reason that names what was refused rather than the bare errno, because
    // "address already in use" and "permission denied" call for different
    // responses from the person who asked for the port.
    [[nodiscard]] bool open(std::uint16_t port, std::string& error) noexcept;

    [[nodiscard]] bool valid() const noexcept { return fd_ >= 0; }
    [[nodiscard]] int fd() const noexcept { return fd_; }

    // The port actually bound, which differs from the one asked for when zero
    // was requested.
    [[nodiscard]] std::uint16_t port() const noexcept { return port_; }

    // Waits up to the timeout for a debugger to connect. Returns false on
    // timeout or when the listener is closed, which a caller treats the same
    // way: no debugger arrived, so the run continues without one.
    [[nodiscard]] bool accept(Connection& out, int timeout_ms) noexcept;

    void close() noexcept;

private:
    int fd_ = -1;
    std::uint16_t port_ = 0;
};


} // namespace occ::obs
