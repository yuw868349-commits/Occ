#include "occ/observer/transport.h"

#include "occ/observer/session.h"

#include "occ/syscall/errno.h"
#include "occ/syscall/syscall.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

namespace occ::obs {

namespace {

// The address a listener binds. Loopback rather than the wildcard address,
// because a socket a debugger uses to read and write a traced process's
// memory is a capability, and a capability should not be reachable from off
// the machine by default.
constexpr std::uint32_t kLoopbackV4 = 0x0100007f; // 127.0.0.1, network order

// The socket address layout the kernel expects. Written out rather than
// included because <netinet/in.h> drags in a libc the rest of this project
// does not use, and because the byte order is the part that has to be right.
struct SockAddrIn {
    std::uint16_t family;
    std::uint16_t port;
    std::uint32_t addr;
    std::uint8_t zero[8];
};

constexpr std::uint16_t kAfInet = 2;
constexpr int kSockStream = 1;
constexpr int kSockCloexec = 0x80000;

// Big endian in the address, which is the one field whose byte order is not
// the target's: the kernel reads a port as network order whatever the
// machine it runs on does with integers.
[[nodiscard]] std::uint16_t to_be16(std::uint16_t v) noexcept {
    return static_cast<std::uint16_t>((v >> 8) | (v << 8));
}

// Packs a port for the kernel. Zero means "any free port", which is how a
// caller avoids a collision without asking the operator to pick one.
[[nodiscard]] std::uint16_t sin_port(std::uint16_t port) noexcept {
    return to_be16(port);
}

[[nodiscard]] std::uint16_t from_be16(std::uint16_t v) noexcept {
    return to_be16(v);
}

} // namespace

// ------------------------------------------------------------------ Listener

bool Listener::accept(Connection& out, int timeout_ms) noexcept {
    if (fd_ < 0) {
        return false;
    }
    // Waiting for readiness first is what makes the timeout real. A blocking
    // accept cannot be given a deadline, and a run that asked for a debugger
    // port would then sit here forever when nobody connects.
    sys::PollFd pfd{};
    pfd.fd = fd_;
    pfd.events = 0x0001; // POLLIN
    pfd.revents = 0;
    const auto ready = sys::poll(&pfd, 1, timeout_ms);
    if (ready.failed() || ready.value == 0) {
        return false;
    }

    const auto fd = sys::accept4(fd_, nullptr, nullptr, kSockCloexec);
    if (fd.failed()) {
        return false;
    }
    out.adopt(static_cast<int>(fd.value));
    return true;
}

Listener::~Listener() noexcept { close(); }

Listener::Listener(Listener&& other) noexcept
    : fd_(other.fd_), port_(other.port_) {
    other.fd_ = -1;
    other.port_ = 0;
}

Listener& Listener::operator=(Listener&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = other.fd_;
        port_ = other.port_;
        other.fd_ = -1;
        other.port_ = 0;
    }
    return *this;
}

bool Listener::open(std::uint16_t port, std::string& error) noexcept {
    close();

    const auto sock = sys::socket(kAfInet, kSockStream | kSockCloexec, 0);
    if (sock.failed()) {
        error = std::string{"cannot create a socket: "} +
                util::strerror(static_cast<int>(sock.value));
        return false;
    }
    fd_ = static_cast<int>(sock.value);

    // The three fields have three different byte orders, which is the whole
    // reason this structure is written out. sin_family is the address family
    // as a host-order number -- 2 for AF_INET -- and swapping its bytes makes
    // it 512, which the kernel rejects with an address-family error that
    // looks like a policy refusal rather than a mistake. sin_port and
    // sin_addr are in network order.
    SockAddrIn addr{};
    addr.family = kAfInet;
    addr.port = sin_port(port);
    addr.addr = kLoopbackV4;

    const auto bound = sys::bind(fd_, &addr, sizeof(addr));
    if (bound.failed()) {
        // The two failures here call for different responses from whoever
        // asked for the port, so they are named rather than reported as a
        // single "bind failed".
        const int err = static_cast<int>(bound.value);
        error = err == sys::kEaddrinuse
                    ? "the port is already in use by another session"
                    : std::string{"cannot bind the port: "} + util::strerror(err);
        close();
        return false;
    }

    // Read the port back, because a request for zero is answered by the
    // kernel and the caller has to be told which one it got.
    SockAddrIn chosen{};
    unsigned int chosen_len = sizeof(chosen);
    if (sys::getsockname(fd_, &chosen, &chosen_len).ok() && chosen_len >= 8) {
        port_ = from_be16(chosen.port);
    } else {
        port_ = port;
    }

    const auto listening = sys::listen(fd_, 1);
    if (listening.failed()) {
        error = std::string{"cannot listen: "} +
                util::strerror(static_cast<int>(listening.value));
        close();
        return false;
    }
    return true;
}

void Listener::close() noexcept {
    if (fd_ >= 0) {
        (void)sys::close(fd_);
        fd_ = -1;
    }
    port_ = 0;
}

// ---------------------------------------------------------------- Connection

Connection::~Connection() noexcept { close(); }

Connection::Connection(Connection&& other) noexcept
    : fd_(other.fd_), decoder_(std::move(other.decoder_)) {
    other.fd_ = -1;
}

Connection& Connection::operator=(Connection&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = other.fd_;
        decoder_ = std::move(other.decoder_);
        other.fd_ = -1;
    }
    return *this;
}

void Connection::adopt(int fd) noexcept {
    close();
    fd_ = fd;
}

void Connection::close() noexcept {
    if (fd_ >= 0) {
        (void)sys::close(fd_);
        fd_ = -1;
    }
    decoder_ = PacketDecoder{};
}

bool Connection::pump() noexcept {
    if (fd_ < 0) {
        return false;
    }
    char buf[4096];
    const auto got = sys::read(fd_, buf, sizeof(buf));
    if (got.ok()) {
        if (got.value == 0) {
            return false; // the peer closed
        }
        decoder_.feed(buf, static_cast<std::size_t>(got.value));
        return true;
    }
    // A read that would block means there is nothing yet, which is not a
    // failure: the caller waits for readiness and comes back.
    if (got.value == -sys::kEintr) {
        return true;
    }
    return false;
}

bool Connection::has_packet() const noexcept {
    return decoder_.has_packet() && !decoder_.retransmit_requested();
}

bool Connection::retransmit_requested() const noexcept {
    return decoder_.retransmit_requested();
}

void Connection::clear_retransmit() noexcept { decoder_.clear_retransmit(); }

const std::string& Connection::last_packet() const noexcept {
    return decoder_.last_packet();
}

bool Connection::take_packet(std::string& out) noexcept {
    if (!decoder_.has_packet()) {
        return false;
    }
    Packet p = decoder_.take();
    // A packet whose checksum failed has no usable payload. It is dropped
    // rather than handed on, because a command read out of a corrupt packet
    // is a command nobody sent: resuming a process, or writing its memory,
    // on the strength of a transmission error is the worst thing a debugger
    // stub can do. The decoder has already recorded the retransmission.
    if (!p.checksum_ok) {
        out.clear();
        return false;
    }
    out.assign(p.data);
    return true;
}

bool Connection::send_packet(std::string_view payload) noexcept {
    if (fd_ < 0) {
        return false;
    }
    const std::string framed = encode_packet(payload);
    std::size_t sent = 0;
    while (sent < framed.size()) {
        const auto wrote =
            sys::write(fd_, framed.data() + sent, framed.size() - sent);
        if (wrote.failed()) {
            if (wrote.value == -sys::kEintr) {
                continue;
            }
            return false;
        }
        sent += static_cast<std::size_t>(wrote.value);
    }
    return true;
}

bool Connection::wait_readable(int timeout_ms) noexcept {
    if (fd_ < 0) {
        return false;
    }
    // POLLIN is 0x0001. It is the only event that matters here: the reply to
    // a packet is written immediately and never waited on, and a debugger
    // that goes quiet is noticed by the timeout rather than by an event.
    sys::PollFd pfd{};
    pfd.fd = fd_;
    pfd.events = 0x0001;
    pfd.revents = 0;
    const auto ready = sys::poll(&pfd, 1, timeout_ms);
    return ready.ok() && ready.value > 0;
}

// The serve loop lives in session.cpp, next to the tracee loop it has to
// interleave with. This file is only the transport: a place to bind, accept,
// and move bytes. Splitting it that way keeps the socket handling testable
// without a process and keeps the loop that owns the tracee in one place.

} // namespace occ::obs
