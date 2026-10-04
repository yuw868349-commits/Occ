#include "occ/observer/watchpoint.h"

#include "occ/syscall/errno.h"
#include "occ/syscall/kabi.h"

#include <cstring>

#include <linux/hw_breakpoint.h>
#include <linux/perf_event.h>

#include <unistd.h>

namespace occ::obs {

namespace {

// The kernel's hw_breakpoint_type values. Restated because the header spells
// them with an HW_BREAKPOINT_ prefix that does not survive being used as a
// small enum.
constexpr std::uint32_t kHwBreakpointExecute = 0;
constexpr std::uint32_t kHwBreakpointWrite = 1;
constexpr std::uint32_t kHwBreakpointReadWrite = 2;
constexpr std::uint32_t kHwBreakpointLen1 = 1;
constexpr std::uint32_t kHwBreakpointLen2 = 2;
constexpr std::uint32_t kHwBreakpointLen4 = 4;
constexpr std::uint32_t kHwBreakpointLen8 = 8;

std::uint32_t to_kernel_kind(WatchKind kind) noexcept {
    switch (kind) {
    case WatchKind::Execute:
        return kHwBreakpointExecute;
    case WatchKind::Write:
        return kHwBreakpointWrite;
    case WatchKind::ReadWrite:
    case WatchKind::Read:
        return kHwBreakpointReadWrite;
    }
    return kHwBreakpointWrite;
}

std::uint32_t to_kernel_len(WatchSize size) noexcept {
    switch (size) {
    case WatchSize::Bytes1:
        return kHwBreakpointLen1;
    case WatchSize::Bytes2:
        return kHwBreakpointLen2;
    case WatchSize::Bytes4:
        return kHwBreakpointLen4;
    case WatchSize::Bytes8:
        return kHwBreakpointLen8;
    }
    return kHwBreakpointLen8;
}

} // namespace

Watchpoints::~Watchpoints() {
    for (const auto& w : watches_) {
        if (w.fd >= 0) {
            (void)sys::close(w.fd);
        }
    }
}

std::uint32_t Watchpoints::hardware_slots() noexcept {
    // The count comes from the kernel rather than from an assumption about
    // the architecture. x86-64 has four in practice, but a virtual machine
    // can report fewer and a kernel built without the feature reports none,
    // and a tracker that assumed four would try to install a watch that can
    // never work.
    //
    // There is no syscall that returns the number directly, so it is probed:
    // create watches on a process until one is refused, then remove them.
    // The probe runs against the calling process on a scratch address, which
    // costs nothing and leaves no state behind.
    struct perf_event_attr attr;
    std::memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_BREAKPOINT;
    attr.size = sizeof(attr);
    attr.bp_type = kHwBreakpointWrite;
    attr.bp_addr = reinterpret_cast<std::uint64_t>(&attr);
    attr.bp_len = kHwBreakpointLen1;
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;

    int fds[16];
    int count = 0;
    for (int i = 0; i < 16; ++i) {
        auto r = sys::perf_event_open(&attr, 0, -1, -1, 0);
        if (r.failed()) {
            break;
        }
        fds[count++] = static_cast<int>(r.value);
    }
    for (int i = 0; i < count; ++i) {
        (void)sys::close(fds[i]);
    }
    return static_cast<std::uint32_t>(count);
}

int Watchpoints::add(int pid, std::uint64_t address, WatchKind kind,
                     WatchSize size, std::string& detail) noexcept {
    if (kind == WatchKind::Read) {
        // The hardware has no read-only encoding. Reporting the limitation
        // is the honest answer; silently installing a read-or-write watch
        // would make the caller's filter wrong in a way it cannot see.
        detail = "a read-only watch is not a hardware breakpoint type";
        return sys::kEinval;
    }

    // A watch is limited to a naturally aligned access of at most eight
    // bytes. An address that is not aligned for its size is refused by the
    // kernel, but refusing it here names the reason.
    const std::uint64_t width = static_cast<std::uint64_t>(size);
    if (width != 0 && (address % width) != 0) {
        detail = "the address is not aligned for the requested width";
        return sys::kEinval;
    }

    // The same address watched twice is a caller mistake, but it is one
    // that would silently consume a scarce hardware slot, so it is folded
    // into the existing watch instead.
    for (const auto& w : watches_) {
        if (w.pid == pid && w.address == address && w.kind == kind &&
            w.size == size) {
            return 0;
        }
    }

    struct perf_event_attr attr;
    std::memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_BREAKPOINT;
    attr.size = sizeof(attr);
    attr.bp_type = to_kernel_kind(kind);
    attr.bp_addr = address;
    attr.bp_len = to_kernel_len(size);
    // Delivered disabled and enabled explicitly, so that a watch installed
    // during setup cannot fire before the caller is ready to read events.
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    // The sample period is one: every access is wanted, not a sample of
    // them. A tracker that sampled would report a write it happened to see
    // rather than all of them.
    attr.sample_period = 1;

    auto r = sys::perf_event_open(&attr, pid, -1, -1, 0);
    if (r.failed()) {
        if (r.error == sys::kEnospc) {
            detail = "all hardware debug registers are in use";
        } else if (r.error == sys::kEacces) {
            detail = "perf events are restricted for this process";
        } else {
            detail = "perf_event_open refused the watch";
        }
        return r.error;
    }

    HardwareWatch w;
    w.fd = static_cast<int>(r.value);
    w.pid = pid;
    w.address = address;
    w.kind = kind;
    w.size = size;
    watches_.push_back(w);
    return 0;
}

int Watchpoints::remove(std::uint64_t address) noexcept {
    for (auto it = watches_.begin(); it != watches_.end(); ++it) {
        if (it->address != address) {
            continue;
        }
        if (it->fd >= 0) {
            (void)sys::close(it->fd);
        }
        watches_.erase(it);
        return 0;
    }
    return 0;
}

int Watchpoints::remove_all_for(int pid) noexcept {
    for (auto it = watches_.begin(); it != watches_.end();) {
        if (it->pid != pid) {
            ++it;
            continue;
        }
        if (it->fd >= 0) {
            (void)sys::close(it->fd);
        }
        it = watches_.erase(it);
    }
    return 0;
}

// -------------------------------------------------------------- decoding
//
// The instruction decode below covers the mov forms the W-to-X tracker
// needs and nothing more. It is deliberately not a general disassembler:
// a general one would be a large amount of code whose correctness could
// not be checked by the tests that exist, and the tracker only ever needs
// to answer "was this a write" for a mov, because that is the instruction
// a compiler emits when it makes a page executable.

namespace {

// The register names in ModRM encoding order, for the base register field.
// These are the 64-bit forms; the 32-bit forms used by an address-size
// override are not distinguished, because the tracker compares the decoded
// base against a watched address rather than against a register value.
constexpr int kRegRax = 0;
constexpr int kRegRcx = 1;
constexpr int kRegRdx = 2;
constexpr int kRegRbx = 3;
constexpr int kRegRsp = 4;
constexpr int kRegRbp = 5;
constexpr int kRegRsi = 6;
constexpr int kRegRdi = 7;

} // namespace

AccessDecode decode_access(const std::uint8_t* code,
                           std::size_t length) noexcept {
    AccessDecode out;
    if (code == nullptr || length == 0) {
        return out;
    }

    std::size_t i = 0;

    // Legacy prefixes. The only ones that change the decode are REX, the
    // operand-size override and the address-size override, and the latter
    // two are ignored rather than mis-decoded: the register the address is
    // based on is the same either way.
    bool rex_w = false;
    for (; i < length && i < 4; ++i) {
        const std::uint8_t b = code[i];
        if (b == 0x66 || b == 0x67 || b == 0xf2 || b == 0xf3) {
            continue;
        }
        if (b >= 0x40 && b <= 0x4f) {
            rex_w = (b & 0x08) != 0;
            continue;
        }
        break;
    }
    if (i >= length) {
        return out;
    }

    const std::uint8_t opcode = code[i++];
    if (opcode != 0x89 && opcode != 0x88 && opcode != 0x8b && opcode != 0x8a) {
        // Not a mov. Reported as undetermined rather than guessed: a wrong
        // is_write would make the tracker act on an access that did not
        // happen.
        return out;
    }

    // 0x89/0x8b are the 32/64-bit forms, 0x88/0x8a the 8-bit ones.
    const bool byte_form = (opcode == 0x88 || opcode == 0x8a);
    // 0x89 and 0x88 have the memory operand as the destination, which makes
    // them writes. 0x8b and 0x8a have it as the source, which makes them
    // reads.
    const bool store = (opcode == 0x89 || opcode == 0x88);

    if (i >= length) {
        return out;
    }
    const std::uint8_t modrm = code[i++];
    const std::uint8_t mod = (modrm >> 6) & 3;
    const std::uint8_t rm = modrm & 7;

    if (mod == 3) {
        // Register to register. No memory access, so this cannot be the
        // instruction that faulted on the watch.
        return out;
    }

    // A SIB byte follows when the base field is rsp/r12.
    int base = static_cast<int>(rm);
    if (rm == 4) {
        if (i >= length) {
            return out;
        }
        const std::uint8_t sib = code[i++];
        base = sib & 7;
    }

    const bool rip_relative = (mod == 0 && (rm == 5 || (rm == 4 && base == 5)));

    std::int64_t displacement = 0;
    if (mod == 1) {
        if (i >= length) {
            return out;
        }
        displacement = static_cast<std::int8_t>(code[i++]);
    } else if (mod == 2 || rip_relative) {
        if (i + 4 > length) {
            return out;
        }
        std::int32_t d = 0;
        std::memcpy(&d, code + i, 4);
        displacement = d;
        i += 4;
    }

    out.valid = true;
    out.is_write = store;
    out.displacement = displacement;

    if (rip_relative) {
        // The access is relative to the instruction pointer, which the
        // caller knows and this decode does not. Reported as -1 so the
        // caller can compute the target from rip plus the instruction
        // length plus the displacement.
        out.base_register = -1;
    } else if (base == kRegRbp && mod == 0) {
        // mod 0 with base rbp is a displacement-only encoding, not a
        // register-indirect access, so there is no base register.
        out.base_register = -1;
    } else {
        out.base_register = base;
    }

    if (byte_form) {
        out.width = 1;
    } else if (rex_w) {
        out.width = 8;
    } else {
        out.width = 4;
    }

    // The unused-constant warnings for the register names are avoided by
    // naming the one that is special; the rest document the encoding for a
    // reader and are otherwise unused.
    (void)kRegRax;
    (void)kRegRcx;
    (void)kRegRdx;
    (void)kRegRbx;
    (void)kRegRsp;
    (void)kRegRsi;
    (void)kRegRdi;

    return out;
}

} // namespace occ::obs
