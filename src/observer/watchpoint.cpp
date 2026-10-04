#include "occ/observer/watchpoint.h"

#include "occ/syscall/errno.h"
#include "occ/syscall/kabi.h"

#include <cstring>

#include <linux/hw_breakpoint.h>
#include <linux/perf_event.h>

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace occ::obs {

namespace {

// The kernel's hw_breakpoint_type values and hw_breakpoint_len values come
// from <linux/hw_breakpoint.h> and are used as the kernel spells them.
//
// They are deliberately not restated here. The type field is a bit field --
// HW_BREAKPOINT_RW is R | W, not the value after W -- so a hand-written
// consecutive enumeration is wrong in a way no compiler or test will catch:
// the kernel rejects a type that names a read watch over an address that is
// not readable with a bare EINVAL, which is indistinguishable from a host
// with no debug registers. That is not hypothetical; this file did exactly
// that and reported every machine as incapable of write tracking.

// The perf sample fields a watch asks for. The instruction pointer says
// which code performed the access and the data address says what it touched;
// a watch that asked for neither would report that something happened
// without saying what.
constexpr std::uint64_t kPerfSampleIp = 1ULL << 0;
constexpr std::uint64_t kPerfSampleAddr = 1ULL << 3;

// The record types this reader decodes out of the ring.
constexpr std::uint32_t kPerfRecordLost = PERF_RECORD_LOST;
constexpr std::uint32_t kPerfRecordSample = PERF_RECORD_SAMPLE;

// The header every ring buffer record starts with. The size field is what
// makes a ring decodable without knowing the record type in advance, which
// is what lets a forward-compatible reader skip a record it does not
// understand instead of losing its place in the ring.
struct RingHeader {
    std::uint32_t type;
    std::uint16_t misc;
    std::uint16_t size;
};

// The byte offsets of the two ring pointers inside the metadata page. They
// are ABI rather than layout: offsetof(perf_event_mmap_page, data_head) is
// 1024 on the 64-bit architecture and the field is followed immediately by
// data_tail. Restated here because the kernel header declares them in a C
// struct and this translation unit is not C.
constexpr std::size_t kRingDataHeadByte = 1024;
constexpr std::size_t kRingDataTailByte = 1032;

std::uint32_t to_kernel_kind(WatchKind kind) noexcept {
    switch (kind) {
    case WatchKind::Execute:
        return HW_BREAKPOINT_X;
    case WatchKind::Write:
        return HW_BREAKPOINT_W;
    case WatchKind::ReadWrite:
    case WatchKind::Read:
        return HW_BREAKPOINT_RW;
    }
    return HW_BREAKPOINT_W;
}

std::uint32_t to_kernel_len(WatchSize size) noexcept {
    switch (size) {
    case WatchSize::Bytes1:
        return HW_BREAKPOINT_LEN_1;
    case WatchSize::Bytes2:
        return HW_BREAKPOINT_LEN_2;
    case WatchSize::Bytes4:
        return HW_BREAKPOINT_LEN_4;
    case WatchSize::Bytes8:
        return HW_BREAKPOINT_LEN_8;
    }
    return HW_BREAKPOINT_LEN_8;
}

// Reads the two ring pointers out of a ring buffer's metadata page:
// data_head is where the producer has written, data_tail is where the
// consumer has read.
//
// The accesses are atomic rather than volatile because the producer is the
// traced process's own CPU and this is the one place in the observer where
// two cores touch the same word without a lock. Reading head before tail is
// the order that matters: a head that moves again is a ring still being
// written to, and reading it first is what makes the tail that follows
// consistent with it.
std::uint64_t ring_load(const void* ring, std::size_t byte_offset) noexcept {
    // The metadata page is declared volatile because it is written by
    // another core, and the pointer is formed with reinterpret_cast rather
    // than static_cast because a byte pointer does not convert to a
    // volatile 64-bit one under the strict aliasing rules this build uses.
    const auto* words = reinterpret_cast<const volatile std::uint64_t*>(
        static_cast<const std::uint8_t*>(ring));
    return __atomic_load_n(words + byte_offset / 8, __ATOMIC_ACQUIRE);
}

void ring_store(void* ring, std::size_t byte_offset,
                std::uint64_t value) noexcept {
    auto* words = reinterpret_cast<volatile std::uint64_t*>(
        static_cast<std::uint8_t*>(ring));
    __atomic_store_n(words + byte_offset / 8, value, __ATOMIC_RELEASE);
}

} // namespace

void Watchpoints::release(HardwareWatch& w) noexcept {
    if (w.ring != nullptr) {
        (void)sys::munmap(w.ring, w.ring_bytes);
        w.ring = nullptr;
        w.ring_bytes = 0;
    }
    w.ring_tail = 0;
    if (w.fd >= 0) {
        (void)sys::close(w.fd);
        w.fd = -1;
    }
}

Watchpoints::~Watchpoints() {
    for (auto& w : watches_) {
        release(w);
    }
}

std::uint32_t Watchpoints::machine_slots() noexcept {
    // The capacity is cached because the probe is destructive in the sense
    // that matters: it counts the registers the calling process has left, so
    // a caller that already holds every register is told zero. That answer is
    // right for "how many can I add" and wrong for "how many exist", and the
    // two questions have to be asked at different moments.
    //
    // The first call therefore records the capacity and every later call
    // returns it. The probe installs and closes its events immediately, so
    // the cache records what was available when the tracker held nothing --
    // which is the machine's capacity, since the registers are per process
    // and the probe runs against this one.
    //
    // The cache is not a claim that the count never changes. A machine can
    // have its debug registers taken by another agent at any time, and the
    // add() path reports that refusal at the moment it happens rather than
    // trusting a cached number.
    static const std::uint32_t capacity = probe_slots();
    return capacity;
}

std::uint32_t Watchpoints::free_slots() const noexcept {
    // The watches already held are subtracted from the machine's capacity.
    // Asking the kernel instead would report the same number here, but only
    // because the probe runs against this process and this process is the one
    // holding the watches -- a coupling that would silently break the moment
    // the watches were installed against a tracee instead.
    const std::uint32_t capacity = machine_slots();
    const std::uint32_t held = static_cast<std::uint32_t>(watches_.size());
    return held >= capacity ? 0u : capacity - held;
}

std::uint32_t Watchpoints::probe_slots() noexcept {
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
    //
    // A created event is not a usable one until it is enabled, so the probe
    // performs the enable as well. Skipping it would count events that the
    // kernel refuses to start, and the count would be a promise the
    // installer cannot keep.
    static int scratch[4];

    struct perf_event_attr attr;
    std::memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_BREAKPOINT;
    attr.size = sizeof(attr);
    attr.bp_type = HW_BREAKPOINT_W;
    attr.bp_addr = reinterpret_cast<std::uint64_t>(scratch);
    // Four bytes, for the reason given in hardware_available: a one-byte
    // write watch is refused by the kernel, so a probe that used one would
    // always report zero slots.
    attr.bp_len = HW_BREAKPOINT_LEN_4;
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
        const int fd = static_cast<int>(r.value);
        auto en = sys::ioctl(fd, PERF_EVENT_IOC_ENABLE, nullptr);
        if (en.failed()) {
            (void)sys::close(fd);
            break;
        }
        fds[count++] = fd;
    }
    for (int i = 0; i < count; ++i) {
        (void)sys::close(fds[i]);
    }
    return static_cast<std::uint32_t>(count);
}

bool Watchpoints::hardware_available() noexcept {
    // A count is not a capability. A seccomp filter in front of this process
    // can make perf_event_open fail with EPERM while the hardware and the
    // kernel both support the event, and a caller that trusted the count
    // would arm a watch, be told it succeeded, and never see it fire.
    //
    // The probe therefore runs the whole path an installed watch runs: open,
    // ring mapping, enable. All three can be refused, and the last one is the
    // easiest to get wrong, because a request number that is off by a shift
    // or an mmap that lands on an unmapped page both surface as a plain
    // failure that looks like a host restriction.
    //
    // The width is 4, not 1. x86-64 has encode and length fields that accept
    // 1, 2, 4 and 8, but the hardware only implements 1 and 4 for a *write*
    // watch: the encoding is taken from the linear address itself, and an
    // odd address has no encoding that can express it. The kernel accepts the
    // request and then refuses it with EINVAL, which is indistinguishable
    // from a machine that cannot do hardware watches at all. A probe that
    // used the smallest width the field can hold would therefore report
    // every x86 machine as incapable, which is what it did.
    static int scratch[4];

    struct perf_event_attr attr;
    std::memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_BREAKPOINT;
    attr.size = sizeof(attr);
    attr.bp_type = HW_BREAKPOINT_W;
    attr.bp_addr = reinterpret_cast<std::uint64_t>(scratch);
    attr.bp_len = HW_BREAKPOINT_LEN_4;
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;

    auto r = sys::perf_event_open(&attr, 0, -1, -1, 0);
    if (r.failed()) {
        return false;
    }
    const int fd = static_cast<int>(r.value);

    const long page = ::sysconf(_SC_PAGESIZE);
    const std::size_t page_size = page > 0 ? static_cast<std::size_t>(page)
                                           : static_cast<std::size_t>(4096);

    auto mm = sys::mmap(nullptr, kWatchRingPages * page_size,
                        PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mm.failed()) {
        (void)sys::close(fd);
        return false;
    }
    void* ring =
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(mm.value));

    auto en = sys::ioctl(fd, PERF_EVENT_IOC_ENABLE, nullptr);
    if (en.failed()) {
        (void)sys::munmap(ring, kWatchRingPages * page_size);
        (void)sys::close(fd);
        return false;
    }

    (void)sys::munmap(ring, kWatchRingPages * page_size);
    (void)sys::close(fd);
    return true;
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

    const std::uint64_t width = static_cast<std::uint64_t>(size);

    // A write watch is limited to a naturally aligned access of at least
    // four bytes and at most eight. One and two byte write watches are not
    // encodable on x86-64: the hardware derives the address encoding from
    // the linear address, and an address that is not four-aligned has no
    // encoding. The kernel refuses those with EINVAL, so they are refused
    // here with the reason attached.
    //
    // A one or two byte *read* watch is a different case and is legal; the
    // restriction is on the write type alone, which is why it is checked
    // here rather than as a general property of the width.
    if (kind == WatchKind::Write && width != 4 && width != 8) {
        detail = "a write watch must be four or eight bytes wide";
        return sys::kEinval;
    }

    // A watch is limited to a naturally aligned access of at most eight
    // bytes. An address that is not aligned for its size is refused by the
    // kernel, but refusing it here names the reason.
    if (width != 0 && (address % width) != 0) {
        detail = "the address is not aligned for the requested width";
        return sys::kEinval;
    }

    // The same address watched twice is a caller mistake, but it is one
    // that would silently consume a scarce hardware slot, so it is folded
    // into the existing watch instead.
    for (auto& w : watches_) {
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
    // Delivered disabled and enabled explicitly below, so that a watch
    // installed during setup cannot fire before its ring buffer is mapped.
    // A watch that fires into an unmapped ring is a lost event, and a lost
    // event in a tracker is a false negative.
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    // The sample period is one: every access is wanted, not a sample of
    // them. A tracker that sampled would report a write it happened to see
    // rather than all of them.
    attr.sample_period = 1;
    attr.sample_type = kPerfSampleIp | kPerfSampleAddr;
    // Wake up on every record rather than after a batch. The observer's loop
    // is stop-driven and already wakes on the target's own stops, so a
    // watch event that waits to be batched would be reported late or not at
    // all on a target that makes few syscalls.
    attr.wakeup_events = 1;

    auto r = sys::perf_event_open(&attr, pid, -1, -1, 0);
    if (r.failed()) {
        if (r.error == sys::kEnospc) {
            detail = "all hardware debug registers are in use";
        } else if (r.error == sys::kEacces) {
            detail = "perf events are restricted for this process";
        } else if (r.error == sys::kEperm) {
            detail = "perf_event_open was refused by a security policy";
        } else {
            detail = "perf_event_open refused the watch";
        }
        return r.error;
    }

    const int fd = static_cast<int>(r.value);

    // The ring buffer. It has to be mapped before the event is enabled, and
    // it has to be a power-of-two page count plus one, because the kernel
    // computes the wrap with a mask.
    const long page = ::sysconf(_SC_PAGESIZE);
    const std::size_t page_size = page > 0 ? static_cast<std::size_t>(page)
                                           : static_cast<std::size_t>(4096);
    const std::size_t ring_bytes = kWatchRingPages * page_size;

    auto mm = sys::mmap(nullptr, ring_bytes, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd, 0);
    if (mm.failed()) {
        (void)sys::close(fd);
        detail = "the watch ring buffer could not be mapped";
        return mm.error;
    }
    // The syscall wrapper returns every value in a long, so an address comes
    // back as an integer and has to be turned back into a pointer before
    // anything can be done with it.
    void* ring =
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(mm.value));

    // Enabling is separate from creation so that the buffer above is already
    // there.
    //
    // The request number is used exactly as the kernel spells it. The
    // PERF_EVENT_IOC_* constants are _IO, which carries a zero size field,
    // so OR-ing a pointer width into bits 16 and up -- which is what the
    // _IOW-shaped commands would need -- changes the request into a
    // different one. The kernel decodes the size out of those bits, does not
    // recognise the result as a command it serves, and answers EPERM, which
    // reads exactly like a host that forbids hardware watches. The argument
    // is null because the command takes none.
    auto en = sys::ioctl(fd, PERF_EVENT_IOC_ENABLE, nullptr);
    if (en.failed()) {
        (void)sys::munmap(ring, ring_bytes);
        (void)sys::close(fd);
        detail = "the watch could not be enabled";
        return en.error;
    }

    HardwareWatch w;
    w.fd = fd;
    w.pid = pid;
    w.address = address;
    w.kind = kind;
    w.size = size;
    w.ring = ring;
    w.ring_bytes = ring_bytes;
    w.ring_data_offset = page_size;
    w.ring_tail = 0;
    watches_.push_back(w);
    return 0;
}

int Watchpoints::remove(std::uint64_t address) noexcept {
    for (auto it = watches_.begin(); it != watches_.end(); ++it) {
        if (it->address != address) {
            continue;
        }
        release(*it);
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
        release(*it);
        it = watches_.erase(it);
    }
    return 0;
}

std::vector<int> Watchpoints::fds() const noexcept {
    std::vector<int> out;
    out.reserve(watches_.size());
    for (const auto& w : watches_) {
        if (w.fd >= 0) {
            out.push_back(w.fd);
        }
    }
    return out;
}

std::size_t Watchpoints::read_events(std::vector<WatchEvent>& out) noexcept {
    const std::size_t before = out.size();

    for (auto& w : watches_) {
        if (w.ring == nullptr || w.ring_data_offset == 0) {
            continue;
        }

        const std::size_t head =
            static_cast<std::size_t>(ring_load(w.ring, kRingDataHeadByte));

        const std::size_t data_bytes = w.ring_bytes - w.ring_data_offset;

        // A head behind the tail means the producer overwrote records the
        // consumer had not taken: the ring wrapped. The tail is snapped to
        // the head so the next read starts from live data rather than from
        // a record that no longer exists. The loss is not reported as an
        // event because there is nothing left to report it from.
        if (head < w.ring_tail) {
            ring_store(w.ring, kRingDataTailByte,
                       static_cast<std::uint64_t>(head));
            w.ring_tail = head;
        }

        if (head > w.ring_tail + data_bytes) {
            // A head past the end of the ring is a foreign or corrupt
            // record rather than a wrap. Recovering to the head is the only
            // safe move; a reader that trusted it would read unmapped
            // memory and report whatever was there.
            ring_store(w.ring, kRingDataTailByte,
                       static_cast<std::uint64_t>(head));
            w.ring_tail = head;
            continue;
        }

        const auto* base = static_cast<const std::uint8_t*>(w.ring) +
                           w.ring_data_offset;

        while (w.ring_tail + sizeof(RingHeader) <= head) {
            RingHeader hdr{};
            std::memcpy(&hdr, base + w.ring_tail, sizeof(hdr));

            if (hdr.size < sizeof(RingHeader) ||
                w.ring_tail + hdr.size > head ||
                w.ring_tail + hdr.size > w.ring_data_offset + data_bytes) {
                // A record that does not fit where it claims is truncated or
                // foreign. The tail is snapped to the head to recover,
                // because a reader that trusted the size would walk off the
                // end of the ring and keep going.
                ring_store(w.ring, kRingDataTailByte,
                           static_cast<std::uint64_t>(head));
                w.ring_tail = head;
                break;
            }

            if (hdr.type == kPerfRecordSample &&
                hdr.size >= sizeof(RingHeader) + 16) {
                // The sample layout is the requested fields in the order
                // their bits are numbered: ip first, then addr.
                std::uint64_t ip = 0;
                std::uint64_t addr = 0;
                std::memcpy(&ip, base + w.ring_tail + sizeof(RingHeader), 8);
                std::memcpy(&addr,
                            base + w.ring_tail + sizeof(RingHeader) + 8, 8);

                WatchEvent ev;
                ev.pid = w.pid;
                ev.rip = ip;
                ev.address = addr;
                ev.kind = w.kind;
                // An execute watch reports no data address: the field is
                // still written by the kernel but describes no memory, so
                // it is reported as absent rather than as address zero,
                // which is a legitimate address on some mappings.
                ev.has_address = (w.kind != WatchKind::Execute);
                out.push_back(ev);
            } else if (hdr.type == kPerfRecordLost &&
                       hdr.size >= sizeof(RingHeader) + 16) {
                // A lost record is a count, not a sample: the kernel could
                // not put the access in the ring because it was full. It is
                // accumulated rather than turned into an event, because
                // there is nothing to point an event at -- but it is not
                // dropped either. A tracker that silently loses writes
                // reports a clean target when the truth is that it was not
                // watching closely enough to know.
                std::uint64_t lost = 0;
                std::memcpy(&lost,
                            base + w.ring_tail + sizeof(RingHeader) + 8, 8);
                lost_samples_ += lost;
            }
            // Any other record type advances the tail without producing an
            // event. Its size is what keeps the walk in step, which is why
            // the size is trusted and the payload is not.

            w.ring_tail += hdr.size;
        }

        // Publishing the tail is what frees the records. Until it is
        // written the producer keeps treating them as unread, and a ring
        // that is never acknowledged wraps.
        ring_store(w.ring, kRingDataTailByte,
                   static_cast<std::uint64_t>(w.ring_tail));
    }

    return out.size() - before;
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
