// Observer layer tests.
//
// Two parts of the observer can be tested without a process to trace, and
// they are the two where a mistake produces something that looks right:
// the remote protocol's framing, where a packet that decodes to the wrong
// payload still decodes, and the instruction decoder, where a wrong answer
// about a memory access is indistinguishable from a correct one until
// something acts on it.
//
// The framing cases are the ones a byte-stream reader gets wrong: a packet
// split across two reads, two packets in one read, a checksum that fails,
// and a payload that contains the framing characters.

#include "occ/observer/rsp.h"
#include "occ/observer/session.h"
#include "occ/observer/transport.h"
#include "occ/observer/watchpoint.h"
#include "occ/observer/wx.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

#include "occ/syscall/syscall.h"

using namespace occ::obs;
using namespace occ::sys;

namespace {

int failures = 0;
int checks = 0;

void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL %s\n", what);
    }
}

// ------------------------------------------------------------- packet framing

void test_checksum() {
    // The checksum is the low byte of the sum of the payload bytes. The
    // values here are the ones the protocol's own examples use, which is
    // what makes them checkable against the specification rather than
    // against this implementation.
    check(encode_packet("?") == "$?#3f", "the stop-reason packet checksum");
    check(encode_packet("g") == "$g#67", "the read-registers packet checksum");

    // m4010c0 sums to 0x6d + 0x34 + 0x30 + 0x31 + 0x30 + 0x63 + 0x30 =
    // 0x1c5, and the checksum is the low byte.
    check(encode_packet("m4010c0") == "$m4010c0#c5",
          "the checksum is the low byte of the payload sum");
}

void test_single_packet() {
    PacketDecoder d;
    const std::string packet = encode_packet("g");
    check(d.feed(packet) == 1, "one packet is decoded from one feed");
    check(d.has_packet(), "the packet is available");

    const Packet p = d.take();
    check(p.data == "g", "the payload is the command");
    check(p.checksum_ok, "the checksum is verified");
    check(!d.has_packet(), "the packet is consumed");
}

void test_split_across_reads() {
    // A socket delivers bytes in whatever chunks it has. A decoder that
    // assumed one packet per read works on a local socket and fails on
    // everything else.
    PacketDecoder d;
    const std::string packet = encode_packet("m4010c0");

    for (std::size_t i = 0; i + 1 < packet.size(); ++i) {
        const std::size_t produced = d.feed(packet.data() + i, 1);
        if (i + 1 < packet.size()) {
            check(produced == 0 || i + 1 == packet.size() - 1,
                  "a partial packet produces nothing until it completes");
        }
    }

    // The final byte completes it.
    check(d.feed(packet.data() + packet.size() - 1, 1) == 1,
          "the last byte completes the packet");
    const Packet p = d.take();
    check(p.data == "m4010c0", "the split packet decodes to its payload");
}

void test_two_packets_in_one_read() {
    PacketDecoder d;
    std::string both = encode_packet("g");
    both += encode_packet("m1000,4");

    check(d.feed(both) == 2, "two packets are decoded from one feed");
    const Packet first = d.take();
    const Packet second = d.take();
    check(first.data == "g", "the first payload");
    check(second.data == "m1000,4", "the second payload");
}

void test_bad_checksum() {
    PacketDecoder d;
    std::string packet = encode_packet("g");
    // Corrupt the last checksum digit. 0 and 1 differ in one bit, so the
    // packet stays well-framed and only the checksum is wrong.
    packet[packet.size() - 1] = (packet[packet.size() - 1] == '0') ? '1' : '0';

    check(d.feed(packet) == 1, "a bad-checksum packet is still recognized");
    const Packet p = d.take();
    check(!p.checksum_ok, "the checksum is reported as bad");
    check(p.data.empty(), "a bad packet yields no payload");
    check(d.retransmit_requested(),
          "a bad checksum asks the far end to retransmit");
}

void test_retransmit_request() {
    PacketDecoder d;
    d.feed("-", 1);
    check(d.retransmit_requested(),
          "a lone minus asks for a retransmission");
    d.clear_retransmit();
    check(!d.retransmit_requested(), "the request can be cleared");
}

void test_acknowledgment() {
    // '+' is an acknowledgment and is not a packet.
    PacketDecoder d;
    check(d.feed("+", 1) == 0, "an acknowledgment is not a packet");
    check(!d.has_packet(), "an acknowledgment produces nothing");

    // A '+' followed by a packet decodes the packet.
    PacketDecoder d2;
    std::string data = "+";
    data += encode_packet("g");
    check(d2.feed(data) == 1, "a packet after an acknowledgment is decoded");
    check(d2.take().data == "g", "the payload after an acknowledgment");
}

void test_escaping() {
    // The framing characters and the escape byte have to survive a payload
    // that contains them.
    const std::string payload = "a$b#c}d";
    const std::string packet = encode_packet(payload);

    // The framing characters must not appear unescaped in the body.
    check(packet.substr(1, packet.size() - 4).find('$') == std::string::npos,
          "a literal $ does not appear unescaped in a packet body");
    check(packet.substr(1, packet.size() - 4).find('#') == std::string::npos,
          "a literal # does not appear unescaped in a packet body");

    PacketDecoder d;
    d.feed(packet);
    const Packet p = d.take();
    check(p.data == payload, "the escaped payload round-trips");
    check(p.checksum_ok, "the escaped packet's checksum is over the escaped "
                         "form");
}

void test_escape_helpers() {
    check(escape("abc") == "abc", "a plain string is not escaped");
    check(escape("$") == "}\x04", "the start byte is escaped");
    check(unescape("}\x04") == "$", "the escape is undone");
    check(unescape(escape("a$b#c}d")) == "a$b#c}d", "escape and unescape are "
                                                    "inverse");
}

void test_split_packet_helper() {
    std::string_view head;
    std::string_view tail;

    split_packet("g", head, tail);
    check(head == "g" && tail.empty(), "a packet with no colon");

    split_packet("m4010c0,10", head, tail);
    check(head == "m" && tail == "4010c0,10",
          "the command letter is split from its arguments");

    split_packet("Z0,401000,4", head, tail);
    check(head == "Z" && tail == "0,401000,4", "a breakpoint packet splits");
}

// ------------------------------------------------------------------- hex

void test_hex_round_trip() {
    check(hex_u64_le(0x1122334455667788ULL) == "8877665544332211",
          "an unsigned 64-bit value is least significant byte first");
    check(hex_u64_be(0x1122334455667788ULL) == "1122334455667788",
          "a big-endian value is most significant byte first");
    check(hex_u32_le(0x11223344U) == "44332211",
          "an unsigned 32-bit value is little-endian");
    check(hex_u8(0xab) == "ab", "a byte is two digits");
    check(hex_u8(0x05) == "05", "a byte keeps its leading zero");

    std::uint64_t v = 0;
    check(parse_hex_u64_le("8877665544332211", v), "a 64-bit value parses");
    check(v == 0x1122334455667788ULL,
          "the parsed value has the byte order undone");

    std::uint32_t w = 0;
    check(parse_hex_u32_le("44332211", w), "a 32-bit value parses");
    check(w == 0x11223344U, "the parsed 32-bit value");

    // Too short is a refusal, not a partial read.
    check(!parse_hex_u64_le("8877", v), "a short value is refused");
    check(!parse_hex_u32_le("ab", w), "a short 32-bit value is refused");
}

void test_hex_number() {
    // A protocol number is not a register value. The difference is both the
    // byte order and the width: an offset or a thread id is written most
    // significant digit first with no padding, so reading one with the
    // register-block reader reverses its digits and refuses it for being
    // shorter than eight.
    check(hex_number(0) == "0", "zero is one digit");
    check(hex_number(1) == "1", "a small number has no padding");
    check(hex_number(10) == "a", "ten is hexadecimal a");
    check(hex_number(0x1092) == "1092",
          "a thread id is most significant digit first");
    check(hex_number(0xdeadbeefULL) == "deadbeef",
          "a large number keeps every digit");

    // The encoding is the one GDB sends, so reading it back has to give the
    // original. A round trip is the property the packet handlers rely on.
    for (const std::uint64_t original : {std::uint64_t{0},
                                         std::uint64_t{1},
                                         std::uint64_t{5},
                                         std::uint64_t{11},
                                         std::uint64_t{255},
                                         std::uint64_t{4242},
                                         std::uint64_t{0x7fffffff},
                                         std::uint64_t{0xffffffffffffffffULL}}) {
        std::uint64_t back = 0;
        check(parse_hex_number(hex_number(original), back) && back == original,
              "a protocol number round trips");
    }

    // The signal numbers the protocol writes with two digits must mean the
    // number a person would write, which is the whole reason a leading zero
    // is not significant.
    std::uint64_t sig = 0;
    check(parse_hex_number("05", sig) && sig == 5, "a padded signal means five");
    check(parse_hex_number("0b", sig) && sig == 11, "a padded signal means eleven");

    // Empty is not a number. A field with nothing in it is malformed rather
    // than zero, because zero is spelled "0".
    check(!parse_hex_number("", sig), "an empty number is refused");
    check(!parse_hex_number("zz", sig), "a non-hexadecimal digit is refused");
    check(!parse_hex_number("0x10", sig), "a prefix is refused");
    check(!parse_hex_number(" 10", sig), "a leading space is refused");

    // Seventeen digits cannot fit, and the value that would wrap is refused
    // rather than silently truncated.
    check(!parse_hex_number("00000000000000000", sig),
          "an over-long number is refused");
    check(!parse_hex_number("fffffffffffffffff", sig),
          "an over-long number does not wrap");
}

void test_hex_bytes() {
    std::vector<std::uint8_t> bytes;
    check(parse_hex_bytes("001122ff", bytes), "a byte string parses");
    check(bytes.size() == 4, "the byte count is half the digit count");
    check(bytes[0] == 0x00 && bytes[1] == 0x11 && bytes[2] == 0x22 &&
              bytes[3] == 0xff,
          "the bytes are in order");

    check(!parse_hex_bytes("abc", bytes), "an odd digit count is refused");
    check(!parse_hex_bytes("zz", bytes), "a non-hex digit is refused");
}

void test_stop_reply() {
    const std::string reply = encode_stop_reply(5, 0x401850);
    check(reply[0] == 'T', "the reply is the T form");
    check(reply.substr(1, 2) == "05", "the signal is two hex digits");
    check(reply.find("10:") != std::string::npos,
          "the reply names rip as register 16");
    // rip 0x401850 little-endian is 50 18 40 00 00 00 00 00, so the digit
    // string begins "501840". Reading it big-endian would give "0000...".
    check(reply.find("10:501840") != std::string::npos,
          "rip is encoded least significant byte first");
    check(reply.back() == ';', "the register list is terminated");
}

// ------------------------------------------------- instruction decoding

// Assembles a byte sequence for a decode test.
AccessDecode decode(std::initializer_list<std::uint8_t> bytes) {
    const std::vector<std::uint8_t> v(bytes);
    return decode_access(v.data(), v.size());
}

void test_decode_store() {
    // 48 89 18 is mov [rax], rbx: a write, based on rax, 8 bytes wide.
    const AccessDecode d = decode({0x48, 0x89, 0x18});
    check(d.valid, "a mov with a memory destination decodes");
    check(d.is_write, "the access is a write");
    check(d.base_register == 0, "the base register is rax");
    check(d.width == 8, "the rex.w prefix makes the access eight bytes");
}

void test_decode_load() {
    // 48 8b 18 is mov rbx, [rax]: a read.
    const AccessDecode d = decode({0x48, 0x8b, 0x18});
    check(d.valid, "a mov with a memory source decodes");
    check(!d.is_write, "the access is a read");
    check(d.base_register == 0, "the base register is rax");
    check(d.width == 8, "the read is eight bytes wide");
}

void test_decode_without_rex() {
    // 89 18 is mov [rax], ebx: a four-byte write.
    const AccessDecode d = decode({0x89, 0x18});
    check(d.valid, "a mov without a rex prefix decodes");
    check(d.is_write, "the access is a write");
    check(d.width == 4, "the access is four bytes without rex.w");
}

void test_decode_byte_form() {
    // 88 18 is mov [rax], bl: a single-byte write. The opcode's low bit is
    // what selects the width, and reading it as the 32-bit form would
    // report a four-byte access for a one-byte store.
    const AccessDecode d = decode({0x88, 0x18});
    check(d.valid, "the byte form decodes");
    check(d.is_write, "the byte form is a write");
    check(d.width == 1, "the byte form is one byte wide");
}

void test_decode_displacement() {
    // 48 89 58 10 is mov [rax+0x10], rbx.
    const AccessDecode d = decode({0x48, 0x89, 0x58, 0x10});
    check(d.valid, "a displaced store decodes");
    check(d.base_register == 0, "the base register is rax");
    check(d.displacement == 0x10, "the displacement is read as a signed byte");

    // 48 89 98 00 01 00 00 is mov [rax+0x100], rbx.
    const AccessDecode d2 = decode({0x48, 0x89, 0x98, 0x00, 0x01, 0x00, 0x00});
    check(d2.valid, "a 32-bit displacement decodes");
    check(d2.displacement == 0x100, "the 32-bit displacement is read");
}

void test_decode_register_to_register() {
    // 48 89 d8 is mov rax, rbx: no memory access at all, so it cannot be
    // the instruction that faulted on a watch.
    const AccessDecode d = decode({0x48, 0x89, 0xd8});
    check(!d.valid, "a register-to-register move is not a memory access");
}

void test_decode_non_mov() {
    // A non-mov opcode is reported as undetermined rather than guessed at.
    // A wrong is_write would make the tracker act on an access that never
    // happened.
    check(!decode({0x48, 0x01, 0x18}).valid, "an add is not decoded");
    check(!decode({0xe8, 0x00, 0x00, 0x00, 0x00}).valid, "a call is not "
                                                         "decoded");
    check(!decode({0x90}).valid, "a nop is not decoded");
    check(!decode_access(nullptr, 0).valid, "an empty buffer does not decode");
}

void test_decode_rip_relative() {
    // 48 89 05 10 00 00 00 is mov [rip+0x10], rax. The base is the
    // instruction pointer, which the decode cannot know, so it is reported
    // as absent rather than as a register.
    const AccessDecode d = decode({0x48, 0x89, 0x05, 0x10, 0x00, 0x00, 0x00});
    check(d.valid, "a rip-relative store decodes");
    check(d.is_write, "the rip-relative store is a write");
    check(d.base_register == -1, "a rip-relative access has no base register");
    check(d.displacement == 0x10, "the rip displacement is read");
}

void test_decode_truncated() {
    // A buffer that ends inside the instruction must not be decoded from
    // whatever happens to be past the end.
    check(!decode({0x48, 0x89}).valid, "a truncated instruction is refused");
    check(!decode({0x48, 0x89, 0x98, 0x00}).valid,
          "a truncated displacement is refused");
}

void test_hardware_slots_probe() {
    // The probe reports what the kernel allows. Zero is a valid answer for
    // a kernel without the capability, so the check is that the call is
    // safe and bounded, not that it is non-zero.
    const std::uint32_t slots = Watchpoints::probe_slots();
    check(slots <= 16, "the hardware slot count is bounded");
    std::fprintf(stderr, "  note: hardware debug registers reported: %u\n",
                 slots);
}

void test_watch_alignment() {
    // A watch has to be naturally aligned for its width. Reporting the
    // refusal here names the reason; the kernel's answer is the same but
    // arrives as a bare errno.
    Watchpoints w;
    std::string detail;

    const int rc = w.add(static_cast<int>(::getpid()), 0x1001, WatchKind::Write,
                         WatchSize::Bytes8, detail);
    check(rc != 0, "a misaligned watch is refused");
    check(detail.find("aligned") != std::string::npos,
          "the refusal names the alignment");

    const int rr = w.add(static_cast<int>(::getpid()), 0x1000, WatchKind::Read,
                         WatchSize::Bytes8, detail);
    check(rr != 0, "a read-only watch is refused");
    check(detail.find("read-only") != std::string::npos,
          "the refusal names the read-only limitation");
}

void test_write_watch_width() {
    // A one or two byte write watch is not encodable on x86-64 and the
    // kernel refuses it with a bare EINVAL, which is indistinguishable from
    // a host with no debug registers at all. The refusal therefore has to
    // happen here, where the reason can be named.
    //
    // This check exists because the tracker used to accept these widths and
    // then report the host as incapable of write tracking.
    Watchpoints w;
    std::string detail;

    const int one = w.add(static_cast<int>(::getpid()), 0x2000,
                          WatchKind::Write, WatchSize::Bytes1, detail);
    check(one != 0, "a one byte write watch is refused");
    check(detail.find("four or eight") != std::string::npos,
          "the refusal names the write watch widths");

    detail.clear();
    const int two = w.add(static_cast<int>(::getpid()), 0x2000,
                          WatchKind::Write, WatchSize::Bytes2, detail);
    check(two != 0, "a two byte write watch is refused");

    // A four byte read watch is legal. The restriction is on the write type
    // alone, which is why the check above is specific to it.
    detail.clear();
    const int four = w.add(static_cast<int>(::getpid()), 0x3000,
                           WatchKind::ReadWrite, WatchSize::Bytes4, detail);
    const bool available = Watchpoints::hardware_available();
    check(!available || four == 0,
          "a four byte read-or-write watch is accepted where hardware exists");
    if (four == 0) {
        std::fprintf(stderr,
                     "  note: installed a four byte read-or-write watch on the "
                     "test process at 0x3000\n");
    }
}

void test_wx_chase_prefers_new_mapping() {
    // A decoder's staging buffer does not exist when the process starts, so
    // arming the watches once at startup cannot reach it. The chase is what
    // moves the scarce debug registers onto a region that appeared while
    // the session was running, and this checks that it covers the region it
    // was asked about rather than reporting a coverage it does not have.
    WriteExecuteTracker tracker;
    Watchpoints watches;

    const int pid = static_cast<int>(::getpid());

    // Two regions the tracker is told about before anything is armed. They
    // stand in for the writable data segments a process image carries.
    std::string detail;
    (void)tracker.watch(pid, 0x100000, 24576, detail);
    (void)tracker.watch(pid, 0x200000, 4096, detail);
    check(tracker.region_count() == 2, "both regions are tracked");

    const ArmReport armed = tracker.arm(watches, pid);
    check(armed.watches_installed <= 4,
          "arming never exceeds the hardware slot count");
    check(armed.bytes_covered <= armed.bytes_total,
          "coverage never exceeds the requested bytes");
    std::fprintf(stderr,
                 "  note: armed %zu watches, covered %llu of %llu bytes\n",
                 armed.watches_installed,
                 static_cast<unsigned long long>(armed.bytes_covered),
                 static_cast<unsigned long long>(armed.bytes_total));

    // The shorter region is served first. That is the point of the ordering:
    // four registers of eight bytes cover thirty-two bytes, and a region of
    // thirty-two bytes or less can be covered completely, while a large one
    // can only be sampled. Longest-first would hand every register to the
    // 24576-byte region and guarantee never seeing a write to the 4096-byte
    // one.
    //
    // The assertion is about the allocation, not about a coverage that only
    // a machine with debug registers can produce. A kernel without them arms
    // nothing and reports every region unwatched, and that is a correct
    // answer that must not fail the test.
    if (armed.watches_installed > 0) {
        bool small_covered = false;
        for (const auto& partial : armed.partial) {
            if (partial.target.base == 0x200000) {
                small_covered = true;
            }
        }
        bool small_unwatched = false;
        for (const auto& u : armed.unwatched) {
            if (u.base == 0x200000) {
                small_unwatched = true;
            }
        }
        check(small_covered || small_unwatched,
              "the short region is served before the long one");
        check(armed.watches_installed <= 32 / 8,
              "no more watches are installed than the registers hold");
    }

    // A region that appears later is chased, and the cold watches on regions
    // nothing wrote to are released to make room for it.
    const ArmReport chased = tracker.chase(watches, pid, 0x300000, 4096);
    check(chased.bytes_total == 4096, "the chased region is measured");
    check(chased.bytes_covered <= 4096, "the chased coverage fits the region");
    std::fprintf(stderr,
                 "  note: chased 0x300000, installed %zu watches, covered "
                 "%llu of %llu bytes\n",
                 chased.watches_installed,
                 static_cast<unsigned long long>(chased.bytes_covered),
                 static_cast<unsigned long long>(chased.bytes_total));

    // Chasing the same region again is idempotent rather than a second set of
    // watches on the same address, which would waste a debug register.
    const ArmReport again = tracker.chase(watches, pid, 0x300000, 4096);
    check(again.watches_installed == 0,
          "re-chasing an armed region installs nothing new");

    // A region that has been written is never evicted: it holds the only
    // evidence a transition can still be built from.
    tracker.note_write(pid, 0x300000, 8, 0x401000);
    const std::size_t before = watches.fds().size();
    (void)tracker.evict_cold(watches, 100);
    check(watches.fds().size() == before,
          "a written region keeps its watches through an eviction");

    tracker.release(watches);
    check(watches.fds().empty(), "releasing the tracker frees every watch");
}

void test_wx_permission_transition() {
    // The transition the tracker exists to report: a region written while it
    // was not executable, then made executable. Reported once, because a
    // decoder that writes a byte at a time would otherwise produce one
    // report per byte.
    WriteExecuteTracker tracker;
    std::string detail;
    (void)tracker.watch(static_cast<int>(::getpid()), 0x400000, 4096, detail);

    RegionPerms data;
    data.readable = true;
    data.writable = true;
    data.executable = false;

    tracker.note_write(static_cast<int>(::getpid()), 0x400000, 4, 0x401000);
    check(!tracker.note_permission(static_cast<int>(::getpid()), 0x400000, data),
          "a region that is still not executable is not a transition");

    RegionPerms code = data;
    code.executable = true;
    check(tracker.note_permission(static_cast<int>(::getpid()), 0x400000, code),
          "becoming executable after a write is a transition");
    check(tracker.transitions().size() == 1, "one transition is recorded");
    check(!tracker.note_permission(static_cast<int>(::getpid()), 0x400000, code),
          "the same transition is not reported twice");
    check(tracker.transitions().size() == 1, "still one transition");

    // A region that became executable without being written is not a
    // transition. It is ordinary relro tightening or lazy binding.
    WriteExecuteTracker plain;
    (void)plain.watch(static_cast<int>(::getpid()), 0x500000, 4096, detail);
    check(!plain.note_permission(static_cast<int>(::getpid()), 0x500000, code),
          "an unwritten region becoming executable is not a transition");
    check(plain.transitions().empty(), "and no transition is recorded");
}

// ------------------------------------------------------------- gdb transport

// The transport is tested against a real socket rather than a mock because
// the things that go wrong here are the things a mock cannot show: a port
// that was not really bound, an accept that returned a descriptor which is
// not the one the listener owns, and a read that blocks instead of
// returning. Each of those is a hang or a wrong connection at run time, and
// both are invisible to an interface that pretends the bytes arrived.

void test_listener_binds_loopback() {
    // Port zero asks the kernel for any free port, which is the only way to
    // run this check without colliding with whatever else is on the machine.
    Listener listener;
    std::string error;
    const bool opened = listener.open(0, error);
    if (!opened) {
        // A sandbox that forbids sockets is a legitimate environment; the
        // check is that the refusal is explained rather than that it
        // succeeds.
        std::fprintf(stderr, "  note: cannot bind a socket here: %s\n",
                     error.c_str());
        check(!error.empty(), "a refusal names its reason");
        return;
    }
    check(listener.valid(), "a bound listener is valid");
    check(listener.port() != 0, "the bound port is reported back");
    check(!error.empty() == false, "a successful bind reports no error");

    // A second bind on the same port has to fail, and has to say why. Two
    // sessions silently sharing a port would mean a debugger attached to one
    // of them drives the other.
    Listener second;
    std::string second_error;
    const bool reopened = second.open(listener.port(), second_error);
    if (reopened) {
        // The kernel allows this when the first socket did not request
        // SO_REUSEADDR, which is the default, so a success here means the
        // platform allows it and there is nothing to assert.
        std::fprintf(stderr, "  note: rebinding an in-use port was allowed\n");
    } else {
        check(!second_error.empty(), "a refused bind names its reason");
    }

    listener.close();
    check(!listener.valid(), "a closed listener is not valid");
}

void test_listener_accept_times_out() {
    Listener listener;
    std::string error;
    if (!listener.open(0, error)) {
        std::fprintf(stderr, "  note: skipping accept test: %s\n", error.c_str());
        return;
    }

    // Nobody connects, so the accept has to give up on its own. Without the
    // deadline this call blocks forever, which is what a run with a debugger
    // port would do to a user who started the run and walked away.
    Connection conn;
    const bool got = listener.accept(conn, 50);
    check(!got, "an accept with no peer times out");
    check(!conn.valid(), "a timed-out accept yields no connection");
}

void test_connection_round_trip() {
    Listener listener;
    std::string error;
    if (!listener.open(0, error)) {
        std::fprintf(stderr, "  note: skipping round trip test: %s\n",
                     error.c_str());
        return;
    }

    // A client socket connected to the listener, which stands in for the
    // debugger. It is made by hand rather than through the Listener because a
    // real debugger connects from outside the process.
    const auto client =
        occ::sys::socket(2 /* AF_INET */, 1 /* SOCK_STREAM */, 0);
    if (client.failed()) {
        std::fprintf(stderr, "  note: cannot create a client socket\n");
        return;
    }

    // 127.0.0.1 in network byte order, and the port the listener reported.
    struct SockAddrIn {
        std::uint16_t family;
        std::uint16_t port;
        std::uint32_t addr;
        std::uint8_t zero[8];
    } addr{};
    // The family is a host-order number. Writing it in network order, as the
    // other two fields are, is a mistake the kernel answers with an
    // address-family error that reads like a policy refusal rather than a
    // bug -- and it is why a listener written this way binds nowhere.
    addr.family = 2; // AF_INET
    addr.port = static_cast<std::uint16_t>((listener.port() >> 8) |
                                            (listener.port() << 8));
    addr.addr = 0x0100007f; // 127.0.0.1

    const auto connected = occ::sys::connect(static_cast<int>(client.value),
                                             &addr, sizeof(addr));
    if (connected.failed()) {
        std::fprintf(stderr, "  note: connect failed, skipping\n");
        (void)occ::sys::close(static_cast<int>(client.value));
        return;
    }

    Connection conn;
    check(listener.accept(conn, 1000), "a waiting accept returns a connection");
    check(conn.valid(), "the accepted connection is valid");

    // The port the listener reports must be the port that was actually
    // bound, or a person told to connect to it connects to nothing.
    check(conn.fd() >= 0, "the connection owns a descriptor");

    // A packet written by the client arrives whole, even when it arrives in
    // pieces: the framing is what makes a byte stream into messages.
    const std::string request = encode_packet("qSupported");
    const std::size_t half = request.size() / 2;
    check(occ::sys::write(static_cast<int>(client.value), request.data(), half).ok(),
          "the first half of a packet is written");
    check(conn.wait_readable(1000), "the connection becomes readable");
    check(conn.pump(), "a partial packet is read");
    check(!conn.has_packet(), "half a packet is not a packet");
    check(occ::sys::write(static_cast<int>(client.value), request.data() + half,
                     request.size() - half)
              .ok(),
          "the second half of a packet is written");
    check(conn.wait_readable(1000), "the connection is readable again");
    check(conn.pump(), "the rest of the packet is read");
    check(conn.has_packet(), "a whole packet is now available");

    std::string packet;
    check(conn.take_packet(packet), "the packet is taken");
    check(packet == "qSupported", "the payload survives the round trip");
    check(!conn.has_packet(), "the queue is empty again");

    // A reply goes back the same way. The framing is applied here rather than
    // by the caller, because a reply sent without it is a byte the debugger
    // discards.
    check(conn.send_packet("OK"), "a reply is sent");
    char buffer[64] = {};
    const auto got = occ::sys::read(static_cast<int>(client.value), buffer,
                               sizeof(buffer));
    check(got.ok(), "the client read the reply");
    if (got.ok()) {
        const std::string_view wire(buffer, static_cast<std::size_t>(got.value));
        check(wire == encode_packet("OK"), "the reply arrives framed");
    }

    // A closed peer is how a debugger that quit is noticed. Reporting it as
    // an error would make a normal disconnect look like a fault.
    (void)occ::sys::close(static_cast<int>(client.value));
    check(conn.wait_readable(1000), "the close is visible as readable");
    check(!conn.pump(), "a closed peer ends the connection");

    conn.close();
    check(!conn.valid(), "a closed connection is not valid");
}

void test_connection_drops_corrupt_packet() {
    // A packet whose checksum is wrong has to be refused rather than acted
    // on. A command read out of a corrupt packet is a command nobody sent,
    // and the worst thing this stub could do with one is resume a process or
    // write its memory on the strength of a transmission error.
    PacketDecoder decoder;
    // 'g' with a checksum that cannot match.
    decoder.feed("$g#00", 5);
    check(decoder.has_packet(), "a corrupt packet is still framed");
    Packet p = decoder.take();
    check(!p.checksum_ok, "a bad checksum is reported");

    // A failed checksum also asks for a retransmission, and that is what the
    // sender does about it: the protocol has no other way to recover a
    // packet that arrived damaged. The retransmission carries the same bytes
    // rather than a rebuilt packet, because the sender's copy is the only one
    // known to be what it meant to send.
    PacketDecoder again;
    again.feed("$g#00", 5);
    check(again.has_packet(), "the corrupt packet is framed");
    (void)again.take();
    check(again.retransmit_requested(),
          "a failed checksum asks for a retransmission");
    // The packet to resend is the last one that arrived intact, not the
    // damaged one: replying with damaged bytes would reproduce the same
    // error. Nothing intact has arrived here, so there is nothing to resend.
    check(again.last_packet().empty(),
          "a damaged packet is not offered for retransmission");
    again.clear_retransmit();
    check(!again.retransmit_requested(), "the request is cleared once handled");

    // A packet that arrives intact does not ask for one, and this is the
    // property that keeps a good connection from retransmitting forever.
    PacketDecoder good;
    good.feed(encode_packet("g"));
    check(good.has_packet(), "an intact packet is framed");
    const Packet ok = good.take();
    check(ok.checksum_ok, "an intact packet passes its checksum");
    check(!good.retransmit_requested(),
          "an intact packet asks for no retransmission");
    check(good.last_packet() == encode_packet("g"),
          "an intact packet is the one to resend");
}

// ------------------------------------------------------- target description

// Counts the regnum attributes in the description. The count is what has to
// agree with the 'g' reply, and it is checked against the wire format rather
// than against a constant so that a register added to one place and not the
// other is caught here instead of by a debugger showing shifted values.
std::size_t count_registers(std::string_view xml) {
    std::size_t seen = 0;
    std::size_t pos = 0;
    while ((pos = xml.find("regnum=\"", pos)) != std::string_view::npos) {
        pos += 8;
        ++seen;
    }
    return seen;
}

void test_target_description() {
    const std::string_view xml = target_description();
    check(!xml.empty(), "the target description is not empty");
    check(xml.find("i386:x86-64") != std::string_view::npos,
          "the description names the x86-64 architecture");
    check(count_registers(xml) == 27,
          "the description carries all twenty-seven registers");

    // The program counter is the register a debugger reads first, and its
    // number is load-bearing: it is the index the 'g' reply and the 'p'
    // packet both use. A description that renumbered it would make every
    // register after it wrong too.
    check(xml.find("name=\"rip\" bitsize=\"64\" type=\"code_ptr\" regnum=\"16\"") !=
              std::string_view::npos,
          "rip is register sixteen at sixty-four bits");

    // eflags and the segment selectors are thirty-two bit. Describing them
    // as sixty-four makes the 'g' reply longer than the debugger expects and
    // every field after them shifts.
    check(xml.find("name=\"eflags\" bitsize=\"32\" regnum=\"17\"") !=
              std::string_view::npos,
          "eflags is thirty-two bits");

    // orig_rax and the two base registers are not architectural state a
    // debugger can set, but a debugger that reads them gets a coherent view
    // of where the thread is. Their presence is why the count is 27 rather
    // than the 24 a plain x86-64 layout would give.
    check(xml.find("name=\"orig_rax\"") != std::string_view::npos,
          "orig_rax is described");
    check(xml.find("name=\"fs_base\"") != std::string_view::npos,
          "fs_base is described");
    check(xml.find("name=\"gs_base\"") != std::string_view::npos,
          "gs_base is described");
}

void test_serve_target_description() {
    const std::string_view xml = target_description();

    // A request for an annex that does not exist answers "l", which is how
    // the protocol says there is nothing here. Answering an error instead
    // makes a debugger that probes for optional files treat the stub as
    // broken.
    check(serve_target_description("features:read:no-such.xml:0,100") == "l",
          "an unknown annex answers l");

    // Malformed packets answer empty, the protocol's "not supported".
    check(serve_target_description("").empty(),
          "an empty request is refused");
    check(serve_target_description("target.xml").empty(),
          "a request without a range is refused");
    check(serve_target_description("target.xml:0").empty(),
          "a range without a length is refused");
    check(serve_target_description("target.xml:zz,10").empty(),
          "a non-hexadecimal offset is refused");

    // A zero-length read answers "l": there is nothing to send and saying
    // "m" would invite the debugger to ask again forever.
    check(serve_target_description("target.xml:0,0") == "l",
          "a zero length answers l");

    // An offset exactly at the end is what the last chunked request looks
    // like, so it answers "l" rather than being treated as out of range.
    const std::string at_end =
        serve_target_description("target.xml:" + hex_number(xml.size()) + ",10");
    check(at_end == "l", "an offset at the end answers l");

    // An offset past the end is out of range, and also answers "l".
    const std::string past_end = serve_target_description(
        "target.xml:" + hex_number(xml.size() + 4096) + ",10");
    check(past_end == "l", "an offset past the end answers l");

    // A short read in the middle is prefixed 'm', meaning more follows. The
    // range is hexadecimal, so the length is written with hex_number rather
    // than spelled as a decimal count -- "64" is one hundred bytes, which is
    // the mistake that makes a stub appear to work in a hand-written test
    // and fail against a real debugger.
    const std::string first =
        serve_target_description("target.xml:0," + hex_number(64));
    check(first.size() == 65, "a chunk is the requested length plus a marker");
    check(!first.empty() && first[0] == 'm', "a partial chunk is marked m");
    check(std::string_view(first).substr(1) == xml.substr(0, 64),
          "the chunk is the document from the requested offset");

    // The length is read as hexadecimal, not as a decimal digit count. This
    // is the property the check above depends on and the one a debugger
    // depends on: it sends "400" meaning one kilobyte and expects a
    // thousand and twenty eight bytes back.
    check(serve_target_description("target.xml:0,10").size() == 17,
          "a length is read as hexadecimal");

    // A read that reaches the end is prefixed 'l', meaning the document is
    // complete. Always answering 'm' is what makes a debugger loop.
    const std::string whole =
        serve_target_description("target.xml:0," + hex_number(xml.size()));
    check(whole.size() == xml.size() + 1, "a full read covers the document");
    check(!whole.empty() && whole[0] == 'l', "a complete chunk is marked l");
    check(std::string_view(whole).substr(1) == xml,
          "a full read is the whole document");

    // A read that asks for more than remains is clamped and marked 'l' rather
    // than being reported as a short read.
    const std::string over =
        serve_target_description("target.xml:0," + hex_number(xml.size() + 1000));
    check(over.size() == xml.size() + 1, "an over-long read is clamped");
    check(over[0] == 'l', "a clamped read is marked l");

    // Reassembling the chunks the way GDB does must reproduce the document
    // exactly. This is the property that matters and the one no single
    // request above establishes.
    std::string rebuilt;
    std::size_t offset = 0;
    for (int guard = 0; guard < 64; ++guard) {
        const std::string chunk =
            serve_target_description("target.xml:" + hex_number(offset) + "," + hex_number(200));
        if (chunk.empty()) {
            check(false, "a chunk request in the middle returned empty");
            break;
        }
        // The marker is the first character and the data follows it, so the
        // last chunk is "l" plus the tail of the document rather than the
        // one-character "l" that means "no such object". Appending from the
        // second character either way is what makes the reassembly
        // independent of where the chunks happened to be cut.
        const bool last = chunk[0] == 'l';
        rebuilt.append(chunk, 1, std::string::npos);
        offset += chunk.size() - 1;
        if (last) {
            break;
        }
    }
    check(rebuilt == xml, "the chunked reads reassemble into the document");
    check(offset == xml.size(), "the reassembly consumed the whole document");
}

void test_parse_vcont() {
    // The capability query. A debugger sends this to decide whether to use
    // vCont at all, and answers with the actions it may then send. Advertising
    // an action that is not implemented is worse than advertising none.
    bool consume = true;
    bool step = true;
    int signal = 0;
    const std::string q = parse_vcont("", 4242, consume, step, signal);
    check(q == "c:C;s:S;r:t",
          "the capability query lists the supported actions");
    check(!consume, "the capability query asks for no resume");
    check(!step, "the capability query sets no step");
    check(signal == 0, "the capability query carries no signal");

    // A continue consumes without stepping.
    consume = false;
    step = true;
    signal = 0;
    std::string reply = parse_vcont(";c", 4242, consume, step, signal);
    check(consume, "a continue consumes");
    check(!step, "a continue does not step");
    check(signal == 0, "a continue carries no signal");
    check(reply == "T" + hex_number(4242), "the reply names the thread");

    // A step consumes and steps.
    consume = false;
    step = false;
    signal = 0;
    (void)parse_vcont(";s", 7, consume, step, signal);
    check(consume, "a step consumes");
    check(step, "a step steps");
    check(signal == 0, "a step carries no signal");

    // The capitalised forms carry the signal to deliver, as a two digit hex
    // number. Signal 5 is SIGTRAP and signal 11 is SIGSEGV, both of which a
    // debugger sends when it wants the target to take a fault deliberately.
    consume = false;
    step = false;
    signal = 0;
    (void)parse_vcont(";C05", 9, consume, step, signal);
    check(consume && !step, "a continue with a signal consumes without stepping");
    check(signal == 5, "the signal is carried through");

    consume = false;
    step = false;
    signal = 0;
    (void)parse_vcont(";S0b", 9, consume, step, signal);
    check(consume && step, "a step with a signal steps");
    check(signal == 11, "the signal is carried through a step");

    // A lowercase verb ignores a trailing condition. "c:thread" selects a
    // thread, not a signal, and reading a thread number as a signal would
    // deliver a signal the debugger never asked for.
    consume = false;
    step = false;
    signal = 0;
    (void)parse_vcont(";c:2", 9, consume, step, signal);
    check(consume && !step, "a threaded continue consumes");
    check(signal == 0, "a thread number is not read as a signal");

    // The first resume action wins. A debugger sends one per thread, and
    // acting on the last would resume a thread the debugger listed first.
    consume = false;
    step = false;
    signal = 0;
    (void)parse_vcont(";s;c", 9, consume, step, signal);
    check(consume && step, "the first action decides the step");

    consume = false;
    step = false;
    signal = 0;
    (void)parse_vcont(";r;c", 9, consume, step, signal);
    check(consume && !step, "an unsupported leading action is skipped");

    // Nothing resumable in the packet means no resume, which the protocol
    // defines as an empty answer.
    consume = true;
    step = true;
    signal = 7;
    check(parse_vcont(";t", 9, consume, step, signal).empty(),
          "a stop-only packet asks for no resume");
    check(!consume, "a stop-only packet does not consume");
    check(!step, "a stop-only packet does not step");
    check(signal == 0, "a stop-only packet clears the signal");
}

} // namespace

int main() {
    test_checksum();
    test_single_packet();
    test_split_across_reads();
    test_two_packets_in_one_read();
    test_bad_checksum();
    test_retransmit_request();
    test_acknowledgment();
    test_escaping();
    test_escape_helpers();
    test_split_packet_helper();
    test_hex_round_trip();
    test_hex_number();
    test_hex_bytes();
    test_stop_reply();
    test_decode_store();
    test_decode_load();
    test_decode_without_rex();
    test_decode_byte_form();
    test_decode_displacement();
    test_decode_register_to_register();
    test_decode_non_mov();
    test_decode_rip_relative();
    test_decode_truncated();
    test_hardware_slots_probe();
    test_watch_alignment();
    test_write_watch_width();
    test_wx_chase_prefers_new_mapping();
    test_wx_permission_transition();
    test_listener_binds_loopback();
    test_listener_accept_times_out();
    test_connection_round_trip();
    test_connection_drops_corrupt_packet();
    test_target_description();
    test_serve_target_description();
    test_parse_vcont();

    std::fprintf(stderr, "%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
