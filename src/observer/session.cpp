#include "occ/observer/session.h"

#include "occ/syscall/errno.h"
#include "occ/util/string.h"

#include <cerrno>

#include <poll.h>
#include <unistd.h>

namespace occ::obs {

namespace {

constexpr int kSigstop = 19;

// The syscall numbers the tracker has to recognise by hand. They are
// restated from the kernel's x86-64 table because there is no syscall that
// reports "what syscall is this" -- the tracee is stopped at the
// instruction, and reading the number out of the register block is the only
// way to know.
//
// Only the ones that change what memory is mapped or what it may be used
// for are listed. A transition to executable requires one of these, and
// nothing else in the syscall table can produce one.
constexpr std::uint64_t kSysMmap = 9;
constexpr std::uint64_t kSysMprotect = 10;
constexpr std::uint64_t kSysMunmap = 11;
constexpr std::uint64_t kSysMremap = 25;

} // namespace

namespace {

// The number of registers the 'g' reply carries. It has to agree with the
// target description below, and a static_assert in fill_gdb_registers is
// what keeps the two from drifting apart. Declared here so both the fill and
// the parse paths can size themselves from one number.
constexpr std::size_t kGdbRegisterCount = 27;

// The register GDB treats as the program counter, per the description below.
constexpr std::size_t kPcRegnum = 16;

// One register out of the block, by the index the target description gives.
//
// The indexing is shared by the 'g', 'p' and 'G' paths so a register cannot
// be written at one index and read at another. That is not a theoretical
// concern: this file did exactly that once, describing twenty-seven
// registers and supplying twenty-four, and every register after the gap
// displayed under the wrong name with a plausible-looking value.
std::uint64_t gdb_register(const Registers& r, std::size_t index) noexcept {
    switch (index) {
    case 0: return r.rax;
    case 1: return r.rbx;
    case 2: return r.rcx;
    case 3: return r.rdx;
    case 4: return r.rsi;
    case 5: return r.rdi;
    case 6: return r.rbp;
    case 7: return r.rsp;
    case 8: return r.r8;
    case 9: return r.r9;
    case 10: return r.r10;
    case 11: return r.r11;
    case 12: return r.r12;
    case 13: return r.r13;
    case 14: return r.r14;
    case 15: return r.r15;
    case 16: return r.rip;
    case 17: return r.eflags;
    case 18: return r.cs;
    case 19: return r.ss;
    case 20: return r.ds;
    case 21: return r.es;
    case 22: return r.fs;
    case 23: return r.gs;
    case 24: return r.orig_rax;
    case 25: return r.fs_base;
    case 26: return r.gs_base;
    default: return 0;
    }
}

// Writes the writable registers of the block back. The three the kernel does
// not accept from SETREGS on this architecture are ignored: the description
// names them because GDB needs a coherent register list, not because the
// debugger can change them, and silently writing zeros into a thread
// selector would be far worse than ignoring the request.
void set_gdb_register(Registers& r, const std::uint64_t* values) noexcept {
    r.rax = values[0];
    r.rbx = values[1];
    r.rcx = values[2];
    r.rdx = values[3];
    r.rsi = values[4];
    r.rdi = values[5];
    r.rbp = values[6];
    r.rsp = values[7];
    r.r8 = values[8];
    r.r9 = values[9];
    r.r10 = values[10];
    r.r11 = values[11];
    r.r12 = values[12];
    r.r13 = values[13];
    r.r14 = values[14];
    r.r15 = values[15];
    r.rip = values[16];
    r.eflags = values[17];
    r.cs = values[18];
    r.ss = values[19];
    r.ds = values[20];
    r.es = values[21];
    r.fs = values[22];
    r.gs = values[23];
    // values[24..26] are orig_rax, fs_base and gs_base. Not writable here.
}

// The register block GDB expects for x86-64, in the order the target
// description below declares. The two lists have to agree: the document
// tells GDB how wide each register is, and this function supplies the bytes
// in that same order. A register described but not supplied leaves the block
// short, and every register after the gap reads as the wrong value -- which
// looks like a target that corrupted its own state rather than a stub that
// filled in too few.
//
// The four segment selectors are 32 bits wide in the description and are
// sent as eight bytes here, because the register block is a fixed-width
// array in the protocol: GDB masks the surplus rather than misreading
// everything after it, and sending four bytes for them would shift the rest.
void fill_gdb_registers(const Registers& r, std::string& out) noexcept {
    out.clear();
    out.reserve(kGdbRegisterCount * 16);
    const std::uint64_t order[] = {
        r.rax, r.rbx, r.rcx, r.rdx, r.rsi, r.rdi, r.rbp, r.rsp,
        r.r8,  r.r9,  r.r10, r.r11, r.r12, r.r13, r.r14, r.r15,
        r.rip, r.eflags, r.cs, r.ss, r.ds, r.es, r.fs, r.gs,
        // The three the description adds past the classic block. orig_rax
        // is the syscall the thread is stopped in, which is the most
        // valuable thing this stub can show a debugger that asked, and the
        // two base registers are the thread and the group selector.
        r.orig_rax, r.fs_base, r.gs_base,
    };
    static_assert(sizeof(order) / sizeof(order[0]) == kGdbRegisterCount,
                  "the register block and the target description disagree on "
                  "how many registers x86-64 has");
    (void)order;
    for (std::size_t i = 0; i < kGdbRegisterCount; ++i) {
        out += hex_u64_le(gdb_register(r, i));
    }
}

// The target description GDB reads through qXfer:features:read.
//
// The register block has to be described here rather than left to the
// protocol's defaults, because the defaults describe a 32-bit i386 target:
// GDB sizes every register from this document, and a register it believes is
// four bytes wide makes every offset in the 'g' packet wrong. The order in
// this list is the order of the 'g' reply, which is also the order
// fill_gdb_registers writes, and the two are cross-checked by the register
// count below.
//
// The names are the ones GDB uses for x86-64. Two are not register names at
// all -- orig_rax and fs_base are the syscall the thread is in and the
// thread pointer -- and they are included because a debugger reading them
// gets a coherent view rather than a shifted one.
//
// The vector size is in bytes and is the x86-64 requirement.
constexpr std::string_view kTargetXml = R"(<?xml version="1.0"?>
<!DOCTYPE target SYSTEM "gdb-target.dtd">
<target version="1.0">
  <architecture>i386:x86-64</architecture>
  <feature name="org.gnu.gdb.i386.core">
    <reg name="rax" bitsize="64" type="int64" regnum="0"/>
    <reg name="rbx" bitsize="64" type="int64" regnum="1"/>
    <reg name="rcx" bitsize="64" type="int64" regnum="2"/>
    <reg name="rdx" bitsize="64" type="int64" regnum="3"/>
    <reg name="rsi" bitsize="64" type="int64" regnum="4"/>
    <reg name="rdi" bitsize="64" type="int64" regnum="5"/>
    <reg name="rbp" bitsize="64" type="data_ptr" regnum="6"/>
    <reg name="rsp" bitsize="64" type="data_ptr" regnum="7"/>
    <reg name="r8" bitsize="64" type="int64" regnum="8"/>
    <reg name="r9" bitsize="64" type="int64" regnum="9"/>
    <reg name="r10" bitsize="64" type="int64" regnum="10"/>
    <reg name="r11" bitsize="64" type="int64" regnum="11"/>
    <reg name="r12" bitsize="64" type="int64" regnum="12"/>
    <reg name="r13" bitsize="64" type="int64" regnum="13"/>
    <reg name="r14" bitsize="64" type="int64" regnum="14"/>
    <reg name="r15" bitsize="64" type="int64" regnum="15"/>
    <reg name="rip" bitsize="64" type="code_ptr" regnum="16"/>
    <reg name="eflags" bitsize="32" regnum="17"/>
    <reg name="cs" bitsize="32" regnum="18"/>
    <reg name="ss" bitsize="32" regnum="19"/>
    <reg name="ds" bitsize="32" regnum="20"/>
    <reg name="es" bitsize="32" regnum="21"/>
    <reg name="fs" bitsize="32" regnum="22"/>
    <reg name="gs" bitsize="32" regnum="23"/>
    <reg name="orig_rax" bitsize="64" type="int64" regnum="24"/>
    <reg name="fs_base" bitsize="64" type="int64" regnum="25"/>
    <reg name="gs_base" bitsize="64" type="int64" regnum="26"/>
  </feature>
</target>
)";

} // namespace

std::string DebugServer::handle(std::string_view packet) noexcept {
    if (packet.empty()) {
        return {};
    }

    const char cmd = packet[0];
    const std::string_view args = packet.size() > 1 ? packet.substr(1)
                                                    : std::string_view{};

    switch (cmd) {
    case '?': {
        // Why did we stop. The stop reply carries the program counter,
        // because that is the register a debugger needs to decide what the
        // stop was and it should not have to ask for it separately. Reading
        // it here rather than answering with zero matters: a debugger that
        // resumes from address zero because this returned zero is looking
        // at a target that has already faulted.
        Registers now{};
        std::uint64_t pc = 0;
        if (tracer_->get_regs(pid_, now).ok()) {
            pc = gdb_register(now, kPcRegnum);
        }
        return encode_stop_reply(gdb_signal_for_trap(), pc);
    }

    case 'v':
        // Only vCont is implemented. A 'v' packet for anything else answers
        // empty, which is what the protocol defines as unsupported, and the
        // debugger falls back to the single-threaded packets it knows.
        if (starts_with(args, "Cont")) {
            return handle_vcont(args.substr(std::string_view{"Cont"}.size()));
        }
        return {};

    case 'q':
        return handle_query(args);

    case 'g':
        return handle_read_registers();

    case 'G':
        return handle_write_registers(args);

    case 'p': {
        // Read one register by number. The numbering is the one the target
        // description gives, which is the same one the 'g' reply is built
        // from -- reading a register by a different index than it was
        // written by is how a debugger ends up displaying one register's
        // value under another's name.
        //
        // The number arrives as a variable-width hexadecimal value, so "p0"
        // asks for the first register and "p1a" for the twenty-sixth. It is
        // read as a protocol number rather than as a register block: the
        // latter wants eight digits and would refuse both.
        std::uint64_t which = 0;
        if (!parse_hex_number(args, which)) {
            return {};
        }
        if (which >= kGdbRegisterCount) {
            return {};
        }
        Registers r{};
        if (tracer_->get_regs(pid_, r).failed()) {
            return "E01";
        }
        return hex_u64_le(gdb_register(r, static_cast<std::size_t>(which)));
    }

    case 'm':
        return handle_read_memory(args);

    case 'M':
        return handle_write_memory(args);

    case 'Z':
    case 'z':
        return handle_breakpoint(packet);

    case 'H':
        return handle_thread();

    case 'c':
        // Continue, optionally with a signal to deliver. The session does
        // not resume inside the handler: the loop owns the tracee and has
        // to be the one that decides when it runs.
        resume_requested_ = true;
        step_ = false;
        resume_signal_ = 0;
        if (!args.empty()) {
            std::uint64_t sig = 0;
            // A signal number is a variable-width hexadecimal value, which
            // the protocol conventionally writes with two digits.
            if (parse_hex_number(args, sig)) {
                resume_signal_ = static_cast<int>(sig);
            }
        }
        return {};

    case 's':
        resume_requested_ = true;
        step_ = true;
        resume_signal_ = 0;
        return {};

    case 'D':
        detached_ = true;
        return "OK";

    case 'k':
        // A kill request. The session reports it as a detach and lets the
        // caller decide; killing a process the user did not ask to kill is
        // not something a debugger stub does on its own.
        detached_ = true;
        return {};

    case '#':
        return {};

    default:
        // Anything else is unsupported, and the empty response is the
        // protocol's way of saying so.
        return {};
    }
}

std::string DebugServer::handle_query(std::string_view kind) noexcept {
    if (starts_with(kind, "Supported")) {
        // The features this stub implements.
        //
        // The list is a contract and it is kept honest in both directions.
        // Claiming a feature that is not implemented makes the debugger take
        // a code path that then fails in a way that looks like a bug in the
        // debugger; omitting one that is implemented makes the debugger fall
        // back to a slower or less precise path it did not have to take.
        // Every entry below names the packet that implements it.
        //
        // vContSupported+ is here because handle_vcont() implements the
        // actions, and qXfer:features:read+ because
        // handle_qxfer_features() serves the target description.
        return "qXfer:features:read+;swbreak+;hwbreak+;vContSupported+";
    }
    if (starts_with(kind, "Xfer:features:read:")) {
        return handle_qxfer_features(
            kind.substr(std::string_view{"Xfer:features:read:"}.size()));
    }
    if (starts_with(kind, "Attached")) {
        // Already attached, because the session was created from a process
        // it owns rather than by asking the target to spawn one.
        return "1";
    }
    if (starts_with(kind, "C")) {
        // The current thread. This stub observes one process, so the answer
        // is always that process.
        return "QC" + hex_number(static_cast<std::uint64_t>(pid_));
    }
    if (starts_with(kind, "fThreadInfo")) {
        return "m" + hex_number(static_cast<std::uint64_t>(pid_));
    }
    if (starts_with(kind, "sThreadInfo")) {
        return "l";
    }
    if (starts_with(kind, "TStatus")) {
        return "T0";
    }
    // qSymbol and every other query this stub does not answer fall through
    // to an empty response, which the protocol defines as "not supported".
    return {};
}

std::string_view target_description() noexcept {
    return kTargetXml;
}

std::string serve_target_description(std::string_view args) noexcept {
    // The packet is "qXfer:features:read:ANNEX:OFFSET,LENGTH" and the
    // dispatcher has already removed the "qXfer:features:read:" prefix, so
    // what arrives here is the annex and the range, separated by one colon.
    //
    // The prefix that comes off is fixed text naming this one object, so the
    // annex is whatever is left before the colon. Only "target.xml" exists;
    // a request for any other annex answers "l", the protocol's way of
    // saying there is no such object, so a debugger that asks for a file it
    // might find elsewhere stops asking rather than treating the absence as
    // an error.
    const std::size_t colon = args.find(':');
    if (colon == std::string_view::npos) {
        return {};
    }
    const std::string_view annex = args.substr(0, colon);
    const std::string_view range = args.substr(colon + 1);

    if (annex != "target.xml") {
        return "l";
    }

    const std::size_t comma = range.find(',');
    if (comma == std::string_view::npos) {
        return {};
    }
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
    if (!parse_hex_number(range.substr(0, comma), offset) ||
        !parse_hex_number(range.substr(comma + 1), length)) {
        return {};
    }

    const std::string_view xml = kTargetXml;

    // An offset at or past the end answers 'l' rather than an error. GDB
    // fetches the document in chunks and the final request legitimately
    // lands exactly at the end, which is not the same as running off it.
    if (offset >= xml.size() || length == 0) {
        return "l";
    }

    // The subtraction cannot underflow: offset is below xml.size() and both
    // are the same type, so available is at least one and take is at most
    // that. The clamp is what keeps a length larger than the remainder from
    // asking substr for more than the document holds.
    const std::size_t available = xml.size() - static_cast<std::size_t>(offset);
    const std::size_t take = length < available
                                 ? static_cast<std::size_t>(length)
                                 : available;

    // The leading 'm' or 'l' says whether more remains. Deciding it here
    // rather than always answering 'm' is what stops GDB from asking for one
    // more chunk forever.
    std::string out;
    out.reserve(take + 1);
    out += (take < available) ? 'm' : 'l';
    // The document is ASCII, so its bytes are already the wire form. The
    // escaping a binary packet needs is applied by encode_packet.
    out.append(xml.substr(static_cast<std::size_t>(offset), take));
    return out;
}

std::string parse_vcont(std::string_view args, int pid, bool& consume,
                        bool& step, int& signal) noexcept {
    // vCont is the multiplexed form of c, s, C and S. The packet is
    // "vCont[;action[:thread-id]]...", where an action is a letter
    // optionally followed by a condition, and an absent thread-id means every
    // thread. The older packets remain the fallback for a debugger that does
    // not use it, so implementing vCont does not replace them.
    //
    // "vCont?" asks which actions are supported. The answer is the list of
    // them, separated by colons, exactly as it would appear in a request,
    // and it is the answer that lets a debugger choose vCont at all.
    //
    // The outputs are cleared before the query is answered rather than
    // after. A debugger sends the query before any action, and a stub that
    // answered it while leaving the caller's flags as they were would resume
    // with whatever an earlier packet happened to set: a debugger that asked
    // what is supported and then asked to continue would single-step.
    consume = false;
    step = false;
    signal = 0;

    if (args.empty()) {
        return "c:C;s:S;r:t";
    }

    std::size_t pos = 0;
    while (pos <= args.size()) {
        const std::size_t next = args.find(';', pos);
        const std::string_view action =
            args.substr(pos, next == std::string_view::npos
                                 ? std::string_view::npos
                                 : next - pos);
        if (!action.empty()) {
            // An action is one letter followed by an optional signal, then an
            // optional thread id: "c", "C05", "s:2", "S0b:2". The signal
            // follows the letter directly and is hexadecimal, which is why it
            // is split off by taking the single leading letter rather than by
            // looking for a separator -- reading "C05" as a whole word finds
            // neither a known verb nor a signal, and the action is then
            // silently dropped and the process never resumes.
            const std::string_view verb = action.substr(0, 1);

            if (verb == "c" || verb == "C" || verb == "s" || verb == "S") {
                step = (verb == "s" || verb == "S");
                consume = true;
                signal = 0;
                // Only the capitalised forms carry a signal. The lowercase
                // "c" and "s" resume without one, and a digit after them
                // would be the thread id rather than a signal, so nothing
                // here is read.
                if (verb == "C" || verb == "S") {
                    const std::size_t sig_begin = 1;
                    std::size_t sig_end = action.find(':');
                    if (sig_end == std::string_view::npos) {
                        sig_end = action.size();
                    }
                    std::uint64_t sig = 0;
                    if (sig_end > sig_begin &&
                        parse_hex_number(action.substr(sig_begin, sig_end - sig_begin),
                                         sig)) {
                        signal = static_cast<int>(sig);
                    }
                }
                break;
            }
            // "r" starts and "t" stops. A remote target may support them, but
            // this stub owns the process it traces and the session loop is
            // the only thing that resumes it, so neither can be honoured
            // here and neither is advertised by vCont?.
        }

        if (next == std::string_view::npos) {
            break;
        }
        pos = next + 1;
    }

    if (!consume) {
        return {};
    }
    // The reply is the thread that was selected, which is the only thread
    // there is. The session does not resume inside the packet layer: the
    // loop owns the tracee and decides when it runs.
    return "T" + hex_number(static_cast<std::uint64_t>(pid));
}


std::string DebugServer::handle_qxfer_features(std::string_view args) noexcept {
    return serve_target_description(args);
}

std::string DebugServer::handle_vcont(std::string_view args) noexcept {
    // The reply is returned unchanged whether or not a resume was requested:
    // "vCont?" produces an answer with no resume behind it, and a resume the
    // loop has not yet performed still answers with the thread it named. The
    // flags come back through the same references the loop reads, so the
    // packet layer does not decide when the tracee runs.
    return parse_vcont(args, pid_, resume_requested_, step_, resume_signal_);
}

std::string DebugServer::handle_read_registers() noexcept {
    Registers r{};
    auto res = tracer_->get_regs(pid_, r);
    if (res.failed()) {
        return "E01";
    }
    std::string out;
    fill_gdb_registers(r, out);
    return out;
}

std::string DebugServer::handle_write_registers(std::string_view args) noexcept {
    // A 'G' packet carries every register, so the length is fixed by the
    // target description rather than by what the debugger happened to send.
    // The sixteen segment selectors and flags are declared 32 bits wide and
    // arrive in sixteen-byte slots, which is why the multiply is by two
    // characters rather than one.
    if (args.size() < kGdbRegisterCount * 16) {
        return "E01";
    }

    std::uint64_t values[kGdbRegisterCount];
    for (std::size_t i = 0; i < kGdbRegisterCount; ++i) {
        if (!parse_hex_u64_le(args.substr(i * 16, 16), values[i])) {
            return "E01";
        }
    }

    // The current values are read rather than zeroed, because three of the
    // registers the block names -- orig_rax, fs_base and gs_base -- are not
    // writable through SETREGS on this architecture. Starting from zero and
    // writing the whole block would clear the thread and group selectors,
    // and a debugger that set one register would corrupt two it never
    // mentioned. Reading first and overwriting only the writable fields is
    // what makes a partial intent harmless.
    Registers r{};
    auto res = tracer_->get_regs(pid_, r);
    if (res.failed()) {
        return "E01";
    }

    set_gdb_register(r, values);

    if (tracer_->set_regs(pid_, r).failed()) {
        return "E01";
    }
    return "OK";
}

std::string DebugServer::handle_read_memory(std::string_view args) noexcept {
    const std::size_t comma = args.find(',');
    if (comma == std::string_view::npos) {
        return "E01";
    }

    std::uint64_t addr = 0;
    std::uint64_t length = 0;
    {
        std::uint64_t a = 0;
        for (char c : args.substr(0, comma)) {
            const int d = (c >= '0' && c <= '9')
                              ? c - '0'
                              : (c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1);
            if (d < 0) {
                return "E01";
            }
            a = (a << 4) | static_cast<std::uint64_t>(d);
        }
        addr = a;
    }
    {
        std::uint64_t l = 0;
        for (char c : args.substr(comma + 1)) {
            const int d = (c >= '0' && c <= '9')
                              ? c - '0'
                              : (c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1);
            if (d < 0) {
                break;
            }
            l = (l << 4) | static_cast<std::uint64_t>(d);
        }
        length = l;
    }

    // The protocol's read is capped by what fits in one packet. A debugger
    // that asks for more than that is asking for a transfer the framing
    // cannot carry, and the cap is what keeps the answer well-formed.
    constexpr std::uint64_t kMaxRead = 4096;
    if (length == 0 || length > kMaxRead) {
        return "E01";
    }

    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(length));
    auto res = tracer_->read_memory(pid_, addr, buffer.data(), buffer.size());
    if (res.failed() || res.value <= 0) {
        return "E01";
    }

    std::string out;
    out.reserve(static_cast<std::size_t>(res.value) * 2);
    for (long i = 0; i < res.value; ++i) {
        out += hex_u8(buffer[static_cast<std::size_t>(i)]);
    }
    return out;
}

std::string DebugServer::handle_write_memory(std::string_view args) noexcept {
    const std::size_t comma = args.find(',');
    if (comma == std::string_view::npos) {
        return "E01";
    }

    std::uint64_t addr = 0;
    for (char c : args.substr(0, comma)) {
        const int d = (c >= '0' && c <= '9')
                          ? c - '0'
                          : (c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1);
        if (d < 0) {
            return "E01";
        }
        addr = (addr << 4) | static_cast<std::uint64_t>(d);
    }

    const std::size_t colon = args.find(':', comma);
    if (colon == std::string_view::npos) {
        return "E01";
    }
    // The length field between the address and the colon is redundant with
    // the payload and the protocol allows it to be omitted. It is not
    // trusted here: the payload is what is written.

    std::vector<std::uint8_t> bytes;
    if (!parse_hex_bytes(args.substr(colon + 1), bytes)) {
        return "E01";
    }

    auto res = tracer_->write_memory(pid_, addr, bytes.data(), bytes.size());
    if (res.failed()) {
        return "E01";
    }
    return "OK";
}

std::string DebugServer::handle_breakpoint(std::string_view args) noexcept {
    // The packet is Z<type>,<addr>,<kind> or z<type>,<addr>,<kind>. The
    // kind is the instruction length on some targets and is ignored here.
    if (args.size() < 2) {
        return "E01";
    }
    const bool insert = args[0] == 'Z';
    const char type = args[1];

    std::string_view rest = args.substr(3);
    const std::size_t comma = rest.find(',');
    if (comma == std::string_view::npos) {
        return "E01";
    }

    std::uint64_t addr = 0;
    for (char c : rest.substr(0, comma)) {
        const int d = (c >= '0' && c <= '9')
                          ? c - '0'
                          : (c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1);
        if (d < 0) {
            return "E01";
        }
        addr = (addr << 4) | static_cast<std::uint64_t>(d);
    }

    if (type != '0') {
        // Only software breakpoints are implemented. A hardware breakpoint
        // request would have to be served from the four debug registers,
        // and those are the scarce resource the write tracker uses; giving
        // them to the debugger without saying so would silently disable the
        // tracker. Reported as unsupported instead.
        return {};
    }

    std::string detail;
    if (insert) {
        const int rc = breakpoints_->add(*tracer_, pid_, addr, detail);
        if (rc != 0) {
            if (events_ != nullptr) {
                auto& e = events_->begin(EventKind::Note);
                e.add("text", std::string_view{"a breakpoint could not be "
                                               "installed"});
                e.add("address", std::string_view{detail});
                events_->commit();
            }
            return "E01";
        }
    } else {
        (void)breakpoints_->remove(*tracer_, pid_, addr);
    }
    return "OK";
}

std::string DebugServer::handle_thread() noexcept {
    // Thread selection. There is one process and its thread id is its pid,
    // so the selection is always satisfied.
    return "OK";
}

// -------------------------------------------------------------------- loop

SessionResult observe(const SessionConfig& config, Writer& events) noexcept {
    SessionResult out;

    if (config.pid <= 0) {
        out.failed = true;
        out.detail = "the session was given no process to observe";
        return out;
    }

    Tracer tracer;
    Breakpoints breakpoints;
    WriteExecuteTracker wx;

    // The watch manager exists whether or not tracking was asked for,
    // because the debugger's hardware breakpoints would need it too. It is
    // created here so that its destructor runs after the tracker has
    // released its own watches, rather than the other way round.
    Watchpoints watchpoints;

    // How a permission change is noticed.
    //
    // A W-to-X transition is two facts at two different times: the region
    // was written, and later the region became executable. The write is seen
    // through a hardware watch, because there is no other way to observe an
    // access without stopping the target. The permission change is seen
    // through mprotect, because that is the only syscall that can make a
    // mapped page executable, and a program that makes one executable
    // without asking the kernel is not something the kernel lets it do.
    //
    // Intercepting mprotect is therefore not a heuristic that might catch a
    // transition. It is the complete set of places the transition can
    // happen, and a tracker that watched only the writes would know that
    // something was written without ever learning that it ran.
    struct PendingProtect {
        int pid;
        std::uint64_t addr;
        std::uint64_t length;
        bool valid;
    };
    std::vector<PendingProtect> pending_protect;

    auto protect_entry_for = [&](int pid) -> PendingProtect& {
        for (auto& p : pending_protect) {
            if (p.pid == pid) {
                p.valid = false;
                return p;
            }
        }
        pending_protect.push_back(PendingProtect{pid, 0, 0, false});
        return pending_protect.back();
    };

    // The target called PTRACE_TRACEME before its exec, so this process is
    // already its tracer and the target is already stopped at the exec
    // boundary. Seizing it here would be refused, because a process that
    // has called TRACEME belongs to its parent and cannot be seized by it
    // twice.
    //
    // What is left is to consume the initial stop. It is the exec stop the
    // kernel raised on the tracer's behalf, and it has to be read before
    // any options can be set: the kernel ignores SETOPTIONS on a process
    // that has not reported its first stop.
    {
        Stop initial = tracer.wait_pid(config.pid);
        if (initial.kind == StopKind::Exited) {
            out.failed = true;
            out.detail = "the target exited before observation began";
            // The reason is reported because "exited" and "the wait failed"
            // are the same kind here, and they have nothing in common: the
            // first is a target that ran, the second is a tracer that never
            // received its first stop. A caller debugging a run that saw
            // nothing needs to know which one happened.
            auto& e = events.begin(EventKind::Note);
            e.add("text", std::string_view{
                             "no first stop: the wait reported a failure or "
                             "an exit"});
            e.add("pid", static_cast<std::uint64_t>(config.pid));
            e.add("errno", static_cast<std::int64_t>(initial.signal));
            e.add("status", static_cast<std::int64_t>(initial.raw_status));
            events.commit();
            return out;
        }
    }

    unsigned long options = kPtraceO_TRACESYSGOOD | kPtraceO_EXITKILL |
                            kPtraceO_TRACEEXEC | kPtraceO_TRACESECCOMP;
    if (config.follow_forks) {
        options |= kPtraceO_TRACEFORK | kPtraceO_TRACEVFORK |
                   kPtraceO_TRACECLONE | kPtraceO_TRACEVFORKDONE;
    }
    auto opts = tracer.set_options(config.pid, options);
    if (opts.failed()) {
        out.failed = true;
        out.detail = "the trace options were refused";
        (void)tracer.detach(config.pid, 0);
        return out;
    }

    DebugServer server(tracer, breakpoints, config.pid, events);

    PacketDecoder decoder;

    // The process has been seized and is stopped. The first resume is what
    // lets it run at all.
    const bool trace_syscalls = config.trace_syscalls;
    auto resume = [&](int pid, int signal) -> int {
        auto r = trace_syscalls ? tracer.syscall(pid, signal)
                                : tracer.cont(pid, signal);
        return r.failed() ? r.error : 0;
    };

    if (resume(config.pid, 0) != 0) {
        out.failed = true;
        out.detail = "the process could not be resumed after being seized";
        (void)tracer.detach(config.pid, 0);
        return out;
    }

    // The traced set. A fork adds to it, an exit removes from it, and the
    // loop ends when it is empty rather than when one particular pid exits.
    std::vector<int> traced{config.pid};

    // ---------------------------------------------------------- wx setup
    //
    // The regions to watch come either from the caller, which is the case
    // when something already knows where the target stages its decoded code,
    // or from the target's own maps, which is the case for a run that does
    // not. The scan is done once here rather than on every stop: a target's
    // regions change when it allocates, and re-reading maps per stop would
    // turn the tracker into the largest cost in the session.
    if (config.track_wx) {
        if (!Watchpoints::hardware_available()) {
            // Reported, not treated as a failure. The rest of the session
            // works without the tracker, and a host that blocks
            // perf_event_open should still produce a full event stream --
            // just one with an honest hole in it.
            out.wx_unavailable = true;
            auto& e = events.begin(EventKind::Note);
            e.add("text", std::string_view{
                             "write tracking is off: this host does not "
                             "permit hardware watch events"});
            events.commit();
        } else {
            std::vector<WatchTarget> wanted = config.wx_regions;

            if (wanted.empty()) {
                for (const auto& r : process_regions(config.pid)) {
                    if (config.wx_anonymous_only && !r.anonymous) {
                        continue;
                    }
                    if (r.length == 0 || r.length > config.wx_max_region_bytes) {
                        continue;
                    }
                    // A region that is neither writable nor executable is
                    // not a candidate: nothing can be written to it and
                    // nothing can run from it without a permission change
                    // first, and the permission change is intercepted
                    // separately.
                    if (r.kind == RegionClass::Other) {
                        continue;
                    }
                    WatchTarget t;
                    t.base = r.base;
                    t.length = r.length;
                    wanted.push_back(t);
                }
            }

            for (const auto& t : wanted) {
                std::string detail;
                (void)wx.watch(config.pid, t.base, t.length, detail);
            }

            const ArmReport arms = wx.arm(watchpoints, config.pid);
            out.wx_regions = wx.region_count();
            out.wx_watches = arms.watches_installed;
            out.wx_bytes_covered = arms.bytes_covered;
            out.wx_bytes_total = arms.bytes_total;
            out.wx_regions_unwatched = arms.unwatched.size();

            // The coverage is reported as an event rather than only in the
            // summary because it is the difference between "watching" and
            // "watching these bytes". A consumer reading the stream has to
            // be able to see that four debug registers covered 32 bytes of a
            // 4096-byte region, because that is the fact that limits what
            // the rest of the stream can claim.
            auto& e = events.begin(EventKind::Note);
            e.add("text", std::string_view{"write tracking armed"});
            e.add("regions", static_cast<std::uint64_t>(out.wx_regions));
            e.add("watches", out.wx_watches);
            e.add("bytes_covered", out.wx_bytes_covered);
            e.add("bytes_total", out.wx_bytes_total);
            e.add("unwatched_regions", out.wx_regions_unwatched);
            events.commit();

            for (const auto& u : arms.unwatched) {
                auto& n = events.begin(EventKind::Note);
                n.add("text", std::string_view{
                                  "a region could not be watched: no debug "
                                  "register was left"});
                n.add_hex("base", u.base);
                n.add("length", u.length);
                events.commit();
            }
        }
    }

    // Drains every watch event and feeds it to the tracker. The event's data
    // address is what is attributed, not the watched address: the kernel
    // reports which address inside the watch actually matched, and a write
    // to the second half of an eight-byte watch is a different fact from one
    // to the first.
    std::vector<WatchEvent> watch_events;
    auto drain_watches = [&]() {
        if (!config.track_wx || out.wx_unavailable) {
            return;
        }
        watch_events.clear();
        (void)watchpoints.read_events(watch_events);
        for (const auto& ev : watch_events) {
            if (ev.kind != WatchKind::Write || !ev.has_address) {
                continue;
            }
            // The width of the access is not in the record -- the kernel
            // reports the address, not how much was touched -- so it comes
            // from the instruction. A decoder that stores a byte at a time
            // and one that stores a quadword at a time produce the same
            // address and different byte counts, and the byte count is what
            // tells them apart.
            std::uint64_t width = 1;
            std::uint8_t code[16];
            const auto rr = tracer.peek(ev.pid, ev.rip);
            if (rr.ok()) {
                (void)tracer.read_memory(ev.pid, ev.rip, code, sizeof(code));
                const AccessDecode d = decode_access(code, sizeof(code));
                if (d.valid && d.width != 0) {
                    width = d.width;
                }
            }
            wx.note_write(ev.pid, ev.address, width, ev.rip);
            auto& e = events.begin(EventKind::MemoryWrite);
            e.add("pid", static_cast<std::uint64_t>(ev.pid));
            e.add_hex("address", ev.address);
            e.add_hex("rip", ev.rip);
            e.add("bytes", width);
            e.add("kind", std::string_view{"watch"});
            events.commit();
        }
        out.wx_lost_samples = watchpoints.lost_samples();
    };

    // The syscall state per process. A syscall stop alternates between
    // entry and exit and the kernel does not say which; the only way to
    // know is to remember what the last one was.
    //
    // A process is entered as "the next stop is an entry". Getting this
    // backwards is not a cosmetic mistake: on an entry stop rax still holds
    // the previous syscall's result, and for the first syscall of a process
    // that is -ENOSYS because the kernel has not run anything yet. Labelling
    // the first stop as an exit therefore reports -ENOSYS as the return
    // value of a syscall that has not happened.
    struct SyscallState {
        int pid;
        bool next_is_entry;
    };
    std::vector<SyscallState> syscall_state;
    auto syscall_entry_for = [&](int pid, bool& at_entry) {
        for (auto& s : syscall_state) {
            if (s.pid == pid) {
                at_entry = s.next_is_entry;
                s.next_is_entry = !s.next_is_entry;
                return;
            }
        }
        // The first stop for a process is an entry, because the process was
        // resumed and the kernel stopped it before running the instruction.
        syscall_state.push_back(SyscallState{pid, false});
        at_entry = true;
    };

    for (;;) {
        if (traced.empty()) {
            break;
        }

        // Serves the debugger between stops. poll with a zero timeout is
        // used rather than a blocking read so that the tracee's stops are
        // never delayed by a debugger that has nothing to say.
        if (config.serve_gdb && config.gdb_read_fd >= 0) {
            struct pollfd pfd {};
            pfd.fd = config.gdb_read_fd;
            pfd.events = POLLIN;
            const int pr = ::poll(&pfd, 1, 0);
            if (pr > 0 && (pfd.revents & POLLIN) != 0) {
                char buffer[4096];
                const ssize_t n = ::read(config.gdb_read_fd, buffer,
                                         sizeof(buffer));
                if (n > 0) {
                    decoder.feed(buffer, static_cast<std::size_t>(n));
                    while (decoder.has_packet()) {
                        const Packet p = decoder.take();
                        if (!p.checksum_ok) {
                            continue;
                        }
                        const std::string reply = server.handle(p.data);
                        const std::string encoded = encode_packet(reply);
                        if (config.gdb_write_fd >= 0) {
                            const ssize_t w = ::write(config.gdb_write_fd,
                                                      encoded.data(),
                                                      encoded.size());
                            (void)w;
                        }
                    }
                } else if (n == 0) {
                    // The debugger disconnected. The process is left to run
                    // rather than killed, because a debugger that goes away
                    // is not a request to end the target.
                    server.clear_resume();
                }
            }
        }

        if (server.detached()) {
            (void)tracer.detach(config.pid, 0);
            break;
        }

        Stop stop = tracer.wait(0);

        ++out.stops;

        if (stop.kind == StopKind::Exited) {
            // The wait reported either an exit or a failure. The signal
            // field carries the errno for a failure, which is how the two
            // are told apart.
            if (stop.term_signal) {
                out.signaled = true;
                out.term_signal = stop.exit_code;
            } else if (stop.signal == 0) {
                out.exit_code = stop.exit_code;
            }

            auto& e = events.begin(EventKind::ProcessExit);
            e.add("pid", static_cast<std::uint64_t>(stop.pid));
            e.add("exit_code", static_cast<std::int64_t>(stop.exit_code));
            e.add("signaled", stop.term_signal);
            e.add("term_signal", static_cast<std::uint64_t>(stop.exit_code));
            events.commit();

            // Remove the process from the traced set. Reaping happens here
            // rather than in a separate pass because a process that has
            // reported its exit status is already reaped by the wait.
            //
            // The watches on it go first. A watch is a debug register held
            // for a process that no longer exists, and leaving it installed
            // would consume one of the four for the rest of the session --
            // on a fork-tracing run, four forks would exhaust the hardware
            // and the tracker would stop working for the rest of the target's
            // life.
            if (config.track_wx && !out.wx_unavailable) {
                drain_watches();
                (void)watchpoints.remove_all_for(stop.pid);
            }
            for (auto it = traced.begin(); it != traced.end(); ++it) {
                if (*it == stop.pid) {
                    traced.erase(it);
                    break;
                }
            }
            if (traced.empty()) {
                break;
            }
            continue;
        }

        if (stop.kind == StopKind::NewChild) {
            // The child inherits the tracing and has to be configured
            // before it is allowed to run, or its first syscall would be
            // missed.
            auto msg = tracer.event_message(stop.pid);
            if (msg.ok()) {
                const int child = static_cast<int>(msg.value);
                traced.push_back(child);
                auto& e = events.begin(EventKind::ProcessSpawn);
                e.add("pid", static_cast<std::uint64_t>(child));
                e.add("parent", static_cast<std::uint64_t>(stop.pid));
                events.commit();
                (void)resume(child, 0);
            }
            (void)resume(stop.pid, 0);
            continue;
        }

        if (stop.kind == StopKind::Exec) {
            // Every cached address is stale after an exec. The breakpoints
            // are dropped rather than carried over, because a breakpoint at
            // an address that meant something in the old image means
            // something else in the new one.
            for (const auto& b : breakpoints.all()) {
                (void)tracer.poke(stop.pid, b.address,
                                  static_cast<std::uint64_t>(b.saved));
            }
            // The same is true of the watches, with one extra problem: a
            // watch placed on the old image's address now points at
            // whatever the new image put there, so it would fire on an
            // access to unrelated memory and call it a write to a region
            // that no longer exists. They are released and the new image's
            // regions are watched instead.
            if (config.track_wx && !out.wx_unavailable) {
                wx.release(watchpoints);
                wx = WriteExecuteTracker();
            }
            auto& e = events.begin(EventKind::Exec);
            e.add("pid", static_cast<std::uint64_t>(stop.pid));
            events.commit();
            (void)resume(stop.pid, 0);
            continue;
        }

        if (stop.kind == StopKind::SeccompTrap) {
            auto& e = events.begin(EventKind::SyscallBlocked);
            e.add("pid", static_cast<std::uint64_t>(stop.pid));
            auto msg = tracer.event_message(stop.pid);
            if (msg.ok()) {
                e.add_hex("nr", static_cast<std::uint64_t>(msg.value));
            }
            events.commit();
            (void)resume(stop.pid, 0);
            continue;
        }

        if (stop.kind == StopKind::SyscallStop) {
            ++out.syscall_stops;

            bool at_entry = false;
            syscall_entry_for(stop.pid, at_entry);

            Registers regs{};
            auto gr = tracer.get_regs(stop.pid, regs);
            if (gr.ok()) {
                const std::uint64_t nr = regs.orig_rax;

                // The permission syscalls are the whole of the transition
                // story. On entry the arguments are the ones the target
                // asked for; on exit the kernel has either applied them or
                // refused, and the maps say which. Reading the maps after
                // the call rather than trusting the arguments is what makes
                // this correct for a call that failed: a refused mprotect
                // leaves the region non-executable, and reporting a
                // transition for it would be a fabrication.
                if (config.track_wx && !out.wx_unavailable) {
                    if (at_entry) {
                        if (nr == kSysMprotect) {
                            PendingProtect& p = protect_entry_for(stop.pid);
                            p.addr = regs.rdi;
                            p.length = regs.rsi;
                            p.valid = true;
                        } else if (nr == kSysMmap) {
                            PendingProtect& p = protect_entry_for(stop.pid);
                            // mmap reports its address in the return value,
                            // not the first argument, so the range is not
                            // known until the call returns. What is known
                            // on entry is the requested protection, which
                            // is enough to decide whether the result is
                            // worth looking at.
                            p.length = regs.rsi;
                            p.valid = true;
                        } else if (nr == kSysMunmap || nr == kSysMremap) {
                            PendingProtect& p = protect_entry_for(stop.pid);
                            p.addr = regs.rdi;
                            p.length = regs.rsi;
                            p.valid = true;
                        }
                    } else {
                        for (auto& p : pending_protect) {
                            if (p.pid != stop.pid || !p.valid) {
                                continue;
                            }
                            p.valid = false;

                            // The address the region ended up at. For mmap
                            // that is the return value; for the others it
                            // is the argument the target passed.
                            const std::uint64_t addr =
                                (nr == kSysMmap && static_cast<long>(regs.rax) >= 0)
                                    ? static_cast<std::uint64_t>(regs.rax)
                                    : p.addr;
                            if (addr == 0) {
                                continue;
                            }

                            // A fresh mapping is where a decoder's staging
                            // buffer appears, so the hardware slots are moved
                            // onto it as soon as it exists. Doing this at
                            // startup alone cannot work: the region does not
                            // exist yet, and the four debug registers would
                            // be spent on the loader's data segments instead.
                            if (nr == kSysMmap) {
                                const ArmReport chased =
                                    wx.chase(watchpoints, stop.pid, addr,
                                             p.length);
                                if (chased.watches_installed > 0) {
                                    out.wx_watches = watchpoints.fds().size();
                                    auto& c =
                                        events.begin(EventKind::Note);
                                    c.add("text", std::string_view{
                                                     "chased a new mapping"});
                                    c.add_hex("base",
                                              addr & ~(std::uint64_t{4095}));
                                    c.add("length", p.length);
                                    c.add("watches",
                                          chased.watches_installed);
                                    c.add("bytes_covered",
                                          chased.bytes_covered);
                                    c.add("bytes_total", chased.bytes_total);
                                    events.commit();
                                }
                            }

                            RegionPerms perms;
                            if (!read_region_perms(stop.pid, addr, perms)) {
                                // The mapping is gone, which is what munmap
                                // does. There is nothing to compare.
                                continue;
                            }
                            if (wx.note_permission(stop.pid, addr, perms)) {
                                wx.flush(stop.pid, events);
                                out.transitions += 1;
                            }
                        }
                    }
                }

                auto& e = events.begin(EventKind::Note);
                if (at_entry) {
                    e.add("text", std::string_view{"syscall entry"});
                    e.add("nr", static_cast<std::int64_t>(
                                    static_cast<long>(regs.orig_rax)));
                    e.add_hex("arg0", regs.rdi);
                    e.add_hex("arg1", regs.rsi);
                    e.add_hex("arg2", regs.rdx);
                } else {
                    e.add("text", std::string_view{"syscall exit"});
                    e.add("ret", static_cast<std::int64_t>(
                                      static_cast<long>(regs.rax)));
                }
                e.add("pid", static_cast<std::uint64_t>(stop.pid));
                events.commit();
            }

            // The watch events are drained after the syscall rather than
            // before, because a write the syscall made is a write that has
            // already happened by the time the kernel reports the call.
            drain_watches();

            (void)resume(stop.pid, 0);
            continue;
        }

        if (stop.kind == StopKind::Signal) {
            // A watch fires as a stop of its own on the traced process, and
            // this is where those stops arrive. The events are drained
            // before anything else because the instruction that wrote is
            // only available while the process is still stopped here.
            drain_watches();

            // A synthetic stop is one ptrace arranged and must not be
            // redelivered. A real signal has to reach the process, or a
            // target that handles SIGSEGV would never see it.
            const int deliver = stop.synthetic ? 0 : stop.signal;

            if (!stop.synthetic && stop.signal != 0) {
                ++out.signals;
                auto& e = events.begin(EventKind::Signal);
                e.add("pid", static_cast<std::uint64_t>(stop.pid));
                e.add("signal", static_cast<std::uint64_t>(stop.signal));
                events.commit();
            }

            // A trap that lands on a breakpoint the session installed is a
            // breakpoint hit, not a stray signal. The distinction is made
            // by looking at rip, because that is the only thing that says
            // where the trap came from.
            Registers regs{};
            if (tracer.get_regs(stop.pid, regs).ok()) {
                if (breakpoints.find(regs.rip - 1) != nullptr) {
                    ++out.breakpoint_hits;
                    auto& e = events.begin(EventKind::BreakpointHit);
                    e.add("pid", static_cast<std::uint64_t>(stop.pid));
                    e.add_hex("address", regs.rip - 1);
                    events.commit();

                    if (server.resume_requested() && !server.step_requested()) {
                        // The debugger asked to continue. The breakpoint
                        // byte has to come out, the instruction has to run,
                        // and the breakpoint has to go back in. That is a
                        // single step with the original byte in place.
                        std::string detail;
                        const int rc = breakpoints.step_over(tracer, stop.pid,
                                                            detail);
                        if (rc == 0) {
                            // The step consumed a wait. The next wait is the
                            // resume the caller asked for.
                            server.clear_resume();
                            (void)resume(stop.pid, 0);
                        } else {
                            (void)resume(stop.pid, 0);
                        }
                        continue;
                    }
                }
            }

            if (server.resume_requested()) {
                const int sig = server.resume_signal();
                const bool step = server.step_requested();
                server.clear_resume();
                if (step) {
                    auto r = tracer.singlestep(stop.pid, sig);
                    (void)r;
                } else {
                    (void)resume(stop.pid, sig);
                }
                continue;
            }

            (void)resume(stop.pid, deliver);
            continue;
        }

        if (stop.kind == StopKind::GroupStop) {
            // A group stop is continued with a zero signal, or with SIGCONT
            // to actually deliver the stop to the group. Continuing with
            // SIGSTOP here would re-stop the process immediately.
            (void)resume(stop.pid, stop.signal == kSigstop ? 0 : stop.signal);
            continue;
        }
    }

    // The session is over. Any process still in the traced set is one whose
    // exit the loop did not see, which happens when the debugger detached
    // first; they are continued so that they are not left stopped.
    for (int pid : traced) {
        (void)tracer.detach(pid, 0);
    }

    // The last drain and the last transition report. A target that wrote its
    // payload and made it executable and then exited within one syscall
    // interval would otherwise have its transition never reported, which is
    // the one case the tracker exists for.
    if (config.track_wx && !out.wx_unavailable) {
        drain_watches();
        // The count is taken before the flush, because flushing is what
        // writes the transitions out and it clears them. Reading it after
        // would always be zero, which is exactly the number a caller would
        // believe if the tracker had found nothing.
        const std::size_t pending = wx.transitions().size();
        wx.flush(config.pid, events);
        out.transitions += pending;
        wx.release(watchpoints);
    }

    return out;
}

} // namespace occ::obs
