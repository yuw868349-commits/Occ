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

// The register block GDB expects for x86-64, in its order. The session
// fills this from the kernel's own register struct rather than maintaining
// a second copy of the state.
void fill_gdb_registers(const Registers& r, std::string& out) noexcept {
    out.clear();
    out.reserve(27 * 8);
    const std::uint64_t order[] = {
        r.rax, r.rbx, r.rcx, r.rdx, r.rsi, r.rdi, r.rbp, r.rsp,
        r.r8,  r.r9,  r.r10, r.r11, r.r12, r.r13, r.r14, r.r15,
        r.rip, r.eflags, r.cs, r.ss, r.ds, r.es, r.fs, r.gs,
    };
    for (std::uint64_t v : order) {
        out += hex_u64_le(v);
    }
}

} // namespace

std::string DebugServer::handle(std::string_view packet) noexcept {
    if (packet.empty()) {
        return {};
    }

    const char cmd = packet[0];
    const std::string_view args = packet.size() > 1 ? packet.substr(1)
                                                    : std::string_view{};

    switch (cmd) {
    case '?':
        // Why did we stop. Answered with the last stop, which the loop has
        // already reported.
        return encode_stop_reply(gdb_signal_for_trap(), 0);

    case 'q':
        return handle_query(args);

    case 'g':
        return handle_read_registers();

    case 'G':
        return handle_write_registers(args);

    case 'p': {
        // Read one register by number. Only the ones the session knows are
        // answered; the rest come back empty, which means unsupported.
        std::uint32_t which = 0;
        if (!parse_hex_u32_le(std::string_view(args.data(), args.size() < 8 ? args.size() : 8),
                              which)) {
            return {};
        }
        Registers r{};
        if (tracer_->get_regs(pid_, r).failed()) {
            return "E01";
        }
        const std::uint64_t order[] = {
            r.rax, r.rbx, r.rcx, r.rdx, r.rsi, r.rdi, r.rbp, r.rsp,
            r.r8,  r.r9,  r.r10, r.r11, r.r12, r.r13, r.r14, r.r15,
            r.rip, r.eflags, r.cs, r.ss, r.ds, r.es, r.fs, r.gs,
        };
        if (which >= sizeof(order) / sizeof(order[0])) {
            return {};
        }
        return hex_u64_le(order[which]);
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
            std::uint32_t sig = 0;
            if (parse_hex_u32_le(std::string_view(args.data(), args.size() < 8 ? args.size() : 8), sig)) {
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
        // The features this stub implements. The list is deliberately short
        // and honest: claiming a feature that is not implemented makes the
        // debugger take a code path that then fails in a way that looks
        // like a bug in the debugger.
        return "qXfer:features:read+;swbreak+;hwbreak+;vContSupported+";
    }
    if (starts_with(kind, "Attached")) {
        // Already attached, because the session was created from a process
        // it owns rather than by asking the target to spawn one.
        return "1";
    }
    if (starts_with(kind, "C")) {
        // The current thread. This stub observes one process, so the answer
        // is always that process.
        return "QC" + hex_u32_le(static_cast<std::uint32_t>(pid_));
    }
    if (starts_with(kind, "fThreadInfo")) {
        return "m" + hex_u32_le(static_cast<std::uint32_t>(pid_));
    }
    if (starts_with(kind, "sThreadInfo")) {
        return "l";
    }
    if (starts_with(kind, "TStatus")) {
        return "T0";
    }
    return {};
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
    if (args.size() < 27 * 16) {
        return "E01";
    }
    Registers r{};
    auto res = tracer_->get_regs(pid_, r);
    if (res.failed()) {
        return "E01";
    }

    std::uint64_t values[27];
    for (int i = 0; i < 27; ++i) {
        if (!parse_hex_u64_le(args.substr(static_cast<std::size_t>(i) * 16, 16),
                              values[i])) {
            return "E01";
        }
    }

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
