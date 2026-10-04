#include "occ/commands.h"

#include "occ/observer/registry.h"
#include "occ/util/log.h"

#include <cstdio>
#include <string>
#include <string_view>

namespace occ {

namespace {

void print_attach_usage() {
    std::fprintf(stderr,
                 "usage: occ attach [-p] <session-id|pid>\n"
                 "\n"
                 "lists the running sessions when no argument is given\n"
                 "\n"
                 "options:\n"
                 "  -p                 the argument is a process id, not a "
                 "session id\n"
                 "  --help             print this text\n");
}

} // namespace

int cmd_attach(int argc, char** argv) {
    std::string key;
    bool by_pid = false;

    for (int i = 0; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_attach_usage();
            return 0;
        }
        if (arg == "-p") {
            by_pid = true;
            continue;
        }
        if (key.empty()) {
            key.assign(arg);
            continue;
        }
        std::fprintf(stderr, "occ attach: unexpected argument '%s'\n",
                     std::string(arg).c_str());
        return 2;
    }

    // With no argument this lists, which is the operation an operator wants
    // first and the one they should not have to look up. Stale records are
    // removed by the listing itself, so what is printed is what is there.
    if (key.empty()) {
        const auto sessions = obs::live_sessions();
        if (sessions.empty()) {
            std::fprintf(stderr, "no running sessions\n");
            return 1;
        }
        for (const auto& s : sessions) {
            std::printf("%s  pid=%d  target=%s\n", s.id.c_str(), s.pid,
                        s.target.c_str());
            if (!s.stream_path.empty()) {
                std::printf("  stream %s\n", s.stream_path.c_str());
            }
        }
        return 0;
    }

    obs::SessionRecord record;
    if (!obs::find_session(key, by_pid, record)) {
        log::error(by_pid ? "no session observes that process id"
                          : "no session with that id");
        return 1;
    }

    // What attach reports is where the session is and where its stream is.
    // Taking control of the process is what occ run does when it starts
    // one; doing it here would put two tracers on one process, and the
    // kernel gives the process to whichever attached last, silently
    // detaching the first.
    std::printf("session %s\n", record.id.c_str());
    std::printf("  pid      %d\n", record.pid);
    std::printf("  target   %s\n", record.target.c_str());
    if (record.stream_path.empty()) {
        std::printf("  stream   (not written to a file)\n");
    } else {
        std::printf("  stream   %s\n", record.stream_path.c_str());
    }
    return 0;
}

} // namespace occ
