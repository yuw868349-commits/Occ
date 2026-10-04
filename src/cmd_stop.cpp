#include "occ/commands.h"

#include "occ/observer/registry.h"
#include "occ/syscall/errno.h"
#include "occ/syscall/syscall.h"
#include "occ/util/log.h"

#include <cstdio>
#include <string>
#include <string_view>

namespace occ {

namespace {

constexpr int kSigterm = 15;
constexpr int kSigkill = 9;

void print_stop_usage() {
    std::fprintf(stderr,
                 "usage: occ stop [-p] <session-id|pid>\n"
                 "\n"
                 "stops a session's process and removes its record\n"
                 "\n"
                 "options:\n"
                 "  -p                 the argument is a process id\n"
                 "  -9                 send SIGKILL instead of SIGTERM\n"
                 "  --forget           remove the record without signalling\n"
                 "  --help             print this text\n");
}

} // namespace

int cmd_stop(int argc, char** argv) {
    std::string key;
    bool by_pid = false;
    bool forget_only = false;
    bool hard = false;

    for (int i = 0; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_stop_usage();
            return 0;
        }
        if (arg == "-p") {
            by_pid = true;
            continue;
        }
        if (arg == "-9" || arg == "--kill") {
            hard = true;
            continue;
        }
        if (arg == "--forget") {
            forget_only = true;
            continue;
        }
        if (key.empty()) {
            key.assign(arg);
            continue;
        }
        std::fprintf(stderr, "occ stop: unexpected argument '%s'\n",
                     std::string(arg).c_str());
        return 2;
    }

    if (key.empty()) {
        std::fprintf(stderr, "occ stop: no session given\n");
        print_stop_usage();
        return 2;
    }

    obs::SessionRecord record;
    if (!obs::find_session(key, by_pid, record)) {
        log::error(by_pid ? "no session observes that process id"
                          : "no session with that id");
        return 1;
    }

    if (forget_only) {
        // The record is removed and the process is left alone. This is the
        // path for a session whose process has been adopted by something
        // else, where signalling it would be the wrong thing to do.
        obs::unregister_session(record.id);
        std::printf("forgot session %s; process %d was not signalled\n",
                    record.id.c_str(), record.pid);
        return 0;
    }

    // SIGTERM first, because a target with its own cleanup should get the
    // chance to run it. SIGKILL is what a caller asks for explicitly.
    const int sig = hard ? kSigkill : kSigterm;
    auto r = sys::kill(record.pid, sig);
    if (r.failed()) {
        if (r.error == sys::kEsrch) {
            // Already gone. The record is stale, which is the state the
            // caller wanted, so this is a success rather than a failure.
            obs::unregister_session(record.id);
            std::printf("session %s had already ended\n", record.id.c_str());
            return 0;
        }
        log::error("the process could not be signalled");
        return 1;
    }

    obs::unregister_session(record.id);
    std::printf("signalled process %d of session %s\n", record.pid,
                record.id.c_str());
    return 0;
}

} // namespace occ
