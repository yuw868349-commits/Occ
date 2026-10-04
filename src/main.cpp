#include "occ/commands.h"
#include "occ/util/log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace {

// isatty is a libc call, not a syscall, but it is one of the few that are
// genuinely part of the C runtime rather than a wrapper this project
// replaces. It reads the terminal state of a file descriptor and has no
// syscall of its own.
[[nodiscard]] bool stdout_is_tty() {
    return ::isatty(1) != 0;
}

void print_usage(const char* argv0) {
    std::fprintf(
        stderr,
        "usage: %s <command> [options]\n"
        "\n"
        "commands:\n"
        "  check <path>       detect target format and print the routing "
        "decision\n"
        "  run <path>         run a target under isolation, start the "
        "observer\n"
        "  attach <session>   reconnect to a running session\n"
        "  stop <session>     tear down a session\n"
        "  doctor             report host capabilities and limitations\n"
        "\n"
        "options:\n"
        "  --help             print this text\n",
        argv0);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }

    // Whether the event stream is wanted is decided here, once, and handed
    // to the commands through the environment. A command that re-derived it
    // could disagree with this call if the descriptor changed in between,
    // and the two answers would be for the same stream.
    if (stdout_is_tty()) {
        ::setenv("OCC_EVENT_STREAM", "0", 1);
    } else {
        ::setenv("OCC_EVENT_STREAM", "1", 1);
    }

    const std::string_view cmd = argv[1];

    if (cmd == "--help" || cmd == "-h" || cmd == "help") {
        print_usage(argv[0]);
        return 0;
    }

    const bool ndjson = !stdout_is_tty();

    if (cmd == "doctor") {
        return occ::cmd_doctor(ndjson);
    }
    if (cmd == "check") {
        return occ::cmd_check(argc - 2, argv + 2);
    }
    if (cmd == "run") {
        return occ::cmd_run(argc - 2, argv + 2);
    }
    if (cmd == "attach") {
        return occ::cmd_attach(argc - 2, argv + 2);
    }
    if (cmd == "stop") {
        return occ::cmd_stop(argc - 2, argv + 2);
    }

    occ::log::error("unknown command");
    print_usage(argv[0]);
    return 2;
}
