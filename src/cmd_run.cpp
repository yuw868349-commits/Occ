#include "occ/commands.h"

#include "occ/observer/event.h"
#include "occ/runner/run.h"
#include "occ/util/log.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace occ {

namespace {

void print_run_usage() {
    std::fprintf(
        stderr,
        "usage: occ run [options] <path> [args...]\n"
        "\n"
        "runs a target under isolation and writes one JSON object per line\n"
        "describing what was observed\n"
        "\n"
        "options:\n"
        "  --root <dir>       use <dir> as the new root (read-only)\n"
        "  --overlay <lower>  run on an overlay of <lower>, writes discarded\n"
        "  --upper <dir>      where overlay writes land; implies --overlay\n"
        "  --work <dir>       overlay work directory\n"
        "  --bind <src>:<dst> bind <src> into the root at <dst>, read-only\n"
        "  --rw-bind <s>:<d>  same, but writable\n"
        "  --memory <bytes>   memory limit for the target\n"
        "  --pids <n>         process count limit for the target\n"
        "  --cpu <percent>    cpu bandwidth, 1..100\n"
        "  --cgroup <dir>     create the run's cgroup under <dir>\n"
        "  --env <K=V>        set an environment variable for the target\n"
        "  --no-events        do not write the event stream\n"
        "  --help             print this text\n"
        "\n"
        "the event stream goes to stdout when it is not a terminal, and is\n"
        "suppressed when it is, so an interactive run shows only the\n"
        "target's own output\n");
}

// Parses an unsigned decimal. Returns false rather than clamping, because a
// limit that silently became a different limit is worse than a run that
// refused to start.
bool parse_u64(std::string_view s, std::uint64_t& out) noexcept {
    if (s.empty()) {
        return false;
    }
    std::uint64_t v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') {
            return false;
        }
        const std::uint64_t digit = static_cast<std::uint64_t>(c - '0');
        if (v > (UINT64_MAX - digit) / 10) {
            return false;
        }
        v = v * 10 + digit;
    }
    out = v;
    return true;
}

// Splits "source:target" into its two halves. A bind mount with no target
// mounts at the same path it came from, which is the common case for a
// directory the target expects to find where the host has it.
bool split_bind(std::string_view spec, std::string& source,
                std::string& target) noexcept {
    const std::size_t colon = spec.find(':');
    if (colon == std::string_view::npos) {
        if (spec.empty()) {
            return false;
        }
        source.assign(spec);
        target.assign(spec);
        return true;
    }
    if (colon == 0 || colon + 1 >= spec.size()) {
        return false;
    }
    source.assign(spec.substr(0, colon));
    target.assign(spec.substr(colon + 1));
    return true;
}

} // namespace

int cmd_run(int argc, char** argv) {
    runner::RunOptions options;

    std::string target;
    std::vector<std::string> target_argv;

    bool no_events = false;

    for (int i = 0; i < argc; ++i) {
        const std::string_view arg = argv[i];

        // Everything after the target path belongs to the target. Parsing
        // stops at the first non-option so that a target's own --help is
        // passed through rather than consumed here.
        if (target.empty() && (arg.empty() || arg[0] != '-')) {
            target.assign(arg);
            continue;
        }

        if (!target.empty()) {
            target_argv.emplace_back(arg);
            continue;
        }

        if (arg == "--help" || arg == "-h") {
            print_run_usage();
            return 0;
        }

        // Each option that takes a value consumes the next argument. A
        // missing value is a usage error, not a silently absent option.
        auto value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "occ run: %s needs a value\n", name);
                return nullptr;
            }
            ++i;
            return argv[i];
        };

        if (arg == "--root") {
            const char* v = value("--root");
            if (v == nullptr) {
                return 2;
            }
            options.root_dir = v;
        } else if (arg == "--overlay") {
            const char* v = value("--overlay");
            if (v == nullptr) {
                return 2;
            }
            options.root_dir = v;
            options.root_kind = isolation::RootKind::Overlay;
        } else if (arg == "--upper") {
            const char* v = value("--upper");
            if (v == nullptr) {
                return 2;
            }
            options.upper_dir = v;
            options.root_kind = isolation::RootKind::Overlay;
        } else if (arg == "--work") {
            const char* v = value("--work");
            if (v == nullptr) {
                return 2;
            }
            options.work_dir = v;
        } else if (arg == "--bind" || arg == "--rw-bind") {
            const char* v = value(arg == "--bind" ? "--bind" : "--rw-bind");
            if (v == nullptr) {
                return 2;
            }
            isolation::ContainerConfig::BindMount mount;
            if (!split_bind(v, mount.source, mount.target)) {
                std::fprintf(stderr, "occ run: bad bind specification '%s'\n",
                             v);
                return 2;
            }
            mount.writable = (arg == "--rw-bind");
            options.extra_mounts.push_back(mount);
        } else if (arg == "--memory") {
            const char* v = value("--memory");
            if (v == nullptr) {
                return 2;
            }
            if (!parse_u64(v, options.memory_bytes)) {
                std::fprintf(stderr, "occ run: bad byte count '%s'\n", v);
                return 2;
            }
        } else if (arg == "--pids") {
            const char* v = value("--pids");
            if (v == nullptr) {
                return 2;
            }
            if (!parse_u64(v, options.pids)) {
                std::fprintf(stderr, "occ run: bad process count '%s'\n", v);
                return 2;
            }
        } else if (arg == "--cpu") {
            const char* v = value("--cpu");
            if (v == nullptr) {
                return 2;
            }
            std::uint64_t percent = 0;
            if (!parse_u64(v, percent) || percent == 0 || percent > 100) {
                std::fprintf(stderr,
                             "occ run: --cpu takes 1..100, got '%s'\n", v);
                return 2;
            }
            options.cpu_percent = static_cast<std::uint32_t>(percent);
        } else if (arg == "--cgroup") {
            const char* v = value("--cgroup");
            if (v == nullptr) {
                return 2;
            }
            options.cgroup_parent = v;
        } else if (arg == "--env") {
            const char* v = value("--env");
            if (v == nullptr) {
                return 2;
            }
            if (std::string_view(v).find('=') == std::string_view::npos) {
                std::fprintf(stderr,
                             "occ run: --env takes KEY=VALUE, got '%s'\n", v);
                return 2;
            }
            options.env.emplace_back(v);
        } else if (arg == "--no-events") {
            no_events = true;
        } else {
            std::fprintf(stderr, "occ run: unknown option '%s'\n",
                         std::string(arg).c_str());
            print_run_usage();
            return 2;
        }
    }

    if (target.empty()) {
        std::fprintf(stderr, "occ run: no target path\n");
        print_run_usage();
        return 2;
    }

    target_argv.insert(target_argv.begin(), target);

    // The stream is the machine-readable output, so it is written when the
    // caller is not a terminal. Writing it to a terminal would bury an
    // interactive session's own output under a line per fact, which is the
    // one case where the facts are not the point. The decision is made once,
    // by the caller of this function, and handed over in the environment;
    // re-deriving it here would let two isatty calls disagree.
    bool emit_events = false;
    if (!no_events) {
        const char* want = ::getenv("OCC_EVENT_STREAM");
        emit_events = want != nullptr && want[0] == '1';
    }

    obs::Writer events;
    // A sink on a closed descriptor makes every emit a no-op without the
    // call sites having to check. The value is one no real stream uses, so
    // an accidental write fails instead of landing somewhere unexpected.
    events.attach(emit_events ? 1 : -1);

    const runner::RunResult result =
        runner::run(target, target_argv, options, events);

    if (result.failed) {
        // A refusal names the reason directly; a failing syscall names the
        // stage it failed at. Both go to stderr, because stdout carries the
        // event stream and a diagnostic written into the stream would be a
        // record of something that did not happen.
        if (!result.failure_detail.empty()) {
            log::error(result.failure_detail);
        } else if (!result.error.detail.empty()) {
            log::error(result.error.detail);
        } else {
            log::error(std::string("the run failed at ") +
                       isolation::stage_name(result.error.stage));
        }
        return 1;
    }

    if (result.signaled) {
        // A target killed by a signal is reported the way a shell reports
        // it, so that a caller scripting occ sees what it would see from
        // any other runner.
        return 128 + result.term_signal;
    }

    return result.exit_code;
}

} // namespace occ
