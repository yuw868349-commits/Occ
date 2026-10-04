#include "occ/runner/run.h"

#include "occ/parser/detect.h"
#include "occ/syscall/errno.h"
#include "occ/syscall/syscall.h"
#include "occ/util/fs.h"
#include "occ/util/log.h"
#include "occ/util/string.h"

namespace occ::runner {

namespace {

// The environment a target gets when the caller did not supply one. Passing
// the whole host environment through would hand the target every variable
// the operator happens to have set, and a variable like LD_PRELOAD changes
// what the binary does. These three are what a static binary needs.
std::vector<std::string> default_environment(const RunOptions& options) {
    if (!options.env.empty()) {
        return options.env;
    }
    std::vector<std::string> env;
    env.emplace_back("PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:"
                     "/usr/bin:/sbin:/bin");
    env.emplace_back("TERM=dumb");
    // A run is not interactive. Making that explicit stops a target from
    // waiting on a terminal that will never answer.
    env.emplace_back("OCC_SANDBOX=1");
    return env;
}

// Reports one loadable segment. Called once per PT_LOAD header, in address
// order, which is the order the kernel itself would map them.
void report_mapping(obs::Writer& events, const parser::DesiredMapping& m,
                    std::size_t index) {
    auto& e = events.begin(obs::EventKind::Mapping);
    e.add("index", static_cast<std::uint64_t>(index));
    e.add_hex("file_offset", m.file_offset);
    e.add_hex("vaddr", m.vaddr);
    e.add_hex("filesz", m.filesz);
    e.add_hex("memsz", m.memsz);
    e.add("readable", (m.flags & parser::kPfRead) != 0);
    e.add("writable", (m.flags & parser::kPfWrite) != 0);
    e.add("executable", (m.flags & parser::kPfExec) != 0);
    // The part of the segment that is not backed by the file has to be
    // zeroed rather than read. Reporting it separately means a consumer does
    // not have to subtract two fields to find the size of the zero region.
    e.add_hex("zero_fill", m.memsz > m.filesz ? m.memsz - m.filesz : 0);
    events.commit();
}

} // namespace

parser::ElfImage inspect(const std::string& path, obs::Writer* events,
                         std::string& read_error) {
    read_error.clear();

    auto bytes = fs::read_file_bytes(path);
    if (!bytes) {
        read_error = "the file could not be read";
        return parser::ElfImage{};
    }

    const parser::ElfImage image =
        parser::ElfImage::parse(ByteSpan{bytes->data(), bytes->size()});

    if (events != nullptr) {
        auto& e = events->begin(obs::EventKind::ImageLoaded);
        e.add("path", path);
        e.add("size", static_cast<std::uint64_t>(bytes->size()));
        e.add("ok", image.ok());
        if (image.ok()) {
            e.add("type", static_cast<std::uint64_t>(image.type()));
            e.add("machine", static_cast<std::uint64_t>(image.machine()));
            e.add_hex("entry", image.entry());
            e.add("phnum", static_cast<std::uint64_t>(image.phnum()));
            e.add("phentsize", static_cast<std::uint64_t>(image.phentsize()));
            e.add_hex("lowest_vaddr", image.lowest_vaddr());
            e.add_hex("highest_vaddr", image.highest_vaddr());
            e.add("interpreter", image.interpreter());
            e.add("executable_stack", image.wants_executable_stack());
            e.add("gnu_relro", image.has_gnu_relro());
        } else {
            e.add("error", parser::load_error_name(image.error()));
            e.add("detail", image.error_detail());
        }
        events->commit();
    }

    return image;
}

RunResult run(const std::string& path, const std::vector<std::string>& argv,
              const RunOptions& options, obs::Writer& events) {
    RunResult out;

    auto& start = events.begin(obs::EventKind::SessionStart);
    start.add("target", path);
    start.add("argv_count", static_cast<std::uint64_t>(argv.size()));
    events.commit();

    const parser::ElfImage image = inspect(path, &events, out.load_detail);
    out.load_error = image.error();

    if (!image.ok()) {
        // A target this build cannot run is refused by name. The detail
        // string is the same one the reader produced, so the reason a run
        // was refused and the reason a parse failed cannot disagree.
        out.failed = true;
        out.failure_detail =
            "the image is not one this build can run: " +
            std::string(parser::load_error_name(image.error()));
        events.note(out.failure_detail);
        auto& end = events.begin(obs::EventKind::SessionEnd);
        end.add("started", false);
        events.commit();
        out.events = events.events_written();
        return out;
    }

    out.image_loaded = true;
    out.entry = image.entry();
    out.mapping_count = image.mappings().size();

    for (std::size_t i = 0; i < image.mappings().size(); ++i) {
        report_mapping(events, image.mappings()[i], i);
    }

    // The container configuration comes from the options and the image. The
    // image contributes the one fact the isolation layer cannot derive on
    // its own: a target that asks for an executable stack is a target whose
    // stack request the caller has to decide about, and the decision is
    // recorded here rather than made silently in the loader.
    isolation::ContainerConfig config;
    config.root_dir = options.root_dir;
    config.upper_dir = options.upper_dir;
    config.work_dir = options.work_dir;
    config.root_kind = options.root_kind;
    config.cgroup_parent = options.cgroup_parent;
    config.limits.memory_bytes = options.memory_bytes;
    config.limits.pids = options.pids;
    config.limits.cpu_percent = options.cpu_percent;
    config.extra_mounts = options.extra_mounts;

    // A run with no root named still needs a root. The host's own / is bound
    // read-only, which is the smallest setup that gives the target a
    // coherent filesystem: it can read the libraries and data files it
    // expects, and it cannot write anything, because the bind is sealed
    // before the pivot. Binding an empty path instead would be refused by
    // the kernel with EINVAL, and a caller who did not name a root has not
    // asked for an empty one -- they have asked for the default.
    if (config.root_dir.empty() &&
        config.root_kind == isolation::RootKind::ReadOnlyBind) {
        config.root_dir = "/";
    }

    const std::vector<std::string> env = default_environment(options);

    isolation::SpawnResult spawned =
        isolation::container_spawn(config, path, argv, env);
    if (!spawned.error.ok()) {
        out.failed = true;
        out.error = spawned.error;
        auto& e = events.begin(obs::EventKind::Note);
        // The literal is named as a string_view: without it the overload set
        // would prefer the bool one and the text would become a truth value.
        e.add("text", std::string_view{"the container could not be started"});
        e.add("stage", isolation::stage_name(spawned.error.stage));
        e.add("errno", static_cast<std::int64_t>(spawned.error.error));
        e.add("detail", spawned.error.detail);
        events.commit();
        auto& end = events.begin(obs::EventKind::SessionEnd);
        end.add("started", false);
        events.commit();
        out.events = events.events_written();
        return out;
    }

    auto& spawn = events.begin(obs::EventKind::ProcessSpawn);
    spawn.add("pid", static_cast<std::uint64_t>(spawned.pid));
    spawn.add("entry", static_cast<std::uint64_t>(image.entry()));
    events.commit();

    // The container releases the target as soon as it has been set up, so
    // there is no stopped process to resume and this call is a no-op. It is
    // deliberately not called: the call is implemented as a non-blocking
    // wait, and a wait that observes the target's exit consumes it, which
    // would leave the reap below with nothing to collect and no exit status
    // to report. The sequence a caller sees is still spawn, then reap.

    const isolation::ContainerResult reaped = isolation::container_reap(spawned.pid);

    if (!reaped.error.ok()) {
        out.failed = true;
        out.error = reaped.error;
    }
    out.exit_code = reaped.exit_code;
    out.term_signal = reaped.term_signal;
    out.signaled = reaped.signaled;

    if (reaped.signaled) {
        auto& e = events.begin(obs::EventKind::Signal);
        e.add("pid", static_cast<std::uint64_t>(spawned.pid));
        e.add("signal", static_cast<std::uint64_t>(reaped.term_signal));
        events.commit();
    }

    auto& exit = events.begin(obs::EventKind::ProcessExit);
    exit.add("pid", static_cast<std::uint64_t>(spawned.pid));
    exit.add("exit_code", static_cast<std::int64_t>(reaped.exit_code));
    exit.add("signaled", reaped.signaled);
    exit.add("term_signal", static_cast<std::uint64_t>(reaped.term_signal));
    events.commit();

    (void)isolation::container_cleanup(config);

    auto& end = events.begin(obs::EventKind::SessionEnd);
    end.add("started", true);
    end.add("exit_code", static_cast<std::int64_t>(out.exit_code));
    end.add("signaled", out.signaled);
    end.add("term_signal", static_cast<std::uint64_t>(out.term_signal));
    events.commit();

    out.events = events.events_written();
    return out;
}

} // namespace occ::runner
