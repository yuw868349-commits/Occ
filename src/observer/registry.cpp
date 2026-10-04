#include "occ/observer/registry.h"

#include "occ/syscall/errno.h"
#include "occ/syscall/syscall.h"
#include "occ/util/fs.h"
#include "occ/util/string.h"

#include <cstdlib>

#include <signal.h>
#include <unistd.h>

namespace occ::obs {

namespace {

constexpr const char* kRegistryDir = "/run/occ/sessions";

// A session id is the pid and the start time, and the pair is what makes it
// unique. A pid alone is reused by the kernel, so an id built from one
// would eventually name a different process, and an attach would reach the
// wrong thing.
std::string make_id(int pid, std::uint64_t started_ns) {
    std::string out;
    append_hex(out, static_cast<std::uint64_t>(pid), 8);
    out.push_back('-');
    append_hex(out, started_ns, 16);
    return out;
}

struct Timespec {
    long sec;
    long nsec;
};

std::uint64_t now_ns() {
    Timespec ts{};
    auto r = sys::clock_gettime(1, &ts);
    if (r.failed()) {
        return 0;
    }
    return static_cast<std::uint64_t>(ts.sec) * 1000000000ULL +
           static_cast<std::uint64_t>(ts.nsec);
}

std::string entry_path(const std::string& id) {
    return std::string(kRegistryDir) + "/" + id + ".session";
}

// The record is a flat key=value file with one field per line. It is read
// by hand often enough that a format a person can read at a terminal is
// worth more than a compact one, and there is no field in it that needs
// escaping beyond a newline, which a path can contain only if someone
// deliberately put one there.
std::string encode(const SessionRecord& r) {
    std::string out;
    out += "id=";
    out += r.id;
    out += "\npid=";
    append_uint(out, static_cast<std::uint64_t>(r.pid));
    out += "\ntarget=";
    out += r.target;
    out += "\nstarted=";
    append_uint(out, r.started_ns);
    out += "\nstream=";
    out += r.stream_path;
    out += "\n";
    return out;
}

bool decode(const std::string& text, const std::string& id,
            const std::string& dir, SessionRecord& out) {
    out = SessionRecord{};
    out.id = id;
    out.directory = dir;

    std::size_t pos = 0;
    while (pos < text.size()) {
        std::size_t end = text.find('\n', pos);
        if (end == std::string::npos) {
            end = text.size();
        }
        const std::string_view line(text.data() + pos, end - pos);
        const std::size_t eq = line.find('=');
        if (eq != std::string_view::npos) {
            const std::string_view key = line.substr(0, eq);
            const std::string_view value = line.substr(eq + 1);
            if (key == "pid") {
                std::uint64_t v = 0;
                for (char c : value) {
                    if (c < '0' || c > '9') {
                        break;
                    }
                    v = v * 10 + static_cast<std::uint64_t>(c - '0');
                }
                out.pid = static_cast<int>(v);
            } else if (key == "target") {
                out.target.assign(value);
            } else if (key == "started") {
                std::uint64_t v = 0;
                for (char c : value) {
                    if (c < '0' || c > '9') {
                        break;
                    }
                    v = v * 10 + static_cast<std::uint64_t>(c - '0');
                }
                out.started_ns = v;
            } else if (key == "stream") {
                out.stream_path.assign(value);
            }
        }
        pos = end + 1;
    }
    return out.pid > 0;
}

} // namespace

std::string registry_root() noexcept {
    // The runtime directory is the right home for this: it is cleared on
    // boot, which is exactly the lifetime of a session record, and it is
    // not world-writable, so another user cannot register a session that
    // this one would then act on.
    (void)fs::mkdir_p(kRegistryDir, 0700);
    return kRegistryDir;
}

bool process_alive(int pid) noexcept {
    if (pid <= 0) {
        return false;
    }
    // Signal zero performs the permission and existence checks without
    // delivering anything. EPERM means the process exists and belongs to
    // someone else, which is still alive.
    auto r = sys::kill(pid, 0);
    if (r.ok()) {
        return true;
    }
    return r.error == sys::kEperm;
}

std::string register_session(const std::string& target, int pid,
                             const std::string& stream_path,
                             std::string& error) noexcept {
    (void)registry_root();
    error.clear();

    SessionRecord r;
    const std::uint64_t started = now_ns();
    r.id = make_id(pid, started);
    r.pid = pid;
    r.target = target;
    r.started_ns = started;
    r.stream_path = stream_path;

    const std::string path = entry_path(r.id);
    if (!fs::write_file(path, encode(r))) {
        error = "the session record could not be written to " + path;
        return {};
    }
    return r.id;
}

bool update_session_pid(const std::string& id, int pid,
                        std::string& error) noexcept {
    error.clear();
    if (id.empty()) {
        return false;
    }

    const std::string path = entry_path(id);
    auto content = fs::read_file(path);
    if (!content) {
        error = "no record for session " + id;
        return false;
    }

    std::string text;
    std::size_t pos = 0;
    while (pos < content->size()) {
        std::size_t end = content->find('\n', pos);
        if (end == std::string::npos) {
            end = content->size();
        }
        const std::string_view line(content->data() + pos, end - pos);
        if (starts_with(line, "pid=")) {
            text += "pid=";
            append_uint(text, static_cast<std::uint64_t>(pid));
        } else {
            text.append(line);
        }
        text.push_back('\n');
        pos = end + 1;
    }

    if (!fs::write_file(path, text)) {
        error = "the session record could not be updated";
        return false;
    }
    return true;
}

void unregister_session(const std::string& id) noexcept {
    if (id.empty()) {
        return;
    }
    (void)fs::remove_file(entry_path(id));
}

std::vector<SessionRecord> live_sessions() noexcept {
    std::vector<SessionRecord> out;
    const std::string root = registry_root();

    for (const std::string& name : fs::list_dir(root)) {
        if (!ends_with(name, ".session")) {
            continue;
        }
        const std::string path = root + "/" + name;
        auto content = fs::read_file(path);
        if (!content) {
            continue;
        }

        const std::string id = name.substr(0, name.size() - 8);
        SessionRecord r;
        if (!decode(*content, id, root, r)) {
            // A record that does not parse is one this build did not write.
            // Removing it would lose whatever wrote it, so it is skipped
            // and left for a person to look at.
            continue;
        }

        if (!process_alive(r.pid)) {
            // The process is gone, so the record is stale. Removing it here
            // rather than leaving it means the registry is self-cleaning
            // and no caller has to check liveness itself.
            (void)fs::remove_file(path);
            continue;
        }

        out.push_back(r);
    }
    return out;
}

bool find_session(const std::string& key, bool by_pid,
                  SessionRecord& out) noexcept {
    for (const auto& r : live_sessions()) {
        if (by_pid) {
            std::string as_text;
            append_uint(as_text, static_cast<std::uint64_t>(r.pid));
            if (as_text == key) {
                out = r;
                return true;
            }
        } else if (r.id == key) {
            out = r;
            return true;
        }
    }
    return false;
}

} // namespace occ::obs
