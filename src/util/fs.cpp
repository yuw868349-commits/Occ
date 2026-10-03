#include "occ/util/fs.h"

#include "occ/syscall/kabi.h"
#include "occ/syscall/syscall.h"
#include "occ/util/string.h"

#include <array>
#include <cstring>

namespace occ::fs {

namespace {

// Directory entries are read with getdents64 into a fixed buffer and parsed
// in place. A per-call dynamic buffer would allocate an unbounded amount for
// a directory with many entries; the fixed buffer bounds the cost of one
// read and the caller loops.
constexpr std::size_t kDirentBufSize = 32u * 1024u;

struct LinuxDirent64 {
    std::uint64_t d_ino;
    std::int64_t d_off;
    unsigned short d_reclen;
    unsigned char d_type;
    char d_name[256];
};

// struct stat as the x86-64 kernel lays it out. The layout is fixed by the
// kernel ABI and does not depend on libc feature macros, which is why it is
// written out rather than taken from <sys/stat.h>: the header's definition
// is guarded by the large-file and time64 feature tests, and a mismatch is
// a silent misread rather than a compile error.
struct StatBuf {
    std::uint64_t st_dev;
    std::uint64_t st_ino;
    std::uint64_t st_nlink;
    std::uint32_t st_mode;
    std::uint32_t st_uid;
    std::uint32_t st_gid;
    std::uint32_t __pad0;
    std::uint64_t st_rdev;
    std::int64_t st_size;
    std::int64_t st_blksize;
    std::int64_t st_blocks;
    std::int64_t st_atime;
    std::int64_t st_atime_nsec;
    std::int64_t st_mtime;
    std::int64_t st_mtime_nsec;
    std::int64_t st_ctime;
    std::int64_t st_ctime_nsec;
    std::int64_t __unused[3];
};

static_assert(sizeof(StatBuf) == 144,
              "struct stat layout does not match the x86-64 kernel ABI");

int open_readonly(const std::string& path) {
    auto r = sys::openat(-100 /* AT_FDCWD */, path.c_str(), 0 /* O_RDONLY */,
                         0);
    return r.ok() ? static_cast<int>(r.value) : -r.error;
}

int open_dir(const std::string& path) {
    auto r = sys::openat(-100, path.c_str(), 0200000 /* O_DIRECTORY */, 0);
    return r.ok() ? static_cast<int>(r.value) : -r.error;
}

std::string strip_trailing_slash(std::string path) {
    while (path.size() > 1 && path.back() == '/') {
        path.pop_back();
    }
    return path;
}

} // namespace

bool exists(const std::string& path) noexcept {
    // newfstatat with a null stat buffer is only valid together with
    // AT_EMPTY_PATH, and it returns EFAULT rather than a useful result
    // otherwise. Passing a real buffer is the version that works on every
    // kernel this project targets.
    StatBuf st{};
    auto r = sys::newfstatat(-100, path.c_str(), &st, 0);
    return r.ok();
}

bool is_dir(const std::string& path) noexcept {
    StatBuf st{};

    auto r = sys::newfstatat(-100, path.c_str(), &st, 0);
    if (r.failed()) {
        return false;
    }
    constexpr std::uint32_t kSIfmt = 0170000;
    constexpr std::uint32_t kSIfdir = 0040000;
    return (st.st_mode & kSIfmt) == kSIfdir;
}

bool is_writable_dir(const std::string& path) noexcept {
    auto r = sys::faccessat2(-100, path.c_str(), 2 /* W_OK */, 0);
    return r.ok();
}

bool is_writable_file(const std::string& path) noexcept {
    return is_writable_dir(path);
}

std::optional<std::string> read_file(const std::string& path) noexcept {
    const int fd = open_readonly(path);
    if (fd < 0) {
        return std::nullopt;
    }

    std::string out;
    std::array<char, 8192> buf{};

    for (;;) {
        auto r = sys::read(fd, buf.data(), buf.size());
        if (r.failed()) {
            (void)sys::close(fd);
            return std::nullopt;
        }
        if (r.value == 0) {
            break;
        }
        if (out.size() + static_cast<std::size_t>(r.value) > kMaxReadSize) {
            (void)sys::close(fd);
            return std::nullopt;
        }
        out.append(buf.data(), static_cast<std::size_t>(r.value));
    }

    (void)sys::close(fd);
    return out;
}

std::optional<std::vector<std::uint8_t>>
read_file_bytes(const std::string& path) noexcept {
    auto text = read_file(path);
    if (!text) {
        return std::nullopt;
    }
    std::vector<std::uint8_t> out(text->size());
    if (!text->empty()) {
        std::memcpy(out.data(), text->data(), text->size());
    }
    return out;
}

bool write_file(const std::string& path, const std::string& content) noexcept {
    auto o = sys::openat(-100, path.c_str(), 01 | 0100 | 01000 /* O_WRONLY |
                                                                  O_CREAT |
                                                                  O_TRUNC */
                         ,
                         0644);
    if (o.failed()) {
        return false;
    }
    const int fd = static_cast<int>(o.value);

    std::size_t off = 0;
    while (off < content.size()) {
        auto w = sys::write(fd, content.data() + off, content.size() - off);
        if (w.failed() || w.value == 0) {
            (void)sys::close(fd);
            return false;
        }
        off += static_cast<std::size_t>(w.value);
    }

    (void)sys::close(fd);
    return true;
}

bool mkdir_p(const std::string& path, unsigned int mode) noexcept {
    if (path.empty()) {
        return false;
    }

    std::string cur;
    cur.reserve(path.size());

    std::size_t i = 0;
    if (path[0] == '/') {
        cur.push_back('/');
        i = 1;
    }

    while (i <= path.size()) {
        if (i == path.size() || path[i] == '/') {
            if (!cur.empty() && exists(cur) && is_dir(cur)) {
                // Already present. Continue into it.
            } else if (!cur.empty() && !exists(cur)) {
                auto r = sys::mkdirat(-100, cur.c_str(), mode);
                if (r.failed() && r.error != 17 /* EEXIST */) {
                    return false;
                }
            }
            if (i < path.size()) {
                if (cur.size() > 1) {
                    cur.push_back('/');
                }
            }
        } else {
            cur.push_back(path[i]);
        }
        ++i;
    }
    return true;
}

bool remove_file(const std::string& path) noexcept {
    auto r = sys::unlinkat(-100, path.c_str(), 0);
    return r.ok();
}

long remove_tree(const std::string& path) noexcept {
    const int fd = open_dir(path);
    if (fd < 0) {
        return -1;
    }

    std::array<char, kDirentBufSize> buf{};
    long removed = 0;

    for (;;) {
        auto r = sys::getdents64(fd, buf.data(), buf.size());
        if (r.failed() || r.value == 0) {
            break;
        }

        std::size_t off = 0;
        while (off < static_cast<std::size_t>(r.value)) {
            auto* d = reinterpret_cast<const LinuxDirent64*>(buf.data() + off);
            const std::string name(d->d_name);

            if (name != "." && name != "..") {
                const std::string child =
                    strip_trailing_slash(path) + "/" + name;
                constexpr unsigned char kDtDir = 4;
                if (d->d_type == kDtDir) {
                    const long n = remove_tree(child);
                    if (n < 0) {
                        (void)sys::close(fd);
                        return -1;
                    }
                    removed += n;
                } else {
                    (void)sys::unlinkat(-100, child.c_str(), 0);
                    ++removed;
                }
            }
            off += d->d_reclen;
        }
    }

    (void)sys::close(fd);

    auto u = sys::unlinkat(-100, path.c_str(), 0200 /* AT_REMOVEDIR */);
    if (u.ok()) {
        ++removed;
    }

    return removed;
}

std::vector<std::string> list_dir(const std::string& path) noexcept {
    std::vector<std::string> out;

    const int fd = open_dir(path);
    if (fd < 0) {
        return out;
    }

    std::array<char, kDirentBufSize> buf{};
    for (;;) {
        auto r = sys::getdents64(fd, buf.data(), buf.size());
        if (r.failed() || r.value == 0) {
            break;
        }

        std::size_t off = 0;
        while (off < static_cast<std::size_t>(r.value)) {
            auto* d = reinterpret_cast<const LinuxDirent64*>(buf.data() + off);
            const std::string name(d->d_name);
            if (name != "." && name != "..") {
                out.push_back(name);
            }
            off += d->d_reclen;
        }
    }

    (void)sys::close(fd);
    return out;
}

std::optional<std::string> read_link(const std::string& path) noexcept {
    std::array<char, 4096> buf{};
    auto r = sys::readlinkat(-100, path.c_str(), buf.data(), buf.size() - 1);
    if (r.failed()) {
        return std::nullopt;
    }
    return std::string(buf.data(), static_cast<std::size_t>(r.value));
}

bool filesystem_supports(const std::string& fstype) noexcept {
    auto content = read_file("/proc/filesystems");
    if (!content) {
        return false;
    }
    for (const auto line : split(*content, '\n')) {
        const auto t = trim(line);
        if (t.empty()) {
            continue;
        }
        // The first column is either "nodev" or a device prefix. The
        // filesystem name is the last whitespace-separated field.
        const auto tab = t.find_last_of(' ');
        const auto name = (tab == std::string_view::npos) ? t : t.substr(tab + 1);
        if (name == fstype) {
            return true;
        }
    }
    return false;
}

bool can_mount_tmpfs() noexcept {
    // A real mount attempt, into a private namespace, cleaned up afterwards.
    // Reading a capability bit tells you what the process was granted; it
    // does not tell you that the kernel will accept the mount, and those two
    // answers differ on a machine where a security module is enforcing.
    const char* dir = "/tmp/occ-doctor-mnt";
    if (!exists(dir) && !mkdir_p(dir, 0755)) {
        return false;
    }

    auto unshare_r = sys::unshare(0x00020000 /* CLONE_NEWNS */);
    if (unshare_r.failed()) {
        return false;
    }

    // Make the whole tree private so the probe mount cannot propagate.
    (void)sys::mount(nullptr, "/", nullptr, sys::kMsRecursive | sys::kMsPrivate, nullptr);

    auto m = sys::mount("tmpfs", dir, "tmpfs", sys::kMsNosuid | sys::kMsNodev, nullptr);
    if (m.failed()) {
        return false;
    }

    (void)sys::umount2(dir, sys::kMntDetach);
    return true;
}

bool can_mount_overlay() noexcept {
    const char* base = "/tmp/occ-doctor-ovl";
    const std::string lower = std::string(base) + "/l";
    const std::string upper = std::string(base) + "/u";
    const std::string work = std::string(base) + "/w";
    const std::string merged = std::string(base) + "/m";

    for (const auto& d : {base, lower.c_str(), upper.c_str(), work.c_str(),
                          merged.c_str()}) {
        if (!exists(d) && !mkdir_p(d, 0755)) {
            return false;
        }
    }

    const std::string opts =
        "lowerdir=" + lower + ",upperdir=" + upper + ",workdir=" + work;

    auto m = sys::mount("overlay", merged.c_str(), "overlay", 0, opts.c_str());
    if (m.failed()) {
        (void)remove_tree(base);
        return false;
    }

    (void)sys::umount2(merged.c_str(), sys::kMntDetach);
    (void)remove_tree(base);
    return true;
}

std::optional<std::uint64_t> proc_stat_field(int pid, int field) noexcept {
    const std::string path = "/proc/" + std::to_string(pid) + "/stat";
    auto content = read_file(path);
    if (!content) {
        return std::nullopt;
    }

    // Field 2 is the command name wrapped in parentheses and it may contain
    // spaces and parentheses. Everything after the last ')' is space
    // separated, so counting starts there.
    const auto rparen = content->rfind(')');
    if (rparen == std::string::npos) {
        return std::nullopt;
    }

    int index = 2;
    std::size_t pos = rparen + 1;
    while (pos < content->size()) {
        while (pos < content->size() && (*content)[pos] == ' ') {
            ++pos;
        }
        const std::size_t start = pos;
        while (pos < content->size() && (*content)[pos] != ' ') {
            ++pos;
        }
        if (start == pos) {
            break;
        }
        ++index;
        if (index == field) {
            std::uint64_t value = 0;
            const auto view =
                std::string_view(*content).substr(start, pos - start);
            for (const char c : view) {
                if (c < '0' || c > '9') {
                    return std::nullopt;
                }
                value = value * 10 + static_cast<std::uint64_t>(c - '0');
            }
            return value;
        }
    }

    return std::nullopt;
}

} // namespace occ::fs
