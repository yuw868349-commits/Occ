#pragma once

// Filesystem helpers built on the syscall wrappers.
//
// These never go through std::filesystem. That library brings in locale
// machinery and error-code paths that this project does not want, and it
// reports failures in a form that is harder to act on than an errno.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace occ::fs {

[[nodiscard]] bool exists(const std::string& path) noexcept;

[[nodiscard]] bool is_dir(const std::string& path) noexcept;

[[nodiscard]] bool is_writable_dir(const std::string& path) noexcept;

[[nodiscard]] bool is_writable_file(const std::string& path) noexcept;

// Returns the whole file. Bounded: files above the limit are refused rather
// than read, so a surprise multi-gigabyte path cannot exhaust memory.
inline constexpr std::size_t kMaxReadSize = 64u * 1024u * 1024u;

[[nodiscard]] std::optional<std::string> read_file(const std::string& path) noexcept;

[[nodiscard]] std::optional<std::vector<std::uint8_t>>
read_file_bytes(const std::string& path) noexcept;

[[nodiscard]] bool write_file(const std::string& path,
                              const std::string& content) noexcept;

[[nodiscard]] bool mkdir_p(const std::string& path, unsigned int mode) noexcept;

// Removes a file. Directories need remove_tree.
[[nodiscard]] bool remove_file(const std::string& path) noexcept;

// Recursively removes. Refuses to follow symlinks out of the tree it was
// given. Returns the number of entries removed, or -1 on the first error.
[[nodiscard]] long remove_tree(const std::string& path) noexcept;

[[nodiscard]] std::vector<std::string> list_dir(const std::string& path) noexcept;

// Makes a path absolute and normal.
//
// "Absolute" means joined onto the current working directory, and
// "normal" means the "." and ".." components are folded away. Both halves
// matter and neither is optional: a path that is merely made absolute still
// contains "..", and inside a container whose root is not the host's root
// those two forms resolve differently.
//
// The result is the lexical resolution, not the physical one. Symlinks are
// not followed, because a container is about to have its own root and a
// symlink resolved against the host's root is not a path inside it. A
// caller that needs the physical path asks the kernel for it inside the
// container, after the pivot, where the answer is the right one.
[[nodiscard]] std::string absolute_path(const std::string& path) noexcept;

// The current working directory, or an empty string when it cannot be read.
// An empty result is not a usable path and a caller that treats it as one
// would build a path out of nothing.
[[nodiscard]] std::string current_directory() noexcept;

// Reads a symlink target.
[[nodiscard]] std::optional<std::string> read_link(const std::string& path) noexcept;

// True when /proc/filesystems lists the named filesystem type.
[[nodiscard]] bool filesystem_supports(const std::string& fstype) noexcept;

// Probes whether a tmpfs can be mounted here, then unmounts it. Used by occ
// doctor and by the isolation setup's preflight, which needs the answer
// before it has committed to a sequence of mounts.
[[nodiscard]] bool can_mount_tmpfs() noexcept;

// Same idea for overlayfs, which has three directory arguments and fails for
// reasons that vary by kernel and by the filesystem underneath.
[[nodiscard]] bool can_mount_overlay() noexcept;

// Reads a /proc/<pid>/stat field by index, counting from 1 per proc(5).
// Field 22 is the process start time, which is what makes a pid unique.
[[nodiscard]] std::optional<std::uint64_t> proc_stat_field(int pid,
                                                           int field) noexcept;

} // namespace occ::fs
