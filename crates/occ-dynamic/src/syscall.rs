//! Syscall numbers, names and which arguments are paths.
//!
//! A number alone is not an observation: `openat(-100, 0x7ffd..., 0, 0)` says
//! nothing a reader can use, while `openat(AT_FDCWD, "/etc/ld.so.cache")` says
//! exactly what the program did. The table exists so the tracer can print names
//! without shipping a copy of every header, and the path map exists so the
//! tracer knows which arguments to dereference.
//!
//! Only x86-64 numbers appear here. They are not portable to AArch64, where
//! `openat` is 56 rather than 257, so this table is never used to interpret a
//! trace from another architecture — the tracer refuses those instead.

/// x86-64 syscall numbers with their names, sorted for binary search.
static TABLE: [(u64, &str); 104] = [
    (0, "read"),
    (1, "write"),
    (2, "open"),
    (3, "close"),
    (4, "stat"),
    (5, "fstat"),
    (6, "lstat"),
    (7, "poll"),
    (8, "lseek"),
    (9, "mmap"),
    (10, "mprotect"),
    (11, "munmap"),
    (12, "brk"),
    (13, "rt_sigaction"),
    (14, "rt_sigprocmask"),
    (16, "ioctl"),
    (17, "pread64"),
    (18, "pwrite64"),
    (19, "readv"),
    (20, "writev"),
    (21, "access"),
    (22, "pipe"),
    (23, "select"),
    (24, "sched_yield"),
    (25, "mremap"),
    (28, "madvise"),
    (32, "dup"),
    (33, "dup2"),
    (35, "nanosleep"),
    (39, "getpid"),
    (41, "socket"),
    (42, "connect"),
    (43, "accept"),
    (44, "sendto"),
    (45, "recvfrom"),
    (49, "bind"),
    (50, "listen"),
    (56, "clone"),
    (57, "fork"),
    (58, "vfork"),
    (59, "execve"),
    (60, "exit"),
    (61, "wait4"),
    (62, "kill"),
    (63, "uname"),
    (72, "fcntl"),
    (74, "fsync"),
    (78, "getdents"),
    (79, "getcwd"),
    (80, "chdir"),
    (82, "rename"),
    (83, "mkdir"),
    (84, "rmdir"),
    (85, "creat"),
    (87, "unlink"),
    (89, "readlink"),
    (90, "chmod"),
    (97, "getrlimit"),
    (101, "ptrace"),
    (102, "getuid"),
    (104, "getgid"),
    (105, "setuid"),
    (107, "geteuid"),
    (108, "getegid"),
    (110, "getppid"),
    (131, "sigaltstack"),
    (137, "statfs"),
    (158, "arch_prctl"),
    (186, "gettid"),
    (202, "futex"),
    (217, "getdents64"),
    (218, "set_tid_address"),
    (221, "fadvise64"),
    (228, "clock_gettime"),
    (230, "clock_nanosleep"),
    (231, "exit_group"),
    (232, "epoll_wait"),
    (233, "epoll_ctl"),
    (257, "openat"),
    (258, "mkdirat"),
    (262, "newfstatat"),
    (263, "unlinkat"),
    (264, "renameat"),
    (267, "readlinkat"),
    (268, "fchmodat"),
    (269, "faccessat"),
    (273, "set_robust_list"),
    (281, "epoll_pwait"),
    (288, "accept4"),
    (289, "signalfd4"),
    (290, "eventfd2"),
    (291, "epoll_create1"),
    (292, "dup3"),
    (293, "pipe2"),
    (302, "prlimit64"),
    (316, "renameat2"),
    (318, "getrandom"),
    (322, "execveat"),
    (332, "statx"),
    (334, "rseq"),
    (435, "clone3"),
    (437, "openat2"),
    (439, "faccessat2"),
    (441, "epoll_pwait2"),
];

/// Name for a syscall number.
///
/// Unknown numbers come back as `sys_<n>` rather than as an empty string: the
/// number is still a fact, and hiding it would make an unlisted syscall look
/// like a decode failure.
pub fn name(nr: u64) -> String {
    match TABLE.binary_search_by_key(&nr, |(n, _)| *n) {
        Ok(i) => TABLE[i].1.to_string(),
        Err(_) => format!("sys_{nr}"),
    }
}

/// Which argument indexes of `nr` point at a NUL-terminated path.
///
/// Dereferencing the wrong argument produces a truncated string of garbage that
/// looks like a filename, so this is an explicit list rather than a guess from
/// the syscall's name.
pub const fn path_args(nr: u64) -> &'static [u8] {
    match nr {
        // open, stat, lstat, access, execve, chdir, mkdir, rmdir, creat,
        // unlink, readlink, chmod
        2 | 4 | 6 | 21 | 59 | 80 | 83 | 84 | 85 | 87 | 89 | 90 => &[0],
        // rename takes two paths.
        82 => &[0, 1],
        // AT_* variants: the path follows the dirfd.
        257 | 258 | 262 | 263 | 267 | 268 | 269 | 437 | 439 => &[1],
        264 | 316 => &[1, 3],
        322 => &[1],
        _ => &[],
    }
}

/// True when the call is one this tool can name precisely.
pub const fn is_known(nr: u64) -> bool {
    // Binary search is not const, so this reports the range the table covers.
    nr <= 441
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_sorted_so_binary_search_is_valid() {
        for pair in TABLE.windows(2) {
            assert!(pair[0].0 < pair[1].0, "{:?} is out of order", pair[1]);
        }
    }

    #[test]
    fn names_match_the_x86_64_abi() {
        assert_eq!(name(0), "read");
        assert_eq!(name(59), "execve");
        assert_eq!(name(257), "openat");
        assert_eq!(name(231), "exit_group");
        assert_eq!(name(9999), "sys_9999");
    }

    #[test]
    fn path_arguments_are_listed_only_where_they_exist() {
        assert_eq!(path_args(59), &[0], "execve's first argument is a path");
        assert_eq!(path_args(257), &[1], "openat's path is the second argument");
        assert_eq!(
            path_args(1),
            &[] as &[u8],
            "write takes a buffer, not a path"
        );
    }
}
