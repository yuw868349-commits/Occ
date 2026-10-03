#include "occ/syscall/syscall.h"

// Syscall numbers come from <sys/syscall.h>. They are not written out by
// hand here: a hand-maintained table of numbers is a table that will be
// wrong on the one architecture nobody tested, and the toolchain already
// carries the authoritative list.

#include <sys/syscall.h>

namespace occ::sys {

namespace {

constexpr long N(long nr) noexcept { return nr; }

} // namespace

// ---------------------------------------------------------------- processes

Result clone3_raw(const void* cl_args, std::size_t size) noexcept {
    return detail::call2(N(SYS_clone3), detail::arg(cl_args),
                         detail::arg(size));
}

Result fork(void) noexcept { return detail::call0(N(SYS_fork)); }

Result execve(const char* path, char* const argv[],
              char* const envp[]) noexcept {
    return detail::call3(N(SYS_execve), detail::arg(path),
                         detail::arg(argv), detail::arg(envp));
}

Result execveat(int dirfd, const char* path, char* const argv[],
                char* const envp[], int flags) noexcept {
    return detail::call5(N(SYS_execveat), dirfd, detail::arg(path),
                         detail::arg(argv), detail::arg(envp), flags);
}

Result wait4(int pid, int* status, int options, void* rusage) noexcept {
    return detail::call4(N(SYS_wait4), pid, detail::arg(status), options,
                         detail::arg(rusage));
}

Result getpid(void) noexcept { return detail::call0(N(SYS_getpid)); }

Result gettid(void) noexcept { return detail::call0(N(SYS_gettid)); }

Result getppid(void) noexcept { return detail::call0(N(SYS_getppid)); }

Result getuid_syscall(void) noexcept { return detail::call0(N(SYS_getuid)); }

Result getgid_syscall(void) noexcept { return detail::call0(N(SYS_getgid)); }

Result geteuid(void) noexcept { return detail::call0(N(SYS_geteuid)); }

Result getegid(void) noexcept { return detail::call0(N(SYS_getegid)); }

Result exit_group(int status) noexcept {
    return detail::call1(N(SYS_exit_group), status);
}

Result kill(int pid, int sig) noexcept {
    return detail::call2(N(SYS_kill), pid, sig);
}

Result tgkill(int tgid, int tid, int sig) noexcept {
    return detail::call3(N(SYS_tgkill), tgid, tid, sig);
}

Result prctl(int option, unsigned long a2, unsigned long a3, unsigned long a4,
             unsigned long a5) noexcept {
    return detail::call5(N(SYS_prctl), option, static_cast<long>(a2),
                         static_cast<long>(a3), static_cast<long>(a4),
                         static_cast<long>(a5));
}

// -------------------------------------------------------------------- files

Result openat(int dirfd, const char* path, int flags,
              unsigned int mode) noexcept {
    return detail::call4(N(SYS_openat), dirfd, detail::arg(path), flags,
                         mode);
}

Result openat2(int dirfd, const char* path, const void* how,
               std::size_t size) noexcept {
    return detail::call4(N(SYS_openat2), dirfd, detail::arg(path),
                         detail::arg(how), detail::arg(size));
}

Result close(int fd) noexcept { return detail::call1(N(SYS_close), fd); }

Result close_range(unsigned int first, unsigned int last, int flags) noexcept {
    return detail::call3(N(SYS_close_range), first, last, flags);
}

Result read(int fd, void* buf, std::size_t count) noexcept {
    return detail::call3(N(SYS_read), fd, detail::arg(buf),
                         detail::arg(count));
}

Result write(int fd, const void* buf, std::size_t count) noexcept {
    return detail::call3(N(SYS_write), fd, detail::arg(buf),
                         detail::arg(count));
}

Result pread64(int fd, void* buf, std::size_t count, long offset) noexcept {
    return detail::call4(N(SYS_pread64), fd, detail::arg(buf),
                         detail::arg(count), offset);
}

Result pwrite64(int fd, const void* buf, std::size_t count,
                long offset) noexcept {
    return detail::call4(N(SYS_pwrite64), fd, detail::arg(buf),
                         detail::arg(count), offset);
}

Result lseek(int fd, long offset, int whence) noexcept {
    return detail::call3(N(SYS_lseek), fd, offset, whence);
}

Result dup(int fd) noexcept { return detail::call1(N(SYS_dup), fd); }

Result dup3(int oldfd, int newfd, int flags) noexcept {
    return detail::call3(N(SYS_dup3), oldfd, newfd, flags);
}

Result fcntl(int fd, int cmd, long arg) noexcept {
    return detail::call3(N(SYS_fcntl), fd, cmd, arg);
}

Result ioctl(int fd, unsigned long request, void* arg) noexcept {
    return detail::call3(N(SYS_ioctl), fd, static_cast<long>(request),
                         detail::arg(arg));
}

Result fstat(int fd, void* statbuf) noexcept {
    return detail::call2(N(SYS_fstat), fd, detail::arg(statbuf));
}

Result newfstatat(int dirfd, const char* path, void* statbuf,
                  int flags) noexcept {
    return detail::call4(N(SYS_newfstatat), dirfd, detail::arg(path),
                         detail::arg(statbuf), flags);
}

Result statx(int dirfd, const char* path, int flags, unsigned int mask,
             void* statxbuf) noexcept {
    return detail::call5(N(SYS_statx), dirfd, detail::arg(path), flags, mask,
                         detail::arg(statxbuf));
}

Result statfs(const char* path, void* buf) noexcept {
    return detail::call2(N(SYS_statfs), detail::arg(path),
                         detail::arg(buf));
}

Result fstatfs(int fd, void* buf) noexcept {
    return detail::call2(N(SYS_fstatfs), fd, detail::arg(buf));
}

Result access(const char* path, int mode) noexcept {
    return detail::call2(N(SYS_access), detail::arg(path), mode);
}

Result faccessat2(int dirfd, const char* path, int mode, int flags) noexcept {
    return detail::call4(N(SYS_faccessat2), dirfd, detail::arg(path), mode,
                         flags);
}

Result readlinkat(int dirfd, const char* path, char* buf,
                  std::size_t size) noexcept {
    return detail::call4(N(SYS_readlinkat), dirfd, detail::arg(path),
                         detail::arg(buf), detail::arg(size));
}

Result getdents64(int fd, void* buf, std::size_t size) noexcept {
    return detail::call3(N(SYS_getdents64), fd, detail::arg(buf),
                         detail::arg(size));
}

Result mkdirat(int dirfd, const char* path, unsigned int mode) noexcept {
    return detail::call3(N(SYS_mkdirat), dirfd, detail::arg(path), mode);
}

Result unlinkat(int dirfd, const char* path, int flags) noexcept {
    return detail::call3(N(SYS_unlinkat), dirfd, detail::arg(path), flags);
}

Result renameat2(int olddirfd, const char* oldpath, int newdirfd,
                 const char* newpath, unsigned int flags) noexcept {
    return detail::call5(N(SYS_renameat2), olddirfd, detail::arg(oldpath),
                         newdirfd, detail::arg(newpath), flags);
}

Result ftruncate(int fd, long length) noexcept {
    return detail::call2(N(SYS_ftruncate), fd, length);
}

Result fsync(int fd) noexcept { return detail::call1(N(SYS_fsync), fd); }

Result pipe2(int pipefd[2], int flags) noexcept {
    return detail::call2(N(SYS_pipe2), detail::arg(pipefd), flags);
}

Result mount_setattr(int dfd, const char* path, unsigned int attr_flags,
                     void* attr, std::size_t size) noexcept {
    return detail::call5(N(SYS_mount_setattr), dfd, detail::arg(path),
                         attr_flags, detail::arg(attr),
                         detail::arg(size));
}

// ------------------------------------------------------------------- memory

Result mmap(void* addr, std::size_t length, int prot, int flags, int fd,
            long offset) noexcept {
    return detail::call6(N(SYS_mmap), detail::arg(addr),
                         detail::arg(length), prot, flags, fd, offset);
}

Result munmap(void* addr, std::size_t length) noexcept {
    return detail::call2(N(SYS_munmap), detail::arg(addr),
                         detail::arg(length));
}

Result mprotect(void* addr, std::size_t len, int prot) noexcept {
    return detail::call3(N(SYS_mprotect), detail::arg(addr),
                         detail::arg(len), prot);
}

Result mremap(void* old_address, std::size_t old_size, std::size_t new_size,
              int flags, void* new_address) noexcept {
    return detail::call5(N(SYS_mremap), detail::arg(old_address),
                         detail::arg(old_size),
                         detail::arg(new_size), flags,
                         detail::arg(new_address));
}

Result memfd_create(const char* name, unsigned int flags) noexcept {
    return detail::call2(N(SYS_memfd_create), detail::arg(name), flags);
}

Result process_vm_readv(int pid, const void* local_iov, unsigned long liovcnt,
                        const void* remote_iov, unsigned long riovcnt,
                        unsigned long flags) noexcept {
    return detail::call6(N(SYS_process_vm_readv), pid,
                         detail::arg(local_iov), detail::arg(liovcnt),
                         detail::arg(remote_iov), detail::arg(riovcnt), detail::arg(flags));
}

Result process_vm_writev(int pid, const void* local_iov, unsigned long liovcnt,
                         const void* remote_iov, unsigned long riovcnt,
                         unsigned long flags) noexcept {
    return detail::call6(N(SYS_process_vm_writev), pid,
                         detail::arg(local_iov), detail::arg(liovcnt),
                         detail::arg(remote_iov), detail::arg(riovcnt), detail::arg(flags));
}

// ------------------------------------------------------------------- mounts

Result mount(const char* source, const char* target, const char* fstype,
             unsigned long flags, const void* data) noexcept {
    return detail::call5(N(SYS_mount), detail::arg(source),
                         detail::arg(target), detail::arg(fstype), detail::arg(flags),
                         detail::arg(data));
}

Result umount2(const char* target, int flags) noexcept {
    return detail::call2(N(SYS_umount2), detail::arg(target), flags);
}

Result pivot_root(const char* new_root, const char* put_old) noexcept {
    return detail::call2(N(SYS_pivot_root), detail::arg(new_root),
                         detail::arg(put_old));
}

Result chdir(const char* path) noexcept {
    return detail::call1(N(SYS_chdir), detail::arg(path));
}

Result fchdir(int fd) noexcept { return detail::call1(N(SYS_fchdir), fd); }

Result chroot(const char* path) noexcept {
    return detail::call1(N(SYS_chroot), detail::arg(path));
}

Result chmod(const char* path, unsigned int mode) noexcept {
    return detail::call2(N(SYS_chmod), detail::arg(path), mode);
}

Result fchmod(int fd, unsigned int mode) noexcept {
    return detail::call2(N(SYS_fchmod), fd, mode);
}

Result chown(const char* path, unsigned int uid, unsigned int gid) noexcept {
    return detail::call3(N(SYS_chown), detail::arg(path), uid, gid);
}

Result fchown(int fd, unsigned int uid, unsigned int gid) noexcept {
    return detail::call3(N(SYS_fchown), fd, uid, gid);
}

// ------------------------------------------------------------------ signals

Result rt_sigaction(int signum, const void* act, void* oldact,
                    std::size_t sigsetsize) noexcept {
    return detail::call4(N(SYS_rt_sigaction), signum, detail::arg(act),
                         detail::arg(oldact),
                         detail::arg(sigsetsize));
}

Result rt_sigprocmask(int how, const void* set, void* oldset,
                      std::size_t sigsetsize) noexcept {
    return detail::call4(N(SYS_rt_sigprocmask), how, detail::arg(set),
                         detail::arg(oldset),
                         detail::arg(sigsetsize));
}

Result rt_sigreturn(void) noexcept {
    return detail::call0(N(SYS_rt_sigreturn));
}

Result rt_sigtimedwait(const void* set, void* info, const void* timeout,
                       std::size_t sigsetsize) noexcept {
    return detail::call4(N(SYS_rt_sigtimedwait), detail::arg(set),
                         detail::arg(info), detail::arg(timeout),
                         detail::arg(sigsetsize));
}

Result signalfd4(int fd, const void* mask, std::size_t sizemask,
                 int flags) noexcept {
    return detail::call4(N(SYS_signalfd4), fd, detail::arg(mask),
                         detail::arg(sizemask), flags);
}

Result sigaltstack(const void* ss, void* old_ss) noexcept {
    return detail::call2(N(SYS_sigaltstack), detail::arg(ss),
                         detail::arg(old_ss));
}

// ------------------------------------------------------------------ tracing

Result ptrace(unsigned long request, int pid, void* addr, void* data) noexcept {
    return detail::call4(N(SYS_ptrace), detail::arg(request), pid, detail::arg(addr),
                         detail::arg(data));
}

Result perf_event_open(const void* attr, int pid, int cpu, int group_fd,
                       unsigned long flags) noexcept {
    return detail::call5(N(SYS_perf_event_open), detail::arg(attr), pid, cpu,
                         group_fd, detail::arg(flags));
}

Result bpf(int cmd, const void* attr, unsigned int size) noexcept {
    return detail::call3(N(SYS_bpf), cmd, detail::arg(attr), detail::arg(size));
}

// -------------------------------------------------------------------- sched

Result sched_setaffinity(int pid, std::size_t cpusetsize,
                         const void* mask) noexcept {
    return detail::call3(N(SYS_sched_setaffinity), pid,
                         detail::arg(cpusetsize), detail::arg(mask));
}

Result sched_getaffinity(int pid, std::size_t cpusetsize,
                         void* mask) noexcept {
    return detail::call3(N(SYS_sched_getaffinity), pid,
                         detail::arg(cpusetsize), detail::arg(mask));
}

Result setpriority(int which, int who, int prio) noexcept {
    return detail::call3(N(SYS_setpriority), which, who, prio);
}

Result getpriority(int which, int who) noexcept {
    return detail::call2(N(SYS_getpriority), which, who);
}

Result unshare(int flags) noexcept {
    return detail::call1(N(SYS_unshare), flags);
}

Result setns(int fd, int nstype) noexcept {
    return detail::call2(N(SYS_setns), fd, nstype);
}

// --------------------------------------------------------------------- misc

Result uname(void* buf) noexcept {
    return detail::call1(N(SYS_uname), detail::arg(buf));
}

Result sysinfo(void* info) noexcept {
    return detail::call1(N(SYS_sysinfo), detail::arg(info));
}

Result getrandom(void* buf, std::size_t buflen, unsigned int flags) noexcept {
    return detail::call3(N(SYS_getrandom), detail::arg(buf),
                         detail::arg(buflen), flags);
}

Result clock_gettime(int clockid, void* tp) noexcept {
    return detail::call2(N(SYS_clock_gettime), clockid, detail::arg(tp));
}

Result nanosleep(const void* req, void* rem) noexcept {
    return detail::call2(N(SYS_nanosleep), detail::arg(req),
                         detail::arg(rem));
}

Result prlimit64(int pid, int resource, const void* newlim,
                 void* oldlim) noexcept {
    return detail::call4(N(SYS_prlimit64), pid, resource,
                         detail::arg(newlim), detail::arg(oldlim));
}

Result socket(int domain, int type, int protocol) noexcept {
    return detail::call3(N(SYS_socket), domain, type, protocol);
}

Result bind(int sockfd, const void* addr, unsigned int addrlen) noexcept {
    return detail::call3(N(SYS_bind), sockfd, detail::arg(addr), addrlen);
}

Result listen(int sockfd, int backlog) noexcept {
    return detail::call2(N(SYS_listen), sockfd, backlog);
}

Result accept4(int sockfd, void* addr, void* addrlen, int flags) noexcept {
    return detail::call4(N(SYS_accept4), sockfd, detail::arg(addr),
                         detail::arg(addrlen), flags);
}

Result connect(int sockfd, const void* addr, unsigned int addrlen) noexcept {
    return detail::call3(N(SYS_connect), sockfd, detail::arg(addr), addrlen);
}

Result setsockopt(int sockfd, int level, int optname, const void* optval,
                  unsigned int optlen) noexcept {
    return detail::call5(N(SYS_setsockopt), sockfd, level, optname,
                         detail::arg(optval), optlen);
}

Result getsockopt(int sockfd, int level, int optname, void* optval,
                  void* optlen) noexcept {
    return detail::call5(N(SYS_getsockopt), sockfd, level, optname,
                         detail::arg(optval), detail::arg(optlen));
}

Result socketpair(int domain, int type, int protocol, int sv[2]) noexcept {
    return detail::call4(N(SYS_socketpair), domain, type, protocol,
                         detail::arg(sv));
}

Result sendmsg(int sockfd, const void* msg, int flags) noexcept {
    return detail::call3(N(SYS_sendmsg), sockfd, detail::arg(msg), flags);
}

Result recvmsg(int sockfd, void* msg, int flags) noexcept {
    return detail::call3(N(SYS_recvmsg), sockfd, detail::arg(msg), flags);
}

Result shutdown(int sockfd, int how) noexcept {
    return detail::call2(N(SYS_shutdown), sockfd, how);
}

Result setsid(void) noexcept { return detail::call0(N(SYS_setsid)); }

Result setresuid(unsigned int ruid, unsigned int euid,
                 unsigned int suid) noexcept {
    return detail::call3(N(SYS_setresuid), ruid, euid, suid);
}

Result setresgid(unsigned int rgid, unsigned int egid,
                 unsigned int sgid) noexcept {
    return detail::call3(N(SYS_setresgid), rgid, egid, sgid);
}

Result setgroups(std::size_t size, const void* list) noexcept {
    return detail::call2(N(SYS_setgroups), detail::arg(size),
                         detail::arg(list));
}

Result capget(void* hdrp, void* datap) noexcept {
    return detail::call2(N(SYS_capget), detail::arg(hdrp),
                         detail::arg(datap));
}

Result capset(void* hdrp, const void* datap) noexcept {
    return detail::call2(N(SYS_capset), detail::arg(hdrp),
                         detail::arg(datap));
}

Result getpgid(int pid) noexcept { return detail::call1(N(SYS_getpgid), pid); }

Result setpgid(int pid, int pgid) noexcept {
    return detail::call2(N(SYS_setpgid), pid, pgid);
}

Result umask_syscall(unsigned int mask) noexcept {
    return detail::call1(N(SYS_umask), detail::arg(mask));
}

Result readlink(const char* path, char* buf, std::size_t size) noexcept {
    return detail::call3(N(SYS_readlink), detail::arg(path), detail::arg(buf),
                         detail::arg(size));
}

Result symlink(const char* target, const char* linkpath) noexcept {
    return detail::call2(N(SYS_symlink), detail::arg(target),
                         detail::arg(linkpath));
}

Result mknodat(int dirfd, const char* path, unsigned int mode,
               unsigned long dev) noexcept {
    return detail::call4(N(SYS_mknodat), dirfd, detail::arg(path),
                         detail::arg(mode), detail::arg(dev));
}

Result mknod(const char* path, unsigned int mode, unsigned long dev) noexcept {
    return detail::call3(N(SYS_mknod), detail::arg(path), detail::arg(mode),
                         detail::arg(dev));
}

Result flock(int fd, int operation) noexcept {
    return detail::call2(N(SYS_flock), fd, operation);
}

Result fchmodat(int dirfd, const char* path, unsigned int mode,
                int flags) noexcept {
    return detail::call4(N(SYS_fchmodat), dirfd, detail::arg(path),
                         detail::arg(mode), flags);
}

Result chdir_syscall(const char* path) noexcept {
    return detail::call1(N(SYS_chdir), detail::arg(path));
}

} // namespace occ::sys
