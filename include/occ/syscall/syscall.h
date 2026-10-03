#pragma once

// Typed syscall wrappers.
//
// One function per syscall, each with the real argument types of that
// syscall. No generic forwarding layer, no variadic template that turns a
// tuple into a syscall.
//
// The x86-64 kernel entry ABI:
//
//   rax = syscall number
//   rdi, rsi, rdx, r10, r8, r9 = arguments one through six
//   syscall
//   rax = return value, or -errno when it lands in [-4095, -1]
//
// Note that r10 carries the fourth argument, not rcx. rcx holds the return
// address after the syscall instruction, so the kernel cannot use it for
// argument passing. This is the single most common way to get a hand-written
// syscall wrapper wrong.

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace occ::sys {

struct Result {
    long value;
    int error;

    [[nodiscard]] constexpr bool ok() const noexcept { return error == 0; }
    [[nodiscard]] constexpr bool failed() const noexcept { return error != 0; }
    [[nodiscard]] constexpr long unwrap() const noexcept { return value; }
    [[nodiscard]] constexpr int err() const noexcept { return error; }

    // For syscalls whose successful return is a file descriptor, zero, or a
    // byte count, the caller usually wants the value. For the rest, the
    // error. Making the caller choose keeps the two from being confused.
    [[nodiscard]] constexpr int as_fd() const noexcept {
        return error == 0 ? static_cast<int>(value) : -error;
    }
};

namespace detail {

// The kernel signals an error by returning a value in [-4095, -1] as an
// unsigned wrap of -errno. Anything outside that window is a real return
// value, including large positive integers and pointers.
[[gnu::always_inline]] inline Result classify(long raw) noexcept {
    if (raw < 0 && raw >= -4095) {
        return Result{raw, static_cast<int>(-raw)};
    }
    return Result{raw, 0};
}

// Every argument reaches the assembly as a long. The wrappers below convert
// each argument explicitly at the call site, so the conversion is visible in
// the source rather than happening implicitly inside an asm constraint.
// -Wsign-conversion is enabled and catches anything left implicit.
template <typename T>
[[gnu::always_inline]] constexpr long arg(T v) noexcept {
    if constexpr (std::is_pointer_v<T>) {
        return static_cast<long>(reinterpret_cast<std::uintptr_t>(v));
    } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        return static_cast<long>(v);
    } else if constexpr (std::is_integral_v<T>) {
        // An unsigned value whose high bit is set would be undefined to
        // convert directly. The kernel reads the register as a bit pattern,
        // so the bit pattern is what is preserved here.
        return static_cast<long>(static_cast<std::int64_t>(v));
    } else {
        static_assert(sizeof(T) == 0, "unsupported syscall argument type");
    }
}

[[gnu::always_inline]] inline Result call6(long nr, long a, long b, long c,
                                           long d, long e, long f) noexcept {
    long r;
    // Argument registers in order: rdi, rsi, rdx, r10, r8, r9. The fourth
    // argument goes in r10, not rcx, because rcx holds the return address
    // after the syscall instruction. Getting this wrong produces EINVAL or
    // worse, and only on the syscalls that take four or more arguments.
    register long r10 __asm__("r10") = d;
    register long r8 __asm__("r8") = e;
    register long r9 __asm__("r9") = f;
    __asm__ volatile("syscall"
                     : "=a"(r)
                     : "a"(nr), "D"(a), "S"(b), "d"(c), "r"(r10), "r"(r8),
                       "r"(r9)
                     : "rcx", "r11", "memory");
    return classify(r);
}

[[gnu::always_inline]] inline Result call5(long nr, long a, long b, long c,
                                           long d, long e) noexcept {
    long r;
    register long r10 __asm__("r10") = d;
    register long r8 __asm__("r8") = e;
    __asm__ volatile("syscall"
                     : "=a"(r)
                     : "a"(nr), "D"(a), "S"(b), "d"(c), "r"(r10), "r"(r8)
                     : "rcx", "r11", "memory");
    return classify(r);
}

[[gnu::always_inline]] inline Result call4(long nr, long a, long b, long c,
                                           long d) noexcept {
    long r;
    register long r10 __asm__("r10") = d;
    __asm__ volatile("syscall"
                     : "=a"(r)
                     : "a"(nr), "D"(a), "S"(b), "d"(c), "r"(r10)
                     : "rcx", "r11", "memory");
    return classify(r);
}

[[gnu::always_inline]] inline Result call3(long nr, long a, long b,
                                           long c) noexcept {
    long r;
    __asm__ volatile("syscall"
                     : "=a"(r)
                     : "a"(nr), "D"(a), "S"(b), "d"(c)
                     : "rcx", "r11", "memory");
    return classify(r);
}

[[gnu::always_inline]] inline Result call2(long nr, long a, long b) noexcept {
    long r;
    __asm__ volatile("syscall"
                     : "=a"(r)
                     : "a"(nr), "D"(a), "S"(b)
                     : "rcx", "r11", "memory");
    return classify(r);
}

[[gnu::always_inline]] inline Result call1(long nr, long a) noexcept {
    long r;
    __asm__ volatile("syscall"
                     : "=a"(r)
                     : "a"(nr), "D"(a)
                     : "rcx", "r11", "memory");
    return classify(r);
}

[[gnu::always_inline]] inline Result call0(long nr) noexcept {
    long r;
    __asm__ volatile("syscall" : "=a"(r) : "a"(nr) : "rcx", "r11", "memory");
    return classify(r);
}

} // namespace detail

// ---------------------------------------------------------------- processes

Result clone3_raw(const void* cl_args, std::size_t size) noexcept;
Result fork(void) noexcept;
Result execve(const char* path, char* const argv[],
              char* const envp[]) noexcept;
Result execveat(int dirfd, const char* path, char* const argv[],
                char* const envp[], int flags) noexcept;
Result wait4(int pid, int* status, int options, void* rusage) noexcept;
Result getpid(void) noexcept;
Result gettid(void) noexcept;
Result getppid(void) noexcept;
Result getuid_syscall(void) noexcept;
Result getgid_syscall(void) noexcept;
Result geteuid(void) noexcept;
Result getegid(void) noexcept;
Result exit_group(int status) noexcept;
Result kill(int pid, int sig) noexcept;
Result tgkill(int tgid, int tid, int sig) noexcept;
Result prctl(int option, unsigned long a2, unsigned long a3,
             unsigned long a4, unsigned long a5) noexcept;

// -------------------------------------------------------------------- files

Result openat(int dirfd, const char* path, int flags,
              unsigned int mode) noexcept;
Result openat2(int dirfd, const char* path, const void* how,
               std::size_t size) noexcept;
Result close(int fd) noexcept;
Result close_range(unsigned int first, unsigned int last, int flags) noexcept;
Result read(int fd, void* buf, std::size_t count) noexcept;
Result write(int fd, const void* buf, std::size_t count) noexcept;
Result pread64(int fd, void* buf, std::size_t count, long offset) noexcept;
Result pwrite64(int fd, const void* buf, std::size_t count,
                long offset) noexcept;
Result lseek(int fd, long offset, int whence) noexcept;
Result dup(int fd) noexcept;
Result dup3(int oldfd, int newfd, int flags) noexcept;
Result fcntl(int fd, int cmd, long arg) noexcept;
Result ioctl(int fd, unsigned long request, void* arg) noexcept;
Result fstat(int fd, void* statbuf) noexcept;
Result newfstatat(int dirfd, const char* path, void* statbuf,
                  int flags) noexcept;
Result statx(int dirfd, const char* path, int flags, unsigned int mask,
             void* statxbuf) noexcept;
Result statfs(const char* path, void* buf) noexcept;
Result fstatfs(int fd, void* buf) noexcept;
Result access(const char* path, int mode) noexcept;
Result faccessat2(int dirfd, const char* path, int mode, int flags) noexcept;
Result readlinkat(int dirfd, const char* path, char* buf,
                  std::size_t size) noexcept;
Result getdents64(int fd, void* buf, std::size_t size) noexcept;
Result mkdirat(int dirfd, const char* path, unsigned int mode) noexcept;
Result unlinkat(int dirfd, const char* path, int flags) noexcept;
Result renameat2(int olddirfd, const char* oldpath, int newdirfd,
                 const char* newpath, unsigned int flags) noexcept;
Result ftruncate(int fd, long length) noexcept;
Result fsync(int fd) noexcept;
Result pipe2(int pipefd[2], int flags) noexcept;
Result mount_setattr(int dfd, const char* path, unsigned int attr_flags,
                     void* attr, std::size_t size) noexcept;

// ------------------------------------------------------------------- memory

Result mmap(void* addr, std::size_t length, int prot, int flags, int fd,
            long offset) noexcept;
Result munmap(void* addr, std::size_t length) noexcept;
Result mprotect(void* addr, std::size_t len, int prot) noexcept;
Result mremap(void* old_address, std::size_t old_size, std::size_t new_size,
              int flags, void* new_address) noexcept;
Result memfd_create(const char* name, unsigned int flags) noexcept;
Result process_vm_readv(int pid, const void* local_iov, unsigned long liovcnt,
                        const void* remote_iov, unsigned long riovcnt,
                        unsigned long flags) noexcept;
Result process_vm_writev(int pid, const void* local_iov, unsigned long liovcnt,
                         const void* remote_iov, unsigned long riovcnt,
                         unsigned long flags) noexcept;

// ------------------------------------------------------------------- mounts

Result mount(const char* source, const char* target, const char* fstype,
             unsigned long flags, const void* data) noexcept;
Result umount2(const char* target, int flags) noexcept;
Result pivot_root(const char* new_root, const char* put_old) noexcept;
Result chdir(const char* path) noexcept;
Result fchdir(int fd) noexcept;
Result chroot(const char* path) noexcept;
Result chmod(const char* path, unsigned int mode) noexcept;
Result fchmod(int fd, unsigned int mode) noexcept;
Result chown(const char* path, unsigned int uid, unsigned int gid) noexcept;
Result fchown(int fd, unsigned int uid, unsigned int gid) noexcept;

// ------------------------------------------------------------------ signals

Result rt_sigaction(int signum, const void* act, void* oldact,
                    std::size_t sigsetsize) noexcept;
Result rt_sigprocmask(int how, const void* set, void* oldset,
                      std::size_t sigsetsize) noexcept;
Result rt_sigreturn(void) noexcept;
Result rt_sigtimedwait(const void* set, void* info, const void* timeout,
                       std::size_t sigsetsize) noexcept;
Result signalfd4(int fd, const void* mask, std::size_t sizemask,
                 int flags) noexcept;
Result sigaltstack(const void* ss, void* old_ss) noexcept;

// ------------------------------------------------------------------ tracing

Result ptrace(unsigned long request, int pid, void* addr, void* data) noexcept;
Result perf_event_open(const void* attr, int pid, int cpu, int group_fd,
                       unsigned long flags) noexcept;
Result bpf(int cmd, const void* attr, unsigned int size) noexcept;

// -------------------------------------------------------------------- sched

Result sched_setaffinity(int pid, std::size_t cpusetsize,
                         const void* mask) noexcept;
Result sched_getaffinity(int pid, std::size_t cpusetsize, void* mask) noexcept;
Result setpriority(int which, int who, int prio) noexcept;
Result getpriority(int which, int who) noexcept;
Result unshare(int flags) noexcept;
Result setns(int fd, int nstype) noexcept;

// --------------------------------------------------------------------- misc

Result uname(void* buf) noexcept;
Result sysinfo(void* info) noexcept;
Result getrandom(void* buf, std::size_t buflen, unsigned int flags) noexcept;
Result clock_gettime(int clockid, void* tp) noexcept;
Result nanosleep(const void* req, void* rem) noexcept;
Result prlimit64(int pid, int resource, const void* newlim,
                 void* oldlim) noexcept;

Result socket(int domain, int type, int protocol) noexcept;
Result bind(int sockfd, const void* addr, unsigned int addrlen) noexcept;
Result listen(int sockfd, int backlog) noexcept;
Result accept4(int sockfd, void* addr, void* addrlen, int flags) noexcept;
Result connect(int sockfd, const void* addr, unsigned int addrlen) noexcept;
Result setsockopt(int sockfd, int level, int optname, const void* optval,
                  unsigned int optlen) noexcept;
Result getsockopt(int sockfd, int level, int optname, void* optval,
                  void* optlen) noexcept;
Result socketpair(int domain, int type, int protocol, int sv[2]) noexcept;
Result sendmsg(int sockfd, const void* msg, int flags) noexcept;
Result recvmsg(int sockfd, void* msg, int flags) noexcept;
Result shutdown(int sockfd, int how) noexcept;
Result setsid(void) noexcept;

Result setresuid(unsigned int ruid, unsigned int euid,
                 unsigned int suid) noexcept;
Result setresgid(unsigned int rgid, unsigned int egid,
                 unsigned int sgid) noexcept;
Result setgroups(std::size_t size, const void* list) noexcept;
Result capget(void* hdrp, void* datap) noexcept;
Result capset(void* hdrp, const void* datap) noexcept;

} // namespace occ::sys
