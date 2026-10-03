#pragma once

// Error numbers.
//
// The values are the x86-64 Linux numbers, which are the same on every
// architecture the kernel supports except for a handful of legacy ports.
// They are written out because the syscall layer must not include <errno.h>:
// that header declares the thread-local errno variable, and the whole point
// of returning a Result is that no thread-local error state exists.
//
// Numbers rather than the code-generation trick of round-tripping through
// libc are used because the mapping has to be exact: a wrong number here
// prints a confident and incorrect error message.

#include <cstddef>
#include <cstdint>

namespace occ::sys {

inline constexpr int kEperm = 1;
inline constexpr int kEnoent = 2;
inline constexpr int kEsrch = 3;
inline constexpr int kEintr = 4;
inline constexpr int kEio = 5;
inline constexpr int kEnxio = 6;
inline constexpr int kE2big = 7;
inline constexpr int kEnoexec = 8;
inline constexpr int kEbadf = 9;
inline constexpr int kEchild = 10;
inline constexpr int kEagain = 11;
inline constexpr int kEnomem = 12;
inline constexpr int kEacces = 13;
inline constexpr int kEfault = 14;
inline constexpr int kEbusy = 16;
inline constexpr int kEexist = 17;
inline constexpr int kExdev = 18;
inline constexpr int kEnodev = 19;
inline constexpr int kEnotdir = 20;
inline constexpr int kEisdir = 21;
inline constexpr int kEinval = 22;
inline constexpr int kEnfile = 23;
inline constexpr int kEmfile = 24;
inline constexpr int kEnotty = 25;
inline constexpr int kEtxtbsy = 26;
inline constexpr int kEfbig = 27;
inline constexpr int kEnospc = 28;
inline constexpr int kEspipe = 29;
inline constexpr int kErofs = 30;
inline constexpr int kEmlink = 31;
inline constexpr int kEpipe = 32;
inline constexpr int kErange = 34;
inline constexpr int kEnametoolong = 36;
inline constexpr int kEnosys = 38;
inline constexpr int kEnotempty = 39;
inline constexpr int kEloop = 40;
inline constexpr int kEnomsg = 42;
inline constexpr int kEnoprotoopt = 92;
inline constexpr int kEopnotsupp = 95;
inline constexpr int kEafnosupport = 97;
inline constexpr int kEaddrinuse = 98;
inline constexpr int kEconnrefused = 111;
inline constexpr int kEtimedout = 110;
inline constexpr int kEstale = 116;
inline constexpr int kEnotsup = 95; // same value as EOPNOTSUPP on Linux
inline constexpr int kEcanceled = 125;

// Not an errno. Returned by helpers that need a failure code distinct from
// every kernel value, so that a caller can tell "the kernel said no" from
// "the input was malformed before the call".
inline constexpr int kEparse = 4096;

} // namespace occ::sys

namespace occ::util {

// The kernel's error descriptions, reproduced from the kernel's own table so
// that this project does not depend on the C library's locale files being
// installed or on the C library's spelling of the message.
//
// Returns a stable string for every value the kernel can produce, and a
// generic one otherwise. Never null.
[[nodiscard]] const char* strerror(int err) noexcept;

} // namespace occ::util
