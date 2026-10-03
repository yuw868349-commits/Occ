#pragma once

// Linux capability bits.
//
// <linux/capability.h> declares these as macros with names like CAP_SYS_ADMIN,
// which collide with the same names in <sys/capability.h> and with anything a
// caller happens to define. This project uses its own enum so the names are
// constants in a namespace and the collision cannot happen.
//
// The bit values are from the kernel's own list and have not changed since
// capability version 3. New capabilities are only ever appended, so the
// numbers below stay correct.

#include <cstdint>

namespace occ::sys {

enum class Cap : std::uint32_t {
    Chown = 0,
    DacOverride = 1,
    DacReadSearch = 2,
    Fowner = 3,
    Fsetid = 4,
    Kill = 5,
    Setgid = 6,
    Setuid = 7,
    Setpcap = 8,
    LinuxImmutable = 9,
    NetBindService = 10,
    NetBroadcast = 11,
    NetAdmin = 12,
    NetRaw = 13,
    IpcLock = 14,
    IpcOwner = 15,
    SysModule = 16,
    SysRawio = 17,
    SysChroot = 18,
    SysPtrace = 19,
    SysPacct = 20,
    SysAdmin = 21,
    SysBoot = 22,
    SysNice = 23,
    SysResource = 24,
    SysTime = 25,
    SysTtyConfig = 26,
    Mknod = 27,
    Lease = 28,
    AuditWrite = 29,
    AuditControl = 30,
    Setfcap = 31,
    MacOverride = 32,
    MacAdmin = 33,
    Syslog = 34,
    WakeAlarm = 35,
    BlockSuspend = 36,
    AuditRead = 37,
    Perfmon = 38,
    Bpf = 39,
    CheckpointRestore = 40,
};

inline constexpr std::uint32_t kCapLast = 40;

[[nodiscard]] const char* cap_name(Cap c) noexcept;

// Parses "CAP_SYS_ADMIN", "SYS_ADMIN", or "sys_admin". Returns false when the
// text matches no capability, rather than silently producing a wrong bit.
[[nodiscard]] bool cap_from_name(const char* name, Cap& out) noexcept;

} // namespace occ::sys
