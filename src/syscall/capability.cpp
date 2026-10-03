#include "occ/syscall/capability.h"

#include <cstring>

namespace occ::sys {

namespace {

struct CapEntry {
    Cap bit;
    const char* upper; // canonical CAP_* spelling without the prefix
};

// Indexed by the capability bit. The kernel's own names, with the CAP_
// prefix removed. A table rather than a switch so that the name-to-bit and
// bit-to-name directions cannot drift apart.
constexpr CapEntry kCaps[] = {
    {Cap::Chown, "CHOWN"},
    {Cap::DacOverride, "DAC_OVERRIDE"},
    {Cap::DacReadSearch, "DAC_READ_SEARCH"},
    {Cap::Fowner, "FOWNER"},
    {Cap::Fsetid, "FSETID"},
    {Cap::Kill, "KILL"},
    {Cap::Setgid, "SETGID"},
    {Cap::Setuid, "SETUID"},
    {Cap::Setpcap, "SETPCAP"},
    {Cap::LinuxImmutable, "LINUX_IMMUTABLE"},
    {Cap::NetBindService, "NET_BIND_SERVICE"},
    {Cap::NetBroadcast, "NET_BROADCAST"},
    {Cap::NetAdmin, "NET_ADMIN"},
    {Cap::NetRaw, "NET_RAW"},
    {Cap::IpcLock, "IPC_LOCK"},
    {Cap::IpcOwner, "IPC_OWNER"},
    {Cap::SysModule, "SYS_MODULE"},
    {Cap::SysRawio, "SYS_RAWIO"},
    {Cap::SysChroot, "SYS_CHROOT"},
    {Cap::SysPtrace, "SYS_PTRACE"},
    {Cap::SysPacct, "SYS_PACCT"},
    {Cap::SysAdmin, "SYS_ADMIN"},
    {Cap::SysBoot, "SYS_BOOT"},
    {Cap::SysNice, "SYS_NICE"},
    {Cap::SysResource, "SYS_RESOURCE"},
    {Cap::SysTime, "SYS_TIME"},
    {Cap::SysTtyConfig, "SYS_TTY_CONFIG"},
    {Cap::Mknod, "MKNOD"},
    {Cap::Lease, "LEASE"},
    {Cap::AuditWrite, "AUDIT_WRITE"},
    {Cap::AuditControl, "AUDIT_CONTROL"},
    {Cap::Setfcap, "SETFCAP"},
    {Cap::MacOverride, "MAC_OVERRIDE"},
    {Cap::MacAdmin, "MAC_ADMIN"},
    {Cap::Syslog, "SYSLOG"},
    {Cap::WakeAlarm, "WAKE_ALARM"},
    {Cap::BlockSuspend, "BLOCK_SUSPEND"},
    {Cap::AuditRead, "AUDIT_READ"},
    {Cap::Perfmon, "PERFMON"},
    {Cap::Bpf, "BPF"},
    {Cap::CheckpointRestore, "CHECKPOINT_RESTORE"},
};

constexpr std::size_t kCapCount = sizeof(kCaps) / sizeof(kCaps[0]);

// Case-insensitive comparison against an ASCII identifier. Written here so
// that <strings.h> is not pulled in for one call.
bool equals_ignore_ascii_case(const char* a, const char* b) noexcept {
    while (*a != '\0' && *b != '\0') {
        char ca = *a;
        char cb = *b;
        if (ca >= 'a' && ca <= 'z') {
            ca = static_cast<char>(ca - 'a' + 'A');
        }
        if (cb >= 'a' && cb <= 'z') {
            cb = static_cast<char>(cb - 'a' + 'A');
        }
        if (ca != cb) {
            return false;
        }
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

} // namespace

const char* cap_name(Cap c) noexcept {
    const auto index = static_cast<std::uint32_t>(c);
    if (index < kCapCount && kCaps[index].bit == c) {
        return kCaps[index].upper;
    }
    return "UNKNOWN";
}

bool cap_from_name(const char* name, Cap& out) noexcept {
    if (name == nullptr) {
        return false;
    }

    const char* body = name;
    // Accept the CAP_ prefix, and accept the lowercase spelling that some
    // configuration files use.
    if (std::strncmp(body, "CAP_", 4) == 0) {
        body += 4;
    } else if (std::strncmp(body, "cap_", 4) == 0) {
        body += 4;
    }

    for (const auto& entry : kCaps) {
        if (equals_ignore_ascii_case(body, entry.upper)) {
            out = entry.bit;
            return true;
        }
    }
    return false;
}

} // namespace occ::sys
