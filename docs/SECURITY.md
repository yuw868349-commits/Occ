# Security

## Threat model

Occ is designed to run software that you do not trust, in an environment
where you control the machine and the kernel.

It is not designed to run software that can attack the kernel. A kernel
exploit from inside the target is outside what this project can prevent.
Namespace, cgroup, and seccomp isolation are kernel features; a bug in the
kernel is not contained by them.

The distinction:

| Threat | Contained |
|---|---|
| Target reads or writes files outside its root | yes, via `pivot_root` and the mount namespace |
| Target sees or signals other processes | yes, via the PID namespace |
| Target opens a network connection | yes, default-deny network namespace |
| Target exhausts host memory, CPU, or pids | yes, via cgroup v2 limits |
| Target makes a syscall it was not granted | yes, via seccomp-BPF |
| Target exploits a kernel bug | no |
| Target attacks the hardware | no |

If your requirement is to contain a kernel exploit, you need a hypervisor.
This project deliberately does not use one. Use a VM in addition to Occ, or
instead of it, depending on what you are doing.

## The "target unmodified" claim

Occ does not write to the target's memory. Concretely:

- Breakpoints are hardware breakpoints by default. No `0xCC` is written
  into target text.
- Memory is read with `process_vm_readv`, which does not stop or alter the
  target.
- `ptrace` is used for stop, continue, single-step, signal delivery, and
  register access.

Two things sit outside this claim. Both are stated here rather than left
for the reader to discover:

1. **uprobes.** The PE engine hooks Wine's `ntdll` with a uprobe. A uprobe
   works by the kernel inserting a trap instruction into the probed
   function. Occ asks for it; the kernel performs the write. Occ itself
   does not write the byte. This is a real modification of target memory,
   the mechanism is the kernel's, and it is on by default for the PE
   engine because function-level observability is the point of the engine.

2. **Software breakpoints.** Disabled by default. Enabled with
   `--allow-int3-fallback`, at which point Occ writes `0xCC` into the
   target. Events emitted from software breakpoints are marked as such.

The claim is "Occ does not write to the target unless you ask it to", not
"nothing is ever written". The first is true. The second is not, and any
tool claiming it would be wrong.

## Privileges

Occ requires root or:

```
CAP_SYS_ADMIN    namespaces, mounts, pivot_root, cgroup
CAP_NET_ADMIN    veth pairs and nftables rules
CAP_BPF          BPF program and map creation
CAP_PERFMON      perf_event_open for breakpoints and tracepoints
CAP_SYS_PTRACE   ptrace and process_vm_readv
CAP_SYS_RESOURCE rlimit changes
```

This is the same requirement class as Docker, strace, and bpftrace. It is
not a gap to be worked around. A tracing tool that cannot trace is not
useful, and pretending otherwise would mean shipping something that fails
in ways the user cannot diagnose.

`occ doctor` reports the effective capability set, so the reason for a
failure is visible before the failure happens.

## Host state that weakens isolation

`occ doctor` checks and reports each of these. None of them stop Occ. All
of them are worth knowing about.

| Setting | Effect when wrong |
|---|---|
| `kernel.unprivileged_bpf_disabled` | Does not apply to Occ, which runs with `CAP_BPF`. Reported for completeness. |
| `kernel.perf_event_paranoid` | Above 2 blocks non-root perf use. Occ runs as root, so this is informational. |
| `binderfs` not available | Blocks the APK engine. Reported as a hard failure for that engine only. |
| cgroup v2 not mounted or not writable | Limits cannot be applied. Occ reports it rather than running without limits and saying nothing. |
| `CONFIG_*` unavailable | `/proc/config.gz` and `/boot/config-$(uname -r)` are both commonly absent on cloud images. `occ doctor` reports `unavailable` and continues; a missing config file is not an error. |
| AppArmor or SELinux in enforcing mode | May deny mounts, ptrace, or BPF. Reported when a profile is enforcing. |

## Untrusted input

These components parse data from the target and are the primary attack
surface besides the kernel itself:

- ELF, PE, ZIP, and AXML parsers.
- The eBPF bytecode emitter, which reads tracefs `format` files.
- The GDB RSP packet parser, which accepts bytes from a socket.
- The adb protocol client.

Each has a libFuzzer harness under `fuzz/`. They are written to validate
before allocating, to bound every length against the buffer actually
received, and to never trust a field that describes the size of another
field.

The parsers are the part of this codebase most likely to be handed
adversarial input on purpose, because that is what a reverse engineering
target is.

## Reporting

Do not open a public issue for a memory safety bug in a parser or a
privilege escalation in the isolation layer. See `SECURITY.md` at the
repository root for the contact path.
