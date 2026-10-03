# Occ

A dynamic execution runtime for reverse engineering.

Occ runs a target binary and reports what it actually does — syscalls,
memory events, register state — without analyzing it. Analysis is left to
the tools you already use: gdb, pwndbg, GEF, Frida, Ghidra, radare2.

Occ does not disassemble. It does not unpack. It does not diff traces. It
executes under isolation and emits facts.

## Status

Only the PE engine (`exe` targets) is implemented. The APK and SYS engine
directories exist in the tree and are not built. See `docs/ROADMAP.md`.

## What this is not

- Not a VM. There is no hardware virtualization anywhere in this codebase.
- Not a sandbox for hostile code in the general case. See `docs/SECURITY.md`.
- Not a debugger. It speaks the GDB Remote Serial Protocol so your debugger
  can attach; it is not one.

## Isolation model

Occ shares the host kernel. Isolation is namespace-, cgroup-, and
seccomp-based. This is the same class of isolation as a container runtime,
not the same class as a hypervisor. The distinction matters and is stated
plainly rather than papered over.

Every run gets:

- A user, PID, network, mount, UTS, IPC, and cgroup namespace.
- A new root built with `pivot_root` over an overlayfs whose upper layer is
  disposable and whose lower layer is never written.
- A cgroup v2 subtree with explicit limits.
- A seccomp-BPF filter, default-deny, built from an in-tree bytecode
  emitter.

The isolation layer is the security boundary of this project and is
fuzzed as such.

## Privilege model

Occ requires root, or `CAP_SYS_ADMIN`, `CAP_NET_ADMIN`, `CAP_BPF`,
`CAP_PERFMON`, `CAP_SYS_PTRACE`, and `CAP_SYS_RESOURCE`. It states this as
a requirement rather than working around it. `occ doctor` reports what the
current host actually grants.

Occ does not attempt to run as an unprivileged user. Container and CI
environments that withhold these capabilities are reported as unsupported
by `occ doctor` rather than silently degraded.

## Observation model

The target is not instrumented by Occ.

- Syscall tracing uses eBPF programs constructed in-tree and attached to
  raw tracepoints. There is no libbpf, no BTF, no CO-RE. Tracepoint field
  offsets are read at runtime from tracefs.
- Function-level monitoring uses hardware breakpoints via
  `perf_event_open(PERF_TYPE_BREAKPOINT)`. No `int3` is written into the
  target by default.
- Memory reads use `process_vm_readv` from outside the target.
- `ptrace` is used for process control only: stop, continue, single-step,
  and signal delivery. It is not used to patch target memory.

`ptrace` being present while "target unmodified" is claimed is not a
contradiction, and `docs/DESIGN.md` explains exactly where the line is.

## Event stream

All observation output is normalized into one event schema and written as
line-delimited JSON to a Unix domain socket. One JSON object per line, no
array wrapper. Dropped events are reported as their own event type rather
than silently lost.

The stream is never multiplexed onto stdout. When stdout is a TTY, `occ run`
prints a session summary table; when it is not, `occ run` prints the socket
path and the session id as a single NDJSON line. The event stream is always
read from the socket.

## Debugger interface

A GDB Remote Serial Protocol server, implemented in-tree, speaks the packet
set listed in `docs/RSP.md`: `qSupported`, `g`, `G`, `m`, `M`, `c`, `s`,
`Z`, `z`, `vCont`, `qfThreadInfo`, `qsThreadInfo`, and `qXfer`. Reads are
served from observer data. Writes go through `process_vm_writev` and
`PTRACE_SETREGSET` at a stop point.

## PE engine

Runs a user-supplied Wine installation inside the isolation layer as the PE
loader and API translation layer. Occ does not modify Wine. It hooks Wine's
Unix-side `ntdll` through a uprobe and turns those probes into events.

Fidelity is bounded by the Wine build supplied. Timing-sensitive,
SEH-detail-sensitive, and undocumented-structure-sensitive anti-analysis
checks will diverge from genuine Windows. Some PE targets will not reach a
state with analytical value under Wine. The documentation says so instead
of implying universal coverage.

A stripped Wine with no dynamic symbol table degrades function-level
observability to syscall-level only. This is a reduction in granularity,
not a failure, and is reported as such.

## Build

Requirements:

- Clang 18 or newer, or GCC 14 or newer.
- CMake 3.20 or newer.
- A C++23 standard library. libc++ is recommended; see `docs/BUILD.md` for
  the exact flags used to produce a fully static binary against the LLVM
  tree, which is what this project ships.

```
cmake -S . -B build -G Ninja
cmake --build build
```

The build produces one statically linked executable.

Tests:

```
OCC_ENABLE_TESTS=ON cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build
```

Fuzz harnesses are a separate, non-static build because sanitizers and
libFuzzer cannot be combined with a fully static link:

```
OCC_ENABLE_FUZZ=ON cmake -S . -B build-fuzz -G Ninja
```

## Layout

```
include/occ/     public headers
  syscall/       typed syscall wrappers, one per syscall
  isolation/     namespaces, rootfs, cgroup, seccomp
  observer/      ebpf, perf, ptrace control, event encoding
  parser/        elf, pe, zip, axml
  engine/        pe (exe) engine
  runner/        gdb rsp, ndjson, adb
  util/          logging, procfs, strings, hexdump
src/             implementation, mirrors include/occ
tests/           unit tests
fuzz/            libFuzzer harnesses
docs/            design notes
tools/           development helpers, not shipped
```

## Documentation

- `docs/DESIGN.md` — architecture, and why each piece is shaped the way it is
- `docs/SECURITY.md` — threat model, what isolation does and does not give you
- `docs/RSP.md` — the packet set, and the exact semantics of each reply
- `docs/EVENTS.md` — the event schema, field by field
- `docs/BUILD.md` — toolchain details, static linking, reproducible flags
- `docs/ROADMAP.md` — what is implemented, what is not, in plain terms
- `COMPAT.md` — hosts this has actually been built and run on

## License

See `LICENSE`.
