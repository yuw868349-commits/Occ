# Design notes

This document records decisions that are not obvious from the code, and
the reasons behind them. If you are reading the source and something looks
strange, the answer is probably here.

## Why everything is hand-written

Occ links against libc++ and the kernel. Nothing else.

That rule has a cost and it is paid deliberately. The components that
matter — the isolation setup, the seccomp emitter, the eBPF emitter, the
parsers, the RSP server — are where the project's correctness lives, and
they are small enough to own outright. A security boundary that delegates
to a dependency is a boundary whose behavior you cannot fully describe.

The rule does not extend to reinventing tools that are not part of the
boundary. Occ does not contain a disassembler or an emulator. Those belong
to the tools that attach to it.

## No version numbers

There is no version string anywhere. Not in the build, not in the CLI, not
in this document.

Version numbers create an implication that this tree has releases with
defined compatibility ranges. It does not. What matters is which host it
was verified on, and that is recorded as fact in `COMPAT.md` rather than
encoded as a number that has to be interpreted.

## Timestamps in output, not in source

Source, comments, and documentation carry no dates. Observation data
carries timing fields, because timing is the observation. A syscall event
without a timestamp is not an event, it is a rumor.

The rule is about authorship metadata, not about data.

## The isolation layer

### Namespaces

`clone3` with all seven namespaces in one call. Doing this in one call
rather than a chain of `unshare` calls removes a window where the process
is partially isolated and a signal could be delivered into an inconsistent
state.

The user namespace maps to the invoking user. Root outside, root inside,
but a different root — the containment is the point, not the privilege
change. Occ itself already runs with the capabilities it needs; the user
namespace is for the target.

`/proc/<pid>/setgroups` is written `deny` before `gid_map`. Skipping this
fails the `gid_map` write on any kernel where an unprivileged mapping is
attempted, and the failure mode is a confusing `EPERM` far from the cause.

### Root filesystem

`pivot_root`, not `chroot`. `chroot` is escapable by a process with
`CAP_SYS_CHROOT` and leaves the old root reachable through open directory
descriptors. `pivot_root` does not.

The sequence is: bind-mount the new root onto itself (because `pivot_root`
requires the new root to be a mount point), set `MS_PRIVATE` on the whole
tree (so nothing propagates back to the host), `chdir` into the new root,
`pivot_root`, then `umount2(MNT_DETACH)` the old root.

`MS_PRIVATE` is applied to the root of the propagation tree at mount time,
before any submounts exist. Applying it later leaves windows where a
propagation event can escape.

### Overlayfs

Lower layer is read-only and shared between runs. Upper and work layers are
per-session and destroyed with the session. A run never writes to the base.

The upper layer is on the same filesystem as the work layer. Overlayfs
requires this and the error when it is violated does not say so clearly.

### cgroup v2

Written directly. No systemd delegation dance, because Occ already requires
root or the equivalent capability set.

Files written, in order: `cgroup.max.descendants` and
`cgroup.max.depth` on the parent to bound runaway creation, then the leaf
`cgroup.procs` for placement, then the limit files. Writing limits before
placing the process means the process may run briefly unbounded.

### seccomp

Generated as raw `struct sock_filter` arrays. The emitter is in
`src/isolation/seccomp.cpp` and produces a jump table of the classic form:
load architecture, load syscall number, binary search over the allowed
set, return `SECCOMP_RET_ERRNO` with `EPERM` for anything unmatched.

The filter is installed after the mount setup and immediately before
`execve`. Installing it earlier would filter Occ's own setup syscalls.
Installing it later would leave a window where the target runs unfiltered.

`SECCOMP_RET_KILL_PROCESS` is not the default. A sandboxed target that
makes a disallowed syscall should learn that it failed, not die silently,
because silent death is indistinguishable from a crash and wastes the
analyst's time. The default is `EPERM`; kill-on-violation is opt-in.

### Network

Default-deny, always. The namespace is created without a network
interface, so "deny" is the absence of a path rather than a rule that
could be misconfigured.

When network is requested, a veth pair is created and nftables rules are
written through netlink directly. There is no `slirp4netns`, no userspace
network stack. The rules are NAT for outbound and a drop for inbound,
which is the minimum that makes a package manager work.

## The observer layer

### eBPF without libbpf

Programs are assembled as raw instruction arrays. Maps are created and
updated through the `bpf` syscall. There is no BTF and no CO-RE.

The consequence is that struct field offsets are not known at compile
time. They are read at runtime from `/sys/kernel/tracing/events/<cat>/<name>/format`,
which the kernel generates and keeps current. This is slower at startup
and correct across kernel versions, which is the right trade for a tool
that must not break when the host is updated.

### Ring buffer and backpressure

The ring buffer has a fixed capacity. When it fills, the kernel drops
events and increments an overrun counter. Occ reads that counter and emits
a `dropped` event carrying the count.

Losing events silently would make every downstream conclusion suspect.
Reporting the loss makes it a fact the analyst can weigh.

The consumer runs on its own thread. The producer is the kernel, so the
only backpressure available is capacity; the consumer is written to drain
faster than the producer can fill, and the drop event is the signal that
it failed to.

### Hardware breakpoints

`perf_event_open(PERF_TYPE_BREAKPOINT)`. Four slots per core is a hardware
limit, not an Occ limit.

When a debugger asks for a fifth breakpoint, Occ does not silently
substitute a software breakpoint. It replies with an error and lets the
debugger decide. A software breakpoint writes `0xCC` into the target,
which changes what the target observes, which changes what it does. That
is a decision for the person driving the debugger, not for the tool.

`--allow-int3-fallback` opts in explicitly. When it is on, software
breakpoints appear in the event stream as their own event type so that no
consumer can mistake one for a hardware breakpoint.

### ptrace

Used for stop, continue, single-step, signal delivery, and register
get/set. Not used for breakpoints and not used to write code bytes.

This is what makes the "target unmodified" claim meaningful. It is a claim
about code, not about the absence of a debugger.

### Write-then-execute tracking

`mprotect` and `mmap` calls that change a page from writable to executable
are emitted as events with a snapshot of the page attached. The page is
read with `process_vm_readv` at the moment of the transition.

Occ does not classify the page. It does not say "this is packed" or "this
is shellcode". It reports that a page became executable and here are its
bytes. Naming things is analysis and analysis is the other tool's job.

## The PE engine

### Why Wine

Running a PE natively means implementing the Win32 and NT API surface the
binary uses. Wine already implements a large fraction of it and is
maintained by people who care about it.

Occ supplies the isolation and the observation. Wine supplies the loader
and the API translation. Neither modifies the other.

### Why the hook is in Wine's ntdll

Wine's Unix-side `ntdll` is where PE requests become Unix syscalls. Hooking
there gives function-level events for the calls that matter, at a
granularity syscall tracing cannot reach, without touching the PE.

The hook is a uprobe. It is worth being precise: a uprobe is the kernel
inserting a trap into the target's text. Occ does not write those bytes;
the kernel does, on Occ's request. The "target unmodified" claim is about
what Occ writes, and a uprobe is not Occ writing. `docs/SECURITY.md` states
this without hedging, because a reader who assumes otherwise will draw the
wrong conclusion about what Occ guarantees.

### Why stripped Wine degrades

A uprobe needs an address. Addresses come from the dynamic symbol table.
`strip` removes the static symbol table but keeps `.dynsym` for exported
functions, so most builds still expose what is needed. A build with
`.dynsym` emptied cannot be probed at function granularity.

When that happens, function-level events stop and syscall-level events
continue. The session reports the reduction. It does not fail, and it does
not pretend the probes are working.

### Fidelity

Wine is not Windows. Stated here so it is not a surprise later:

- TLS callback ordering differs.
- SEH unwinding differs in the details.
- `NtQuerySystemInformation` classes return different or missing fields.
- Anti-debug checks that read the PEB, that hide threads from debuggers,
  or that time instructions will produce answers that differ from Windows.
- Structures that are undocumented on Windows may be approximations in
  Wine.

Some targets will not run far enough to be worth analyzing. That is a
property of the target, and the honest thing is to say so rather than to
imply that every PE works.

## The event schema

One struct-to-JSON mapping, defined in `docs/EVENTS.md`. Every observer
source produces one of a fixed set of event types. There is no free-form
field bag.

A schema that admits arbitrary keys cannot be parsed without guessing. The
fixed set is smaller and is worth the constraint.

## The runner

### Why GDB RSP

It is the one debugging protocol that gdb, gdb-multiarch, pwndbg, GEF, and
IDA's remote debugger already speak. Implementing it means the front end
is whatever the analyst already has open, not a new TUI to learn.

### Where reads and writes come from

Reads are served from observer data, which is collected without stopping
the target.

Writes are different. Changing a register or a memory location breaks the
"unmodified" property, and that is fine — it is what the analyst asked
for. Writes go through `process_vm_writev` and `PTRACE_SETREGSET`, and
they are applied at a stop point so the target is not running while its
state is being rewritten.

### Session state

A session is a directory under `/run/occ/<id>`, mode 0700. It holds a lock
file, the target's pid together with the start time from
`/proc/<pid>/stat`, the socket paths, and the overlay paths.

The pid alone is not enough to identify a process. Pids are reused. The
start time makes the pair unique.

`occ attach` and `occ run` both sweep `/run/occ` on startup. A session
whose pid and start time no longer describe a live process is treated as
crashed: its mounts are detached, its cgroup is removed, and its directory
is deleted. Leaving stale mounts around eventually fills the mount table
with entries that cannot be cleared without a reboot.
