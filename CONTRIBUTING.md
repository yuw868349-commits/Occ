# Working notes

Context for anyone — person or tool — editing this repository. These are
conventions, not preferences. A patch that breaks one of them will be asked
to change.

## Style

**No dates or timestamps in source, comments, commit messages, or
documentation.** This does not apply to observed data: event records carry
timing fields because timing is the observation. It applies to authorship
metadata.

**No version numbers anywhere.** Not in `project()`, not in comments, not in
CLI output. See `docs/DESIGN.md` for why.

**English only.** Code, comments, commit messages, documentation. No
exceptions.

**No filler.** No "leverage", no "seamlessly", no "cutting-edge". A comment
that restates the code in prose is deleted. A comment that explains why the
code is not the obvious thing is kept.

**No emoji.** Not in source, not in docs, not in commit messages.

**No bare TODOs.** If something is unfinished, either it is tracked with a
concrete statement of what remains and why it is not done yet, or there is
no comment. `// TODO: fix this` is noise.

**No stub function bodies presented as complete.** A function that returns
a hardcoded value to satisfy the linker is worse than no function, because
it looks finished.

## Writing documentation

Write like a man page or an RFC. Declarative sentences. Prefer a table or a
code block to a paragraph when the content is enumerable.

State limitations plainly. "This does not work when X" is a sentence worth
writing. Implying coverage that does not exist wastes the reader's time
worse than the limitation would have.

Do not explain what the reader already knows. Do not open with a summary of
the document they are about to read.

## Code

C++23. The binary links libc++ and the kernel and nothing else. No
third-party library, no vendored source, no package manager.

**Syscalls** go through `occ::sys::*` wrappers in
`include/occ/syscall/syscall.h`. Each wrapper has an explicit typed
signature. Do not call `syscall()` from anywhere else and do not add a
generic forwarding template.

**Headers** use `#pragma once`. Every header includes what it uses.

**Errors** are returned as `occ::sys::Result`, not thrown. The core library
is built with `-fno-exceptions`. Check every result that can fail. A call
whose result is `[[nodiscard]]` and cast to void needs a comment saying why
the failure is acceptable.

**Integer conversions** are checked at compile time where the ranges are
known. `-Wconversion` and `-Wsign-conversion` are on and are not to be
silenced with a cast unless the cast is provably safe.

**Parsers** take a `ByteSpan` and return a result type. No parser assumes a
null terminator, no parser trusts a length field without bounding it
against the buffer it came from, and no parser allocates before validating.

**Memory** is owned by containers. There is no `new` and no `delete` in
this codebase. `malloc` appears only where an interface demands it.

## Commits

One change per commit. The message states what changed and why, in the
imperative, in a sentence a maintainer would write. No "fixes issue", no
list of files, no restatement of the diff.

Good:

```
Add SELinux-labelled tmpfs to the mount setup

The target expects /dev/shm to exist. Without it, Wine aborts during
startup rather than at the first shm_open, which makes the failure
appear to come from the PE loader.
```

Bad:

```
Fix stuff

- updated mount code
- misc fixes
```

## Testing

Every component that parses untrusted input has a harness in `fuzz/`. A new
parser without one is not finished.

Tests live in `tests/` and run under `ctest`. A test that only runs with
root is marked as such and skipped otherwise, because a suite that cannot
run on a development machine is a suite that stops being run.
