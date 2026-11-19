# occ

Binary analysis over MCP, built for a model to drive.

Not a CLI with a JSON mode bolted on. The unit of exchange is a **handle**, every
answer carries its **evidence**, and the session remembers what has been
established so far. A file goes in; a chain of checkable conclusions comes out.

```
$ occ                # speaks MCP over stdio
$ occ --http :8787   # same tools over POST /mcp
```

## The problem this solves

A model analysing a binary through a normal tool loop spends its context on
transport. Ask `objdump` about a function, get 4 KB of disassembly, ask about the
caller, get the same function again from a different angle. By the third turn the
transcript is mostly repetition, and by the tenth the model is reasoning about its
own earlier output instead of about the binary.

Three concrete failures follow, and each one has a design answer here:

| Failure | Answer in this codebase |
| --- | --- |
| Payload re-sending | Addresses are handles. `fn:0x6d30` is 10 bytes on the wire; the function stays on the server. |
| Confident nonsense | Every conclusion is a set of weighted observations, combined into a confidence, with the counter-evidence and the gaps kept next to it. |
| Amnesia between turns | A session holds images, names, notes, hypotheses, traces and snapshots. `fn:0x6d30` means the same thing in turn 40 as in turn 1. |

## Five rules the code follows

**1. Handles, not payloads.** Fourteen kinds — `img:`, `sec:`, `sym:`, `fn:`,
`bb:`, `insn:`, `str:`, `blob:`, `imp:`, `exp:`, `rel:`, `hyp:`, `xref:`,
`trace:` — all resolved by one argument type. A caller that holds a handle never
re-sends what it refers to. Blob handles are content-addressed, so identical bytes
in two images get the same handle.

**2. Every answer is checkable.** The envelope carries `evidence[]` (kind, source,
address, detail, signed weight), `confidence`, `alternatives`, `caveats`, and
`truncated`. A model can tell "I verified this" from "I inferred this" from "I
could not see this", which is the distinction that stops a plausible story from
becoming a reported finding.

**3. The session is the unit of work.** `session.save`/`session.restore` persist
images, names, notes and hypotheses without caching analyses — they are
recomputed, because a cached disassembly is a stale disassembly. Hypotheses carry
sequence numbers, so a reply can distinguish a prediction made before the evidence
from one made after.

**4. Tools are intents, not command mirrors.** There is no `disassemble(start,
end)`. There is `code.function target=<handle>`, `code.explore`, `code.taint`,
`trace.run`. The engine decides how much to decode, how far to follow control
flow, and how much to render; the caller says how much it wants to read.

**5. Zero dependencies.** `Cargo.lock` contains twelve path crates and no
registry source. JSON, SHA-256, demangling, ELF/PE/Mach-O/WASM parsing, x86,
ARM32, AArch64, RISC-V and WASM decoding, DWARF, and the MCP transport are all
implemented here. The only external symbols are the Linux system calls in
`occ-sys`.

## One exchange, asked and answered

Trimmed output from a real session against `/bin/ls`.

```jsonc
// binary.open path="/bin/ls"
{
  "ok": true,
  "handle": "img:0",
  "summary": "elf x86_64 64-bit pie (/bin/ls) — 405 functions, 700 strings, 89.3% of executable bytes decoded as instructions",
  "data": {
    "identity": { "format": "elf", "arch": "x86_64", "bits": 64, "entry": "0x6d30",
                  "image_base": "0x0", "linkage": "pie", "imports": 120,
                  "interpreter": "/lib64/ld-linux-x86-64.so.2",
                  "sha256": "0148f5ab3062a905281d8deb9645363da5131011c9e7b6dcaa38b504e41b68ea" },
    "fingerprint": { "compiler": "unknown", "language": "c", "linkage": "pie",
                     "runtime": "glibc", "stripped": false, "has_dwarf": false },
    "coverage": { "code_bytes": 85818, "insn_bytes": 76659, "unparsed_bytes": 0,
                  "decoded_bytes": 76659, "covered_bytes": 83188, "hole_bytes": 6529,
                  "functions": 405, "insns": 18947, "blocks": 4121,
                  "fraction": 0.8933, "accounted_fraction": 0.8933,
                  "body_fraction": 0.9215 }
  },
  "next": ["code.functions with image=0, sort=insns, limit=20", "binary.strings ..."]
}
```

```jsonc
// code.function target="0x6d30"
{
  "ok": true,
  "handle": "fn:0x6d30",
  "summary": "_start at 0x6d30: 13 insns, 1 blocks, complexity 2, 0 callers",
  "data": { "function": { "size": 38, "import_calls": ["__libc_start_main"] } },
  "evidence": [
    { "kind": "format", "source": "loader", "weight": 0.8,
      "detail": "the image header names this address as the entry point" },
    { "kind": "cfg", "source": "heuristic", "weight": 0.2,
      "detail": "entry matches a compiler prologue" }
  ],
  "confidence": 0.84,
  "caveats": []
}
```

```jsonc
// hypothesis.create target="fn:0x6d30" statement="calls __libc_start_main"
{ "ok": true, "handle": "hyp:1", "summary": "hypothesis 1 recorded",
  "next": ["hypothesis.evidence id=1 detail=... support=true weight=0.6"] }
```

The same weight scale that makes `0.84` mean something is what makes a refutation
usable: `hypothesis.evidence` with `support=false` recomputes the verdict, and a
statement the binary contradicts comes back `refuted`, not `inconclusive`.

## Tools

Thirty-eight, in seven groups. The names carry the intent.

**Session — what state exists**

| Tool | Answers |
| --- | --- |
| `session.info` | Open images, hypotheses, notes. First call when resuming a conversation. |
| `session.save` | Write paths, names, notes and hypotheses to a file for a later process. |
| `session.restore` | Reopen those images; analyses are recomputed, not restored. |

**Binary — what is in the file**

| Tool | Answers |
| --- | --- |
| `binary.open` | Load, analyse and fingerprint a file. Every other tool takes the returned image id. |
| `binary.info` | Headers, segments, sections, fingerprint, coverage. |
| `binary.sections` | Section map, filterable by name or kind. |
| `binary.imports` | Imports, with the GOT/IAT slot each occupies when the container records it. |
| `binary.exports` | Exported symbols and their addresses. |
| `binary.symbols` | Symbol search; says whether the image is stripped, which changes how far the names can be trusted. |
| `binary.relocations` | Relocations with target, kind and addend — where function pointers live. |
| `binary.strings` | ASCII and UTF-16 strings with addresses, addressable as `str:<addr>`. |
| `binary.hexdump` | Hex and ASCII, up to 4096 bytes. |
| `binary.entropy` | Per-section entropy or a sliding window, with the threshold used. |
| `binary.signatures` | Learn function signatures from a build that still has symbols, then name the same code stripped. |
| `binary.fingerprint` | Compiler, language, linkage, runtime, packer suspicion, each with its evidence. |

**Code — what it does**

| Tool | Answers |
| --- | --- |
| `code.functions` | Recovered functions with size, block count and call degree; sortable. |
| `code.function` | One function: blocks, edges, callers, callees, imports called, strings referenced, complexity, boundary evidence. |
| `code.disasm` | Instructions from an address, each with flow kind and resolved target. |
| `code.blocks` | Control-flow graph with typed edges — loop and branch structure without reading instructions. |
| `code.decompile` | Pseudo-C: lifting, SSA, structuring, rendering. Unmodelled statements stay as comments and the `goto` count is reported. |
| `code.explore` | Execute one function along its paths; each path reports the inputs that reach it and the value it returns. |
| `code.taint` | Propagate sources through a function and report which argument of which call they arrive at, with a block path. |
| `code.xrefs` | Who refers to this, and what this refers to. |
| `code.callgraph` | Breadth-first callers or callees, bounded by depth and node count. |
| `code.search` | Instructions by mnemonic, text, or immediate value; each hit names its function. |
| `code.indirect` | What an indirect branch or call can reach: the cases of a `switch`, the entries of a function-pointer table, and the value the jump went through. |

**Debug — what the compiler recorded**

| Tool | Answers |
| --- | --- |
| `debug.info` | Compilation units, producer, language, source files; whether DWARF exists at all. |
| `debug.line` | Address to `file:line:column`, and the enclosing debug function. |
| `debug.functions` | Functions as the compiler declared them, not as recovery guessed them. |
| `debug.function` | Parameters and locals with resolved types and the location valid at an address. |
| `debug.types` | Types with sizes, member offsets and pointees. |
| `debug.variables` | Every variable placed at an address, with type and location expression. |

**Hypotheses — what has been established**

| Tool | Answers |
| --- | --- |
| `hypothesis.create` | Record a falsifiable statement about a handle. |
| `hypothesis.evidence` | Attach an observation; returns the recomputed verdict: confirmed, refuted, or inconclusive. |
| `hypothesis.list` | Every hypothesis with its verdict and evidence count. |

**Diff and scan — comparing and matching**

| Tool | Answers |
| --- | --- |
| `diff.functions` | Identical, modified, rewritten, added, removed — matched by name, then by structural hash. |
| `scan.pattern` | YARA-shaped rules: text with `nocase`/`wide`, hex with `??` and `[n-m]`, `and`/`or`/`not`, `N of them`, `at`, `in`. |

**Trace — what it actually does**

| Tool | Answers |
| --- | --- |
| `trace.run` | Run under `ptrace` and record the instruction path, every call edge including indirect ones, syscalls with path arguments, signals. Compares the run against the recovered functions. |
| `trace.get` | Query a recorded run: coverage per function, block-by-block hits, without running anything again. |

## The reply envelope

Every tool returns the same shape. A caller that has read one reply has read all
of them.

```jsonc
{
  "ok": true,               // branch on this, never on prose
  "handle": "fn:0x6d30",    // what to ask about next
  "summary": "...",         // one line, enough to decide whether to read on
  "data": { },              // structure, already trimmed to the depth asked for
  "evidence": [ ],          // why the engine believes it
  "confidence": 0.84,       // noisy-OR over the evidence
  "alternatives": [ ],      // explanations weighed and set aside
  "caveats": [ ],           // what this answer does not cover
  "next": [ ],              // concrete follow-up calls
  "tokens_estimate": 120,   // what this reply cost
  "truncated": false        // was it cut to fit the depth budget
}
```

Failures use the same envelope with `ok: false`, a `summary` that names the cause,
and `next` entries that are valid calls:

```jsonc
// code.function target="nope"
{ "ok": false,
  "summary": "\"nope\" is not an address; expected 0x401000 or 4198400",
  "next": ["binary.sections — find an addressable region", ...] }
```

`depth=summary|detail|raw` (default `detail`) caps items, bytes, instructions and
evidence entries per reply. `summary` answers "what is this", `detail` answers "how
is it built", `raw` gives the full listing with original instruction bytes so the
caller can re-decode independently.

## What it reads

| Container | Depth |
| --- | --- |
| ELF | Both classes, both byte orders, `ET_REL`/`ET_EXEC`/`ET_DYN`/core. Section and program headers, `.symtab` and `.dynsym` merged and de-duplicated, relocations, PLT/GOT binding, interpreter and `DT_NEEDED`. |
| PE/COFF | PE32 and PE32+, imports with IAT slots, exports, base relocations, Rich header detection, and raw `COFF` objects. |
| Mach-O | Thin and fat, `MH_EXECUTE`/`DYLIB`/`BUNDLE`/`OBJECT`, load commands, `LC_SYMTAB`, `__DWARF` sections. |
| WASM | The type, import, function and code sections, the signature each function index names, and the function bodies decoded from bytecode. |

Architectures: x86 and x86-64 (full decode, decompile, symbolic execution,
taint), ARM 32-bit in both instruction sets (A32 and Thumb, with the
conditional blocks, the extends, the parallel additions and the exchanged
ones; full decode, decompile, symbolic execution), AArch64
and RISC-V 32/64 (full instruction decode, decompile, symbolic execution,
taint), WebAssembly (full decode, and decompile by walking the structured
regions the bytecode is built from).

Loading does what a loader does: mapped segments and permissions, symbol tables,
import/export resolution, **PLT/GOT and IAT binding** so `call [rip+0x2f4a]`
becomes `call puts`, section classification, entropy profiling, string extraction
over mapped sections.

Analysis is recursive descent from entry points, with symbol sizes authoritative
when present, tail calls distinguished from intra-function jumps, and basic
blocks plus cyclomatic complexity per function.

A `switch` that the compiler turned into a table is recovered rather than lost.
Recursive descent stops at the indirect jump that performs the dispatch, and the
case bodies behind it are unreachable from there, so a recovery that does not
read the table ends the function at the jump and reports a body three quarters
short. The dispatch is resolved in the two shapes compilers actually emit — the
table built by `lea` / `movsxd` / `add` before a `jmp` through a register, and
the absolute-pointer table read by `jmp [base + index*8]` — and the entries
become real edges, so the case bodies are decoded, belong to the function, and
decompile as a `switch` with one case per entry.

## Decompilation, execution and taint

`code.decompile` runs four passes over the same IR that `code.explore` and
`code.taint` consume, so an answer from one can be read against the other without
a translation step: lift to statements over named values, convert to SSA with phi
nodes at joins, propagate copies and drop dead assignments, then structure with
dominators and post-dominators.

Only the first pass knows the architecture. The register file, the condition
codes, the stack frame and the widths live in a per-architecture `Target` and
`Lifter`; SSA, structuring, rendering, path exploration and taint see one
architecture-independent IR. That split is not cosmetic: x86 compares through a
flags register that survives any number of instructions, while AArch64's
condition flags are set by the instruction that computes them and consumed by
the branch that tests them, and RISC-V has no flags at all — its branch names
the two registers it compares. A single `match image.arch` inside the lifter
would have had to hold all three at once.

The width travels with the comparison. `cmp %edi,%edi` compares 32-bit values,
and a `js` after it asks whether the low half of the register is negative, so
the branch condition sign-extends what the comparison read. A lifter that
compared the whole 64-bit register would answer a different question, and the
solver would look for an input in a range the machine never examined.

Two numbers keep the output honest. `unstructured` counts the constructs that
needed `goto`, and every instruction the lifter could not model is emitted as a
comment in place. A body with `unstructured: 0` can be read as written; a body
with `unstructured: 17` is a partial rendering and says so.

Every `goto` in the output has a label to land on. A block the structure has
to reach from two places is emitted once, under a label at the end of the
function, and every path that reaches it jumps there — not once inline and
once again at the end, which leaves the reader working out that the two
copies are the same code. A jump to a label that is never defined is worse
than no structure at all, because the reader cannot follow it.

`libc.so.6` is the honest test of all this: it is hand-written, heavily
optimised code with shapes no compiler emits. `cargo run -p occ-decompile
--example decompile_symbol -- /lib/x86_64-linux-gnu/libc.so.6 warn` decompiles
one symbol from it, and the whole library can be walked the same way without
running out of stack.

WebAssembly is the one architecture that does not go through that pipeline, and
the reason is in the bytecode rather than in the implementation. A WASM
function is not a graph of basic blocks: it is a tree of `block`, `loop`, `if`
and `end`, and the instructions between them carry a stack whose height
depends on the path taken to reach them. Lifting that into a register machine
would mean inventing a register for every stack slot, inventing a condition
code for every branch, and then recovering the structure the bytecode already
states explicitly.

So the WASM lifter walks the tree instead, and the shape it produces is the
shape the source had: `if` becomes an `if`, a `loop` whose only back edge is
at its end becomes a `do`/`while`, a `br_table` becomes a `switch`. A branch
that leaves a region carries the values the region's result type names, and
the two halves of an `if` that leave different values are joined by a
synthetic phi, exactly as a join would be in the other architectures. A
label is a `break` and a `continue` where the bytecode says so, and what the
module does not state — the register a local lives in, the name of a table —
is named the way the module names it.

Two details are worth stating because they are where a WASM lifter usually
goes wrong. The first is reachability: a `br_table` has one branch per case,
and they do not all leave the same number of regions, so a region is dead
only when every branch out of it leaves it — a per-frame decision, not a
per-function one. The second is the operand stack: a branch truncates the
stack to the height its target frame was opened at, and the values above that
height are the ones it carries, so a `br` out of a region with a result moves
the result with it.

One shape no compiler emits is a branch back to the top of its own loop. A
compiler folds its `continue` into a conditional around the rest of the body,
so a `br 0` in the middle of a loop never appears in compiled output; the
decoder still has to read it and the decompiler still has to say it as
`continue`. `tests/fixtures/fixture.wasm3` is written by hand for it, built by
`scripts/wasm3_fixture.py`, which writes the sections out because there is no
WAT assembler in the build environment — a fixture that needs a tool the build
does not have is a test that skips itself.

`code.explore` forks the state at branch conditions that depend on an argument,
accumulates constraints along each path, solves them for the inputs that reach it,
and reports the returned value. A path that hits a bound is reported as `truncated`
— never as dead code, which is the failure mode that makes a model conclude a
branch is unreachable when the tool merely gave up.

## Dynamic tracing

`trace.run` is the part static analysis cannot do. It runs the program under
`ptrace`, single-steps it, and records what happened: the executed address path,
every call edge with its call site and target, indirect calls with the slot they
came through, syscalls with their path arguments and return values, and signals
with fault addresses.

It is a debugger, not a sandbox: the program runs natively with the caller's
privileges and its effects are real. It is bounded by a step count and a wall
clock, and a cut run reports `truncated` with the reason.

```
trace.run image=0 steps=20000
→ "ran /tmp/t for 1566 step(s) in 13 ms: exited 4; 11 of 14 recovered function(s)
   executed, 3 never did, 0 executed address(es) belong to no function,
   3 indirect call site(s), 2 syscall kind(s)"
```

That last part is the point. A static view says "these 14 functions exist"; the
trace says which 3 never ran, which executed bytes belong to no recovered
function, and where each indirect call actually went. `trace.get` then answers
per-branch questions — *did this block ever execute* — against the same run
without running anything again.

Three details that only show up when the tracer is real:

- **Single-stepping, not breakpoints.** A breakpoint can only confirm what the
  static view already predicted. A step-by-step path can contradict it, which is
  the useful direction. The cost is ~5 system calls per instruction, which is why
  the step and time budgets exist and why a cut run says so.
- **Every thread is resumed per stop.** `waitpid` reports one stop at a time; if
  the other threads are left suspended, `pthread_join` deadlocks. Each stop
  resumes all parked threads for one instruction.
- **Image addresses are distinguished from external ones.** An address is "in the
  image" only if `addr - load_base` falls inside a mapped section. Otherwise it is
  counted in `external_steps` and kept out of the call-edge table, so a libc
  address never masquerades as a low file offset.
- **A program that dies leaves no core file.** The tracee runs with the caller's
  privileges in the caller's directory, so a core dump would land in the middle
  of a source tree and say nothing the trace has not already said. The child's
  core-dump limit is lowered before it execs, and the tracer's own report — the
  signal, the faulting address, the instruction — is the artifact that survives.

## The C ABI

`occ-ffi` is the same analysis behind a C interface: thirteen functions, one
hand-written header, no build script and no dependency. A caller needs a C
compiler and nothing else.

```c
#include "occ.h"

int main(int argc, char **argv) {
    char err[256];
    OccSession *s = occ_open(argv[1], err, sizeof err);
    if (!s) { fprintf(stderr, "%s\n", err); return 1; }
    int64_t image = occ_image_at(s, 0);
    char *json = occ_function(s, image, "main", err, sizeof err);
    if (json) { puts(json); occ_free(json); }
    occ_close(s);
    return 0;
}
```

There are two error channels, and they mean different things:

| Return | Means |
| --- | --- |
| `NULL` | The call did not happen: a null pointer, invalid UTF-8, an image that is not open, or a panic caught at the language boundary. `err` says which. |
| a JSON document | The call happened. `"ok": true` with the payload, or `"error": true` when the analysis declined the question. |

Conflating the two is how a caller ends up reporting a bad argument as a fact
about the binary, so they stay apart: `NULL` is the absence of an answer, and
`"error": true` is an answer that says no.

Strings belong to the caller and are released with `occ_free()`. A string stays
valid after `occ_close()`, because it is a copy and not a view into the
session. The release profile unwinds rather than aborts, so a panic inside the
analysis is caught at the boundary and reported through `err` instead of
taking the caller's process down.

`scripts/ffi_check.sh` builds the library and proves three things: a C99
program links and runs against it, the same header compiles as C++17 without
changes, and the exported symbols are the ones the header promises — a rename
in Rust cannot silently produce a library that links but answers nothing.

## Differential semantics

The lifter turns instructions into expressions and the interpreter evaluates
them. Every step in that chain can be wrong in a way no unit test sees, because
a unit test encodes the author's expectation and the hardware has none.

So the semantics are checked against the hardware. `semantic_diff` runs every
integer encoding the decoder recognises twice: once on the real processor under
`ptrace`, and once through the lifted IR and the concrete interpreter. A
disagreement is a bug in one of the two, and the report names the register or
the flag.

```
$ cargo run --release -p occ-dynamic --example semantic_diff -- --cases 16
encodings tested   9584
agree              111157
differ             0
not arithmetic     17417
no statements      15648
skipped: faulted on the processor     3275
skipped: segment register operand     7360
skipped: stack-relative operand       15904
```

Those numbers are the same on every run, and that is a property the tool has to
earn rather than assume. The registers the traced process starts with point
into a mapped region, and where the kernel puts that region decides whether a
division overflows and whether a memory operand lands inside it — so an
address that moves with ASLR means the same seed tests a different set of
encodings each time. The regions are therefore mapped at fixed addresses, and
the run says so out loud when it could not get them and is not comparable with
another run's.

The first run disagreed on 11094 cases. Every one was a real difference — a
carry flag computed from the result instead of from the operand, a shift count
masked by the width instead of by the machine's five-bit rule, a multiply that
reused the unsigned high half for a signed product — and fixing them moved the
number to zero.

The exclusions are counted and printed, because an exclusion that is not
counted is an exclusion nobody can check. `faulted on the processor` is an
input the processor rejected, such as a divide by zero or a shift past the
width: the run says so instead of comparing two different things. `segment
register operand` is a move that names a segment, `stack-relative operand` one
that reads through the stack pointer — the lifter promotes those to named
locals on purpose, so the promotion is a fact about the output rather than
about the machine. `not arithmetic` and `no statements` count the encodings
the comparison does not model.

## Value-set analysis

`occ-vsa` answers a question the rest of the toolkit can only guess at: what
can an indirect branch or call reach, and why.

A `switch` compiled to a jump table reaches its case bodies through a value
the program computed. A dispatch through a table of function pointers reaches
its callees the same way. Both look like `jmp rax` in a listing, and both are
opaque without the values.

The pass keeps a value set per register — a finite set, an arithmetic
progression, a symbol with a known offset, or nothing known — and from that it
reads the table behind a computed jump and names the entries.

Two things make the answer checkable:

- **The table is read out of the image, and a run of entries ends where the
  first entry stops naming code.** Not at the end of the section: a ten-entry
  table sitting next to another table is how a switch grows cases it never had.
- **The case values come from the data flow, not from the table.** A compiler
  that sees `case 100..106` takes a hundred off the index before it looks
  anything up, so the table holds positions and the case value of a position
  is the position plus a hundred.

The two readings are independent — one is what the value analysis says, the
other is what the instruction stream says — and the pass reports a
disagreement rather than picking one.

```console
$ cargo run --release -p occ-vsa --example report ./switch
./switch: 18 functions, 61 blocks
calls           1 sites,      0 resolved (0.0%)
jumps           5 sites,      3 resolved (60.0%)
tables          1 read, 0 two-level; syntactic resolver found 2
jump      0x134e in     0x1330 via rax   7 cases  (the table was recovered: 7 cases, and the index carries -100, so the first case is 100)
        case    100 -> 0x1388
        case    101 -> 0x13a0
        case    102 -> 0x13b8
```

The same numbers are behind `code.indirect`, so the pass is available over MCP
without leaving the session.

## Naming library code in a stripped image

A stripped static executable is mostly a library, and every name in it is
gone. `binary.signatures` learns signatures from a build that still has
symbols and applies them to the same code stripped.

A signature is a function's first 32 bytes plus a mask, one flag per byte.
Learning records those bytes as they are; a mask entry says whether a byte has
to match, so a call displacement or a RIP-relative address that the linker
chose differently on a second build does not have to. The file format is text,
because a signature is a claim about code and a reader should be able to see
what it claims:

```
# occ signatures v1
# name	bytes-hex	mask-bits
memcpy	488b0511b709004885c0	11111111111111111111
```

Three rules make the answer safe to act on:

- **A function that already has a name is never renamed.** The symbol table
  wins over a byte pattern, every time.
- **Two signatures of equal length that both match name nothing.** The
  collision is reported with both names, because a wrong name is worse than
  `sub_4f3a20`: the reader stops looking.
- **A signature never matches a prefix it did not compare in full.** There is
  no partial credit and no fuzzy score.

Sixteen bytes was not enough. A static link holds hundreds of bodies that open
with the same `endbr64; push r15; push r14` and diverge later; a sixteen-byte
prefix produced 468 collisions against 580 names, which is a database that
names almost nothing it could not also have named wrongly. Thirty-two bytes
fixed it. On a static `-O1` build of a fifty-line program, learning from the
unstripped binary and applying to the stripped one named 953 of 1044 functions,
and every name it placed agreed with the symbol table it was checked against.

## Fuzzing the boundary

Everything above reads a file the caller did not write. `scripts/fuzz.py`
mutates seeds and feeds them to the parsers until one of them does something it
should not.

A failure is a panic, an abort, a segfault, or a batch that does not finish
inside the time allowed — a hang is a failure until it is proven otherwise, and
it is reported rather than retried quietly. A rejected input is not a failure:
the parser is supposed to refuse most of what this produces, and the refusal
is the bounds checking working.

Ten million runs at the load depth:

```
10000000 iterations
  5999715 loaded
  4000285 rejected
        0 panics
        0 crashes
        0 timeouts
  1225.3s  (8161/s)
```

The mutator is in Rust because a file write per iteration costs more than the
parse it is trying to test. The script owns the parts Python is good at:
choosing seeds, sizing batches, surviving a hang, and saying what happened.

## Zero dependencies

`Cargo.lock` has fourteen packages, all of them workspace members, and not one
`source = "registry"` line. That is checkable in one command:

```bash
grep -c 'source = "registry' Cargo.lock                                   # 0
cargo tree --workspace --edges normal --prefix none | grep -v '^occ' \
  | grep -c ' v'                                                          # 0
```

What a dependency would have been, and what replaced it:

| Would be | Replacement | Where |
| --- | --- | --- |
| `serde` + `serde_json` | Value tree, parser, writer, schema | `occ-json` |
| `thiserror` | Hand-written `Display` and `Error` | `occ-core/src/error.rs` |
| `object` | ELF, PE/COFF, Mach-O, WASM parsers | `occ-bin` |
| `cpp_demangle` + `rustc-demangle` | Itanium demangler and Rust hash stripping | `occ-bin/src/demangle.rs` |
| `sha2` | SHA-256 | `occ-bin/src/sha256.rs` |
| `memmap2` | `mmap` declared directly, with an owned lifetime | `occ-sys/src/mem.rs` |
| `libc` / `nix` | `extern "C"` declarations for the fifteen symbols used | `occ-sys` |
| `iced-x86` | Table-driven x86 decoder | `occ-decode/src/x86.rs` |
| `gimli` | DWARF 2–5 reader, line program, location lists | `occ-dwarf` |
| `rmcp` / `tokio` | JSON-RPC 2.0 over stdio and a hand-rolled HTTP loop | `occ-mcp` |

The trade is deliberate and has a cost: the decoders cover what this tool needs
to answer questions about real binaries, not every encoding in the vendor manual.
Where coverage stops, the answer says so rather than guessing — and the counts
say which kind of stop it was. `insn_bytes` is what the decoder understood.
`unparsed_bytes` is what it had to emit as `.word`: read, aligned, not
understood. `hole_bytes` is inside a recovered body with no unit covering it at
all — the gap a body jumps over. `fraction` is the first over `code_bytes`;
`body_fraction` asks the narrower question, how much of the code that was found
was actually read.

## Install

```bash
cargo build --release          # Rust 1.85+, edition 2024
./target/release/occ        # MCP over stdio
```

Register it with a client:

```json
{
  "mcpServers": {
    "occ": {
      "command": "/absolute/path/to/occ",
      "args": []
    }
  }
}
```

Then: *"Load /usr/bin/curl, find what it does with the `-x` argument, and show me
the evidence."*

## Layout

| Crate | Responsibility | Lines |
| --- | --- | --- |
| `occ-core` | Handles, addresses, evidence, hypotheses, image IR, depth budgets. No I/O. | 2.2k |
| `occ-json` | JSON value, parser, writer, schema validation. | 1.5k |
| `occ-sys` | `ptrace`, `waitpid`, `mmap`, `kill`, `setpgid`, `setrlimit`, `atexit` as `extern "C"`. | 1.1k |
| `occ-bin` | Container parsing: ELF, PE/COFF, Mach-O, WASM, `ar` archives, demangling, SHA-256. | 5.8k |
| `occ-decode` | x86, ARM32 in both instruction sets, AArch64, RISC-V and WASM instruction decoding. | 16.7k |
| `occ-analyze` | Recursive descent, CFG, cross references, strings, entropy, fingerprint, jump tables. | 4.0k |
| `occ-dwarf` | DWARF units, abbreviations, line program, location lists. | 2.0k |
| `occ-decompile` | Lift, SSA, cleanup, structuring, reshaping, rendering; the WASM structured walk. | 18.5k |
| `occ-sym` | Path exploration, constraint solving, taint propagation over that IR. | 2.8k |
| `occ-dynamic` | `ptrace` tracing, `/proc/pid/maps`, syscall names, trace/static correlation. | 1.3k |
| `occ-vsa` | Value sets: what an indirect branch or call can reach, and why. | 1.8k |
| `occ-session` | Session state, handle resolution, envelopes, persistence, diff, pattern scan. | 2.7k |
| `occ-mcp` | The 39 tool definitions, JSON-RPC transport, stdio and HTTP. | 3.6k |
| `occ-ffi` | The C interface: thirteen functions, one hand-written header. | 0.7k |

64.7k lines of Rust, 625 tests, 82 end-to-end MCP checks. The line count is the
code, not the comments and not the blank lines, and it leaves out the SVE
encoding tables, which are generated from the architecture manual rather than
written by hand and which run to a hundred thousand lines on their own.

## Verification

Unit tests are necessary and not sufficient: most of the serious bugs found while
building this were invisible to them and obvious against a real file.

```bash
cargo test --workspace                                         # 625 tests
cargo clippy --workspace --all-targets -- -D warnings          # clean
python3 scripts/mcp_smoke.py target/release/occ             # 82 checks
scripts/ffi_check.sh /bin/true                                 # the C interface
cargo run --release -p occ-dynamic --example semantic_diff -- --cases 16
python3 scripts/fuzz.py                                         # ten million runs
```

One test compiles the same C source for three architectures and asks each one
the same questions. The decompiler has to recover the same parameters, the same
control flow and the same global address; the symbolic executor has to reach
the same four answers for the same five inputs, explore all four paths, and
carry the taint to the same three branches. A lifter that treats a signed
comparison as an unsigned one still produces paths — just the wrong ones — and
the answers say so.

The suite also checks the tracer's effect on the machine around it: a program
that dies under the tracer must leave no core file in the working directory, and
the fixtures the tests compile must not pile up in `/tmp`. Both are assertions,
not habits, because both were true the other way once.

`scripts/mcp_smoke.py` is a real MCP client: JSON-RPC handshake, `tools/list`,
schema validation, then every tool against `/bin/ls`, a cross-compiled PE, a
Mach-O object, a COFF object, a WASM module and a compiled trace fixture. It
asserts on things like *the entry point reaches `__libc_start_main`*, *string
addresses are not zero*, *the tracee's stdout does not corrupt the protocol
stream*, *a PIE breakpoint actually hits*, and *changing argv changes block
coverage by exactly one block*.

Three of those commands check the parts that cannot be checked by a unit test.
`semantic_diff` runs the lifted semantics against the processor and requires
zero disagreements; `fuzz.py` feeds ten million mutated binaries to the parsers
and requires zero panics, crashes and hangs; `ffi_check.sh` compiles a C
program against the library and requires the exported symbols to be the ones
the header promises.

`semantic_diff` is also the one command whose numbers are only meaningful if
they repeat, so the run is arranged to repeat: the regions the traced process
sees are mapped at fixed addresses, and the report says so when it could not
get them. A differential run that covers a different set of encodings each
time cannot be compared with the last one, and a difference between two such
runs is not a finding.

### The opcode tables are measured, not written down

A disassembler is a table of several thousand facts, and a table typed from
memory is a table with several thousand chances to be wrong in a way no unit
test looks at. Every x86 table in `occ-decode` is therefore built by
construction: the script lays an encoding into a sixteen-byte slot, hands the
buffer to two independent disassemblers, and records what they agree on.

| Script | Produces | Cross-checked against |
| --- | --- | --- |
| `scripts/vector_tables.py` | `vector_tables.rs` (VEX) | GNU objdump 2.42 + LLVM 23 |
| `scripts/evex_tables.py` | `evex_ops.rs` (EVEX) | LLVM 23 (GNU lacks the 128/256-bit forms) |
| `scripts/vsib_tables.py` | `vsib_tables.rs` (gather/scatter) | GNU objdump 2.42 + LLVM 23 |
| `scripts/compare_tables.py` | `cmp_ops.rs` (comparisons) | GNU objdump 2.42 + LLVM 23 |

The agreement rule is what makes this usable: an encoding the two channels
describe differently is dropped, not guessed. A few hundred encodings are
dropped that way, and the table says nothing about them rather than something
wrong.

The check that matters is the one over real code. `scripts/diff_all.py` runs
`objdump -d` and the decoder over every executable and every archive member on
the machine and compares instruction boundaries, not text.

An instruction boundary is the load-bearing claim: get one length wrong and
every following instruction is compared at the wrong address. Text is compared
too, and the differences that remain are aliases — the same instruction under
another name, `vcmpeqps` for `vcmpps` — which the report counts separately
rather than hiding.

Two comparisons need a different rule than the rest, and the rule is stated
rather than worked around. A relocatable object has no load address: `objdump`
restarts its address column at zero for every section, while the decoder keeps
addresses unique by using the section's file offset, so those two listings are
compared by position. An `ar` archive is not one image at all: each member is
written out and checked as the object it is.

The sweep is not a clean sheet, and saying so is part of the method. Of the
forty files it flagged on this machine, thirty-four now compare instruction for
instruction. The six that remain have two causes, both recorded under Limits: an
AVX-512 shift form the table does not carry, and 32-bit libraries whose `.text`
holds data that `objdump` reads one byte at a time where the decoder follows the
encoding.

The same rule covers a class that can be enumerated rather than swept.
Thumb-32's `0xFA` space — the register shifts, the extends, and the parallel
additions and subtractions — is one first halfword plus sixteen bits of second,
so `scripts/t32_fa_scan.py` reads all of its encodings with `llvm-objdump`,
`cargo run -p occ-decode --example t32_fa_table` writes the decoder's reading of
the same space, and `scripts/t32_fa_cmp.py` compares the two. That comparison
found three mistakes in the class: a checksum group carried over from A32 that
the Thumb space does not have, a byte-reversal reading that took its source from
the wrong halfword, and an extend group that stopped at half its range. The
remaining differences are all encodings where the reference reads a field the
architecture manual marks as ignored; the report names them as such rather than
counting them as agreement.

Bugs this discipline caught, each of which had passing unit tests at the time:

- **dynsym index off by one** — `__libc_start_main` resolved as `__mempcpy_chk`,
  because ELF symbol indices count the reserved null symbol. Found by diffing
  against `readelf -rW`.
- **Function recovery running into the next function** — a 5-instruction `_start`
  absorbed 125 instructions from a neighbour. Fixed with symbol-size bounds plus
  tail-call classification.
- **PIE image base reported as `0x4000`** — zero was filtered out of the segment
  minimum, so every runtime address landed a segment short and no PIE breakpoint
  could ever be armed.
- **Strings at address `0x0`** — non-allocated sections have no load address, and
  reporting one handle for hundreds of different strings silently aliased them.
- **Relocatable objects with every section at `0x0`** — `.o`/`.obj` files have no
  load address and their symbol values are section-relative, so every function
  landed at zero. Found by compiling the same source to ELF, COFF and Mach-O
  objects and requiring two distinct functions from each.
- **Prologue predicates that were always false** — the AArch64 and RISC-V bit
  masks cleared the bits they then compared, so those two architectures had no
  prologue recognition at all. `clippy::bad_bit_mask` found it; the fix is pinned
  by tests over real instruction encodings.
- **One file, two answers about linkage** — the header said `pie`, the fingerprint
  said `linkage?`, because the analyzer re-derived the load model from section
  names instead of reading `e_type`. Linkage now comes from the container only.
- **Syscall names that were all `sys_18446744073709551615`** — single-stepping
  produces no syscall-stop, so `orig_rax` is empty; the tracer reads `rax`.
- **Every `goto` pointing at a label that did not exist** — the structurer
  jumped to a block it had already emitted but never defined the label. Across
  2000 functions of glibc that was 16359 jumps into nothing. The structure is
  now built twice: the first pass finds the shared blocks, the second treats
  them as boundaries and emits each one once, under a label.
- **A stack overflow that killed the server** — a block with one predecessor
  forwards a variable read to it, and the blocks of a loop form a ring, so the
  forwarding never ended. One function of glibc was enough to abort the
  process. A read now tracks what is in flight; a value in a ring has no first
  definition and is reported as unknown.
- **A switch that vanished from the decompiled output** — a `switch` compiled
  to a table is reached through an indirect jump, which recursive descent stops
  at, so the case bodies were never decoded. The table is now resolved and its
  entries become real edges.
- **A fixture that never ran and the suite was green anyway** — the jump-table
  test had no `main`, so it never linked, so the test skipped itself and
  reported success. A compiler that is present but fails is now a failure.
- **A 32-bit comparison answered with a 64-bit value** — `cmp %edi,%edi` compares
  the low half of the register, but the branch condition read the whole thing, so
  `js` asked whether the upper half was negative. The width now travels with the
  comparison, and the solver looks for an answer in the range the machine
  actually examined.
- **Every RISC-V global at address zero** — `__global_pointer$` is an absolute
  symbol, and the loader treated it as undefined and dropped its address. Every
  access through `gp` was then off by the whole data segment.
- **A branch against an unknown** — RISC-V's `x0` is hardwired to zero: reads
  return zero and writes are discarded. Lifted as an ordinary register, `beq a0,
  zero` compared against a value nothing had ever set, and the solver had no
  constant to work from.
- **`xsavec` decoded as one byte** — the state-save family (`0F C7 /3` through
  `/5`) had no entry at all, so the decoder emitted the opcode alone and every
  following instruction in the function was compared at the wrong address. The
  report showed a hundred `not decoded` lines; the cause was one missing row.
- **Absolute moves read eight address bytes in 32-bit code** — `A1` takes an
  address as wide as the mode, and `67` switches it to the other width. The
  decoder read the 64-bit width everywhere, so every `movabs` in a 32-bit
  library over-read by four bytes.
- **The instructions 64-bit mode dropped were missing entirely** — `push es`,
  `daa`, `pusha`, `aam`, `les`, `bound`, `enter`, and the second spelling of the
  byte-form ALU group at `0x82`. They are rare in code written this century and
  ordinary in libraries that predate it, and each one cost a byte of alignment.
- **An undecodable opcode did not consume its prefix** — `rep` in front of an
  unknown byte is a two-byte instruction in the listing, and answering one byte
  left the prefix behind and moved the stream by one.
- **`movddup` renamed to `movhlps`** — the two share an opcode, and the
  register-form rename was applied without checking the prefix that decides
  which of them the bytes mean.
- **`ar` archives were recognised and refused** — the format probe knew the
  magic and the loader answered "cannot parse it yet". A `.a` is a container of
  containers, and it now opens as one image per member, named
  `libfoo.a(member.o)`.

## Limits

Stated because the tool is supposed to state them, not because they are temporary.

- **Decompilation covers x86-64, x86-32, ARM32, AArch64, RISC-V and
  WebAssembly; path exploration and taint cover all of them except
  WebAssembly.** A WASM function is decompiled by walking the structured
  regions the bytecode is built from, which produces a body, but not the
  register-level program the symbolic executor explores: there is no
  register file to fork and no condition flags to read, so `code.explore`
  and `code.paths` report that they have nothing to execute rather than
  guessing. The operand stack could be modelled as that state, and that
  work has not been done.
- **ARM32's 32-bit Thumb space is not finished.** The 16-bit set is, and in
  the 32-bit set the conditional branches, the extends, the parallel
  additions and subtractions, the exchanged forms, the multiply family,
  the load and store multiples, the load and store singles and the
  exclusive accesses are read. The table branches (`tbb`, `tbh`), the
  acquire and release forms, and the coprocessor space — VFP and NEON —
  are not. The CRC32 extension is read in neither instruction set: it is
  an ARMv8 instruction written in the A32 space, and the Thumb space at
  that encoding has nothing behind it, which is why the decoder reports
  those halfwords as undefined rather than naming an instruction the
  Thumb machine cannot execute. An encoding this decoder cannot read is
  reported as an undefined instruction or as a coprocessor instruction,
  never as a guess at what it might have been.
- **AVX-512 coverage is uneven by construction.** The EVEX table holds the
  encodings both disassemblers agreed on, which is most of the set but not all of
  it: `vpshldw`, `vdbpsadbw`, `vrsqrt14ps` and a few dozen others are missing,
  and the sweep over the family reports them as gaps rather than as guesses. The
  immediate-shift forms at `0F 73` are among them, and on this machine the gap
  fires in one library — `libgfortran` — where the instruction decodes as bytes
  and the listing drifts until the next boundary. Compilers do not emit the rest
  of the missing set below 512-bit vectors, so code written with intrinsics is
  where to expect them.
- **Data inside `.text` is read as code, by both sides, to different depths.**
  A jump table or a literal that sits in an executable section is decoded as
  instructions. `objdump` answers one byte and gives up on an encoding it does
  not recognise; this decoder follows the encoding as far as the bytes go. Both
  are guessing, the guesses differ in length, and the comparison reports the
  difference instead of hiding it. Two 32-bit libraries on this machine are the
  only places it fires.
- **Tracing is x86-64 Linux only.** It requires a Linux host and an x86 or x86-64
  image; PE and Mach-O can be loaded and analysed, not executed here.
- **Static analysis cannot see** runtime-resolved function pointers, `dlopen`'d
  symbols, direct syscalls, or anything a packer decrypts in memory. Hypothesis
  replies list these as unchecked rather than implying absence. `trace.run` is the
  answer for the cases where the program can be run.
- **Indirect calls** resolve to `CallTarget::Indirect { via }` when the slot is
  known, and the value-set pass reads the table behind a computed jump when it
  can. What it cannot do is follow a pointer the program builds at run time:
  there is no points-to analysis, and a table whose entries are written rather
  than read is not a table this pass can see.
- **DWARF** covers versions 2 to 5, without split DWARF or compressed
  `.zdebug_*` sections. A variable whose location cannot be determined is reported
  as such, not guessed.
- **No decompilation of hand-written assembly idioms** beyond the patterns the
  lifter models; those instructions appear as comments with their bytes intact.
- **Dispatch is resolved in two shapes, and they are different shapes.** The
  syntactic resolver handles a table built by `lea` / `movsxd` / `add` before a
  `jmp` through a register, and an absolute-pointer table read by
  `jmp [base + index*8]`. The value-set pass reads a table out of the image
  whenever the address of the lookup has the shape `base + index * stride`,
  which covers both and also the tables a compiler writes for a function
  pointer dispatch. What neither can do is follow a jump through a slot the
  program computed at run time, or a table whose entries are values rather than
  targets — which is what a compiler emits for a small switch, and which is
  not a dispatch at all.
- **Case numbers come from the offset the compiler took off the index.** A
  compiler that normalises the discriminant (`sub edi, 'a'`) leaves the offset
  in the instruction stream, and both readers of it — the syntactic resolver
  and the value-set pass — use it, so the recovered `case` values are the
  source's. The two readings are independent and the pass reports a
  disagreement rather than picking one, but a compiler that transforms the
  index in a way neither can see leaves the case values numbered from zero
  again.

## License

MIT. See [`LICENSE`](LICENSE).
