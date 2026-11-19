# scripts

Development-side instruments. Nothing here is part of the library: the crates
have no registry dependency, and these scripts are what measured the opcode
tables and what checks the decoders against implementations written by other
people. They are kept because a claim that a table is right is only worth
something if the method that produced it can be run again.

## Requirements

| | |
| --- | --- |
| Python | 3.8 or newer; standard library only unless noted |
| `llvm-objdump` | LLVM 23 — the primary reference for every differential test |
| `objdump` | GNU binutils 2.42 — the second opinion on x86 |
| `capstone` | optional, `thumb_diff32.py` only (`pip install capstone`) |

A script that cannot find its reference says so and exits non-zero. None of them
skip silently: a check that quietly passes because its reference was missing is
worse than no check.

## Generators

These print Rust source to standard output. The output is redirected into the
crate and committed, so building this project needs no Python at all and the
table in the repository is reviewable as a diff.

| Script | Writes | Built from |
| --- | --- | --- |
| `vector_tables.py` | `crates/occ-decode/src/x86/vector_tables.rs` | VEX maps, GNU objdump + LLVM |
| `evex_tables.py` | `crates/occ-decode/src/x86/evex_ops.rs` | EVEX maps, LLVM only (GNU lacks the 128/256-bit forms) |
| `vsib_tables.py` | `crates/occ-decode/src/x86/vsib_tables.rs` | gather/scatter, GNU objdump + LLVM |
| `compare_tables.py` | `crates/occ-decode/src/x86/cmp_ops.rs` | comparison predicates, GNU objdump + LLVM |

The rule they share: an encoding the two references describe differently is
dropped, not guessed. A missing row is a refusal to answer; a wrong row is a
wrong answer.

```bash
python3 scripts/vector_tables.py > crates/occ-decode/src/x86/vector_tables.rs
```

## Differential tests

Each of these reads encodings through this decoder and through at least one
independent implementation of the same manual, then reports where they differ.
A difference in mnemonic and a difference in operands are counted separately,
because the first means the encoding was read as a different instruction and the
second means it was read as the right one with the wrong operand.

| Script | Covers | Compares against |
| --- | --- | --- |
| `diff_all.py` | every executable and archive member on the machine | GNU objdump |
| `objdump_diff.py` | one file, instruction boundaries then text | GNU objdump |
| `x86_verify.py` | instruction boundaries on real code | GNU objdump |
| `sse_coverage.py` | the legacy `0f38` and `0f3a` maps, byte for byte | GNU objdump |
| `vex_coverage.py` | every VEX table row | GNU objdump |
| `dwarf_verify.py` | DWARF: functions, parameters, line rows, variables | binutils, over every `-O` level |
| `thumb_diff32.py` | the whole 32-bit Thumb space, exhaustively | capstone + LLVM |
| `t32_fa_scan.py` | the T32 `0xFA` class — writes the reference reading | `llvm-objdump` |
| `t32_fa_cmp.py` | the T32 `0xFA` class — reads both sides and diffs them | `llvm-objdump` |
| `t32_fa_diff.py` | two recorded listings of the same class | whichever two files are given |

`t32_fa_scan.py` and `t32_fa_cmp.py` cover the same class by different routes:
the first records what the reference says so the record can be inspected by
hand, the second does the whole round in one process. The second is what found
the three decoder bugs described in the top-level README.

## Shared modules

Imported by the scripts above; running them directly does nothing useful.

| Script | Provides |
| --- | --- |
| `llvm_disasm.py` | batched `llvm-objdump` invocation and listing parser |
| `truth_probe.py` | one encoding through both disassemblers, with disagreement reported |
| `legacy_cmp_names.py` | the condition-encoded names of the legacy SSE comparisons |
| `thumb_ref.py` | the reference-reading layer for Thumb (capstone + LLVM) |
| `thumb_canon.py` | canonical form of a Thumb instruction, for textual comparison |

## End-to-end

These talk to a built binary rather than to a library function.

| Script | Checks |
| --- | --- |
| `mcp_smoke.py` | the MCP server over stdio as a real client: handshake, schema, every tool |
| `ffi_check.sh` | the C interface: C99 link, C++17 compile, exported symbol set |
| `fuzz.py` | ten million mutated containers through the parsers; zero panics, hangs or crashes |

## Fixtures

| Script | Writes |
| --- | --- |
| `wasm3_fixture.py` | `tests/fixtures/fixture.wasm3`, by hand — no WAT assembler is assumed to exist |

## What was removed, and why

Forty-two `thumb_*.py` measuring scripts accumulated while the Thumb decoder was
being derived from the architecture by measurement: flip a bit and see what
moves, print a region as a table, solve a field layout. That work is finished —
the decoder is written, `thumb_diff32.py` covers the space exhaustively and
`t32_fa_cmp.py` reports zero real disagreements.

Thirty-nine of them printed tables for a human to read, had no caller, and
described measurements already recorded in the decoder's own comments. They were
removed rather than kept as archaeology; the method they implemented is described
in the top-level README under *The opcode tables are measured, not written down*,
and the scripts themselves remain in the git history for anyone who wants to see
how a particular field was pinned down.

The three that remain are the ones with a future: `thumb_ref.py` and
`thumb_canon.py` are the reference layer, and `thumb_diff32.py` is a test that
can be re-run when the `tbb`/`tbh`, acquire/release or VFP/NEON gaps are filled.
