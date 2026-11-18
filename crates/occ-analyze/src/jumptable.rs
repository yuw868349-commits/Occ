//! Computed jumps: finding the table behind a `jmp`, and the cases it reaches.
//!
//! A `switch` with enough cases compiles to a table lookup followed by a jump
//! through it, and a recursive-descent recovery that stops at the jump loses
//! every case body — the function looks small, decompiles to a table read, and
//! reports nothing wrong. That is the failure mode this module exists to
//! remove.
//!
//! # What the compilers emit
//!
//! Two shapes, both handled here:
//!
//! ```text
//! ; GCC/Clang, position independent: 32-bit offsets from the table base
//! lea     rdx, [rip+0xea4]        ; 0x202c
//! movsxd  rax, dword ptr [rdx + rdi*4]
//! add     rax, rdx
//! jmp     rax
//!
//! ; non-PIE: absolute 64-bit pointers in the table
//! jmp     qword ptr [0x404020 + rax*8]
//! ```
//!
//! The first shape needs a short backward slice inside the block, because the
//! jump itself is through a register and the table address was materialised
//! several instructions earlier. The slice is bounded and syntactic: it looks
//! for `add <jmp reg>, <base reg>`, then `lea <base reg>, [rip+X]`, then a
//! scaled load into the jumped register. Anything less specific is left
//! unresolved rather than guessed at.

use std::collections::BTreeSet;

use occ_core::{Addr, Instruction, JumpTable};
use occ_core::FlowKind;

use crate::view::View;

/// Most entries to follow. A real dispatch table is far smaller; a bigger one
/// is a data structure indexed by a jump, and following it would decode the
/// heap as code.
const MAX_ENTRIES: usize = 4096;

/// How far back the slice may look for the table address.
const MAX_LOOKBACK: usize = 24;

/// Smallest run of executable targets that counts as a table.
const MIN_ENTRIES: usize = 3;

/// Resolve the table behind `jmp`, using the instructions decoded before it in
/// the same block. `window` must end with `jmp`.
pub fn resolve(view: &View, window: &[&Instruction], jmp: &Instruction) -> Option<JumpTable> {
    if jmp.flow != FlowKind::IndirectBranch {
        return None;
    }
    let (base, entry_bytes, relative, index_reg) = shape(window, jmp)?;
    let limit = case_count(window, index_reg.as_deref());
    let first = first_case(window, index_reg.as_deref());
    let targets = read_targets(view, base, entry_bytes, relative, limit);
    if targets.len() < MIN_ENTRIES {
        return None;
    }
    Some(JumpTable {
        at: jmp.addr,
        base,
        entry_bytes,
        relative,
        count: targets.len() as u32,
        index_reg,
        first_case: first,
        targets,
    })
}

/// The case value of the first entry, as the code states it.
///
/// A compiler that sees `case 100..106` subtracts 100 from the index
/// before the lookup:
///
/// ```text
/// sub     $0x64, %edi
/// cmp     $0x6, %edi
/// ja      .Ldefault
/// ```
///
/// The table then holds positions, and the position is the case value less
/// 100. The subtraction is the only statement that says so.
///
/// The instruction has to write the register the table was indexed by, and
/// it has to be the last one that does: a later write would mean the
/// subtraction was not what reached the table.
fn first_case(window: &[&Instruction], index_reg: Option<&str>) -> Option<i64> {
    let index_reg = index_reg?;
    let mut found = None;
    for insn in window {
        let Some((dst, amount)) = offset_of_index(&insn.text) else {
            continue;
        };
        if family(dst) != family(index_reg) {
            continue;
        }
        // `sub $0x64, %edi` lowers the index by 64, so the first case is 64.
        // `add $-0x64, %edi` says the same thing the other way round.
        found = Some(match amount {
            Some(amount) => amount,
            None => continue,
        });
    }
    found
}

/// `sub edi, 0x64` → ("edi", 64), and `add edi, -0x64` → ("edi", 64).
///
/// The spelling is the one this decoder prints: the source operand comes
/// first, so the immediate is the second half.
fn offset_of_index(text: &str) -> Option<(&str, Option<i64>)> {
    let (op, rest) = match text.split_once(' ') {
        Some((op, rest)) => (op, rest),
        None => return None,
    };
    let sign = match op {
        "sub" => 1i64,
        "add" => -1i64,
        _ => return None,
    };
    let (reg, amount) = match rest.split_once(", ") {
        Some((reg, amount)) => (reg.trim(), amount.trim()),
        None => return None,
    };
    let reg = reg.split_whitespace().next().unwrap_or(reg);
    if reg.is_empty() || !reg.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    let value = immediate(amount)?;
    Some((reg, Some(value.wrapping_mul(sign))))
}

/// How many cases the switch has, as the bound check states it.
///
/// A compiler that builds a jump table checks the index against the last
/// case before the lookup:
///
/// ```text
/// cmp     $0x9, %edi
/// ja      .Ldefault
/// ```
///
/// The constant is the number of cases minus one, and it is the only
/// statement in the code that says how long the table is. Without it the
/// table is read until an entry stops naming code, which runs into the
/// next table in the same section and invents cases that the switch never
/// had — which is not a rare accident: two switches in one file put their
/// tables next to each other.
///
/// The check is only trusted when it is on the register the table was
/// indexed by. A comparison on some other register says nothing about
/// how long this table is.
fn case_count(window: &[&Instruction], index_reg: Option<&str>) -> Option<usize> {
    let index_reg = index_reg?;
    let mut found = None;
    for pair in window.windows(2) {
        let (cmp, br) = (pair[0], pair[1]);
        if !is_above_branch(&br.text) {
            continue;
        }
        let Some((reg, imm)) = cmp_against_immediate(&cmp.text) else {
            continue;
        };
        if family(reg) != family(index_reg) {
            continue;
        }
        if imm < 0 || imm as usize >= MAX_ENTRIES {
            continue;
        }
        // The last such check before the jump is the one that bounds the
        // table; an earlier one is some other comparison.
        found = Some(imm as usize + 1);
    }
    found
}

/// `ja` and `jg`: the branches that mean "above the bound, leave the switch".
fn is_above_branch(text: &str) -> bool {
    text.starts_with("ja ") || text.starts_with("jg ")
}

/// `cmp edi, 0x9` → ("edi", 9).
fn cmp_against_immediate(text: &str) -> Option<(&str, i64)> {
    let rest = text.strip_prefix("cmp ")?;
    let (lhs, rhs) = rest.split_once(", ")?;
    let reg = lhs.trim();
    if reg.is_empty() || !reg.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    Some((reg, immediate(rhs.trim())?))
}

/// An immediate as this decoder prints it: `0x9`, `-0x64`.
fn immediate(text: &str) -> Option<i64> {
    let (negative, digits) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    let digits = digits.strip_prefix("0x").unwrap_or(digits);
    let value = i64::from_str_radix(digits, 16).ok()?;
    Some(if negative { -value } else { value })
}

/// The register the switch counts cases in, traced through a widening copy.
///
/// Compilers widen the index before the scaled load — `movzx eax, sil` — and
/// the bound check that says how long the table is stays on the narrow
/// register. Following the copy does two things: it names an expression whose
/// value the case numbers describe, and it lets the bound check be found at
/// all. Without it the switch was printed against the computed jump address,
/// so `case 0x0` claimed to be the value the register holds at the jump.
fn traced_index(window: &[&Instruction], index: &str) -> String {
    let mut reg = family(index).to_string();
    for _ in 0..4 {
        let mut next = None;
        for insn in window.iter().rev() {
            let Some((dst, src)) = move_pair(&insn.text) else {
                continue;
            };
            if family(dst) != reg {
                continue;
            }
            // The last write to the register is the one that reaches the
            // table. A copy from another register renames the value; one from
            // memory or an immediate ends the chain, because `[rbx]` is not a
            // register the case numbers can be counted in.
            next = src.map(|s| family(s).to_string());
            break;
        }
        match next {
            Some(src) if src != reg => reg = src,
            _ => break,
        }
    }
    reg
}

/// `movzx eax, sil` → ("eax", Some("sil")); `mov eax, [rbx]` → ("eax", None).
fn move_pair(text: &str) -> Option<(&str, Option<&str>)> {
    let (op, rest) = text.split_once(' ')?;
    if !matches!(op, "mov" | "movzx" | "movsx" | "movsxd") {
        return None;
    }
    let (dst, src) = rest.split_once(", ")?;
    let dst = dst.trim();
    let src = src.trim();
    let src = if !src.is_empty() && src.chars().all(|c| c.is_ascii_alphanumeric()) {
        Some(src)
    } else {
        None
    };
    Some((dst, src))
}

/// The 64-bit name of a register, whatever width it was written at.
///
/// `edi` and `rdi` are one register, and a bound check written at 32 bits
/// names the same index the table read named at 64.
fn family(reg: &str) -> &str {
    match reg {
        "al" | "ah" | "ax" | "eax" | "rax" => "rax",
        "cl" | "ch" | "cx" | "ecx" | "rcx" => "rcx",
        "dl" | "dh" | "dx" | "edx" | "rdx" => "rdx",
        "sil" | "si" | "esi" | "rsi" => "rsi",
        "dil" | "di" | "edi" | "rdi" => "rdi",
        "bpl" | "bl" | "bh" | "bx" | "ebx" | "rbx" => "rbx",
        "spl" | "sp" | "esp" | "rsp" => "rsp",
        "r8b" | "r8w" | "r8d" | "r8" => "r8",
        "r9b" | "r9w" | "r9d" | "r9" => "r9",
        "r10b" | "r10w" | "r10d" | "r10" => "r10",
        "r11b" | "r11w" | "r11d" | "r11" => "r11",
        "r12b" | "r12w" | "r12d" | "r12" => "r12",
        "r13b" | "r13w" | "r13d" | "r13" => "r13",
        "r14b" | "r14w" | "r14d" | "r14" => "r14",
        "r15b" | "r15w" | "r15d" | "r15" => "r15",
        other => other,
    }
}

/// Where the table lives and how an entry is read.
fn shape(
    window: &[&Instruction],
    jmp: &Instruction,
) -> Option<(Addr, u8, bool, Option<String>)> {
    let operand = jmp.text.strip_prefix("jmp ")?;
    let before = &window[..window.len().saturating_sub(1)];

    // `jmp qword ptr [0x4020 + rax*8]`: the table address is in the operand.
    if operand.contains('[') {
        let mem = Mem::parse(operand)?;
        let width = match mem.scale {
            4 => 4,
            8 => 8,
            _ => return None,
        };
        // The address is usually a displacement, but position-independent code
        // loads it into a register first:
        //
        //     lea     rax, [rip + 0x85741]   # 0x4a61a0
        //     jmp     qword ptr [rax + rdx*8]
        //
        // Following the `lea` is the only way to reach that table, and a
        // switch whose table is never found leaves every case body unvisited.
        let base = match mem.disp {
            Some(disp) => Addr(disp as u64),
            None => lea_into(before, mem.base.as_deref()?)?,
        };
        return Some((base, width, false, mem.index));
    }

    // `jmp rax`, with the target computed or loaded into the register first.
    let reg = operand.trim();
    if reg.is_empty() || !reg.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    let start = window.len().saturating_sub(MAX_LOOKBACK);
    let slice = &window[start..before.len()];

    // Computed comes first and must win: a relative table is also loaded into
    // the register before the `add` that rebases it, so the load shape alone
    // cannot tell the two apart. Reading a relative table as absolute yields
    // no targets at all, which used to cost the whole switch body.
    if let Some(found) = computed_target(slice, reg) {
        return Some(found);
    }
    loaded_target(slice, reg)
}

/// `movsxd rcx, dword ptr [r10 + rax*4]` then `add rcx, r10` then `jmp rcx`:
/// the entries are 32-bit offsets from the table base, so `relative` is true.
///
/// The rebase has two spellings and both occur in glibc's string routines:
///
/// ```text
/// add  rcx, r10          ; table entry in rcx
/// lea  r10, [r10 + r9]   ; entry in r9, rebased in place
/// ```
///
/// The `lea` form is an `add` that writes the base register, and a table behind
/// it is just as real; missing it left `__strcasecmp_l_sse2` with a 7 KiB hole
/// in a 7.6 KiB body.
fn computed_target(slice: &[&Instruction], reg: &str) -> Option<(Addr, u8, bool, Option<String>)> {
    let (base_reg, entry_reg, at) = rebase(slice, reg)?;

    let base = lea_into(&slice[..at], base_reg)?;

    // Confirm the entry width rather than assuming it: a scaled load into the
    // entry register means 32-bit relative offsets when the scale is 4.
    let mut entry = None;
    for (at, insn) in slice[..at].iter().enumerate().rev() {
        if let Some((dst, mem)) = load_pair(&insn.text)
            && dst == entry_reg
            && mem.base.as_deref() == Some(base_reg)
            && matches!(mem.scale, 4 | 8)
        {
            // The load overwrites the index register with the entry, so the
            // value the cases count is whatever it held before this
            // instruction — trace it through the copies before the load, not
            // through the load.
            let index = mem.index.map(|r| traced_index(&slice[..at], &r));
            entry = Some((mem.scale as u8, index));
            break;
        }
    }
    let (entry_bytes, index) = entry?;
    Some((base, entry_bytes, true, index))
}

/// How the table entry reached the jumped register.
///
/// Returns the register holding the table base, the register the entry was
/// loaded into, and the index of the rebasing instruction so the search for the
/// base stops before it.
fn rebase<'a>(slice: &[&'a Instruction], reg: &'a str) -> Option<(&'a str, &'a str, usize)> {
    for (at, insn) in slice.iter().enumerate().rev() {
        if let Some((dst, src)) = add_pair(&insn.text)
            && dst == reg
        {
            // `add rcx, r10`: the entry is in rcx, the base in r10.
            return Some((src, reg, at));
        }
        if let Some((base, index)) = rebase_lea(&insn.text, reg) {
            return Some((base, index, at));
        }
        if writes_register(&insn.text, reg) {
            return None;
        }
    }
    None
}

/// `lea r10, [r10 + r9]` → ("r10", "r9"): the base register and the entry
/// register, when the destination is the jumped register and one of the two
/// operands is that same register.
fn rebase_lea<'a>(text: &'a str, reg: &str) -> Option<(&'a str, &'a str)> {
    let rest = text.strip_prefix("lea ")?;
    let (dst, src) = rest.split_once(", ")?;
    if dst.trim() != reg {
        return None;
    }
    let open = src.find('[')?;
    let close = src[open..].find(']')? + open;
    let mut regs: Vec<&str> = Vec::new();
    for term in src[open + 1..close].split(['+', '-']) {
        let term = term.trim();
        if term.is_empty() || term == "rip" || term.contains('*') {
            continue;
        }
        if term.starts_with(|c: char| c.is_ascii_digit()) || term.starts_with("0x") {
            continue;
        }
        regs.push(term);
    }
    if regs.len() != 2 {
        return None;
    }
    if regs[0] == reg {
        Some((regs[0], regs[1]))
    } else if regs[1] == reg {
        Some((regs[1], regs[0]))
    } else {
        None
    }
}

/// `mov rax, qword ptr [table + rdx*8]` then `jmp rax`: the entry is a pointer,
/// so `relative` is false.
///
/// A function-pointer call looks similar — `mov rax, [rbp - 8]; jmp rax` — and
/// is not a table: it has no index and no scale, which is what the checks below
/// reject.
fn loaded_target(slice: &[&Instruction], reg: &str) -> Option<(Addr, u8, bool, Option<String>)> {
    for (at, insn) in slice.iter().enumerate().rev() {
        let Some((dst, mem)) = load_pair(&insn.text) else {
            continue;
        };
        if dst != reg {
            continue;
        }
        if !matches!(mem.scale, 4 | 8) || mem.index.is_none() {
            continue;
        }
        let base = match mem.disp {
            Some(disp) => Addr(disp as u64),
            None => lea_into(&slice[..at], mem.base.as_deref()?)?,
        };
        let index = mem.index.map(|r| traced_index(&slice[..at], &r));
        return Some((base, mem.scale as u8, false, index));
    }
    None
}

fn read_targets(
    view: &View,
    base: Addr,
    width: u8,
    relative: bool,
    limit: Option<usize>,
) -> Vec<Addr> {
    let mut out: Vec<Addr> = Vec::new();
    let mut seen: BTreeSet<u64> = BTreeSet::new();
    for i in 0..MAX_ENTRIES {
        // The bound check said how many cases there are, and the table is
        // exactly that long. Reading past it reads the next table.
        if let Some(limit) = limit
            && i >= limit
        {
            break;
        }
        let slot = base.add((i * usize::from(width)) as u64);
        let Some(raw) = view.read_uint(slot, usize::from(width)) else {
            break;
        };
        let target = if relative {
            // Signed, and counted from the table base: the linker stores the
            // distance so the table stays position independent.
            let off = (raw as u32 as i32) as i64;
            Addr((base.0 as i64 + off) as u64)
        } else {
            Addr(raw)
        };
        if !view.is_exec(target) {
            break;
        }
        // A table entry that points back at the table is data, not a case.
        if target == base {
            break;
        }
        // Several cases can share one body — `case 1: case 2:` compiles to two
        // entries naming the same address — so a repeated target is a fact
        // about the switch, not the end of the table. Stopping here kept only
        // the first few entries and left every other case body undecoded.
        if !seen.insert(target.0) {
            continue;
        }
        out.push(target);
    }
    out
}

/// A memory operand, in the spelling this decoder produces.
///
/// `base` is the unscaled register, `index` the scaled one: `[rdx + rdi*4]` has
/// base `rdx` and index `rdi` with scale 4, and confusing the two makes every
/// table lookup look like it indexes off the wrong register.
struct Mem {
    base: Option<String>,
    index: Option<String>,
    scale: u32,
    disp: Option<i64>,
}

impl Mem {
    fn parse(text: &str) -> Option<Mem> {
        let open = text.find('[')?;
        let close = text[open..].find(']')? + open;
        let inner = &text[open + 1..close];
        let mut mem = Mem {
            base: None,
            index: None,
            scale: 1,
            disp: None,
        };
        for term in inner.split(['+', '-']) {
            let term = term.trim();
            if term.is_empty() || term == "rip" {
                continue;
            }
            if let Some((reg, scale)) = term.split_once('*') {
                mem.index = Some(reg.trim().to_string());
                mem.scale = scale.trim().parse().ok()?;
                continue;
            }
            if let Some(hex) = term.strip_prefix("0x") {
                let v = i64::from_str_radix(hex, 16).ok()?;
                mem.disp = Some(if inner.contains('-') { -v } else { v });
                continue;
            }
            if term.starts_with(|c: char| c.is_ascii_digit()) {
                mem.disp = Some(term.parse().ok()?);
                continue;
            }
            mem.base = Some(term.to_string());
        }
        Some(mem)
    }
}

/// `add rax, rdx` → ("rax", "rdx"). Register-to-register only.
fn add_pair(text: &str) -> Option<(&str, &str)> {
    let rest = text.strip_prefix("add ")?;
    let (dst, src) = rest.split_once(", ")?;
    if src.contains(['[', '*']) {
        return None;
    }
    Some((dst.trim(), src.trim()))
}

/// `lea rdx, qword ptr [rip+0xea4]  # 0x202c` → ("rdx", 0x202c).
///
/// The resolved address is taken from the decoder's own `#` annotation, which
/// is the same value it put in `Instruction::mem_reads`; re-deriving it from
/// the displacement would duplicate the RIP arithmetic.
fn lea_target(text: &str) -> Option<(&str, Addr)> {
    let rest = text.strip_prefix("lea ")?;
    let (dst, _) = rest.split_once(", ")?;
    let hash = text.find('#')?;
    let addr = text[hash + 1..].trim();
    let value = u64::from_str_radix(addr.strip_prefix("0x")?, 16).ok()?;
    Some((dst.trim(), Addr(value)))
}

/// Address the nearest preceding `lea` loaded into `reg`.
///
/// The walk stops at the first instruction that overwrites the register
/// without being a `lea`. An intervening `add rax, rdx` means the register
/// holds something else by the time the table is indexed, and naming the
/// older address would read the wrong table — or a table that is not there.
fn lea_into(before: &[&Instruction], reg: &str) -> Option<Addr> {
    for insn in before.iter().rev() {
        if let Some((dst, addr)) = lea_target(&insn.text) {
            if family(dst) == family(reg) {
                return Some(addr);
            }
            continue;
        }
        if writes_register(&insn.text, reg) {
            return None;
        }
    }
    None
}

/// True when the instruction's first operand is `reg`, i.e. it writes it.
///
/// Text-level and deliberately conservative: comparisons, tests, pushes and
/// branches do not write their first operand, and a memory destination is not
/// a register. The one case that matters here is distinguishing `mov rax, ...`
/// from `cmp rax, ...` while walking back to a `lea`.
fn writes_register(text: &str, reg: &str) -> bool {
    let Some((op, rest)) = text.split_once(' ') else {
        return false;
    };
    if matches!(
        op,
        "cmp" | "test" | "push" | "pop" | "call" | "jmp" | "nop" | "ret" | "leave"
    ) || op.starts_with('j')
    {
        return false;
    }
    let dst = rest.split(',').next().unwrap_or("").trim();
    if dst.is_empty() || !dst.chars().all(|c| c.is_ascii_alphanumeric()) {
        return false;
    }
    family(dst) == family(reg)
}

/// `movsxd rax, dword ptr [rdx + rdi*4]` → ("rax", mem).
fn load_pair(text: &str) -> Option<(&str, Mem)> {
    let (_, rest) = text.split_once(' ')?;
    let (dst, src) = rest.split_once(", ")?;
    let mem = Mem::parse(src)?;
    Some((dst.trim(), mem))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_operand_terms() {
        let m = Mem::parse("qword ptr [0x4020 + rax*8]").unwrap();
        assert_eq!(m.scale, 8);
        assert_eq!(m.disp, Some(0x4020));
        let m = Mem::parse("dword ptr [rdx + rdi*4]").unwrap();
        assert_eq!(m.base.as_deref(), Some("rdx"));
        assert_eq!(m.scale, 4);
        assert_eq!(m.disp, None);
        let m = Mem::parse("qword ptr [rip+0xea4]").unwrap();
        assert_eq!(m.disp, Some(0xea4));
    }

    #[test]
    fn a_table_loaded_into_a_register_is_found_through_its_lea() {
        // Position-independent code loads the address first, so the jump
        // operand has no displacement of its own:
        //     lea rax, [rip + 0x85741]   # 0x4a61a0
        //     jmp qword ptr [rax + rdx*8]
        let owned = [
            insn("movzx edx, byte ptr [rsi + rax]"),
            insn("lea rax, qword ptr [rip+0x85741]  # 0x4a61a0"),
            insn("jmp qword ptr [rax + rdx*8]"),
        ];
        let window: Vec<&Instruction> = owned.iter().collect();
        assert_eq!(
            shape(&window, window.last().unwrap()),
            Some((Addr(0x4a61a0), 8, false, Some("rdx".to_string())))
        );
    }

    #[test]
    fn a_lea_rebase_keeps_the_table_relative() {
        // glibc's string routines rebase with `lea` instead of `add`:
        //     lea    r10, [rip + 0x277ff]   # 0x488500
        //     movslq r9, [r10 + r9*4]
        //     lea    r10, [r10 + r9]
        //     jmp    r10
        // The entries are 32-bit offsets from the table base, so reading them
        // as absolute addresses yields no targets at all.
        let owned = [
            insn("lea r10, qword ptr [rip+0x277ff]  # 0x488500"),
            insn("movslq r9, dword ptr [r10 + r9*4]"),
            insn("lea r10, qword ptr [r10 + r9]"),
            insn("jmp r10"),
        ];
        let window: Vec<&Instruction> = owned.iter().collect();
        assert_eq!(
            shape(&window, window.last().unwrap()),
            Some((Addr(0x488500), 4, true, Some("r9".to_string())))
        );
    }

    #[test]
    fn a_function_pointer_load_is_not_a_table() {
        // `mov rax, [rbp - 8]; jmp rax` is a call through a local, not a
        // switch: no index and no scale.
        let owned = [
            insn("mov rax, qword ptr [rbp - 0x8]"),
            insn("jmp rax"),
        ];
        let window: Vec<&Instruction> = owned.iter().collect();
        assert_eq!(shape(&window, window.last().unwrap()), None);
    }

    #[test]
    fn rip_relative_lea_reads_the_annotation() {
        let (reg, addr) =
            lea_target("lea rdx, qword ptr [rip+0xea4]  # 0x202c").unwrap();
        assert_eq!(reg, "rdx");
        assert_eq!(addr, Addr(0x202c));
    }

    #[test]
    fn register_add_is_not_confused_with_memory_add() {
        assert_eq!(add_pair("add rax, rdx"), Some(("rax", "rdx")));
        assert_eq!(add_pair("add rax, qword ptr [rbx]"), None);
    }

    #[test]
    fn a_bound_check_says_how_many_cases_there_are() {
        let owned = [
            insn("cmp edi, 0x9"),
            insn("ja 0x1040 (rel -0x1ed)"),
            insn("jmp rax"),
        ];
        let window: Vec<&Instruction> = owned.iter().collect();
        assert_eq!(case_count(&window, Some("rdi")), Some(10));
    }

    #[test]
    fn a_subtraction_before_the_lookup_names_the_first_case() {
        let owned = [
            insn("sub edi, 0x64"),
            insn("cmp edi, 0x6"),
            insn("ja 0x1044"),
            insn("jmp rax"),
        ];
        let window: Vec<&Instruction> = owned.iter().collect();
        assert_eq!(first_case(&window, Some("rdi")), Some(100));
    }

    #[test]
    fn an_addition_names_the_first_case_the_other_way_round() {
        let owned = [insn("add edi, -0x64"), insn("jmp rax")];
        let window: Vec<&Instruction> = owned.iter().collect();
        assert_eq!(first_case(&window, Some("rdi")), Some(100));
    }

    #[test]
    fn a_subtraction_on_another_register_names_nothing() {
        let owned = [insn("sub esi, 0x64"), insn("jmp rax")];
        let window: Vec<&Instruction> = owned.iter().collect();
        assert_eq!(first_case(&window, Some("rdi")), None);
    }

    #[test]
    fn a_bound_check_on_another_register_bounds_nothing() {
        let owned = [insn("cmp esi, 0x9"), insn("ja 0x1040")];
        let window: Vec<&Instruction> = owned.iter().collect();
        assert_eq!(case_count(&window, Some("rdi")), None);
    }

    #[test]
    fn a_comparison_without_a_branch_bounds_nothing() {
        let owned = [insn("cmp edi, 0x9"), insn("mov eax, 1")];
        let window: Vec<&Instruction> = owned.iter().collect();
        assert_eq!(case_count(&window, Some("rdi")), None);
    }

    #[test]
    fn the_last_bound_check_is_the_one_that_counts() {
        let owned = [
            insn("cmp edi, 0x3"),
            insn("ja 0x1040"),
            insn("cmp edi, 0x9"),
            insn("ja 0x1044"),
        ];
        let window: Vec<&Instruction> = owned.iter().collect();
        assert_eq!(case_count(&window, Some("rdi")), Some(10));
    }

    #[test]
    fn register_widths_do_not_change_the_register() {
        assert_eq!(family("edi"), family("rdi"));
        assert_eq!(family("ax"), family("rax"));
        assert_ne!(family("esi"), family("rdi"));
    }

    #[test]
    fn a_comparison_against_an_immediate_is_read() {
        assert_eq!(cmp_against_immediate("cmp edi, 0x9"), Some(("edi", 9)));
        assert_eq!(cmp_against_immediate("cmp edi, 0x0"), Some(("edi", 0)));
        assert_eq!(cmp_against_immediate("cmp eax, esi"), None);
    }

    #[test]
    fn a_table_whose_cases_share_a_body_keeps_its_later_entries() {
        // `case 1: case 2:` compiles to two slots holding one address. Reading
        // used to stop at the first repeat, which kept the first two cases of a
        // real switch and left the rest of the function undecoded.
        let mut data = vec![0u8; 0x100];
        for (slot, offset) in [(0usize, 0x10u32), (4, 0x10), (8, 0x20), (12, 0x40)] {
            data[slot..slot + 4].copy_from_slice(&offset.to_le_bytes());
        }
        let image = crate::testutil::image_with_code(&data);
        let view = View::new(&image, &data);
        let targets = read_targets(&view, Addr(0x1000), 4, true, None);
        assert_eq!(targets, vec![Addr(0x1010), Addr(0x1020), Addr(0x1040)]);
    }

    #[test]
    fn a_widened_index_is_traced_back_to_the_value_the_cases_count() {
        // `movzx eax, sil` feeds the table, and the bound check that says how
        // many cases there are stays on `sil`. Naming the index `rax` printed
        // the switch against the computed jump address.
        let owned = [
            insn("movzx eax, sil"),
            insn("movsxd rax, dword ptr [rdx + rax*4]"),
            insn("add rax, rdx"),
            insn("jmp rax"),
        ];
        let window: Vec<&Instruction> = owned.iter().collect();
        assert_eq!(traced_index(&window[..1], "rax"), "rsi");
        // Without a widening copy the register is its own index.
        assert_eq!(traced_index(&window[..1], "rdi"), "rdi");
    }

    fn insn(text: &str) -> Instruction {
        Instruction {
            addr: Addr(0x1000),
            len: 2,
            bytes: [0; occ_core::MAX_INSN_LEN],
            text: text.into(),
            mnemonic: text.split(' ').next().unwrap_or("t").into(),
            flow: FlowKind::Next,
            target: None,
            mem_reads: Vec::new(),
            mem_writes: Vec::new(),
        }
    }
}
