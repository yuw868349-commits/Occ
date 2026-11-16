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

    // `jmp qword ptr [0x4020 + rax*8]`: the table address is in the operand.
    if operand.contains('[') {
        let mem = Mem::parse(operand)?;
        let base = mem.disp?;
        let width = match mem.scale {
            4 => 4,
            8 => 8,
            _ => return None,
        };
        return Some((Addr(base as u64), width, false, mem.index));
    }

    // `jmp rax`: walk back for the three instructions that built the target.
    let reg = operand.trim();
    if reg.is_empty() || !reg.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    let start = window.len().saturating_sub(MAX_LOOKBACK);
    let slice = &window[start..window.len().saturating_sub(1)];

    let mut base_reg: Option<&str> = None;
    for insn in slice.iter().rev() {
        if let Some((dst, src)) = add_pair(&insn.text)
            && dst == reg
        {
            base_reg = Some(src);
            break;
        }
    }
    let base_reg = base_reg?;

    let mut base = None;
    for insn in slice.iter().rev() {
        if let Some((dst, addr)) = lea_target(&insn.text)
            && dst == base_reg
        {
            base = Some(addr);
            break;
        }
    }
    let base = base?;

    // Confirm the entry width rather than assuming it: a scaled load into the
    // jumped register with scale 4 means 32-bit relative offsets.
    let mut entry = None;
    for insn in slice.iter().rev() {
        if let Some((dst, mem)) = load_pair(&insn.text)
            && dst == reg
            && mem.base.as_deref() == Some(base_reg)
            && matches!(mem.scale, 4 | 8)
        {
            entry = Some((mem.scale as u8, mem.index));
            break;
        }
    }
    let (entry_bytes, index) = entry?;
    Some((base, entry_bytes, true, index))
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
        if !seen.insert(target.0) {
            break;
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
