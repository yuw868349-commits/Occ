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
    let targets = read_targets(view, base, entry_bytes, relative);
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
        targets,
    })
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

fn read_targets(view: &View, base: Addr, width: u8, relative: bool) -> Vec<Addr> {
    let mut out: Vec<Addr> = Vec::new();
    let mut seen: BTreeSet<u64> = BTreeSet::new();
    for i in 0..MAX_ENTRIES {
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
}
