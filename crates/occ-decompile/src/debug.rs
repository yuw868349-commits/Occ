//! Source-level names and types, taken from debug information.
//!
//! The lifter works from machine code alone, so it names what it sees: `a0`
//! for the first argument register, `local_8_u64` for a frame slot. When the
//! image carries DWARF, the compiler has already answered those questions —
//! the parameter is `p`, the slot is `count`, the argument is
//! `struct point *`. This module carries that answer into the lifted program.
//!
//! # What it does not do
//!
//! It does not invent a location. A variable whose debug location is a
//! register this decompiler does not model, or a frame base that is not the
//! canonical frame address, keeps the name the lifter gave it. A wrong name
//! is worse than a mechanical one: it makes the reader trust a value that is
//! not there.
//!
//! # Frame slots
//!
//! DWARF locates a stack variable relative to the frame base, which on the
//! targets here is the canonical frame address: the stack pointer at the
//! call, before the callee adjusted it. The lifter's slot numbers are relative
//! to the same address, so the two agree up to the return address that a
//! `call` pushes. [`DebugInfo::slot_of`] applies that one offset.

use std::collections::HashMap;

use occ_core::Addr;

use crate::ir::{Expr, Stmt};
use crate::lift::slot_name;
use crate::Program;

/// Everything the debug information says about one function.
#[derive(Clone, Debug, Default)]
pub struct DebugInfo {
    /// Name as written in the source.
    pub name: Option<String>,
    /// Source file that declares the function.
    pub file: Option<String>,
    /// Declared return type, already rendered as source text.
    pub return_type: Option<String>,
    /// Parameters, in declaration order.
    pub params: Vec<DebugVar>,
    /// Local variables.
    pub locals: Vec<DebugVar>,
    /// Aggregate types the function's variables use.
    pub types: Vec<DebugType>,
    /// Line table entries covering the function, sorted by address.
    pub lines: Vec<DebugLine>,
    /// True when the frame base is the canonical frame address.
    ///
    /// That is what makes `DW_OP_fbreg` convertible to a lifter slot.
    pub frame_base_is_cfa: bool,
    /// The register the frame base lives in, when it is not the CFA.
    ///
    /// Clang at `-O0` writes `DW_AT_frame_base` as `DW_OP_reg6` on x86-64,
    /// which is `rbp`: every `fbreg` offset is relative to the frame pointer,
    /// not to the CFA. Treating the two as the same put every local four
    /// bytes off.
    pub frame_base_reg: Option<String>,
}

/// Where a variable lives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DebugLoc {
    /// A frame slot, as `DW_OP_fbreg` states it: bytes from the frame base.
    Frame(i64),
    /// A register, by the architecture's canonical name (`rdi`, `x0`, `a0`).
    Reg(String),
    /// A location this decompiler cannot map.
    Unknown,
}

/// One named variable.
#[derive(Clone, Debug)]
pub struct DebugVar {
    /// Source name.
    pub name: String,
    /// Declared type, as source text.
    pub type_name: Option<String>,
    /// Where it lives.
    pub loc: DebugLoc,
}

/// One aggregate type.
#[derive(Clone, Debug)]
pub struct DebugType {
    /// Type name, as the source spells it.
    pub name: String,
    /// `struct`, `union`, `class` or another DWARF kind.
    pub kind: String,
    /// Size in bytes, when stated.
    pub size: Option<u64>,
    /// Members, in declaration order.
    pub members: Vec<DebugMember>,
}

impl DebugType {
    /// Member at `offset`, when one starts exactly there.
    pub fn member_at(&self, offset: u64) -> Option<&DebugMember> {
        self.members.iter().find(|m| m.offset == offset)
    }
}

/// One struct or union member.
#[derive(Clone, Debug)]
pub struct DebugMember {
    /// Member name.
    pub name: String,
    /// Declared type.
    pub type_name: Option<String>,
    /// Byte offset from the start of the aggregate.
    pub offset: u64,
    /// Size in bytes, when stated.
    pub size: Option<u64>,
}

/// One line-table row that covers this function.
#[derive(Clone, Debug)]
pub struct DebugLine {
    /// First address the row covers.
    pub addr: Addr,
    /// Source line.
    pub line: u32,
    /// Source file, when it differs from the function's file.
    pub file: Option<String>,
}

/// A source range, for the header comment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceRef {
    /// File that contains the function.
    pub file: String,
    /// First source line the body covers.
    pub first_line: u32,
    /// Last source line the body covers.
    pub last_line: u32,
}

impl DebugInfo {
    /// The lifter slot a frame-relative location names.
    ///
    /// `cfa_slot` is the slot of the canonical frame address: 8 on x86-64,
    /// where the call pushed a return address above the caller's stack
    /// pointer, and 4 on x86-32. The other targets keep the return address in
    /// a register, so the frame base is the entry stack pointer and the offset
    /// is zero.
    pub fn slot_of(&self, offset: i64, frame_base_slot: Option<i64>) -> Option<i64> {
        Some(frame_base_slot?.wrapping_add(offset))
    }

    /// Source line covering `addr`, from the rows this function owns.
    pub fn line_at(&self, addr: Addr) -> Option<(&str, u32)> {
        let idx = self.lines.partition_point(|l| l.addr.0 <= addr.0);
        let row = idx.checked_sub(1).and_then(|i| self.lines.get(i))?;
        let file = row.file.as_deref().or(self.file.as_deref())?;
        Some((file, row.line))
    }

    /// The source range the function spans, when the line table says.
    pub fn source_range(&self, file_fallback: Option<&str>) -> Option<SourceRef> {
        let file = self.file.as_deref().or(file_fallback)?;
        let first = self.lines.first()?.line;
        let last = self.lines.iter().map(|l| l.line).max()?;
        Some(SourceRef {
            file: file.to_string(),
            first_line: first.min(last),
            last_line: last,
        })
    }

    /// The type of a variable, for the signature.
    pub fn param_types(&self) -> Vec<Option<String>> {
        self.params
            .iter()
            .map(|p| p.type_name.clone())
            .collect()
    }

    /// Render a parameter declaration: `struct point *p`, or `u64 p` when the
    /// type is unknown.
    pub fn param_decl(&self, i: usize) -> String {
        let Some(p) = self.params.get(i) else {
            return String::new();
        };
        match p.type_name.as_deref() {
            Some(t) => format!("{t} {}", p.name),
            None => format!("u64 {}", p.name),
        }
    }
}

/// Apply debug names to a lifted program.
///
/// `cfa_slot` is the frame slot of the canonical frame address, as
/// [`crate::lift::Target::return_address_slot`] states it. Parameters whose
/// debug location is a register rename the parameter alias the lifter already
/// created, so the body reads the source name rather than `a0`; frame-located
/// parameters and locals rename the slot they occupy.
///
/// Names that cannot be mapped are left alone, and the function keeps saying
/// `local_8_u64`, which is true, rather than `count`, which would be a guess.
pub fn apply(program: &mut Program, info: &DebugInfo, frame_base_slot: Option<i64>) {
    let mut renames: Vec<(String, String)> = Vec::new();
    let mut aliases: HashMap<String, String> = program.param_aliases.iter().cloned().collect();
    let mut alias_order: Vec<String> = program.param_aliases.iter().map(|(r, _)| r.clone()).collect();

    for (i, p) in info.params.iter().enumerate() {
        match &p.loc {
            DebugLoc::Reg(reg) => {
                if aliases.contains_key(reg) {
                    aliases.insert(reg.clone(), p.name.clone());
                } else {
                    // A register the ABI probe did not count as an argument:
                    // the debug info says it carries one, so name it.
                    alias_order.push(reg.clone());
                    aliases.insert(reg.clone(), p.name.clone());
                }
            }
            DebugLoc::Frame(off) => {
                if let Some(slot) = info.slot_of(*off, frame_base_slot) {
                    for width in [1u8, 2, 4, 8, 10] {
                        renames.push((slot_name(slot, width), p.name.clone()));
                    }
                }
                // A parameter the debug info places in a frame slot arrived in
                // an argument register: the body starts with `mov [rbp-0x24],
                // edi`, and the slot is what the code reads afterwards. Naming
                // only the slot leaves `a = (u32)rdi` in the output, and the
                // copy pass folds `a` away, so the parameter loses its name at
                // exactly the place a reader looks first. The ABI register the
                // parameter occupies is known, and it is the same value.
                if let Some(reg) = program.param_regs.get(i) {
                    if !aliases.contains_key(reg) {
                        alias_order.push(reg.clone());
                    }
                    aliases.insert(reg.clone(), p.name.clone());
                }
            }
            DebugLoc::Unknown => {}
        }
    }
    for v in &info.locals {
        if let DebugLoc::Frame(off) = &v.loc
            && let Some(slot) = info.slot_of(*off, frame_base_slot)
        {
            for width in [1u8, 2, 4, 8, 10] {
                renames.push((slot_name(slot, width), v.name.clone()));
            }
        }
    }

    // Longest base first, so `local_8_u64` is tried before `local_8`.
    renames.sort_by_key(|(base, _)| std::cmp::Reverse(base.len()));
    let map: HashMap<&str, &str> = renames
        .iter()
        .map(|(base, name)| (base.as_str(), name.as_str()))
        .collect();

    if !renames.is_empty() {
        let rename = |v: &str| -> Option<String> {
            if let Some(name) = map.get(v) {
                return Some((*name).to_string());
            }
            let (base, rest) = v.split_once('.')?;
            let name = map.get(base)?;
            Some(format!("{name}.{rest}"))
        };
        for b in &mut program.blocks {
            for s in &mut b.stmts {
                rename_stmt(s, &rename);
            }
            // The condition that ends a block is not a statement. A branch
            // test that kept `local_14_u32` while the assignment above it read
            // `n` was the visible half of a half-renamed body.
            match &mut b.term {
                crate::ir::Term::Branch { cond, .. } => *cond = cond.rename_vars(&rename),
                crate::ir::Term::Switch { index, .. } => *index = index.rename_vars(&rename),
                crate::ir::Term::Ret(Some(v)) => *v = v.rename_vars(&rename),
                crate::ir::Term::Indirect { target } => *target = target.rename_vars(&rename),
                _ => {}
            }
        }
    }

    if !info.params.is_empty() {
        program.params = info.params.iter().map(|p| p.name.clone()).collect();
    }
    if !aliases.is_empty() {
        program.param_aliases = alias_order
            .into_iter()
            .filter_map(|r| aliases.get(&r).map(|a| (r.clone(), a.clone())))
            .collect();
    }
    if let Some(name) = &info.name {
        if let Some(note) = program
            .notes
            .iter_mut()
            .find(|n| n.starts_with("parameters are inferred"))
        {
            *note = format!(
                "parameter names and types come from debug information; the signature is {name}'s"
            );
        } else {
            program.notes.push(format!(
                "names and types come from debug information for {name}"
            ));
        }
    }

    // Member accesses last: the base now carries its source name, which is the
    // name the type table keys on. The expression still holds the register
    // name — aliases are applied at render time — so the lookup goes through
    // the same table the renderer will use.
    let alias_of = |v: &str| -> String {
        if let Some(a) = aliases.get(v) {
            return a.clone();
        }
        if let Some((base, rest)) = v.split_once('.')
            && let Some(a) = aliases.get(base)
        {
            return format!("{a}.{rest}");
        }
        v.to_string()
    };
    for b in &mut program.blocks {
        for s in &mut b.stmts {
            match s {
                Stmt::Assign { value, .. } => {
                    if let Some(m) = rewrite_member_load(info, value, &alias_of) {
                        *value = m;
                    }
                }
                Stmt::Store { value, .. } => {
                    if let Some(m) = rewrite_member_load(info, value, &alias_of) {
                        *value = m;
                    }
                }
                Stmt::Call { args, .. } => {
                    for a in args.iter_mut() {
                        if let Some(m) = rewrite_member_load(info, a, &alias_of) {
                            *a = m;
                        }
                    }
                }
                Stmt::Comment(_) => {}
            }
        }
    }
}

fn rename_stmt(s: &mut Stmt, f: &dyn Fn(&str) -> Option<String>) {
    match s {
        Stmt::Assign { dest, value } => {
            if let Some(n) = f(dest) {
                *dest = n;
            }
            *value = value.rename_vars(f);
        }
        Stmt::Store { addr, value, .. } => {
            *addr = addr.rename_vars(f);
            *value = value.rename_vars(f);
        }
        Stmt::Call { args, ret, .. } => {
            for a in args.iter_mut() {
                *a = a.rename_vars(f);
            }
            if let Some(r) = ret
                && let Some(n) = f(r)
            {
                *r = n;
            }
        }
        Stmt::Comment(_) => {}
    }
}

/// The name a slot would print under if a debug variable owned it.
///
/// Kept public for the tools that need to explain a mapping without applying
/// it, and for tests.
pub fn slot_for(off: i64, frame_base_slot: i64, width: u8) -> String {
    slot_name(frame_base_slot.wrapping_add(off), width)
}

/// A value's type, when a debug variable of that name states one.
pub fn type_of<'a>(info: &'a DebugInfo, name: &str) -> Option<&'a str> {
    let base = name.split_once('.').map(|(b, _)| b).unwrap_or(name);
    info.params
        .iter()
        .chain(info.locals.iter())
        .find(|v| v.name == base)
        .and_then(|v| v.type_name.as_deref())
}

/// True when `ty` names a pointer.
pub fn is_pointer(ty: &str) -> bool {
    ty.contains('*') || ty.ends_with("ptr")
}

/// The aggregate a pointer type points at, when the type table names one.
pub fn pointee<'a>(info: &'a DebugInfo, ty: &str) -> Option<&'a DebugType> {
    let base = ty
        .trim()
        .trim_end_matches('*')
        .trim()
        .trim_start_matches("const ")
        .trim();
    // DWARF spells an aggregate reference both ways: the DIE's name is
    // `point`, while a variable's type prints as `struct point`.
    let bare = base
        .strip_prefix("struct ")
        .or_else(|| base.strip_prefix("union "))
        .or_else(|| base.strip_prefix("class "))
        .unwrap_or(base);
    info.types
        .iter()
        .find(|t| t.name == base || t.name == bare)
}

/// The expression for a member access, when one applies.
///
/// A load through `base + offset` where `base` is a pointer to an aggregate
/// and a member starts at `offset` is that member. Returning `None` keeps the
/// explicit `*(T *)(base + n)` form, which is what the machine said.
pub fn member_access<'a>(
    info: &'a DebugInfo,
    base: &str,
    offset: u64,
) -> Option<(&'a DebugMember, &'a DebugType)> {
    let ty = type_of(info, base)?;
    if !is_pointer(ty) {
        return None;
    }
    let agg = pointee(info, ty)?;
    let member = agg.member_at(offset)?;
    Some((member, agg))
}

/// Rebuild a `Load` expression as a member access, when debug types allow.
///
/// This is the expression-level form of [`member_access`]: `*(u32 *)(p + 4)`
/// where `p` points at `struct point` and `y` sits at offset 4 is `p->y`.
pub fn rewrite_member_load(
    info: &DebugInfo,
    e: &Expr,
    alias: &dyn Fn(&str) -> String,
) -> Option<Expr> {
    let Expr::Load { addr, size, signed } = e else {
        return None;
    };
    // `p->x` where `x` is the first member loads through the pointer itself:
    // there is no `+ 0` in the machine code to key on, so the bare variable is
    // the zero-offset case rather than a different shape.
    let (base, off) = match addr.as_ref() {
        Expr::Var(v) => (v, 0u64),
        Expr::Bin {
            op: crate::ir::BinOp::Add,
            lhs,
            rhs,
            ..
        } => match (lhs.as_ref(), rhs.as_ref()) {
            (Expr::Var(base), Expr::Const(off)) => (base, *off),
            _ => return None,
        },
        _ => return None,
    };
    let (member, _) = member_access(info, &alias(base), off)?;
    if member.size.is_some_and(|s| u64::from(*size) != s) {
        return None;
    }
    let _ = signed;
    Some(Expr::Member {
        base: Box::new(Expr::Var(base.clone())),
        offset: off,
        size: *size,
        name: member.name.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Stmt;

    fn info() -> DebugInfo {
        DebugInfo {
            name: Some("total".into()),
            file: Some("sum.c".into()),
            frame_base_is_cfa: true,
            params: vec![
                DebugVar {
                    name: "p".into(),
                    type_name: Some("struct point *".into()),
                    loc: DebugLoc::Reg("rdi".into()),
                },
                DebugVar {
                    name: "n".into(),
                    type_name: Some("int".into()),
                    loc: DebugLoc::Reg("rsi".into()),
                },
            ],
            locals: vec![DebugVar {
                name: "total".into(),
                type_name: Some("int".into()),
                loc: DebugLoc::Frame(-20),
            }],
            ..DebugInfo::default()
        }
    }

    #[test]
    fn frame_slots_account_for_the_return_address() {
        let d = info();
        // `fbreg -20` is rbp-4 in a frame whose CFA is rbp+16, and rbp sits
        // eight bytes below the entry stack pointer: slot -12.
        assert_eq!(d.slot_of(-20, Some(8)), Some(-12));
        // No frame base, no mapping: guessing would put the name on the wrong
        // slot.
        let mut d = d;
        d.frame_base_is_cfa = false;
        assert_eq!(d.slot_of(-20, None), None);
    }

    #[test]
    fn renames_frame_locals_and_register_parameters() {
        let mut p = Program {
            entry: 0,
            blocks: vec![crate::ir::Block {
                id: 0,
                addr: Addr(0x1000),
                stmts: vec![
                    Stmt::Assign {
                        dest: "local_c_u32.1".into(),
                        value: Expr::var("local_c_u32"),
                    },
                    Stmt::Call {
                        name: "f".into(),
                        args: vec![Expr::var("rdi")],
                        ret: None,
                        addr: Addr(0x1004),
                    },
                ],
                term: crate::ir::Term::Ret(None),
            }],
            preds: vec![Vec::new()],
            succs: vec![Vec::new()],
            params: vec!["a0".into(), "a1".into()],
            param_regs: vec!["rdi".into(), "rsi".into()],
            param_aliases: vec![
                ("rdi".into(), "a0".into()),
                ("rsi".into(), "a1".into()),
            ],
            ret_type: None,
            notes: Vec::new(),
        };
        apply(&mut p, &info(), Some(8));
        assert_eq!(p.params, ["p", "n"]);
        assert_eq!(p.param_aliases[0], ("rdi".into(), "p".into()));
        match &p.blocks[0].stmts[0] {
            Stmt::Assign { dest, .. } => assert_eq!(dest, "total.1"),
            _ => panic!("expected assign"),
        }
        match &p.blocks[0].stmts[1] {
            // The body still names the register; the renderer turns it into
            // the parameter name through `param_aliases`, which is where the
            // debug name was written.
            Stmt::Call { args, .. } => assert_eq!(args, &[Expr::var("rdi")]),
            _ => panic!("expected call"),
        }
        assert!(p.param_aliases.contains(&("rdi".into(), "p".into())));
    }

    #[test]
    fn line_lookup_uses_the_last_row_at_or_before_the_address() {
        let d = DebugInfo {
            file: Some("sum.c".into()),
            lines: vec![
                DebugLine {
                    addr: Addr(0x1000),
                    line: 3,
                    file: None,
                },
                DebugLine {
                    addr: Addr(0x1010),
                    line: 7,
                    file: None,
                },
            ],
            ..DebugInfo::default()
        };
        assert_eq!(d.line_at(Addr(0x1008)), Some(("sum.c", 3)));
        assert_eq!(d.line_at(Addr(0x1010)), Some(("sum.c", 7)));
        assert_eq!(d.line_at(Addr(0xfff)), None);
    }

    #[test]
    fn member_access_needs_a_pointer_to_a_named_aggregate() {
        let d = DebugInfo {
            params: vec![DebugVar {
                name: "p".into(),
                type_name: Some("struct point *".into()),
                loc: DebugLoc::Reg("rdi".into()),
            }],
            types: vec![DebugType {
                name: "point".into(),
                kind: "struct".into(),
                size: Some(8),
                members: vec![
                    DebugMember {
                        name: "x".into(),
                        type_name: Some("int".into()),
                        offset: 0,
                        size: Some(4),
                    },
                    DebugMember {
                        name: "y".into(),
                        type_name: Some("int".into()),
                        offset: 4,
                        size: Some(4),
                    },
                ],
            }],
            ..DebugInfo::default()
        };
        let (m, _) = member_access(&d, "p", 4).unwrap();
        assert_eq!(m.name, "y");
        assert!(member_access(&d, "p", 2).is_none());
    }
}
