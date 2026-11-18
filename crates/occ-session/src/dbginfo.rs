//! Debug information, in the shape the decompiler reads it.
//!
//! The DWARF reader in `occ-dwarf` answers questions about a file; the
//! decompiler wants one answer about one function, in the form its
//! [`DebugInfo`] names. This module is that translation, kept out of the MCP
//! layer so a test, a command-line tool and the server all describe a
//! function the same way.
//!
//! What it will not do is guess. A variable whose location is a DWARF
//! expression the lifter does not model stays unnamed rather than being
//! assigned to a plausible slot.

use occ_core::Function;

use crate::ImageFile;

/// The name a DWARF register number refers to on this architecture.
///
/// DWARF numbers registers its own way, and it is not the numbering the
/// decoder uses: on x86-64 DWARF's 1 is `rdx` while the decoding table's 1 is
/// `rcx`. Mapping by index would put every parameter in the wrong register,
/// so the table is spelled out.
pub fn dwarf_reg_name(arch: occ_core::Arch, n: u64) -> Option<String> {
    use occ_core::Arch;
    let name = match arch {
        Arch::X86_64 => match n {
            0 => "rax",
            1 => "rdx",
            2 => "rcx",
            3 => "rbx",
            4 => "rsi",
            5 => "rdi",
            6 => "rbp",
            7 => "rsp",
            8..=15 => return Some(format!("r{n}")),
            _ => return None,
        },
        Arch::X86 => match n {
            0 => "eax",
            1 => "ecx",
            2 => "edx",
            3 => "ebx",
            4 => "esp",
            5 => "ebp",
            6 => "esi",
            7 => "edi",
            _ => return None,
        },
        Arch::Aarch64 => match n {
            0..=30 => return Some(format!("x{n}")),
            31 => "sp",
            _ => return None,
        },
        Arch::RiscV32 | Arch::RiscV64 => match n {
            0 => "zero",
            1 => "ra",
            2 => "sp",
            3 => "gp",
            4 => "tp",
            5 => "t0",
            6 => "t1",
            7 => "t2",
            8 => "s0",
            9 => "s1",
            10..=17 => return Some(format!("a{}", n - 10)),
            18..=27 => return Some(format!("s{}", n - 16)),
            28..=31 => return Some(format!("t{}", n - 25)),
            _ => return None,
        },
        _ => return None,
    };
    Some(name.to_string())
}

/// Read a DWARF location expression into the form the decompiler can apply.
///
/// Only the two forms that map to something the lifter models are accepted:
/// a frame-relative offset and a whole register. A composite expression
/// (`breg5+8 deref_size 4`, `entry_value(...)`) is left unknown rather than
/// approximated, because a name on the wrong value is worse than `local_8`.
pub fn dwarf_location(
    at: occ_core::Addr,
    var: &occ_dwarf::Variable,
    arch: occ_core::Arch,
) -> occ_decompile::debug::DebugLoc {
    use occ_decompile::debug::DebugLoc;
    let Some(text) = var.location_at(at) else {
        return DebugLoc::Unknown;
    };
    if let Some(rest) = text.strip_prefix("fbreg")
        && let Ok(off) = rest.parse::<i64>()
    {
        return DebugLoc::Frame(off);
    }
    if let Some(rest) = text.strip_prefix("reg")
        && let Ok(n) = rest.parse::<u64>()
        && let Some(name) = dwarf_reg_name(arch, n)
    {
        return DebugLoc::Reg(name);
    }
    DebugLoc::Unknown
}

/// Build the debug view of one function from the image's DWARF.
pub fn debug_info_for(file: &ImageFile, f: &Function) -> Option<occ_decompile::debug::DebugInfo> {
    use occ_decompile::debug::{DebugInfo, DebugLine, DebugMember, DebugType, DebugVar};
    let dw = file.dwarf.as_ref()?;
    let df = dw.function_for(f.entry)?;
    let arch = file.image.arch;
    let types: Vec<DebugType> = dw
        .types()
        .filter(|t| !t.members.is_empty())
        .map(|t| DebugType {
            name: t.name.clone(),
            kind: t.kind.clone(),
            size: t.byte_size,
            members: t
                .members
                .iter()
                .map(|m| DebugMember {
                    name: m.name.clone(),
                    type_name: m.type_name.clone(),
                    offset: m.offset,
                    size: m.size,
                })
                .collect(),
        })
        .collect();
    // DWARF names the aggregate `point` while C spells the type `struct
    // point`. The tag belongs in a declaration, so put it back.
    let kinds: Vec<(String, String)> = types
        .iter()
        .filter(|t| matches!(t.kind.as_str(), "struct" | "union" | "class" | "enum"))
        .map(|t| (t.name.clone(), t.kind.clone()))
        .collect();
    let qualify = |ty: &Option<String>| -> Option<String> {
        let text = ty.as_deref()?;
        for (name, kind) in &kinds {
            let bare = text == name;
            let compound = text
                .strip_prefix(name)
                .is_some_and(|rest| rest.starts_with(' ') || rest.starts_with('*'));
            if bare || compound {
                return Some(format!("{kind} {text}"));
            }
        }
        Some(text.to_string())
    };
    let vars = |list: &[occ_dwarf::Variable]| -> Vec<DebugVar> {
        list.iter()
            .filter(|v| !v.name.is_empty() && v.name != "?")
            .map(|v| DebugVar {
                name: v.name.clone(),
                type_name: qualify(&v.type_name),
                loc: dwarf_location(f.entry, v, arch),
            })
            .collect()
    };
    let mut lines: Vec<DebugLine> = Vec::new();
    for u in &dw.units {
        for row in &u.lines.rows {
            if row.end_sequence || row.addr.0 < df.low_pc.0 || row.addr.0 >= df.high_pc.0 {
                continue;
            }
            lines.push(DebugLine {
                addr: row.addr,
                line: row.line,
                file: (df.decl_file.as_deref() != Some(row.file.as_str()))
                    .then(|| row.file.clone()),
            });
        }
    }
    lines.sort_by_key(|l| l.addr.0);
    lines.dedup_by_key(|l| l.addr.0);
    let (frame_base_is_cfa, frame_base_reg) = match df.frame_base.as_deref() {
        None => (true, None),
        Some(text) if text.contains("call_frame_cfa") => (true, None),
        Some(text) => (
            false,
            text.strip_prefix("reg")
                .and_then(|n| n.parse::<u64>().ok())
                .and_then(|n| dwarf_reg_name(arch, n)),
        ),
    };
    Some(DebugInfo {
        name: Some(df.name.clone()),
        file: df.decl_file.clone().or_else(|| {
            dw.units
                .iter()
                .find(|u| u.functions.iter().any(|g| std::ptr::eq(g, df)))
                .and_then(|u| u.name.clone())
        }),
        return_type: qualify(&df.return_type),
        params: vars(&df.params),
        locals: vars(&df.locals),
        types,
        lines,
        // `DW_AT_frame_base` is normally `DW_OP_call_frame_cfa`. When it names
        // a register instead, the base is that register's slot, which the
        // decompiler resolves after lifting the prologue.
        frame_base_is_cfa,
        frame_base_reg,
    })
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dwarf_registers_map_to_their_architectural_names() {
        use occ_core::Arch;
        assert_eq!(dwarf_reg_name(Arch::X86_64, 5).as_deref(), Some("rdi"));
        assert_eq!(dwarf_reg_name(Arch::X86_64, 6).as_deref(), Some("rbp"));
        assert_eq!(dwarf_reg_name(Arch::Aarch64, 0).as_deref(), Some("x0"));
        assert_eq!(dwarf_reg_name(Arch::Aarch64, 31).as_deref(), Some("sp"));
        assert_eq!(dwarf_reg_name(Arch::RiscV64, 10).as_deref(), Some("a0"));
        assert_eq!(dwarf_reg_name(Arch::RiscV64, 2).as_deref(), Some("sp"));
        // A number with no architectural meaning is not a register.
        assert_eq!(dwarf_reg_name(Arch::X86_64, 200), None);
    }
}
