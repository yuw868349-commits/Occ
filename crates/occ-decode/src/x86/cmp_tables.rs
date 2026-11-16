//! The comparison instructions.
//!
//! Some instructions are named after the condition they test: `vcmpeqps`
//! and `vcmpneqps` are the same opcode with a different immediate byte,
//! and a listing that calls them both `vcmpps` has lost the only thing
//! that tells them apart.
//!
//! There are eight conditions and sixteen families, and the machine has
//! one spelling for each pair. Every name here was read out of a
//! disassembler; the script that does this is `scripts/compare_tables.py`.

/// One family of comparison instructions.
pub struct CmpOp {
    /// The opcode byte.
    pub op: u8,
    /// The opcode map.
    pub map: u8,
    /// The mandatory prefix.
    pub pp: u8,
    /// The operand-width bit.
    pub w: bool,
    /// The name for each condition, selected by the immediate byte.
    pub names: [&'static str; 8],
}

impl CmpOp {
    /// The name for the given condition.
    pub fn name(&self, cond: u8) -> &'static str {
        self.names[usize::from(cond) & 7]
    }
}

/// `const` constructor, so the table stays readable.
pub const fn cmp_op(
    op: u8,
    map: u8,
    pp: u8,
    w: bool,
    names: [&'static str; 8],
) -> CmpOp {
    CmpOp {
        op,
        map,
        pp,
        w,
        names,
    }
}

/// The family for one key, if the table has one.
pub fn lookup(op: u8, map: u8, pp: u8, w: bool) -> Option<&'static CmpOp> {
    CMP_OPS.iter()
        .find(|row| row.op == op && row.map == map && row.pp == pp && row.w == w)
}

include!("cmp_ops.rs");
