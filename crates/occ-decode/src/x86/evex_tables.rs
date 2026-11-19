//! The EVEX opcode table.
//!
//! Every entry was produced by building the encoding and asking an
//! independent decoder what it means, rather than by writing the names down
//! from memory. The script that does this is `scripts/evex_tables.py`.
//!
//! The EVEX encoding names an instruction with more fields than the legacy
//! ones do: the opcode map, the mandatory prefix, the `W` bit and the
//! opcode byte all take part, and the vector length can change the name
//! itself. The table therefore records, for one opcode, a row for every
//! combination of those fields, and each row holds the name for each of the
//! three vector lengths.
//!
//! The rows are kept sorted by the key so that the lookup is a binary
//! search rather than a scan.

use super::vector::{Form, MemSize};

/// One row of the EVEX table.
pub struct EvexOp {
    /// The opcode byte.
    pub op: u8,
    /// The opcode map: 1 is `0f38`, 2 is `0f38` through `f3`, 3 is `0f3a`,
    /// and 5 and 6 are `0f38` through `f3` and `f2`.
    pub map: u8,
    /// The mandatory prefix: 0 is none, 1 is `66`, 2 is `f3`, 3 is `f2`.
    pub pp: u8,
    /// The operand-width bit.
    pub w: bool,
    /// The name for each vector length: 128, 256 and 512 bits.
    pub names: [&'static str; 3],
    /// The shape of the operands.
    pub form: Form,
    /// The size of the memory operand.
    pub mem: MemSize,
    /// A trailing immediate byte.
    pub imm8: bool,
    /// True when the instruction keeps a 128-bit destination at every
    /// vector length.
    ///
    /// The scalar instructions are the case: `vrndscaless` works on one
    /// element and names one lane, so the prefix's `L` field says nothing
    /// about the registers it uses. An encoding that sets it anyway still
    /// names a 128-bit destination, and the listing says `xmm`.
    pub xmm_dest: bool,
    /// The width of one element of the memory operand, in bytes, or zero
    /// when the instruction has no broadcast form.
    ///
    /// The broadcast bit turns the memory operand into one element that
    /// is copied to every lane, and the listing says so with `{1toN}`.
    /// The N is the vector length divided by this width, which is why the
    /// width has to be recorded: the encoding does not name it.
    pub elem: u8,
}

impl EvexOp {
    /// The name for the given vector length.
    pub fn name(&self, ll: u8) -> &'static str {
        self.names[usize::from(ll) & 3]
    }
}

/// `const` constructor, so the table stays readable.
#[allow(clippy::too_many_arguments)]
pub const fn evex_op(
    op: u8,
    map: u8,
    pp: u8,
    w: bool,
    names: [&'static str; 3],
    form: Form,
    mem: MemSize,
    imm8: bool,
    elem: u8,
    xmm_dest: bool,
) -> EvexOp {
    EvexOp {
        op,
        map,
        pp,
        w,
        names,
        form,
        mem,
        imm8,
        elem,
        xmm_dest,
    }
}

/// The key a row is looked up by.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct EvexKey {
    /// The opcode byte.
    pub op: u8,
    /// The opcode map.
    pub map: u8,
    /// The mandatory prefix.
    pub pp: u8,
    /// The operand-width bit.
    pub w: bool,
}

/// The row for one key, if the table has one.
pub fn lookup(op: u8, map: u8, pp: u8, w: bool) -> Option<&'static EvexOp> {
    let key = EvexKey { op, map, pp, w };
    EVEX_OPS
        .binary_search_by(|row| {
            (row.op, row.map, row.pp, row.w).cmp(&(key.op, key.map, key.pp, key.w))
        })
        .ok()
        .map(|i| &EVEX_OPS[i])
}

include!("evex_ops.rs");
