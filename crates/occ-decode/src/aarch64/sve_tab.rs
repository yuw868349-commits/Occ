//! Shared machinery for the generated SVE2 tables.
//!
//! A class is a cube of the encoding space -- a mask and the value the
//! masked bits carry -- plus the operands the family prints.  An operand
//! reads a bit field and looks the value up in a table; `IV_UNUSED` marks
//! a field value the family never prints, which rejects the word so the
//! next class can try.  The classes are ordered most-specific-first.

/// A table entry that no word of the family produces.
pub(super) const IV_UNUSED: i64 = i64::MIN;

/// Operand kinds; the rendering of each is in `iv_operand`.
pub(super) const KZ: u8 = 0;
pub(super) const KZIDX: u8 = 1;
pub(super) const KP: u8 = 2;
pub(super) const KPD: u8 = 3;
pub(super) const KSIMD: u8 = 4;
pub(super) const KSCALAR: u8 = 5;
pub(super) const KIMM: u8 = 6;
pub(super) const KZLIST: u8 = 7;

/// One printed operand: the bit field it reads and what each value of that
/// field prints as.
pub(super) struct IvOp {
    pub kind: u8,
    pub elem: char,
    /// Values per table entry: one, or two for an index or a list.
    pub stride: u8,
    pub bits: &'static [u8],
    pub table: &'static [i64],
}

/// One family: the cube it occupies and the operands it prints.
pub(super) struct IvClass {
    pub mask: u32,
    pub val: u32,
    pub mn: &'static str,
    pub ops: &'static [IvOp],
}

/// The first class that claims the word, with its rendered text.
pub(super) fn iv_scan(
    classes: &'static [IvClass],
    cands: &[u16],
    word: u32,
) -> Option<(&'static str, String)> {
    for &i in cands {
        let c = &classes[i as usize];
        if word & c.mask != c.val {
            continue;
        }
        let mut text = String::new();
        text.push_str(c.mn);
        text.push('\t');
        let mut first = true;
        let mut ok = true;
        for op in c.ops {
            match iv_operand(op, word) {
                Some(t) => {
                    if !first {
                        text.push_str(", ");
                    }
                    first = false;
                    text.push_str(&t);
                }
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            return Some((c.mn, text));
        }
    }
    None
}

/// The printed form of one operand, or `None` when the field carries a
/// value the family never uses.
fn iv_operand(op: &IvOp, word: u32) -> Option<String> {
    let mut key = 0usize;
    for (i, b) in op.bits.iter().enumerate() {
        if (word >> b) & 1 != 0 {
            key |= 1 << i;
        }
    }
    let base = key * op.stride as usize;
    let v0 = *op.table.get(base)?;
    if v0 == IV_UNUSED {
        return None;
    }
    let v1 = if op.stride > 1 { op.table[base + 1] } else { 0 };
    let out = match op.kind {
        KZ => format!("z{}.{}", v0 & 31, op.elem),
        KZIDX => format!("z{}.{}[{}]", v0 & 31, op.elem, v1),
        KP => {
            if op.elem == ' ' {
                format!("p{}", v0 & 15)
            } else {
                format!("p{}/{}", v0 & 15, op.elem)
            }
        }
        KPD => format!("p{}.{}", v0 & 15, op.elem),
        KSIMD => format!("v{}.{}{}", v0 & 31, v1, op.elem),
        KSCALAR => format!("{}{}", op.elem, v0 & 31),
        KIMM => {
            if op.elem == 'x' {
                format!("#0x{:x}", v0)
            } else {
                format!("#{}", v0)
            }
        }
        KZLIST => {
            let n = v1 + 1;
            let lo = v0 & 31;
            let body = if n == 1 {
                format!("z{}.{}", lo, op.elem)
            } else if n == 2 {
                format!("z{}.{}, z{}.{}", lo, op.elem, (lo + 1) & 31, op.elem)
            } else if lo + n <= 32 {
                format!("z{}.{} - z{}.{}", lo, op.elem, (lo + n - 1) & 31, op.elem)
            } else {
                let mut s = String::new();
                for i in 0..n {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    s.push_str(&format!("z{}.{}", (lo + i) & 31, op.elem));
                }
                s
            };
            format!("{{ {} }}", body)
        }
        _ => return None,
    };
    Some(out)
}
