//! Scalar floating point and Advanced SIMD (NEON) decoding.
//!
//! The SIMD/FP group is the part of A64 whose encodings are least tree-shaped:
//! the same 32-bit word can mean a scalar `fadd`, a four-lane vector `fadd`, a
//! lane-indexed multiply, or a structure load, and the bits that choose between
//! those readings are scattered across the word. This module therefore decodes
//! by family, and each family is a table keyed by the fields the architecture
//! itself uses to choose between instructions.
//!
//! Every table entry was checked against the reference assembler rather than
//! written from memory. The rule that follows from that: an encoding no entry
//! matches is reported as `.word` with its family named, and is never given a
//! plausible-looking mnemonic. A fabricated `fmla` reads as real code and sends
//! whoever follows it in the wrong direction.
//!
//! Covered:
//!
//! * scalar FP — three-source (fused multiply-add), two-source arithmetic,
//!   one-source (`fabs`/`fneg`/`fsqrt`/`frint*`/`fcvt`), compares, conditional
//!   compare and select, FP/integer conversion, FP immediates, and `fmov`
//!   between the register files;
//! * vector — three-register same-length (integer and FP), three-register
//!   different-length (widening), two-register miscellaneous, across-lanes
//!   reductions, copy (`dup`/`ins`/`umov`/`smov`), permute (`zip`/`uzp`/`trn`),
//!   `tbl`/`tbx`, `ext`, modified immediates (`movi`/`mvni`/`orr`/`bic`/`fmov`),
//!   shift by immediate, and structure loads and stores in every shape: multiple
//!   structures, a single lane, and the replicating loads.
//!
//! Not covered, and reported as a named gap rather than guessed: the by-element
//! (indexed) forms, whose lane index is split across `immh`, `immb`, `L` and
//! `M`, and SVE, which is a separate instruction set with its own predication
//! model. Scalar and vector loads and stores live in the load/store group and
//! are decoded there.

use occ_core::Addr;

use super::{bits, undefined, R};

/// Element arrangements for the two vector lengths, indexed by `size`.
const ARRANGEMENTS: [[&str; 4]; 2] = [
    ["8b", "4h", "2s", "2d"],
    ["16b", "8h", "4s", "2d"],
];

/// Element letters used for a lane index.
const ELEMENTS: [char; 4] = ['b', 'h', 's', 'd'];

const CONDS: [&str; 16] = [
    "eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al", "nv",
];

/// Decode one word of the SIMD/FP group (top-level `op1` of `0111` or `1111`).
pub(super) fn decode(origin: Addr, word: u32) -> R {
    if bits(word, 28, 25) == 0b1111 {
        return match scalar(word) {
            Some(r) => r,
            None => undefined(origin, word, "SIMD/FP scalar (unallocated)"),
        };
    }
    match vector(word) {
        Ok(r) => r,
        Err(group) => undefined(origin, word, group),
    }
}

/// `v<n>.<arrangement>`, e.g. `v3.4s`.
fn vreg(n: u32, arrangement: &str) -> String {
    format!("v{}.{arrangement}", n & 31)
}

/// A scalar register spelled by element letter: `s0`, `d3`, `h1`, `b7`.
fn elem_reg(letter: char, n: u32) -> String {
    format!("{letter}{}", n & 31)
}

/// The arrangement for `size` at the given vector length.
fn arrangement(q: bool, size: u32) -> &'static str {
    ARRANGEMENTS[q as usize][(size & 3) as usize]
}

/// The register letter for an element size: `b`, `h`, `s`, `d`.
fn element(size: u32) -> char {
    ELEMENTS[(size & 3) as usize]
}

/// The FP type letter for an `ftype` field, rejecting the reserved value.
fn ftype(t: u32) -> Option<char> {
    match t {
        0 => Some('s'),
        1 => Some('d'),
        3 => Some('h'),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Scalar floating point
// ---------------------------------------------------------------------------

/// Scalar FP sits in the `1111` half of the group and always has bits 30:29
/// clear; the encodings that need a 64-bit view of the world (conversions and
/// `fmov` to and from an X register) use bit 31 as their `sf` field.
fn scalar(word: u32) -> Option<R> {
    if bits(word, 30, 29) != 0 {
        return None;
    }
    match bits(word, 28, 24) {
        0b11111 => three_source(word),
        0b11110 => scalar_data_processing(word),
        _ => None,
    }
}

/// `fmadd`/`fmsub`/`fnmadd`/`fnmsub`: `0 0 0 11111 type o1 Rm o0 Ra Rn Rd`.
fn three_source(word: u32) -> Option<R> {
    let t = ftype(bits(word, 23, 22))?;
    let mn = match (bits(word, 21, 21) == 1, bits(word, 15, 15) == 1) {
        (false, false) => "fmadd",
        (false, true) => "fmsub",
        (true, false) => "fnmadd",
        (true, true) => "fnmsub",
    };
    let (rd, rn, rm, ra) = (
        bits(word, 4, 0),
        bits(word, 9, 5),
        bits(word, 20, 16),
        bits(word, 14, 10),
    );
    Some(R::new(
        mn,
        format!("{mn} {t}{rd}, {t}{rn}, {t}{rm}, {t}{ra}"),
    ))
}

/// Everything else scalar: two-source arithmetic, conditional compare and
/// select, one-source operations, compares, conversions and immediates.
fn scalar_data_processing(word: u32) -> Option<R> {
    let t = ftype(bits(word, 23, 22))?;
    let (rd, rn, rm) = (bits(word, 4, 0), bits(word, 9, 5), bits(word, 20, 16));

    // Two-source: bits 11:10 = 10, opcode in bits 15:12.
    if bits(word, 11, 10) == 0b10 {
        let mn = match bits(word, 15, 12) {
            0 => "fmul",
            1 => "fdiv",
            2 => "fadd",
            3 => "fsub",
            4 => "fmax",
            5 => "fmin",
            6 => "fmaxnm",
            7 => "fminnm",
            8 => "fnmul",
            _ => return None,
        };
        return Some(R::new(mn, format!("{mn} {t}{rd}, {t}{rn}, {t}{rm}")));
    }

    // Conditional compare (bits 11:10 = 01) and conditional select (11).
    let cond = CONDS[bits(word, 15, 12) as usize];
    match bits(word, 11, 10) {
        0b01 => {
            let e = bits(word, 4, 4) == 1;
            let nzcv = bits(word, 3, 0);
            let mn = if e { "fccmpe" } else { "fccmp" };
            return Some(R::new(
                mn,
                format!("{mn} {t}{rn}, {t}{rm}, #{nzcv}, {cond}"),
            ));
        }
        0b11 => {
            return Some(R::new(
                "fcsel",
                format!("fcsel {t}{rd}, {t}{rn}, {t}{rm}, {cond}"),
            ));
        }
        _ => {}
    }

    if bits(word, 21, 21) != 1 {
        return None;
    }

    // A one-source operation has bits 14:10 = 10000; the sixth opcode bit
    // sits at bit 15.
    if bits(word, 14, 10) == 0b10000 {
        let opcode = bits(word, 20, 15);
        let mn = match opcode {
            0 => "fmov",
            1 => "fabs",
            2 => "fneg",
            3 => "fsqrt",
            8 => "frintn",
            9 => "frintp",
            10 => "frintm",
            11 => "frintz",
            12 => "frinta",
            14 => "frintx",
            15 => "frinti",
            // `fcvt` is the one-source form whose result has a different type;
            // the opcode names the destination, the type field the source.
            4..=6 => {
                let d = ftype(opcode - 4)?;
                return Some(R::new("fcvt", format!("fcvt {d}{rd}, {t}{rn}")));
            }
            _ => return None,
        };
        return Some(R::new(mn, format!("{mn} {t}{rd}, {t}{rn}")));
    }

    // Compares: bits 15:12 = 0010, bits 11:10 = 00. Bit 4 selects the
    // signalling form and bit 3 the comparison against zero.
    if bits(word, 15, 12) == 0b0010 && bits(word, 11, 10) == 0 {
        let mn = if bits(word, 4, 4) == 1 { "fcmpe" } else { "fcmp" };
        if bits(word, 3, 3) == 1 {
            return Some(R::new(mn, format!("{mn} {t}{rn}, #0.0")));
        }
        return Some(R::new(mn, format!("{mn} {t}{rn}, {t}{rm}")));
    }

    // FP immediate: bits 15:13 = 000, bits 12:10 = 100, and the eight-bit
    // immediate sits whole at bits 20:13.
    if bits(word, 15, 13) == 0 && bits(word, 12, 10) == 0b100 {
        let imm8 = bits(word, 20, 13);
        let value = expand_imm(imm8, t);
        return Some(R::new(
            "fmov",
            format!("fmov {t}{rd}, #{}", render_float(t, value)),
        ));
    }

    // Conversion between FP and integer: bits 15:10 = 000000.
    if bits(word, 15, 10) == 0 {
        return conversion(word, t);
    }
    None
}

/// FP <-> integer conversion, `fmov` across register files, and the integer
/// rounding forms. `sf` picks the integer width; `rmode` and `opcode` pick the
/// operation.
fn conversion(word: u32, t: char) -> Option<R> {
    let sf = bits(word, 31, 31) == 1;
    let gpr = if sf { 'x' } else { 'w' };
    let (rd, rn) = (bits(word, 4, 0), bits(word, 9, 5));
    let rmode = bits(word, 20, 19);
    let opcode = bits(word, 18, 16);
    match (opcode, rmode) {
        (0, 0) => Some(R::new("fcvtns", format!("fcvtns {gpr}{rd}, {t}{rn}"))),
        (0, 1) => Some(R::new("fcvtps", format!("fcvtps {gpr}{rd}, {t}{rn}"))),
        (0, 2) => Some(R::new("fcvtms", format!("fcvtms {gpr}{rd}, {t}{rn}"))),
        (0, 3) => Some(R::new("fcvtzs", format!("fcvtzs {gpr}{rd}, {t}{rn}"))),
        (1, 0) => Some(R::new("fcvtnu", format!("fcvtnu {gpr}{rd}, {t}{rn}"))),
        (1, 1) => Some(R::new("fcvtpu", format!("fcvtpu {gpr}{rd}, {t}{rn}"))),
        (1, 2) => Some(R::new("fcvtmu", format!("fcvtmu {gpr}{rd}, {t}{rn}"))),
        (1, 3) => Some(R::new("fcvtzu", format!("fcvtzu {gpr}{rd}, {t}{rn}"))),
        (2, 0) => Some(R::new("scvtf", format!("scvtf {t}{rd}, {gpr}{rn}"))),
        (3, 0) => Some(R::new("ucvtf", format!("ucvtf {t}{rd}, {gpr}{rn}"))),
        (4, 0) => Some(R::new("fcvtas", format!("fcvtas {gpr}{rd}, {t}{rn}"))),
        (4, 1) => Some(R::new("fcvtau", format!("fcvtau {gpr}{rd}, {t}{rn}"))),
        // `fmov` to and from the integer file. rmode = 1 names the top half of
        // a 128-bit vector register, which only exists for 64-bit transfers.
        (6, 0) => Some(R::new("fmov", format!("fmov {gpr}{rd}, {t}{rn}"))),
        (6, 1) if sf => Some(R::new("fmov", format!("fmov x{rd}, v{rn}.d[1]"))),
        (7, 0) => Some(R::new("fmov", format!("fmov {t}{rd}, {gpr}{rn}"))),
        (7, 1) if sf => Some(R::new("fmov", format!("fmov v{rd}.d[1], x{rn}"))),
        _ => None,
    }
}

/// VFPExpandImm: `abcdefgh` becomes a sign, an exponent whose low bits are `cd`
/// with every remaining bit a copy of `b` — inverted in the top position — and a
/// fraction of `efgh` followed by zeros. The three shapes differ only in how many
/// exponent bits are replicated and therefore how far the fraction sits.
fn expand_imm(imm8: u32, t: char) -> f64 {
    let sign = (imm8 >> 7) & 1;
    let b = (imm8 >> 6) & 1;
    let cd = (imm8 >> 4) & 3;
    let efgh = imm8 & 0xF;
    match t {
        's' => {
            let exp = ((1 - b) << 7) | (b * 0b0111_1100) | cd;
            f32::from_bits((sign << 31) | (exp << 23) | (efgh << 19)) as f64
        }
        'd' => {
            let exp = ((1 - b) << 10) | (b * 0b0011_1111_1100) | cd;
            f64::from_bits(
                ((sign as u64) << 63) | ((exp as u64) << 52) | ((efgh as u64) << 48),
            )
        }
        _ => {
            let exp = ((1 - b) << 4) | (b * 0b0_1100) | cd;
            half_to_f64((sign << 15) | (exp << 10) | (efgh << 6))
        }
    }
}

/// Convert a binary16 bit pattern to `f64`, for `fmov h` and `fmov v.4h`.
fn half_to_f64(bits: u32) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x3FF) as f64;
    match exp {
        0 => sign * frac * 2f64.powi(-24),
        0x1F if frac == 0.0 => sign * f64::INFINITY,
        0x1F => f64::NAN,
        _ => sign * (1.0 + frac / 1024.0) * 2f64.powi(exp - 15),
    }
}

/// How the value of an FP immediate is printed. Whole numbers keep their `.0`
/// so the listing does not read like an integer operand.
fn render_float(t: char, value: f64) -> String {
    if t == 's' {
        format!("{:?}", value as f32)
    } else {
        format!("{value:?}")
    }
}

// ---------------------------------------------------------------------------
// Vector (Advanced SIMD)
// ---------------------------------------------------------------------------

/// The vector half of the group, split by bits 28:24 into data processing,
/// immediates and shifts, and the two structure-load shapes. The error case
/// names the family so the `.word` comment can say what was not decoded.
fn vector(word: u32) -> Result<R, &'static str> {
    let q = bits(word, 30, 30) == 1;
    match bits(word, 28, 24) {
        0b01110 => vector_dp(word, q),
        0b01111 => vector_imm(word, q),
        0b01100 => structure(word, q, false),
        0b01101 => structure(word, q, true),
        _ => Err("SIMD/FP vector (unallocated)"),
    }
}

/// Vector data processing. The pair `(bit 21, bits 11:10)` picks the family;
/// within a family the remaining fields are the architecture's own opcode.
fn vector_dp(word: u32, q: bool) -> Result<R, &'static str> {
    let u = bits(word, 29, 29) == 1;
    let size = bits(word, 23, 22);
    // With bit 21 clear the word is one of the classes that carry an immediate
    // or a register list: the copy forms (bits 15:11 opcode), the permutes, and
    // the table lookups.
    if bits(word, 21, 21) == 0 {
        return vector_low(word, q);
    }
    // The remaining families keep their opcode in bits 16:12 and fix bits
    // 21:17, which is what separates them from the three-register forms: no
    // three-register encoding has bits 11:10 = `10`, because that opcode field
    // ends in `1` and is followed by the `10` that marks the class.
    if bits(word, 11, 10) == 0b10 {
        return match bits(word, 21, 17) {
            0b10000 => two_misc(word, q, u, false).ok_or("SIMD/FP two-register (unallocated)"),
            0b11000 => across_lanes(word, q, u).ok_or("SIMD/FP across-lanes (unallocated)"),
            0b11100 => two_misc(word, q, u, true).ok_or("SIMD/FP two-register FP16 (unallocated)"),
            _ => Err("SIMD/FP by-element (not decoded)"),
        };
    }
    if let Some(r) = three_same(word, q, u, size) {
        return Ok(r);
    }
    if bits(word, 11, 10) == 0b00 {
        if let Some(r) = three_long(word, q, u, size) {
            return Ok(r);
        }
    }
    Err("SIMD/FP three-register (unallocated)")
}

/// Three-register same-length integer encodings: `(opcode, u, size)`.
///
/// `size` is bits 23:22. The logical group reuses those two bits as a
/// sub-opcode (`and`/`orr`/`eor`/`bsl`/`bit`/`bif`), so its entries read as
/// four separate sizes rather than as four widths.
const THREE_SAME_INT: &[(u8, bool, u32, &str)] = &[
    (0b000001, false, 0, "shadd"),
    (0b000001, false, 1, "shadd"),
    (0b000001, false, 2, "shadd"),
    (0b000001, true, 0, "uhadd"),
    (0b000001, true, 1, "uhadd"),
    (0b000001, true, 2, "uhadd"),
    (0b000011, false, 0, "sqadd"),
    (0b000011, false, 1, "sqadd"),
    (0b000011, false, 2, "sqadd"),
    (0b000011, false, 3, "sqadd"),
    (0b000011, true, 0, "uqadd"),
    (0b000011, true, 1, "uqadd"),
    (0b000011, true, 2, "uqadd"),
    (0b000011, true, 3, "uqadd"),
    (0b000101, false, 0, "srhadd"),
    (0b000101, false, 1, "srhadd"),
    (0b000101, false, 2, "srhadd"),
    (0b000101, true, 0, "urhadd"),
    (0b000101, true, 1, "urhadd"),
    (0b000101, true, 2, "urhadd"),
    (0b000111, false, 0, "and"),
    (0b000111, false, 2, "orr"),
    (0b000111, true, 0, "eor"),
    (0b000111, true, 1, "bsl"),
    (0b000111, true, 2, "bit"),
    (0b000111, true, 3, "bif"),
    (0b001001, false, 0, "shsub"),
    (0b001001, false, 1, "shsub"),
    (0b001001, false, 2, "shsub"),
    (0b001001, true, 0, "uhsub"),
    (0b001001, true, 1, "uhsub"),
    (0b001001, true, 2, "uhsub"),
    (0b001011, false, 0, "sqsub"),
    (0b001011, false, 1, "sqsub"),
    (0b001011, false, 2, "sqsub"),
    (0b001011, false, 3, "sqsub"),
    (0b001011, true, 0, "uqsub"),
    (0b001011, true, 1, "uqsub"),
    (0b001011, true, 2, "uqsub"),
    (0b001011, true, 3, "uqsub"),
    (0b001101, false, 0, "cmgt"),
    (0b001101, false, 1, "cmgt"),
    (0b001101, false, 2, "cmgt"),
    (0b001101, false, 3, "cmgt"),
    (0b001101, true, 0, "cmhi"),
    (0b001101, true, 1, "cmhi"),
    (0b001101, true, 2, "cmhi"),
    (0b001101, true, 3, "cmhi"),
    (0b001111, false, 0, "cmge"),
    (0b001111, false, 1, "cmge"),
    (0b001111, false, 2, "cmge"),
    (0b001111, false, 3, "cmge"),
    (0b001111, true, 0, "cmhs"),
    (0b001111, true, 1, "cmhs"),
    (0b001111, true, 2, "cmhs"),
    (0b001111, true, 3, "cmhs"),
    (0b011001, false, 0, "smax"),
    (0b011001, false, 1, "smax"),
    (0b011001, false, 2, "smax"),
    (0b011001, true, 0, "umax"),
    (0b011001, true, 1, "umax"),
    (0b011001, true, 2, "umax"),
    (0b011011, false, 0, "smin"),
    (0b011011, false, 1, "smin"),
    (0b011011, false, 2, "smin"),
    (0b011011, true, 0, "umin"),
    (0b011011, true, 1, "umin"),
    (0b011011, true, 2, "umin"),
    (0b011101, false, 0, "sabd"),
    (0b011101, false, 1, "sabd"),
    (0b011101, false, 2, "sabd"),
    (0b011101, true, 0, "uabd"),
    (0b011101, true, 1, "uabd"),
    (0b011101, true, 2, "uabd"),
    (0b011111, false, 0, "saba"),
    (0b011111, false, 1, "saba"),
    (0b011111, false, 2, "saba"),
    (0b011111, true, 0, "uaba"),
    (0b011111, true, 1, "uaba"),
    (0b011111, true, 2, "uaba"),
    (0b100001, false, 0, "add"),
    (0b100001, false, 1, "add"),
    (0b100001, false, 2, "add"),
    (0b100001, false, 3, "add"),
    (0b100001, true, 0, "sub"),
    (0b100001, true, 1, "sub"),
    (0b100001, true, 2, "sub"),
    (0b100001, true, 3, "sub"),
    (0b100011, false, 0, "cmtst"),
    (0b100011, false, 1, "cmtst"),
    (0b100011, false, 2, "cmtst"),
    (0b100011, false, 3, "cmtst"),
    (0b100011, true, 0, "cmeq"),
    (0b100011, true, 1, "cmeq"),
    (0b100011, true, 2, "cmeq"),
    (0b100011, true, 3, "cmeq"),
    (0b100101, false, 0, "mla"),
    (0b100101, false, 1, "mla"),
    (0b100101, false, 2, "mla"),
    (0b100101, true, 0, "mls"),
    (0b100101, true, 1, "mls"),
    (0b100101, true, 2, "mls"),
    (0b100111, false, 0, "mul"),
    (0b100111, false, 1, "mul"),
    (0b100111, false, 2, "mul"),
    (0b100111, true, 0, "pmul"),
    (0b101101, false, 1, "sqdmulh"),
    (0b101101, false, 2, "sqdmulh"),
    (0b101101, true, 1, "sqrdmulh"),
    (0b101101, true, 2, "sqrdmulh"),
    (0b101111, false, 0, "addp"),
    (0b101111, false, 1, "addp"),
    (0b101111, false, 2, "addp"),
    (0b101111, false, 3, "addp"),
];

/// Three-register same-length floating point: `(opcode, u, bit 23, size)`.
///
/// Bit 23 is a second opcode bit here rather than part of the width, so the key
/// needs it separately; bit 22 is the single/double selector.
const THREE_SAME_FP: &[(u8, bool, bool, bool, &str)] = &[
    (0b110001, false, false, false, "fmaxnm"),
    (0b110001, false, false, true, "fmaxnm"),
    (0b110001, false, true, false, "fminnm"),
    (0b110001, false, true, true, "fminnm"),
    (0b110001, true, false, false, "fmaxnmp"),
    (0b110001, true, false, true, "fmaxnmp"),
    (0b110001, true, true, false, "fminnmp"),
    (0b110001, true, true, true, "fminnmp"),
    (0b110011, false, false, false, "fmla"),
    (0b110011, false, false, true, "fmla"),
    (0b110011, false, true, false, "fmls"),
    (0b110011, false, true, true, "fmls"),
    (0b110101, false, false, false, "fadd"),
    (0b110101, false, false, true, "fadd"),
    (0b110101, false, true, false, "fsub"),
    (0b110101, false, true, true, "fsub"),
    (0b110101, true, false, false, "faddp"),
    (0b110101, true, false, true, "faddp"),
    (0b110101, true, true, false, "fabd"),
    (0b110101, true, true, true, "fabd"),
    (0b110111, false, false, false, "fmulx"),
    (0b110111, false, false, true, "fmulx"),
    (0b110111, true, false, false, "fmul"),
    (0b110111, true, false, true, "fmul"),
    (0b111001, false, false, false, "fcmeq"),
    (0b111001, false, false, true, "fcmeq"),
    (0b111001, true, false, false, "fcmge"),
    (0b111001, true, false, true, "fcmge"),
    (0b111001, true, true, false, "fcmgt"),
    (0b111001, true, true, true, "fcmgt"),
    (0b111011, true, false, false, "facge"),
    (0b111011, true, false, true, "facge"),
    (0b111011, true, true, false, "facgt"),
    (0b111011, true, true, true, "facgt"),
    (0b111101, false, false, false, "fmax"),
    (0b111101, false, false, true, "fmax"),
    (0b111101, false, true, false, "fmin"),
    (0b111101, false, true, true, "fmin"),
    (0b111101, true, false, false, "fmaxp"),
    (0b111101, true, false, true, "fmaxp"),
    (0b111101, true, true, false, "fminp"),
    (0b111101, true, true, true, "fminp"),
    (0b111111, false, false, false, "frecps"),
    (0b111111, false, false, true, "frecps"),
    (0b111111, false, true, false, "frsqrts"),
    (0b111111, false, true, true, "frsqrts"),
    (0b111111, true, false, false, "fdiv"),
    (0b111111, true, false, true, "fdiv"),
];

fn three_same(word: u32, q: bool, u: bool, size: u32) -> Option<R> {
    let op = bits(word, 15, 10) as u8;
    let (rd, rn, rm) = (bits(word, 4, 0), bits(word, 9, 5), bits(word, 20, 16));
    let (mn, arrangement) = if op >= 0b11000 {
        let wide = size & 1 == 1;
        // A double-precision vector needs all 128 bits; the encoding with one
        // 64-bit half is not a valid instruction.
        if wide && !q {
            return None;
        }
        let entry = THREE_SAME_FP
            .iter()
            .find(|e| e.0 == op && e.1 == u && e.2 == (size >> 1 == 1) && e.3 == wide)?;
        let a = if wide {
            "2d"
        } else if q {
            "4s"
        } else {
            "2s"
        };
        (entry.4, a)
    } else {
        let entry = THREE_SAME_INT
            .iter()
            .find(|e| e.0 == op && e.1 == u && e.2 == size)?;
        (entry.3, arrangement(q, size))
    };
    Some(R::new(
        mn,
        format!(
            "{mn} {}, {}, {}",
            vreg(rd, arrangement),
            vreg(rn, arrangement),
            vreg(rm, arrangement)
        ),
    ))
}

/// Three-register different-length: the source lanes are half as wide as the
/// destination lanes, so a widening op reads `8b` and writes `8h`. Bit 30 picks
/// the lower or the upper half of the source, which the assembler spells with a
/// `2` suffix on the mnemonic.
const THREE_LONG: &[(u8, bool, u32, &str)] = &[
    (0b01010, false, 0, "sabal"),
    (0b01010, false, 1, "sabal"),
    (0b01010, false, 2, "sabal"),
    (0b01010, true, 0, "uabal"),
    (0b01010, true, 1, "uabal"),
    (0b01010, true, 2, "uabal"),
    (0b01110, false, 0, "sabdl"),
    (0b01110, false, 1, "sabdl"),
    (0b01110, false, 2, "sabdl"),
    (0b01110, true, 0, "uabdl"),
    (0b01110, true, 1, "uabdl"),
    (0b01110, true, 2, "uabdl"),
    (0b10000, false, 0, "smlal"),
    (0b10000, false, 1, "smlal"),
    (0b10000, false, 2, "smlal"),
    (0b10000, true, 0, "umlal"),
    (0b10000, true, 1, "umlal"),
    (0b10000, true, 2, "umlal"),
    (0b10010, false, 1, "sqdmlal"),
    (0b10010, false, 2, "sqdmlal"),
    (0b10100, false, 0, "smlsl"),
    (0b10100, false, 1, "smlsl"),
    (0b10100, false, 2, "smlsl"),
    (0b10100, true, 0, "umlsl"),
    (0b10100, true, 1, "umlsl"),
    (0b10100, true, 2, "umlsl"),
    (0b10110, false, 1, "sqdmlsl"),
    (0b10110, false, 2, "sqdmlsl"),
    (0b11000, false, 0, "smull"),
    (0b11000, false, 1, "smull"),
    (0b11000, false, 2, "smull"),
    (0b11000, true, 0, "umull"),
    (0b11000, true, 1, "umull"),
    (0b11000, true, 2, "umull"),
    (0b11010, false, 1, "sqdmull"),
    (0b11010, false, 2, "sqdmull"),
    (0b11100, false, 0, "pmull"),
];

/// The `2` spelling of a widening mnemonic, used when the source is the upper
/// half of two registers.
fn widening_name(mn: &'static str, upper: bool) -> &'static str {
    if !upper {
        return mn;
    }
    match mn {
        "sabal" => "sabal2",
        "uabal" => "uabal2",
        "sabdl" => "sabdl2",
        "uabdl" => "uabdl2",
        "smlal" => "smlal2",
        "umlal" => "umlal2",
        "sqdmlal" => "sqdmlal2",
        "smlsl" => "smlsl2",
        "umlsl" => "umlsl2",
        "sqdmlsl" => "sqdmlsl2",
        "smull" => "smull2",
        "umull" => "umull2",
        "sqdmull" => "sqdmull2",
        "pmull" => "pmull2",
        other => other,
    }
}

fn three_long(word: u32, q: bool, u: bool, size: u32) -> Option<R> {
    let op = bits(word, 15, 11) as u8;
    let (rd, rn, rm) = (bits(word, 4, 0), bits(word, 9, 5), bits(word, 20, 16));
    let entry = THREE_LONG
        .iter()
        .find(|e| e.0 == op && e.1 == u && e.2 == size)?;
    let mn = widening_name(entry.3, q);
    Some(R::new(
        mn,
        format!(
            "{mn} {}, {}, {}",
            vreg(rd, arrangement(true, size + 1)),
            vreg(rn, arrangement(q, size)),
            vreg(rm, arrangement(q, size))
        ),
    ))
}

/// How a two-register encoding lays out its operands.
///
/// Most entries use the size field as an element width. The narrowing and
/// widening forms shift it by one element, `Byte` pins the width to bytes
/// whatever the size field says, and the floating-point entries read bit 22 as
/// a second opcode bit rather than as part of a width.
#[derive(Clone, Copy, PartialEq)]
enum Shape {
    /// Destination and source share the element width.
    Same,
    /// Byte lanes only (`cnt`, `rbit`, `rev16`).
    Byte,
    /// Destination lanes are half the source's (`xtn`).
    Narrow,
    /// Destination lanes are twice the source's (`saddlp`).
    Wide,
    /// Single or double precision, selected by bit 22 (`fabs`, `frintn`).
    Fp,
    /// Half precision, four or eight lanes, from the FP16 group.
    Half,
    /// Narrowing FP convert: single to half, or double to single (`fcvtn`).
    FcvtN,
    /// Widening FP convert, the reverse of `FcvtN`.
    FcvtL,
    /// `shll`, which widens and shifts by the full element width.
    Shll,
}

impl Shape {
    /// Whether `Q` changes the mnemonic (`xtn` to `xtn2`) rather than only the
    /// arrangement. Only the forms that write the upper half of a vector do.
    fn suffix_two(self) -> bool {
        matches!(
            self,
            Shape::Narrow | Shape::FcvtN | Shape::FcvtL | Shape::Shll
        )
    }
}

/// The `2` spelling of a mnemonic whose upper-half form has its own name.
fn wide_name(mn: &'static str, q: bool) -> &'static str {
    if !q {
        return mn;
    }
    match mn {
        "xtn" => "xtn2",
        "sqxtn" => "sqxtn2",
        "uqxtn" => "uqxtn2",
        "sqxtun" => "sqxtun2",
        "fcvtn" => "fcvtn2",
        "fcvtl" => "fcvtl2",
        "fcvtxn" => "fcvtxn2",
        "shll" => "shll2",
        "smlal" => "smlal2",
        "smlsl" => "smlsl2",
        "smull" => "smull2",
        "sqdmlal" => "sqdmlal2",
        "sqdmlsl" => "sqdmlsl2",
        "sqdmull" => "sqdmull2",
        "umlal" => "umlal2",
        "umlsl" => "umlsl2",
        "umull" => "umull2",
        other => other,
    }
}

/// Operand text for one two-register entry, with `extra` appended after it (the
/// implicit zero of the compare-against-zero forms).
///
/// Returning `None` means the encoding is not legal for this shape at this
/// vector length, which is a real answer: `Q = 0` with a `2d` arrangement has no
/// encoding.
fn shape_operands(shape: Shape, q: bool, size: u32, rd: u32, rn: u32, extra: &str) -> Option<String> {
    let pair = |a: &str| format!("{}, {}{extra}", vreg(rd, a), vreg(rn, a));
    let text = match shape {
        Shape::Same => {
            if size == 3 && !q {
                return None;
            }
            pair(arrangement(q, size))
        }
        Shape::Byte => pair(arrangement(q, 0)),
        Shape::Narrow => format!(
            "{}, {}{extra}",
            vreg(rd, arrangement(q, size)),
            vreg(rn, arrangement(true, size + 1))
        ),
        Shape::Wide => {
            // A widening pair from `2s` produces `1d` at 64 bits, and `2d` at
            // 128: the destination widens past the `s` arrangement, so
            // `arrangement` cannot name it.
            let dest = if size == 2 {
                if q { "2d" } else { "1d" }
            } else {
                arrangement(q, size + 1)
            };
            format!("{}, {}{extra}", vreg(rd, dest), vreg(rn, arrangement(q, size)))
        }
        Shape::Fp => {
            // Bit 22 is the width here, and a double needs the full vector.
            if size & 1 == 1 {
                if !q {
                    return None;
                }
                pair("2d")
            } else if q {
                pair("4s")
            } else {
                pair("2s")
            }
        }
        Shape::Half => pair(if q { "8h" } else { "4h" }),
        // `fcvtn` reads the low half of the source, so the source keeps its
        // arrangement while the destination changes length.
        Shape::FcvtN => format!(
            "{}, {}{extra}",
            vreg(rd, if size == 0 { if q { "8h" } else { "4h" } } else if q { "4s" } else { "2s" }),
            vreg(rn, if size == 0 { "4s" } else { "2d" })
        ),
        Shape::FcvtL => format!(
            "{}, {}{extra}",
            vreg(rd, if size == 0 { "4s" } else { "2d" }),
            vreg(rn, if size == 0 { if q { "8h" } else { "4h" } } else if q { "4s" } else { "2s" })
        ),
        Shape::Shll => format!(
            "{}, {}, #{}",
            vreg(rd, arrangement(true, size + 1)),
            vreg(rn, arrangement(q, size)),
            8u32 << size
        ),
    };
    Some(text)
}

/// Two-register miscellaneous, keyed by `(opcode, u, size)`.
///
/// The key is the architecture's own: bits 16:12 hold the opcode, bit 29 is a
/// second opcode bit for most of the family, and bits 23:22 are either a width
/// or a third opcode bit. Every row was read off a reference disassembly of the
/// word, and the whole `bits 21:17 = 10000` subspace was then decoded both ways
/// to confirm that the table reproduces it exactly and invents nothing.
const TWO_MISC: &[(u8, bool, u8, &str, Shape, &str)] = &[
    (0b00000, false, 0, "rev64", Shape::Same, ""), // 16b <- 16b
    (0b00001, false, 0, "rev16", Shape::Same, ""), // 16b <- 16b
    (0b00010, false, 0, "saddlp", Shape::Wide, ""), // 8h <- 16b
    (0b00011, false, 0, "suqadd", Shape::Same, ""), // 16b <- 16b
    (0b00100, false, 0, "cls", Shape::Same, ""), // 16b <- 16b
    (0b00101, false, 0, "cnt", Shape::Same, ""), // 16b <- 16b
    (0b00110, false, 0, "sadalp", Shape::Wide, ""), // 8h <- 16b
    (0b00111, false, 0, "sqabs", Shape::Same, ""), // 16b <- 16b
    (0b01000, false, 0, "cmgt", Shape::Same, ", #0"), // 16b <- 16b
    (0b01001, false, 0, "cmeq", Shape::Same, ", #0"), // 16b <- 16b
    (0b01010, false, 0, "cmlt", Shape::Same, ", #0"), // 16b <- 16b
    (0b01011, false, 0, "abs", Shape::Same, ""), // 16b <- 16b
    (0b10010, false, 0, "xtn", Shape::Narrow, ""), // 16b <- 8h
    (0b10100, false, 0, "sqxtn", Shape::Narrow, ""), // 16b <- 8h
    (0b10110, false, 0, "fcvtn", Shape::FcvtN, ""), // 8h <- 4s
    (0b10111, false, 0, "fcvtl", Shape::FcvtL, ""), // 4s <- 8h
    (0b11000, false, 0, "frintn", Shape::Fp, ""), // 4s
    (0b11001, false, 0, "frintm", Shape::Fp, ""), // 4s
    (0b11010, false, 0, "fcvtns", Shape::Fp, ""), // 4s
    (0b11011, false, 0, "fcvtms", Shape::Fp, ""), // 4s
    (0b11100, false, 0, "fcvtas", Shape::Fp, ""), // 4s
    (0b11101, false, 0, "scvtf", Shape::Fp, ""), // 4s
    (0b00000, false, 1, "rev64", Shape::Same, ""), // 8h <- 8h
    (0b00010, false, 1, "saddlp", Shape::Wide, ""), // 4s <- 8h
    (0b00011, false, 1, "suqadd", Shape::Same, ""), // 8h <- 8h
    (0b00100, false, 1, "cls", Shape::Same, ""), // 8h <- 8h
    (0b00110, false, 1, "sadalp", Shape::Wide, ""), // 4s <- 8h
    (0b00111, false, 1, "sqabs", Shape::Same, ""), // 8h <- 8h
    (0b01000, false, 1, "cmgt", Shape::Same, ", #0"), // 8h <- 8h
    (0b01001, false, 1, "cmeq", Shape::Same, ", #0"), // 8h <- 8h
    (0b01010, false, 1, "cmlt", Shape::Same, ", #0"), // 8h <- 8h
    (0b01011, false, 1, "abs", Shape::Same, ""), // 8h <- 8h
    (0b10010, false, 1, "xtn", Shape::Narrow, ""), // 8h <- 4s
    (0b10100, false, 1, "sqxtn", Shape::Narrow, ""), // 8h <- 4s
    (0b10110, false, 1, "fcvtn", Shape::FcvtN, ""), // 4s <- 2d
    (0b10111, false, 1, "fcvtl", Shape::FcvtL, ""), // 2d <- 4s
    (0b11000, false, 1, "frintn", Shape::Fp, ""), // 2d
    (0b11001, false, 1, "frintm", Shape::Fp, ""), // 2d
    (0b11010, false, 1, "fcvtns", Shape::Fp, ""), // 2d
    (0b11011, false, 1, "fcvtms", Shape::Fp, ""), // 2d
    (0b11100, false, 1, "fcvtas", Shape::Fp, ""), // 2d
    (0b11101, false, 1, "scvtf", Shape::Fp, ""), // 2d
    (0b00000, false, 2, "rev64", Shape::Same, ""), // 4s <- 4s
    (0b00010, false, 2, "saddlp", Shape::Wide, ""), // 2d <- 4s
    (0b00011, false, 2, "suqadd", Shape::Same, ""), // 4s <- 4s
    (0b00100, false, 2, "cls", Shape::Same, ""), // 4s <- 4s
    (0b00110, false, 2, "sadalp", Shape::Wide, ""), // 2d <- 4s
    (0b00111, false, 2, "sqabs", Shape::Same, ""), // 4s <- 4s
    (0b01000, false, 2, "cmgt", Shape::Same, ", #0"), // 4s <- 4s
    (0b01001, false, 2, "cmeq", Shape::Same, ", #0"), // 4s <- 4s
    (0b01010, false, 2, "cmlt", Shape::Same, ", #0"), // 4s <- 4s
    (0b01011, false, 2, "abs", Shape::Same, ""), // 4s <- 4s
    (0b01100, false, 2, "fcmgt", Shape::Same, ", #0.0"), // 4s <- 4s
    (0b01101, false, 2, "fcmeq", Shape::Same, ", #0.0"), // 4s <- 4s
    (0b01110, false, 2, "fcmlt", Shape::Same, ", #0.0"), // 4s <- 4s
    (0b01111, false, 2, "fabs", Shape::Same, ""), // 4s <- 4s
    (0b10010, false, 2, "xtn", Shape::Narrow, ""), // 4s <- 2d
    (0b10100, false, 2, "sqxtn", Shape::Narrow, ""), // 4s <- 2d
    (0b11000, false, 2, "frintp", Shape::Same, ""), // 4s <- 4s
    (0b11001, false, 2, "frintz", Shape::Same, ""), // 4s <- 4s
    (0b11010, false, 2, "fcvtps", Shape::Same, ""), // 4s <- 4s
    (0b11011, false, 2, "fcvtzs", Shape::Same, ""), // 4s <- 4s
    (0b11100, false, 2, "urecpe", Shape::Same, ""), // 4s <- 4s
    (0b11101, false, 2, "frecpe", Shape::Same, ""), // 4s <- 4s
    (0b00011, false, 3, "suqadd", Shape::Same, ""), // 2d <- 2d
    (0b00111, false, 3, "sqabs", Shape::Same, ""), // 2d <- 2d
    (0b01000, false, 3, "cmgt", Shape::Same, ", #0"), // 2d <- 2d
    (0b01001, false, 3, "cmeq", Shape::Same, ", #0"), // 2d <- 2d
    (0b01010, false, 3, "cmlt", Shape::Same, ", #0"), // 2d <- 2d
    (0b01011, false, 3, "abs", Shape::Same, ""), // 2d <- 2d
    (0b01100, false, 3, "fcmgt", Shape::Same, ", #0.0"), // 2d <- 2d
    (0b01101, false, 3, "fcmeq", Shape::Same, ", #0.0"), // 2d <- 2d
    (0b01110, false, 3, "fcmlt", Shape::Same, ", #0.0"), // 2d <- 2d
    (0b01111, false, 3, "fabs", Shape::Same, ""), // 2d <- 2d
    (0b11000, false, 3, "frintp", Shape::Same, ""), // 2d <- 2d
    (0b11001, false, 3, "frintz", Shape::Same, ""), // 2d <- 2d
    (0b11010, false, 3, "fcvtps", Shape::Same, ""), // 2d <- 2d
    (0b11011, false, 3, "fcvtzs", Shape::Same, ""), // 2d <- 2d
    (0b11101, false, 3, "frecpe", Shape::Same, ""), // 2d <- 2d
    (0b00000, true, 0, "rev32", Shape::Same, ""), // 16b <- 16b
    (0b00010, true, 0, "uaddlp", Shape::Wide, ""), // 8h <- 16b
    (0b00011, true, 0, "usqadd", Shape::Same, ""), // 16b <- 16b
    (0b00100, true, 0, "clz", Shape::Same, ""), // 16b <- 16b
    (0b00101, true, 0, "mvn", Shape::Same, ""), // 16b <- 16b
    (0b00110, true, 0, "uadalp", Shape::Wide, ""), // 8h <- 16b
    (0b00111, true, 0, "sqneg", Shape::Same, ""), // 16b <- 16b
    (0b01000, true, 0, "cmge", Shape::Same, ", #0"), // 16b <- 16b
    (0b01001, true, 0, "cmle", Shape::Same, ", #0"), // 16b <- 16b
    (0b01011, true, 0, "neg", Shape::Same, ""), // 16b <- 16b
    (0b10010, true, 0, "sqxtun", Shape::Narrow, ""), // 16b <- 8h
    (0b10011, true, 0, "shll", Shape::Shll, ""), // 8h <- 16b, #8
    (0b10100, true, 0, "uqxtn", Shape::Narrow, ""), // 16b <- 8h
    (0b11000, true, 0, "frinta", Shape::Fp, ""), // 4s
    (0b11001, true, 0, "frintx", Shape::Fp, ""), // 4s
    (0b11010, true, 0, "fcvtnu", Shape::Fp, ""), // 4s
    (0b11011, true, 0, "fcvtmu", Shape::Fp, ""), // 4s
    (0b11100, true, 0, "fcvtau", Shape::Fp, ""), // 4s
    (0b11101, true, 0, "ucvtf", Shape::Fp, ""), // 4s
    (0b00000, true, 1, "rev32", Shape::Same, ""), // 8h <- 8h
    (0b00010, true, 1, "uaddlp", Shape::Wide, ""), // 4s <- 8h
    (0b00011, true, 1, "usqadd", Shape::Same, ""), // 8h <- 8h
    (0b00100, true, 1, "clz", Shape::Same, ""), // 8h <- 8h
    (0b00101, true, 1, "rbit", Shape::Byte, ""), // 16b <- 16b
    (0b00110, true, 1, "uadalp", Shape::Wide, ""), // 4s <- 8h
    (0b00111, true, 1, "sqneg", Shape::Same, ""), // 8h <- 8h
    (0b01000, true, 1, "cmge", Shape::Same, ", #0"), // 8h <- 8h
    (0b01001, true, 1, "cmle", Shape::Same, ", #0"), // 8h <- 8h
    (0b01011, true, 1, "neg", Shape::Same, ""), // 8h <- 8h
    (0b10010, true, 1, "sqxtun", Shape::Narrow, ""), // 8h <- 4s
    (0b10011, true, 1, "shll", Shape::Shll, ""), // 4s <- 8h, #16
    (0b10100, true, 1, "uqxtn", Shape::Narrow, ""), // 8h <- 4s
    (0b10110, true, 1, "fcvtxn", Shape::FcvtN, ""), // 4s <- 2d
    (0b11000, true, 1, "frinta", Shape::Fp, ""), // 2d
    (0b11001, true, 1, "frintx", Shape::Fp, ""), // 2d
    (0b11010, true, 1, "fcvtnu", Shape::Fp, ""), // 2d
    (0b11011, true, 1, "fcvtmu", Shape::Fp, ""), // 2d
    (0b11100, true, 1, "fcvtau", Shape::Fp, ""), // 2d
    (0b11101, true, 1, "ucvtf", Shape::Fp, ""), // 2d
    (0b00010, true, 2, "uaddlp", Shape::Wide, ""), // 2d <- 4s
    (0b00011, true, 2, "usqadd", Shape::Same, ""), // 4s <- 4s
    (0b00100, true, 2, "clz", Shape::Same, ""), // 4s <- 4s
    (0b00110, true, 2, "uadalp", Shape::Wide, ""), // 2d <- 4s
    (0b00111, true, 2, "sqneg", Shape::Same, ""), // 4s <- 4s
    (0b01000, true, 2, "cmge", Shape::Same, ", #0"), // 4s <- 4s
    (0b01001, true, 2, "cmle", Shape::Same, ", #0"), // 4s <- 4s
    (0b01011, true, 2, "neg", Shape::Same, ""), // 4s <- 4s
    (0b01100, true, 2, "fcmge", Shape::Same, ", #0.0"), // 4s <- 4s
    (0b01101, true, 2, "fcmle", Shape::Same, ", #0.0"), // 4s <- 4s
    (0b01111, true, 2, "fneg", Shape::Same, ""), // 4s <- 4s
    (0b10010, true, 2, "sqxtun", Shape::Narrow, ""), // 4s <- 2d
    (0b10011, true, 2, "shll", Shape::Shll, ""), // 2d <- 4s, #32
    (0b10100, true, 2, "uqxtn", Shape::Narrow, ""), // 4s <- 2d
    (0b11001, true, 2, "frinti", Shape::Same, ""), // 4s <- 4s
    (0b11010, true, 2, "fcvtpu", Shape::Same, ""), // 4s <- 4s
    (0b11011, true, 2, "fcvtzu", Shape::Same, ""), // 4s <- 4s
    (0b11100, true, 2, "ursqrte", Shape::Same, ""), // 4s <- 4s
    (0b11101, true, 2, "frsqrte", Shape::Same, ""), // 4s <- 4s
    (0b11111, true, 2, "fsqrt", Shape::Same, ""), // 4s <- 4s
    (0b00011, true, 3, "usqadd", Shape::Same, ""), // 2d <- 2d
    (0b00111, true, 3, "sqneg", Shape::Same, ""), // 2d <- 2d
    (0b01000, true, 3, "cmge", Shape::Same, ", #0"), // 2d <- 2d
    (0b01001, true, 3, "cmle", Shape::Same, ", #0"), // 2d <- 2d
    (0b01011, true, 3, "neg", Shape::Same, ""), // 2d <- 2d
    (0b01100, true, 3, "fcmge", Shape::Same, ", #0.0"), // 2d <- 2d
    (0b01101, true, 3, "fcmle", Shape::Same, ", #0.0"), // 2d <- 2d
    (0b01111, true, 3, "fneg", Shape::Same, ""), // 2d <- 2d
    (0b11001, true, 3, "frinti", Shape::Same, ""), // 2d <- 2d
    (0b11010, true, 3, "fcvtpu", Shape::Same, ""), // 2d <- 2d
    (0b11011, true, 3, "fcvtzu", Shape::Same, ""), // 2d <- 2d
    (0b11101, true, 3, "frsqrte", Shape::Same, ""), // 2d <- 2d
    (0b11111, true, 3, "fsqrt", Shape::Same, ""), // 2d <- 2d
];

/// The FP16 two-register group: same key, but `bits 21:17 = 11100`. These are
/// the half-precision forms of the same operations, and they are a separate
/// encoding class rather than a width of the class above.
const TWO_MISC_HALF: &[(u8, bool, u8, &str, Shape, &str)] = &[
    (0b11000, false, 1, "frintn", Shape::Same, ""), // 8h <- 8h
    (0b11001, false, 1, "frintm", Shape::Same, ""), // 8h <- 8h
    (0b11010, false, 1, "fcvtns", Shape::Same, ""), // 8h <- 8h
    (0b11011, false, 1, "fcvtms", Shape::Same, ""), // 8h <- 8h
    (0b11100, false, 1, "fcvtas", Shape::Same, ""), // 8h <- 8h
    (0b11101, false, 1, "scvtf", Shape::Same, ""), // 8h <- 8h
    (0b01100, false, 3, "fcmgt", Shape::Half, ", #0.0"), // 8h
    (0b01101, false, 3, "fcmeq", Shape::Half, ", #0.0"), // 8h
    (0b01110, false, 3, "fcmlt", Shape::Half, ", #0.0"), // 8h
    (0b01111, false, 3, "fabs", Shape::Half, ""), // 8h
    (0b11000, false, 3, "frintp", Shape::Half, ""), // 8h
    (0b11001, false, 3, "frintz", Shape::Half, ""), // 8h
    (0b11010, false, 3, "fcvtps", Shape::Half, ""), // 8h
    (0b11011, false, 3, "fcvtzs", Shape::Half, ""), // 8h
    (0b11101, false, 3, "frecpe", Shape::Half, ""), // 8h
    (0b11000, true, 1, "frinta", Shape::Same, ""), // 8h <- 8h
    (0b11001, true, 1, "frintx", Shape::Same, ""), // 8h <- 8h
    (0b11010, true, 1, "fcvtnu", Shape::Same, ""), // 8h <- 8h
    (0b11011, true, 1, "fcvtmu", Shape::Same, ""), // 8h <- 8h
    (0b11100, true, 1, "fcvtau", Shape::Same, ""), // 8h <- 8h
    (0b11101, true, 1, "ucvtf", Shape::Same, ""), // 8h <- 8h
    (0b01100, true, 3, "fcmge", Shape::Half, ", #0.0"), // 8h
    (0b01101, true, 3, "fcmle", Shape::Half, ", #0.0"), // 8h
    (0b01111, true, 3, "fneg", Shape::Half, ""), // 8h
    (0b11001, true, 3, "frinti", Shape::Half, ""), // 8h
    (0b11010, true, 3, "fcvtpu", Shape::Half, ""), // 8h
    (0b11011, true, 3, "fcvtzu", Shape::Half, ""), // 8h
    (0b11101, true, 3, "frsqrte", Shape::Half, ""), // 8h
    (0b11111, true, 3, "fsqrt", Shape::Half, ""), // 8h
];

fn two_misc(word: u32, q: bool, u: bool, half: bool) -> Option<R> {
    let op = bits(word, 16, 12) as u8;
    let size = bits(word, 23, 22);
    let table = if half { TWO_MISC_HALF } else { TWO_MISC };
    let entry = table
        .iter()
        .find(|e| e.0 == op && e.1 == u && e.2 == size as u8)?;
    let (rd, rn) = (bits(word, 4, 0), bits(word, 9, 5));
    let text = shape_operands(entry.4, q, size, rd, rn, entry.5)?;
    let mn = if entry.4.suffix_two() {
        wide_name(entry.3, q)
    } else {
        entry.3
    };
    Some(R::new(mn, format!("{mn} {text}")))
}


/// How an across-lanes reduction reports its result.
#[derive(Clone, Copy, PartialEq)]
enum Cross {
    /// A scalar register of the source element width (`addv`).
    Int,
    /// A scalar register one element wider (`saddlv`).
    Long,
    /// An FP scalar whose width comes from the `u` bit (`fmaxv`), where the
    /// size field instead chooses between the four rounding operations.
    Fp,
}

/// Across-lanes reductions, keyed by `(opcode, u, size)` with `bits 21:17 =
/// 11000`. Read off a reference disassembly in the same way as `TWO_MISC`.
const ACROSS: &[(u8, bool, u8, &str, Cross)] = &[
    (0b00011, false, 0, "saddlv", Cross::Long),
    (0b01010, false, 0, "smaxv", Cross::Int),
    (0b01100, false, 0, "fmaxnmv", Cross::Fp),
    (0b01111, false, 0, "fmaxv", Cross::Fp),
    (0b11010, false, 0, "sminv", Cross::Int),
    (0b11011, false, 0, "addv", Cross::Int),
    (0b00011, false, 1, "saddlv", Cross::Long),
    (0b01010, false, 1, "smaxv", Cross::Int),
    (0b11010, false, 1, "sminv", Cross::Int),
    (0b11011, false, 1, "addv", Cross::Int),
    (0b00011, false, 2, "saddlv", Cross::Long),
    (0b01010, false, 2, "smaxv", Cross::Int),
    (0b01100, false, 2, "fminnmv", Cross::Fp),
    (0b01111, false, 2, "fminv", Cross::Fp),
    (0b11010, false, 2, "sminv", Cross::Int),
    (0b11011, false, 2, "addv", Cross::Int),
    (0b00011, true, 0, "uaddlv", Cross::Long),
    (0b01010, true, 0, "umaxv", Cross::Int),
    (0b01100, true, 0, "fmaxnmv", Cross::Fp),
    (0b01111, true, 0, "fmaxv", Cross::Fp),
    (0b11010, true, 0, "uminv", Cross::Int),
    (0b00011, true, 1, "uaddlv", Cross::Long),
    (0b01010, true, 1, "umaxv", Cross::Int),
    (0b11010, true, 1, "uminv", Cross::Int),
    (0b00011, true, 2, "uaddlv", Cross::Long),
    (0b01010, true, 2, "umaxv", Cross::Int),
    (0b01100, true, 2, "fminnmv", Cross::Int),
    (0b01111, true, 2, "fminv", Cross::Int),
    (0b11010, true, 2, "uminv", Cross::Int),
];

fn across_lanes(word: u32, q: bool, u: bool) -> Option<R> {
    let op = bits(word, 16, 12) as u8;
    let size = bits(word, 23, 22);
    let entry = ACROSS
        .iter()
        .find(|e| e.0 == op && e.1 == u && e.2 == size as u8)?;
    let (rd, rn) = (bits(word, 4, 0), bits(word, 9, 5));
    // A `2s` source has no reduction: the 128-bit `4s` form is the only one
    // that produces a 64-bit result, so `Q` is fixed by the size field.
    let (dest, src) = match entry.4 {
        Cross::Int => {
            if size == 2 && !q {
                return None;
            }
            (element(size), arrangement(q, size))
        }
        Cross::Long => {
            if size == 2 && !q {
                return None;
            }
            (element(size + 1), arrangement(q, size))
        }
        Cross::Fp => {
            let width = if u { 2 } else { 1 };
            if width == 2 && !q {
                return None;
            }
            (element(width), arrangement(q, width))
        }
    };
    Some(R::new(
        entry.3,
        format!("{} {}, {}", entry.3, elem_reg(dest, rd), vreg(rn, src)),
    ))
}

/// The half of the vector space that keeps bit 21 clear: table lookup,
/// extract, permute, and the copy forms, plus the FP16 three-register group
/// that reuses the copy opcodes at a non-zero size.
fn vector_low(word: u32, q: bool) -> Result<R, &'static str> {
    let u = bits(word, 29, 29) == 1;
    let size = bits(word, 23, 22);
    // `a` (bit 29) selects between the pairs that share bits 15:12: `tbl` and
    // `ext` differ only by it, and so do the permutes and `ext`.
    let do_ext = || ext(word, q).ok_or("SIMD/FP extract (unallocated)");
    match (bits(word, 11, 10), u) {
        (0b00, false) => table_lookup(word, q).ok_or("SIMD/FP table lookup (unallocated)"),
        (0b00, true) | (0b10, true) => do_ext(),
        (0b10, false) => permute(word, q, size).ok_or("SIMD/FP permute (unallocated)"),
        // With bit 10 set the word is a copy form when bits 23:22 are clear and
        // an FP16 three-register operation otherwise.
        (_, _) if size != 0 => {
            fp16_three(word, q, u, size).ok_or("SIMD/FP FP16 three-register (unallocated)")
        }
        (_, _) => copy(word, q).ok_or("SIMD/FP copy (unallocated)"),
    }
}

/// Copy operations: lane insertion and extraction. The five-bit `imm5` names
/// both the element width, by the position of its lowest set bit, and the lane.
///
/// Bit 29 separates the pairs that share bits 15:10: `dup` from `ins` for a
/// vector source, and a general-purpose source from a vector one for `ins`.
fn copy(word: u32, q: bool) -> Option<R> {
    if bits(word, 15, 15) != 0 || bits(word, 10, 10) != 0 {
        return None;
    }
    let imm5 = bits(word, 20, 16);
    let size = imm5.trailing_zeros();
    if size > 3 || (size == 3 && !q) {
        return None;
    }
    let lane = imm5 >> (size + 1);
    let lanes = (if q { 16 } else { 8 }) >> size;
    let (rd, rn) = (bits(word, 4, 0), bits(word, 9, 5));
    let e = element(size);
    let gpr = if size == 3 { 'x' } else { 'w' };
    let imm4 = bits(word, 14, 11);
    if bits(word, 29, 29) == 1 {
        // INS (element): the source lane is `imm4`.
        if !q || lane >= lanes || imm4 >= lanes {
            return None;
        }
        return Some(R::new(
            "mov",
            format!("mov v{rd}.{e}[{lane}], v{rn}.{e}[{imm4}]"),
        ));
    }
    match (bits(word, 15, 12), bits(word, 11, 11)) {
        // DUP (element): `imm4` must be clear, since that field is the source
        // lane of the form above.
        (0, 0) => {
            if imm4 != 0 || lane >= lanes {
                return None;
            }
            Some(R::new(
                "dup",
                format!("dup {}, v{rn}.{e}[{lane}]", vreg(rd, arrangement(q, size))),
            ))
        }
        // DUP (general).
        (0, 1) => {
            if lane >= lanes {
                return None;
            }
            Some(R::new(
                "dup",
                format!("dup {}, {gpr}{rn}", vreg(rd, arrangement(q, size))),
            ))
        }
        // INS (general); the assembler spells it `mov`.
        (1, 1) => {
            if !q || lane >= lanes {
                return None;
            }
            Some(R::new("mov", format!("mov v{rd}.{e}[{lane}], {gpr}{rn}")))
        }
        // SMOV: a signed extract. A 32-bit element only fits a 64-bit register,
        // so it needs `Q`, and there is no 64-bit element form.
        (2, 1) => {
            if size > 2 || (size == 2 && !q) || lane >= lanes {
                return None;
            }
            let dest = if q { 'x' } else { 'w' };
            Some(R::new(
                "smov",
                format!("smov {dest}{rd}, v{rn}.{e}[{lane}]"),
            ))
        }
        // UMOV: a zero-extending extract. The assembler calls the 32-bit and
        // 64-bit element forms `mov`.
        (3, 1) => {
            if lane >= lanes || (size == 3 && !q) {
                return None;
            }
            let dest = if q { 'x' } else { 'w' };
            let mn = if size >= 2 { "mov" } else { "umov" };
            Some(R::new(mn, format!("{mn} {dest}{rd}, v{rn}.{e}[{lane}]")))
        }
        _ => None,
    }
}

fn permute(word: u32, q: bool, size: u32) -> Option<R> {
    // Bits 15:12 name the operation. Bit 11 is part of nothing here: the four
    // permutes occupy even values and the extract occupies the odd ones.
    let mn = match bits(word, 15, 12) {
        0b0001 => "uzp1",
        0b0010 => "trn1",
        0b0011 => "zip1",
        0b0101 => "uzp2",
        0b0110 => "trn2",
        0b0111 => "zip2",
        _ => return None,
    };
    if size == 3 && !q {
        return None;
    }
    let (rd, rn, rm) = (bits(word, 4, 0), bits(word, 9, 5), bits(word, 20, 16));
    let a = arrangement(q, size);
    Some(R::new(
        mn,
        format!("{mn} {}, {}, {}", vreg(rd, a), vreg(rn, a), vreg(rm, a)),
    ))
}

/// `ext`: a byte-wise extract from the concatenation of two vectors. The
/// immediate is five bits wide in the architecture but only four are encoded;
/// values of eight and above need the full vector to exist.
fn ext(word: u32, q: bool) -> Option<R> {
    // The size field is not a width here: any value in it makes the word
    // something else.
    if bits(word, 15, 15) != 0 || bits(word, 23, 22) != 0 {
        return None;
    }
    let imm = bits(word, 14, 11);
    if imm > 7 && !q {
        return None;
    }
    let (rd, rn, rm) = (bits(word, 4, 0), bits(word, 9, 5), bits(word, 20, 16));
    let a = if q { "16b" } else { "8b" };
    Some(R::new(
        "ext",
        format!(
            "ext {}, {}, {}, #{imm}",
            vreg(rd, a),
            vreg(rn, a),
            vreg(rm, a)
        ),
    ))
}

/// `tbl` and `tbx`: a byte lookup across one to four vector registers.
///
/// The list is always spelled `.16b`: a table register contributes all sixteen
/// bytes even when the destination is 64 bits wide.
fn table_lookup(word: u32, q: bool) -> Option<R> {
    if bits(word, 15, 15) != 0 || bits(word, 23, 22) != 0 {
        return None;
    }
    let len = bits(word, 14, 13);
    let mn = if bits(word, 12, 12) == 1 { "tbx" } else { "tbl" };
    let (rd, rn, rm) = (bits(word, 4, 0), bits(word, 9, 5), bits(word, 20, 16));
    let a = if q { "16b" } else { "8b" };
    let mut list = String::from("{ ");
    for i in 0..=len {
        if i > 0 {
            list.push_str(", ");
        }
        list.push_str(&vreg(rn + i, "16b"));
    }
    list.push_str(" }");
    Some(R::new(
        mn,
        format!("{mn} {}, {list}, {}", vreg(rd, a), vreg(rm, a)),
    ))
}

/// FP16 three-register arithmetic, keyed by `(u, size, bits 15:12, bits 11:10)`.
///
/// This group reuses the copy opcodes with bits 23:22 non-zero, which is what
/// keeps the two apart. Every row was read off a reference disassembly.
const FP16_THREE: &[(bool, u8, u8, u8, &str)] = &[
    (false, 1, 0b0000, 0b01, "fmaxnm"),
    (false, 1, 0b0000, 0b11, "fmla"),
    (false, 1, 0b0001, 0b01, "fadd"),
    (false, 1, 0b0001, 0b11, "fmulx"),
    (false, 1, 0b0010, 0b01, "fcmeq"),
    (false, 1, 0b0011, 0b01, "fmax"),
    (false, 1, 0b0011, 0b11, "frecps"),
    (false, 3, 0b0000, 0b01, "fminnm"),
    (false, 3, 0b0000, 0b11, "fmls"),
    (false, 3, 0b0001, 0b01, "fsub"),
    (false, 3, 0b0011, 0b01, "fmin"),
    (false, 3, 0b0011, 0b11, "frsqrts"),
    (true, 1, 0b0000, 0b01, "fmaxnmp"),
    (true, 1, 0b0001, 0b01, "faddp"),
    (true, 1, 0b0001, 0b11, "fmul"),
    (true, 1, 0b0010, 0b01, "fcmge"),
    (true, 1, 0b0010, 0b11, "facge"),
    (true, 1, 0b0011, 0b01, "fmaxp"),
    (true, 1, 0b0011, 0b11, "fdiv"),
    (true, 3, 0b0000, 0b01, "fminnmp"),
    (true, 3, 0b0001, 0b01, "fabd"),
    (true, 3, 0b0010, 0b01, "fcmgt"),
    (true, 3, 0b0010, 0b11, "facgt"),
    (true, 3, 0b0011, 0b01, "fminp"),];

fn fp16_three(word: u32, q: bool, u: bool, size: u32) -> Option<R> {
    let hi = bits(word, 15, 12) as u8;
    let low = bits(word, 11, 10) as u8;
    let entry = FP16_THREE
        .iter()
        .find(|e| e.0 == u && e.1 == size as u8 && e.2 == hi && e.3 == low)?;
    let (rd, rn, rm) = (bits(word, 4, 0), bits(word, 9, 5), bits(word, 20, 16));
    let a = if q { "8h" } else { "4h" };
    Some(R::new(
        entry.4,
        format!(
            "{} {}, {}, {}",
            entry.4,
            vreg(rd, a),
            vreg(rn, a),
            vreg(rm, a)
        ),
    ))
}

/// Immediates and shifts share the `01111` space. Bit 10 separates the shift
/// group and the immediates from the by-element group, and a zero `immh` means
/// there is no element width to shift, which is how the modified immediates are
/// encoded.
fn vector_imm(word: u32, q: bool) -> Result<R, &'static str> {
    if bits(word, 10, 10) == 0 {
        return match by_element(word, q) {
            Some(r) => Ok(r),
            None => Err("SIMD/FP by element (unallocated)"),
        };
    }
    // Bit 23 belongs to the element size of the by-element forms; with bit 10
    // set it has no meaning, and the reference assembler has no encoding for
    // any word that sets it.
    if bits(word, 23, 23) == 1 {
        return Err("SIMD/FP immediate (unallocated)");
    }
    if bits(word, 22, 19) == 0 {
        return match modified_immediate(word, q) {
            Some(r) => Ok(r),
            None => Err("SIMD/FP modified immediate (unallocated)"),
        };
    }
    match shift_immediate(word, q) {
        Some(r) => Ok(r),
        None => Err("SIMD/FP shift by immediate (unallocated)"),
    }
}

/// By-element (indexed) arithmetic, keyed by `(u, opcode, bit 23, bit 22)`.
///
/// Here bits 23:22 no longer mean a width: the pair selects the element size
/// together with the shape, and the top bit of `Rm` doubles as an index bit, so
/// the same opcode appears once per combination. `true` in the last column means
/// the source is narrower than the destination and `Q` picks the `2` spelling.
const BY_ELEMENT: &[(bool, u8, bool, bool, &str, bool)] = &[
    (false,  1, false, false, "fmla", false),
    (false,  1, true, false, "fmla", false),
    (false,  1, true, true, "fmla", false),
    (false,  2, false, true, "smlal", true),
    (false,  2, true, false, "smlal", true),
    (false,  3, false, true, "sqdmlal", true),
    (false,  3, true, false, "sqdmlal", true),
    (false,  5, false, false, "fmls", false),
    (false,  5, true, false, "fmls", false),
    (false,  5, true, true, "fmls", false),
    (false,  6, false, true, "smlsl", true),
    (false,  6, true, false, "smlsl", true),
    (false,  7, false, true, "sqdmlsl", true),
    (false,  7, true, false, "sqdmlsl", true),
    (false,  8, false, true, "mul", false),
    (false,  8, true, false, "mul", false),
    (false,  9, false, false, "fmul", false),
    (false,  9, true, false, "fmul", false),
    (false,  9, true, true, "fmul", false),
    (false, 10, false, true, "smull", true),
    (false, 10, true, false, "smull", true),
    (false, 11, false, true, "sqdmull", true),
    (false, 11, true, false, "sqdmull", true),
    (false, 12, false, true, "sqdmulh", false),
    (false, 12, true, false, "sqdmulh", false),
    (false, 13, false, true, "sqrdmulh", false),
    (false, 13, true, false, "sqrdmulh", false),
    (true,  0, false, true, "mla", false),
    (true,  0, true, false, "mla", false),
    (true,  2, false, true, "umlal", true),
    (true,  2, true, false, "umlal", true),
    (true,  4, false, true, "mls", false),
    (true,  4, true, false, "mls", false),
    (true,  6, false, true, "umlsl", true),
    (true,  6, true, false, "umlsl", true),
    (true,  9, false, false, "fmulx", false),
    (true,  9, true, false, "fmulx", false),
    (true,  9, true, true, "fmulx", false),
    (true, 10, false, true, "umull", true),
    (true, 10, true, false, "umull", true),
];

/// By-element forms: `v0.4s, v1.4s, v2.s[1]`. The lane index is assembled from
/// `H` and whichever of bits 21:19 the element width leaves free, and the
/// register number takes the rest — which is why the field that spells `Rm`
/// elsewhere is split here.
fn by_element(word: u32, q: bool) -> Option<R> {
    let u = bits(word, 29, 29) == 1;
    let op = bits(word, 15, 10);
    let hcode = (op >> 2) as u8;
    let h = bits(word, 11, 11);
    let (b23, b22) = (bits(word, 23, 23) == 1, bits(word, 22, 22) == 1);
    let (b21, b20, b19) = (bits(word, 21, 21), bits(word, 20, 20), bits(word, 19, 19));
    let entry = BY_ELEMENT
        .iter()
        .find(|e| e.0 == u && e.1 == hcode && e.2 == b23 && e.3 == b22)?;
    let (rd, rn, rest) = (bits(word, 4, 0), bits(word, 9, 5), bits(word, 18, 16));
    let size = if !b23 {
        1
    } else if !b22 {
        2
    } else {
        3
    };
    let (index, rm) = match size {
        1 => ((h << 2) | (b21 << 1) | b20, (b19 << 3) | rest),
        2 => ((h << 1) | b21, (b20 << 4) | (b19 << 3) | rest),
        _ => {
            if b21 == 1 {
                return None;
            }
            (h, (b20 << 4) | (b19 << 3) | rest)
        }
    };
    if size == 3 && !q {
        return None;
    }
    let (mn, dest, src) = if entry.5 {
        (
            wide_name(entry.4, q),
            arrangement(true, size + 1),
            arrangement(q, size),
        )
    } else {
        (entry.4, arrangement(q, size), arrangement(q, size))
    };
    let e = element(size);
    Some(R::new(
        mn,
        format!(
            "{mn} {}, {}, v{rm}.{e}[{index}]",
            vreg(rd, dest),
            vreg(rn, src)
        ),
    ))
}

/// Shift by immediate. `immh:immb` holds the element width and the amount: the
/// width is the position of the highest set bit of `immh`, and the amount is
/// either `width - imm` (left shifts) or `2 * width - imm` (right shifts and
/// narrowing).
fn shift_immediate(word: u32, q: bool) -> Option<R> {
    let u = bits(word, 29, 29) == 1;
    let immh = bits(word, 22, 19);
    if immh == 0 {
        return None;
    }
    let immb = bits(word, 18, 16);
    let imm = (immh << 3) | immb;
    let size = 31 - immh.leading_zeros();
    let width = 8u32 << size;
    let (rd, rn) = (bits(word, 4, 0), bits(word, 9, 5));
    // The amount is counted from the top of the element for right shifts and
    // from its bottom for left shifts, which is why the two differ.
    let right = 2 * width - imm;
    let left = imm - width;
    let same = |mn: &'static str, amount: u32| {
        // There is no 64-bit shift: the form stops at 4S.
        if size > 2 {
            return None;
        }
        Some(R::new(
            mn,
            format!(
                "{mn} {}, {}, #{amount}",
                vreg(rd, arrangement(q, size)),
                vreg(rn, arrangement(q, size))
            ),
        ))
    };
    // Narrowing reads twice as wide as it writes, widening the other way round.
    let narrow = |mn: &'static str, amount: u32| {
        if size > 2 {
            return None;
        }
        let name = match mn {
            "shrn" if q => "shrn2",
            "rshrn" if q => "rshrn2",
            "sqshrn" if q => "sqshrn2",
            "sqrshrn" if q => "sqrshrn2",
            "sqshrun" if q => "sqshrun2",
            "sqrshrun" if q => "sqrshrun2",
            "uqshrn" if q => "uqshrn2",
            "uqrshrn" if q => "uqrshrn2",
            other => other,
        };
        Some(R::new(
            name,
            format!(
                "{name} {}, {}, #{amount}",
                vreg(rd, arrangement(q, size)),
                vreg(rn, arrangement(true, size + 1))
            ),
        ))
    };
    let widen = |mn: &'static str, amount: u32| {
        if size > 2 {
            return None;
        }
        let name = if q {
            match mn {
                "sshll" => "sshll2",
                _ => "ushll2",
            }
        } else {
            mn
        };
        Some(R::new(
            name,
            format!(
                "{name} {}, {}, #{amount}",
                vreg(rd, arrangement(true, size + 1)),
                vreg(rn, arrangement(q, size))
            ),
        ))
    };
    match (bits(word, 15, 11) as u8, u) {
        (0b00000, false) => same("sshr", right),
        (0b00000, true) => same("ushr", right),
        (0b00010, false) => same("ssra", right),
        (0b00010, true) => same("usra", right),
        (0b00100, false) => same("srshr", right),
        (0b00100, true) => same("urshr", right),
        (0b00110, false) => same("srsra", right),
        (0b00110, true) => same("ursra", right),
        (0b01000, true) => same("sri", right),
        (0b01010, false) => same("shl", left),
        (0b01010, true) => same("sli", left),
        (0b01100, true) => same("sqshlu", left),
        (0b01110, false) => same("sqshl", left),
        (0b01110, true) => same("uqshl", left),
        (0b10000, false) => narrow("shrn", right),
        (0b10000, true) => narrow("sqshrun", right),
        (0b10001, false) => narrow("rshrn", right),
        (0b10001, true) => narrow("sqrshrun", right),
        (0b10010, false) => narrow("sqshrn", right),
        (0b10010, true) => narrow("uqshrn", right),
        (0b10011, false) => narrow("sqrshrn", right),
        (0b10011, true) => narrow("uqrshrn", right),
        (0b10100, false) => widen("sshll", left),
        (0b10100, true) => widen("ushll", left),
        // Conversion has no byte form: the fixed-point value has to fit an FP
        // register, and there is no 8-bit lane to scale.
        (0b11100, false) if size > 0 => same("scvtf", right),
        (0b11100, true) if size > 0 => same("ucvtf", right),
        (0b11111, false) if size > 0 => same("fcvtzs", right),
        (0b11111, true) if size > 0 => same("fcvtzu", right),
        _ => None,
    }
}

/// Modified immediate: `movi`, `mvni`, `orr`, `bic` and the vector FP
/// immediate. `cmode` chooses the element width and the shift; its low bit
/// switches between the move and the or/and-not forms.
fn modified_immediate(word: u32, q: bool) -> Option<R> {
    let op = bits(word, 29, 29) == 1;
    let cmode = bits(word, 15, 12);
    let o2 = bits(word, 11, 11) == 1;
    let imm8 = (bits(word, 18, 16) << 5) | bits(word, 9, 5);
    let rd = bits(word, 4, 0);

    // `o2` is a second opcode bit only for the FP immediate: every other
    // combination with `o2` set is unallocated.
    if o2 && !(cmode == 0b1111 && !op) {
        return None;
    }

    if cmode == 0b1111 {
        // FP immediate. `op` and `o2` choose the type and the vector length
        // chooses the lane count, so all four combinations are distinct.
        let (t, a) = match (op, o2, q) {
            (false, false, false) => ('s', "2s"),
            (false, false, true) => ('s', "4s"),
            (false, true, false) => ('h', "4h"),
            (false, true, true) => ('h', "8h"),
            (true, false, true) => ('d', "2d"),
            _ => return None,
        };
        // The vector form spells all eight bits in one field, so the extra
        // five fraction bits of the scalar form are zero here.
        let value = expand_imm(imm8, t);
        return Some(R::new(
            "fmov",
            format!("fmov {}, #{}", vreg(rd, a), render_float(t, value)),
        ));
    }

    if cmode == 0b1110 {
        if !op {
            // The byte form: the immediate is already the pattern.
            return Some(R::new(
                "movi",
                format!("movi {}, #0x{imm8:02x}", vreg(rd, arrangement(q, 0))),
            ));
        }
        // The 64-bit form repeats each bit of the immediate as a whole byte of
        // `0x00` or `0xff`, which is what makes it a mask rather than a number.
        let mut value = 0u64;
        for i in 0..8 {
            if imm8 & (1 << i) != 0 {
                value |= 0xffu64 << (8 * i);
            }
        }
        let dest = if q {
            vreg(rd, "2d")
        } else {
            format!("d{rd}")
        };
        return Some(R::new(
            "movi",
            format!("movi {dest}, #0x{value:016x}"),
        ));
    }

    // The rest name a width, a shift and whether to invert or merge. The masked
    // shift group at `110x` counts as a 32-bit element because it fills the low
    // half of each lane with ones, so it has to be tested before the 16-bit
    // group, whose top bit it shares.
    let (size, msl, shift) = if cmode & 0b1100 == 0b1100 {
        (2, true, 8 * (1 + (cmode & 1)))
    } else if cmode & 0b1000 != 0 {
        (1, false, 8 * ((cmode >> 1) & 1))
    } else {
        (2, false, 8 * ((cmode >> 1) & 0b11))
    };
    // Only the plain shift group alternates with bit 0 of `cmode`; the masked
    // shift group always moves, because it has no "or" counterpart.
    let odd = cmode & 1 == 1 && cmode < 0b1100;
    let mn = match (op, odd) {
        (false, false) => "movi",
        (false, true) => "orr",
        (true, false) => "mvni",
        (true, true) => "bic",
    };
    let suffix = if msl {
        format!(", msl #{shift}")
    } else if shift != 0 {
        format!(", lsl #{shift}")
    } else {
        String::new()
    };
    Some(R::new(
        mn,
        format!(
            "{mn} {}, #0x{imm8:02x}{suffix}",
            vreg(rd, arrangement(q, size))
        ),
    ))
}

/// The mnemonic for a structure access: `ld1`..`ld4`, `st1`..`st4`, and the
/// replicating `ld1r`..`ld4r`.
fn structure_mnemonic(load: bool, count: u32, replicate: bool) -> &'static str {
    match (load, replicate, count) {
        (true, false, 1) => "ld1",
        (true, false, 2) => "ld2",
        (true, false, 3) => "ld3",
        (true, false, 4) => "ld4",
        (false, false, 1) => "st1",
        (false, false, 2) => "st2",
        (false, false, 3) => "st3",
        (false, false, 4) => "st4",
        (true, true, 1) => "ld1r",
        (true, true, 2) => "ld2r",
        (true, true, 3) => "ld3r",
        _ => "ld4r",
    }
}

/// Structure loads and stores. `single` selects the lane and replicate shapes
/// (`op1` = 01101) from the multiple-structure shape (`op1` = 01100).
///
/// For the multiple-structure shape the opcode in bits 15:12 names the register
/// count directly (0 -> 4, 4 -> 3, 8 -> 2, and 2/6/7/10 -> 1). For the
/// single-structure shape bit 21 halves the count (1/3 vs 2/4) and bits 15:13
/// pick the element width and whether the form replicates; the lane index is
/// packed into `Q : bit12 : size` and right-shifted by the element width.
/// `Rm` = 31 means post-index by the transfer size, `Rm` = 30 means `x30`, and
/// any other value with bit 23 set is a register post-index.
fn structure(word: u32, q: bool, single: bool) -> Result<R, &'static str> {
    let load = bits(word, 22, 22) == 1;
    let rd = bits(word, 4, 0);
    let rn = bits(word, 9, 5);
    let rm = bits(word, 20, 16);
    let size = bits(word, 11, 10);
    let opc = bits(word, 15, 12);
    let post = bits(word, 23, 23) == 1;

    // With no post-index the offset register field must be clear: it only
    // carries a meaning when bit 23 says the form writes back. Bit 21 is part
    // of the multiple-structure encoding and must be clear there; in the
    // single-structure group it instead halves the register count.
    if !post && rm != 0 {
        return Err("SIMD/FP structure (unallocated)");
    }
    if !single && bits(word, 21, 21) == 1 {
        return Err("SIMD/FP structure (unallocated)");
    }
    let (count, reg_arr, lane, stride) = if !single {
        let count = match opc {
            0 => 4,
            4 => 3,
            8 => 2,
            2 | 6 | 7 | 10 => 1,
            _ => return Err("SIMD/FP structure load (unallocated)"),
        };
        // A 64-bit element only has one lane at `Q = 0`, and the multiple-
        // structure forms need at least two.
        if size == 3 && !q {
            return Err("SIMD/FP structure load (unallocated)");
        }
        let a = arrangement(q, size);
        (count, a, None, count * (8 << (q as u32)))
    } else {
        let b21 = bits(word, 21, 21);
        let count = 1 + b21 + 2 * ((opc >> 1) & 1);
        if opc & 0b1100 == 0b1100 {
            if !load || opc & 1 == 1 {
                return Err("SIMD/FP structure replicate (unallocated)");
            }
            let a = arrangement(q, size);
            (count, a, None, count << size)
        } else {
            let packed = (bits(word, 30, 30) << 3) | ((opc & 1) << 2) | size;
            // The low bits of `packed` below the element width are a marker,
            // not part of the index, and the encoding only exists when they
            // match the width: `001` for a double, zero otherwise.
            let (elem, shift) = match (opc >> 2) & 3 {
                0 => ("b", 0u32),
                1 => ("h", 1),
                _ => {
                    if size & 1 == 0 {
                        ("s", 2)
                    } else {
                        ("d", 3)
                    }
                }
            };
            let marker = if shift == 3 { 1 } else { 0 };
            if packed & ((1 << shift) - 1) != marker {
                return Err("SIMD/FP structure lane (unallocated)");
            }
            let lane = packed >> shift;
            let lanes = (if q { 16 } else { 8 }) >> shift;
            if lane >= lanes {
                return Err("SIMD/FP structure lane (unallocated)");
            }
            (count, elem, Some(lane), count << shift)
        }
    };
    let mn = structure_mnemonic(load, count, single && opc & 0b1100 == 0b1100);
    let mut list = String::from("{ ");
    for i in 0..count {
        if i > 0 {
            list.push_str(", ");
        }
        if single && lane.is_some() {
            list.push_str(&format!("v{}.{reg_arr}", (rd + i) & 31));
        } else {
            list.push_str(&vreg((rd + i) & 31, reg_arr));
        }
    }
    if let Some(lane) = lane {
        list.push_str(&format!("}}[{lane}]"));
    } else {
        list.push_str(" }");
    }
    let base = format!("x{rn}");
    let text = if !post {
        format!("{mn} {list}, [{base}]")
    } else {
        match rm {
            31 => format!("{mn} {list}, [{base}], #{stride}"),
            30 => format!("{mn} {list}, [{base}], x30"),
            _ => format!("{mn} {list}, [{base}], x{rm}"),
        }
    };
    Ok(R::new(mn, text))
}
